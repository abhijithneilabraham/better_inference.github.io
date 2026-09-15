---
layout: default
title: "What I learned getting Gemma-4-31B to 150 tok/s on one B300"
---

# What I learned getting Gemma-4-31B to 150 tok/s on one B300

I spent a couple of weeks getting a 31B model to decode faster on a single GPU. The end number is 150 tok/s single-stream, exact output, which beats the best lossless result I could get out of SGLang by about 7%. That's the part that goes on a slide.

The part that's actually useful is everything that went wrong on the way, because most of it wasn't about the model at all. It was about a kernel that was numerically correct and still crashed, a version of FlashInfer that was one minor release too old, and a roofline calculation that told me the number I ended on is still six times slower than the hardware allows.

This is written as a tutorial. Not "here's the config, copy it" — the config is in the [repo](https://github.com/abhijithneilabraham/benchmark_gemma#readme) — but how to think when the stock setup doesn't work and you have to decide whether to give up, switch engines, or fork something. I'll use what actually happened as the worked example, with the real numbers.

Setup, so the numbers mean something: one NVIDIA B300 (sm_103, about 6.85 TB/s of HBM), CUDA 13. Workload is 10k input tokens, 1500 output, temperature 0.6, top_p 0.95, one request at a time, reporting p50 output tok/s. Model is `nvidia/Gemma-4-31B-IT-NVFP4` with `google/gemma-4-31B-it-assistant` as the MTP drafter for speculative decoding.

## Build the harness before you touch a single flag

This sounds boring. It's the thing that made everything else possible.

Before trying any engine, I wrote a small benchmark client and froze a dataset: 30 prompts at the workload above. Same prompts, same sampling, same warmup, every run. It reports output tok/s p50, TTFT, and whether all 30 requests succeeded.

The dataset is [`longctx30.jsonl`](https://github.com/abhijithneilabraham/benchmark_gemma/blob/main/longctx30.jsonl), built by [`make_longctx_dataset.py`](https://github.com/abhijithneilabraham/benchmark_gemma/blob/main/make_longctx_dataset.py) from public-domain nonfiction — Darwin, Thucydides, Adam Smith, Grant's memoirs, twelve books in all. Each prompt is two ~5k-token passages from two different books, followed by one of five tasks (summarize, Q&A, extract facts, outline, rewrite), which is what makes the 1500-token outputs realistic rather than padded. Every prompt lands within a few tokens of 10k input, measured with the same tokenizer the client uses. If you're building your own, the two things to hold fixed are the token count and the task mix; the actual text matters less than you'd think.

One thing to be straight about: the numbers in this post were measured on an earlier prompt set of exactly this shape. `longctx30.jsonl` is the public reproduction set — same token budget, same five tasks, same two-passage structure, different source text. I'd expect it to land within a few percent, but I haven't re-run the full ladder on it, so treat the figures here as measured on the original and reproducible on this.

The reason this matters: over the project I ran something like thirty distinct configurations across three engines and four different vLLM builds. If the harness had drifted even slightly between them, none of the comparisons would mean anything. A 3% difference between two configs is real when the harness is frozen and noise when it isn't.

The other thing the harness gave me was a correctness gate. Every config, before I looked at tok/s, got a greedy request: write a Fibonacci function, then compute 17 × 23. If the output wasn't coherent and didn't say 391, I stopped. Garbled output is a bug, not a slow config, and a bug that produces tokens fast is the most dangerous kind because it looks like a win.

## Try everything stock first, and write down every failure

I started with what you'd expect: SGLang and vLLM, BF16 and NVFP4, with and without speculative decoding. Here's what the first pass looked like, single-stream:

| Engine | Weights | Drafter | tok/s | What happened |
|---|---|---|---|---|
| SGLang | NVFP4 | MTP k6 | 139.6 | worked out of the box |
| SGLang | NVFP4 | none | 78.9 | the no-spec floor |
| vLLM 0.24 | BF16 | MTP k6 | 139.4 | worked |
| vLLM 0.24 | NVFP4 | any | — | `tie_weights NotImplementedError` |
| vLLM nightly | NVFP4 | none | 80.9 | loads, but no spec decoding |
| vLLM nightly | NVFP4 | MTP | — | `Trtllm-gen kernels not found: headDimQk=512` |

Two things jump out that I didn't expect going in.

First, speculative decoding is doing most of the work. NVFP4 without MTP is 79–81 tok/s on either engine. With MTP it's ~140. The quantization matters much less than the drafter at batch 1. I'd assumed the opposite.

Second, vLLM with plain BF16 weights and MTP was already at 139.4 — essentially tied with SGLang's NVFP4 result. So the "obvious" win of NVFP4 + MTP on vLLM was worth chasing precisely because both halves worked separately and only the combination failed.

Write the failures down verbatim. Those two error strings turned out to be walls 1 and 4 of nine, and having the exact text saved a lot of re-running later.

One correction to something I nearly wrote wrong in a post: every SGLang run used `--attention-backend triton`. I never ran SGLang with FlashInfer. So the head-to-head at the end is "vLLM on a patched FlashInfer FA2 path" against "SGLang on Triton," and that's a fair thing to say but not the same as "both used FlashInfer." Check your own configs before you write the comparison sentence.

## Read the checkpoint, not the model card

Gemma-4-31B has 60 layers. Fifty are sliding-window attention with head_dim 256. Ten are full global attention with head_dim **512**. They're every sixth layer.

<img src="assets/layers.svg" alt="Sixty layers, ten with head_dim 512 highlighted" width="800">

That single fact is the root of almost everything that follows. Most attention kernels cap at head_dim 256. trtllm-gen, which vLLM auto-selects for decode on Blackwell, has no dense 512 kernel at all. And it's not a knob you can turn down — 512 is the trained shape of those layers' Q/K/V projections. Reinterpret them as 256 and you read the weights wrong.

The second thing I found by opening the checkpoint's quantization config rather than assuming: `exclude_modules` lists every `self_attn`, plus `lm_head` and vision. Attention is **not** quantized. Only the MLP linears are NVFP4. Q, K, V, O and the output head are BF16.

This reframed the whole problem. I'd been thinking "how do I make FP4 attention work at head_dim 512," which sounds like a kernel-writing project. The real problem was "how do I get a fast kernel to accept a 512-wide BF16 head," which is a routing and version problem. Much smaller.

It also has a consequence for where the bytes go per forward pass, which I'll come back to.

<img src="assets/bytes.svg" alt="Bytes per forward: attention 16.96 GB unquantized, MLP 10.40 GB in FP4" width="800">

The unquantized attention weights are bigger than the quantized MLP. 56% of the bytes. Hold that thought.

## The point where you fork

Here's the decision I want to be honest about, because I think people get it wrong in both directions.

At this point I had: a working SGLang config at 139.6, a working vLLM BF16 config at 139.4, and a vLLM NVFP4 config that failed on load. The easy move is to stop. 139.6 is a fine number. Nobody would have faulted it.

The reason I didn't: both failures were *specific*. `tie_weights NotImplementedError` is one code path in one file. `headDimQk=512` is a kernel-selection decision, not a missing kernel — FlashInfer's own supported-head-sizes list includes 512. Neither error said "this can't work." They said "nobody wired this up." That's the signal to fork.

The signal *not* to fork is when the error is architectural — the engine genuinely doesn't have the abstraction you need, and you'd be rewriting a subsystem. I didn't hit that here, but it's the question to ask before every patch: am I connecting two things that exist, or am I building a thing that doesn't?

Some things that made the fork survivable:

**Patch against a pinned commit, and record it.** Everything is against vLLM `main @ 50ac1c7`, written down in `BASE_VLLM_COMMIT.txt`. Nightly moves; your patches shouldn't.

**One patch per wall.** Seven patches, each minimal, each commented with the exact error it clears. When one stops applying after an upstream change, you know exactly what to re-do.

**Make the Dockerfile self-verifying.** `Dockerfile.patched` upgrades FlashInfer and then runs a check that fails the build if the head_dim ≥ 512 fix isn't present. Better to fail at build time than three hours later on a real request.

**Keep a safety net you can flip with an env var.** `VLLM_GEMMA4_512_TRITON=1` routes the 512 layers to Triton instead of FA2. It's slower (115.7) but always correct. When something's broken at 2am, having a known-good fallback that's one flag away is worth more than the 30 tok/s.

## Nine walls

Each one was a different error, and each fix moved strictly further. No fix for wall 7 accidentally fixed wall 3. That's what makes it feel like a maze instead of a bug.

<img src="assets/walls.svg" alt="Nine sequential failures and their fixes" width="800">

A few worth expanding:

**Walls 1–3 are load-time.** `tie_weights` because the head is excluded from quantization but the quant method's tie function raises on the base class — catch it, do a plain tensor tie. The MTP drafter's centroids embedding was a bare `nn.Linear` that couldn't accept packed FP4 weights — swap for a quant-aware `ReplicatedLinear`. Marlin's NVFP4 kernel needs 64-aligned output dims and Gemma has an 8608-wide projection — skip Marlin in auto-selection. None of these are interesting. All of them block you.

**Wall 4 is the real one.** On the SM100 family, vLLM's FlashInfer backend hard-selects trtllm-gen for decode. trtllm-gen has no 512 kernel. Patch 02 adds a flag that, when head_dim is 512, forces the native FA2 path instead. That's about twenty lines.

**Walls 5 and 6 are the same problem twice.** FA2 can't read an FP8 KV cache. But only the ten 512 layers use FA2. So only those ten get BF16 KV, via vLLM's `kv_cache_dtype_skip_layers`, and the other fifty stay FP8. Then again for the drafter, which has its own 512 layer. I think this is the most reusable trick in the project: per-layer dtype splits chosen by *which kernel each layer needs*, not by accuracy.

**Wall 7 wasn't our code.** The bundled FlashInfer was 0.6.13. The FA2 head_dim ≥ 512 tile-selection fix landed in 0.6.14 (PR #3576). Without it the tile chooser returns a config that fails a register-budget check with a scary-looking `Invalid configuration` error. A version bump. But there's no 0.6.14 cubin package — the 512 entry was dropped to keep the wheel under GitHub's size limit — so you pin `flashinfer-python==0.6.14` with `flashinfer-cubin==0.6.13` and set `FLASHINFER_DISABLE_VERSION_CHECK=1`. This is fine, and it took me a while to be sure it was fine: the 512 kernel JIT-compiles from the Python package's headers, so the cubin version is irrelevant for that kernel specifically.

**Wall 8** is a workspace buffer. 394 MiB default, FA2-512 wants ~767 MiB. Set it to 2 GiB and move on.

## A kernel that's correct and still crashes

Wall 9 deserves its own section because it's a different kind of bug and the fix generalizes.

After wall 8, the server started. Warmup ran. CUDA graph capture ran. Everything looked fine. Then the first real request faulted with `CUDA illegal memory access`.

With `--enforce-eager`, it worked. Coherent output, Fibonacci correct, 17 × 23 = 391, 30 of 30 requests. At 46.7 tok/s.

So the FA2-512 kernel is numerically correct. It just isn't safe under CUDA graph replay on this architecture. The thing that's easy to miss: warmup and graph capture use dummy data. A kernel can pass capture and still fault on real inputs. "Server came up" tells you nothing.

<img src="assets/graph-modes.svg" alt="Same forward pass under eager, FULL cudagraph, and PIECEWISE" width="800">

vLLM resolves one CUDA-graph mode per model from the minimum support level across its attention groups. Patch 07 makes any 512-wide group report `NEVER`, so vLLM refuses to put it in a full graph — it fails fast at startup instead of faulting later. That's a diagnostic, not a fix.

The fix is a launch flag: `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`. Attention becomes a graph split point. Every attention op runs eagerly, on FA2's proven-correct path. Everything between attention ops — the FP4 GEMMs, the norms, most of the compute — stays CUDA-graphed. The crashing full-capture decode path is never built.

46.7 → 145.9.

The general lesson: **eager vs. graphs is a diagnostic.** If a kernel works with `--enforce-eager` and faults with graphs, you have a graph-safety bug, not a math bug, and you probably don't need to touch the kernel. You need to keep that one op out of the graph. Before this I'd have assumed a CUDA IMA meant a kernel patch.

Why it's graph-unsafe isn't fully settled. My best current read, from FlashInfer's own docstrings and an open upstream issue with the same fingerprint (#3929, on the trtllm-gen path), is split-KV CTA-count variance — the number of launched CTAs depends on KV length, and a planned-vs-replayed mismatch surfaces as a replay-time fault. PR #3576 added eager-only tests. I wouldn't have expected it to work under graphs either.

The honest limitation: vLLM's one-mode-per-model rule means I can't graph the fifty 256 attentions and only eager the ten 512s. It's all attention eager or none. PIECEWISE is the achievable optimum on this vLLM.

## The exact knobs

Once the fast path worked, two things were left to tune, and both are exact — same output tokens, just faster or slower.

`num_speculative_tokens` (k) is how many tokens the drafter attempts per pass. MTP verifies every draft against the target, so k never changes what's kept. The Gemma-4 assistant registers with `n_predict=1`, which makes k a plain loop count with a ceiling of 128.

| k | tok/s | | block size (k=8) | tok/s |
|---|---|---|---|---|
| 5 | 141.2 | | 16 | **150.0** |
| 6 | 145.9 | | 32 | 148.2 |
| 7 | 147.9 | | 64 | 146.8 |
| 8 | **150.0** | | ≥128 | re-routes to trtllm-gen, breaks |

The k curve is still rising at 8. I'd sweep 10, 12, 16 next. Block size has to stay ≤ 64 — at 128 vLLM picks trtllm-gen again and the 512 gap comes straight back.

That's the whole sweep. Four values of k, three of block size. When the structural stuff is done, tuning is short.

## Compute the roofline before you celebrate

At batch 1, decode should be memory-bound: tok/s ≈ bandwidth ÷ bytes read per token. From the bytes figure earlier, one target forward reads about 31.4 GB including KV, which at 6.85 TB/s is roughly 4.6 ms. The measured step at k=8 is 29.7 ms.

<img src="assets/roofline.svg" alt="4.6 ms roofline versus 29.7 ms measured: 6.5x gap" width="800">

6.5× off. At 150 tok/s the model is overhead-bound, not bandwidth-bound. Roughly 85% of each step is launch overhead, dispatch, and the drafter's own forwards — not reading weights.

This is the single most useful number in the project, and I computed it last. I should have done it first. It invalidates most of the tuning menu: no KV dtype change or block-size tweak closes a 6.5× gap. It also says the attention-quantization idea — which would cut the roofline from 4.4 ms to ~2.3 ms, since attention is 56% of the bytes — is worth nothing until the overhead gap closes, because the roofline isn't what's binding.

The lever that *would* close it is FULL CUDA graphs. Which is the one thing the FA2-512 bug blocks. So the next 2× is behind that bug, not behind any further quantization. That's a less satisfying answer than "try FP8 KV" but it's the true one.

<img src="assets/ladder.svg" alt="46.7 to 115.7 to 145.9 to 150.0 tok/s, with SGLang 139.6 marked" width="800">

SGLang also posts 159.4 with a relaxed acceptance threshold. That's lossy — it changes the output — and the equivalent knob doesn't exist in vLLM v1, so it isn't a fair comparison. Exact against exact, vLLM is ahead by 7.5%.

## Four things I got wrong

My first-pass notes had a list of "quick wins." Going back through vLLM source and FlashInfer's PRs, four of them were wrong. Recording them because they're the kind of wrong that sounds right.

**"The eight eager drafter forwards are eating the step."** The drafter has its own CUDA-graph dispatcher and PIECEWISE already propagates to it. The k=8 loop is graphed. There's no drafter win left — and setting `speculative_config.enforce_eager` silently drops it to no graphs at all and wrecks k=8.

**"Re-enable FP8 KV on the 512 layers."** Global KV at 10k context is 0.82 GB in BF16. FP8 saves 0.41 GB, about 0.06 ms of a 29.7 ms step. 0.2%. Not a speed lever.

**"Add a lossy acceptance threshold like SGLang for +14%."** vLLM v1's rejection sampler has three modes: standard, synthetic, block. No posterior threshold. "Synthetic" accepts without verifying, which is garbage, not a quality trade. SGLang's 159.4 is a number you can't reach in vLLM v1.

**"FA2-512 isn't tested on sm_103; wait for a version that supports it."** PR #3576 gates SM100+; sm_103 was always in scope. The bug is graph-safety, not arch coverage, and no version through 0.6.15 fixes it. Waiting won't help.

## The decision tree, in retrospect

If I compress the whole thing into what to do when the stock setup fails:

<img src="assets/decision.svg" alt="Decision flow: load error, dtype mismatch, kernel not found, crash — and the eager test" width="800">

Every box on the amber path was a different error. None of the fixes were kernel rewrites. Two were version bumps. One was a config mode. The rest were twenty-line routing or dtype patches.

The question I'd ask at each step now: *is this error telling me the thing doesn't exist, or that nobody connected it?* Almost every wall here was the second kind.

## What I'd do differently

Compute the roofline on day one. It costs an hour with the checkpoint config and a calculator and it tells you which tuning knobs are worth touching. I did it on the last day and it retroactively invalidated a week of "quick win" ideas.

Test eager vs. graphs the moment anything faults. It's the fastest way to split "kernel is wrong" from "kernel is graph-unsafe," and those have completely different fixes.

And gate every config on a real request before looking at throughput. I'd have shipped at least one broken config as a win otherwise.

The recipe, all seven patches, the self-verifying Dockerfile and the correctness script are in [`fpa4fix/`](https://github.com/abhijithneilabraham/benchmark_gemma/tree/main/fpa4fix). The full engineering log is [`FPA4FIX.md`](https://github.com/abhijithneilabraham/benchmark_gemma/blob/main/FPA4FIX.md), and the mechanism-level detail — file and line for every change, plus the roofline math — is in [`TECHNICAL_DEEP_DIVE.md`](https://github.com/abhijithneilabraham/benchmark_gemma/blob/main/TECHNICAL_DEEP_DIVE.md).

---

*Every number here is from the harness runs recorded in [benchmark_gemma](https://github.com/abhijithneilabraham/benchmark_gemma). Nothing is interpolated. Where a mechanism is a best current theory rather than confirmed, it says so.*
