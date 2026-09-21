---
layout: default
title: "Learning inference: How to reduce the cold start time of an LLM server"
permalink: /cold-start/
---

# Learning inference: How to reduce the cold start time of an LLM server

Getting a model to answer fast once it is running is one problem. Getting it to start in the first place is a different problem, and it gets less attention. In an earlier article I looked at tokens per second once a server is warm. In this one I look at the seconds before that, the time between starting the process and the moment it can answer anything at all.

I will use vLLM serving a 14 billion parameter model as the example, because the numbers are large enough to be worth fixing and small enough that every step fits on one GPU.

## What is cold start

Cold start is the time from starting the server process to the moment it returns the first token. Before that moment the GPU can be sitting there doing nothing useful for you, even though it is fully occupied loading things and compiling things.

Here is why it matters. If you run one long lived server and never restart it, cold start is a one time cost you can mostly ignore. But if your server restarts often, because it autoscales, because it runs inside a container that gets rescheduled, because you are iterating on a config in development, or because it shares a GPU with something else and has to give it back, then cold start happens over and over, and it starts to matter as much as steady state speed does.

In the setup I used, an unmodified server took 160 seconds to go from a cold process to its first token. That is nearly three minutes where the GPU is not serving anyone.

## Experiment Setup

I want to be precise about the setup, because every number in this article comes from it, and if you change any part of it the numbers will move.

**Hardware.** One NVIDIA H100 80GB. Everything here is a single GPU, so there is no distributed startup cost, no NCCL handshake between nodes. A multi GPU setup would add its own phase on top of everything measured here.

**Model.** `Qwen/Qwen3-14B` in bf16, about 27.5 GiB of weights.

**Engine.** vLLM 0.11.0 with transformers 4.56.2, using vLLM's own Python interface directly rather than the OpenAI compatible server, so the numbers are the engine's cost and not the API server's cost on top of it. I also ran the same experiments against vLLM 0.28.0 later, and I mention the differences where they matter.

**What counts as cold start here.** I measured wall clock time from the moment the process starts importing Python packages to the moment a single test generation returns. That last part matters. It is easy to time "the server says it is ready" and miss that a server can report ready and still fail, or still be significantly slower, on the first real request. I always ran one generation and included its time.

**The benchmark harness.** Each measurement is a fresh Python process, not a warm loop, because I wanted every run to have a genuinely new CUDA context, matching what a real restart looks like. Between runs I wait for the GPU's memory to fully clear, and if a run leaves an orphaned process holding memory, the harness kills it before the next one starts. Most configurations were run three or five times and I report the mean with the standard deviation, the same way you would report a p50 with its spread. The page cache, which I explain below, was only dropped for the two experiments that specifically measure disk behaviour. Dropping it everywhere would add about 75 seconds of constant disk noise on top of whatever else was being measured, and would make it impossible to see small effects.

## Where do the 160 seconds go

Here is the breakdown for the unmodified server, cold in every sense: nothing cached on disk, no compiled kernels saved from a previous run.

```
159.6s total
├──  78.6s   reading the weights off disk
├──  31.9s   torch.compile building fast GPU kernels
├──   5.7s   capturing CUDA graphs
└──  43.4s   Python imports, engine setup, everything else
```

<img src="assets/waterfall.png" alt="A stacked bar chart showing cold start time falling from 159.6 seconds to 18.2 seconds as page cache, compile cache, smaller graph capture, and finally eager execution are each applied in turn" width="800">

Two things dominate, and they are two completely different kinds of cost. One is disk. One is compilation. Both of them are things you pay once and then, if you are not careful, throw away and pay again on the next restart.

## The first big cost: reading weights off disk

The model is 27.5 GiB. Reading that off disk, cold, took 78.6 seconds. That is measured, not estimated. I confirmed it separately with a plain `dd` read off a freshly dropped cache, and got 432 megabytes per second, which is the actual ceiling of the disk, not something vLLM's loader is doing inefficiently. I also tried reading two shards of the file in parallel to see if that would help. It did not. Two parallel readers together got 395 megabytes per second, slightly worse than one reader alone, so the disk does not reward parallel reads here.

The fix does not touch the disk at all. Linux keeps a copy of any file it has recently read in spare RAM, called the page cache. If the same file is read again and it is still in that cache, the read comes from RAM instead of disk, and it is fast. On this machine, with 196 GiB of RAM and a 27.5 GiB model, the whole model fits in the cache with room to spare.

So before starting the server, read the file once:

```bash
cat ~/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/*/*.safetensors > /dev/null
```

That line does nothing except force Linux to load the file into RAM. The next time vLLM reads it, the same 78.6 seconds becomes 4.6 seconds. Here is the harness command for the cold case and the warm case, so you can see exactly what changes:

```bash
# cold: drop the page cache first, then measure
sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'

python3 harness/run_once.py \
  --model Qwen/Qwen3-14B \
  --out result.json \
  --do-generate \
  --clean-compile-cache
```

That gave 159.60 seconds, averaged over five runs, with a spread of about 1.7 seconds.

```bash
# warm: read the weights into page cache first, then measure the same command
cat ~/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/*/*.safetensors > /dev/null

python3 harness/run_once.py \
  --model Qwen/Qwen3-14B \
  --out result.json \
  --do-generate \
  --clean-compile-cache
```

That gave 72.63 seconds. The `--clean-compile-cache` flag is still there in both commands, on purpose, because I only wanted to isolate the disk. The compiler cost is still fully paid in both of these runs. That is the next thing to fix.

## The second big cost: compiling the model

`torch.compile` takes the model's Python code and turns it into optimised GPU kernels. This is what makes vLLM fast once it is running, and it is also expensive to do. On this model it took 31.9 seconds, cold.

The important thing is that this work produces a real artifact on disk. vLLM writes the compiled result to `~/.cache/vllm/torch_compile_cache`, keyed by a hash of the model and the engine configuration. If that folder already has the right entry in it, the compiler does not have to redo the work. The only reason it took 31.9 seconds in the two commands above is that I deliberately deleted that folder first, with `--clean-compile-cache`, to measure the true cold cost. In normal use you would never do that.

So the fix is simply: do not delete it. Leave the flag off.

```bash
python3 harness/run_once.py \
  --model Qwen/Qwen3-14B \
  --out result.json \
  --do-generate
```

With the page cache still warm from the previous step, and the compile cache now left alone, this gave 40.24 seconds. Compare the three numbers so far:

| configuration | time | what changed |
|---|---|---|
| cold disk, cold compile cache | 159.60s | nothing yet, this is the baseline |
| warm page cache | 72.63s | pre read the weights once |
| warm page cache and warm compile cache | 40.24s | also stopped deleting the compiled kernels |

That is a 4x improvement, and neither step changes what the model computes or how fast it runs once it is warm. You are not trading anything away. You are just not throwing away work you already did.

## Squeezing the remaining 40 seconds

Two more changes get this down further, and this is where it gets more specific to how vLLM captures CUDA graphs.

A CUDA graph records a whole sequence of GPU operations once, so they can be replayed as a single unit later instead of being launched one at a time from Python. vLLM captures a separate graph for a range of batch sizes so it has a fast path ready no matter how many requests arrive together. By default it captures graphs for 67 different batch sizes, and capturing all of them cold takes 5.7 seconds. Most of that is unnecessary if you only ever expect to serve a handful of concurrent requests.

```bash
python3 harness/run_once.py \
  --model Qwen/Qwen3-14B \
  --out result.json \
  --do-generate \
  --cuda-graph-sizes 1,2,4,8
```

Capturing only four sizes instead of 67 brought the total down to 34.74 seconds. This is worth a caveat. It is free if you only ever serve small batches, but if real traffic later arrives in batches of 32 or 64, execution falls back to a slower, uncaptured path for those sizes, and I measured that fallback costing about 11 percent of steady state throughput at batch 64. Choose the sizes you capture based on the concurrency you actually expect to serve, not by minimising the list blindly.

Stacking three more changes together, pinning the KV cache size instead of letting vLLM measure it with a profiling pass, telling vLLM to only build the lighter "piecewise" graph instead of the full one, and running the engine in the same process instead of spawning a child process for it, brought the number to 29.48 seconds:

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0

python3 harness/run_once.py \
  --model Qwen/Qwen3-14B \
  --out result.json \
  --do-generate \
  --kv-cache-memory-bytes 32212254720 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --cuda-graph-sizes 1,2,4,8
```

And if you are willing to drop compilation entirely, using `--enforce-eager`, the number falls to 18.20 seconds:

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0

python3 harness/run_once.py \
  --model Qwen/Qwen3-14B \
  --out result.json \
  --do-generate \
  --enforce-eager \
  --kv-cache-memory-bytes 32212254720
```

This last one is a real trade, not a free win. Without compilation, every token generated afterwards costs somewhere between 10 and 20 percent more time, for as long as the server stays up. It is worth it for a short lived job that starts, does a small amount of work, and exits. It is usually not worth it for a server meant to stay up and serve traffic for hours, where the steady state cost adds up to far more than the 11 seconds you saved at startup.

## The floor, and what is actually in it

18.2 seconds is close to the floor for this model on this hardware, so it is worth asking what is actually left in it once nothing can be cut further. I instrumented one run in detail, and it breaks down like this:

| what | seconds |
|---|---|
| vLLM inspecting its own model registry | 6.5 |
| reading the weights, now from page cache | 4.5 |
| importing the vLLM package | 3.4 |
| plugin discovery and assembling the config | about 2 |
| importing torch | 1.0 |
| distributed setup, even with one GPU | about 1 |
| reserving the KV cache | about 1 |

The single biggest item is not weight loading. It is vLLM inspecting the model class to work out its capabilities, which is pure CPU work that happens before any weight is touched, and it happens even though the result is cached. None of this is GPU time. All of it is Python and CPU overhead that has nothing to do with the size of the model or the size of the GPU.

## Two things that looked like they would help and did not

**Pinning the KV cache size to skip the memory profiling step.** vLLM normally runs a short profiling pass to work out how much GPU memory it can safely give to the KV cache. I assumed telling it the number directly would skip that pass. It does not. It still runs the same profiling forward pass; it only skips a small accounting step afterwards. The measured saving was 0.3 seconds. I kept the flag in the stacked configuration above because it is harmless, not because it is doing real work on its own.

**Running the engine in the same process instead of a child process.** vLLM normally spawns a separate process for the engine core, which means torch and vLLM get imported twice, once in the parent and once in the child, each with its own CUDA context. Turning that off should save one whole import and one whole context setup. Measured on its own, it did not produce a consistent number, sometimes it looked like a small win and sometimes it looked like nothing, with more noise than signal. I only kept it because it is a requirement for reaching the 1.1 second engine initialisation in the fully eager configuration, not because it helps by itself.

Neither of these is a criticism of vLLM. They are a reminder that a change which sounds like it should save time needs to be measured, not assumed.

## Changing a setting without paying for any of this again

Everything above is about a full process restart. A more common situation is smaller: you want to change one setting on a server that is already running well. Whether that is free or expensive depends entirely on what the setting is.

Sampling settings, temperature, top_p, top_k, the maximum number of tokens, the random seed, cost nothing. They are attached to each request, not to the engine, so there is no restart at all.

Waking a sleeping engine, which I get to below, costs about half a second.

A restart where you change `max_num_seqs`, `gpu_memory_utilization`, or turn on prefix caching costs about 35 to 40 seconds, the same as any warm restart, because none of those touch the compiled kernels. The compile cache is still a hit.

A restart where you change `max_model_len` or the KV cache dtype costs 74 to 84 seconds, because those change the shapes or the data types the compiled kernels were built for, so the old compiled kernels no longer match and have to be rebuilt from scratch.

I did not want to guess at this distinction from timing alone, because a slow run and a cache miss can look similar. vLLM prints the exact folder name it is using for the compiled cache, and that name is a hash of everything that affects the compiled result. If two runs print the same hash, they are provably sharing the same compiled kernels. If the hash changes, it is provably a miss. I checked every one of the settings above this way, not just by looking at the clock.

One more honest note. Even a full hit is not free. It still costs 35 to 40 seconds, because part of the compilation step, the part called Dynamo tracing, is not saved to disk at all on this version of vLLM and has to be redone on every single process start, hit or miss. Only the heavier Inductor part of compilation is actually cached. I confirmed this stops being true on a newer vLLM, 0.28.0, where that tracing step is itself cached and a hit drops from about 6.5 seconds to about 0.35 seconds. That same newer version also changed which settings count as a miss, `cuda_graph_sizes` became one of them, where it used to be free, so upgrading is not purely a win without re-checking your own settings against it.

## The thing that beats every fix above

Everything so far reduces the cost of a restart. The best answer is often to not restart at all.

vLLM can put a running engine to sleep, which moves its weights off the GPU and frees the memory, while keeping the process, the imports, and the compiled kernels exactly as they were.

```python
llm.sleep(level=1)   # about 15 seconds, frees the GPU memory
llm.wake_up()        # about 0.55 seconds, ready again
```

Waking up is not a restart in miniature. Nothing gets re-imported, nothing gets recompiled, no new CUDA context is created. It is the same process picking its weights back up. Half a second to go from asleep to serving its next token, against 29 to 40 seconds for the fastest full restarts above.

Use this whenever you need to hand the GPU to something else temporarily and want the same server back quickly, rather than tearing it down and paying any of the costs in this article again.

## What I would actually do

If the machine is starting completely fresh, with nothing cached anywhere, the two free wins are to read the weight file into page cache while the rest of startup is happening, and to ship a pre built compile cache instead of building one from scratch on first boot. Together those two changes alone took this example from 159.6 seconds to 40.2 seconds, with nothing given up.

If the server restarts with a stable configuration, stack the smaller graph capture list, the pinned KV cache size, and piecewise graph mode, which get you to about 29.5 seconds, or accept the throughput trade of eager execution for about 18.2 seconds if the process is short lived.

If you only need to change a setting, check whether it is a `SamplingParams` field first, because those are free. If it is an engine setting, expect a 35 to 40 second restart unless it touches shapes or dtypes, in which case expect closer to 75 to 85 seconds.

And if the process itself can stay alive, prefer sleeping and waking it over restarting it. Half a second beats every number in this article by a wide margin.

## Where the code is

Every harness script, every raw result file, and the full set of caveats and provenance notes are in the [inference_research](https://github.com/abhijithneilabraham/inference_research) repository. The main README has the complete table of configurations with their measured standard deviations, the version comparison against vLLM 0.28.0, and the Docker packaging work for actually deploying a pre warmed image, which is its own article's worth of detail and did not fit here.

*Every number in this article is from measured runs recorded in that repository, most of them averaged over three or five repeats with the spread reported alongside. Nothing here is estimated or rounded up to make a cleaner story.*
