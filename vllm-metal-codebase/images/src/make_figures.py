"""Render the figures for this article. Run: python3 make_figures.py (needs rsvg-convert)."""
import html, os

SHA = "42e4b2a313a61406dacbf3723bd4571812eaff47"
GH = f"https://github.com/vllm-project/vllm-metal/blob/{SHA}/"
OUT = os.path.dirname(os.path.abspath(__file__))

def e(s): return html.escape(s, quote=True)

def src(path, line=None, label=None):
    url = GH + path + (f"#L{line}" if line else "")
    lab = label or (path + (f":{line}" if line else ""))
    return f'<a class="src" href="{url}" target="_blank" rel="noopener">{e(lab)}</a>'

# ---------------------------------------------------------------- SVG helpers
def box(x, y, w, h, k, title, sub=None, mono=False, sub2=None, r=7, tcls=None):
    t = f'<rect class="b-{k}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}"/>'
    ty = y + (20 if sub else h / 2 + 5)
    if sub and not sub2 and h >= 56: ty = y + 23
    if sub and h < 50: ty = y + 17
    t += f'<text class="{tcls or ("cm" if mono else "n")}" x="{x+12}" y="{ty}">{e(title)}</text>'
    if sub:
        t += f'<text class="d" x="{x+12}" y="{ty+18}">{e(sub)}</text>'
    if sub2:
        t += f'<text class="d" x="{x+12}" y="{ty+33}">{e(sub2)}</text>'
    return t

def txt(x, y, s, cls="d", anchor="start"):
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{e(s)}</text>'

def arrow(d, cls="ar"):
    return f'<path class="{cls}" d="{d}"/>'

def num(x, y, n):
    return (f'<circle class="badge" cx="{x}" cy="{y}" r="10"/>'
            f'<text class="badge-t" x="{x}" y="{y+4}" text-anchor="middle">{n}</text>')

def svg(w, h, label, body, minw=760):
    return (f'<div class="fig"><svg viewBox="0 0 {w} {h}" style="min-width:{minw}px" role="img" '
            f'aria-label="{e(label)}">{body}</svg></div>')

# ---------------------------------------------------------------- Fig 1: layers
def fig_layers():
    rows = [
        ("prog", "Your program", "asks for text", [
            ("vllm serve", "OpenAI-style HTTP server"), ("LLM(...)", "the Python API"),
            ("curl / client", "sends prompts, gets tokens")]),
        ("vllm", "vLLM", "plans the work", [
            ("API server", "tokenize, stream back"), ("EngineCore", "the main loop"),
            ("Scheduler", "who runs, how many tokens"), ("KV block manager", "block ids per request")]),
        ("glue", "vllm-metal: glue", "joins vLLM to the Mac", [
            ("register()  compat  envs", "entry point and patches"), ("MetalPlatform", "platform.py: checks, defaults"),
            ("MetalWorker", "v1/worker.py: lifecycle")]),
        ("runner", "vllm-metal: runner", "runs one step", [
            ("MetalModelRunner", "build batch, forward, sample"), ("ModelLifecycle + Adapter", "load and drive the model"),
            ("ModelCachePolicy", "KV specs, memory plan"), ("DecodePipeline, drafters", "overlap, spec decode")]),
        ("attn", "vllm-metal: attention", "paged KV cache", [
            ("wrappers (patching.py)", "swap the attention layer"), ("PagedAttentionContext", "per-step metadata"),
            ("runtimes", "SDPA, MLA, Hybrid"), ("KVCacheStorage", "owns the KV memory")]),
        ("kern", "native ops", "C++ and Metal", [
            ("paged_ops.cpp", "MLX primitives (nanobind)"), ("kernels_v2/*.metal", "paged attention, MLA, TurboQuant"),
            ("GDN lazy kernels", "for hybrid models, JIT")]),
        ("mlx", "MLX", "the array library", [
            ("mlx_lm / mlx_vlm", "the model code itself"), ("lazy arrays", "build a graph, then mx.async_eval"),
            ("mx.quantized_matmul", "4-bit and 8-bit weights")]),
    ]
    W, X0, XC, CW = 960, 20, 212, 728
    b = ""
    y = 16
    for k, lab, sub, chips in rows:
        h = 56
        b += f'<rect class="lane-{k}" x="{X0}" y="{y}" width="{W-40}" height="{h}" rx="8"/>'
        b += f'<text class="h k-{k}" x="{X0+14}" y="{y+24}">{e(lab)}</text>'
        b += txt(X0 + 14, y + 42, sub)
        n = len(chips); cw = (CW - 10 * (n - 1)) / n
        for i, (t, s) in enumerate(chips):
            b += box(XC + i * (cw + 10), y + 6, cw, h - 12, k, t, s, r=5)
        y += h + 8
    # hardware row
    h = 92
    b += f'<rect class="lane-hw" x="{X0}" y="{y}" width="{W-40}" height="{h}" rx="8"/>'
    b += f'<text class="h k-hw" x="{X0+14}" y="{y+24}">Apple Silicon</text>'
    b += txt(X0 + 14, y + 42, "one memory for all")
    b += box(XC, y + 8, 200, 30, "hw", "CPU cores", r=5)
    b += box(XC + 210, y + 8, 200, 30, "hw", "GPU cores", r=5)
    b += txt(XC + 422, y + 28, "both read and write the same memory chips")
    segs = [(170, "mlx", "model weights"), (230, "attn", "KV pool (blocks)"), (110, "attn2", "state pools"),
            (218, "hw", "macOS and other apps")]
    x = XC
    for w, k, t in segs:
        b += f'<rect class="b-{k}" x="{x}" y="{y+46}" width="{w}" height="38" rx="4"/>'
        b += txt(x + 10, y + 70, t, "n")
        x += w
    y += h + 10
    return svg(W, y, "The layers of vllm-metal, from your program down to the Apple Silicon unified memory", b)

# ---------------------------------------------------------------- Fig 2: startup
def fig_startup():
    W = 960
    lanes = [("vllm", "API server process", 20), ("glue", "EngineCore process", 330), ("runner", "Worker (inside EngineCore by default)", 640)]
    LW = 300
    steps = [
        (0, "find the plugin", "entry point vllm.platform_plugins"),
        (0, "_register()", "env defaults, compat patches, MetalPlatform"),
        (0, "build VllmConfig", "check_and_update_config: worker, checks"),
        (1, "start EngineCore", "load plugins, build the uni executor"),
        (2, "init_device", "MLX device, wired limit, pick the runner"),
        (2, "load_model", "loaders: mlx_lm, mlx_vlm, GGUF, AWQ"),
        (2, "get_kv_cache_spec", "ModelCachePolicy: full, sliding, MLA, state"),
        (2, "determine_available_memory", "limit x util - weights - overhead"),
        (2, "initialize_from_config", "KVCacheStorage, runtime, patch_model"),
        (2, "compile_or_warm_up_model", "a dummy forward, load the kernels"),
        (1, "busy loop", "schedule, execute_model, sample_tokens"),
    ]
    top, step, bh = 74, 62, 46
    H = top + len(steps) * step + 10
    b = ""
    for k, lab, x in lanes:
        b += f'<rect class="lane-{k}" x="{x}" y="14" width="{LW}" height="{H-24}" rx="8"/>'
        b += f'<text class="h k-{k}" x="{x+14}" y="40">{e(lab)}</text>'
    prev = None
    for i, (li, t, s) in enumerate(steps):
        k, _, x = lanes[li]
        y = top + i * step
        b += box(x + 34, y, LW - 48, bh, k, t, s, mono=True, r=6)
        b += num(x + 18, y + bh / 2, i + 1)
        cx = x + 34 + (LW - 48) / 2
        if prev:
            px, py = prev
            if px == cx:
                b += arrow(f"M{px},{py} L{cx},{y-2}")
            else:
                m = py + (y - py) / 2
                b += arrow(f"M{px},{py} L{px},{m} L{cx},{m} L{cx},{y-2}")
        prev = (cx, y + bh)
    return svg(W, H, "Startup sequence of vllm-metal across the API server, EngineCore and worker", b)

# ---------------------------------------------------------------- Fig 3: one step
def fig_step():
    W = 960
    b = box(20, 14, 920, 48, "vllm", "vLLM Scheduler", "SchedulerOutput: which requests run, how many tokens each, which block ids they own")
    L = [("reconcile requests", "drop finished, preempted; restore resumed"),
         ("sort the requests", "new and continuing prompts = prefill; the rest = decode"),
         ("add draft tokens", "spec decode: a decode row is [last token, drafts...]"),
         ("pack one batch", "decode rows first, then prefill chunks"),
         ("prepare_grouped()", "PagedAttentionContext: slots, block tables, cu_seqlens"),
         ("model forward", "mlx_lm model, cache = OffsetCache stubs; logits"),
         ("mx.async_eval()", "the GPU starts; Python returns at once")]
    R = [("wait for the logits", "or hand the step to the decode pipeline"),
         ("grammar mask, sample", "MLX argmax, or vLLM's Sampler on the CPU"),
         ("verify drafts", "verify_greedy: keep the matching prefix + 1"),
         ("update RequestState", "new tokens, sequence lengths"),
         ("drafter.propose()", "draft tokens for the next step"),
         ("ModelRunnerOutput", "sampled tokens back to the scheduler")]
    b += f'<rect class="lane-runner" x="20" y="78" width="450" height="{40+len(L)*60}" rx="8"/>'
    b += f'<text class="h k-runner" x="34" y="102">execute_model(): build and submit</text>'
    b += f'<rect class="lane-runner" x="490" y="78" width="450" height="{40+len(L)*60}" rx="8"/>'
    b += f'<text class="h k-runner" x="504" y="102">sample_tokens(): wait and sample</text>'
    for col, items, x0 in ((0, L, 34), (1, R, 504)):
        for i, (t, s) in enumerate(items):
            y = 114 + i * 60
            b += box(x0, y, 422, 44, "runner", t, s, mono=(t.endswith(")") or t.startswith("Model")), r=5)
            if i: b += arrow(f"M{x0+211},{y-15} L{x0+211},{y-2}")
    b += arrow("M245,62 L245,76")
    b += arrow(f"M456,{114+6*60+22} C478,{114+6*60+22} 478,136 500,136")
    b += arrow(f"M926,{114+5*60+22} L950,{114+5*60+22} L950,38 L944,38")
    # packed batch strip
    y0 = 114 + 7 * 60 + 20
    b += txt(20, y0 + 6, "One packed batch for this step (varlen: each request keeps its own length)", "n")
    segs = [("d1", 1, "runner"), ("d2", 1, "runner"), ("d3+3 drafts", 4, "glue"), ("prefill A", 6, "attn"), ("prefill B", 4, "mlx")]
    x, s = 20, 34
    cu = [0]
    for lab, n, k in segs:
        for j in range(n):
            b += f'<rect class="b-{k}" x="{x+j*s}" y="{y0+18}" width="{s-3}" height="30" rx="3"/>'
        b += txt(x + 3, y0 + 64, lab, "d")
        x += n * s
        cu.append(cu[-1] + n)
    xs = 20
    for i, c in enumerate(cu):
        cx = 20 + c * s - (1.5 if c else 0)
        b += f'<line class="tick" x1="{cx}" y1="{y0+12}" x2="{cx}" y2="{y0+52}"/>'
        b += txt(cx, y0 + 84, str(c), "cm", "middle")
    b += txt(20 + 16 * s + 16, y0 + 38, "cu_seqlens = [0, 1, 2, 6, 12, 16]", "cm")
    b += txt(20 + 16 * s + 16, y0 + 56, "decode rows first, prefill after", "d")
    H = y0 + 100
    return svg(W, H, "One engine step: execute_model builds and submits the batch, sample_tokens waits and samples", b)

# ---------------------------------------------------------------- Fig 4: decode pipeline
def fig_pipeline():
    W = 960
    b = ""
    def lane(y, label):
        return txt(20, y + 20, label, "n")
    b += txt(20, 30, "Without the pipeline: the GPU waits while Python samples and builds the next batch", "h k-hw")
    b += lane(44, "CPU") + lane(80, "GPU")
    x = 90
    for i in range(3):
        b += f'<rect class="b-runner" x="{x}" y="44" width="80" height="28" rx="4"/>' + txt(x + 8, 62, f"build {i+1}", "d")
        b += f'<rect class="b-kern" x="{x+80}" y="80" width="120" height="28" rx="4"/>' + txt(x + 88, 98, f"forward {i+1}", "d")
        b += f'<rect class="b-attn" x="{x+200}" y="44" width="70" height="28" rx="4"/>' + txt(x + 206, 62, f"sample {i+1}", "d")
        x += 270
    b += txt(20, 150, "With the pipeline: Python builds step n+1 while the GPU still runs step n; tokens stay on the GPU", "h k-attn")
    b += lane(164, "CPU") + lane(200, "GPU")
    x = 90
    for i in range(5):
        b += f'<rect class="b-runner" x="{x}" y="164" width="80" height="28" rx="4"/>' + txt(x + 8, 182, f"build {i+1}", "d")
        b += f'<rect class="b-kern" x="{x+80}" y="200" width="120" height="28" rx="4"/>' + txt(x + 88, 218, f"forward {i+1}", "d")
        x += 125
    b += txt(20, 260, "Used only for plain decode steps with async scheduling on: no new prompts, no spec decode, no grammar.", "d")
    return svg(W, 272, "Timeline of decode steps with and without the decode pipeline", b)

# ---------------------------------------------------------------- Fig 5: attention patch
def fig_attn():
    W = 960
    b = ""
    b += txt(20, 30, "Before: the model as mlx_lm loads it", "h k-mlx")
    b += f'<rect class="lane-mlx" x="20" y="42" width="290" height="120" rx="8"/>'
    b += box(34, 54, 262, 40, "mlx", "DecoderLayer", "mlp, norms, ...")
    b += box(34, 104, 262, 46, "mlx", "self_attn = Attention", "its own KV cache, one request at a time")
    b += txt(20, 200, "After: walk_and_wrap() swaps one attribute", "h k-attn")
    b += f'<rect class="lane-attn" x="20" y="212" width="290" height="160" rx="8"/>'
    b += box(34, 224, 262, 40, "mlx", "DecoderLayer", "unchanged")
    b += box(34, 274, 262, 46, "attn", "self_attn = SDPA wrapper", "keeps the original projections")
    b += txt(34, 344, "setattr(layer, \"self_attn\", wrapper)", "cm")
    b += txt(34, 362, "the model file is never edited", "d")
    chain = [("get_context()", "this step's batch metadata"),
             ("prepare_sdpa_qkv()", "q, k, v projections, norms, RoPE per segment"),
             ("reshape_and_cache()", "write new K and V at slot_mapping"),
             ("paged_attention_primitive()", "read all K and V through block_tables"),
             ("o_proj(out)", "back to the model, which never noticed")]
    for i, (t, s) in enumerate(chain):
        y = 54 + i * 64
        b += box(350, y, 300, 48, "kern" if 2 <= i <= 3 else "attn", t, s, mono=True)
        if i: b += arrow(f"M500,{y-16} L500,{y-2}")
    b += arrow("M296,297 C322,297 322,78 346,78")
    b += box(690, 54, 250, 70, "runner", "PagedAttentionContext", "set by the runner before forward:", sub2="slot_mapping, block_tables, cu_seqlens")
    b += arrow("M690,90 L652,80")
    b += box(690, 190, 250, 140, "attn", "KVCacheStorage", "one pool, one region per layer", sub2="blocks of 16 tokens, K and V")
    b += arrow("M650,190 C672,190 672,230 688,232")
    b += arrow("M688,280 C670,280 670,262 652,262")
    b += txt(704, 296, "write: one slot per new token", "d")
    b += txt(704, 314, "read: every block in the table", "d")
    return svg(W, 386, "How vllm-metal replaces an mlx_lm attention layer with a paged attention wrapper", b)

# ---------------------------------------------------------------- Fig 6: KV layout
def fig_kv():
    W = 960
    b = txt(20, 30, "The KV pool: one region per layer, each region cut into blocks (layout LBNHC: layer, block, token, head, content)", "n")
    # regions
    x = 20
    for i, lab in enumerate(["layer 0", "layer 1", "layer 2", "...", "layer L-1"]):
        w = 176
        b += f'<rect class="b-attn" x="{x}" y="44" width="{w-6}" height="36" rx="4"/>'
        b += txt(x + 10, 67, lab, "n")
        x += w + 7
    # layer 0 blocks
    b += txt(60, 112, "Layer 0, cut into blocks. Request A owns blocks 7, 2 and 9, in that order.", "d")
    owned = {7: 0, 2: 1, 9: 2}
    for i in range(16):
        x = 20 + i * 57
        k = "runner" if i in owned else "soft"
        b += f'<rect class="b-{k}" x="{x}" y="122" width="52" height="40" rx="4"/>'
        b += txt(x + 26, 147, str(i), "cm", "middle")
    b += arrow("M44,82 L44,118")
    # block table
    b += txt(20, 200, "Block table for request A", "n")
    for i, p in enumerate([7, 2, 9]):
        x = 20 + i * 92
        b += f'<rect class="b-runner" x="{x}" y="212" width="86" height="54" rx="4"/>'
        b += txt(x + 43, 232, f"logical {i}", "d", "middle")
        b += txt(x + 43, 254, f"block {p}", "cm", "middle")
    # slot calc
    calc = ["Where does token 37 of request A go?",
            "logical block = 37 // 16 = 2   ->  physical block 9",
            "offset inside the block = 37 % 16 = 5",
            "slot = 9 x 16 + 5 = 149   (slot_mapping[that token] = 149)"]
    for i, l in enumerate(calc):
        b += txt(20, 300 + i * 22, l, "n" if i == 0 else "cm")
    # block 9 zoom
    bx, by = 470, 196
    b += txt(bx, by, "Inside block 9: 16 token rows", "n")
    for r in range(16):
        y = by + 12 + r * 14
        k = "kern" if r == 5 else "attn"
        b += f'<rect class="b-{k}" x="{bx}" y="{y}" width="170" height="12" rx="2"/>'
        b += f'<rect class="b-{k}" x="{bx+174}" y="{y}" width="170" height="12" rx="2"/>'
        if r in (0, 5, 15): b += txt(bx + 352, y + 10, f"token {r}", "d")
    y = by + 12 + 16 * 14 + 16
    b += txt(bx, y, "head 0: [ K | V ]", "cm") + txt(bx + 174, y, "head 1: [ K | V ]", "cm")
    b += txt(bx, y + 20, "row 5 is slot 149: K and V of every KV head for token 37", "d")
    b += arrow(f"M{20+9*57+26},164 C{20+9*57+26},180 {bx+120},180 {bx+120},{by-14}")
    return svg(W, y + 34, "Layout of the paged KV cache in memory, with a worked example of a slot mapping", b)

# ---------------------------------------------------------------- Fig 7: kernel dispatch
def fig_dispatch():
    W = 960
    b = box(20, 14, 340, 46, "kern", "paged_attention_primitive", "eval_gpu() in paged_ops.cpp picks a kernel", mono=True)
    qs = [("Mixed batch with long decode rows?", "a decode row has 4096+ tokens of context", "split", "long decode rows go to the per-token path; the rest go on below", "runner"),
          ("Prefill on an M5 chip?", "fp16/bf16, head size 64..512, block 8/16/32", "pagedattention_nax.metal", "uses the M5 neural accelerators (NAX)", "kern"),
          ("Prefill with a tiled head size?", "head size 64, 96, 128, 256 or 512", "pagedattention_tiled.metal", "FlashAttention-2 style tiles of 32 queries", "kern"),
          ("Otherwise: per-token kernel", "one GPU thread group per (head, query token)", "pagedattention.metal", "GQA decode, split-KV + reduce, or one pass", "kern")]
    for i, (q, qd, a, ad, k) in enumerate(qs):
        y = 84 + i * 74
        b += box(20, y, 400, 54, "soft", q, qd)
        b += arrow(f"M420,{y+27} L520,{y+27}")
        b += txt(470, y + 20, "yes" if i < 3 else "", "d", "middle")
        b += box(524, y, 416, 54, k, a, ad, mono=(i > 0))
        if i: b += arrow(f"M200,{y-20} L200,{y-2}")
        if i < 3: b += txt(214, y + 70, "no", "d")
    b += arrow("M200,60 L200,82")
    return svg(W, 84 + 4 * 74 + 4, "How the paged attention primitive chooses which Metal kernel to run", b)

# ---------------------------------------------------------------- Fig 8: spec decode
def fig_spec():
    W = 960
    b = ""
    b += box(20, 30, 280, 74, "glue", "1. propose", "a drafter guesses k tokens", sub2="ngram, draft_model, gemma4_mtp, dflash, dspark")
    b += box(340, 30, 280, 74, "runner", "2. verify", "next step packs [last, d1, d2, d3]", sub2="one forward checks all guesses")
    b += box(660, 30, 280, 74, "attn", "3. accept", "verify_greedy keeps the matching", sub2="prefix, plus one token from the model")
    b += arrow("M300,67 L336,67") + arrow("M620,67 L656,67")
    b += arrow("M800,104 C800,150 160,150 160,108")
    b += txt(480, 168, "Greedy sampling only. Async scheduling is turned off. Hybrid models are not supported.", "d", "middle")
    return svg(W, 182, "The propose, verify and accept loop of speculative decoding", b)

# ---------------------------------------------------------------- Fig 9: everything
def fig_all():
    W, H = 1240, 1000
    b = ""
    b += '<rect class="lane-vllm" x="20" y="14" width="1200" height="128" rx="10"/>'
    b += '<text class="h k-vllm" x="36" y="40">vLLM (upstream): plans the work</text>'
    b += box(36, 54, 190, 72, "prog", "Your program", "vllm serve, LLM(),", sub2="an HTTP client")
    b += box(266, 54, 220, 72, "vllm", "API server", "tokenizes, streams tokens", sub2="back to the client")
    b += box(526, 54, 300, 72, "vllm", "EngineCore + Scheduler", "picks requests and token counts,", sub2="hands out KV block ids")
    b += box(866, 54, 338, 72, "vllm", "Sampler, structured output", "borrowed by the runner when", sub2="sampling is not plain argmax")
    b += arrow("M226,90 L262,90") + arrow("M486,90 L522,90")
    b += num(246, 112, 1) + num(506, 112, 2)
    b += '<rect class="lane-glue" x="20" y="160" width="270" height="392" rx="10"/>'
    b += '<text class="h k-glue" x="36" y="186">Glue</text>'
    b += box(36, 200, 238, 128, "glue", "MetalWorker", "init_device, load_model,", sub2="determine_available_memory,")
    b += txt(48, 268, "initialize_from_config, warm up,") + txt(48, 283, "execute_model, sample_tokens,") + txt(48, 298, "sleep, wake_up")
    b += box(36, 342, 238, 76, "glue", "MetalPlatform", "startup checks, memory budget,", sub2="worker class, offload setup")
    b += box(36, 432, 238, 60, "glue", "register(), compat, envs", "entry point, patches, settings")
    b += box(36, 504, 238, 38, "soft", "MPSWorker (experimental)", r=5)
    b += '<rect class="lane-runner" x="306" y="160" width="500" height="392" rx="10"/>'
    b += '<text class="h k-runner" x="322" y="186">MetalModelRunner: runs one step</text>'
    helpers = [("ModelLifecycle", "mlx_lm, mlx_vlm, GGUF, AWQ"), ("ModelAdapter", "per-model quirks"),
               ("ModelCachePolicy", "KV specs, runtime, memory plan"), ("DecodePipeline", "overlap decode steps"),
               ("drafters", "ngram, draft model, MTP, DFlash"), ("LoRA, multimodal", "adapters, image encoders"),
               ("pooling, diffusion", "embeddings, block diffusion")]
    for i, (t, s_) in enumerate(helpers):
        b += box(322, 200 + i * 48, 210, 42, "soft", t, s_, r=5)
    flow = [("build the batch", "decode rows first, then prefill"),
            ("PagedAttentionContext", "slots, block tables, cu_seqlens"),
            ("model forward (mlx_lm)", "then logits, mx.async_eval"),
            ("sample, verify drafts", "then drafter.propose()")]
    for i, (t, s_) in enumerate(flow):
        y = 200 + i * 70
        b += box(548, y, 242, 54, "runner", t, s_)
        if i: b += arrow(f"M669,{y-16} L669,{y-2}")
    b += num(772, 216, 4) + num(772, 356, 5) + num(772, 426, 8)
    b += arrow("M669,126 L669,196")
    b += num(686, 152, 3)
    b += txt(652, 156, "via MetalWorker.execute_model()", "cm", "end")
    b += '<rect class="lane-attn" x="822" y="160" width="398" height="392" rx="10"/>'
    b += '<text class="h k-attn" x="838" y="186">Attention and kernels</text>'
    b += box(838, 200, 366, 54, "attn", "wrapped attention layers", "patching.py swaps self_attn, linear_attn, conv")
    b += box(838, 268, 178, 66, "attn", "runtimes", "SDPA, MLA, Hybrid", sub2="gdn, shortconv, mamba2")
    b += box(1026, 268, 178, 66, "attn", "KVCacheStorage", "one region per layer,", sub2="release / reallocate")
    b += box(838, 348, 366, 66, "kern", "paged_ops.cpp", "write: reshape_and_cache", sub2="read: paged_attention_primitive")
    b += box(838, 428, 366, 66, "kern", "kernels_v2/*.metal", "NAX (M5), tiled (FA2 style), per-token,", sub2="MLA, TurboQuant, GDN state kernels")
    b += txt(838, 522, "prebuilt by metal/build.py")
    b += arrow("M927,254 L927,264") + arrow("M927,334 L927,344") + arrow("M927,414 L927,424")
    b += arrow("M1115,348 L1115,338")
    b += num(1186, 366, 7)
    b += arrow("M790,297 C810,297 812,227 834,227")
    b += num(806, 250, 6)
    b += arrow("M790,467 L814,467 L814,132", "arb")
    b += num(814, 510, 9)
    b += '<rect class="lane-hw" x="20" y="572" width="1200" height="226" rx="10"/>'
    b += '<text class="h k-hw" x="36" y="598">Apple Silicon: CPU and GPU share one unified memory</text>'
    segs = [(270, "hw", "macOS and other apps", "share the same budget"),
            (256, "mlx", "model weights", "loaded once by ModelLifecycle"),
            (244, "glue", "offload host pool", "KV blocks that left the pool"),
            (280, "attn", "KV pool", "blocks of 16 tokens, lazy"),
            (134, "attn2", "state pools", "hybrid models")]
    x = 36
    for w, k, t, s_ in segs:
        b += box(x, 612, w - 6, 62, k, t, s_)
        x += w
    b += arrow("M1040,494 L1040,608")
    b += box(36, 700, 380, 82, "soft", "second Mac (pipeline parallel)", "layers split across Macs; hidden states", sub2="sent over the mx.distributed ring")
    b += box(430, 700, 380, 82, "soft", "disk tier (KV offload)", "a folder for KV blocks that do not fit", sub2="in the host pool")
    b += box(824, 700, 380, 82, "soft", "sleep(level=1)", "frees the KV pool, keeps the weights;", sub2="wake_up() makes the same layout again")
    b += arrow("M685,674 L685,696") + arrow("M1040,674 L1040,696")
    leg = ["1 A prompt arrives at the API server.",
           "2 The scheduler picks requests, token counts and KV block ids.",
           "3 EngineCore calls the worker, which calls the runner.",
           "4 The runner packs one batch and sets the attention context.",
           "5 The mlx_lm model runs; MLX builds a lazy graph.",
           "6 Each wrapped attention layer reads the context.",
           "7 Native ops write new K and V, then read old K and V.",
           "8 The runner samples tokens and proposes drafts.",
           "9 Tokens go back to the scheduler, then stream to you."]
    for i, l in enumerate(leg):
        col, row = divmod(i, 5)
        b += num(36 + col * 600, 832 + row * 30, l[0])
        b += txt(56 + col * 600, 836 + row * 30, l[2:], "n")
    return svg(W, H, "The whole vllm-metal system on one diagram, from the program to the unified memory", b, minw=1000)

# ---------------------------------------------------------------- write standalone SVG and PNG files
STYLE = """.n { font: 600 13px -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; fill: #17202b; }
.d { font: 400 11.5px -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; fill: #576375; }
.cm { font: 500 12px ui-monospace, Menlo, Consolas, monospace; fill: #17202b; }
.h { font: 700 11.5px -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; letter-spacing: .07em; text-transform: uppercase; }
.ar { stroke: #576375; stroke-width: 1.5; fill: none; marker-end: url(#arr); }
.arb { stroke: #2a66c9; stroke-width: 1.8; fill: none; stroke-dasharray: 5 4; marker-end: url(#arrb); }
.tick { stroke: #17202b; stroke-width: 1.5; }
.badge { fill: #17202b; }
.badge-t { font: 700 11px -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; fill: #f5f6f8; }
.b-vllm { fill: #e5eeff; stroke: #2a66c9; stroke-width: 1.3; }
.b-glue { fill: #efe8ff; stroke: #7448c9; stroke-width: 1.3; }
.b-runner { fill: #fff1dc; stroke: #b56f0c; stroke-width: 1.3; }
.b-attn { fill: #e0f4ea; stroke: #1b8455; stroke-width: 1.3; }
.b-attn2 { fill: #eef9f3; stroke: #1b8455; stroke-width: 1.3; stroke-dasharray: 4 3; }
.b-kern { fill: #fde8e6; stroke: #c2433a; stroke-width: 1.3; }
.b-mlx { fill: #ddf3f5; stroke: #0d7f8c; stroke-width: 1.3; }
.b-hw { fill: #e9edf2; stroke: #5a6577; stroke-width: 1.3; }
.b-prog { fill: #f1f3f6; stroke: #3f4a5a; stroke-width: 1.3; }
.b-soft { fill: #edf0f4; stroke: #c6ced9; stroke-width: 1.2; }
[class^="lane-"] { stroke-width: 1; fill-opacity: .45; }
.lane-vllm { fill: #e5eeff; stroke: #2a66c9; stroke-opacity: .35; }
.lane-glue { fill: #efe8ff; stroke: #7448c9; stroke-opacity: .35; }
.lane-runner { fill: #fff1dc; stroke: #b56f0c; stroke-opacity: .35; }
.lane-attn { fill: #e0f4ea; stroke: #1b8455; stroke-opacity: .35; }
.lane-kern { fill: #fde8e6; stroke: #c2433a; stroke-opacity: .35; }
.lane-mlx { fill: #ddf3f5; stroke: #0d7f8c; stroke-opacity: .35; }
.lane-hw { fill: #e9edf2; stroke: #5a6577; stroke-opacity: .35; }
.lane-prog { fill: #f1f3f6; stroke: #3f4a5a; stroke-opacity: .35; }
.k-vllm { fill: #2a66c9; } .k-glue { fill: #7448c9; } .k-runner { fill: #b56f0c; }
.k-attn { fill: #1b8455; } .k-kern { fill: #c2433a; } .k-mlx { fill: #0d7f8c; }
.k-hw { fill: #5a6577; } .k-prog { fill: #3f4a5a; }
.mk { fill: #576375; } .mkb { fill: #2a66c9; }"""
DEFS = """<defs>
    <marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path fill="#576375" d="M0,0 L9,4.5 L0,9 z"/></marker>
    <marker id="arrb" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path fill="#2a66c9" d="M0,0 L9,4.5 L0,9 z"/></marker>
  </defs>"""
NAMES = [("01-the-layers", fig_layers), ("02-startup", fig_startup), ("03-one-step", fig_step),
         ("04-decode-pipeline", fig_pipeline), ("05-attention-wrapper", fig_attn), ("06-kv-cache-layout", fig_kv),
         ("07-kernel-choice", fig_dispatch), ("08-spec-decode", fig_spec), ("09-the-whole-picture", fig_all)]
import re, subprocess
for name, f in NAMES:
    s = re.search(r"<svg viewBox.*?</svg>", f(), re.S).group(0)
    w, h = re.search(r'viewBox="0 0 (\d+) (\d+)"', s).groups()
    s = re.sub(r' style="min-width:\d+px"', "", s)
    s = s.replace("<svg ", f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" ', 1)
    s = s.replace(">", f'><style>{STYLE}</style>{DEFS}<rect width="100%" height="100%" fill="#ffffff"/>', 1)
    path = os.path.join(OUT, name + ".svg")
    open(path, "w").write(s + "\n")
    subprocess.run(["rsvg-convert", "-w", "1600", path, "-o", os.path.join(OUT, "..", name + ".png")], check=True)
