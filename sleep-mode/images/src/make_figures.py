"""Render the figures for this article. Run: python3 make_figures.py (needs rsvg-convert)."""
import os, subprocess
OUT = os.path.dirname(os.path.abspath(__file__))
STYLE = open(os.path.join(OUT, "_style.css")).read()
DEFS = open(os.path.join(OUT, "_defs.svg")).read()

def svg(name, w, h, label, title, tag, body):
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{label}">\n'
         f'<style>{STYLE}</style>{DEFS}\n<rect class="bg" width="{w}" height="{h}" rx="6"/>\n'
         f'<text class="h" x="24" y="34">{title}</text>\n<text class="t" x="{w-24}" y="34" text-anchor="end">{tag}</text>\n{body}\n</svg>\n')
    p = os.path.join(OUT, name + ".svg")
    open(p, "w").write(s)
    subprocess.run(["rsvg-convert", "-w", "1600", p, "-o", os.path.join(OUT, "..", name + ".png")], check=True)

# 01 one memory, before and after sleep
def bar(y, title, segs):
    out = f'<text class="n" x="24" y="{y-10}">{title}</text>'
    x = 24
    for w, cls, a, b in segs:
        out += f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="44" rx="4"/>'
        out += f'<text class="n" x="{x+10}" y="{y+19}" style="font-size:12px">{a}</text><text class="d" x="{x+10}" y="{y+35}">{b}</text>'
        x += w
    return out
body = '<text class="m s" x="24" y="66">one Mac with 8 GB of unified memory (example; the model is Qwen3-0.6B)</text>'
body += bar(104, "Awake, serving", [
    (300, "box", "macOS and your other apps", "always there"),
    (200, "blue", "model weights", "about 1.2 GB"),
    (252, "amber", "KV cache pool", "grows as it is used")])
body += bar(194, "After sleep(level=1)", [
    (300, "box", "macOS and your other apps", "always there"),
    (200, "blue", "model weights", "stay in place"),
    (252, "good", "given back to macOS", "0.65 GiB freed in the test")])
body += bar(284, "After wake_up()", [
    (300, "box", "macOS and your other apps", "always there"),
    (200, "blue", "model weights", "never moved"),
    (252, "amber", "new, empty KV cache pool", "made again in 0.015 s")])
body += ('<line class="rule" x1="24" y1="348" x2="776" y2="348"/>'
         '<text class="cap" x="24" y="372">The weights never move, because on a Mac there is no second, cheaper memory to move them to. Only the KV</text>'
         '<text class="cap" x="24" y="392">cache goes away, and it comes back empty. The first request after wake_up() has to fill it again.</text>')
svg("01-one-memory-before-and-after", 800, 408, "Unified memory of a Mac while the engine is awake, after sleep level 1, and after wake_up", "One memory, before and after sleep", "MEMORY", body)

# 02 what the sleep levels mean
def card(x, y, w, h, cls, title, lines):
    out = f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/><text class="n" x="{x+14}" y="{y+24}">{title}</text>'
    for i, l in enumerate(lines):
        out += f'<text class="d" x="{x+14}" y="{y+46+i*17}" style="font-size:12px">{l}</text>'
    return out
body = '<text class="n" x="24" y="68">vLLM on a GPU server</text><text class="n" x="416" y="68">vllm-metal on a Mac</text>'
body += card(24, 80, 360, 96, "blue", "level 1", [
    "weights: copied from the GPU to CPU memory",
    "KV cache: thrown away",
    "wake_up copies the weights back (seconds)"])
body += card(24, 190, 360, 96, "purple", "level 2", [
    "weights: thrown away",
    "KV cache: thrown away",
    "wake_up loads the weights from disk again"])
body += card(416, 80, 360, 96, "good", "level 1", [
    "weights: stay where they are",
    "KV cache: thrown away, memory goes back to macOS",
    "wake_up makes a new empty KV cache (well under 1 s)"])
body += card(416, 190, 360, 96, "box", "level 2", [
    "not supported yet",
    "needs a way to load the weights again",
    "the call fails with a clear error"])
body += ('<line class="rule" x1="24" y1="306" x2="776" y2="306"/>'
         '<text class="cap" x="24" y="330">On a GPU, "CPU memory" is a second, cheaper place, so moving the weights there frees GPU memory. On a Mac the</text>'
         '<text class="cap" x="24" y="350">CPU and the GPU share one memory, so the copy would free nothing. That is why level 1 means less on a Mac.</text>')
svg("02-sleep-levels", 800, 366, "What sleep level 1 and level 2 do on a GPU server and on a Mac with vllm-metal", "What the sleep levels mean", "LEVELS", body)

# 03 the path inside, for sleep and for wake
def step(x, y, w, cls, a, b):
    return (f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="50" rx="5"/>'
            f'<text class="m w" x="{x+10}" y="{y+20}">{a}</text><text class="d" x="{x+10}" y="{y+38}">{b}</text>')
body = '<text class="n" x="24" y="68">sleep</text>'
body += step(24, 80, 170, "box", "LLM.sleep(level=1)", "what you call")
body += '<path class="arrow" d="M194,105 L214,105"/>'
body += step(216, 80, 170, "box", "MetalWorker.sleep", "level must be 1")
body += '<path class="arrow" d="M386,105 L406,105"/>'
body += step(408, 80, 170, "amber", "runner.release_kv_", "refuses mid-step")
body += '<path class="arrow" d="M578,105 L598,105"/>'
body += step(600, 80, 176, "good", "storage.release()", "buffers = [], memory freed")
body += '<text class="n" x="24" y="176">wake</text>'
body += step(24, 188, 170, "box", "LLM.wake_up()", "what you call")
body += '<path class="arrow" d="M194,213 L214,213"/>'
body += step(216, 188, 170, "box", "MetalWorker.wake_up", "tag kv_cache")
body += '<path class="arrow" d="M386,213 L406,213"/>'
body += step(408, 188, 170, "amber", "runner.restore_kv_", "same config as before")
body += '<path class="arrow" d="M578,213 L598,213"/>'
body += step(600, 188, 176, "good", "storage.reallocate()", "checks the layout is equal")
body += ('<line class="rule" x1="24" y1="262" x2="776" y2="262"/>'
         '<text class="cap" x="24" y="286">The scheduler keeps its block tables through the sleep. That only works because the new pool has the same size and</text>'
         '<text class="cap" x="24" y="306">the same region offsets as the old one, so block 17 still means the same place. reallocate() checks this and stops if not.</text>')
svg("03-inside-sleep-and-wake", 800, 322, "The chain of calls inside vllm-metal for sleep and for wake, ending at the KV cache storage", "The path inside, for sleep and for wake", "CODE", body)
