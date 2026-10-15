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

# 02 sleep with KV offloading on
def row(y, title, segs):
    out = f'<text class="n" x="24" y="{y-10}">{title}</text>'
    for x, w, cls, a, b, dash in segs:
        st = ' style="stroke-dasharray:5 4"' if dash else ""
        out += f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="44" rx="4"{st}/>'
        out += f'<text class="n" x="{x+10}" y="{y+19}" style="font-size:12px">{a}</text><text class="d" x="{x+10}" y="{y+35}">{b}</text>'
    return out
body = ('<text class="t" x="24" y="68">UNIFIED MEMORY</text><text class="t" x="600" y="68">ON DISK</text>'
        '<line class="rule" x1="584" y1="56" x2="584" y2="352" style="stroke-dasharray:4 4"/>')
body += row(104, "Awake, serving", [
    (24, 150, "blue", "model weights", "in place", False),
    (174, 190, "amber", "KV cache pool", "in use", False),
    (364, 200, "purple", "host pool", "holds offloaded blocks", False),
    (600, 176, "box", "disk tier (optional)", "holds offloaded blocks", False)])
body += row(194, "After sleep(level=1)", [
    (24, 150, "blue", "model weights", "stay in place", False),
    (174, 190, "good", "given back to macOS", "KV cache pool freed", False),
    (364, 200, "purple", "host pool, emptied", "its memory is still held", True),
    (600, 176, "box", "disk tier", "keeps its blocks", False)])
body += row(284, "After wake_up()", [
    (24, 150, "blue", "model weights", "never moved", False),
    (174, 190, "amber", "new, empty KV pool", "same layout as before", False),
    (364, 200, "purple", "host pool", "fills again", False),
    (600, 176, "box", "disk tier", "old prompts restore", False)])
body += ('<line class="rule" x1="24" y1="352" x2="776" y2="352"/>'
         '<text class="cap" x="24" y="376">Moving the KV cache into the host pool at sleep would free nothing: both are in the same memory.</text>'
         '<text class="cap" x="24" y="396">Sleep empties the host pool but keeps its memory. A prompt stored before sleep still restores after</text>'
         '<text class="cap" x="24" y="416">wake: 2,320 tokens in the test, from a 2 GiB host pool with the disk tier.</text>')
svg("02-sleep-with-kv-offloading", 800, 432, "What happens to the KV cache pool, the offload host pool and the disk tier when vllm-metal sleeps and wakes", "Sleep with KV offloading on", "OFFLOAD", body)
