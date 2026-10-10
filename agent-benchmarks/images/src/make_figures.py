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

# 01 where agent time goes (example numbers)
S = 1.6
def bar(y, parts):
    x, out = 150, ""
    for w, c, lab in parts:
        out += f'<rect class="{c}" x="{x:.0f}" y="{y}" width="{w*S:.0f}" height="34" rx="3"/>'
        if w * S > 46:
            out += f'<text class="m w" x="{x+w*S/2:.0f}" y="{y+22}" text-anchor="middle">{lab}</text>'
        x += w * S
    return out, x
b1, x1 = bar(90, [(45, "amber", "TTFT 45 s"), (200, "blue", "decode 200 s"), (90, "box", "tools 90 s")])
b2, x2 = bar(196, [(12, "amber", ""), (80, "blue", "decode 80 s"), (90, "box", "tools 90 s")])
body = (f'<text class="m s" x="24" y="66">one agent task: 30 turns, 400 output tokens per turn, 3 s of tool work per turn (example numbers)</text>'
        f'<text class="n" x="24" y="112">before</text>{b1}<text class="m w" x="{x1+10:.0f}" y="112">335 s</text>'
        f'<text class="d" x="150" y="146">TTFT 1.5 s per turn, 60 tokens per second</text>'
        f'<text class="n" x="24" y="218">after</text>{b2}<text class="m g" x="{x2+10:.0f}" y="218">182 s</text>'
        f'<text class="d" x="150" y="252">prefix cache: TTFT 0.4 s. speculative decoding + FP8: 150 tokens per second</text>'
        f'<line class="rule" x1="24" y1="276" x2="776" y2="276"/>'
        f'<text class="m w" x="24" y="304">task time = sum over turns of ( TTFT + output tokens / TPS ) + tool time</text>'
        f'<text class="cap" x="24" y="334">An agent does many turns. Each turn pays TTFT and decode again. Speed changes how many tasks finish</text>'
        f'<text class="cap" x="24" y="354">inside a benchmark time limit, and how long a developer waits.</text>')
svg("01-where-agent-time-goes", 800, 370, "Where the time of one agent task goes: TTFT, decode and tool work for each turn, before and after optimisation, with example numbers", "Where the time of one agent task goes", "AGENT TIME", body)

# 02 the TPS ceiling
rows = [("bf16 weights", "62 GB", 129, "blue"), ("FP8 weights", "31 GB", 258, "amber"), ("FP4 weights", "15.5 GB", 516, "good")]
body = '<text class="m s" x="24" y="66">example: 31B dense model, one GPU with about 8 TB/s memory bandwidth, one request</text>'
for i, (a, b, v, c) in enumerate(rows):
    y = 88 + i * 50
    body += (f'<text class="n" x="24" y="{y+22}">{a}</text><text class="m s" x="150" y="{y+22}">{b}</text>'
             f'<rect class="{c}" x="230" y="{y}" width="{v}" height="32" rx="3"/><text class="m w" x="{240+v}" y="{y+22}">{v} tok/s max</text>')
body += ('<line class="rule" x1="24" y1="252" x2="776" y2="252"/>'
         '<text class="m w" x="24" y="280">TPS ceiling = memory bandwidth / bytes read for each token</text>'
         '<text class="m w" x="24" y="302">            = 8,000 GB/s / 62 GB = 129 tokens per second (bf16)</text>'
         '<text class="cap" x="24" y="334">In decode, each token reads all active weights from memory one time. Fewer bytes give a higher ceiling.</text>'
         '<text class="cap" x="24" y="354">Speculative decoding accepts more than one token for each read, so it can go above this line.</text>')
svg("02-the-tps-ceiling", 800, 370, "The decode speed ceiling for one request: memory bandwidth divided by the bytes of weights read per token, for bf16, FP8 and FP4 weights of a 31B model", "The decode speed ceiling", "ROOFLINE", body)

# 03 the leaderboards
items = [("Artificial Analysis", "quality index, TPS, TTFT, price", "the leaderboard team runs it", "blue"),
         ("SWE-bench", "real GitHub issues fixed", "you run it, then submit logs", "amber"),
         ("Terminal-Bench", "tasks in a real terminal", "you run it, then submit", "amber"),
         ("Aider Polyglot", "225 exercises, 6 languages, cost", "you run it, then open a PR", "amber"),
         ("LiveCodeBench", "new contest problems, by date", "maintainers or you run it", "purple"),
         ("LMArena WebDev", "human votes between two answers", "users vote; the team adds models", "good"),
         ("OpenRouter", "use, plus live provider speed", "measured from real traffic", "good")]
body = ('<text class="n" x="24" y="66" style="font-size:12px">leaderboard</text>'
        '<text class="n" x="230" y="66" style="font-size:12px">what it ranks</text>'
        '<text class="n" x="520" y="66" style="font-size:12px">who runs the test</text>')
for i, (a, b, c, k) in enumerate(items):
    y = 78 + i * 40
    body += (f'<rect class="{k}" x="24" y="{y}" width="752" height="32" rx="4"/>'
             f'<text class="n" x="38" y="{y+21}" style="font-size:12.5px">{a}</text>'
             f'<text class="d" x="230" y="{y+21}" style="fill:#dfe6ee">{b}</text><text class="d" x="520" y="{y+21}">{c}</text>')
body += '<text class="cap" x="24" y="374">blue: they measure you. amber: you measure, they check. green: real users decide. purple: mixed.</text>'
svg("03-the-leaderboards", 800, 390, "Seven leaderboards for coding models and agents, what each ranks, and who runs the test", "Seven leaderboards and who runs the test", "LEADERBOARDS", body)
print("done")
