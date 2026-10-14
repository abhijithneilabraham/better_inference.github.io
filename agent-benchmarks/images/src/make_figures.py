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

# 01 two topics
def col(x, k, title, rows):
    out = f'<rect class="{k}" x="{x}" y="56" width="360" height="250" rx="6"/><text class="n" x="{x+16}" y="84">{title}</text>'
    for i, (a, b) in enumerate(rows):
        y = 116 + i * 38
        out += f'<text class="d" x="{x+16}" y="{y}">{a}</text><text class="n" x="{x+16}" y="{y+17}" style="font-size:12px">{b}</text>'
    return out
body = col(24, "amber", "Part A: how good is the coding agent?", [
    ("the question", "Does it fix the bug? Does it finish the task?"),
    ("what changes the answer", "the model, the harness, the server settings"),
    ("unit", "pass@1: percent of tasks solved"),
    ("tools", "SWE-bench, Terminal-Bench, Aider, LiveCodeBench"),
    ("leaderboards", "SWE-bench, Terminal-Bench, Aider, LMArena")])
body += col(416, "blue", "Part B: how fast does the server make tokens?", [
    ("the question", "How many tokens per second, for each user and in total?"),
    ("what changes the answer", "the GPU, the engine (vLLM, SGLang), its settings"),
    ("unit", "TPS, TTFT, total throughput"),
    ("tools", "vllm bench serve, sglang.bench_serving"),
    ("leaderboards", "Artificial Analysis, OpenRouter, MLPerf")])
body += ('<text class="cap" x="24" y="336">Two separate measurements. A model can be clever and slow, or fast and weak. TPS is the same idea for chat,</text>'
         '<text class="cap" x="24" y="356">agents, RAG or batch jobs. Part C shows the one place where the two meet: agent task time.</text>')
svg("02-two-topics", 800, 372, "Two separate topics: coding agent quality and LLM inference speed, with the question, unit, tools and leaderboards of each", "Two topics in one article", "OVERVIEW", body)

# 02 two kinds of TPS (example numbers)
rows = [(1, 120, 120), (8, 95, 760), (32, 60, 1920), (128, 25, 3200)]
body = ('<text class="m s" x="24" y="66">one GPU, one model, more users at the same time (example numbers; the shape is typical)</text>'
        '<text class="n" x="24" y="96" style="font-size:12px">users</text>'
        '<text class="n" x="110" y="96" style="font-size:12px">speed for each user (tok/s)</text>'
        '<text class="n" x="440" y="96" style="font-size:12px">total for the server (tok/s)</text>')
for i, (u, per, tot) in enumerate(rows):
    y = 110 + i * 46
    body += (f'<text class="n" x="24" y="{y+22}">{u}</text>'
             f'<rect class="amber" x="110" y="{y}" width="{per*2.2:.0f}" height="30" rx="3"/><text class="m w" x="{118+per*2.2:.0f}" y="{y+20}">{per}</text>'
             f'<rect class="blue" x="440" y="{y}" width="{tot*0.085:.0f}" height="30" rx="3"/><text class="m w" x="{448+tot*0.085:.0f}" y="{y+20}">{tot}</text>')
body += ('<line class="rule" x1="24" y1="300" x2="776" y2="300"/>'
         '<text class="cap" x="24" y="326">More users: each user gets slower, but the server makes many more tokens in total. A speed board shows the left</text>'
         '<text class="cap" x="24" y="346">column. Your GPU bill depends on the right column. Always say which TPS you mean, and at how many users.</text>')
svg("03-two-kinds-of-tps", 800, 362, "Per-user TPS falls and total server throughput rises as more users share one GPU, with example numbers", "Two kinds of TPS", "THROUGHPUT", body)

# 03 the TPS ceiling
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
svg("04-the-tps-ceiling", 800, 370, "The decode speed ceiling for one request: memory bandwidth divided by the bytes of weights read per token, for bf16, FP8 and FP4 weights of a 31B model", "The decode speed ceiling", "ROOFLINE", body)

# 04 where agent time goes (example numbers)
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
svg("05-where-agent-time-goes", 800, 370, "Where the time of one agent task goes: TTFT, decode and tool work for each turn, before and after optimisation, with example numbers", "Where the time of one agent task goes", "AGENT TIME", body)

# 01a / 01b the leaderboards
def boards(name, title, tag, label, items, legend):
    body = ('<text class="n" x="24" y="66" style="font-size:12px">leaderboard</text>'
            '<text class="n" x="200" y="66" style="font-size:12px">what it ranks</text>'
            '<text class="n" x="510" y="66" style="font-size:12px">who runs the test</text>')
    for i, (a, b, c, k) in enumerate(items):
        y = 78 + i * 40
        body += (f'<rect class="{k}" x="24" y="{y}" width="752" height="32" rx="4"/>'
                 f'<text class="n" x="38" y="{y+21}" style="font-size:12.5px">{a}</text>'
                 f'<text class="d" x="200" y="{y+21}" style="fill:#dfe6ee">{b}</text><text class="d" x="510" y="{y+21}">{c}</text>')
    h = 78 + len(items) * 40 + 30
    body += f'<text class="cap" x="24" y="{h}">{legend}</text>'
    svg(name, 800, h + 16, label, title, tag, body)

boards("01a-coding-agent-leaderboards", "Coding agent leaderboards", "AGENT QUALITY",
       "Five coding agent leaderboards, what each ranks and who runs the test",
       [("SWE-bench", "real GitHub issues fixed", "you run it, then submit logs", "amber"),
        ("Terminal-Bench", "tasks in a real terminal", "you run it, then submit", "amber"),
        ("Aider Polyglot", "225 exercises, 6 languages, cost", "you run it, then open a PR", "amber"),
        ("LiveCodeBench", "new contest problems, by date", "maintainers or you run it", "purple"),
        ("LMArena WebDev", "human votes between two answers", "users vote; the team adds models", "good")],
       "amber: you measure, they check. purple: mixed. green: users vote.")
boards("01b-inference-engine-leaderboards", "Inference engine leaderboards", "INFERENCE SPEED",
       "Three inference engine leaderboards, what each ranks and who runs the test",
       [("Artificial Analysis", "speed for one user, TTFT, price", "the leaderboard team measures you", "blue"),
        ("OpenRouter", "live provider speed and latency", "measured from real traffic", "good"),
        ("MLPerf Inference", "total throughput of hardware + software", "you run it; MLCommons reviews", "amber")],
       "blue: they measure you. green: real traffic decides. amber: you measure, they check.")
# 06 my Terminal-Bench 2.1 architecture
def node(x, y, w, h, k, title, sub=""):
    out = f'<rect class="{k}" x="{x}" y="{y}" width="{w}" height="{h}" rx="5"/><text class="n" x="{x+w/2:.0f}" y="{y+22}" text-anchor="middle">{title}</text>'
    if sub:
        out += f'<text class="d" x="{x+w/2:.0f}" y="{y+40}" text-anchor="middle">{sub}</text>'
    return out
def arrow(x1, y1, x2, y2, c="arrow"):
    return f'<line class="{c}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>'
body = ('<text class="m s" x="24" y="66">Terminal-Bench 2.1, one graded answer per task: 86% (72 of the 84 tasks that Artificial Analysis grades)</text>'
        + node(250, 84, 300, 50, "box", "1. Solve: GPT-5.6-sol (max effort)", "Terminus 2 harness, clean snapshot of the task")
        + arrow(400, 134, 400, 156)
        + node(250, 158, 300, 50, "purple", "2. Judge: Claude Fable 5", "runs the work; Opus 4.8 if Fable refuses")
        + arrow(550, 183, 612, 183, "arrowg") + '<text class="m g" x="560" y="176">PASS</text>'
        + node(616, 158, 160, 50, "good", "ship it", "")
        + arrow(400, 208, 400, 230, "arrowr") + '<text class="m r" x="410" y="224">FAIL</text>'
        + node(250, 232, 300, 50, "amber", "3. One backup, from a clean copy", "by task type: Fable, then Opus, then Grok 4.5")
        + arrow(400, 282, 400, 304)
        + node(250, 306, 300, 50, "purple", "4. Judge the backup: GPT-5.6-sol", "a different family from the backup")
        + arrow(550, 331, 612, 331, "arrowg") + '<text class="m g" x="560" y="324">PASS</text>'
        + node(616, 306, 160, 50, "good", "ship the backup", "")
        + arrow(400, 356, 400, 378, "arrowr") + '<text class="m r" x="410" y="372">FAIL</text>'
        + node(250, 380, 300, 50, "blue", "5. Salvage: ship the first answer", "a FAIL often means \"could not verify\"")
        + node(24, 158, 196, 124, "box", "time guard", "")
        + '<text class="d" x="122" y="206" text-anchor="middle">past 60% of the time:</text><text class="d" x="122" y="222" text-anchor="middle">no new backup</text>'
        + '<text class="d" x="122" y="246" text-anchor="middle">past 90%: ship what</text><text class="d" x="122" y="262" text-anchor="middle">is on disk</text>'
        + '<text class="cap" x="24" y="462">Every judge comes from a different model family than the solver it checks, and decides PASS or FAIL by running the work.</text>')
svg("06-my-tb21-architecture", 800, 480, "My Terminal-Bench 2.1 agent: GPT-5.6-sol solves, Claude Fable 5 judges, one domain-routed backup from a clean copy judged by GPT-5.6-sol, salvage of the first answer, and a time guard", "My Terminal-Bench 2.1 agent", "86% ON TB 2.1", body)
print("done")
