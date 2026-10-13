"""Render each table in tables.md as a figure. Run: python3 make_tables.py (needs rsvg-convert)."""
import os, re, subprocess
from html import escape
OUT = os.path.dirname(os.path.abspath(__file__))
STYLE = open(os.path.join(OUT, "_style.css")).read()
CW, CWB, LH = 6.4, 6.9, 17   # average char width (regular, bold) at 12px, line height

def clean(c):
    return re.sub(r"\*\*|`", "", c.strip())

def wrap(text, px, cw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if len(t) * cw > px and cur:
            lines.append(cur); cur = w
        else:
            cur = t
    return lines + [cur] if cur else lines or [""]

def render(name, title, rows):
    head, body = rows[0], rows[1:]
    ncol = len(head)
    W = 1000 if ncol >= 4 and max(len(" ".join(r)) for r in rows) > 160 else 800
    avail = W - 48
    nat = [min(max(len(r[c]) * (CWB if c == 0 or r is head else CW) for r in rows) + 24, 460) for c in range(ncol)]
    nat = [max(n, 90) for n in nat]
    widths = [n * avail / sum(nat) for n in nat] if sum(nat) > avail else [n + (avail - sum(nat)) / ncol for n in nat]
    y, svg = 58, ""
    for ri, r in enumerate(rows):
        bold = ri == 0
        cells = [wrap(r[c], widths[c] - 24, CWB if bold or c == 0 else CW) for c in range(ncol)]
        h = max(len(x) for x in cells) * LH + 16
        cls = "box" if ri == 0 else ("rowa" if ri % 2 else "rowb")
        svg += f'<rect class="{cls}" x="24" y="{y}" width="{avail}" height="{h}" rx="3"/>'
        x = 24
        for c in range(ncol):
            tc = "n" if bold or c == 0 else "c"
            for li, line in enumerate(cells[c]):
                svg += f'<text class="{tc}" x="{x+12:.0f}" y="{y+22+li*LH}">{escape(line)}</text>'
            x += widths[c]
        y += h + 4
    H = y + 14
    style = STYLE + ".rowa{fill:#1b2531}.rowb{fill:#161d26;stroke:#2e3a47;stroke-width:1}.c{font:400 12px -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;fill:#dfe6ee}.n{font-size:12px}"
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{escape(title)}">\n'
         f'<style>{style}</style>\n<rect class="bg" width="{W}" height="{H}" rx="6"/>\n'
         f'<text class="h" x="24" y="36">{escape(title)}</text>\n{svg}\n</svg>\n')
    p = os.path.join(OUT, name + ".svg")
    open(p, "w").write(s)
    subprocess.run(["rsvg-convert", "-w", str(W * 2), p, "-o", os.path.join(OUT, "..", name + ".png")], check=True)

src = open(os.path.join(OUT, "tables.md")).read()
for block in src.split("\n## ")[1:]:
    first, *lines = block.strip().split("\n")
    name, title = [x.strip() for x in first.split("|", 1)]
    rows = [[clean(c) for c in l.strip().strip("|").split("|")] for l in lines if l.startswith("|") and not re.match(r"^\|[-| :]+\|$", l)]
    render(name, title, rows)
print("done")
