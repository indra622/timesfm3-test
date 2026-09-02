"""Render the standalone docs Markdown to PDF with markdown-it + KaTeX + headless Chrome.

Usage:
  uv run --with mdit-py-plugins python scripts/build_paper_pdf.py <paper.md> <out.pdf> <tmp.html>

Requires Google Chrome (headless) and network access for the KaTeX CDN.
"""
from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.footnote import footnote_plugin

SRC = Path(sys.argv[1])
OUT_PDF = Path(sys.argv[2])
HTML_OUT = Path(sys.argv[3])
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

md = (
    MarkdownIt("gfm-like", {"linkify": False})
    .use(footnote_plugin)
    .use(dollarmath_plugin, double_inline=False)
)


def math_inline(content: str) -> str:
    return f'<span class="math-inline">{html.escape(content)}</span>'


def math_block(content: str, *_args) -> str:
    return f'<div class="math-block">{html.escape(content)}</div>'


md.add_render_rule("math_inline", lambda self, tokens, idx, options, env: math_inline(tokens[idx].content))
md.add_render_rule("math_block", lambda self, tokens, idx, options, env: math_block(tokens[idx].content))

# make image paths absolute so the file:// page resolves them
docs_dir = SRC.parent.resolve()


def image_rule(self, tokens, idx, options, env):
    tok = tokens[idx]
    src = tok.attrGet("src")
    if src and not src.startswith(("http", "file:", "/")):
        tok.attrSet("src", (docs_dir / src).resolve().as_uri())
    alt = tok.content
    return f'<img src="{html.escape(tok.attrGet("src"))}" alt="{html.escape(alt)}">'


md.add_render_rule("image", image_rule)

body = md.render(SRC.read_text(encoding="utf-8"))

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
html { -webkit-print-color-adjust: exact; }
body { font-family: "Apple SD Gothic Neo", "Pretendard", "Noto Sans KR", "Helvetica Neue", Arial, sans-serif;
       font-size: 10.5pt; line-height: 1.55; color: #1b1f24; margin: 0; }
h1 { font-size: 20pt; color: #0f2a4a; border-bottom: 2px solid #1f4e79; padding-bottom: 4px; margin: 26px 0 12px; }
h1.title { font-size: 26pt; border: none; text-align: center; margin: 40mm 0 6px; }
h2.subtitle { font-size: 13pt; font-weight: 500; color: #34495e; text-align: center; border: none; margin: 0 0 30px; }
h1.break { page-break-before: always; }
h2 { font-size: 14.5pt; color: #1f3b5c; margin: 20px 0 8px; }
h3 { font-size: 12pt; color: #24425f; margin: 16px 0 6px; }
p { margin: 6px 0 8px; }
blockquote { border-left: 4px solid #1f4e79; background: #f2f6fa; margin: 10px 0; padding: 8px 12px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 9.2pt; page-break-inside: avoid; }
th, td { border: 1px solid #c9d3de; padding: 4px 6px; vertical-align: top; }
th { background: #e8eef5; color: #0f2a4a; }
code { font-family: "JetBrains Mono", "SF Mono", Menlo, monospace; font-size: 0.88em; background: #f3f4f6; padding: 0 3px; border-radius: 3px; }
pre { background: #f3f4f6; padding: 10px; border-radius: 4px; font-size: 8.8pt; overflow-x: auto; white-space: pre-wrap; word-break: break-all; }
pre code { background: none; padding: 0; }
img { max-width: 100%; display: block; margin: 10px auto 4px; page-break-inside: avoid; }
em { color: #333; }
p > em:only-child { display: block; font-size: 9.3pt; color: #444; margin-top: 2px; }
ul, ol { padding-left: 22px; }
li { margin: 2px 0; }
sup.footnote-ref a, .footnote-backref { text-decoration: none; color: #1f4e79; }
section.footnotes { font-size: 9pt; }
section.footnotes ol { padding-left: 20px; }
hr.footnotes-sep { display: none; }
.footnote-backref { display: none; }
.math-block { text-align: center; margin: 8px 0; }
"""

# Title page treatment: first h1/h2 become title/subtitle; remaining plain h1 break the page,
# except 초록 and 키워드 which stay in flow with the title.

body = re.sub(r"<h1>(.*?)</h1>", r'<h1 class="title">\1</h1>', body, count=1)
body = re.sub(r"<h2>(.*?)</h2>", r'<h2 class="subtitle">\1</h2>', body, count=1)


def h1_break(m):
    text = m.group(1)
    if text.startswith(("초록", "키워드")):
        return f"<h1>{text}</h1>"
    return f'<h1 class="break">{text}</h1>'


body = re.sub(r"<h1>(.*?)</h1>", h1_break, body)

page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>{html.escape(SRC.stem)}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<style>{CSS}</style></head><body>
{body}
<script>
document.querySelectorAll('.math-inline').forEach(el => {{ katex.render(el.textContent, el, {{throwOnError: false, displayMode: false}}); }});
document.querySelectorAll('.math-block').forEach(el => {{ katex.render(el.textContent, el, {{throwOnError: false, displayMode: true}}); }});
document.body.setAttribute('data-math-done', '1');
</script></body></html>"""
HTML_OUT.write_text(page, encoding="utf-8")

cmd = [
    CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
    "--run-all-compositor-stages-before-draw", "--virtual-time-budget=15000",
    f"--print-to-pdf={OUT_PDF}", HTML_OUT.resolve().as_uri(),
]
subprocess.run(cmd, check=True, capture_output=True, text=True)
print("pdf", OUT_PDF, OUT_PDF.stat().st_size, "bytes")
