"""Build docs/researcher_dossier.html from docs/RESEARCHER_DOSSIER.md.

A small, dependency-free Markdown-to-HTML converter for the subset the dossier
uses: headings, paragraphs, bullet and numbered lists, pipe tables, fenced code,
horizontal rules, block images and inline code, bold, italic and links. Images
are embedded as data URIs so the page is a single file; a sibling named
``<stem>_dark.<ext>`` is shown instead in dark mode. Every heading gets a unique
id, and parts (``#``) and sections (``##``) are listed in a table of contents.

Usage: python tools/build_dossier.py [--src docs/RESEARCHER_DOSSIER.md] [--out docs/researcher_dossier.html]
"""

from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import re
from pathlib import Path

CSS = """
:root{--bg:#fbfaf6;--ink:#242a2a;--muted:#626b6b;--line:#d8dcd5;--surface:#f0f1eb;--accent:#225b65;--body:'Source Serif 4',Georgia,'Times New Roman',serif;--sans:'IBM Plex Sans',Arial,Helvetica,sans-serif;color-scheme:light}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#171d1f;--ink:#e0e5e2;--muted:#a6b2b2;--line:#3b484b;--surface:#222c2e;--accent:#91cbd0;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#171d1f;--ink:#e0e5e2;--muted:#a6b2b2;--line:#3b484b;--surface:#222c2e;--accent:#91cbd0;color-scheme:dark}
*{box-sizing:border-box}html{scroll-padding-top:2rem}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--body);font-size:18px;line-height:1.65;font-variant-numeric:lining-nums tabular-nums}
a{color:var(--accent);text-underline-offset:.18em;overflow-wrap:anywhere}
a:focus-visible,.table-wrap:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:4px}
main{max-width:1140px;margin:auto;padding:64px 32px 100px;min-width:0}
main>p,main>ul,main>ol,main>h1,main>h2,main>h3,main>pre,main>hr,nav{max-width:760px;margin-left:auto;margin-right:auto}
h1,h2,h3{text-wrap:balance;line-height:1.2}
h1{font-size:2.2rem;margin-top:3.2rem;padding-top:1.2rem;border-top:1px solid var(--line)}
h1.title{font-size:clamp(2.2rem,5vw,3.4rem);border:0;margin-top:0;padding-top:0;letter-spacing:-.02em}
h2{font-size:1.45rem;margin-top:2.4rem}h3{font-size:1.15rem;margin-top:2rem}
p{margin:1.05em auto}strong{font-weight:600}
code{font-size:.82em;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}
pre{white-space:pre-wrap;background:var(--surface);padding:16px;border:1px solid var(--line)}
hr{border:0;border-top:1px solid var(--line);margin:2.5rem auto}
nav{font-family:var(--sans);font-size:14px;padding:18px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin-top:24px;margin-bottom:24px}
nav ol{list-style:none;margin:0;padding:0}nav>ol>li{margin:.45em 0}nav ol ol{padding-left:1.2em;font-size:13px;color:var(--muted)}
nav a{text-decoration:none}
.table-wrap{max-width:100%;overflow-x:auto;margin:22px 0 30px;border-bottom:1px solid var(--line)}
table{border-collapse:collapse;font-family:var(--sans);font-size:13px;line-height:1.5;min-width:100%}
th,td{padding:8px 11px;border-top:1px solid var(--line);vertical-align:top;text-align:left}
th{font-weight:600;background:var(--surface)}td.num,th.num{text-align:right}
figure{margin:30px 0 36px;padding:16px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
figure img{display:block;max-width:100%;height:auto;margin:auto}.fig-has-dark img{max-width:min(100%,680px)}
figcaption{font-family:var(--sans);font-size:13px;color:var(--muted);margin:12px auto 0;max-width:95ch}
.fig-dark{display:none}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .fig-has-dark .fig-light{display:none}:root:not([data-theme="light"]) .fig-has-dark .fig-dark{display:block}}
:root[data-theme="dark"] .fig-has-dark .fig-light{display:none}:root[data-theme="dark"] .fig-has-dark .fig-dark{display:block}
.theme{position:fixed;top:12px;right:12px;font:13px var(--sans);background:var(--surface);color:var(--ink);border:1px solid var(--line);padding:6px 10px;cursor:pointer}
@media(max-width:600px){body{font-size:17px}main{padding:32px 16px 60px}h1{font-size:1.7rem}h2{font-size:1.25rem}th,td{padding:7px}}
@media print{.theme,nav{display:none}.table-wrap{overflow:visible}main{padding:0}}
"""

SCRIPT = """
(function(){var b=document.querySelector('.theme');var r=document.documentElement;
function cur(){return r.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light')}
b.addEventListener('click',function(){r.dataset.theme=cur()==='dark'?'light':'dark';});})();
"""

FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600"
         "&family=Source+Serif+4:ital,wght@0,400;0,600;0,700;1,400&display=swap")


def slugify(text: str, used: set[str]) -> str:
    """Return a unique, URL-safe id for a heading's plain text."""
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"
    slug, n = base, 2
    while slug in used:
        slug, n = f"{base}-{n}", n + 1
    used.add(slug)
    return slug


def inline(text: str) -> str:
    """Convert inline Markdown (code, links, bold, italic) to escaped HTML."""
    codes: list[str] = []

    def stash(m: re.Match) -> str:
        codes.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text, quote=False)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)",
                  lambda m: f'<a href="{html.escape(safe_href(m.group(2)))}">{m.group(1)}</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], text)


def plain(text: str) -> str:
    """Strip inline Markdown for use in ids and the table of contents."""
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return re.sub(r"\*\*?|\[|\]\([^)]*\)", "", text)


def split_row(line: str) -> list[str]:
    """Split a pipe-table row on unescaped pipes and unescape the rest."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", s)]


ROOT = Path(__file__).resolve().parent.parent
SAFE_SCHEMES = ("http://", "https://", "mailto:", "#")


def safe_href(url: str) -> str:
    """Only web, mail, fragment and relative links are rendered; any other scheme becomes a fragment."""
    u = url.strip()
    if u.startswith(SAFE_SCHEMES) or ":" not in u.split("/", 1)[0]:
        return u
    return "#"


def data_uri(path: Path) -> str:
    """Inline a figure that lives inside the repository; anything outside it is refused."""
    if ROOT not in path.resolve().parents:
        raise SystemExit(f"figure outside the repository refused: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def figure(alt: str, src: str, md_dir: Path) -> str:
    """Embed a block image, with its ``_dark`` sibling for dark mode when present."""
    path = (md_dir / src).resolve()
    if not path.exists():
        raise SystemExit(f"figure not found: {path}")
    dark = path.with_name(f"{path.stem}_dark{path.suffix}")
    alt_e = html.escape(alt)
    if dark.exists():
        imgs = (f'<img class="fig-light" src="{data_uri(path)}" alt="{alt_e}">'
                f'<img class="fig-dark" src="{data_uri(dark)}" alt="{alt_e}">')
        cls = ' class="fig-has-dark"'
    else:
        imgs, cls = f'<img src="{data_uri(path)}" alt="{alt_e}">', ""
    return f"<figure{cls}>{imgs}<figcaption>{alt_e}</figcaption></figure>"


def table(rows: list[str]) -> str:
    head = split_row(rows[0])
    aligns = [("num" if c.strip().endswith(":") else "") for c in split_row(rows[1])]
    out = ['<div class="table-wrap" tabindex="0"><table><thead><tr>']
    out += [f'<th class="{aligns[i] if i < len(aligns) else ""}">{inline(c)}</th>' for i, c in enumerate(head)]
    out.append("</tr></thead><tbody>")
    for r in rows[2:]:
        cells = split_row(r)
        out.append("<tr>" + "".join(
            f'<td class="{aligns[i] if i < len(aligns) else ""}">{inline(c)}</td>' for i, c in enumerate(cells)) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out).replace(' class=""', "")


def convert(md: str, md_dir: Path) -> tuple[str, list[tuple[int, str, str]], str]:
    """Return (body html, headings as (level, id, text), document title)."""
    lines = md.splitlines()
    out: list[str] = []
    heads: list[tuple[int, str, str]] = []
    used: set[str] = set()
    title = "DoomDiT Researcher Dossier"
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(lines[i + 1:j])) + "</code></pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,3}) (.+)$", line)
        if m:
            level, text = len(m.group(1)), m.group(2).strip()
            if not heads and level == 1 and not out:
                title = plain(text)
                out.append(f'<h1 class="title">{inline(text)}</h1>')
                heads.append((0, "", text))
            else:
                hid = slugify(plain(text), used)
                heads.append((level, hid, plain(text)))
                out.append(f'<h{level} id="{hid}">{inline(text)}</h{level}>')
            i += 1
            continue
        if re.match(r"^-{3,}$", line.strip()):
            out.append("<hr>")
            i += 1
            continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", line)
        if m:
            out.append(figure(m.group(1), m.group(2), md_dir))
            i += 1
            continue
        if line.lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            out.append(table(lines[i:j]))
            i = j
            continue
        m = re.match(r"^(\s*)([-*]|\d+\.) ", line)
        if m:
            ordered = m.group(2)[0].isdigit()
            items: list[str] = []
            while i < len(lines) and lines[i].strip():
                mm = re.match(r"^\s*([-*]|\d+\.) (.*)$", lines[i])
                if mm:
                    items.append(mm.group(2))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(t)}</li>" for t in items) + f"</{tag}>")
            continue
        para = [line.strip()]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,3} |\||```|!\[|\s*([-*]|\d+\.) |-{3,}$)", lines[i]):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out), heads, title


def toc(heads: list[tuple[int, str, str]]) -> str:
    """Nested list of parts (h1) with their sections (h2)."""
    parts: list[str] = []
    open_sub = False
    for level, hid, text in heads:
        if level == 1:
            if open_sub:
                parts.append("</ol></li>")
                open_sub = False
            elif parts:
                parts.append("</li>")
            parts.append(f'<li><a href="#{hid}">{html.escape(text)}</a>')
        elif level == 2 and parts:
            if not open_sub:
                parts.append("<ol>")
                open_sub = True
            parts.append(f'<li><a href="#{hid}">{html.escape(text)}</a></li>')
    parts.append("</ol></li>" if open_sub else "</li>")
    return '<nav aria-label="Contents"><strong>Contents</strong><ol>' + "".join(parts) + "</ol></nav>"


def build(src: Path, out: Path) -> None:
    body, heads, title = convert(src.read_text(encoding="utf-8"), src.parent)
    nav = toc(heads)
    first_part = body.find("<hr>")
    body = body[:first_part] + nav + body[first_part:] if first_part >= 0 else nav + body
    page = (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{html.escape(title)}</title>"
        f"<link rel=\"stylesheet\" href=\"{FONTS}\"><style>{CSS}</style></head>\n"
        "<body><button class=\"theme\" type=\"button\" aria-label=\"Toggle light and dark\">Light / dark</button>"
        f"<main>\n{body}\n</main><script>{SCRIPT}</script></body></html>\n"
    )
    out.write_text(page, encoding="utf-8")


def main() -> None:
    root = ROOT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=root / "docs/RESEARCHER_DOSSIER.md")
    ap.add_argument("--out", type=Path, default=root / "docs/researcher_dossier.html")
    args = ap.parse_args()
    build(args.src, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
