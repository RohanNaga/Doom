"""
The dossier builder passes hand-authored inline SVG figures through unchanged and refuses active content.

A raw `<figure>` block in `docs/RESEARCHER_DOSSIER.md` must reach the page byte for byte: escaping it would print
the SVG source as text. The pass-through is the one place raw HTML enters the page, so a block carrying a script,
an event handler, an element that is not drawing or caption markup, or a link out of the figure is refused, as `data_uri` refuses a figure from
outside the repository.

    python -m pytest paper/fixtures/test_build_dossier.py -q
"""
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import build_dossier  # noqa: E402

FIGURE = """<figure>
<div class="svg-wrap" tabindex="0"><svg viewBox="0 0 100 40" role="img" aria-label="A to B">
<defs><marker id="t-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>
<line x1="10" y1="20" x2="90" y2="20" stroke="currentColor" marker-end="url(#t-arrow)"/>
<text x="50" y="14" font-size="12" text-anchor="middle" fill="currentColor">z_{r-1} &amp; u_{r-1} * not *italic*</text>
</svg></div>
<figcaption>A claim with <code>code</code>.</figcaption>
</figure>"""


def body(md):
    return build_dossier.convert(md, REPO / "docs")[0]


def test_a_raw_figure_block_passes_through_verbatim():
    out = body(f"Before.\n\n{FIGURE}\n\nAfter *this*.\n")
    assert FIGURE in out
    assert out.count("<figure>") == 1
    assert "<p>Before.</p>" in out and "<p>After <em>this</em>.</p>" in out


def test_a_figure_directly_after_a_paragraph_line_is_not_swallowed_into_it():
    out = body(f"Lead-in sentence.\n{FIGURE}\n")
    assert "<p>Lead-in sentence.</p>" in out
    assert FIGURE in out


def test_html_outside_a_figure_block_is_still_escaped():
    out = body("A <b>bold</b> claim.\n")
    assert "&lt;b&gt;bold&lt;/b&gt;" in out


@pytest.mark.parametrize("bad", [
    '<script>alert(1)</script>',
    '<rect onload="alert(1)"/>',
    '<foreignObject><div>x</div></foreignObject>',
    '<a href="https://example.com"><text>x</text></a>',
    '<image href="data:image/png;base64,AAAA"/>',
    '<use xlink:href="javascript:alert(1)"/>',
    '<a href=javascript:alert(1)><text>x</text></a>',
    '<set attributeName="href" to="javascript:alert(1)"/>',
    '<animate attributeName="href" values="javascript:alert(1)"/>',
    '<meta http-equiv="refresh" content="0;url=https://example.com">',
    '<rect style="fill:url(https://example.com/x.svg)"/>',
])
def test_active_content_or_an_outside_reference_in_a_figure_is_refused(bad):
    block = FIGURE.replace("</svg>", bad + "</svg>")
    with pytest.raises(SystemExit):
        body(block)


def test_an_unterminated_figure_is_refused():
    with pytest.raises(SystemExit):
        body("<figure>\n<svg viewBox=\"0 0 1 1\"></svg>\n")


def test_the_committed_page_is_the_build_of_the_committed_markdown(tmp_path):
    out = tmp_path / "dossier.html"
    build_dossier.build(REPO / "docs/RESEARCHER_DOSSIER.md", out)
    assert out.read_bytes() == (REPO / "docs/researcher_dossier.html").read_bytes()
