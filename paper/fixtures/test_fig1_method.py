"""`paper/figures/fig1_method.tex`: the method figure compiles standalone at the text column's printed width.

The figure is drawn at 5.5 in (`\\linewidth` under corl_2026.sty) so `\\includegraphics[width=\\linewidth]` prints
its labels at their nominal 7 and 8 pt; a page of any other width would scale every label. It sets the style's
text font (Times) and must embed no Type 3 font.

    python -m pytest paper/fixtures/test_fig1_method.py -q
"""
import os
import re
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(os.path.dirname(HERE), "figures")
SOURCE = os.path.join(FIGURES, "fig1_method.tex")
TEXT_WIDTH_IN = 5.5

pytestmark = pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")


@pytest.fixture(scope="module")
def pdf(tmp_path_factory):
    out = tmp_path_factory.mktemp("fig1")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={out}",
                        SOURCE], cwd=FIGURES, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-2000:]
    return os.path.join(out, "fig1_method.pdf")


def test_the_page_is_the_text_column_wide(pdf):
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", open(pdf, "rb").read())
    assert float(m.group(1)) / 72 == pytest.approx(TEXT_WIDTH_IN, abs=0.01)


def test_the_text_font_is_the_styles_times_and_no_font_is_type3(pdf):
    data = open(pdf, "rb").read()
    assert b"/Type3" not in data
    assert b"NimbusRomNo9L" in data or b"Times" in data


def producer(data):
    m = re.search(rb"/Producer\s*\(([^)]*)\)", data)
    return m.group(1) if m else None


def test_the_committed_pdf_is_the_build_of_the_committed_source(pdf):
    # the source omits the creation date and trailer id, so an unchanged figure rebuilds byte-identical, but
    # only under the pdfTeX that wrote the committed file; another version changes bytes, not the figure
    built, committed = open(pdf, "rb").read(), open(os.path.join(FIGURES, "fig1_method.pdf"), "rb").read()
    if producer(built) != producer(committed):
        pytest.skip(f"committed PDF from {producer(committed)!r}, this build from {producer(built)!r}")
    assert built == committed, "paper/figures/fig1_method.pdf is stale: rebuild it from fig1_method.tex"
