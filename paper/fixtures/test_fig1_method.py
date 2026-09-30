"""`paper/figures/fig1_method.tex`: the method figure compiles standalone at the text column's printed width and
keeps the figure standard (`paper/FIGURE_STANDARDS.md`) and the round-1 review's agreed edits.

The figure is drawn at 5.5 in (`\\linewidth` under corl_2026.sty) and included at scale 1, so its type prints at its
nominal size: 7 pt text, 8 pt panel letters, 6 pt math scripts, nothing under the 6 pt floor. It sets the style's
text font (Times) and embeds no Type 3 font. The three backbone outlines carry the only colour (Okabe-Ito, as in
`paper/figstyle.py`); nothing is filled with a colour, hatched or rounded. The images are real frames: each asset
under `paper/figures/assets/` is re-derived here from the rollout strip it names.

    python -m pytest paper/fixtures/test_fig1_method.py -q
"""
import os
import re
import shutil
import subprocess
import sys
import zlib

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
FIGURES = os.path.join(PAPER, "figures")
ASSETS = os.path.join(FIGURES, "assets")
STRIPS = os.path.join(FIGURES, "figure1")
SOURCE = os.path.join(FIGURES, "fig1_method.tex")
TEXT_WIDTH_IN = 5.5
MIN_BP = 6 * 72 / 72.27          # the 6 pt floor in PDF points (TeX pt are 1/72.27 in)
SCENE_ROWS = 208                  # rows 0 to 207, the scored scene without the HUD

sys.path.insert(0, PAPER)
pytestmark = pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")


@pytest.fixture(scope="module")
def pdf(tmp_path_factory):
    out = tmp_path_factory.mktemp("fig1")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={out}",
                        SOURCE], cwd=FIGURES, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-2000:]
    return os.path.join(out, "fig1_method.pdf")


def content_streams(data):
    """The decoded page and form streams of a PDF, skipping images and embedded font programs."""
    out = []
    for m in re.finditer(rb"obj\s*<<(.*?)>>\s*stream\r?\n", data, re.S):
        head = m.group(1)
        if b"/Image" in head or b"/Length1" in head or b"/FlateDecode" not in head:
            continue
        end = data.find(b"endstream", m.end())
        out.append(zlib.decompressobj().decompress(data[m.end():end]))
    return out


def pdf_text(path):
    if shutil.which("pdftotext") is None:
        pytest.skip("pdftotext is not installed")
    r = subprocess.run(["pdftotext", path, "-"], capture_output=True, text=True, timeout=60)
    return " ".join(r.stdout.split())


def test_the_page_is_the_text_column_wide(pdf):
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", open(pdf, "rb").read())
    assert float(m.group(1)) / 72 == pytest.approx(TEXT_WIDTH_IN, abs=0.01)


def test_the_text_font_is_the_styles_times_and_no_font_is_type3(pdf):
    data = open(pdf, "rb").read()
    assert b"/Type3" not in data
    assert b"NimbusRomNo9L" in data or b"Times" in data


def test_nothing_prints_under_six_points(pdf):
    data = open(pdf, "rb").read()
    # Computer Modern's 5 pt design sizes appear only when a math script falls to 5 pt
    assert not re.search(rb"/[A-Z]{6}\+CM(MI|R|SY)5\b", data)
    sizes = {float(s) for body in content_streams(data) for s in re.findall(rb"([\d.]+)\s+Tf\b", body)}
    assert sizes, "no text operators found"
    assert min(sizes) >= MIN_BP - 1e-3, sorted(sizes)


def test_the_backbone_outlines_carry_the_only_colour_and_nothing_is_filled_with_it(pdf):
    import figstyle as fs

    backbone = {fs.BACKBONES[b].colour for b in fs.BACKBONE_ORDER}
    data = open(pdf, "rb").read()
    strokes, fills = set(), set()
    for body in content_streams(data):
        for r, g, b, op in re.findall(rb"(?<![\d.])([\d.]+) ([\d.]+) ([\d.]+) (RG|rg)\b", body):
            rgb = "#%02X%02X%02X" % tuple(round(float(v) * 255) for v in (r, g, b))
            if len({r, g, b}) > 1:                       # a grey (r = g = b) is not a colour
                (strokes if op == b"RG" else fills).add(rgb)
    assert strokes == backbone, strokes
    assert not fills, f"coloured fills {fills}"
    assert b"/PatternType" not in data, "hatching"      # pgf declares an empty /Pattern resource; a fill adds one
    code = "\n".join(re.split(r"(?<!\\)%", line)[0] for line in open(SOURCE).read().splitlines())
    assert "rounded corners" not in code
    assert not re.search(r"pattern\s*=|\\usetikzlibrary\{[^}]*patterns", code)


def test_the_panels_carry_the_agreed_labels(pdf):
    text = pdf_text(pdf)
    for phrase in ("4 training maps,", "500 episodes each", "13 unseen maps,", "24 each", "19 executed buttons",
                   "context and the noisy next latent stacked on channels", "10-step DDIM", "Adaptation",
                   "world model:", "map adaptation", "rank-16 LoRA,", "8 episodes)",
                   "decoder fine-tuning", "(MSE + 0.1 LPIPS)", "decoder D (stock)",
                   "Models and training"):
        assert phrase in text, phrase
    assert "arena" not in text.lower()          # Rohan's vocabulary: unseen maps and training maps, never arenas
    # provenance and the auxiliary quantities live in the caption (paper/FIGURES.md), not in the figure
    for gone in ("same decoder", "Arnold", "150 s", "gap to the", "zero-shot latent skill", "860M"):
        assert gone not in text, gone


def asset_source(name):
    """The strip and the reduction factor an asset name encodes: fig1_<stem>_truth0[_half].png."""
    m = re.fullmatch(r"fig1_(.+)_truth0(_half)?\.png", name)
    assert m, name
    return os.path.join(STRIPS, f"{m.group(1)}_strip.png"), 2 if m.group(2) else 1


def test_every_image_is_a_real_frame_cropped_from_its_strip():
    Image = pytest.importorskip("PIL.Image")
    names = sorted(n for n in os.listdir(ASSETS) if n.startswith("fig1_"))
    assert len(names) == 7                               # one gameplay frame and two stacks of three
    for name in names:
        strip, factor = asset_source(name)
        truth = Image.open(strip).convert("RGB").crop((0, 0, 320, SCENE_ROWS))
        want = truth.reduce(factor) if factor > 1 else truth
        got = Image.open(os.path.join(ASSETS, name)).convert("RGB")
        assert got.size == want.size and got.tobytes() == want.tobytes(), name


def test_the_source_draws_exactly_the_assets_on_disk():
    source = open(SOURCE).read()
    used = set(re.findall(r"assets/fig1_([A-Za-z0-9_]+?)_truth0\.png", source))
    used |= {f"{stem}_half" for stem in re.findall(r"\{(train_map\d+_[a-z_]+_ep\d+_s\d+|unseen_arena\d+_[a-z_]+_ep"
                                                   r"\d+_s\d+)\}", source)}
    on_disk = {re.fullmatch(r"fig1_(.+)_truth0(_half)?\.png", n).group(1) + ("_half" if n.endswith("_half.png")
               else "") for n in os.listdir(ASSETS) if n.startswith("fig1_")}
    assert used == on_disk


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


def test_the_150_alternate_is_the_same_figure_at_1_5_in(tmp_path):
    """`fig1_method_150.tex`, the 1.5 in alternate for the page-4 fit: the text column wide, 1.5 in tall, the same
    fonts and images, nothing under 6 pt, and its committed PDF is the build of its source."""
    source = os.path.join(FIGURES, "fig1_method_150.tex")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={tmp_path}",
                        source], cwd=FIGURES, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-2000:]
    built = open(os.path.join(tmp_path, "fig1_method_150.pdf"), "rb").read()
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", built)
    assert (float(m.group(1)) / 72, float(m.group(2)) / 72) == (pytest.approx(TEXT_WIDTH_IN, abs=0.01),
                                                                pytest.approx(1.5, abs=0.01))
    assert b"/Type3" not in built and (b"NimbusRomNo9L" in built or b"Times" in built)
    assert not re.search(rb"/[A-Z]{6}\+CM(MI|R|SY)5\b", built)
    sizes = {float(s) for body in content_streams(built) for s in re.findall(rb"([\d.]+)\s+Tf\b", body)}
    assert sizes and min(sizes) >= MIN_BP - 1e-3, sorted(sizes)
    # the same assets, and the same panel labels, as the 1.8 in figure
    main_src, alt_src = open(SOURCE).read(), open(source).read()
    assert set(re.findall(r"assets/(fig1_[^}]+)", alt_src)) == set(re.findall(r"assets/(fig1_[^}]+)", main_src))
    assert re.findall(r"\\panel\{(\w)\}\{([^}]+)\}", alt_src) == re.findall(r"\\panel\{(\w)\}\{([^}]+)\}", main_src)
    committed = open(os.path.join(FIGURES, "fig1_method_150.pdf"), "rb").read()
    if producer(built) != producer(committed):
        pytest.skip(f"committed PDF from {producer(committed)!r}, this build from {producer(built)!r}")
    assert built == committed, "paper/figures/fig1_method_150.pdf is stale: rebuild it from fig1_method_150.tex"
