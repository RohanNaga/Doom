"""
The CoRL 2026 PhysWM draft (`paper/main.tex`, `paper/appendix.tex`, `paper/refs.bib`, `paper/FIGURES.md`).

What is pinned, because each is a way the draft can go wrong without anyone noticing:

  * the template is used as shipped: `\\documentclass{article}` with `\\usepackage{corl_2026}` and no option
    (the submission build), no page-geometry package of our own, and `corl_2026.sty` / `corlabbrvnat.bst`
    byte-identical to the Overleaf download committed in f0500f9;
  * every citation resolves to a `refs.bib` entry, and every entry names the arXiv id or URL it was
    verified against;
  * every placeholder (`\\tbd`, `\\todo{}`) and every provisional number (`\\prov{}`) sits in a paragraph
    whose comment names the run it depends on (`% depends:`);
  * the abstract is four to six sentences, as the template asks;
  * every figure and table label appears in `FIGURES.md`;
  * the appendix comes after the bibliography;
  * and, when pdflatex is installed, the documented build (pdflatex, bibtex, pdflatex, pdflatex) runs clean,
    leaves no undefined citation or reference, keeps the body within four pages, and embeds a figure whose
    file exists while boxing one whose file does not.

    python -m pytest paper/fixtures/test_paper_skeleton.py -q
"""
import hashlib
import os
import re
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
SOURCES = ("main.tex", "appendix.tex")

# sha256 of the template files as committed in f0500f9 (Rohan's Overleaf download)
SHIPPED = {
    "corl_2026.sty": "62f38cd8df7ad718796617595c8230da136fc83ea1d7e1ba80084113409ec957",
    "corlabbrvnat.bst": "eb5c200b6c5a81cd774155c79c28d3a32ac8a5e25bb246a40dda74deb94dfa71",
}
MARKS = (r"\tbd", r"\todo{", r"\prov{")
DEFINITION = re.compile(r"^\s*\\(newcommand|renewcommand|providecommand|def)\b")


def read(name):
    with open(os.path.join(PAPER, name)) as f:
        return f.read()


def strip_comment(line):
    """The line up to its first unescaped %."""
    return re.split(r"(?<!\\)%", line, maxsplit=1)[0]


def cited_keys(tex):
    keys = set()
    for m in re.finditer(r"\\cite[pt]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex):
        keys.update(k.strip() for k in m.group(1).split(",") if k.strip())
    return keys


def bib_entries(bib):
    """{key: body} for every @type{key, ...} entry."""
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", bib, re.S):
        entries[m.group(2)] = m.group(3)
    return entries


def paragraphs(tex):
    """Blocks of consecutive non-blank lines."""
    return [p for p in re.split(r"\n\s*\n", tex) if p.strip()]


def test_the_template_is_used_as_shipped():
    for name, digest in SHIPPED.items():
        with open(os.path.join(PAPER, name), "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == digest, f"{name} differs from the Overleaf download"
    tex = read("main.tex")
    body = "\n".join(strip_comment(ln) for ln in tex.splitlines())
    assert re.search(r"\\documentclass\{article\}", body)
    assert re.search(r"\\usepackage\{corl_2026\}", body), "the submission build loads corl_2026 with no option"
    for forbidden in ("fullpage", "geometry", "times", "natbib"):
        assert not re.search(r"\\usepackage(\[[^\]]*\])?\{[^}]*\b%s\b" % forbidden, body), forbidden
    assert r"\keywords{" in body


def test_every_citation_resolves_to_a_verified_bib_entry():
    entries = bib_entries(read("refs.bib"))
    cited = set()
    for name in SOURCES:
        cited |= cited_keys(read(name))
    assert cited, "the draft cites nothing"
    missing = sorted(cited - set(entries))
    assert not missing, f"cited but not in refs.bib: {missing}"
    for key in cited:
        body = entries[key]
        assert re.search(r"(eprint|url|howpublished)\s*=", body), f"{key} names no arXiv id or URL"


def test_every_placeholder_names_what_it_depends_on():
    for name in SOURCES:
        for block in paragraphs(read(name)):
            lines = [ln for ln in block.splitlines() if not DEFINITION.match(ln)]
            marked = any(mark in strip_comment(ln) for ln in lines for mark in MARKS)
            if marked:
                assert "depends:" in block, f"{name}: placeholder without a '% depends:' comment:\n{block[:400]}"


def test_the_abstract_is_four_to_six_sentences():
    tex = read("main.tex")
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    assert m, "no abstract"
    text = " ".join(strip_comment(ln) for ln in m.group(1).splitlines())
    text = re.sub(r"\\[a-zA-Z]+\*?(\{[^}]*\})?", " ", text)          # macros are not sentence ends
    text = re.sub(r"\d\.\d", "0", text)                              # decimals are not sentence ends
    sentences = [s for s in re.split(r"[.!?](?:\s|$)", text) if len(s.split()) > 3]
    assert 4 <= len(sentences) <= 6, f"{len(sentences)} sentences"


def test_every_figure_and_table_is_listed_in_figures_md():
    listed = read("FIGURES.md")
    for name in SOURCES:
        for label in re.findall(r"\\label\{((?:fig|tab):[^}]*)\}", read(name)):
            assert f"`{label}`" in listed, f"{label} ({name}) is not in FIGURES.md"


def test_the_appendix_comes_after_the_bibliography():
    tex = "\n".join(strip_comment(ln) for ln in read("main.tex").splitlines())
    bib = tex.find(r"\bibliography{refs}")
    app = tex.find(r"\input{appendix}")
    assert bib != -1 and app != -1 and bib < app


@pytest.mark.skipif(shutil.which("pdflatex") is None or shutil.which("bibtex") is None,
                    reason="pdflatex or bibtex not installed")
def test_the_documented_build_is_clean_and_the_body_fits_four_pages(tmp_path):
    work = tmp_path / "paper"
    shutil.copytree(PAPER, work, ignore=shutil.ignore_patterns("fixtures", "*.pdf", "*.aux", "*.log", "*.bbl",
                                                               "*.blg", "*.out", "__pycache__"))
    # the build copies figures/, so the file that exists today embeds; the pending ones must box, not fail
    for step in (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main"], ["bibtex", "main"],
                 ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main"],
                 ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main"]):
        proc = subprocess.run(step, cwd=work, capture_output=True, text=True)
        assert proc.returncode == 0, f"{' '.join(step)} failed:\n{proc.stdout[-3000:]}"
    log = (work / "main.log").read_text(errors="replace")
    assert "undefined" not in log.lower().replace("undefined control", ""), "an undefined citation or reference"
    assert "There were undefined references" not in log
    blg = (work / "main.blg").read_text(errors="replace")
    assert "Warning--I didn't find a database entry" not in blg
    aux = (work / "main.aux").read_text(errors="replace")
    m = re.search(r"\\newlabel\{body-end\}\{\{[^}]*\}\{(\d+)\}", aux)
    assert m, "the body-end label is missing"
    assert int(m.group(1)) <= 4, f"the body ends on page {m.group(1)}"
    assert "sd35_70k_live_vs_ema_rollout_strip.jpg" in log, "an existing figure file was not embedded"
    assert (work / "main.pdf").stat().st_size > 10000
