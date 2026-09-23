"""Statements about GameNGen that the 2026-09-22 review (H5) checked against the paper's v2 text.

Each test pins one corrected statement so a later edit cannot quietly reintroduce the claim:

  * the decoder is tuned with MSE + 0.1 LPIPS, which is OUR deviation: GameNGen tunes it with MSE
    alone (v2 section 3.2.2);
  * GameNGen trains on a random 70M-example subset (v2 section 4.2), not on the whole generated pool;
  * GameNGen's step table shows LPIPS improving from 1 to 4 steps (0.255 to 0.198), the same
    direction as ours, and only the step count of our 4-step column matches its sampler;
  * "no unexplained modelling gap" is not a claim the floors can support;
  * the rows are a system comparison under one recipe, not a comparison in which the backbone is
    the only variable.

    python -m pytest paper/fixtures/test_paper_claims.py -q
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def read(*parts):
    with open(os.path.join(REPO, *parts)) as f:
        return f.read()


def test_the_paper_does_not_credit_the_lpips_term_to_gamengen():
    tex = read("paper", "main.tex")
    assert not re.search(r"LPIPS,?\s*as in GameNGen", tex), "MSE + LPIPS is our deviation"
    assert "GameNGen tunes its decoder with MSE alone" in tex


def test_the_paper_states_the_gamengen_training_set_correctly():
    tex = read("paper", "main.tex")
    assert "over 900M frames" not in tex
    assert "70M" in tex and "900M" in tex and "v2" in tex


def test_the_dossier_quotes_gamengen_table_3():
    doc = read("docs", "RESEARCHER_DOSSIER.md")
    assert "the opposite LPIPS pattern" not in doc
    assert "0.255" in doc and "0.198" in doc
    assert "like-for-like sampler setting for that comparison" not in doc
    assert "Only the step count matches" in doc


def test_no_modelling_gap_is_not_claimed():
    doc = read("docs", "RESEARCHER_DOSSIER.md")
    assert "floor-subtracted PSNR does not control" in doc
    rc = read("RESEARCH_CONTEXT.md")
    line = [ln for ln in rc.splitlines() if "no unexplained modelling gap" in ln]
    assert all("[Withdrawn 2026-09-22" in ln for ln in line), "the log still asserts the claim unqualified"


def test_the_framing_is_a_system_comparison():
    for name in ("README.md", "REQUIREMENTS.md", "TECHNICAL_PLAN.md"):
        text = read(name)
        assert "the backbone is the only variable" not in text, name
        assert "system comparison" in text, name


def test_the_launcher_labels_what_is_gamengens_and_what_is_ours():
    text = read("scripts", "spiderman", "launch_nexttic.sh")
    assert "is GameNGen's own conditioning" not in text
    assert "GameNGen's spacing" not in text or "inference" in text
