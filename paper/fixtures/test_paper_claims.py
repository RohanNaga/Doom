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


def rendered_sentences(tex):
    """The sentences of a TeX source with its comments removed (a rough split, enough for these checks)."""
    body = "\n".join(re.split(r"(?<!\\)%", ln, maxsplit=1)[0] for ln in tex.splitlines())
    return [s for s in re.split(r"(?<=[.;])\s+", " ".join(body.split())) if s]


def test_the_paper_does_not_credit_the_lpips_term_to_gamengen():
    """Wherever the paper describes GameNGen's decoder tune, it says MSE and never LPIPS (v2 section 3.2.2).

    The Sep 27 joint review reworded the sentence ("Following GameNGen, we ... tune it with MSE"), so the
    check is on what any such sentence says, not on one fixed wording.
    """
    tex = read("paper", "main.tex")
    assert not re.search(r"LPIPS,?\s*as in GameNGen", tex), "MSE + LPIPS is our deviation"
    about_the_tune = [s for s in rendered_sentences(tex)
                      if "GameNGen" in s and re.search(r"\btun(e|es|ed|ing)\b", s)]
    for s in about_the_tune:
        assert "MSE" in s and "LPIPS" not in s, s


def test_the_paper_states_the_gamengen_training_set_correctly():
    """If the paper states GameNGen's training-set size, it names the version: 70M examples is v2, 900M frames v1.

    The Sep 27 joint review cut the training-budget sentence from the related work (cut c9), so the size may be
    absent; it may never read "over 900M frames".
    """
    tex = read("paper", "main.tex")
    assert "over 900M frames" not in tex
    rendered = " ".join(rendered_sentences(tex))
    if "70M" in rendered or "900M" in rendered:
        assert "70M" in rendered and "v2" in rendered, "the 70M subset is arXiv v2's"
        if "900M" in rendered:
            assert "v1" in rendered, "900M generated frames is arXiv v1's statement"


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


def test_the_trainer_help_does_not_claim_gamengens_spacing_or_conditioning():
    """Text only: the help strings of --tic-stride and --action-history (review H5)."""
    src = read("train_wm.py")
    assert "the per-tic dataset, GameNGen's spacing" not in src
    assert "GameNGen's action conditioning: one token" not in src
    assert "GameNGen context-noise augmentation" not in src
    import train_wm
    helps = {a.dest: a.help or "" for a in train_wm.build_parser()._actions}
    assert "inference" in helps["tic_stride"] and "not stated" in helps["tic_stride"]
    assert "ours" in helps["action_history"]


def test_the_design_note_names_the_current_unseen_subset():
    note = read(".claude", "analyses", "nexttic-design-2026-09-20.md")
    assert "| unseen (`arenas_678`) | 0:60 | 20 |" not in note
    assert "60:120" in note


def test_the_launcher_labels_what_is_gamengens_and_what_is_ours():
    text = read("scripts", "spiderman", "launch_nexttic.sh")
    assert "is GameNGen's own conditioning" not in text
    assert "GameNGen's spacing" not in text or "inference" in text
