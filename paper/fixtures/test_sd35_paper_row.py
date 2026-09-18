"""The SD 3.5 row in the paper registry, and the second VAE-ceiling line it needs.

Every other row in the table was trained in the SD KL-f8 latent space, so one "VAE ceiling"
reference line covers them all and `paperdata.reference` may take it from whichever run reports it
first. The SD 3.5 row has a different autoencoder and therefore a different reconstruction ceiling,
measured by `vae_gate_score.py` on the same windows `eval_tf.py` scores. These checks pin that it
gets its own line, read from the gate, and that it is kept out of the shared reference so the
existing rows' ceiling does not move.

    python -m pytest paper/fixtures/test_sd35_paper_row.py -q
"""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
REPO = os.path.dirname(PAPER)
sys.path.insert(0, HERE)
sys.path.insert(0, PAPER)

import make_fixtures  # noqa: E402
import paperdata  # noqa: E402

RUN = "035-sd35-l32-aligned"
GATE = "vae_gate_sd35"
SYNC = os.path.join(PAPER, "sync_results.sh")


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("results_sd35"))
    make_fixtures.build(root, seed=0)
    return root


@pytest.fixture(scope="module")
def tables(fixture_root, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("tables_sd35"))
    proc = subprocess.run([sys.executable, os.path.join(PAPER, "make_tables.py"),
                           "--results-root", fixture_root, "--out", out],
                          capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, f"make_tables failed:\n{proc.stdout}\n{proc.stderr}"
    return out


def registry():
    return paperdata.load_registry()


def spec():
    return next(s for s in registry()["rows"] if s["run"] == RUN)


# ---- the registry entry ------------------------------------------------------------------------

def test_the_row_is_registered_as_a_placeholder_in_the_main_table():
    s = spec()
    assert s["placeholder"] is True, "the run directory does not exist yet"
    assert "main" in s["groups"]
    assert s["exposure"] and s["exposure_source"]


def test_the_row_declares_its_own_autoencoder_and_where_the_ceiling_was_measured():
    s = spec()
    assert s["own_vae"]["gate"] == GATE
    assert s["own_vae"]["decoder"], "which decoder entry of the gate's metrics.json to read"
    assert s["own_vae"]["label"]


def test_the_other_rows_still_share_one_ceiling():
    others = [s for s in registry()["rows"] if s["run"] != RUN]
    assert not any(s.get("own_vae") for s in others), "this change is additive to the existing rows"


def test_sync_pulls_the_row_and_its_gate():
    body = open(SYNC).read()
    assert RUN in body and GATE in body


# ---- the ceiling accessor ----------------------------------------------------------------------

def test_the_ceiling_is_read_from_the_gate_not_from_the_row(fixture_root):
    reg = registry()
    row = next(r for r in paperdata.load_rows(fixture_root, reg, "main") if r.run == RUN)
    assert not row.exists, "the fixture deliberately has no 035 run directory"
    gate = json.load(open(os.path.join(fixture_root, GATE, "metrics.json")))
    decoder = row.own_vae["decoder"]
    for corpus in ("seen", "unseen"):
        for key in ("psnr", "lpips"):
            assert row.own_vae_ceiling(corpus, key) == gate["metrics"][corpus][decoder][key]["mean"]


def test_a_row_without_its_own_autoencoder_has_no_gate_ceiling(fixture_root):
    reg = registry()
    row = next(r for r in paperdata.load_rows(fixture_root, reg, "main") if r.run == "030-dit-l32-aligned")
    assert row.own_vae is None
    assert row.own_vae_ceiling("seen", "psnr") is None


def test_the_sixteen_channel_row_is_left_out_of_the_shared_reference(fixture_root):
    """The shared ceiling must stay the SD 1.x number the finished rows were measured against."""
    reg = registry()
    rows = paperdata.load_rows(fixture_root, reg, "main")
    dit = next(r for r in rows if r.run == "030-dit-l32-aligned")
    assert paperdata.reference(rows, "seen", "vae_psnr") == dit.tf_mean("seen", "vae_psnr")


# ---- the table ---------------------------------------------------------------------------------

def test_the_main_table_carries_a_second_ceiling_line_with_the_gate_numbers(tables, fixture_root):
    body = open(os.path.join(tables, "main.tex")).read()
    gate = json.load(open(os.path.join(fixture_root, GATE, "metrics.json")))
    decoder = spec()["own_vae"]["decoder"]
    assert spec()["own_vae"]["label"] in body
    assert f"{gate['metrics']['seen'][decoder]['psnr']['mean']:.2f}" in body
    assert f"{gate['metrics']['unseen'][decoder]['lpips']['mean']:.3f}" in body


def test_the_row_label_is_marked_so_its_ceiling_is_not_read_off_the_shared_line(tables):
    body = open(os.path.join(tables, "main.tex")).read()
    note = spec()["vae_note"]
    assert "$^{" + note + "}$" in body, "the model row needs the marker"
    assert note in body.split("footnotesize", 1)[1], "and the footnote has to explain it"


def test_the_other_tables_are_unchanged_by_the_new_row(tables):
    for name in ("seeds.tex", "cm.tex", "grid.tex"):
        body = open(os.path.join(tables, name)).read()
        assert spec()["own_vae"]["label"] not in body, name
