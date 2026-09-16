"""
End-to-end test of the paper builders on fabricated result directories.

Builds a fixture results root with the real schemas (`make_fixtures.py`), runs
`make_figures.py` and `make_tables.py` against it, and checks that every figure and table
exists, that the numbers in the tables are the fixture's numbers (read back from the fixture
files, never typed here), that missing artifacts show as "n/a" with a warning instead of a
crash, and that an empty results root still produces complete output. When pdflatex is on
the path, the tables are compiled in a standalone document.

    python -m pytest paper/fixtures/test_paper_build.py -q
"""
import json
import os
import shutil
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

FIGURES = ["drift_psnr.pdf", "drift_lpips.pdf", "drift.pdf", "tf_seen_unseen.pdf", "blur_control.pdf"]
TABLES = ["main.tex", "transfer.tex", "ema.tex", "seeds.tex", "cm.tex", "grid.tex", "horizons.tex", "blur.tex",
          "paired.tex", "idm.tex", "late_drop.tex"]


def run_builder(script, root, out, extra=()):
    cmd = [sys.executable, os.path.join(PAPER, script), "--results-root", root, "--out", out, *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, f"{script} failed:\nSTDOUT\n{proc.stdout}\nSTDERR\n{proc.stderr}"
    return proc


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("results"))
    make_fixtures.build(root, seed=0)
    return root


@pytest.fixture(scope="module")
def built(fixture_root, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("paper_out"))
    figs = run_builder("make_figures.py", fixture_root, os.path.join(out, "figures"), ["--png"])
    tabs = run_builder("make_tables.py", fixture_root, os.path.join(out, "tables"))
    return out, figs, tabs


def read(path):
    return open(path).read()


def test_all_outputs_exist(built):
    out, _, _ = built
    for f in FIGURES:
        p = os.path.join(out, "figures", f)
        assert os.path.exists(p) and os.path.getsize(p) > 1000, f
    for t in TABLES:
        p = os.path.join(out, "tables", t)
        assert os.path.exists(p) and os.path.getsize(p) > 100, t


def test_main_table_numbers_come_from_files(built, fixture_root):
    out, _, _ = built
    main = read(os.path.join(out, "tables", "main.tex"))
    for run in ("030-dit-l32-aligned", "031-unet-l32-aligned", "033-pixart-l32-aligned"):
        seen = json.load(open(os.path.join(fixture_root, run, "eval_tf_seen", "metrics.json")))
        unseen = json.load(open(os.path.join(fixture_root, run, "eval_tf_unseen", "metrics.json")))
        drift = json.load(open(os.path.join(fixture_root, run, "rollout_metrics_seen", "drift.json")))
        fvd16 = json.load(open(os.path.join(fixture_root, run, "rollout_metrics_seen", "fvd16.json")))["fvd"]
        cfg = json.load(open(os.path.join(fixture_root, run, "config.json")))
        for text in (f"{seen['psnr_raw']['mean']:.2f}", f"{seen['lpips_raw']['mean']:.3f}",
                     f"{unseen['psnr_raw']['mean']:.2f}", f"{drift['psnr@64']:.2f}", f"{drift['lpips@64']:.3f}",
                     f"{fvd16:.0f}", f"{drift['idm_top1_mean']:.3f}", f"{cfg['params'] / 1e6:.0f}M"):
            assert text in main, f"{run}: {text} not in main.tex"
    # reference rows
    seen = json.load(open(os.path.join(fixture_root, "030-dit-l32-aligned", "eval_tf_seen", "metrics.json")))
    assert f"{seen['copy_psnr_raw']['mean']:.2f}" in main
    assert f"{seen['vae_psnr']['mean']:.2f}" in main and f"{seen['vae_lpips']['mean']:.3f}" in main
    audit = json.load(open(os.path.join(fixture_root, "030-dit-l32-aligned", "audit", "audit.json")))
    assert f"{audit['blur_sweep']['64']['0.0']['copy_seed_lpips']:.3f}" in main
    # the compute-matched table carries the 2.5k rows, not the main table
    assert "2.5k" not in main
    assert "2.5k" in read(os.path.join(out, "tables", "cm.tex"))


def test_missing_artifacts_render_as_na_with_warnings(built):
    out, figs, tabs = built
    main = read(os.path.join(out, "tables", "main.tex"))
    assert "UniDiffuser" in main and "n/a" in main
    unidiffuser_line = [ln for ln in main.splitlines() if "UniDiffuser" in ln][0]
    assert unidiffuser_line.count("n/a") >= 8
    skyreels_line = [ln for ln in main.splitlines() if "SkyReels" in ln][0]
    assert "n/a" in skyreels_line                       # no FVD for the video row
    assert "\\S" in skyreels_line                         # its IDM footnote marker survives
    cm = read(os.path.join(out, "tables", "cm.tex"))
    pixart_cm = [ln for ln in cm.splitlines() if "PixArt" in ln][0]
    assert "n/a" in pixart_cm                             # no rollout for 062
    for proc in (figs, tabs):
        assert "040-unidiffuser-l32-aligned" in proc.stderr, "missing run must be warned about"
    assert "no audit" in tabs.stderr or "audit" in tabs.stderr


def test_ci_columns_use_audit_bootstrap(built, fixture_root):
    out, _, _ = built
    audit = json.load(open(os.path.join(fixture_root, "031-unet-l32-aligned", "audit", "audit.json")))
    lo, hi = audit["bootstrap"]["tf_seen"]["psnr_raw"]["ci95"]
    horizons = read(os.path.join(out, "tables", "horizons.tex"))
    tf_ci = f"[{lo:.2f}, {hi:.2f}]"
    transfer = read(os.path.join(out, "tables", "transfer.tex"))
    assert tf_ci in transfer, "transfer table must carry the audit CI for the seen mean"
    lo64, hi64 = audit["bootstrap"]["rollout"]["psnr_h64_s0.0"]["ci95"]
    assert f"[{lo64:.2f}, {hi64:.2f}]" in horizons
    # the audit gives no CI for horizon 16: the cell keeps the value with no interval and no crash
    assert "n/a" not in [ln for ln in horizons.splitlines() if "U-Net" in ln][0]


def test_transfer_drop_is_seen_minus_unseen(built, fixture_root):
    out, _, _ = built
    transfer = read(os.path.join(out, "tables", "transfer.tex"))
    seen = json.load(open(os.path.join(fixture_root, "031-unet-l32-aligned", "eval_tf_seen", "metrics.json")))["psnr_raw"]["mean"]
    unseen = json.load(open(os.path.join(fixture_root, "031-unet-l32-aligned", "eval_tf_unseen", "metrics.json")))["psnr_raw"]["mean"]
    assert f"{seen - unseen:.2f}" in transfer
    unet_line = [ln for ln in transfer.splitlines() if "U-Net" in ln][0]
    assert "n/a" in unet_line                             # the 15-map corpus is a placeholder


def test_empty_results_root_does_not_crash(tmp_path):
    root = str(tmp_path / "empty")
    os.makedirs(root)
    out = str(tmp_path / "out")
    figs = run_builder("make_figures.py", root, os.path.join(out, "figures"))
    tabs = run_builder("make_tables.py", root, os.path.join(out, "tables"))
    for t in TABLES:
        text = read(os.path.join(out, "tables", t))
        assert "n/a" in text, t
    for f in FIGURES:
        assert os.path.exists(os.path.join(out, "figures", f)), f
    assert "warning" in figs.stderr and "warning" in tabs.stderr


def test_loader_bootstrap_matches_rollout_audit_convention():
    # episode resampling with the same RandomState sequence as rollout_audit.episode_resamples
    episodes = [1, 1, 2, 2, 3, 3]
    draws = paperdata.episode_resamples(episodes, n_boot=3, seed=0)
    import numpy as np
    rng = np.random.RandomState(0)
    uniq = np.unique(episodes)
    for d in draws:
        picked = uniq[rng.randint(len(uniq), size=len(uniq))]
        expect = np.concatenate([np.flatnonzero(np.asarray(episodes) == e) for e in picked])
        assert np.array_equal(d, expect)


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_tables_compile(built, tmp_path):
    out, _, _ = built
    doc = tmp_path / "preview.tex"
    body = "\n".join(f"\\section*{{\\detokenize{{{t}}}}}\\input{{{os.path.join(out, 'tables', t)}}}" for t in TABLES)
    doc.write_text("\\documentclass{article}\\usepackage{booktabs,multirow,amsmath,graphicx}\\begin{document}"
                   + body + "\\end{document}")
    proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", str(doc)],
                          capture_output=True, text=True, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert (tmp_path / "preview.pdf").exists()
