"""CPU gates for the adaptation scoring driver (`score_adapt.py`).

* **Censoring.** A curve that never reaches a target within the steps scored is right-censored (a flag and
  the last step scored), never "reached at the last step".
* **One row per (map, step, seed, weights)**, with the decoder-free latent ratio as a column beside PSNR
  and LPIPS, the gains signed so that positive is better, the guards at the guard steps only, and the
  step-0 row equal to the frozen model scored on the same held-out windows. A rerun adds nothing.

    python -m pytest paper/fixtures/test_score_adapt.py -q
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

os.environ.setdefault("ACCELERATE_USE_CPU", "1")

import adapt_fixtures as F  # noqa: E402
import adapt_split  # noqa: E402
import adapt_wm  # noqa: E402
import backbones  # noqa: E402
import eval_tf  # noqa: E402
import score_adapt  # noqa: E402


# ---------------------------------------------------------------------------------------
# cost and censoring
# ---------------------------------------------------------------------------------------

def curve(*pts, metric="heldout_psnr_gain"):
    return [{"step": s, metric: v} for s, v in pts]


def test_the_cost_is_the_first_step_that_reaches_the_target():
    c = score_adapt.adaptation_cost(curve((0, -0.5), (250, 0.1), (500, 0.6), (1000, 0.4)), "heldout_psnr_gain", 0.5)
    assert c["reached"] and not c["censored"] and c["step"] == 500 and c["value"] == 0.6
    assert c["last_step_scored"] == 1000


def test_a_curve_that_never_reaches_the_target_is_right_censored_not_reached_at_the_end():
    c = score_adapt.adaptation_cost(curve((0, -0.5), (1000, 0.2), (2000, 0.45)), "heldout_psnr_gain", 0.5)
    assert c["censored"] and not c["reached"] and c["step"] is None
    assert c["last_step_scored"] == 2000 and c["last_value"] == 0.45


def test_the_zero_shot_point_can_already_reach_the_target_and_lower_can_be_better():
    assert score_adapt.adaptation_cost(curve((0, 0.7), (250, 0.9)), "heldout_psnr_gain", 0.5)["step"] == 0
    rows = curve((0, 1.2), (250, 0.95), (500, 0.8), metric="heldout_latent_mse_ratio")
    c = score_adapt.adaptation_cost(rows, "heldout_latent_mse_ratio", 0.9, higher_is_better=False)
    assert c["step"] == 500


def test_unscored_points_do_not_count():
    rows = curve((0, -0.5), (250, None), (500, 0.1))
    c = score_adapt.adaptation_cost(rows, "heldout_psnr_gain", 0.0)
    assert c["steps_scored"] == [0, 500] and c["step"] == 500


@pytest.mark.parametrize("spec,want", [("all", {0, 250, 2000}), ("none", set()), ("0,last", {0, 2000}),
                                       ("first,250", {0, 250}), ("250,7", {250})])
def test_the_guard_steps_parse(spec, want):
    assert score_adapt.parse_steps(spec, [0, 250, 2000]) == want


def test_the_gains_are_signed_so_that_positive_is_better():
    m = {"psnr_raw": {"mean": 22.5}, "persist_psnr_raw": {"mean": 21.5}, "lpips_raw": {"mean": 0.18},
         "persist_lpips_raw": {"mean": 0.20}, "latent_mse": {"mean": 0.3}, "copy_latent_mse": {"mean": 0.4},
         "latent_mse_ratio": {"mean": 0.8}, "psnr_dec": {"mean": 20.0, "n": 256}}
    row = score_adapt.eval_columns("heldout", m)
    assert row["heldout_psnr_gain"] == pytest.approx(1.0) and row["heldout_lpips_gain"] == pytest.approx(0.02)
    assert row["heldout_latent_mse_ratio"] == 0.8 and row["heldout_latent_ratio_of_means"] == pytest.approx(0.75)
    assert row["heldout_windows"] == 256
    assert score_adapt.eval_columns("heldout", {"psnr_dec": {"mean": 1, "n": 3}})["heldout_psnr_gain"] is None


def write_per_window(path, rows):
    import csv
    keys = ["index", "episode", "map", "start", "latent_mse", "copy_latent_mse", "latent_mse_ratio", "psnr_raw",
            "persist_psnr_raw", "lpips_raw", "persist_lpips_raw"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"index": i, "map": 17, "start": 10 * i, **r})


def test_the_latent_skill_is_a_geometric_mean_that_static_windows_cannot_dominate(tmp_path):
    import math
    # three ordinary windows (R = 0.5, 0.5, 2.0), one near-static window (R = 1000), one exact copy (undefined)
    rows = [{"episode": 1, "latent_mse": 0.5, "copy_latent_mse": 1.0, "psnr_raw": 22.0, "persist_psnr_raw": 21.0,
             "lpips_raw": 0.10, "persist_lpips_raw": 0.20},
            {"episode": 1, "latent_mse": 0.25, "copy_latent_mse": 0.5, "psnr_raw": 20.0, "persist_psnr_raw": 20.5,
             "lpips_raw": 0.30, "persist_lpips_raw": 0.25},
            {"episode": 3, "latent_mse": 2.0, "copy_latent_mse": 1.0, "psnr_raw": 19.0, "persist_psnr_raw": 19.0,
             "lpips_raw": 0.20, "persist_lpips_raw": 0.20},
            {"episode": 3, "latent_mse": 0.01, "copy_latent_mse": 0.00001, "psnr_raw": 30.0, "persist_psnr_raw": 45.0,
             "lpips_raw": 0.05, "persist_lpips_raw": 0.01},
            {"episode": 5, "latent_mse": 0.01, "copy_latent_mse": 0.0, "psnr_raw": 25.0, "persist_psnr_raw": 99.0,
             "lpips_raw": 0.05, "persist_lpips_raw": 0.0}]
    src, out = tmp_path / "per_window.csv", tmp_path / "paired_windows.csv"
    write_per_window(src, rows)
    c = score_adapt.window_columns("heldout", str(src), str(out))
    ratios = [0.5, 0.5, 2.0, 1000.0]
    assert c["heldout_latent_skill"] == pytest.approx(-10 * sum(math.log10(r) for r in ratios) / 4)
    assert c["heldout_latent_ratio_mean"] == pytest.approx(sum(ratios) / 4)      # dominated by the static window
    assert c["heldout_latent_skill_windows"] == 4 and c["heldout_latent_ratio_undefined"] == 1
    assert c["heldout_psnr_gain_paired"] == pytest.approx((1.0 - 0.5 + 0.0 - 15.0 - 74.0) / 5)
    assert c["heldout_lpips_gain_paired"] == pytest.approx((0.10 - 0.05 + 0.0 - 0.04 - 0.05) / 5)
    assert c["heldout_episodes"] == 3 and c["heldout_paired_windows"] == 5
    import csv
    kept = list(csv.DictReader(open(out)))
    assert [int(k["episode"]) for k in kept] == [1, 1, 3, 3, 5] and kept[4]["latent_skill"] == "nan"
    assert float(kept[0]["latent_skill"]) == pytest.approx(10 * math.log10(2))
    assert c["heldout_per_window"] == str(src) and c["heldout_paired"] == str(out)


def test_without_raw_frames_the_paired_gains_are_absent_not_zero(tmp_path):
    src = tmp_path / "per_window.csv"
    write_per_window(src, [{"episode": 1, "latent_mse": 0.5, "copy_latent_mse": 1.0}])
    c = score_adapt.window_columns("heldout", str(src), str(tmp_path / "p.csv"))
    assert c["heldout_psnr_gain_paired"] is None and c["heldout_paired_windows"] == 0
    assert c["heldout_latent_skill"] == pytest.approx(10 * 0.30103, rel=1e-4)


# ---------------------------------------------------------------------------------------
# a run, scored end to end
# ---------------------------------------------------------------------------------------

@pytest.fixture
def tiny(monkeypatch):
    F.serve_tiny_pixart(monkeypatch)
    F.stub_lpips(monkeypatch)


def test_every_checkpoint_is_scored_per_weights_and_step_zero_is_the_frozen_score(tiny, tmp_path, capsys):
    d = F.map_corpus(tmp_path / "lat")
    src = F.write_source_snapshot(tmp_path / "040-unet-nexttic" / "snap_0200000.pt")
    split = str(tmp_path / "split_adapt_unseen_map17_seed0.json")
    adapt_split.main(["--episodes", ",".join(map(str, F.EPISODES)), "--set", "unseen", "--map", str(F.MAP),
                      "--latents-dir", d, "--windows", "12", "--context-frames", str(F.CTX), "--out", split])
    run = tmp_path / "unet_unseen_map17_s0"
    assert adapt_wm.main(adapt_wm.build_parser().parse_args(
        ["--source", src, "--adapt-split", split, "--latents-dir", d, "--results-dir", str(run),
         "--per-gpu-batch", "4", "--global-batch", "4", "--step-grid", "0,2", "--lora-rank", "4", "--warmup", "1",
         "--lr", "1e-2", "--num-workers", "0", "--parity-windows", "4", "--no-wandb"])) == 0
    vae = F.tiny_vae(tmp_path / "vae")
    val = F.map_corpus(tmp_path / "val", episodes=(6000, 6001), map_id=2)
    val_split = tmp_path / "split_val.json"
    val_split.write_text(json.dumps({"val": [6000, 6001]}))
    argv = ["score", "--run-dir", str(run), "--vae-path", vae, "--directional-windows", "2", "--device", "cpu",
            "--trainmap-latents-dir", val, "--trainmap-split", str(val_split), "--trainmap-windows", "6",
            "--guard-steps", "last", "--steps", "2", "--batch-size", "4", "--no-wandb"]
    assert score_adapt.main(argv) == 0
    rows = score_adapt.read_rows(str(run / "scores.jsonl"))
    assert sorted((r["step"], r["weights"]) for r in rows) == [(0, "ema"), (0, "live"), (2, "ema"), (2, "live")]
    by = {(r["step"], r["weights"]): r for r in rows}
    r0 = by[(0, "live")]
    assert r0["map"] == "unseen_map17" and r0["seed"] == 0 and r0["heldout_windows"] == 12
    assert r0["heldout_latent_mse_ratio"] is not None and r0["heldout_psnr"] is None      # no raw frames here
    # the per-window rows are kept, and the latent skill is -10 x the mean log10 ratio over them
    import csv
    import math
    for (step, w), r in by.items():
        assert r["weights"] == w
        per = list(csv.DictReader(open(r["heldout_per_window"])))
        paired = list(csv.DictReader(open(r["heldout_paired"])))
        assert len(per) == len(paired) == 12
        ratios = [float(p["latent_mse"]) / float(p["copy_latent_mse"]) for p in per]
        assert r["heldout_latent_skill"] == pytest.approx(-10 * sum(math.log10(x) for x in ratios) / 12)
        assert r["heldout_latent_ratio_mean"] == pytest.approx(r["heldout_latent_mse_ratio"], rel=1e-5)
    # at step 0 the live adapter and its EMA are the same tensors: scored once, copied, and said so
    assert by[(0, "ema")]["same_tensors_as"] == "live"
    assert by[(0, "ema")]["heldout_psnr_dec"] == r0["heldout_psnr_dec"]
    # guards only where asked
    assert r0["directional_correct_frac"] is None and r0["trainmap_psnr_dec"] is None
    for w in ("live", "ema"):
        assert by[(2, w)]["directional_windows"] == 4 and by[(2, w)]["trainmap_windows"] == 6
    assert "same_tensors_as" not in by[(2, "ema")]
    # the step-0 row IS the frozen EMA model on the same held-out windows
    out = str(tmp_path / "frozen")
    eval_tf.main(eval_tf.build_parser().parse_args(
        ["--ckpt", src, "--backbone", "pixart", "--pixart-path", backbones.PIXART_DEFAULT, "--latent-channels", "4",
         "--vae-path", vae, "--latents-dir", d, "--split", split, "--subset", "val", "--windows-file", split,
         "--use-ema", "--context-frames", str(F.CTX), "--num-actions", "3", "--noise-buckets", "4",
         "--batch-size", "4", "--steps", "2", "--save-images", "0", "--num-workers", "0", "--out-dir", out]))
    frozen = json.load(open(os.path.join(out, "metrics.json")))
    for k in ("psnr_dec", "lpips_dec", "latent_mse_ratio"):
        assert r0[f"heldout_{k}"] == frozen[k]["mean"], k
    # a rerun scores nothing new; a target nobody reaches is censored at the last step scored
    assert score_adapt.main(argv) == 0
    assert len(score_adapt.read_rows(str(run / "scores.jsonl"))) == 4
    capsys.readouterr()
    score_adapt.main(["cost", "--scores", str(run / "scores.jsonl"), "--metric", "heldout_psnr_dec",
                      "--target", "1000"])
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    assert len(lines) == 2 and all(c["censored"] and c["step"] is None and c["last_step_scored"] == 2 for c in lines)
