"""Offline review reproductions for eaf565a; assertions describe the required behavior.

These tests intentionally fail until the reviewed defects are fixed. All scorer files
are in memory: no model downloads, GPU, W&B calls, or result directories are needed.
Run with PYTHONDONTWRITEBYTECODE=1 and pytest -p no:cacheprovider.
"""
import builtins
import csv
import io
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import adapt_split
import score_adapt


@pytest.fixture
def memory_files(monkeypatch):
    files = {}
    real_open = builtins.open

    class Writer(io.StringIO):
        def __init__(self, path, initial=""):
            super().__init__(initial)
            self.path = path
            self.seek(0, io.SEEK_END)

        def close(self):
            if not self.closed:
                files[self.path] = self.getvalue()
            super().close()

    def open_memory(path, mode="r", *args, **kwargs):
        path = os.fspath(path)
        if not path.startswith("/review/"):
            return real_open(path, mode, *args, **kwargs)
        if "w" in mode or "a" in mode:
            return Writer(path, files.get(path, "") if "a" in mode else "")
        return io.StringIO(files[path])

    monkeypatch.setattr(builtins, "open", open_memory)
    return files


@pytest.fixture
def scoring(monkeypatch, memory_files):
    """Exercise the real score driver and paired summarizer with tiny mocked evaluator output."""
    import eval_identity

    files = memory_files
    split = "/review/split.json"
    run = "/review/run"
    path = run + "/adapter_0000000.pt"
    files[split] = json.dumps({"meta": {"latents_dir": "/review/lat", "set": "arenas13", "map": 8}})
    state = {"a": torch.tensor([0.0])}
    ck = {
        "recipe": {"backbone": "unet", "latent_channels": 4, "context_frames": 32,
                   "num_actions": 29, "noise_buckets": 10},
        "certificate": {"step_grid": [0, 250, 500, 1000, 2000, 4000], "adapt_episodes_k": 8},
        "split": {"path": split, "sha256": "split-hash"},
        "adapter": state, "adapter_ema": state, "map": "arenas13_map08", "episodes": list(range(8)),
        "adapt_args": {"seed": 0}, "source": {"sha256": "source-hash", "weights": "ema"},
        "adapter_config": {"rank": 16, "alpha": 16, "include_mlp": False,
                           "parts": ["control", "input_proj", "noise_emb"]},
    }
    monkeypatch.setattr(score_adapt, "run_checkpoints", lambda _: [(0, path)])
    monkeypatch.setattr(torch, "load", lambda *a, **k: ck)
    monkeypatch.setattr(adapt_split, "sha256_bytes", lambda _: "split-hash")
    monkeypatch.setattr(adapt_split, "git_state", lambda: {"commit": "eaf565a", "dirty": 0})
    monkeypatch.setattr(eval_identity, "sha256_file", lambda _: "checkpoint-hash")
    monkeypatch.setattr(score_adapt, "read_rows",
                        lambda p: [json.loads(s) for s in files.get(p, "").splitlines() if s])
    calls = []

    def evaluate(argv):
        calls.append(argv)
        out = argv[argv.index("--out-dir") + 1]
        # Distinguish two legitimate sampling configurations in the saved evidence.
        mse = float(argv[argv.index("--steps") + 1])
        files[out + "/per_window.csv"] = (
            "episode,start,index,latent_mse,copy_latent_mse,psnr_raw,persist_psnr_raw,lpips_raw,persist_lpips_raw\n"
            f"20,32,0,{mse},1,22,21,0.1,0.2\n"
        )
        return {"psnr_dec": {"mean": 24.0, "n": 1}, "copy_psnr_dec": {"mean": 22.0},
                "psnr_raw": {"mean": 22.0}, "persist_psnr_raw": {"mean": 21.0}}

    monkeypatch.setattr(score_adapt, "run_eval_tf", evaluate)
    argv = ["score", "--run-dir", run, "--weights", "live", "--guard-steps", "none", "--no-wandb"]
    return SimpleNamespace(files=files, calls=calls, argv=argv, scores=run + "/scores.jsonl")


@pytest.mark.parametrize("changed", [
    ["--latent-scale", "0.5"],
    ["--latent-shift", "0.2"],
    ["--vae-subfolder", "other-decoder"],
    ["--latents-dir", "/review/another-corpus"],
])
def test_changed_evaluation_inputs_are_not_silently_cached(scoring, changed):
    score_adapt.main(scoring.argv)
    score_adapt.main(scoring.argv + changed)
    assert len(scoring.calls) == 2, "an output-affecting input changed, but the previous row was reused"


def test_rescoring_keeps_the_old_rows_per_window_evidence(scoring):
    score_adapt.main(scoring.argv)
    old = score_adapt.read_rows(scoring.scores)[0]
    before = scoring.files[old["heldout_per_window"]]
    score_adapt.main(scoring.argv + ["--steps", "20"])
    assert len(scoring.calls) == 2
    assert scoring.files[old["heldout_per_window"]] == before, "a later score overwrote the earlier row's CSV"


def test_cost_does_not_pool_stock_and_tuned_curves(monkeypatch, capsys):
    rows = []
    for decoder, values in (("stock", (2.0, 2.0)), ("tuned", (0.0, 0.1))):
        for step, value in zip((0, 4000), values):
            rows.append({"run": "run", "map": "map08", "seed": 0, "weights": "live", "step": step,
                         "decoder": decoder, "eval_seed": 0, "sampler_steps": 10,
                         "windows_key": "held_out_windows", "metric": value})
    monkeypatch.setattr(score_adapt, "read_rows", lambda _: rows)
    try:
        score_adapt.main(["cost", "--scores", "unused", "--metric", "metric", "--target", "1"])
    except (ValueError, SystemExit):
        return  # Rejecting an ambiguous mixture is also correct.
    costs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(costs) == 2 and sum(c["censored"] for c in costs) == 1, costs


def test_nan_is_not_an_observation_extending_the_censoring_budget():
    result = score_adapt.adaptation_cost(
        [{"step": 0, "gain": 0.0}, {"step": 250, "gain": 0.1}, {"step": 4000, "gain": float("nan")}],
        "gain", 1.0,
    )
    assert result["censored"]
    assert result["last_step_scored"] == 250, result


def test_default_guards_cover_every_curve_point():
    args = score_adapt.build_parser().parse_args(["score", "--run-dir", "unused"])
    grid = [0, 250, 500, 1000, 2000, 4000]
    assert score_adapt.parse_steps(args.guard_steps, grid) == set(grid)


def test_a_perfect_prediction_counts_in_the_arithmetic_ratio(memory_files):
    src, dst = "/review/input.csv", "/review/paired.csv"
    memory_files[src] = (
        "episode,start,index,latent_mse,copy_latent_mse\n"
        "1,32,0,0,1\n"
        "2,32,1,1,1\n"
    )
    result = score_adapt.window_columns("heldout", src, dst)
    assert result["heldout_latent_ratio_mean"] == pytest.approx(0.5), result


def test_one_record_jsonl_manifest_is_accepted(memory_files):
    path = "/review/worker_00.jsonl"
    memory_files[path] = json.dumps({"episode_id": 16, "map_id": 8}) + "\n"
    assert adapt_split.manifest_maps(path) == {8: [16]}


def test_existing_split_refuses_changed_corpus_provenance(memory_files, monkeypatch):
    path = "/review/split.json"
    old = {"adapt": [1], "held_out": [2], "held_out_windows": [[2, 32]],
           "meta": {"latents_dir": "/review/old-corpus", "source_sha256": "old"}}
    memory_files[path] = json.dumps(old)
    new = {**old, "meta": {"latents_dir": "/review/fresh-corpus", "source_sha256": "fresh"}}
    real_exists = os.path.exists
    monkeypatch.setattr(os.path, "exists", lambda p: p in memory_files or real_exists(p))
    with pytest.raises(SystemExit):
        adapt_split.write_split(new, path)


def test_split_requires_32_valid_windows_from_each_held_out_episode(monkeypatch):
    monkeypatch.setattr(adapt_split, "window_starts",
                        lambda directory, ids, *a: [(e, np.arange(31)) for e in ids])
    monkeypatch.setattr(adapt_split, "git_state", lambda: {"commit": "eaf565a", "dirty": 0})
    with pytest.raises((ValueError, SystemExit)):
        adapt_split.build_split([1, 2], "arenas13", 8, "/review/lat", n_adapt=1,
                                n_held_out=1, ladder=(1,), step_curve_k=1)
