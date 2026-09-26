"""`eval_tf.py` scores recorded windows and LoRA adaptation checkpoints.

* **The windows are the recorded ones.** `--windows-file <adaptation split>` scores exactly the
  [episode, start] pairs the split recorded, every one from a held-out episode, and nothing else.
* **The legacy cross-check reproduces the frozen scores.** Scored against the map's own distance-study
  split, the legacy windows keep the study's dataset indices and so its noise keys; their per-window
  numbers equal the study draw's numbers on the same windows to float precision (a window's batch
  companions move the last digits, which is a property of eval_tf.py batching, not of the windows).
* **An adaptation checkpoint rebuilds exactly.** The frozen source it names (checked by hash), with the
  adapter live or EMA, gives the same predictions as the adapted model it was saved from; a source whose
  hash has changed is refused.

    python -m pytest paper/fixtures/test_adapt_eval.py -q
"""
import csv
import json
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

os.environ.setdefault("ACCELERATE_USE_CPU", "1")

import adapt_fixtures as F  # noqa: E402
import adapt_split  # noqa: E402
import backbones  # noqa: E402
import eval_tf  # noqa: E402
import lora  # noqa: E402


@pytest.fixture
def tiny(monkeypatch):
    F.serve_tiny_pixart(monkeypatch)
    F.stub_lpips(monkeypatch)


@pytest.fixture
def one_map(tmp_path):
    """A 10-episode map, its distance-study split file and its adaptation split (seed 0)."""
    d = F.map_corpus(tmp_path / "lat")
    map_split = tmp_path / "split_unseen_map17.json"
    map_split.write_text(json.dumps({"val": list(F.EPISODES), "meta": {"set": "unseen", "map": F.MAP}}))
    out = tmp_path / "split_adapt.json"
    adapt_split.main(["--map-split", str(map_split), "--latents-dir", d, "--seed", "0", *F.SPLIT_FLAGS,
                      "--context-frames", str(F.CTX), "--legacy-windows", "40", "--out", str(out)])
    return d, str(map_split), str(out), tmp_path


def run_eval(tmp_path, ckpt, split, latents, name, extra=()):
    out = str(tmp_path / name)
    eval_tf.main(eval_tf.build_parser().parse_args(
        ["--ckpt", ckpt, "--backbone", "pixart", "--pixart-path", backbones.PIXART_DEFAULT, "--latent-channels", "4",
         "--vae-path", str(tmp_path / "vae"), "--latents-dir", latents, "--split", split, "--subset", "val",
         "--context-frames", str(F.CTX), "--num-actions", "3", "--noise-buckets", "4", "--batch-size", "5",
         "--steps", "2", "--save-images", "0", "--num-workers", "0", "--out-dir", out, *extra]))
    with open(os.path.join(out, "per_window.csv")) as f:
        rows = list(csv.DictReader(f))
    return json.load(open(os.path.join(out, "metrics.json"))), rows


def pairs(rows):
    return sorted((int(r["episode"]), int(r["start"])) for r in rows)


def test_the_scored_windows_are_the_recorded_held_out_windows(tiny, one_map):
    d, _, split, tmp = one_map
    F.tiny_vae(tmp / "vae")
    src = F.write_source_snapshot(tmp / "src" / "snap_0200000.pt")
    s = json.load(open(split))
    m, rows = run_eval(tmp, src, split, d, "held", ["--use-ema", "--windows-file", split])
    assert pairs(rows) == sorted(map(tuple, s[adapt_split.WINDOW_KEY]))
    assert {int(r["episode"]) for r in rows} <= set(s["held_out"])
    assert m["psnr_dec"]["n"] == len(s[adapt_split.WINDOW_KEY]) == 12
    assert m["config"]["windows_file"] == split and m["config"]["windows_key"] == adapt_split.WINDOW_KEY


def test_the_legacy_windows_reproduce_the_studys_per_window_scores(tiny, one_map):
    d, map_split, split, tmp = one_map
    F.tiny_vae(tmp / "vae")
    src = F.write_source_snapshot(tmp / "src" / "snap_0200000.pt")
    s = json.load(open(split))
    _, study = run_eval(tmp, src, map_split, d, "study", ["--use-ema", "--num-windows", "40"])
    _, legacy = run_eval(tmp, src, map_split, d, "legacy",
                         ["--use-ema", "--windows-file", split, "--windows-key", adapt_split.LEGACY_KEY])
    held = set(s["held_out"])
    want = {(int(r["episode"]), int(r["start"])): r for r in study if int(r["episode"]) in held}
    assert pairs(legacy) == sorted(want) and 0 < len(legacy) < 40
    for r in legacy:
        w = want[(int(r["episode"]), int(r["start"]))]
        assert r["index"] == w["index"]                       # same dataset index, hence the same noise keys
        # the same window, noise and weights; only the batch it shares moves the last float digits
        for k in ("psnr_dec", "latent_mse", "copy_latent_mse", "lpips_dec"):
            assert float(r[k]) == pytest.approx(float(w[k]), rel=1e-4, abs=1e-6), k


def test_without_a_windows_file_the_draw_is_unchanged(tiny, one_map):
    d, map_split, _, tmp = one_map
    F.tiny_vae(tmp / "vae")
    src = F.write_source_snapshot(tmp / "src" / "snap_0200000.pt")
    m, rows = run_eval(tmp, src, map_split, d, "plain", ["--use-ema", "--num-windows", "9", "--seed", "4"])
    from doom_data import TicWindowDataset
    ds = TicWindowDataset(d, list(F.EPISODES), F.CTX, action_history=F.CTX)
    assert sorted(int(r["index"]) for r in rows) == eval_tf.draw_windows(len(ds), 9, 4).tolist()
    assert m["config"]["windows_file"] == ""


def adapted_checkpoint(tmp, weights="ema", perturb=0.05, **cfg_over):
    """An adaptation checkpoint in the trainer's format, and the adapted model it was saved from."""
    from eval_identity import sha256_file
    src = F.write_source_snapshot(tmp / "src" / "snap_0200000.pt")
    model = F.tiny_source_model(seed=9)
    lora.load_source_weights(model, torch.load(src, weights_only=False), weights)
    cfg = lora.adapter_config("pixart", 4, 8, 0.0, False, lora.FULL_PARTS, seed=3, **cfg_over)
    cfg["targets"] = lora.inject_lora(model, cfg["rank"], cfg["alpha"], cfg["dropout"], cfg["include_mlp"], cfg["seed"])
    names = lora.trained_names(model, "pixart", cfg["parts"])
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in names:
                p.add_(perturb * torch.randn(p.shape, generator=g))
    ema = {n: p.detach() * 0.5 for n, p in model.named_parameters() if n in names}
    ck = lora.adapter_checkpoint(model, names, ema, cfg,
                                 {"path": src, "sha256": sha256_file(src), "weights": weights, "step": 200000},
                                 F.source_args(), step=250)
    path = tmp / "adapt" / "adapter_0000250.pt"
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    torch.save(ck, str(path))
    return str(path), model, ema, src


def ns(ckpt, use_ema):
    return eval_tf.build_parser().parse_args(["--ckpt", ckpt, "--backbone", "pixart", "--latents-dir", "x",
                                              "--split", "x", "--out-dir", "x", "--context-frames", str(F.CTX),
                                              "--num-actions", "3", "--noise-buckets", "4"]
                                             + (["--use-ema"] if use_ema else []))


def inputs(n=3, seed=5):
    g = torch.Generator().manual_seed(seed)
    return dict(x=torch.randn(n, 4, 32, 40, generator=g), t=torch.tensor([10, 500, 990])[:n],
                action=torch.randint(0, 2, (n, F.CTX, F.BITS), generator=g).float(),
                context=torch.randn(n, 4 * F.CTX, 32, 40, generator=g), noise_bucket=torch.tensor([0, 1, 3])[:n])


@pytest.mark.parametrize("weights", ["ema", "live"])
def test_an_adaptation_checkpoint_rebuilds_the_adapted_model(tiny, tmp_path, weights):
    path, model, ema, _ = adapted_checkpoint(tmp_path, weights)
    ck = torch.load(path, weights_only=False)
    trained = eval_tf.checkpoint_interface(ck, ns(path, False))
    assert trained["action_history"] == F.CTX and trained["tic_stride"] == 1
    kw = inputs()
    live, step, objective = eval_tf.load_model(ns(path, False), "cpu", 4, trained)
    assert step == 250 and objective == "v"
    with torch.no_grad():
        assert torch.equal(live(**kw), model.eval()(**kw))
        with_ema, _, _ = eval_tf.load_model(ns(path, True), "cpu", 4, trained)
        for n, p in model.named_parameters():
            if n in ema:
                p.copy_(ema[n])
        assert torch.equal(with_ema(**kw), model(**kw))


def test_a_source_whose_bytes_changed_is_refused(tiny, tmp_path):
    path, _, _, src = adapted_checkpoint(tmp_path)
    ck = torch.load(src, weights_only=False)
    ck["step"] = 1
    torch.save(ck, src)
    trained = eval_tf.checkpoint_interface(torch.load(path, weights_only=False), ns(path, False))
    with pytest.raises(SystemExit, match="SHA-256"):
        eval_tf.load_model(ns(path, False), "cpu", 4, trained)
