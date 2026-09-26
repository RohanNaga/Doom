"""CPU gates for the LoRA adaptation trainer (`adapt_wm.py`) on a tiny next-tic PixArt.

What has to hold, or an adaptation curve cannot be read against the zero-shot numbers:

* **Step 0 is the frozen model's score.** The source's EMA tensors go INTO the model (not only into the EMA
  copy, as `train_wm.py --init-from` does), and the step-0 checkpoint scored by `eval_tf.py` on the split's
  held-out windows equals the frozen snapshot scored with `--use-ema` on the same windows, window by window.
  The launch certificate states the prediction parity, and a nonzero adapter at initialisation is refused.
* **Only the adapter and the parts are saved**, at exactly the grid steps, with the certificate (git,
  args, seed, source path and hash, map, split hash) and enough to rebuild them; the round trip through
  `eval_tf.load_model` gives the saved tensors on top of the source's frozen weights.
* **Windows come only from the adaptation episodes**, or from the first k of the pool under the data grid.
* **The batch rule**: the global batch and the micro-batch are separate, and accumulation needs its flag.

    python -m pytest paper/fixtures/test_adapt_train.py -q
"""
import csv
import glob
import json
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

# the Accelerator is a process-wide singleton; pin it to the CPU before anything builds one
os.environ.setdefault("ACCELERATE_USE_CPU", "1")

import adapt_fixtures as F  # noqa: E402
import adapt_split  # noqa: E402
import adapt_wm  # noqa: E402
import backbones  # noqa: E402
import eval_tf  # noqa: E402
import lora  # noqa: E402


@pytest.fixture
def tiny(monkeypatch):
    F.serve_tiny_pixart(monkeypatch)
    F.stub_lpips(monkeypatch)


@pytest.fixture
def setup(tmp_path):
    """A source snapshot, a 10-episode map and its adaptation split."""
    d = F.map_corpus(tmp_path / "lat")
    src = F.write_source_snapshot(tmp_path / "040-unet-nexttic" / "snap_0200000.pt")
    split = tmp_path / "split_adapt_unseen_map17_seed0.json"
    adapt_split.main(["--episodes", ",".join(map(str, F.EPISODES)), "--set", "unseen", "--map", str(F.MAP),
                      "--latents-dir", d, "--seed", "0", *F.SPLIT_FLAGS, "--context-frames", str(F.CTX),
                      "--out", str(split)])
    return {"lat": d, "src": src, "split": str(split), "tmp": tmp_path}


def train(s, name="run", extra=()):
    out = s["tmp"] / name
    argv = ["--source", s["src"], "--adapt-split", s["split"], "--latents-dir", s["lat"], "--results-dir", str(out),
            "--per-gpu-batch", "4", "--global-batch", "4", "--step-grid", "0,2,3", "--lora-rank", "4",
            "--lora-alpha", "4", "--warmup", "1", "--lr", "1e-2", "--num-workers", "0", "--log-every", "1",
            "--parity-windows", "4", "--no-wandb", *extra]
    assert adapt_wm.main(adapt_wm.build_parser().parse_args(argv)) == 0
    events = [json.loads(ln) for ln in open(out / "log.jsonl")]
    return out, events


def load(path):
    return torch.load(str(path), map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------------------
# flags
# ---------------------------------------------------------------------------------------

def test_accumulation_needs_its_flag_and_the_batch_must_divide():
    assert adapt_wm.resolve_batch(32, 32, 1) == 1
    assert adapt_wm.resolve_batch(32, 8, 4) == 1
    with pytest.raises(SystemExit, match="accumulation of 4"):
        adapt_wm.resolve_batch(32, 8, 1)
    assert adapt_wm.resolve_batch(32, 8, 1, allow_accumulation=True) == 4
    with pytest.raises(SystemExit, match="cannot reach"):
        adapt_wm.resolve_batch(32, 12, 1, allow_accumulation=True)


def test_the_step_grid_parses_and_its_last_step_is_the_run_length():
    assert adapt_wm.parse_grid(adapt_wm.DEFAULT_GRID) == [0, 250, 500, 1000, 2000, 4000]
    assert adapt_wm.parse_grid("1000,0,250,250") == [0, 250, 1000]
    for bad in ("0", "", "a,b", "-1,5"):
        with pytest.raises(SystemExit):
            adapt_wm.parse_grid(bad)


def test_the_defaults_are_the_designs():
    a = adapt_wm.build_parser().parse_args(["--source", "s.pt", "--results-dir", "r"])
    assert (a.lora_rank, a.lora_alpha, a.lora_dropout, a.lora_mlp) == (16, 16.0, 0.0, False)
    assert lora.parse_parts(a.full_parts) == ("control", "input_proj", "noise_emb")
    assert (a.lr, a.warmup, a.ema_decay, a.global_batch, a.source_weights) == (1e-4, 100, 0.999, 32, "ema")
    assert adapt_wm.parse_grid(a.step_grid) == [0, 250, 500, 1000, 2000, 4000] and a.wandb is True


def test_the_recipe_is_the_sources_own():
    r = adapt_wm.source_recipe(F.source_args(noise_aug_max=0.5, noise_buckets=7))
    assert (r["noise_aug_max"], r["noise_buckets"], r["context_frames"]) == (0.5, 7, F.CTX)
    assert (r["action_history"], r["control_bits"], r["tic_stride"], r["objective"]) == (F.CTX, F.BITS, 1, "v")
    with pytest.raises(SystemExit, match="no args"):
        adapt_wm.source_recipe({})


# ---------------------------------------------------------------------------------------
# a run
# ---------------------------------------------------------------------------------------

def test_a_run_writes_the_adapter_at_exactly_the_grid_steps_with_its_certificate(tiny, setup):
    out, events = train(setup)
    files = sorted(os.path.basename(p) for p in glob.glob(str(out / "*.pt")))
    assert files == ["adapter_0000000.pt", "adapter_0000002.pt", "adapter_0000003.pt"]
    ck = load(out / "adapter_0000003.pt")
    assert ck["step"] == 3 and lora.is_adapter_checkpoint(ck)
    # the adapter and the three parts, nothing of the frozen backbone
    names = set(ck["adapter"])
    assert names == set(ck["adapter_ema"])
    assert any(n.endswith("lora_A") for n in names) and "bucket_embedder.weight" in names
    assert "transformer.pos_embed.proj.weight" in names and "control_history.pos" in names
    assert not any(n.endswith(".to_q.weight") or n.startswith("transformer.transformer_blocks.0.norm") for n in names)
    assert all(t.dtype == torch.float32 for t in ck["adapter"].values())
    # the certificate: git, args, seed, source path and hash, map, split hash, effective batch, parity
    from eval_identity import sha256_file
    c = ck["certificate"]
    assert c["source"]["path"] == os.path.abspath(setup["src"]) and c["source"]["sha256"] == sha256_file(setup["src"])
    assert c["source"]["weights"] == "ema" and c["source"]["step"] == 200000
    assert c["split"]["sha256"] == adapt_split.sha256_bytes(setup["split"]) and c["map"] == "unseen_map17"
    assert c["git"]["commit"] and c["seed"] == 0 and c["args"]["lora_rank"] == 4
    assert c["effective_batch"] == 4 and c["accum"] == 1 and c["step0_parity_max_abs"] == 0.0
    line = ck["certificate_line"]
    assert line.startswith("ADAPT_CERTIFICATE ") and "effective_batch=4" in line and "step0_parity_max_abs=0" in line
    assert f"split_sha256={c['split']['sha256']}" in line and "source_weights=ema" in line
    # the source run's args ride along, so every evaluator rebuilds the source graph
    assert ck["args"] == load(setup["src"])["args"]
    kinds = [e["event"] for e in events]
    assert kinds[:2] == ["start", "certificate"] and kinds[-1] == "end"
    assert [e["step"] for e in events if e["event"] == "val"] == [0, 2, 3]
    assert [e["step"] for e in events if e["event"] == "checkpoint"] == [0, 2, 3]
    assert max(e["step"] for e in events if e["event"] == "train") == 3
    assert json.load(open(out / "config.json"))["certificate_line"] == line


def test_step_zero_starts_from_the_source_ema_and_training_moves_only_the_adapter(tiny, setup):
    out, _ = train(setup)
    src = load(setup["src"])
    zero, last = load(out / "adapter_0000000.pt"), load(out / "adapter_0000003.pt")
    for n, t in zero["adapter"].items():
        if n.endswith("lora_B"):
            assert not t.any(), n
        elif not n.endswith("lora_A"):
            # the parts start from the source's EMA tensors, not its live ones
            assert torch.equal(t, src["ema"][n].float()), n
            assert not torch.equal(t, src["model"][n].float()), n
        assert torch.equal(zero["adapter_ema"][n], t)
    assert any(last["adapter"][n].any() for n in last["adapter"] if n.endswith("lora_B"))
    assert any(not torch.equal(last["adapter"][n], last["adapter_ema"][n]) for n in last["adapter"])


def test_the_saved_adapter_round_trips_through_the_evaluator(tiny, setup):
    out, _ = train(setup)
    path = str(out / "adapter_0000003.pt")
    ck, src = load(path), load(setup["src"])
    ns = eval_tf.build_parser().parse_args(["--ckpt", path, "--backbone", "pixart", "--latents-dir", "x",
                                            "--split", "x", "--out-dir", "x", "--context-frames", str(F.CTX),
                                            "--num-actions", "3", "--noise-buckets", "4", "--use-ema"])
    model, step, _ = eval_tf.load_model(ns, "cpu", 4, eval_tf.checkpoint_interface(ck, ns))
    assert step == 3
    params = dict(model.named_parameters())
    for n, t in ck["adapter_ema"].items():
        assert torch.equal(params[n], t), n
    for n, p in params.items():
        if n not in ck["adapter"]:
            assert torch.equal(p, src["ema"][n].float()), n          # the frozen backbone is the source's EMA


def score(s, ckpt, name, extra=()):
    out = str(s["tmp"] / name)
    eval_tf.main(eval_tf.build_parser().parse_args(
        ["--ckpt", ckpt, "--backbone", "pixart", "--pixart-path", backbones.PIXART_DEFAULT, "--latent-channels", "4",
         "--vae-path", str(s["tmp"] / "vae"), "--latents-dir", s["lat"], "--split", s["split"], "--subset", "val",
         "--windows-file", s["split"], "--context-frames", str(F.CTX), "--num-actions", "3", "--noise-buckets", "4",
         "--batch-size", "4", "--steps", "3", "--save-images", "0", "--num-workers", "0", "--out-dir", out, *extra]))
    with open(os.path.join(out, "per_window.csv")) as f:
        return list(csv.DictReader(f))


def test_the_step_zero_score_equals_the_frozen_score_on_the_same_windows(tiny, setup):
    """The zero-shot number on the held-out draw IS the step-0 row, live adapter or EMA adapter alike."""
    F.tiny_vae(setup["tmp"] / "vae")
    out, _ = train(setup)
    frozen = score(setup, setup["src"], "frozen_ema", ["--use-ema"])
    for use_ema in ([], ["--use-ema"]):
        step0 = score(setup, str(out / "adapter_0000000.pt"), f"step0{len(use_ema)}", use_ema)
        assert len(step0) == len(frozen) == 12
        for a, b in zip(step0, frozen):
            assert (a["episode"], a["start"]) == (b["episode"], b["start"])
            for k in ("psnr_dec", "lpips_dec", "latent_mse", "latent_mse_ratio", "hud_psnr_dec"):
                assert a[k] == b[k], (k, a[k], b[k])
    # and the frozen LIVE weights score differently, so the equality above is the EMA's, not a coincidence
    live = score(setup, setup["src"], "frozen_live")
    assert any(a["psnr_dec"] != b["psnr_dec"] for a, b in zip(live, frozen))


def test_the_full_fine_tune_reference_is_the_same_path_at_rank_zero(tiny, setup):
    out, events = train(setup, "full_ft", ["--lora-rank", "0", "--full-parts", "all"])
    ck, src = load(out / "adapter_0000003.pt"), load(setup["src"])
    assert set(ck["adapter"]) == set(src["ema"]) and not any("lora_" in n for n in ck["adapter"])
    assert ck["certificate"]["rank"] == 0 and "parts=all" in ck["certificate_line"]
    start = [e for e in events if e["event"] == "start"][0]
    assert start["params"]["trainable"] == start["params"]["total"]
    ns = eval_tf.build_parser().parse_args(["--ckpt", str(out / "adapter_0000003.pt"), "--backbone", "pixart",
                                            "--latents-dir", "x", "--split", "x", "--out-dir", "x",
                                            "--context-frames", str(F.CTX), "--num-actions", "3",
                                            "--noise-buckets", "4"])
    model, _, _ = eval_tf.load_model(ns, "cpu", 4, eval_tf.checkpoint_interface(ck, ns))
    for n, p in model.named_parameters():
        assert torch.equal(p, ck["adapter"][n]), n
    assert any(not torch.equal(ck["adapter"][n], src["ema"][n].float()) for n in ck["adapter"])


def test_training_windows_come_only_from_the_adaptation_episodes(tiny, setup, monkeypatch):
    seen = {}
    real = adapt_wm.window_datasets

    def spy(recipe, latents_dir, train_ids, held_ids, windows):
        seen["train"], seen["held"] = list(train_ids), list(held_ids)
        tr, va = real(recipe, latents_dir, train_ids, held_ids, windows)
        seen["episodes"] = sorted(int(e[0]) for e in tr.episodes)
        return tr, va

    monkeypatch.setattr(adapt_wm, "window_datasets", spy)
    split = json.load(open(setup["split"]))
    train(setup, "full")
    # by default the step-curve rung: the first step_curve_k (4 here) of the adapt list
    assert seen["train"] == split["train"] == sorted(split["adapt"][:4]) == seen["episodes"]
    assert seen["held"] == split["held_out"]
    train(setup, "k2", ["--adapt-episodes-k", "2"])
    assert seen["episodes"] == sorted(split["adapt"][:2])
    train(setup, "k6", ["--adapt-episodes-k", "6"])
    assert seen["episodes"] == sorted(split["adapt"])
    assert not set(seen["episodes"]) & set(split["held_out"])


def test_a_used_results_directory_is_refused(tiny, setup):
    train(setup)
    with pytest.raises(SystemExit, match="already holds adapter checkpoints"):
        train(setup)


def test_a_nonzero_adapter_at_initialisation_stops_the_run(tiny, setup, monkeypatch):
    real = lora.inject_lora

    def broken(model, *a, **k):
        names = real(model, *a, **k)
        with torch.no_grad():
            for _, p in lora.lora_parameters(model):
                p.fill_(0.1)
        return names

    monkeypatch.setattr(lora, "inject_lora", broken)
    with pytest.raises(SystemExit, match="step-0 parity failed"):
        train(setup)


def test_a_fit_check_runs_on_synthetic_windows_and_writes_no_checkpoint(tiny, setup):
    out = setup["tmp"] / "fit"
    argv = ["--source", setup["src"], "--results-dir", str(out), "--per-gpu-batch", "2", "--global-batch", "4",
            "--allow-accumulation", "--fit-check", "2", "--num-workers", "0", "--lora-rank", "2", "--no-wandb"]
    assert adapt_wm.main(adapt_wm.build_parser().parse_args(argv)) == 0
    events = [json.loads(ln) for ln in open(out / "log.jsonl")]
    fit = [e for e in events if e["event"] == "fit_check"][0]
    assert fit["steps"] == 2 and fit["accum"] == 2 and fit["global_batch"] == 4 and fit["data"] == "synthetic"
    assert not glob.glob(str(out / "*.pt")) and not os.path.exists(out / "config.json")
