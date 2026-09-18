"""CPU gates for the twelve-cell knob grid's trainer flags.

Each cell of the grid turns exactly one knob relative to the base PixArt-alpha recipe, so every
new flag has to default to the behaviour the five finished rows were trained with, and has to be
readable back out of a checkpoint by the evaluators. These checks pin both halves on the CPU:

* `--train-fraction`: nested seeded subsets of the *training* episode list only.
* `--objective {v,eps}`: the same betas, the same x0 estimate at a fixed (x_t, t), the objective
  recorded in the checkpoint and read back by the samplers.
* `--action-inject {token,adaln}`: additive injection into PixArt's adaLN-single path, zero-init
  so step 0 is still the pretrained model.
* the tiny fixture path: five real training updates and a two-window evaluation for each cell's
  flag combination, with a DiT-S/2 stand-in for the heavy warm-started backbones.

The PixArt and SD 3.5 transformers themselves are not built here (network access and gigabytes of
weights); `verify_sd35.py` and a GPU run gate those.

    python -m pytest paper/fixtures/test_grid_flags.py -q
"""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import backbones  # noqa: E402
from diffusion_v import VDiffusion, checkpoint_objective  # noqa: E402
from doom_data import select_train_episodes  # noqa: E402

TRAIN_IDS = list(range(100, 164))          # 64 training episodes
VAL_IDS = list(range(200, 208))


# ---- --train-fraction ------------------------------------------------------------------------

def test_full_fraction_is_the_whole_training_list():
    assert select_train_episodes(TRAIN_IDS, 1.0) == sorted(TRAIN_IDS)


def test_fraction_sizes_are_rounded_counts():
    for frac, n in ((0.5, 32), (0.25, 16), (0.125, 8)):
        assert len(select_train_episodes(TRAIN_IDS, frac)) == n


def test_subsets_are_nested_so_the_ladder_varies_only_in_size():
    eighth, quarter, half = (set(select_train_episodes(TRAIN_IDS, f)) for f in (0.125, 0.25, 0.5))
    assert eighth < quarter < half < set(TRAIN_IDS)


def test_subset_is_seeded_and_reproducible():
    assert select_train_episodes(TRAIN_IDS, 0.25, seed=0) == select_train_episodes(TRAIN_IDS, 0.25, seed=0)
    assert select_train_episodes(TRAIN_IDS, 0.25, seed=1) != select_train_episodes(TRAIN_IDS, 0.25, seed=0)


def test_subset_never_reaches_the_validation_episodes():
    # the function only ever sees the train list, which is what keeps validation and evaluation
    # identical across the data cells
    assert not set(select_train_episodes(TRAIN_IDS, 0.125)) & set(VAL_IDS)


def test_at_least_one_episode_survives_a_tiny_fraction():
    assert len(select_train_episodes(TRAIN_IDS, 0.001)) == 1


def test_bad_fractions_are_refused():
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            select_train_episodes(TRAIN_IDS, bad)


def test_ids_come_back_sorted_and_unique():
    got = select_train_episodes([5, 1, 3, 2, 4, 6, 7, 8], 0.5)
    assert got == sorted(set(got))


# ---- --objective {v,eps} ---------------------------------------------------------------------

FIXED_T = (0, 1, 37, 500, 998, 999)


def _fixed_batch(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, 4, 8, 10, generator=g)
    noise = torch.randn(n, 4, 8, 10, generator=g)
    return x0, noise


def test_both_objectives_share_the_ddpm_betas():
    v, eps = VDiffusion(), VDiffusion(objective="eps")
    assert torch.equal(v.sqrt_abar, eps.sqrt_abar)
    assert torch.equal(v.sqrt_1m_abar, eps.sqrt_1m_abar)
    assert v.num_steps == eps.num_steps == 1000


def test_unknown_objective_is_refused():
    with pytest.raises(ValueError):
        VDiffusion(objective="x0")


def test_the_two_parameterisations_give_the_same_x0_at_a_fixed_xt_and_t():
    """The gate Astra asked for: a perfect v prediction and the equivalent perfect eps prediction
    must reconstruct the same x0 from the same (x_t, t), to float tolerance."""
    dv, de = VDiffusion(objective="v"), VDiffusion(objective="eps")
    x0, noise = _fixed_batch()
    worst = 0.0
    for t_int in FIXED_T:
        t = torch.full((x0.shape[0],), t_int, dtype=torch.long)
        xt = dv.q_sample(x0, t, noise)
        v_pred = dv.target(x0, t, noise)                  # = v_target
        eps_pred = dv.eps_from_v(xt, t, v_pred)           # the same network output, re-expressed
        x0_v = dv.x0_from_pred(xt, t, v_pred)
        x0_e = de.x0_from_pred(xt, t, eps_pred)
        worst = max(worst, float((x0_v - x0_e).abs().max()), float((x0_v - x0).abs().max()))
    assert worst < 1e-4, f"x0 estimates disagree by {worst}"


def test_eps_target_is_the_noise_and_v_target_is_the_velocity():
    x0, noise = _fixed_batch()
    t = torch.full((x0.shape[0],), 400, dtype=torch.long)
    assert torch.equal(VDiffusion(objective="eps").target(x0, t, noise), noise)
    dv = VDiffusion(objective="v")
    assert torch.equal(dv.target(x0, t, noise), dv.v_target(x0, t, noise))


def test_a_perfect_model_samples_the_same_latent_in_both_parameterisations():
    """Full respaced DDIM with an oracle: identical trajectories, so the eps branch of the sampler
    is the same sampler and not a second recipe."""
    dv, de = VDiffusion(objective="v"), VDiffusion(objective="eps")
    x0, _ = _fixed_batch(n=2, seed=3)
    start = torch.randn(2, 4, 8, 10, generator=torch.Generator().manual_seed(11))

    # each oracle needs the true noise at (x_t, t), which x0 determines
    def v_oracle(xt, t):
        a, s = dv._coef(t, xt.ndim)
        eps = (xt - a * x0) / s.clamp_min(1e-8)
        return dv.v_target(x0, t, eps)

    def eps_oracle(xt, t):
        a, s = de._coef(t, xt.ndim)
        return (xt - a * x0) / s.clamp_min(1e-8)

    out_v = dv.ddim_sample(v_oracle, x0.shape, steps=5, noise=start)
    out_e = de.ddim_sample(eps_oracle, x0.shape, steps=5, noise=start)
    assert torch.allclose(out_v, out_e, atol=1e-4), float((out_v - out_e).abs().max())
    assert torch.allclose(out_v, x0, atol=1e-3), float((out_v - x0).abs().max())


def test_checkpoint_objective_defaults_to_velocity_for_the_finished_rows():
    # every checkpoint written before the flag existed is a velocity model and carries no key
    assert checkpoint_objective({"args": {"backbone": "pixart"}}) == "v"
    assert checkpoint_objective({}) == "v"
    assert checkpoint_objective({"args": None}) == "v"


def test_checkpoint_objective_reads_what_training_recorded():
    assert checkpoint_objective({"args": {"objective": "eps"}}) == "eps"


def test_checkpoint_objective_override_wins():
    assert checkpoint_objective({"args": {"objective": "eps"}}, "v") == "v"
    assert checkpoint_objective({"args": {"objective": "eps"}}, "auto") == "eps"


@pytest.mark.parametrize("script", ["eval_tf.py", "rollout_eval.py"])
def test_the_evaluators_take_an_objective_flag_defaulting_to_auto(script):
    import subprocess
    out = subprocess.run([sys.executable, os.path.join(REPO, script), "--help"],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "--objective" in out.stdout
    assert "auto" in out.stdout


# ---- --warm-start none -----------------------------------------------------------------------

def test_scratch_keeps_the_pixart_repo_for_its_config_but_loads_no_weights():
    source, scratch = backbones.resolve_warm_start("pixart", "none")
    assert (source, scratch) == (backbones.PIXART_DEFAULT, True)


def test_scratch_is_case_and_space_insensitive():
    for spelling in ("none", "NONE", " None "):
        assert backbones.resolve_warm_start("pixart", spelling)[1] is True


def test_a_named_warm_start_is_left_alone():
    assert backbones.resolve_warm_start("pixart", "/weights/pixart") == ("/weights/pixart", False)
    assert backbones.resolve_warm_start("pixart", None) == (None, False)


def test_the_dit_scratch_path_is_simply_no_checkpoint():
    assert backbones.resolve_warm_start("dit", "none") == (None, True)


def test_scratch_is_refused_where_it_is_not_implemented():
    for backbone in ("unet", "unidiffuser", "sd35"):
        with pytest.raises(NotImplementedError):
            backbones.resolve_warm_start(backbone, "none")


# ---- --action-inject {token,adaln} -----------------------------------------------------------

def _tiny_pixart():
    """A 16k-parameter PixArt transformer with PixArt's own module graph, built from a config so the
    adaLN-single injection can be gated without the 611M checkpoint."""
    from diffusers import PixArtTransformer2DModel
    return PixArtTransformer2DModel(num_attention_heads=2, attention_head_dim=8, in_channels=4, out_channels=8,
                                    num_layers=2, caption_channels=32, sample_size=64, patch_size=2,
                                    cross_attention_dim=16, use_additional_conditions=False, norm_num_groups=2)


def _tiny_world_model(inject):
    return backbones.PixArtWorldModel(num_actions=3, context_frames=2, noise_buckets=4, action_dropout=0.0,
                                      grad_ckpt=False, action_inject=inject, transformer=_tiny_pixart())


def _batch(model, n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = model.latent_channels
    return (torch.randn(n, c, 32, 40, generator=g), torch.full((n,), 300, dtype=torch.long),
            torch.tensor([1, 2])[:n], torch.randn(n, c * model.context_frames, 32, 40, generator=g),
            torch.tensor([0, 3])[:n])


def test_the_adaln_injection_wrapper_is_transparent_until_something_is_stashed():
    from diffusers.models.normalization import AdaLayerNormSingle
    norm = AdaLayerNormSingle(16, use_additional_conditions=False)
    plain = norm.emb
    t = torch.tensor([13, 700])
    before = plain(t, resolution=None, aspect_ratio=None, batch_size=2, hidden_dtype=torch.float32)
    norm.emb = backbones.AdaLNActionInjection(plain)
    after = norm.emb(t, resolution=None, aspect_ratio=None, batch_size=2, hidden_dtype=torch.float32)
    assert torch.equal(before, after)
    norm.emb.extra = torch.ones_like(after)
    assert torch.allclose(norm.emb(t, resolution=None, aspect_ratio=None, batch_size=2, hidden_dtype=torch.float32),
                          after + 1.0)


def test_both_injection_modes_return_the_velocity_shape():
    for inject in backbones.ACTION_INJECTIONS:
        m = _tiny_world_model(inject)
        out = m(*_batch(m))
        assert out.shape == (2, m.latent_channels, 32, 40), inject


def test_adaln_injection_is_zero_initialised_so_step_zero_is_still_the_pretrained_model():
    m = _tiny_world_model("adaln")
    assert float(m.action_embedder.weight.detach().abs().max()) == 0.0
    assert float(m.bucket_embedder.weight.detach().abs().max()) == 0.0
    assert float(m.null_caption.detach().abs().max()) == 0.0
    x, t, act, ctx, bucket = _batch(m)
    with torch.no_grad():
        mixed = m(x, t, act, ctx, bucket)
        other = m(x, t, act * 0, ctx, bucket * 0)
    assert torch.equal(mixed, other), "a zero-initialised injection must not depend on the action yet"


def test_adaln_injection_widths_follow_the_transformer():
    m = _tiny_world_model("adaln")
    inner = m.transformer.config.num_attention_heads * m.transformer.config.attention_head_dim
    assert m.action_embedder.weight.shape == (3 + 1, inner)      # +1 null row for action dropout
    assert m.bucket_embedder.weight.shape == (4, inner)
    tok = _tiny_world_model("token")
    assert tok.action_embedder.weight.shape[1] == tok.transformer.config.caption_channels


def test_both_injection_embedders_receive_gradient():
    for inject in backbones.ACTION_INJECTIONS:
        m = _tiny_world_model(inject)
        m.train()
        m(*_batch(m)).square().mean().backward()
        for name in ("action_embedder", "bucket_embedder"):
            g = getattr(m, name).weight.grad
            assert g is not None and float(g.abs().sum()) > 0, f"{inject}/{name} got no gradient"


def test_the_adaln_stash_is_cleared_after_a_forward():
    m = _tiny_world_model("adaln")
    with torch.no_grad():
        m(*_batch(m))
    assert m.transformer.adaln_single.emb.extra is None


def test_an_unknown_injection_is_refused():
    with pytest.raises(ValueError):
        _tiny_world_model("film")


def test_the_injection_knob_is_refused_for_the_other_backbones():
    for backbone in ("dit", "unet", "unidiffuser", "sd35"):
        with pytest.raises(NotImplementedError):
            backbones.build_model(backbone, 3, 2, action_inject="adaln")


def test_the_trainer_exposes_every_cell_knob():
    import subprocess
    train = subprocess.run([sys.executable, os.path.join(REPO, "train_wm.py"), "--help"],
                           capture_output=True, text=True, cwd=REPO)
    assert train.returncode == 0, train.stderr[-2000:]
    for flag in ("--train-fraction", "--objective", "--action-inject", "--warm-start",
                 "--context-frames", "--noise-aug-max", "--lr"):
        assert flag in train.stdout, flag
    assert "adaln" in train.stdout and "none" in train.stdout


# ---- the tiny fixture path: five updates and a two-window evaluation per cell -----------------

# one knob per cell, exactly as scripts/spiderman/launch_cell.sh sets them
CELL_FLAGS = {
    "base30k": [],
    "data-1_8": ["--train-fraction", "0.125"],
    "data-1_4": ["--train-fraction", "0.25"],
    "data-1_2": ["--train-fraction", "0.5"],
    "scratch": ["--warm-start", "none"],
    "eps": ["--objective", "eps"],
    "ctx8": ["--context-frames", "8"],
    "ctx16": ["--context-frames", "16"],
    "noaug": ["--noise-aug-max", "0.0"],
    "adaln": ["--action-inject", "adaln"],
    "lr1e-4": ["--lr", "1e-4"],
    "lr2.5e-5": ["--lr", "2.5e-5"],
}
TINY_PIXART = dict(num_attention_heads=2, attention_head_dim=8, in_channels=4, out_channels=8, num_layers=2,
                   caption_channels=32, sample_size=64, patch_size=2, cross_attention_dim=16,
                   use_additional_conditions=False, norm_num_groups=2)
FIXTURE_EPISODES, FIXTURE_FRAMES = 10, 40


@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory):
    """A latent corpus with the real on-disk layout: ep_*_latents.npy plus verified-transition metadata."""
    import json

    import numpy as np
    d = tmp_path_factory.mktemp("latents")
    rng = np.random.RandomState(0)
    for ep in range(FIXTURE_EPISODES):
        np.save(d / f"ep_{ep:05d}_latents.npy",
                rng.randn(FIXTURE_FRAMES, 4, 32, 40).astype(np.float16))
        np.savez(d / f"ep_{ep:05d}_meta.npz", action=rng.randint(0, 3, FIXTURE_FRAMES).astype(np.int64),
                 tic=(np.arange(FIXTURE_FRAMES) * 4).astype(np.int64),
                 map_id=np.full(FIXTURE_FRAMES, 1, np.int64), chain_id=np.zeros(FIXTURE_FRAMES, np.int64))
    split = str(d / "split.json")
    json.dump({"train": list(range(8)), "val": [8, 9]}, open(split, "w"))
    return str(d), split


@pytest.fixture
def tiny_hub(monkeypatch):
    """Serve a 16k-parameter transformer wherever PixArt would pull the 611M checkpoint, so the real
    `build_model` -> `resolve_warm_start` -> `PixArtWorldModel` path runs on the CPU."""
    from diffusers import PixArtTransformer2DModel as P
    monkeypatch.setattr(P, "from_pretrained", classmethod(lambda cls, *a, **k: cls(**TINY_PIXART)))
    monkeypatch.setattr(P, "load_config", classmethod(lambda cls, *a, **k: dict(TINY_PIXART)))
    monkeypatch.setenv("ACCELERATE_USE_CPU", "1")


def _train_cell(flags, corpus, out_dir):
    import train_wm
    latents, split = corpus
    argv = ["--backbone", "pixart", "--warm-start", backbones.PIXART_DEFAULT,
            "--latents-dir", latents, "--split", split, "--results-dir", out_dir,
            "--num-actions", "3", "--noise-buckets", "4", "--context-frames", "32",
            "--per-gpu-batch", "2", "--global-batch", "2", "--steps", "5", "--warmup", "2",
            "--val-every", "5", "--val-windows", "2", "--ckpt-every", "5", "--ema-every", "1",
            "--num-workers", "0", "--action-dropout", "0.0", "--require-verified-transitions",
            "--seed", "0"] + flags
    train_wm.main(train_wm.build_parser().parse_args(argv))


@torch.no_grad()
def _evaluate_two_windows(out_dir, corpus, cell):
    """Teacher-forced sampling of two held-out windows, the way eval_tf.py does it minus the VAE:
    rebuild the graph the checkpoint records, sample in the parameterization it records."""
    from doom_data import LatentWindowDataset, load_split
    latents, split_path = corpus
    ck = torch.load(os.path.join(out_dir, "best.pt"), map_location="cpu", weights_only=False)
    trained = ck["args"]
    model = backbones.build_model("pixart", trained["num_actions"], trained["context_frames"],
                                  trained["noise_buckets"], grad_ckpt=False,
                                  warm_start=backbones.PIXART_DEFAULT, action_dropout=0.0,
                                  action_inject=trained.get("action_inject", "token"))
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()}, strict=True)
    model.eval()
    diffusion = VDiffusion(objective=checkpoint_objective(ck))
    ds = LatentWindowDataset(latents, load_split(split_path)["val"], trained["context_frames"])
    ctx = torch.stack([ds[i][0] for i in range(2)])
    tgt = torch.stack([ds[i][1] for i in range(2)])
    act = torch.stack([ds[i][2] for i in range(2)])
    bucket = torch.zeros(2, dtype=torch.long)
    pred = diffusion.ddim_sample(lambda xt, t: model(xt, t, act, ctx, bucket), tgt.shape, steps=2)
    assert pred.shape == tgt.shape, cell
    assert torch.isfinite(pred).all(), f"{cell}: non-finite sample"
    return diffusion.objective, trained


@pytest.mark.parametrize("cell", list(CELL_FLAGS))
def test_every_cell_trains_five_updates_and_evaluates_two_windows(cell, tiny_corpus, tiny_hub, tmp_path):
    import json
    out = str(tmp_path / cell)
    _train_cell(CELL_FLAGS[cell], tiny_corpus, out)

    events = [json.loads(line) for line in open(os.path.join(out, "log.jsonl"))]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "end", cell
    assert [e for e in events if e["event"] == "val"], f"{cell}: no validation ran"
    assert events[-1]["step"] == 5, cell
    assert all(e["val_loss"] == e["val_loss"] for e in events if e["event"] == "val"), f"{cell}: NaN val loss"

    cfg = json.load(open(os.path.join(out, "config.json")))
    recorded = {"train_fraction": cfg["train_fraction"], "objective": cfg["objective"],
                "context_frames": cfg["context_frames"], "noise_aug_max": cfg["noise_aug_max"],
                "lr": cfg["lr"], "warm_start": cfg["warm_start"], "action_inject": cfg["action_inject"]}
    expected = dict(zip(CELL_FLAGS[cell][::2], CELL_FLAGS[cell][1::2]))
    for flag, value in expected.items():
        key = flag.lstrip("-").replace("-", "_")
        got = recorded[key]
        same = float(got) == float(value) if isinstance(got, (int, float)) else str(got) == value
        assert same, f"{cell}: config.json records {key}={got}, launched with {value}"

    episodes = json.load(open(os.path.join(out, "train_episodes.json")))
    assert episodes["num_episodes"] == max(1, round(cfg["train_fraction"] * 8)), cell
    assert not set(episodes["episodes"]) & {8, 9}, f"{cell}: a validation episode leaked into training"

    objective, trained = _evaluate_two_windows(out, tiny_corpus, cell)
    assert objective == ("eps" if cell == "eps" else "v"), cell
    assert trained["action_inject"] == ("adaln" if cell == "adaln" else "token"), cell
