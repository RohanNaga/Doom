"""The few-step sampler can be evaluated with other timestep spacings; the default stays bit-identical.

The 2026-09-22 review (M1): uniform-in-t spacing over the trained linear betas puts the 4-step
sampler at t = [999, 666, 333, 0], two calls at essentially pure noise, so the step sweep did not
show the best 4-step quality the model can reach. `--timestep-spacing {linear,trailing,karras}` on
`eval_tf.py` and `rollout_eval.py` changes only which trained indices are visited. `linear` is the
default and is `VDiffusion.ddim_sample` itself.

    python -m pytest paper/fixtures/test_timestep_spacing.py -q
"""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import timestep_spacing as tsp  # noqa: E402
from diffusion_v import VDiffusion  # noqa: E402


def toy_model(xt, t):
    """A deterministic stand-in network whose output depends on both the input and the timestep."""
    return 0.3 * xt + 0.001 * t.float().view(-1, 1, 1, 1)


def test_linear_is_the_samplers_own_list():
    assert tsp.timesteps("linear", 4).tolist() == [999, 666, 333, 0]
    for n in (1, 2, 8, 50):
        want = torch.linspace(999, 0, n).round().long()
        assert torch.equal(tsp.timesteps("linear", n), want)


@pytest.mark.parametrize("objective", ["v", "eps"])
@pytest.mark.parametrize("eta", [0.0, 0.5])
@pytest.mark.parametrize("steps", [1, 4, 50])
def test_the_explicit_loop_reproduces_ddim_sample_bit_for_bit(objective, eta, steps):
    d = VDiffusion(objective=objective)
    shape = (2, 4, 8, 10)
    noise = torch.randn(shape, generator=torch.Generator().manual_seed(0))

    def nfn(i):
        return torch.randn(shape, generator=torch.Generator().manual_seed(100 + i))

    ref = d.ddim_sample(toy_model, shape, steps=steps, eta=eta, noise=noise, noise_fn=nfn)
    got = tsp.ddim_sample_at(d, toy_model, shape, tsp.timesteps("linear", steps), eta=eta, noise=noise,
                             noise_fn=nfn)
    assert torch.equal(ref, got)


def test_the_default_spacing_calls_ddim_sample_itself(monkeypatch):
    d = VDiffusion()
    calls = []
    real = d.ddim_sample

    def spy(*a, **kw):
        calls.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(d, "ddim_sample", spy)
    noise = torch.randn(1, 4, 8, 10)
    out = tsp.sample(d, toy_model, noise.shape, steps=4, spacing="linear", noise=noise)
    assert len(calls) == 1 and calls[0]["steps"] == 4
    assert torch.equal(out, real(toy_model, noise.shape, steps=4, noise=noise))


def test_trailing_starts_at_the_last_index_and_never_samples_zero():
    assert tsp.timesteps("trailing", 4).tolist() == [999, 749, 499, 249]
    for n in (1, 3, 8, 50):
        t = tsp.timesteps("trailing", n)
        assert len(t) == n and int(t[0]) == 999 and int(t[-1]) > 0
        assert bool((t[1:] < t[:-1]).all())


def test_karras_is_strictly_decreasing_and_denser_near_clean():
    d = VDiffusion()
    for n in (2, 4, 8, 16, 50):
        t = tsp.timesteps("karras", n, d.num_steps, d.sqrt_abar)
        assert len(t) == n and int(t[0]) == 999 and int(t[-1]) == 0
        assert bool((t[1:] < t[:-1]).all()), t.tolist()
    k4 = tsp.timesteps("karras", 4, d.num_steps, d.sqrt_abar)
    assert sorted(k4.tolist())[1] < 333, "karras should spend more of four calls at low noise than linear"


def test_other_spacings_sample_through_the_same_update_rule():
    d = VDiffusion()
    noise = torch.randn(1, 4, 8, 10, generator=torch.Generator().manual_seed(1))
    for spacing in ("trailing", "karras"):
        out = tsp.sample(d, toy_model, noise.shape, steps=4, spacing=spacing, noise=noise)
        ts = tsp.timesteps(spacing, 4, d.num_steps, d.sqrt_abar)
        assert torch.equal(out, tsp.ddim_sample_at(d, toy_model, noise.shape, ts, noise=noise))
        assert not torch.equal(out, d.ddim_sample(toy_model, noise.shape, steps=4, noise=noise))


def test_an_unknown_spacing_is_refused():
    with pytest.raises(ValueError, match="unknown timestep spacing"):
        tsp.timesteps("cosine", 4)


def test_both_evaluators_take_the_flag_with_linear_as_default():
    import eval_tf
    import rollout_eval
    a = eval_tf.build_parser().parse_args(["--ckpt", "x", "--backbone", "unet", "--latents-dir", "l",
                                           "--split", "s", "--out-dir", "o"])
    assert a.timestep_spacing == "linear"
    b = eval_tf.build_parser().parse_args(["--ckpt", "x", "--backbone", "unet", "--latents-dir", "l",
                                           "--split", "s", "--out-dir", "o", "--timestep-spacing", "karras"])
    assert b.timestep_spacing == "karras"
    assert rollout_eval.build_parser().parse_args([]).timestep_spacing == "linear"
    with pytest.raises(SystemExit):
        eval_tf.build_parser().parse_args(["--ckpt", "x", "--backbone", "unet", "--latents-dir", "l",
                                           "--split", "s", "--out-dir", "o", "--timestep-spacing", "cosine"])


def test_both_evaluators_sample_through_the_spacing_switch():
    for name in ("eval_tf.py", "rollout_eval.py"):
        src = open(os.path.join(REPO, name)).read()
        assert "spacing=args.timestep_spacing" in src, name
        assert "diffusion.ddim_sample(" not in src, f"{name} bypasses the spacing switch"
