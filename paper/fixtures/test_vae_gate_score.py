"""The paired episode-level bootstrap behind the VAE ceiling gate's go/no-go.

The gate decides a corpus re-encode on "+1 dB paired with the CI excluding zero", so the
interval has to be right. Windows inside one episode share context and scene, which makes
frames anything but independent: resampling frames would shrink the interval by an order of
magnitude and turn noise into a decision. These checks pin the effect recovery, the null, the
resampling unit and determinism.

    python -m pytest paper/fixtures/test_vae_gate_score.py -q
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from vae_gate_score import paired_bootstrap, summarise  # noqa: E402

EPISODES, PER_EPISODE = 20, 100


def _paired_sample(delta, seed=7):
    """(episode ids, candidate, incumbent) with a per-episode level shift and a known delta."""
    eps = np.repeat(np.arange(EPISODES), PER_EPISODE)
    rng = np.random.RandomState(seed)
    incumbent = rng.normal(28.0, 1.5, eps.size) + rng.normal(0, 1.0, EPISODES)[eps]
    candidate = incumbent + delta + rng.normal(0, 0.2, eps.size)
    return eps, candidate, incumbent


def test_recovers_a_known_paired_gain():
    eps, cand, base = _paired_sample(1.5)
    r = paired_bootstrap(eps, cand, base, 1000, 0)
    assert abs(r["delta"] - 1.5) < 0.1
    assert r["excludes_zero"] and r["ci95"][0] > 0
    assert r["episodes"] == EPISODES


def test_null_interval_straddles_zero():
    eps, _, base = _paired_sample(0.0)
    noise = np.random.RandomState(1).normal(0, 0.5, base.size)
    r = paired_bootstrap(eps, base, base + noise, 1000, 0)
    assert not r["excludes_zero"]
    assert r["ci95"][0] < 0 < r["ci95"][1]


def test_one_episode_gives_a_degenerate_interval():
    """With a single episode there is nothing to resample, so the interval must collapse.

    This is the resampling unit stated exactly: a frame-level bootstrap over the same frames
    would still spread, because the per-frame deltas vary.
    """
    eps = np.zeros(200, dtype=int)
    d = np.random.RandomState(11).normal(1.5, 0.9, 200)      # per-frame deltas genuinely vary
    r = paired_bootstrap(eps, d, np.zeros_like(d), 1000, 0)
    assert r["episodes"] == 1
    assert r["ci95"][0] == r["ci95"][1] == r["delta"]


def test_interval_ignores_how_many_frames_each_episode_contributes():
    """Duplicating every frame inside each episode cannot change an episode-level interval.

    It would shrink a frame-level one by sqrt(2), so this pins the unit without a threshold.
    """
    eps, cand, base = _paired_sample(1.5)
    once = paired_bootstrap(eps, cand, base, 1000, 0)
    order = np.argsort(np.concatenate([eps, eps]), kind="stable")
    twice = paired_bootstrap(np.concatenate([eps, eps])[order],
                             np.concatenate([cand, cand])[order],
                             np.concatenate([base, base])[order], 1000, 0)
    assert twice["episodes"] == once["episodes"] == EPISODES
    assert np.allclose(twice["ci95"], once["ci95"], atol=1e-12)
    assert np.isclose(twice["delta"], once["delta"], atol=1e-12)


def test_is_deterministic_for_a_given_seed():
    eps, cand, base = _paired_sample(1.5)
    assert paired_bootstrap(eps, cand, base, 1000, 0) == paired_bootstrap(eps, cand, base, 1000, 0)
    assert paired_bootstrap(eps, cand, base, 1000, 1) != paired_bootstrap(eps, cand, base, 1000, 0)


def test_summarise_reports_mean_sem_and_count():
    assert summarise(np.array([1.0, 2.0, 3.0])) == {"mean": 2.0, "sem": 0.5773502691896258, "n": 3}
