"""Synthetic per-tic latent episodes, in exactly the layout `encode_parquet.py --every-tic` writes.

Shared by every next-tic test. Each frame's latent is filled with its own row index, so an
assertion can read a tensor and say which tic it came from; that is what makes the
"which action conditions which target" and "oldest frame first" rules checkable rather than
merely asserted.
"""
import os

import numpy as np

CHANNELS = 4
LATENT_HW = (32, 40)
STRIDE = 4          # the recorded action repeat: the agent decides every 4th tic


def write_pertic_episode(out_dir, ep, actions, *, deaths=None, tics=None, map_ids=None,
                         decisions=None, channels=CHANNELS, buttons=None):
    """One `ep_XXXXX_latents.npy` / `ep_XXXXX_meta.npz` pair. Latent of row t is filled with t.

    `decisions` defaults to the clean grid (every row whose tic is divisible by 4), which is what a
    recording with no interrupted decisions produces.
    """
    os.makedirs(out_dir, exist_ok=True)
    actions = np.asarray(actions, dtype=np.int64)
    T = len(actions)
    tics = np.arange(T, dtype=np.int64) if tics is None else np.asarray(tics, dtype=np.int64)
    deaths = np.zeros(T, dtype=np.int64) if deaths is None else np.asarray(deaths, dtype=np.int64)
    map_ids = np.full(T, 7, dtype=np.int64) if map_ids is None else np.asarray(map_ids, dtype=np.int64)
    if decisions is None:
        decisions = (tics % STRIDE) == 0
    decisions = np.asarray(decisions, dtype=bool)
    chain = np.full(T, -1, dtype=np.int64)
    chain[decisions] = 0
    lat = np.stack([np.full((channels,) + LATENT_HW, t, dtype=np.float16) for t in range(T)])
    np.save(os.path.join(out_dir, f"ep_{ep:05d}_latents.npy"), lat)
    np.savez(os.path.join(out_dir, f"ep_{ep:05d}_meta.npz"),
             action=actions, buttons=np.array(["100000000"] * T) if buttons is None else np.asarray(buttons),
             tic=tics, deaths=deaths, map_id=map_ids,
             episode_id=np.full(T, ep, dtype=np.int64), is_decision=decisions, chain_id=chain)
    return lat


def write_stride4_episode(out_dir, ep, actions, *, channels=CHANNELS, map_id=7):
    """The stride-4 layout, for the tests that check the old path is untouched: one row per decision."""
    os.makedirs(out_dir, exist_ok=True)
    actions = np.asarray(actions, dtype=np.int64)
    T = len(actions)
    lat = np.stack([np.full((channels,) + LATENT_HW, t, dtype=np.float16) for t in range(T)])
    np.save(os.path.join(out_dir, f"ep_{ep:05d}_latents.npy"), lat)
    np.savez(os.path.join(out_dir, f"ep_{ep:05d}_meta.npz"),
             action=actions, tic=np.arange(T, dtype=np.int64) * STRIDE,
             map_id=np.full(T, map_id, dtype=np.int64),
             episode_id=np.full(T, ep, dtype=np.int64),
             chain_id=np.zeros(T, dtype=np.int64))
    return lat


def held_actions(ids, repeat=STRIDE):
    """One action id per tic from a list of per-decision ids: each held for `repeat` tics."""
    return np.repeat(np.asarray(ids, dtype=np.int64), repeat)


def frame_index_of(tensor):
    """Which row a context slice or a target came from, read back out of its constant fill."""
    v = np.unique(np.asarray(tensor, dtype=np.float32))
    assert len(v) == 1, f"not a single-frame slice: {v[:5]}"
    return int(v[0])
