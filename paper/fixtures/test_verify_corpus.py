"""`verify_corpus.py` has to fail on the mismatches a re-encode can actually produce.

A structural check is only worth running if it goes red for the right reasons, so each test here
breaks one property of a mirrored corpus and asserts the report names it: a dropped episode, a
short episode, an edited metadata column, a wrong channel count, a wrong dtype. The counting
tests pin the frame/chain/transition identity `frames - chains == transitions`.

    python -m pytest paper/fixtures/test_verify_corpus.py -q
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from verify_corpus import chain_count, verify  # noqa: E402


def write_corpus(d, channels, episodes=((0, [0, 0, 0, 1, 1]), (1, [0, 0, 0]))):
    os.makedirs(d, exist_ok=True)
    for ep, chains in episodes:
        t = len(chains)
        np.save(os.path.join(d, f"ep_{ep:05d}_latents.npy"),
                np.zeros((t, channels, 32, 40), dtype=np.float16))
        np.savez(os.path.join(d, f"ep_{ep:05d}_meta.npz"),
                 action=np.arange(t), tic=np.arange(t) * 4, chain_id=np.array(chains),
                 map_id=np.full(t, 7), episode_id=np.full(t, ep))
    return d


def pair(tmp_path, **kw):
    new = write_corpus(str(tmp_path / "new"), 16, **kw)
    ref = write_corpus(str(tmp_path / "ref"), 4)
    return new, ref


def test_mirrored_corpus_passes_and_counts(tmp_path):
    new, ref = pair(tmp_path)
    r = verify(new, ref, 16, 4)
    assert r["ok"], r["problems"]
    assert r["episodes"] == 2 and r["frames"] == 8 and r["chains"] == 3
    assert r["transitions"] == r["frames"] - r["chains"] == 5
    assert r["maps"] == {"7": 2}


def test_chain_count_counts_boundaries_not_unique_values():
    assert chain_count(np.array([0, 0, 1, 1, 0, 0])) == 3      # an id reused later is a new chain
    assert chain_count(np.array([])) == 0
    assert chain_count(np.array([4])) == 1


def test_missing_episode_is_reported(tmp_path):
    new, ref = pair(tmp_path)
    os.remove(os.path.join(new, "ep_00001_latents.npy"))
    r = verify(new, ref, 16, 4)
    assert not r["ok"] and any("missing" in p for p in r["problems"])


def test_short_episode_is_reported(tmp_path):
    new, ref = pair(tmp_path)
    np.save(os.path.join(new, "ep_00000_latents.npy"), np.zeros((4, 16, 32, 40), dtype=np.float16))
    r = verify(new, ref, 16, 4)
    assert not r["ok"] and any("frames vs" in p for p in r["problems"])


def test_edited_metadata_column_is_reported(tmp_path):
    new, ref = pair(tmp_path)
    np.savez(os.path.join(new, "ep_00000_meta.npz"), action=np.zeros(5), tic=np.arange(5) * 4,
             chain_id=np.array([0, 0, 0, 1, 1]), map_id=np.full(5, 7), episode_id=np.zeros(5))
    r = verify(new, ref, 16, 4)
    assert not r["ok"] and any("column action differs" in p for p in r["problems"])


def test_wrong_channel_count_is_reported(tmp_path):
    new, ref = pair(tmp_path)
    r = verify(new, ref, 4, 4)       # ask for 4 channels from the 16-channel corpus
    assert not r["ok"] and any("latent shape" in p for p in r["problems"])


def test_wrong_dtype_is_reported(tmp_path):
    new, ref = pair(tmp_path)
    np.save(os.path.join(new, "ep_00000_latents.npy"), np.zeros((5, 16, 32, 40), dtype=np.float32))
    r = verify(new, ref, 16, 4)
    assert not r["ok"] and any("dtype" in p for p in r["problems"])


def test_transition_stats_mismatch_is_reported(tmp_path):
    new, ref = pair(tmp_path)
    stats = str(tmp_path / "transition_stats.json")
    with open(stats, "w") as f:
        f.write('{"transitions": 99, "chains": 3}')
    r = verify(new, ref, 16, 4, transition_stats=stats)
    assert not r["ok"] and any("99" in p for p in r["problems"])
    with open(stats, "w") as f:
        f.write('{"transitions": 5, "chains": 3}')
    assert verify(new, ref, 16, 4, transition_stats=stats)["ok"]
