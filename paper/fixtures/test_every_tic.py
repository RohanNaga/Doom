"""`encode_parquet.py --every-tic` keeps every tic and still marks the decisions.

The open GameNGen reproduction stores consecutive tics and repeats each action for four of them, so a
stride-1 experiment needs per-tic latents of the corpora we already train and evaluate on. The one
property that makes such a corpus safe to build is that it must not be a *different* corpus: taking
the rows where `is_decision` is true has to give back exactly the stride-4 latents
`--align-decisions` produces today, with the same `chain_id`s. Otherwise a stride-1 result and a
stride-4 result would not be comparable and the experiment would be worthless.

These tests assert that reduction property on synthetic trajectories, plus that the default path is
untouched and that the flag refuses the cases where per-tic output is impossible or ambiguous.
The helpers and the deterministic stub encoder come from `test_decision_only.py`.

    python -m pytest paper/fixtures/test_every_tic.py -q
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

pytest.importorskip("pyarrow")
pytest.importorskip("torch")

from test_decision_only import (ATTACK, CLEAN, FORWARD, SKIP, STUCK, StubVAE,  # noqa: E402
                                canonical_for, trajectory, write_parquet)

import encode_parquet  # noqa: E402
from transitions import decision_rows  # noqa: E402

# a trajectory with an anti-stuck override and a death, so the decision grid is genuinely irregular
MESSY = ([(0, FORWARD, SKIP, None)] * 3 + [(0, STUCK, 40, None)] + [(1, ATTACK, SKIP, None)] * 3
         + [(0, FORWARD, SKIP, 2)] + [(1, ATTACK, SKIP, None)] * 4)


def run(tmp_path, rows, name, canon, *, every_tic=False, align=False, stored=1):
    """Encode one synthetic episode on the CPU with the stub encoder."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    out = str(tmp_path / name)
    os.makedirs(out, exist_ok=True)
    write_parquet(os.path.join(out, "ep_00000.parquet"), rows, stored)
    encode_parquet.CANONICAL = canon
    with ThreadPoolExecutor(2) as pool:
        r = encode_parquet.encode_episode(os.path.join(out, "ep_00000.parquet"), out, StubVAE(), "cpu",
                                          torch.float32, SKIP, 8, pool, align_decisions=align,
                                          every_tic=every_tic)
    return (r, np.load(os.path.join(out, "ep_00000_latents.npy")),
            np.load(os.path.join(out, "ep_00000_meta.npz")))


@pytest.mark.parametrize("decisions,label", [(CLEAN, "clean"), (MESSY, "messy")])
def test_is_decision_rows_reproduce_the_aligned_latents_exactly(tmp_path, decisions, label):
    """The property the whole corpus rests on."""
    per, _, _ = trajectory(decisions)
    canon = canonical_for(decisions)
    _, lat_t, meta_t = run(tmp_path, per, f"pertic_{label}", canon, every_tic=True)
    _, lat_a, meta_a = run(tmp_path, per, f"aligned_{label}", canon, align=True)
    sel = meta_t["is_decision"].astype(bool)
    assert sel.sum() == lat_a.shape[0] > 0
    assert np.array_equal(lat_t[sel], lat_a)
    assert np.array_equal(meta_t["chain_id"][sel], meta_a["chain_id"])
    for k in ("action", "buttons", "tic", "health", "map_id", "episode_id"):
        assert np.array_equal(meta_t[k][sel], meta_a[k]), k


def test_every_tic_keeps_every_row_in_order(tmp_path):
    per, _, _ = trajectory(CLEAN)
    r, lat, meta = run(tmp_path, per, "all", canonical_for(CLEAN), every_tic=True)
    assert r["frames"] == len(per) == lat.shape[0]
    assert np.array_equal(meta["tic"], np.arange(len(per)))
    assert lat.shape[1:] == (4, 32, 40) and lat.dtype == np.float16


def test_chain_id_is_minus_one_off_the_decision_rows(tmp_path):
    """A non-decision tic belongs to no transition chain, and must not claim chain 0."""
    per, _, _ = trajectory(CLEAN)
    _, _, meta = run(tmp_path, per, "chain", canonical_for(CLEAN), every_tic=True)
    sel = meta["is_decision"].astype(bool)
    assert set(np.unique(meta["chain_id"][~sel]).tolist()) <= {-1}
    assert (meta["chain_id"][sel] >= 0).all()


def test_is_decision_matches_transitions_directly(tmp_path):
    """Cross-check against `transitions.decision_rows`, not just against the other encoder path."""
    per, _, _ = trajectory(MESSY)
    canon = canonical_for(MESSY)
    _, _, meta = run(tmp_path, per, "xcheck", canon, every_tic=True)
    import pyarrow.parquet as pq
    t = pq.read_table(str(tmp_path / "xcheck" / "ep_00000.parquet"))
    keep, chain = decision_rows(t["action"].to_numpy(zero_copy_only=False),
                                np.array(t["buttons"].to_pylist()),
                                t["deaths"].to_numpy(zero_copy_only=False), SKIP, canon, 1)
    assert np.array_equal(np.flatnonzero(meta["is_decision"]), keep)
    assert np.array_equal(meta["chain_id"][keep], chain)


def test_decision_spacing_inside_a_chain_is_the_stride(tmp_path):
    _, _, meta = run(tmp_path, trajectory(CLEAN)[0], "spacing", canonical_for(CLEAN), every_tic=True)
    k = np.flatnonzero(meta["is_decision"])
    c = meta["chain_id"][k]
    same = c[1:] == c[:-1]
    assert same.any() and np.all(np.diff(meta["tic"][k])[same] == SKIP)


def test_default_behaviour_is_unchanged(tmp_path):
    """Without the flag, nothing about the existing output moves."""
    per, _, _ = trajectory(CLEAN)
    _, lat, meta = run(tmp_path, per, "plain", canonical_for(CLEAN))
    assert lat.shape[0] == len(np.flatnonzero(np.arange(len(per)) % SKIP == 0))
    assert "is_decision" not in meta.files and "chain_id" not in meta.files


def test_every_tic_refuses_a_decision_only_recording(tmp_path):
    """Those tics were never rendered, so per-tic output cannot be reconstructed from the file."""
    _, dec, _ = trajectory(CLEAN)
    with pytest.raises(ValueError, match="stored_tic_stride"):
        run(tmp_path, dec, "decisiononly", canonical_for(CLEAN), every_tic=True, stored=SKIP)


def test_every_tic_and_align_decisions_together_are_refused(tmp_path):
    """One keeps every row and the other selects a subset; asking for both is ambiguous."""
    per, _, _ = trajectory(CLEAN)
    with pytest.raises(ValueError, match="align-decisions"):
        run(tmp_path, per, "both", canonical_for(CLEAN), every_tic=True, align=True)


def test_parser_exposes_the_flag_and_defaults_it_off():
    a = encode_parquet.build_parser().parse_args(["--in-dir", "x", "--out-dir", "y"])
    assert a.every_tic is False
    b = encode_parquet.build_parser().parse_args(["--in-dir", "x", "--out-dir", "y", "--every-tic"])
    assert b.every_tic is True


def test_verify_corpus_reduce_decisions_accepts_a_matching_pair(tmp_path):
    """`verify_corpus.py --reduce-decisions` is how the equivalence is checked on the real corpus."""
    import verify_corpus
    per, _, _ = trajectory(MESSY)
    canon = canonical_for(MESSY)
    run(tmp_path, per, "pt", canon, every_tic=True)
    run(tmp_path, per, "al", canon, align=True)
    r = verify_corpus.verify(str(tmp_path / "pt"), str(tmp_path / "al"), 4, 4, reduce_decisions=True)
    assert r["ok"], r["problems"]
    assert r["latent_diff"]["max_abs"] == 0.0
    assert r["frames"] == r["ref_frames"] > 0


def test_verify_corpus_reduce_decisions_catches_a_corrupted_latent(tmp_path):
    import verify_corpus
    per, _, _ = trajectory(CLEAN)
    canon = canonical_for(CLEAN)
    run(tmp_path, per, "pt2", canon, every_tic=True)
    run(tmp_path, per, "al2", canon, align=True)
    p = str(tmp_path / "al2" / "ep_00000_latents.npy")
    a = np.load(p)
    a[0, 0, 0, 0] = np.float16(a[0, 0, 0, 0] + 1.0)
    np.save(p, a)
    r = verify_corpus.verify(str(tmp_path / "pt2"), str(tmp_path / "al2"), 4, 4, reduce_decisions=True)
    assert not r["ok"] and any("latent" in x for x in r["problems"])
    assert r["latent_diff"]["max_abs"] >= 1.0


def test_verify_corpus_reduce_decisions_needs_the_is_decision_column(tmp_path):
    import verify_corpus
    per, _, _ = trajectory(CLEAN)
    canon = canonical_for(CLEAN)
    run(tmp_path, per, "plain2", canon)          # stride-4 output, no is_decision
    run(tmp_path, per, "al3", canon, align=True)
    r = verify_corpus.verify(str(tmp_path / "plain2"), str(tmp_path / "al3"), 4, 4, reduce_decisions=True)
    assert not r["ok"] and any("is_decision" in x for x in r["problems"])


def test_verify_corpus_default_mode_is_unchanged(tmp_path):
    """Without the flag the tool compares the rows as stored, as it always has."""
    import verify_corpus
    per, _, _ = trajectory(CLEAN)
    canon = canonical_for(CLEAN)
    run(tmp_path, per, "a1", canon, align=True)
    run(tmp_path, per, "a2", canon, align=True)
    r = verify_corpus.verify(str(tmp_path / "a1"), str(tmp_path / "a2"), 4, 4)
    assert r["ok"], r["problems"]


def test_meta_records_the_mode(tmp_path):
    """A consumer must be able to tell a per-tic latent directory from a stride-4 one."""
    import json
    per, _, _ = trajectory(CLEAN)
    out = str(tmp_path / "meta")
    os.makedirs(out, exist_ok=True)
    a = encode_parquet.build_parser().parse_args(["--in-dir", out, "--out-dir", out, "--every-tic"])
    encode_parquet.write_meta(a, {"latent_channels": 4, "scaling_factor": 0.18215, "shift_factor": None},
                              0.18215, None, None, 1)
    m = json.load(open(os.path.join(out, "encode_meta_00.json")))
    assert m["every_tic"] is True and m["stride"] == 4
