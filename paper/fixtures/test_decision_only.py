"""`record_arnold.py --decision-only` must be the same corpus, sampled once per decision.

Nothing here needs ViZDoom. A decision list is expanded into the rows each of the recorder's two
branches would write from that same trajectory, and the tests assert the part that has to hold on
any machine: the alignment in `encode_parquet.py --align-decisions` keeps the same decisions, chains
them the same way, and writes the same latents, whether the file stores every tic or only the tics
the agent decided on. Divergence of the engine itself between the two stepping modes is a separate,
measured question; it cannot be settled here and is recorded in `release/DENSE_CORPUS.md`.

    python -m pytest paper/fixtures/test_decision_only.py -q
"""
import io
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from transitions import (EPISODE_META_KEY, canonical_table, decision_rows,  # noqa: E402
                         stored_tic_stride, valid_transitions)

SKIP = 4
FORWARD = "100000000"    # canonical control bits of action 0
ATTACK = "000000100"     # canonical control bits of action 1
STUCK = "100100010"      # the anti-stuck manual override: never canonical for any action


def trajectory(decisions):
    """Rows both recorder branches would write from one list of decisions.

    A decision is `(action, buttons, n_tics, died_after)`: `died_after` is the number of tics stepped
    before the death was seen, or None. Per tic the recorder stores a row before each single-tic step
    and stops at the death, so a cut-short decision leaves `died_after` rows; decision-only stores one
    row and hands the whole skip to the engine, so its tic counter advances by `n_tics` regardless.
    Returns (per_tic_rows, decision_rows, decision_index_of_per_tic_row); a row is
    (tic, action, buttons, deaths).
    """
    per, dec, owner = [], [], []
    tic_per = tic_dec = deaths = 0
    for d, (action, buttons, n_tics, died_after) in enumerate(decisions):
        dec.append((tic_dec, action, buttons, deaths))
        stepped = n_tics if died_after is None else died_after
        for i in range(stepped):
            per.append((tic_per + i, action, buttons, deaths))
            owner.append(d)
        tic_per += stepped
        tic_dec += n_tics
        if died_after is not None:
            deaths += 1
    return per, dec, np.array(owner)


def frame_png(tic):
    """A frame whose bytes are a function of the tic, so two recordings of the same tic must match."""
    from PIL import Image
    a = np.zeros((240, 320, 3), np.uint8)
    a[:, :, 0] = tic % 256
    a[:, :, 1] = (tic // 256) % 256
    a[tic % 240, :, 2] = 255
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def write_parquet(path, rows, stored):
    """One episode file in exactly the schema and metadata `record_arnold.py` writes."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = len(rows)
    tic, action, buttons, deaths = (list(c) for c in zip(*rows))
    t = pa.table({
        "episode_id": pa.array([0] * n, pa.int32()), "map_id": pa.array([3] * n, pa.int8()),
        "tic": pa.array(tic, pa.int32()), "action": pa.array(action, pa.int16()),
        "buttons": pa.array(buttons, pa.string()), "health": pa.array([100] * n, pa.int16()),
        "ammo": pa.array([50] * n, pa.int16()), "kills": pa.array([0] * n, pa.int16()),
        "deaths": pa.array(deaths, pa.int16()), "frags": pa.array([0] * n, pa.int16()),
        "pos_x": pa.array([1.0] * n, pa.float32()), "pos_y": pa.array([2.0] * n, pa.float32()),
        "angle": pa.array([3.0] * n, pa.float32()),
        "frame": pa.array([frame_png(v) for v in tic], pa.binary()),
    })
    prov = {"seed_scheme": "doomdit-episode-v1", "corpus_id": "test", "episode_id": 0, "map_id": 3}
    if stored > 1:
        prov["stored_tic_stride"] = stored
    t = t.replace_schema_metadata({EPISODE_META_KEY: json.dumps(prov, sort_keys=True).encode()})
    pq.write_table(t, path, compression=None, row_group_size=256)
    return t


def columns(rows):
    tic, action, buttons, deaths = (np.array(c) for c in zip(*rows))
    return action, buttons, deaths


def aligned(rows, stored, canonical):
    """(kept rows, chain ids, transition sources) the encoder would take from this recording."""
    action, buttons, deaths = columns(rows)
    keep, chain = decision_rows(action, buttons, deaths, SKIP, canonical, stored)
    src, _ = valid_transitions(action, buttons, deaths, SKIP // stored, canonical)
    return keep, chain, src


def canonical_for(decisions):
    """The canonical table the encoder builds; taken from the per-tic rows, as the real corpus is."""
    per, _, _ = trajectory(decisions)
    action, buttons, _ = columns(per)
    return canonical_table(action, buttons)


CLEAN = [(0, FORWARD, SKIP, None)] * 6 + [(1, ATTACK, SKIP, None)] * 6


def test_stored_stride_defaults_to_one_and_round_trips(tmp_path):
    per_tic = write_parquet(str(tmp_path / "pertic.parquet"), trajectory(CLEAN)[0], stored=1)
    dec = write_parquet(str(tmp_path / "dec.parquet"), trajectory(CLEAN)[1], stored=SKIP)
    assert stored_tic_stride(per_tic.schema.metadata) == 1      # absent key means a per-tic recording
    assert stored_tic_stride(dec.schema.metadata) == SKIP
    assert stored_tic_stride(None) == 1
    assert stored_tic_stride({b"other": b"{}"}) == 1


def test_decision_only_keeps_the_same_decisions_and_chains():
    per, dec, owner = trajectory(CLEAN)
    canon = canonical_for(CLEAN)
    keep_p, chain_p, src_p = aligned(per, 1, canon)
    keep_d, chain_d, src_d = aligned(dec, SKIP, canon)
    assert len(keep_d) == len(keep_p) > 0
    assert np.array_equal(owner[keep_p], keep_d)                 # the same decisions survive
    assert np.array_equal(owner[src_p], src_d)                   # as sources of the same transitions
    assert np.array_equal(chain_p, chain_d)                      # chained the same way
    assert np.array_equal(np.array([per[i][0] for i in keep_p]), np.array([dec[i][0] for i in keep_d]))


def test_death_inside_a_skip_cuts_the_chain_in_both_modes():
    decisions = [(0, FORWARD, SKIP, None)] * 4 + [(0, FORWARD, SKIP, 2)] + [(0, FORWARD, SKIP, None)] * 4
    per, dec, owner = trajectory(decisions)
    canon = canonical_for(decisions)
    keep_p, chain_p, src_p = aligned(per, 1, canon)
    keep_d, chain_d, src_d = aligned(dec, SKIP, canon)
    assert np.array_equal(owner[keep_p], keep_d)
    assert np.array_equal(owner[src_p], src_d)
    assert 4 not in src_d                                        # the decision the death cut short
    assert 4 in keep_d                                           # but its frame is still a chain target
    assert np.array_equal(chain_p, chain_d)
    assert len(set(chain_d.tolist())) == 2                       # one chain per life, not one across
    assert len(keep_d) - len(set(chain_d.tolist())) == len(src_d)   # frames - chains == transitions
    # the death is the only thing that splits them: without it the same decisions form a single chain
    chain_clean = aligned(trajectory([(0, FORWARD, SKIP, None)] * 9)[1], SKIP, canon)[1]
    assert len(set(chain_clean.tolist())) == 1


def test_anti_stuck_override_is_excluded_in_both_modes():
    decisions = [(0, FORWARD, SKIP, None)] * 30 + [(0, STUCK, 40, None)] + [(0, FORWARD, SKIP, None)] * 30
    per, dec, owner = trajectory(decisions)
    canon = canonical_for(decisions)
    keep_p, chain_p, src_p = aligned(per, 1, canon)
    keep_d, chain_d, src_d = aligned(dec, SKIP, canon)
    assert canon[0] == FORWARD
    assert np.array_equal(owner[keep_p], keep_d)
    assert np.array_equal(owner[src_p], src_d)
    assert np.array_equal(chain_p, chain_d)
    assert 30 not in src_d                                       # the override decision itself
    assert len(set(chain_d.tolist())) == 2                       # and it breaks the chain in two


def test_override_tics_are_weighted_differently_by_the_two_canonical_tables():
    """The one place the two layouts genuinely disagree, and the reason decision-only is the safer vote.

    `canonical_table` takes the modal control bits per action over every stored row. A 40-tic anti-stuck
    override is one decision but forty per-tic rows, so per tic it votes forty times and decision-only
    votes once. With overrides a minority of decisions both tables agree; engineered to a minority of
    decisions but a majority of tics, only the per-tic table is captured by the override.
    """
    decisions = [(0, FORWARD, SKIP, None)] * 8 + [(0, STUCK, 40, None)]
    per, dec, _ = trajectory(decisions)
    assert canonical_table(*columns(per)[:2])[0] == STUCK        # 40 override tics beat 32 forward tics
    assert canonical_table(*columns(dec)[:2])[0] == FORWARD      # 1 override decision loses to 8
    many = [(0, FORWARD, SKIP, None)] * 30 + [(0, STUCK, 40, None)]
    assert canonical_table(*columns(trajectory(many)[0])[:2])[0] == FORWARD
    assert canonical_table(*columns(trajectory(many)[1])[:2])[0] == FORWARD


def test_stride_must_be_a_multiple_of_the_stored_stride():
    per, _, _ = trajectory(CLEAN)
    action, buttons, deaths = columns(per)
    with pytest.raises(ValueError):
        decision_rows(action, buttons, deaths, stride=6, stored=SKIP)


def test_map_id_offset_keeps_two_wads_apart_and_refuses_to_overflow():
    """deathmatch_simple's MAP01 and full_deathmatch's arena 1 are both map 1 to the engine."""
    record_arnold = pytest.importorskip("record_arnold")
    assert record_arnold.stored_map_id(1, 0) == 1            # the arena keeps its own label
    assert record_arnold.stored_map_id(1, 100) == 101        # the second WAD's MAP01 does not collide
    assert record_arnold.stored_map_id(8, 0) == 8
    for bad in ((1, 127), (1, -1), (0, 0), (100, 100)):
        with pytest.raises(ValueError):                      # the stored column is int8
            record_arnold.stored_map_id(*bad)


class StubVAE:
    """Deterministic stand-in for the AutoencoderKL: an 8x average pool, so a latent still depends on
    its frame and two encodings of the same frame can be compared."""

    def encode(self, x):
        import torch

        class Dist:
            pass

        d, out = Dist(), torch.nn.functional.avg_pool2d(x, 8)[:, :1].repeat(1, 4, 1, 1)
        d.mean = out
        return type("Enc", (), {"latent_dist": d})()


def encode(tmp_path, rows, stored, canon, name):
    """Run `encode_parquet.encode_episode` on CPU with the stub encoder and return (latents, meta)."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    import encode_parquet

    src = str(tmp_path / f"{name}.parquet")
    write_parquet(src, rows, stored)
    out = str(tmp_path / name)
    os.makedirs(out, exist_ok=True)
    os.rename(src, os.path.join(out, "ep_00000.parquet"))
    encode_parquet.CANONICAL = canon
    with ThreadPoolExecutor(2) as pool:
        r = encode_parquet.encode_episode(os.path.join(out, "ep_00000.parquet"), out, StubVAE(), "cpu",
                                          torch.float32, SKIP, 8, pool, align_decisions=True)
    return r, np.load(os.path.join(out, "ep_00000_latents.npy")), np.load(os.path.join(out, "ep_00000_meta.npz"))


def test_encoder_produces_the_same_latents_from_either_layout(tmp_path):
    per, dec, _ = trajectory(CLEAN)
    canon = canonical_for(CLEAN)
    r_p, lat_p, meta_p = encode(tmp_path, per, 1, canon, "pertic")
    r_d, lat_d, meta_d = encode(tmp_path, dec, SKIP, canon, "decision")
    assert r_p["frames"] == r_d["frames"] > 0
    assert lat_p.shape == lat_d.shape and lat_p.shape[1:] == (4, 32, 40)
    assert np.array_equal(lat_p, lat_d)
    for k in ("action", "buttons", "tic", "chain_id", "map_id", "episode_id", "health"):
        assert np.array_equal(meta_p[k], meta_d[k]), k
    assert np.all(np.diff(meta_d["tic"])[meta_d["chain_id"][1:] == meta_d["chain_id"][:-1]] == SKIP)


def test_encode_wan_refuses_a_decision_only_recording(tmp_path):
    """The video layouts need the tics between decisions, which this recording never rendered."""
    encode_wan = pytest.importorskip("encode_wan")
    _, dec, _ = trajectory(CLEAN)
    path = str(tmp_path / "ep_00000.parquet")
    write_parquet(path, dec, stored=SKIP)
    with pytest.raises(ValueError, match="stored_tic_stride"):
        encode_wan.encode_episode(path, str(tmp_path), None, None, None, 1)


def test_unaligned_encoding_keeps_every_decision_row(tmp_path):
    """Without --align-decisions, `tic % stride == 0` is wrong for a file that already holds decisions."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    import encode_parquet

    _, dec, _ = trajectory(CLEAN)
    out = str(tmp_path / "plain")
    os.makedirs(out, exist_ok=True)
    write_parquet(os.path.join(out, "ep_00000.parquet"), dec, stored=SKIP)
    with ThreadPoolExecutor(2) as pool:
        r = encode_parquet.encode_episode(os.path.join(out, "ep_00000.parquet"), out, StubVAE(), "cpu",
                                          torch.float32, SKIP, 8, pool, align_decisions=False)
    assert r["frames"] == len(dec)
