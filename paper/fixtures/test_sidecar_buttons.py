"""The encoder writes the EXECUTED 19-button vector, and repairs the sidecars that hold raw requests.

Writing `np.array(t["buttons"].to_pylist())` made a `<U2506` column out of Arnold's widest request
string, so every `_meta.npz` was 38 to 49 MB instead of about 200 KB: 76 GB of sidecar over the
2,000-episode corpus, for 19 characters of control per row. The fix normalises the column at write
time and `--normalize-sidecars` shrinks the already-encoded episodes without touching a latent.

The raw request is not discarded: `buttons_raw_len` and `switch_requested_index` keep, per row, how
long Arnold's string was and where its weapon-select press sat, so an unexecuted switch stays
recoverable from the sidecar alone.

    python -m pytest paper/fixtures/test_sidecar_buttons.py -q
"""
import json
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

from test_decision_only import ATTACK, FORWARD, SKIP, StubVAE, canonical_for, trajectory, write_parquet  # noqa: E402

import encode_parquet  # noqa: E402
from transitions import EXECUTED_BUTTONS, normalize_buttons  # noqa: E402

SWITCH_IN = FORWARD + "0000" + "1"                     # 14 chars: a switch the engine executed
SWITCH_OUT = FORWARD + "0" * 2493 + "1"                # 2503 chars: a switch it never saw


def _episode(widths):
    """One life of per-tic rows, each carrying the button string given for it."""
    return [(t, 0 if b.startswith("1") else 1, b, 0) for t, b in enumerate(widths)]


def _encode(tmp_path, rows, name, *, every_tic=True):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    out = str(tmp_path / name)
    os.makedirs(out, exist_ok=True)
    write_parquet(os.path.join(out, "ep_00000.parquet"), rows, 1)
    encode_parquet.CANONICAL = {0: FORWARD, 1: ATTACK}
    with ThreadPoolExecutor(2) as pool:
        r = encode_parquet.encode_episode(os.path.join(out, "ep_00000.parquet"), out, StubVAE(), "cpu",
                                          torch.float32, SKIP, 8, pool, every_tic=every_tic)
    return out, r, np.load(os.path.join(out, "ep_00000_meta.npz"))


MIXED = [FORWARD] * 18 + [SWITCH_IN] + [SWITCH_OUT]


def test_the_sidecar_buttons_column_is_the_fixed_executed_width(tmp_path):
    _, _, m = _encode(tmp_path, _episode(MIXED), "mixed")
    assert m["buttons"].dtype == np.dtype(f"<U{EXECUTED_BUTTONS}")
    assert m["buttons"].tolist() == [normalize_buttons(s) for s in MIXED]
    assert m["buttons"][-1] == FORWARD + "0" * 10, "the trailing 1 at index 2502 was never executed"
    assert m["buttons"][-2][13] == "1", "a switch inside the engine's 19 buttons survives"


def test_the_sidecar_is_small_again(tmp_path):
    """The whole point: 19 characters per row, not 2,506."""
    d, _, _ = _encode(tmp_path, _episode(MIXED), "small")
    size = os.path.getsize(os.path.join(d, "ep_00000_meta.npz"))
    assert size < 8 * 1024, f"{size} bytes of sidecar for 20 rows"
    assert np.load(os.path.join(d, "ep_00000_meta.npz"))["buttons"].nbytes == 20 * EXECUTED_BUTTONS * 4


def test_the_raw_request_stays_recoverable_from_the_sidecar(tmp_path):
    """Rohan's rule: a valid control input is never discarded, only moved somewhere honest."""
    _, _, m = _encode(tmp_path, _episode(MIXED), "raw")
    assert m["buttons_raw_len"].dtype == np.int16
    assert m["buttons_raw_len"].tolist() == [9] * 18 + [14, 2503]
    assert m["switch_requested_index"].dtype == np.int32
    assert m["switch_requested_index"].tolist() == [-1] * 18 + [13, 2502]
    unexecuted = m["switch_requested_index"] >= EXECUTED_BUTTONS
    assert unexecuted.sum() == 1


def test_the_episode_summary_reports_the_raw_widths_and_the_lost_switches(tmp_path):
    _, r, _ = _encode(tmp_path, _episode(MIXED), "summary")
    assert r["buttons_raw_max_width"] == 2503
    assert r["unexecuted_switch_rows"] == 1
    assert r["buttons_rows_over_executed"] == 1


def test_the_stride_four_path_writes_the_same_normalised_column(tmp_path):
    """The executed control is a property of the recording, not of which rows the encoder keeps."""
    rows = _episode([FORWARD] * 8 + [SWITCH_OUT] * 8)
    _, _, m = _encode(tmp_path, rows, "stride4", every_tic=False)
    assert m["buttons"].dtype == np.dtype(f"<U{EXECUTED_BUTTONS}")
    assert set(m["buttons"].tolist()) <= {FORWARD + "0" * 10}


def test_the_encoder_records_the_executed_control_contract(tmp_path):
    import argparse
    out = str(tmp_path / "meta")
    os.makedirs(out)
    a = argparse.Namespace(vae_id=None, vae_subfolder=None, legacy=False, stride=4, out_dir=out,
                           align_decisions=False, every_tic=True, shard=0)
    encode_parquet.write_meta(a, {"latent_channels": 4}, 0.18215, None, None)
    meta = json.load(open(os.path.join(out, "encode_meta_00.json")))
    assert meta["executed_buttons"] == EXECUTED_BUTTONS
    assert "buttons[:19]" in meta["buttons_column"]


# ---------------------------------------------------------------------------------------
# repairing the sidecars that are already on disk
# ---------------------------------------------------------------------------------------

def _wide_sidecar(tmp_path, name, widths, stored=None):
    """An episode encoded the old way: the raw request strings straight into the `.npz`.

    `stored` overrides what lands in the sidecar, for the cases where the column is not something
    the encoder would ever have produced.
    """
    d, _, _ = _encode(tmp_path, _episode(widths), name)
    p = os.path.join(d, "ep_00000_meta.npz")
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files}
    cols["buttons"] = np.array(widths if stored is None else stored)   # <U2503, as the defect wrote it
    for gone in ("buttons_raw_len", "switch_requested_index"):
        cols.pop(gone)
    np.savez(p, **cols)
    return d, p


def _paths(d):
    return [os.path.join(d, "ep_00000.parquet")]


def test_normalize_sidecars_shrinks_an_already_encoded_episode(tmp_path):
    d, p = _wide_sidecar(tmp_path, "repair", MIXED)
    assert np.load(p)["buttons"].dtype.itemsize // 4 == 2503
    before = os.path.getsize(p)
    r = encode_parquet.normalize_sidecars(_paths(d), d)
    assert r == {"episodes": 1, "normalized": 1, "unchanged": 0, "missing": [], "refused": []}
    with np.load(p) as z:
        assert z["buttons"].dtype == np.dtype(f"<U{EXECUTED_BUTTONS}")
        assert z["buttons"].tolist() == [normalize_buttons(s) for s in MIXED]
        assert z["buttons_raw_len"].tolist() == [9] * 18 + [14, 2503]
        assert z["switch_requested_index"].tolist() == [-1] * 18 + [13, 2502]
        assert z["tic"].tolist() == list(range(20)), "every other column is copied through"
        assert z["is_decision"].dtype == bool
    assert os.path.getsize(p) < before


def test_normalize_sidecars_is_idempotent(tmp_path):
    d, p = _wide_sidecar(tmp_path, "twice", MIXED)
    encode_parquet.normalize_sidecars(_paths(d), d)
    again = encode_parquet.normalize_sidecars(_paths(d), d)
    assert again["normalized"] == 0 and again["unchanged"] == 1


def test_normalize_sidecars_writes_nothing_under_dry_run(tmp_path):
    d, p = _wide_sidecar(tmp_path, "dry", MIXED)
    before = open(p, "rb").read()
    r = encode_parquet.normalize_sidecars(_paths(d), d, dry_run=True)
    assert r["normalized"] == 1
    assert open(p, "rb").read() == before


def test_normalize_sidecars_refuses_a_non_binary_column(tmp_path):
    """A '2' is a corrupted column, not a wide one; the file must be left alone for a human to look at."""
    d, p = _wide_sidecar(tmp_path, "bad", [FORWARD] * 20,
                         stored=[FORWARD + "00000"] * 19 + ["12000000000000"])
    r = encode_parquet.normalize_sidecars(_paths(d), d)
    assert r["normalized"] == 0 and len(r["refused"]) == 1
    assert "not binary" in r["refused"][0]
    assert np.load(p)["buttons"].dtype.itemsize // 4 == 14, "the file is left exactly as it was"


# ---------------------------------------------------------------------------------------
# the repair reads the recording, so it can tell a wide sidecar from a WRONG one
# ---------------------------------------------------------------------------------------

def test_normalize_sidecars_refuses_a_row_count_that_does_not_match_the_recording(tmp_path):
    d, p = _wide_sidecar(tmp_path, "short", MIXED)
    with np.load(p) as z:
        cols = {k: z[k][:12] for k in z.files}
    np.savez(p, **cols)
    r = encode_parquet.normalize_sidecars(_paths(d), d)
    assert r["normalized"] == 0 and "12 sidecar rows vs 20 raw rows" in r["refused"][0]


def test_normalize_sidecars_refuses_a_ragged_buttons_column(tmp_path):
    """Two button rows beside one tic row: the old sidecar-only repair happily rewrote this."""
    d, p = _wide_sidecar(tmp_path, "ragged", MIXED)
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files}
    cols["buttons"] = np.array(MIXED + MIXED)
    np.savez(p, **cols)
    r = encode_parquet.normalize_sidecars(_paths(d), d)
    assert r["normalized"] == 0 and r["refused"]
    assert "buttons" in r["refused"][0]
    assert np.load(p)["buttons"].shape == (40,), "the file is left exactly as it was"


def test_normalize_sidecars_refuses_tics_that_differ_from_the_recording(tmp_path):
    d, p = _wide_sidecar(tmp_path, "tics", MIXED)
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files}
    cols["tic"] = np.asarray(cols["tic"]) + 1000
    np.savez(p, **cols)
    r = encode_parquet.normalize_sidecars(_paths(d), d)
    assert r["normalized"] == 0 and "tics differ" in r["refused"][0]


def test_normalize_sidecars_refuses_controls_that_differ_from_the_recording(tmp_path):
    """A sidecar from a different episode would otherwise be "repaired" into looking correct."""
    d, p = _wide_sidecar(tmp_path, "otherep", MIXED)
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files}
    cols["buttons"] = np.array([ATTACK] * 20)
    np.savez(p, **cols)
    r = encode_parquet.normalize_sidecars(_paths(d), d)
    assert r["normalized"] == 0 and "differ from the raw recording" in r["refused"][0]


def test_normalize_sidecars_reports_an_episode_with_no_sidecar(tmp_path):
    d, p = _wide_sidecar(tmp_path, "gone", MIXED)
    os.remove(p)
    r = encode_parquet.normalize_sidecars(_paths(d), d)
    assert r["missing"] == ["ep_00000"] and r["episodes"] == 0


def test_a_short_provenance_column_is_rewritten_not_called_unchanged(tmp_path):
    """Presence is not enough: a truncated `buttons_raw_len` described the wrong rows."""
    d, _, _ = _encode(tmp_path, _episode(MIXED), "prov")
    p = os.path.join(d, "ep_00000_meta.npz")
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files}
    cols["buttons_raw_len"] = cols["buttons_raw_len"][:5]
    np.savez(p, **cols)
    r = encode_parquet.normalize_sidecars(_paths(d), d)
    assert r["normalized"] == 1 and r["refused"] == []
    with np.load(p) as z:
        assert z["buttons_raw_len"].tolist() == [9] * 18 + [14, 2503]
        assert len(z["switch_requested_index"]) == 20


def test_rebuild_sidecar_masks_normalises_the_buttons_it_rewrites(tmp_path):
    """The mask repair already rewrites the file, so it shrinks it on the way through."""
    d, p = _wide_sidecar(tmp_path, "masks", MIXED)
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files}
    cols["is_decision"] = np.zeros(len(cols["tic"]), dtype=bool)      # force a mask change
    np.savez(p, **cols)
    r = encode_parquet.rebuild_sidecar_masks([os.path.join(d, "ep_00000.parquet")], d,
                                             {0: FORWARD, 1: ATTACK}, SKIP)
    assert r["refused"] == [] and r["changed"] == 1
    with np.load(p) as z:
        assert z["buttons"].dtype == np.dtype(f"<U{EXECUTED_BUTTONS}")
        assert z["buttons_raw_len"].tolist() == [9] * 18 + [14, 2503]


def test_the_mask_repair_compares_the_executed_control_not_the_raw_string(tmp_path):
    """A normalised sidecar beside a raw recording is the SAME control; refusing it blocks the repair."""
    d, _, _ = _encode(tmp_path, _episode(MIXED), "agree")
    with np.load(os.path.join(d, "ep_00000_meta.npz")) as z:
        cols = {k: z[k] for k in z.files}
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(d, "ep_00000.parquet"), columns=["action", "buttons", "deaths", "tic"])
    assert encode_parquet._columns_agree(cols, t, "buttons")
    cols["buttons"] = np.array([normalize_buttons(ATTACK)] * len(cols["tic"]))
    assert not encode_parquet._columns_agree(cols, t, "buttons")


def test_the_cli_repairs_a_directory_without_touching_a_gpu_or_a_frame(tmp_path, monkeypatch, capsys):
    """The repair must run on a busy machine: no VAE, no CUDA probe, and never the `frame` column."""
    import torch
    import pyarrow.parquet as pq
    d, p = _wide_sidecar(tmp_path, "cli", MIXED)
    monkeypatch.setattr(torch.cuda, "is_available",
                        lambda: pytest.fail("--normalize-sidecars must not touch a GPU"))
    real = pq.read_table

    def no_frames(path, **kw):
        assert "frame" not in (kw.get("columns") or []), "the repair must not decode frames"
        assert kw.get("columns"), "the repair must name the columns it needs"
        return real(path, **kw)
    monkeypatch.setattr(pq, "read_table", no_frames)
    encode_parquet.main(encode_parquet.build_parser().parse_args(
        ["--normalize-sidecars", "--in-dir", d, "--out-dir", d]))
    assert "DONE" in capsys.readouterr().out
    assert np.load(p)["buttons"].dtype == np.dtype(f"<U{EXECUTED_BUTTONS}")


def test_the_cli_requires_in_dir_now_that_it_checks_the_recording(tmp_path, capsys):
    d, _ = _wide_sidecar(tmp_path, "noin", MIXED)
    with pytest.raises(SystemExit):
        encode_parquet.build_parser().parse_args(["--normalize-sidecars", "--out-dir", d])
    assert "--in-dir" in capsys.readouterr().err
    # and main refuses it too, for a caller that builds the namespace itself
    a = encode_parquet.build_parser().parse_args(["--normalize-sidecars", "--in-dir", d, "--out-dir", d])
    a.in_dir = ""
    with pytest.raises(SystemExit, match="--in-dir is required"):
        encode_parquet.main(a)


def test_the_cli_exits_nonzero_when_a_sidecar_is_refused(tmp_path):
    d, _ = _wide_sidecar(tmp_path, "clibad", [FORWARD] * 20,
                         stored=[FORWARD + "00000"] * 19 + ["12000000000000"])
    with pytest.raises(SystemExit, match="refused"):
        encode_parquet.main(encode_parquet.build_parser().parse_args(
            ["--normalize-sidecars", "--in-dir", d, "--out-dir", d]))


def test_an_empty_directory_is_reported_not_crashed(tmp_path):
    d = str(tmp_path / "empty")
    os.makedirs(d)
    assert encode_parquet.normalize_sidecars([], d) == {"episodes": 0, "normalized": 0, "unchanged": 0,
                                                        "missing": [], "refused": []}


def test_the_canonical_table_is_unchanged_by_the_normalisation(tmp_path):
    """The table keys on `str(b)[:9]`, and normalisation never moves those nine characters."""
    per, _, _ = trajectory([(0, FORWARD, SKIP, None)] * 4 + [(1, ATTACK, SKIP, None)] * 4)
    raw = canonical_for([(0, FORWARD, SKIP, None)] * 4 + [(1, ATTACK, SKIP, None)] * 4)
    _, _, m = _encode(tmp_path, [(t, a, b, d) for t, a, b, d in per], "canon")
    from transitions import canonical_table
    assert canonical_table(m["action"], m["buttons"]) == raw
