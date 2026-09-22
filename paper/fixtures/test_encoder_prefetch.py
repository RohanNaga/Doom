"""`encode_parquet.py` prepares the next batch of frames while the card encodes this one.

The per-tic encode is GPU-bound: measured on an idle RTX A6000, the VAE forward alone sustains 95
frames/s at batch 64 in bf16 while the whole encoder managed 83, and the difference is the host
sitting idle on the parquet read, the PNG decode and the `np.stack` of every batch in turn. A single
producer thread closes that gap.

The corpus it writes is already on disk, so the one thing this change must not do is move a byte.
These tests pin that: `batch_stream` yields exactly the arrays, in exactly the order and at exactly
the batch sizes the old serial loop produced (which is reimplemented here verbatim as
`serial_batches`), and a whole synthetic episode encoded through it with the deterministic stub
encoder lands the same latents and the same sidecar as the serial path. The batch sizes matter as
much as the contents: under bf16 autocast cuDNN chooses its convolution algorithm from the batch
shape, so a regrouped batch would re-encode the same frame differently.

    python -m pytest paper/fixtures/test_encoder_prefetch.py -q
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

pytest.importorskip("pyarrow")
pytest.importorskip("torch")

from test_decision_only import (ATTACK, FORWARD, SKIP, STUCK, StubVAE,  # noqa: E402
                                canonical_for, trajectory, write_parquet)

import encode_parquet  # noqa: E402

# long enough to span several batches at every batch size below, with an anti-stuck override and a
# death so the decision grid is irregular and `keep` is genuinely non-contiguous
LONG = ([(0, FORWARD, SKIP, None)] * 9 + [(1, ATTACK, SKIP, None)] * 8
        + [(0, STUCK, 40, None)] + [(1, ATTACK, SKIP, 2)])


def serial_batches(frames_col, keep, batch_size, pool):
    """The loop body `encode_episode` ran before prefetching, unchanged. The reference to beat."""
    for i in range(0, len(keep), batch_size):
        idx = keep[i:i + batch_size]
        raw = [frames_col[int(j)].as_py() for j in idx]
        yield np.stack(list(pool.map(encode_parquet.decode, raw)))


@pytest.fixture
def episode(tmp_path):
    """(parquet path, arrow table) of one per-tic recording of `LONG`."""
    import pyarrow.parquet as pq
    rows, _, _ = trajectory(LONG)
    path = str(tmp_path / "ep_00000.parquet")
    write_parquet(path, rows, 1)
    return path, pq.read_table(path)


def both(table, keep, batch_size, prefetch=2, threads=3):
    """(serial batches, prefetched batches) for the same selection."""
    with ThreadPoolExecutor(threads) as pool:
        want = list(serial_batches(table["frame"], keep, batch_size, pool))
        got = list(encode_parquet.batch_stream(table["frame"], keep, batch_size, pool, prefetch))
    return want, got


@pytest.mark.parametrize("batch_size", [1, 7, 16, 64, 4096])
@pytest.mark.parametrize("prefetch", [0, 1, 2, 5])
def test_prefetched_batches_are_the_serial_batches(episode, batch_size, prefetch):
    """Same count, same order, same bytes, at every batch size and every lookahead."""
    _, table = episode
    keep = np.arange(table.num_rows)
    want, got = both(table, keep, batch_size, prefetch)
    assert len(got) == len(want)
    assert [b.shape for b in got] == [b.shape for b in want]
    assert b"".join(b.tobytes() for b in got) == b"".join(b.tobytes() for b in want)


def test_batch_sizes_are_unchanged(episode):
    """The cuDNN algorithm is chosen from the batch shape, so no batch may be regrouped."""
    _, table = episode
    n, bs = table.num_rows, 16
    assert n > 3 * bs and n % bs                      # several full batches and a short tail
    _, got = both(table, np.arange(n), bs)
    assert [len(b) for b in got] == [min(bs, n - i) for i in range(0, n, bs)]


def test_a_non_contiguous_row_selection_is_followed_exactly(episode):
    """`--align-decisions` hands `batch_stream` a sparse `keep`; the rows must still be those rows."""
    _, table = episode
    tic = table["tic"].to_numpy(zero_copy_only=False)
    keep = np.flatnonzero(tic % SKIP == 0)
    assert 0 < len(keep) < table.num_rows
    want, got = both(table, keep, 7)
    assert b"".join(b.tobytes() for b in got) == b"".join(b.tobytes() for b in want)
    # and they really are the selected rows, not the first len(keep) rows
    with ThreadPoolExecutor(2) as pool:
        first = list(encode_parquet.batch_stream(table["frame"], np.arange(len(keep)), 7, pool))
    assert not np.array_equal(np.concatenate(got), np.concatenate(first))


def test_an_empty_selection_yields_nothing(episode):
    _, table = episode
    _, got = both(table, np.arange(0), 16)
    assert got == []


def test_a_decode_failure_surfaces_instead_of_hanging(episode, monkeypatch):
    """A producer thread must not swallow the error that used to be raised on the caller's stack."""
    _, table = episode

    def boom(_b):
        raise ValueError("bad png")

    monkeypatch.setattr(encode_parquet, "decode", boom)
    with ThreadPoolExecutor(2) as pool:
        with pytest.raises(ValueError, match="bad png"):
            list(encode_parquet.batch_stream(table["frame"], np.arange(table.num_rows), 8, pool))


def encode_one(tmp_path, name, rows, canon, stream=None):
    """Encode one synthetic per-tic episode with the stub encoder; `stream` overrides `batch_stream`."""
    import torch
    out = str(tmp_path / name)
    os.makedirs(out)
    src = os.path.join(out, "ep_00000.parquet")
    write_parquet(src, rows, 1)
    encode_parquet.CANONICAL = canon
    saved = encode_parquet.batch_stream
    if stream is not None:
        encode_parquet.batch_stream = stream
    try:
        with ThreadPoolExecutor(3) as pool:
            encode_parquet.encode_episode(src, out, StubVAE(), "cpu", torch.float32, SKIP, 16, pool,
                                          every_tic=True)
    finally:
        encode_parquet.batch_stream = saved
    return out


def test_every_tic_output_is_byte_identical_to_the_serial_encoder(tmp_path):
    """The whole point: the latents and the sidecar this writes are the ones already on disk.

    The `.npy` is compared byte for byte. The `.npz` is compared array by array because `np.savez`
    stamps each zip member with the wall clock, which would make a raw file comparison a clock test
    rather than a content one.
    """
    rows, _, _ = trajectory(LONG)
    canon = canonical_for(LONG)
    old = encode_one(tmp_path, "serial", rows, canon,
                     stream=lambda fc, keep, bs, pool, prefetch=2: serial_batches(fc, keep, bs, pool))
    new = encode_one(tmp_path, "prefetch", rows, canon)

    a = open(os.path.join(old, "ep_00000_latents.npy"), "rb").read()
    b = open(os.path.join(new, "ep_00000_latents.npy"), "rb").read()
    assert a == b and len(a) > 0
    with np.load(os.path.join(old, "ep_00000_meta.npz")) as za, \
         np.load(os.path.join(new, "ep_00000_meta.npz")) as zb:
        assert sorted(za.files) == sorted(zb.files)
        for k in za.files:
            assert za[k].dtype == zb[k].dtype and za[k].shape == zb[k].shape, k
            assert za[k].tobytes() == zb[k].tobytes(), k
