"""Every random stream in an evaluator is keyed, and a raw target is the frame that was asked for.

Two defects that move a number without crashing:

  * `RawFrames.get` located the raw frame with `np.searchsorted` and used the result unchecked.
    `searchsorted` returns an INSERTION POINT, so a tic the recording does not hold scored the NEXT
    frame, or past the end the last one. An off-by-one raw reference moves PSNR by a fraction of a
    dB and looks healthy.
  * the initial noise and the sampler's stochastic term are keyed per window, but the CONTEXT
    corruption came from `torch.randn_like`, i.e. the global generator. With `--infer-noise > 0` the
    values a window received then depended on the batch size and the sampler's step count, so half
    the comparison was paired and half was not.

    python -m pytest paper/fixtures/test_eval_noise_keys.py -q
"""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

pytest.importorskip("pyarrow")

import eval_tf  # noqa: E402
import rollout_eval  # noqa: E402


# ---------------------------------------------------------------------------------------
# the raw target is the frame at exactly that tic
# ---------------------------------------------------------------------------------------

def _recording(path, tics):
    """A parquet recording holding one distinguishable frame per tic in `tics`."""
    import io
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    blobs = []
    for t in tics:
        buf = io.BytesIO()
        Image.new("RGB", (4, 4), (int(t) % 256, 0, 0)).save(buf, format="PNG")
        blobs.append(buf.getvalue())
    pq.write_table(pa.table({"tic": pa.array([int(t) for t in tics], pa.int32()),
                             "frame": pa.array(blobs, pa.binary())}), path, compression=None)


def test_the_frame_returned_is_the_frame_asked_for(tmp_path):
    _recording(os.path.join(tmp_path, "ep_00000.parquet"), range(10))
    raw = eval_tf.RawFrames(str(tmp_path))
    for t in (0, 5, 9):
        assert int(raw.get(0, t)[0, 0, 0]) == t


def test_a_tic_the_recording_does_not_hold_is_refused(tmp_path):
    """The defect: `searchsorted` for a missing tic returns the index of the NEXT one, so the wrong
    frame was scored in silence."""
    _recording(os.path.join(tmp_path, "ep_00000.parquet"), [0, 1, 2, 5, 6])
    raw = eval_tf.RawFrames(str(tmp_path))
    assert int(np.searchsorted([0, 1, 2, 5, 6], 3)) == 3, "the old lookup would have scored tic 5"
    with pytest.raises(KeyError, match="no frame at tic 3"):
        raw.get(0, 3)


def test_a_tic_past_the_end_is_refused(tmp_path):
    _recording(os.path.join(tmp_path, "ep_00000.parquet"), range(5))
    raw = eval_tf.RawFrames(str(tmp_path))
    with pytest.raises(KeyError, match="no frame at tic 99"):
        raw.get(0, 99)
    with pytest.raises(KeyError, match="no frame at tic -1"):
        raw.get(0, -1)


def test_the_refusal_says_what_the_recording_holds(tmp_path):
    _recording(os.path.join(tmp_path, "ep_00000.parquet"), range(3, 9))
    raw = eval_tf.RawFrames(str(tmp_path))
    with pytest.raises(KeyError, match=r"recorded tics 3\.\.8, 6 rows"):
        raw.get(0, 100)


# ---------------------------------------------------------------------------------------
# the context-noise stream is keyed like the others
# ---------------------------------------------------------------------------------------

def test_the_context_stream_is_separate_from_the_initial_and_eta_streams():
    shape = (2, 4, 4, 5)
    keys = [(0, 11, 3, 0), (0, 11, 4, 0)]
    init = eval_tf.window_noise(shape, keys)
    ctx = eval_tf.window_noise(shape, keys, purpose="context")
    eta = eval_tf.window_noise(shape, keys, purpose="eta")
    assert not torch.allclose(init, ctx)
    assert not torch.allclose(ctx, eta)
    assert torch.equal(eval_tf.window_noise(shape, keys, purpose="context"), ctx), "not reproducible"


def test_the_context_noise_of_one_window_does_not_depend_on_the_batch():
    """What keying buys: the same window's context corruption in a batch of eight and on its own."""
    keys = [(0, 11, s, 0) for s in range(8)]
    big = eval_tf.window_noise((8, 4, 4, 5), keys, purpose="context")
    one = eval_tf.window_noise((1, 4, 4, 5), keys[5:6], purpose="context")
    assert torch.allclose(big[5], one[0])


def test_the_rollout_context_corruption_takes_the_keys():
    ctx = torch.zeros(2, 8, 4, 5)
    keys = [(0, 11, 3, 0), (0, 11, 4, 0)]
    a, bucket = rollout_eval.fixed_noise(ctx, 0.35, 0.7, 10, keys)
    b, _ = rollout_eval.fixed_noise(ctx, 0.35, 0.7, 10, keys)
    assert torch.equal(a, b), "the corruption is not reproducible from the keys"
    assert bucket.tolist() == [5, 5]
    # the same rollout, alone in its batch, gets the same corruption
    one, _ = rollout_eval.fixed_noise(ctx[1:2], 0.35, 0.7, 10, keys[1:])
    assert torch.allclose(a[1], one[0], atol=1e-6)


def test_the_unkeyed_rollout_corruption_was_not_paired():
    """Reproduces the defect: without keys the values come from the global generator, so two calls
    with the same arguments differ and a batch change re-deals every rollout."""
    ctx = torch.zeros(2, 8, 4, 5)
    torch.manual_seed(0)
    a, _ = rollout_eval.fixed_noise(ctx, 0.35, 0.7, 10)
    b, _ = rollout_eval.fixed_noise(ctx, 0.35, 0.7, 10)
    assert not torch.allclose(a, b)


def test_no_evaluator_corrupts_context_from_the_global_generator():
    for name in ("eval_tf.py", "rollout_eval.py"):
        code = open(os.path.join(REPO, name)).read()
        body = code[code.find("def main") if name == "eval_tf.py" else 0:]
        assert "randn_like(run)" not in body, f"{name}: the context noise is unkeyed again"
    roll = open(os.path.join(REPO, "rollout_eval.py")).read()
    # the one remaining call is the documented fallback for a caller that passes no keys
    code_lines = [ln for ln in roll.splitlines()
                  if "torch.randn_like" in ln and not ln.lstrip().startswith("#")
                  and "`torch.randn_like`" not in ln]
    assert len(code_lines) == 1 and code_lines[0].strip() == "eps = torch.randn_like(ctx)", code_lines
    assert 'purpose="context"' in roll
    assert 'purpose="context"' in open(os.path.join(REPO, "eval_tf.py")).read()


def test_the_rollout_uses_one_key_tuple_for_every_stream():
    roll = open(os.path.join(REPO, "rollout_eval.py")).read()
    assert "noise_keys = rollout_noise_keys(noise_seed, chunk, h)" in roll
    assert "window_noise(shape, noise_keys)" in roll
    assert "eta_noise_fn(shape, noise_keys, device)" in roll
    assert "args.noise_buckets, noise_keys)" in roll
