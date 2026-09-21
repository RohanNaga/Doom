"""A local-only checkpoint goes straight to a file, with the same bytes as the byte buffer wrote.

`train_wm.save_checkpoint` serialized every checkpoint into a `BytesIO` and kept the bytes, even
with no remote destination to feed them to. For an SD 3.5 recovery checkpoint that is an extra
~37 GB of host memory on top of the fp32 model copy and the two EMA copies the caller already
holds, at the moment a second training job is running on the same host.

    python -m pytest paper/fixtures/test_checkpoint_save.py -q
"""
import io
import os
import subprocess
import sys
import zipfile

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

os.environ.setdefault("ACCELERATE_USE_CPU", "1")

import train_wm  # noqa: E402


def _ck_obj():
    return {"model": {"w": torch.arange(64, dtype=torch.float32).reshape(8, 8)},
            "ema": [torch.ones(4, 4)], "step": 7, "args": {"backbone": "dit"}}


def test_a_local_only_save_builds_no_byte_buffer(tmp_path, monkeypatch):
    """37 GB of avoidable allocation for an SD 3.5 recovery checkpoint, next to the fp32 model copy
    and two EMA copies the caller already holds.

    `io.BytesIO` is counted by the FRAME that constructs it, because torch's own serialization code
    builds buffers of its own for unrelated reasons; only a construction inside `train_wm` is the
    defect."""
    made = []
    real_buffer = io.BytesIO

    def spy(*a, **k):
        made.append(os.path.basename(sys._getframe(1).f_code.co_filename))
        return real_buffer(*a, **k)

    monkeypatch.setattr(io, "BytesIO", spy)
    seen = []
    real_save = torch.save
    monkeypatch.setattr(torch, "save", lambda obj, f, *a, **k: seen.append(f) or real_save(obj, f, *a, **k))
    path = str(tmp_path / "0000005.pt")
    assert train_wm.save_checkpoint(_ck_obj(), path) == [path]
    assert "train_wm.py" not in made, made
    assert seen and not any(isinstance(f, real_buffer) for f in seen), seen
    assert getattr(seen[0], "name", "").endswith(".tmp"), "the write was not renamed into place"


def test_the_bytes_are_the_ones_the_buffer_used_to_produce(tmp_path):
    """Byte-identical to the old path, which is the reason the object is serialized into an open
    file rather than into a path: `torch.save(obj, "0000005.pt.tmp")` names the zip archive after
    the temporary file, so every byte offset after the header would move."""
    obj = _ck_obj()
    path = str(tmp_path / "a.pt")
    train_wm.save_checkpoint(obj, path)
    buf = io.BytesIO()
    torch.save(obj, buf)
    assert open(path, "rb").read() == buf.getvalue()


def test_the_checkpoint_does_not_carry_its_temporary_name(tmp_path):
    for name in ("0000005.pt", "best.pt", "snap_0010000.pt"):
        p = str(tmp_path / name)
        train_wm.save_checkpoint(_ck_obj(), p)
        assert zipfile.ZipFile(p).namelist()[0].startswith("archive/"), name


def test_the_reloaded_checkpoint_is_the_object(tmp_path):
    obj = _ck_obj()
    path = str(tmp_path / "b.pt")
    train_wm.save_checkpoint(obj, path)
    back = torch.load(path, map_location="cpu", weights_only=False)
    assert back["step"] == 7 and back["args"] == {"backbone": "dit"}
    assert torch.equal(back["model"]["w"], obj["model"]["w"])


def test_a_failed_write_leaves_no_partial_checkpoint(tmp_path, monkeypatch):
    path = str(tmp_path / "c.pt")
    train_wm.save_checkpoint(_ck_obj(), path)
    good = open(path, "rb").read()

    def half_then_die(obj, f, *a, **k):
        f.write(b"\x00" * 1024)               # a partially flushed file, as a full disk leaves
        raise RuntimeError("injected")

    monkeypatch.setattr(torch, "save", half_then_die)
    with pytest.raises(RuntimeError, match="injected"):
        train_wm.save_checkpoint(_ck_obj(), path)
    assert open(path, "rb").read() == good, "the previous checkpoint was clobbered"
    assert not os.path.exists(path + ".tmp"), "the partial temporary survived"


def test_a_local_disk_error_is_reported_and_cleaned_up(tmp_path, monkeypatch):
    path = str(tmp_path / "d.pt")

    def no_space(obj, f, *a, **k):
        f.write(b"\x00" * 16)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(torch, "save", no_space)
    with pytest.raises(RuntimeError, match="could not write checkpoint"):
        train_wm.save_checkpoint(_ck_obj(), path)
    assert not os.path.exists(path) and not os.path.exists(path + ".tmp")


def test_the_remote_branch_still_serialises_because_ssh_needs_stdin(tmp_path, monkeypatch):
    sent = {}

    class R:
        returncode = 0
        stderr = b""

    def fake_run(cmd, input=None, **kw):
        sent["bytes"] = input
        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    obj = _ck_obj()
    path = str(tmp_path / "run" / "0000010.pt")
    os.makedirs(os.path.dirname(path))
    written = train_wm.save_checkpoint(obj, path, remote="user@host:/copy")
    assert written == ["/copy/run/0000010.pt", path]
    buf = io.BytesIO()
    torch.save(obj, buf)
    assert sent["bytes"] == buf.getvalue()
    assert open(path, "rb").read() == buf.getvalue()


def test_the_disk_check_has_a_size_to_compare_without_serialising(tmp_path):
    obj = _ck_obj()
    path = str(tmp_path / "e.pt")
    train_wm.save_checkpoint(obj, path)
    est = train_wm.tensor_bytes(obj)
    real = os.path.getsize(path)
    assert 64 * 4 + 16 * 4 == est, est          # the two tensors, exactly
    assert est <= real <= est + 8192, (est, real)


def test_a_shared_storage_is_counted_once():
    """The EMA of a model that aliases a parameter must not double the estimate."""
    w = torch.zeros(1000)
    assert train_wm.tensor_bytes({"a": w, "b": w}) == 4000
    assert train_wm.tensor_bytes({"a": w, "b": w.clone()}) == 8000
