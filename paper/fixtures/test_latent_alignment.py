"""The stored-latent alignment gate catches a shard whose latents are off by a row.

`check_latent_alignment.py` (review 2026-09-22, section 7.1 item 2) re-encodes a few stored rows per
shard with the settings the shard recorded, compares them with what is on disk, and decodes the
stored rows against the raw frames with deliberately shifted negative controls. These tests build a
two-shard synthetic corpus with a stub autoencoder on the CPU: the encoder is an 8x8 average pool
and the decoder a nearest upsample, so decode(encode(frame)) is a blocky copy of the frame and the
unshifted alignment is measurably the best one.

    python -m pytest paper/fixtures/test_latent_alignment.py -q
"""
import io
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import check_latent_alignment as cla  # noqa: E402
from encode_parquet import encode_batch  # noqa: E402

T, BATCH = 40, 16          # batches [0,16) [16,32) and a tail of 8 rows


class StubVAE:
    """encode: 8x8 average pool of the RGB plus one zero channel; decode: nearest 8x upsample."""

    def encode(self, x):
        z = torch.nn.functional.avg_pool2d(x, 8)
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=torch.cat([z, torch.zeros_like(z[:, :1])], 1)))

    def decode(self, z):
        return SimpleNamespace(sample=torch.nn.functional.interpolate(z[:, :3], scale_factor=8, mode="nearest"))


def frame(i):
    """A smooth pattern that moves with i, so neighbouring rows are distinguishable after pooling."""
    y, x = np.mgrid[0:240, 0:320].astype(np.float64)
    r = 127 + 100 * np.sin(2 * np.pi * (x / 80.0 + 0.11 * i))
    g = 127 + 100 * np.cos(2 * np.pi * (y / 60.0 - 0.07 * i))
    b = 127 + 100 * np.sin(2 * np.pi * ((x + y) / 100.0 + 0.05 * i))
    return np.stack([r, g, b], -1).astype(np.uint8)


def write_recording(path, ep, n=T):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    pngs = []
    for t in range(n):
        buf = io.BytesIO()
        Image.fromarray(frame(ep * 1000 + t)).save(buf, format="PNG")
        pngs.append(buf.getvalue())
    pq.write_table(pa.table({"tic": pa.array(range(n), pa.int32()), "frame": pa.array(pngs, pa.binary())}),
                   path, row_group_size=256)


def encode_episode(ep, n=T):
    frames = np.stack([frame(ep * 1000 + t) for t in range(n)])
    return np.concatenate([encode_batch(StubVAE(), frames[a:a + BATCH], "cpu", torch.float32, False, 1.0, None)
                           for a in range(0, n, BATCH)])


def corpus(tmp_path, shards=((0, 1), (2, 3)), shift_shard=None, perturb_shard=None, meta_for=None):
    """Raw recordings and a latent corpus of `len(shards)` shards, in the encoder's own layout."""
    raw, lat = tmp_path / "raw", tmp_path / "lat"
    raw.mkdir(parents=True, exist_ok=True)
    lat.mkdir(parents=True, exist_ok=True)
    for s, eps in enumerate(shards):
        lines = []
        for ep in eps:
            write_recording(str(raw / f"ep_{ep:05d}.parquet"), ep)
            z = encode_episode(ep)
            if s == shift_shard:          # latent row i holds frame i+1: off by one against the sidecar
                z = np.concatenate([z[1:], z[-1:]])
            if s == perturb_shard:
                z = (z.astype(np.float32) + 0.05).astype(np.float16)
            np.save(str(lat / f"ep_{ep:05d}_latents.npy"), z)
            np.savez(str(lat / f"ep_{ep:05d}_meta.npz"), tic=np.arange(T, dtype=np.int64),
                     episode_id=np.full(T, ep, dtype=np.int64))
            lines.append(json.dumps({"episode": f"ep_{ep:05d}", "frames": T}))
        (lat / f"episodes_{s:02d}.jsonl").write_text("\n".join(lines) + "\n")
        if meta_for is None or s in meta_for:
            (lat / f"encode_meta_{s:02d}.json").write_text(json.dumps({
                "vae_id": "stub", "vae_subfolder": "", "scaling_factor_applied": 1.0,
                "shift_factor_applied": None, "every_tic": True, "legacy": False,
                "args": {"batch_size": BATCH, "dtype": "fp32"}}))
    return str(lat), str(raw)


def run(lat, raw, **kw):
    return cla.check(lat, raw, device="cpu", vae=StubVAE(), **kw)


# ---------------------------------------------------------------------------------------

def test_a_clean_corpus_passes_every_shard(tmp_path):
    rep = run(*corpus(tmp_path))
    assert rep["ok"], rep
    for name in ("00", "01"):
        s = rep["shards"][name]
        assert s["ok"] and s["fraction_identical"] == 1.0 and s["mean_abs_diff"] == 0.0
        v = s["vae_psnr"]
        assert all(v["0"] > v[k] for k in ("-4", "-1", "+1", "+4")), v
        # both decoded through one decoder: identical latents decode to identical scores
        assert abs(s["vae_psnr_reencoded"] - v["0"]) < 1e-6


def test_a_shard_shifted_by_one_row_is_caught_and_the_others_are_not(tmp_path):
    rep = run(*corpus(tmp_path, shift_shard=1))
    assert not rep["ok"]
    good, bad = rep["shards"]["00"], rep["shards"]["01"]
    assert good["ok"], good["problems"]
    assert not bad["ok"]
    text = " ".join(bad["problems"])
    assert "shifted" in text and "mean |re-encoded - stored|" in text
    # the shifted control that wins is the one that undoes the shift
    v = bad["vae_psnr"]
    assert v["-1"] > v["0"], v


def test_the_report_is_per_shard_never_only_pooled(tmp_path):
    rep = run(*corpus(tmp_path, shards=((0,), (1,), (2,))))
    assert set(rep["shards"]) == {"00", "01", "02"}
    for s in rep["shards"].values():
        assert set(s["vae_psnr"]) == {"-4", "-1", "0", "+1", "+4"}
        assert {"max_abs_diff", "mean_abs_diff", "fraction_identical", "episodes", "encode_meta"} <= set(s)


def test_perturbed_latents_fail_the_difference_thresholds(tmp_path):
    rep = run(*corpus(tmp_path, perturb_shard=0))
    s = rep["shards"]["00"]
    assert not s["ok"]
    text = " ".join(s["problems"])
    assert "mean |re-encoded - stored|" in text and "bit-identical" in text
    assert rep["shards"]["01"]["ok"]


def test_the_thresholds_are_the_callers(tmp_path):
    lat, raw = corpus(tmp_path, perturb_shard=0)
    rep = run(lat, raw, max_abs_diff=1.0, min_identical=0.0)
    assert rep["shards"]["00"]["ok"], rep["shards"]["00"]["problems"]


def test_the_tail_batch_is_always_checked():
    rng = np.random.RandomState(0)
    blocks = cla.sample_rows(40, 16, rng)
    assert (32, 40) in blocks and len(blocks) == 2
    assert all(a % 16 == 0 for a, _ in blocks), "a block must start on an encoder batch boundary"
    assert cla.sample_rows(32, 16, rng)[-1] == (16, 32)
    assert cla.sample_rows(10, 16, rng) == [(0, 10)]


def test_a_shard_with_no_recorded_settings_fails_coverage(tmp_path):
    lat, raw = corpus(tmp_path, meta_for={0})
    rep = run(lat, raw)
    assert not rep["ok"]
    assert any("encode_meta_01.json" in p for p in rep["coverage_problems"])
    assert "01" not in rep["shards"]


def test_explicit_episodes_are_checked_inside_their_own_shards(tmp_path):
    lat, raw = corpus(tmp_path)
    rep = run(lat, raw, episodes=[0, 3])
    assert rep["ok"], rep
    assert rep["shards"]["00"]["episodes"] == [0] and rep["shards"]["01"]["episodes"] == [3]
    bad = run(lat, raw, episodes=[0, 9])
    assert not bad["ok"] and any("not encoded" in p for p in bad["coverage_problems"])


# ---------------------------------------------------------------------------------------
# coverage, one contract, finite values (Astra's review of the first version, 2026-09-23)
# ---------------------------------------------------------------------------------------

def test_a_missing_shard_log_can_no_longer_hide_a_shifted_shard(tmp_path):
    """The reproduction: delete shard 1's episodes log and its shifted latents went unchecked."""
    lat, raw = corpus(tmp_path, shift_shard=1)
    os.remove(os.path.join(lat, "episodes_01.jsonl"))
    rep = run(lat, raw)
    assert not rep["ok"]
    text = " ".join(rep["coverage_problems"])
    assert "belong to no shard log" in text and "[2, 3]" in text
    assert "encode_meta_01.json has no episodes_01.jsonl" in text


def test_no_shard_logs_at_all_fails(tmp_path):
    lat, raw = corpus(tmp_path)
    for f in os.listdir(lat):
        if f.startswith("episodes_"):
            os.remove(os.path.join(lat, f))
    rep = run(lat, raw)
    assert not rep["ok"] and any("no episodes_NN.jsonl" in p for p in rep["coverage_problems"])


def test_an_episode_claimed_by_two_shards_fails(tmp_path):
    lat, raw = corpus(tmp_path)
    with open(os.path.join(lat, "episodes_01.jsonl"), "a") as f:
        f.write(json.dumps({"episode": "ep_00000", "frames": T}) + "\n")
    rep = run(lat, raw)
    assert not rep["ok"] and any("listed by shards 00 and 01" in p for p in rep["coverage_problems"])


@pytest.mark.parametrize("key,value", [("scaling_factor_applied", 0.5), ("vae_id", "other/vae"),
                                       ("shift_factor_applied", 0.06)])
def test_shards_under_different_latent_contracts_fail(tmp_path, key, value):
    lat, raw = corpus(tmp_path)
    path = os.path.join(lat, "encode_meta_01.json")
    m = json.load(open(path))
    m[key] = value
    json.dump(m, open(path, "w"))
    rep = run(lat, raw)
    assert not rep["ok"]
    assert rep["contract_problems"], rep
    assert any("different" in p for p in rep["contract_problems"])


def test_a_different_batch_size_is_a_different_contract(tmp_path):
    lat, raw = corpus(tmp_path)
    path = os.path.join(lat, "encode_meta_01.json")
    m = json.load(open(path))
    m["args"]["batch_size"] = 8
    json.dump(m, open(path, "w"))
    rep = run(lat, raw)
    assert not rep["ok"] and any("batch_size" in p for p in rep["contract_problems"])


def test_a_sampled_nan_fails(tmp_path):
    """NaN > threshold is False, so a NaN in the sampled rows used to pass every comparison."""
    lat, raw = corpus(tmp_path)
    for ep in (0, 1):
        path = os.path.join(lat, f"ep_{ep:05d}_latents.npy")
        z = np.load(path)
        z[:, 0, 0, 0] = np.nan
        np.save(path, z)
    rep = run(lat, raw)
    assert not rep["ok"] and not rep["shards"]["00"]["ok"]
    assert any("non-finite" in p for p in rep["shards"]["00"]["problems"])
    assert rep["shards"]["01"]["ok"]


def test_non_finite_statistics_fail():
    diffs = np.array([np.nan, 0.0], np.float32)
    psnr = {s: [20.0] for s in cla.SHIFTS}
    stats, problems = cla.shard_verdict(diffs, 2, 2, psnr, [20.0], 1.0, 0.0)
    assert problems and "non-finite" in problems[0]


def test_the_cli_exits_nonzero_on_a_failing_shard_and_writes_the_report(tmp_path, monkeypatch):
    monkeypatch.setattr(cla, "load_encoder", lambda meta, device, cache_dir=None: StubVAE())
    lat, raw = corpus(tmp_path / "bad", shift_shard=1)
    out = tmp_path / "report.json"
    args = cla.build_parser().parse_args(["--latents-dir", lat, "--parquet-dir", raw, "--device", "cpu",
                                          "--out", str(out)])
    assert cla.main(args) == 2
    assert json.loads(out.read_text())["shards"]["01"]["ok"] is False
    lat, raw = corpus(tmp_path / "good")
    args = cla.build_parser().parse_args(["--latents-dir", lat, "--parquet-dir", raw, "--device", "cpu"])
    assert cla.main(args) == 0


def test_the_defaults_are_the_sep20_measurement():
    a = cla.build_parser().parse_args(["--latents-dir", "x", "--parquet-dir", "y"])
    assert a.max_abs_diff == 1e-3 and a.min_identical == 0.5 and a.episodes_per_shard == 2


def test_the_gates_run_it_per_space_on_train_and_val(tmp_path):
    e = {**os.environ, "DRY": "1", "DOOM_ROOT": str(tmp_path), "VAES": "sd15,sd35"}
    p = subprocess.run(["bash", os.path.join(REPO, "scripts", "cluster", "gates.sh")],
                       capture_output=True, text=True, env=e)
    assert p.returncode == 0, p.stderr
    lines = [ln for ln in p.stdout.splitlines() if "check_latent_alignment.py" in ln]
    assert len(lines) == 4, lines
    dirs = {ln.split("--latents-dir ")[1].split()[0] for ln in lines}
    assert dirs == {f"{tmp_path}/latents_arnold_dense_pertic/arenas",
                    f"{tmp_path}/latents_arnold_dense_pertic_sd35/arenas",
                    f"{tmp_path}/latents_arnold_dense_pertic_eval/val",
                    f"{tmp_path}/latents_arnold_dense_pertic_eval_sd35/val"}
    for ln in lines:
        assert f"--parquet-dir {tmp_path}/raw_arnold_dense/arenas" in ln and "--out " in ln
    order = p.stdout.splitlines()
    first = min(i for i, ln in enumerate(order) if "check_latent_alignment.py" in ln)
    assert first < min(i for i, ln in enumerate(order) if "--min-accuracy" in ln)
