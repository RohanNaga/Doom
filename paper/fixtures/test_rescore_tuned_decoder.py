"""The tuned-decoder column (`scripts/spiderman/rescore_tuned_decoder.sh`): six reads, only the decoder changed.

Astra's review of 2026-09-26 (section 5, item 5): the tuned columns are eval_tf.py reads of the U-Net
200k EMA, the SD 3.5 newest snapshot and the PixArt newest snapshot at horizons 1 and 4 on 512 val
windows, identical to the stock-decoder reads except for `--vae-path`, `--latent-scale` and
`--latent-shift` (eval_tf.py:417-420). These tests pin every token of the six commands against the
gate-5 readback they must match (scripts/cluster/gates.sh readback_cmd), the output directories, the
newest-snapshot choice, and the failure accounting of a real (stubbed) run.

    python -m pytest paper/fixtures/test_rescore_tuned_decoder.py -q
"""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESCORE = os.path.join(REPO, "scripts", "spiderman", "rescore_tuned_decoder.sh")

STUB = '''#!/usr/bin/env python3
"""Stands in for eval_tf.py: logs its argv and writes <out-dir>/metrics.json, or exits 3 when the
checkpoint path contains STUB_FAIL."""
import os, sys
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(" ".join(sys.argv) + "\\n")
out, ck = sys.argv[sys.argv.index("--out-dir") + 1], sys.argv[sys.argv.index("--ckpt") + 1]
if os.environ.get("STUB_FAIL") and os.environ["STUB_FAIL"] in ck:
    sys.exit(3)
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "metrics.json"), "w").write("{}")
'''

SNAPSHOTS = {"040-unet-nexttic": [190000, 200000],
             "042-sd35-nexttic": [150000, 160000],
             "041-pixart-nexttic": [190000, 200000]}


def root_with_runs(tmp_path, decoders=True):
    root = tmp_path / "root"
    for run, steps in SNAPSHOTS.items():
        (root / "results_spiderman" / run).mkdir(parents=True)
        for s in steps:
            (root / "results_spiderman" / run / f"snap_{s:07d}.pt").write_bytes(b"w")
    if decoders:
        for name in ("vae_decoder_sd1x_mse", "vae_decoder_sd35_mse"):
            (root / name / "vae").mkdir(parents=True)
            (root / name / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"w")
    return root


def rescore(tmp_path, root, **env):
    """Run the script with a throwaway root and stub interpreters. Returns (proc, calls)."""
    stub = tmp_path / "stubpy"
    stub.write_text(STUB)
    stub.chmod(0o755)
    log = tmp_path / "stub.log"
    if log.exists():
        log.unlink()
    e = {k: v for k, v in os.environ.items() if not k.endswith("_STEP") and k not in ("DRY", "FORCE")}
    e.update(DOOM_ROOT=str(root), PY_UNET=str(stub), PY_SD35=str(stub), STUB_LOG=str(log), **env)
    proc = subprocess.run(["bash", RESCORE, "3"], capture_output=True, text=True, env=e, timeout=60)
    return proc, (log.read_text().splitlines() if log.exists() else [])


def dry(tmp_path, root, **env):
    p, calls = rescore(tmp_path, root, DRY="1", **env)
    assert p.returncode == 0, p.stderr
    assert calls == []
    return [ln.split() for ln in p.stdout.splitlines() if ln.startswith("DRY ")]


def expected(root, run, bb, step, k, py):
    """The gate-5 readback with --use-ema and 512 windows, and the tuned decoder's flags."""
    d, r = str(root), f"{root}/results_spiderman/{run}"
    repo = f"{d}/repo_launch2" if bb == "pixart" else f"{d}/repo_launch"
    s = "_sd35" if bb == "sd35" else ""
    src = {"unet": ["--sd-path", "CompVis/stable-diffusion-v1-4"],
           "pixart": ["--pixart-path", "PixArt-alpha/PixArt-XL-2-512x512"],
           "sd35": ["--sd35-path", "stabilityai/stable-diffusion-3.5-medium"]}[bb]
    dec = (["--vae-path", f"{d}/vae_decoder_sd35_mse/vae", "--latent-scale", "1.5305", "--latent-shift", "0.0609"]
           if bb == "sd35" else ["--vae-path", f"{d}/vae_decoder_sd1x_mse/vae", "--latent-scale", "0.18215"])
    return ([py, f"{repo}/eval_tf.py", "--backbone", bb, "--latent-channels", "16" if bb == "sd35" else "4",
             "--ckpt", f"{r}/snap_{step:07d}.pt", "--use-ema", "--tic-stride", "1", "--horizon-tics", str(k),
             "--latents-dir", f"{d}/latents_arnold_dense_pertic_eval{s}/val",
             "--split", f"{d}/latents_arnold_dense_pertic_eval{s}/split_val.json", "--subset", "val",
             "--parquet-dir", f"{d}/raw_arnold_dense/arenas", "--num-windows", "512", "--batch-size", "16",
             "--steps", "10", "--context-frames", "32", "--num-actions", "29", "--hf-cache", f"{d}/hf/hub",
             *src, *dec, "--out-dir", f"{r}/steward_{step}/tf_ema_h{k}_tuneddec"])


def test_dry_prints_exactly_the_six_reads(tmp_path):
    root = root_with_runs(tmp_path)
    lines = dry(tmp_path, root)
    stub = str(tmp_path / "stubpy")
    rows = (("040-unet-nexttic", "unet", 200000), ("042-sd35-nexttic", "sd35", 160000),
            ("041-pixart-nexttic", "pixart", 200000))
    want = [(run, bb, step, k) for run, bb, step in rows for k in (1, 4)]
    assert len(lines) == 6
    for ln, (run, bb, step, k) in zip(lines, want):
        assert ln[1:4] == [run, f"h{k}", "CUDA_VISIBLE_DEVICES=3"]
        assert ln[4:] == expected(root, run, bb, step, k, stub), (run, k)


def test_only_the_decoder_flags_differ_from_the_stock_read(tmp_path):
    """The stock SD 3.5 read names its autoencoder with --vae-subfolder; the tuned one is a plain directory."""
    root = root_with_runs(tmp_path)
    decoder_flags = {"--vae-path", "--vae-subfolder", "--latent-scale", "--latent-shift", "--out-dir"}
    for ln in dry(tmp_path, root):
        argv = ln[4:]
        assert "--vae-subfolder" not in argv and "--wandb-run" not in argv
        # --vae-path, --latent-scale, --out-dir, and --latent-shift for SD 3.5 only
        assert sum(a in decoder_flags for a in argv) == (4 if "sd35" in argv else 3)


def test_the_steps_can_be_pinned(tmp_path):
    root = root_with_runs(tmp_path)
    lines = dry(tmp_path, root, SD35_STEP="150000", PIXART_STEP="190000", UNET_STEP="190000")
    cks = {ln[1]: ln[ln.index("--ckpt") + 1] for ln in lines}
    assert cks["042-sd35-nexttic"].endswith("snap_0150000.pt")
    assert cks["041-pixart-nexttic"].endswith("snap_0190000.pt")
    assert cks["040-unet-nexttic"].endswith("snap_0190000.pt")


def test_dry_touches_nothing(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    lines = dry(tmp_path, root)
    assert len(lines) == 6 and list(root.iterdir()) == []
    assert any("snap_<newest>.pt" in " ".join(ln) for ln in lines)


def test_a_run_scores_all_six_and_skips_them_the_second_time(tmp_path):
    root = root_with_runs(tmp_path)
    p, calls = rescore(tmp_path, root)
    assert p.returncode == 0, p.stderr
    assert len(calls) == 6 and "RESCORE_DONE" in p.stdout
    for k in (1, 4):
        assert (root / "results_spiderman" / "042-sd35-nexttic" / "steward_160000" / f"tf_ema_h{k}_tuneddec"
                / "metrics.json").exists()
    p, calls = rescore(tmp_path, root)
    assert p.returncode == 0 and calls == [] and p.stdout.count("already scored") == 6
    p, calls = rescore(tmp_path, root, FORCE="1")
    assert len(calls) == 6


def test_a_failed_read_fails_the_script_but_not_the_other_reads(tmp_path):
    root = root_with_runs(tmp_path)
    p, calls = rescore(tmp_path, root, STUB_FAIL="042-sd35")
    assert p.returncode == 1 and "RESCORE_DONE" not in p.stdout and "RESCORE_FAILED" in p.stderr
    assert len(calls) == 6
    assert "042-sd35-nexttic h1: eval_tf.py exit 3" in p.stderr


def test_an_unfinished_decoder_is_never_scored(tmp_path):
    root = root_with_runs(tmp_path, decoders=False)
    p, calls = rescore(tmp_path, root)
    assert p.returncode == 1 and calls == []
    assert p.stderr.count("no decoder weights") == 6
