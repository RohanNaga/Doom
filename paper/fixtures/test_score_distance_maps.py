"""`scripts/spiderman/score_distance_maps.sh`: the per-map scoring pass of the distance study.

The launcher runs the unchanged `eval_tf.py` once per map and horizon, on the per-map split files
`distance_study.py splits` wrote, with the memo's settings: 256 windows, EMA weights, no loader
workers, the paper's 10 sampler steps as a knob, the stock decoder (SD 3.5 with its own decoder
flags), and never `--wandb-run`, because thirty one-step reads would overwrite the headline
`eval/ema_h1` series of the training run. It is resumable: a completed map is skipped, a map scored
under a different key is refused rather than overwritten, and a failed map is retried alone.

The DRY path prints every command. The real path is run against a throwaway root with a stub
interpreter that records its argv and writes `metrics.json`, as the other launcher tests do.

    python -m pytest paper/fixtures/test_score_distance_maps.py -q
"""
import json
import os
import re
import stat
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SCRIPT = os.path.join(REPO, "scripts", "spiderman", "score_distance_maps.sh")

MAPS = (("val", 2, [6002, 6006]), ("val", 3, [6003]), ("unseen2", 18, [7, 20]))


def write_splits(d, maps=MAPS):
    os.makedirs(d, exist_ok=True)
    for s, m, eps in maps:
        with open(os.path.join(d, f"split_{s}_map{m:02d}.json"), "w") as f:
            json.dump({"val": eps, "meta": {"set": s, "map": m, "num_windows": 256}}, f)
    with open(os.path.join(d, "index.json"), "w") as f:
        json.dump({"maps": []}, f)


def launch(root, *args, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("PY", "CKPT", "MAPS", "STEPS", "HORIZONS", "RUN"))}
    e.update({"DOOM_ROOT": str(root), "RUN_REPO": str(root / "repo"), "SPLITS": str(root / "splits"),
              **{k: str(v) for k, v in env.items()}})
    return subprocess.run(["bash", SCRIPT, *[str(a) for a in args]], capture_output=True, text=True, env=e)


def dry(root, *args, **env):
    return launch(root, *args, DRY=1, **env)


def evals(out):
    return [ln for ln in out.splitlines() if ln.startswith("DRY eval_tf ")]


def flag(line, name):
    toks = line.split()
    return toks[toks.index(name) + 1] if name in toks else None


@pytest.fixture
def root(tmp_path):
    write_splits(str(tmp_path / "splits"))
    return tmp_path


def test_script_is_valid_bash():
    assert subprocess.run(["bash", "-n", SCRIPT]).returncode == 0


def test_dry_prints_one_command_per_map_and_horizon_with_every_one_tic_read_first(root):
    r = dry(root, 3, "unet", CKPT="/x/snap_0200000.pt")
    assert r.returncode == 0, r.stderr
    lines = evals(r.stdout)
    assert len(lines) == 3 * 2
    horizons = [flag(ln, "--horizon-tics") for ln in lines]
    assert horizons == ["1", "1", "1", "4", "4", "4"]
    outs = [flag(ln, "--out-dir") for ln in lines]
    assert len(set(outs)) == 6
    base = f"{root}/results_spiderman/distance_study/040-unet-nexttic/snap_0200000_ema_ddim10"
    assert outs[0] == f"{base}/val_map02_h1" and outs[-1] == f"{base}/unseen2_map18_h4"


def test_every_command_carries_the_memo_settings_and_never_wandb(root):
    lines = evals(dry(root, 3, "unet", CKPT="/x/snap_0200000.pt").stdout)
    for ln in lines:
        assert "eval_tf.py" in ln and "--wandb" not in ln
        for name, want in (("--num-windows", "256"), ("--num-workers", "0"), ("--steps", "10"), ("--subset", "val"),
                           ("--tic-stride", "1"), ("--latent-channels", "4"), ("--backbone", "unet"),
                           ("--ckpt", "/x/snap_0200000.pt"), ("--context-frames", "32")):
            assert flag(ln, name) == want, (name, ln)
        assert "--use-ema" in ln.split()
        assert "--vae-path" not in ln                                    # the stock SD 1.x decoder
    by_split = {os.path.basename(flag(ln, "--split")): ln for ln in lines if flag(ln, "--horizon-tics") == "1"}
    v = by_split["split_val_map02.json"]
    assert flag(v, "--latents-dir") == f"{root}/latents_arnold_dense_pertic_eval/val"
    assert flag(v, "--parquet-dir") == f"{root}/raw_arnold_dense/arenas"
    u = by_split["split_unseen2_map18.json"]
    assert flag(u, "--latents-dir") == f"{root}/latents_arnold_eval_pertic/unseen2"
    assert flag(u, "--parquet-dir") == f"{root}/raw_arnold_eval/unseen2"
    with open(SCRIPT) as f:
        code = "\n".join(ln.split("#", 1)[0] for ln in f)                 # comments may explain; code may not pass it
    assert "--wandb" not in code


@pytest.mark.parametrize("backbone", ["unet", "sd35", "pixart"])
def test_the_unchanged_eval_tf_parser_accepts_every_command(root, backbone):
    pytest.importorskip("torch")
    sys.path.insert(0, REPO)
    import eval_tf
    for ln in evals(dry(root, 3, backbone, CKPT="/x/c.pt").stdout):
        argv = ln.split()
        args = eval_tf.build_parser().parse_args(argv[argv.index(os.path.join(str(root), "repo", "eval_tf.py")) + 1:])
        assert args.use_ema and args.num_windows == 256 and args.num_workers == 0 and args.steps == 10
        assert args.wandb_run == "" and args.subset == "val" and args.backbone == backbone


def test_sd35_reads_its_own_latents_through_its_stock_decoder(root):
    lines = evals(dry(root, 3, "sd35", CKPT="/x/snap_0150000.pt").stdout)
    assert len(lines) == 6
    for ln in lines:
        assert flag(ln, "--latent-channels") == "16" and flag(ln, "--backbone") == "sd35"
        assert "--vae-path stabilityai/stable-diffusion-3.5-medium --vae-subfolder vae " \
               "--latent-scale 1.5305 --latent-shift 0.0609" in ln
        assert "_sd35/" in flag(ln, "--latents-dir")
        assert "/042-sd35-nexttic/snap_0150000_ema_ddim10/" in flag(ln, "--out-dir")


def test_steps_horizons_run_and_maps_are_knobs(root):
    lines = evals(dry(root, 3, "unet", CKPT="/x/c.pt", STEPS=4, HORIZONS="4", RUN="044-unet-x",
                      MAPS="val:3 unseen2").stdout)
    assert [os.path.basename(flag(ln, "--split")) for ln in lines] == ["split_val_map03.json",
                                                                       "split_unseen2_map18.json"]
    assert all(flag(ln, "--steps") == "4" and flag(ln, "--horizon-tics") == "4" for ln in lines)
    assert all("/044-unet-x/c_ema_ddim4/" in flag(ln, "--out-dir") for ln in lines)


def test_without_a_checkpoint_it_picks_the_run_s_latest_ema_checkpoint(root):
    r = dry(root, 3, "unet")
    assert "pick_checkpoint.py --results-dir " + f"{root}/results_spiderman/040-unet-nexttic --require-ema" in r.stdout
    assert all(flag(ln, "--ckpt") == "PICKED" for ln in evals(r.stdout))


def test_a_space_the_backbone_cannot_read_is_refused(root):
    assert dry(root, 3, "unet", SPACE="sd35").returncode == 2
    assert dry(root, 3, "sd35", SPACE="sd1").returncode == 2


def test_the_sealed_test_corpus_is_never_scored(root):
    write_splits(str(root / "splits"), [("test", 4, [7000])])
    r = dry(root, 3, "unet", CKPT="/x/c.pt")
    assert r.returncode == 2 and "test" in r.stderr
    assert not evals(r.stdout)


def test_no_split_files_is_an_error(tmp_path):
    r = dry(tmp_path, 3, "unet", CKPT="/x/c.pt")
    assert r.returncode == 2 and "split" in r.stderr


# ---------------------------------------------------------------------------------------------
# the real path, against a stub interpreter
# ---------------------------------------------------------------------------------------------

STUB = '''#!{python}
"""Stands in for the interpreter: `-c` snippets run for real; pick_checkpoint.py and eval_tf.py are faked."""
import json, os, sys
if sys.argv[1] == "-c":
    os.execv({python!r}, [{python!r}] + sys.argv[1:])
script = os.path.basename(sys.argv[1])
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps({{"argv": sys.argv[1:], "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}}) + "\\n")
def arg(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None
if script == "pick_checkpoint.py":
    path = arg("--ckpt") or os.path.join(arg("--results-dir"), "snap_0200000.pt")
    print(path, 200000, os.environ.get("STUB_EMA", "1"), "sha-" + os.path.basename(path))
    sys.exit(0)
if script == "eval_tf.py":
    out = arg("--out-dir")
    if os.path.basename(out) in os.environ.get("STUB_FAIL", "").split():
        sys.exit(1)
    os.makedirs(out, exist_ok=True)
    json.dump({{"psnr_raw": {{"mean": 21.0}}, "persist_psnr_raw": {{"mean": 20.0}}, "sampling_frames_per_s": 19.5}},
              open(os.path.join(out, "metrics.json"), "w"))
    sys.exit(0)
sys.exit(3)
'''


@pytest.fixture
def real(root):
    stub = root / "stubpy"
    stub.write_text(STUB.format(python=sys.executable))
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    (root / "repo").mkdir()
    for s, _, eps in MAPS:
        d = root / ("latents_arnold_dense_pertic_eval" if s == "val" else "latents_arnold_eval_pertic") / s
        d.mkdir(parents=True, exist_ok=True)
        for e in eps:
            (d / f"ep_{e:05d}_latents.npy").write_bytes(b"")
            (d / f"ep_{e:05d}_meta.npz").write_bytes(b"")
    log = root / "stub.log"

    def go(*args, **env):
        return launch(root, *args, PY_UNET=stub, STUB_LOG=log, **env)

    def calls(script="eval_tf.py"):
        if not log.exists():
            return []
        rows = [json.loads(ln) for ln in log.read_text().splitlines()]
        return [r for r in rows if os.path.basename(r["argv"][0]) == script]
    return go, calls, log


def test_it_scores_every_map_then_skips_what_is_complete(root, real):
    go, calls, log = real
    r = go(3, "unet", CKPT="/x/snap_0200000.pt")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SCORE_DISTANCE_DONE" in r.stdout
    tf = calls()
    assert len(tf) == 6 and all(c["cuda"] == "3" for c in tf)
    assert all("--wandb-run" not in c["argv"] for c in tf)
    out = root / "results_spiderman" / "distance_study" / "040-unet-nexttic" / "snap_0200000_ema_ddim10"
    for s, m, _ in MAPS:
        for k in (1, 4):
            d = out / f"{s}_map{m:02d}_h{k}"
            assert (d / "metrics.json").exists()
            key = (d / "score_key.txt").read_text()
            assert "sha256=sha-snap_0200000.pt" in key and f"horizon={k}" in key and "sampler=ddim10" in key
    log.unlink()
    r = go(3, "unet", CKPT="/x/snap_0200000.pt")
    assert r.returncode == 0 and calls() == []                           # everything complete: nothing rerun


def test_a_score_made_under_another_key_is_refused_unless_rescored(root, real):
    go, calls, log = real
    assert go(3, "unet", CKPT="/x/snap_0200000.pt", MAPS="val:3", HORIZONS="1").returncode == 0
    d = root / "results_spiderman" / "distance_study" / "040-unet-nexttic" / "snap_0200000_ema_ddim10" / "val_map03_h1"
    (d / "score_key.txt").write_text("ckpt=something else\n")
    log.unlink()
    r = go(3, "unet", CKPT="/x/snap_0200000.pt", MAPS="val:3", HORIZONS="1")
    assert r.returncode != 0 and "SCORE_DISTANCE_FAILED" in r.stdout + r.stderr and calls() == []
    r = go(3, "unet", CKPT="/x/snap_0200000.pt", MAPS="val:3", HORIZONS="1", RESCORE=1)
    assert r.returncode == 0 and len(calls()) == 1


def test_a_failed_map_is_reported_and_retried_alone(root, real):
    go, calls, log = real
    r = go(3, "unet", CKPT="/x/snap_0200000.pt", HORIZONS="1", STUB_FAIL="val_map03_h1")
    assert r.returncode != 0 and "SCORE_DISTANCE_FAILED" in r.stdout + r.stderr
    d = root / "results_spiderman" / "distance_study" / "040-unet-nexttic" / "snap_0200000_ema_ddim10"
    assert not (d / "val_map03_h1" / "score_key.txt").exists()
    log.unlink()
    r = go(3, "unet", CKPT="/x/snap_0200000.pt", HORIZONS="1")
    assert r.returncode == 0
    assert [os.path.basename(c["argv"][c["argv"].index("--out-dir") + 1]) for c in calls()] == ["val_map03_h1"]


def test_a_split_naming_an_episode_the_space_does_not_hold_is_not_scored(root, real):
    go, calls, _ = real
    os.remove(root / "latents_arnold_eval_pertic" / "unseen2" / "ep_00020_meta.npz")
    r = go(3, "unet", CKPT="/x/snap_0200000.pt", HORIZONS="1")
    assert r.returncode != 0 and "20" in r.stdout + r.stderr
    scored = [os.path.basename(c["argv"][c["argv"].index("--out-dir") + 1]) for c in calls()]
    assert scored == ["val_map02_h1", "val_map03_h1"]


def test_a_checkpoint_without_an_ema_is_refused(root, real):
    go, calls, _ = real
    r = go(3, "unet", CKPT="/x/best.pt", STUB_EMA="0")
    assert r.returncode != 0 and calls() == [] and re.search(r"EMA", r.stderr)
