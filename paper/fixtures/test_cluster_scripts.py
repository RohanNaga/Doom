"""The rented-node scripts: zero to the two next-tic runs on a fresh 8x H100 box.

`scripts/cluster/` takes a node that has nothing but a CUDA driver and brings it to two training
runs, using only the public repo and the public Hugging Face dataset. Nothing here may touch a real
node: every launcher is exercised through `DRY=1` with `DOOM_ROOT` pointed at a throwaway
directory, which is the rule `test_launcher_safety.py` enforces over this whole directory.

    python -m pytest paper/fixtures/test_cluster_scripts.py -q
"""
import json
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CLUSTER = os.path.join(REPO, "scripts", "cluster")

SETUP = os.path.join(CLUSTER, "setup_node.sh")
FETCH = os.path.join(CLUSTER, "fetch_dataset.sh")
ENCODE = os.path.join(CLUSTER, "encode_all.sh")
GATES = os.path.join(CLUSTER, "gates.sh")
LAUNCH = os.path.join(CLUSTER, "launch_runs.sh")
STATUS = os.path.join(CLUSTER, "status.sh")
REQUIREMENTS = os.path.join(CLUSTER, "requirements.txt")
README = os.path.join(CLUSTER, "README.md")
SCRIPTS = [SETUP, FETCH, ENCODE, GATES, LAUNCH, STATUS]


def dry(script, args=(), **env):
    """Run a cluster script with its side effects disabled and its data root thrown away."""
    root = env.pop("root", "/tmp/cluster-dry-root")
    e = {**os.environ, "DRY": "1", "DOOM_ROOT": root, **env}
    r = subprocess.run(["bash", script, *args], capture_output=True, text=True, env=e)
    assert r.returncode == 0, f"{os.path.basename(script)} DRY exited {r.returncode}: {r.stderr}"
    return r.stdout


def dry_fail(script, args=(), **env):
    """Same, for the paths that are supposed to refuse."""
    root = env.pop("root", "/tmp/cluster-dry-root")
    e = {**os.environ, "DRY": "1", "DOOM_ROOT": root, **env}
    return subprocess.run(["bash", script, *args], capture_output=True, text=True, env=e)


def source_of(path):
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------------------
# every script, the same three safety properties
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("script", SCRIPTS, ids=os.path.basename)
def test_the_scripts_parse(script):
    r = subprocess.run(["bash", "-n", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("script", SCRIPTS, ids=os.path.basename)
def test_dry_touches_nothing_and_prints_a_command(script, tmp_path):
    out = dry(script, root=str(tmp_path))
    assert any(ln.startswith("DRY ") for ln in out.splitlines()), out
    assert list(tmp_path.iterdir()) == [], f"{os.path.basename(script)} wrote under the data root"


@pytest.mark.parametrize("script", SCRIPTS, ids=os.path.basename)
def test_the_data_root_override_reaches_every_printed_command(script, tmp_path):
    out = dry(script, root=str(tmp_path))
    assert str(tmp_path) in out, f"{os.path.basename(script)} ignored DOOM_ROOT:\n{out}"
    assert "/data/doom" not in out, "the default root leaked past the override"


@pytest.mark.parametrize("script", SCRIPTS, ids=os.path.basename)
def test_every_script_runs_under_set_u(script):
    """An unset variable must stop the script, not silently expand to nothing on a rented node."""
    assert re.search(r"^set -u$", source_of(script), re.M), "no `set -u`"


@pytest.mark.parametrize("script", SCRIPTS, ids=os.path.basename)
def test_the_default_root_is_the_nvme_path(script):
    assert "DOOM_ROOT:-/data/doom" in source_of(script)


# ---------------------------------------------------------------------------------------
# setup_node.sh and requirements.txt
# ---------------------------------------------------------------------------------------

def test_setup_builds_the_venv_and_the_checkout_under_the_data_root(tmp_path):
    out = dry(SETUP, root=str(tmp_path))
    assert f"{tmp_path}/env" in out, "the venv is not under DOOM_ROOT"
    assert f"{tmp_path}/repo" in out, "the checkout is not under DOOM_ROOT"
    assert f"{tmp_path}/NODE.json" in out, "NODE.json is not written under DOOM_ROOT"


def test_setup_installs_the_pinned_requirements_file(tmp_path):
    out = dry(SETUP, root=str(tmp_path))
    assert "requirements.txt" in out
    assert "-r " in out, "pip is not given a requirements file"


def test_setup_clones_the_public_repo_at_a_named_commit(tmp_path):
    out = dry(SETUP, root=str(tmp_path), COMMIT="deadbeef")
    assert "github.com/RohanNaga/Doom" in out
    assert "deadbeef" in out, "COMMIT never reached the checkout"


def test_setup_verifies_the_driver_and_a_cuda_tensor_op(tmp_path):
    out = dry(SETUP, root=str(tmp_path))
    assert "nvidia-smi" in out
    assert "torch.cuda" in out, "no CUDA tensor op in the verification step"


def test_setup_only_accepts_python_310_or_311(tmp_path):
    """3.12+ is not what the pinned wheels were chosen for, and the script must say so."""
    src = source_of(SETUP)
    assert "python3.11" in src and "python3.10" in src
    out = dry(SETUP, root=str(tmp_path), PY_BIN="/usr/bin/false")
    assert "3.10" in out or "3.11" in out


def test_requirements_are_all_pinned():
    lines = [ln.strip() for ln in source_of(REQUIREMENTS).splitlines()]
    pkgs = [ln for ln in lines if ln and not ln.startswith("#") and not ln.startswith("--")]
    assert pkgs, "no requirements at all"
    for ln in pkgs:
        assert "==" in ln, f"{ln!r} is not pinned to an exact version"


def test_requirements_cover_everything_the_repo_imports():
    """Every third-party import the encode, train and evaluation paths reach for."""
    text = source_of(REQUIREMENTS)
    for pkg in ("torch", "torchvision", "numpy", "diffusers", "transformers", "accelerate",
                "pyarrow", "pillow", "lpips", "scipy", "timm", "huggingface_hub", "hf_xet",
                "safetensors"):
        assert re.search(rf"^{pkg}\b", text, re.M | re.I), f"{pkg} is not pinned in requirements.txt"


def test_requirements_take_torch_from_a_cuda_wheel_index():
    """A default-index torch on this node would be the CPU build on some platforms."""
    assert "download.pytorch.org/whl/cu" in source_of(REQUIREMENTS)


# ---------------------------------------------------------------------------------------
# fetch_dataset.sh
# ---------------------------------------------------------------------------------------

def plan(out):
    """The `DRY plan <corpus> <dir> <lo:hi> files=N first=.. last=..` lines, by corpus."""
    rows = {}
    for ln in out.splitlines():
        m = re.match(r"DRY plan (\S+) (\S+) (\S+) files=(\d+) first=(\S+) last=(\S+)", ln)
        if m:
            rows[m.group(1)] = dict(dir=m.group(2), ids=m.group(3), files=int(m.group(4)),
                                    first=m.group(5), last=m.group(6))
    return rows


def test_fetch_takes_exactly_the_id_ranges_the_two_runs_need(tmp_path):
    rows = plan(dry(FETCH, root=str(tmp_path)))
    assert set(rows) == {"train", "val", "test", "unseen"}, rows
    assert rows["train"] == dict(dir="arenas", ids="0:2000", files=2000,
                                 first="arenas/ep_00000.parquet", last="arenas/ep_01999.parquet")
    assert rows["val"] == dict(dir="arenas", ids="6000:6100", files=100,
                               first="arenas/ep_06000.parquet", last="arenas/ep_06099.parquet")
    assert rows["test"] == dict(dir="arenas", ids="7000:7100", files=100,
                                first="arenas/ep_07000.parquet", last="arenas/ep_07099.parquet")
    assert rows["unseen"] == dict(dir="arenas_678", ids="0:60", files=60,
                                  first="arenas_678/ep_00000.parquet",
                                  last="arenas_678/ep_00059.parquet")


def test_the_file_list_stops_at_every_range_boundary(tmp_path):
    """A prefix glob would have pulled 100 unseen episodes for the 60 the corpus is; these are
    exact names, so the boundary is the boundary."""
    files = [ln.split(None, 2)[2] for ln in dry(FETCH, root=str(tmp_path), LIST="1").splitlines()
             if ln.startswith("DRY file ")]
    assert len(files) == 2260, len(files)
    assert len(set(files)) == 2260, "the download list repeats a file"
    for present in ("arenas/ep_00000.parquet", "arenas/ep_01999.parquet",
                    "arenas/ep_06000.parquet", "arenas/ep_06099.parquet",
                    "arenas/ep_07000.parquet", "arenas/ep_07099.parquet",
                    "arenas_678/ep_00000.parquet", "arenas_678/ep_00059.parquet"):
        assert present in files, present
    for absent in ("arenas/ep_02000.parquet", "arenas/ep_05999.parquet",
                   "arenas/ep_06100.parquet", "arenas/ep_07100.parquet",
                   "arenas_678/ep_00060.parquet"):
        assert absent not in files, f"{absent} is outside the ranges the runs use"


def test_fetch_downloads_the_metadata_the_encode_depends_on(tmp_path):
    out = dry(FETCH, root=str(tmp_path))
    for name in ("dense_split.json", "canonical_controls.json", "md5_arenas.txt",
                 "md5_arenas_678.txt", "README.md"):
        assert name in out, f"{name} is never fetched"


def test_fetch_writes_where_the_encoder_reads(tmp_path):
    out = dry(FETCH, root=str(tmp_path))
    assert f"--local-dir {tmp_path}/raw_arnold_dense" in out
    assert "--repo-type dataset" in out
    assert "RohanNaga/doom-dense-arnold" in out
    assert "--force" not in out, "a forced re-download is not resumable"


def test_fetch_uses_the_nodes_own_hf_client(tmp_path):
    assert f"{tmp_path}/env/bin/hf " in dry(FETCH, root=str(tmp_path))


def test_every_downloaded_file_is_md5_checked_against_its_manifest(tmp_path):
    out = dry(FETCH, root=str(tmp_path))
    assert f"DRY verify arenas 2200 file(s) against {tmp_path}/raw_arnold_dense/md5_arenas.txt" in out
    assert f"DRY verify arenas_678 60 file(s) against {tmp_path}/raw_arnold_dense/md5_arenas_678.txt" in out
    assert "jobs=" in out, "the md5 pass is not parallel"


def test_a_mismatch_is_loud_and_fatal():
    src = source_of(FETCH)
    assert "FETCH_DATASET_FAILED" in src
    assert "md5sum" in src and "--quiet" in src, "no md5 check at all"
    assert re.search(r"grep -E .*FAILED", src), "mismatches are never listed"


def test_the_rest_of_the_corpus_is_opt_in(tmp_path):
    off = dry(FETCH, root=str(tmp_path))
    assert "FULL=1" in off, "the script never says how to get the rest"
    assert "arenas/*.parquet" not in off
    on = dry(FETCH, root=str(tmp_path), FULL="1")
    assert "arenas/*.parquet" in on and "arenas_678/*.parquet" in on


def test_fetch_reports_bytes_elapsed_and_rate(tmp_path):
    out = dry(FETCH, root=str(tmp_path))
    assert "DRY report bytes" in out and "mb_per_s" in out


# ---------------------------------------------------------------------------------------
# encode_all.sh
# ---------------------------------------------------------------------------------------

def encoder_commands(out):
    """The `encode_parquet.py` commands `encode_nexttic.sh` itself emits under DRY."""
    return [ln for ln in out.splitlines() if "encode_parquet.py" in ln and "--canonical-only" not in ln]


def test_every_encode_uses_the_datasets_canonical_table(tmp_path):
    """The table is a property of the recording, not of the autoencoder: one file, shipped with the
    dataset, passed to every shard and every corpus in both latent spaces. Rebuilding it per shard
    is what the Sep 20 host-memory outage looks like."""
    out = dry(ENCODE, root=str(tmp_path))
    cmds = encoder_commands(out)
    assert cmds, out
    canon = f"--canonical {tmp_path}/raw_arnold_dense/canonical_controls.json"
    for c in cmds:
        assert canon in c, c
    assert "never rebuilt" in out


def test_the_training_corpus_is_sharded_across_the_named_gpus(tmp_path):
    out = dry(ENCODE, root=str(tmp_path), GPUS="0,1,2,3", VAES="sd15")
    train = [c for c in encoder_commands(out) if "--episode-ids 0:2000" in c]
    assert len(train) == 4, train
    for i in range(4):
        assert any(f"--shard {i}/4" in c and f"--device cuda:{i}" in c for c in train), i


def test_the_evaluation_corpora_are_encoded_on_one_card(tmp_path):
    out = dry(ENCODE, root=str(tmp_path), GPUS="0,1,2,3", VAES="sd15", EVAL_GPU="3")
    evals = [c for c in encoder_commands(out) if "--episode-ids 0:2000" not in c]
    assert {"6000:6100", "7000:7100", "0:60"} <= {c.split("--episode-ids ")[1].split()[0] for c in evals}
    assert all("--device cuda:3" in c for c in evals), evals
    assert all("--shard" not in c for c in evals), "an evaluation corpus was sharded"


def test_both_latent_spaces_are_encoded_from_the_same_recording(tmp_path):
    out = dry(ENCODE, root=str(tmp_path), GPUS="0,1", VAES="sd15,sd35")
    cmds = encoder_commands(out)
    sd35 = [c for c in cmds if "stable-diffusion-3.5-medium" in c]
    assert sd35, "no 16-channel encode"
    for c in sd35:
        assert "--latent-channels 16" in c and "--scaling-factor 1.5305" in c and "--shift-factor 0.0609" in c
        assert "latents_arnold_dense_pertic_sd35" in c or "latents_arnold_dense_pertic_eval_sd35" in c
    sd15 = [c for c in cmds if c not in sd35]
    assert sd15 and all("_sd35" not in c for c in sd15)


def test_every_encode_keeps_every_tic(tmp_path):
    for c in encoder_commands(dry(ENCODE, root=str(tmp_path))):
        assert "--every-tic" in c, c


def test_the_node_venv_is_the_interpreter_for_both_spaces(tmp_path):
    for c in encoder_commands(dry(ENCODE, root=str(tmp_path))):
        assert f"{tmp_path}/env/bin/python" in c, c


def test_the_audit_and_the_split_publish_come_after_the_encodes(tmp_path):
    out = dry(ENCODE, root=str(tmp_path), GPUS="0,1", VAES="sd15")
    lines = out.splitlines()
    last_encode = max(i for i, ln in enumerate(lines) if "encode_parquet.py" in ln)
    audit = min(i for i, ln in enumerate(lines) if "check_action_alignment.py" in ln)
    splits = min(i for i, ln in enumerate(lines) if "make_dense_eval_splits.py" in ln)
    assert last_encode < audit < splits, out
    assert "--audit-only" in out
    for ids in ("6000:6100", "7000:7100", "0:60"):
        assert any("make_dense_eval_splits.py" in ln and f"--expect-ids {ids}" in ln for ln in lines), ids


def test_each_shard_gets_its_own_log(tmp_path):
    out = dry(ENCODE, root=str(tmp_path), GPUS="0,1", VAES="sd15")
    for i in range(2):
        assert f"{tmp_path}/logs/encode_sd15_train_shard{i}.log" in out, i
    assert f"{tmp_path}/logs/encode_sd15_evals.log" in out


def test_the_inventory_counts_every_corpus_in_both_spaces(tmp_path):
    out = dry(ENCODE, root=str(tmp_path), VAES="sd15,sd35")
    assert "DRY inventory" in out
    for d in ("latents_arnold_dense_pertic/arenas", "latents_arnold_dense_pertic_eval/val",
              "latents_arnold_dense_pertic_eval/test", "latents_arnold_dense_pertic_eval/arenas_678",
              "latents_arnold_dense_pertic_sd35/arenas", "latents_arnold_dense_pertic_eval_sd35/val"):
        assert f"{tmp_path}/{d}" in out, d


def test_preflight_refuses_when_the_canonical_table_is_not_there(tmp_path):
    """CHECK=1 is the read-only preflight: it must stop rather than let the encoder rebuild a
    per-shard table of its own."""
    e = {**os.environ, "DOOM_ROOT": str(tmp_path), "CHECK": "1", "PY": sys.executable}
    r = subprocess.run(["bash", ENCODE], capture_output=True, text=True, env=e)
    assert r.returncode != 0
    assert "canonical" in (r.stdout + r.stderr).lower(), r.stdout + r.stderr
    assert list(tmp_path.iterdir()) == [], "the preflight wrote under the data root"


# ---------------------------------------------------------------------------------------
# gates.sh
# ---------------------------------------------------------------------------------------

def first_index(lines, needle):
    return min(i for i, ln in enumerate(lines) if needle in ln)


def test_the_gates_run_in_the_order_the_launch_protocol_sets(tmp_path):
    lines = dry(GATES, root=str(tmp_path)).splitlines()
    audit = first_index(lines, "--audit-only")
    align = first_index(lines, "--min-accuracy")
    fit = first_index(lines, "--fit-check")
    smoke = first_index(lines, "--steps 300")
    readback = first_index(lines, "eval_tf.py")
    assert audit < align < fit < smoke < readback, "\n".join(lines)


def test_the_sidecar_audit_is_gate_one_and_covers_both_spaces(tmp_path):
    out = dry(GATES, root=str(tmp_path), VAES="sd15,sd35")
    audits = [ln for ln in out.splitlines() if "--audit-only" in ln]
    assert len(audits) == 2, audits
    assert any("latents_arnold_dense_pertic_eval/val" in ln for ln in audits)
    assert any("latents_arnold_dense_pertic_eval_sd35/val" in ln for ln in audits)
    for ln in audits:
        assert f"--audit-parquet-dir {tmp_path}/raw_arnold_dense/arenas" in ln


def test_the_alignment_gate_carries_the_protocols_thresholds(tmp_path):
    out = dry(GATES, root=str(tmp_path))
    line = [ln for ln in out.splitlines() if "--min-accuracy" in ln][0]
    for flag in ("--min-yaw 0.25", "--min-move 0.05", "--min-rows 1000", "--min-episodes 20",
                 "--min-per-class 100", "--min-accuracy 0.95", "--margin 0.20",
                 "--bootstrap 1000", "--episodes 100", "--seed 0"):
        assert flag in line, f"{flag} missing from the alignment gate"
    assert "latents_arnold_dense_pertic_eval/val" in line, "the gate must score the val corpus"


def test_the_alignment_exit_codes_are_read_the_way_the_scorer_means_them(tmp_path):
    aligned = dry(GATES, root=str(tmp_path), EXPLAIN="0")
    assert "aligned" in aligned and "misaligned" not in aligned
    assert "misaligned" in dry(GATES, root=str(tmp_path), EXPLAIN="2")
    inconclusive = dry(GATES, root=str(tmp_path), EXPLAIN="3")
    assert "inconclusive" in inconclusive
    assert "not approval" in inconclusive, "exit 3 must never read as a pass"


def test_both_backbones_are_fit_checked_at_the_launch_configuration(tmp_path):
    out = dry(GATES, root=str(tmp_path))
    fits = [ln for ln in out.splitlines() if "--fit-check" in ln]
    assert len(fits) == 2, fits
    assert any("--backbone unet" in ln for ln in fits)
    assert any("--backbone sd35" in ln for ln in fits)
    for ln in fits:
        assert "--fit-check 20" in ln, "FIT=20 is what the plan asks for"
        assert "--per-gpu-batch 32" in ln and "--global-batch 32" in ln, "the fit must be the launch batch"
    assert "peak_mem_gb" in out and "steps_per_s" in out, "the fit numbers are never recorded"


def test_the_smoke_writes_to_a_throwaway_results_dir(tmp_path):
    out = dry(GATES, root=str(tmp_path))
    smoke = [ln for ln in out.splitlines() if "--steps 300" in ln]
    assert smoke, out
    for ln in smoke:
        # the launcher names the production directory first; the override has to be the last word
        last = ln.rsplit("--results-dir ", 1)[1].split()[0]
        assert last.startswith(f"{tmp_path}/results_smoke"), last
        assert "results_spiderman" not in last
        for flag in ("--val-every 100", "--ckpt-every 300", "--snapshot-every 300", "--local-snapshots"):
            assert flag in ln, flag


def test_the_smoke_must_leave_a_recovery_checkpoint_and_a_snapshot(tmp_path):
    out = dry(GATES, root=str(tmp_path))
    assert "0000300.pt" in out and "snap_0000300.pt" in out


def test_the_smoke_refuses_to_resume_a_production_run(tmp_path):
    """`launch_nexttic.sh` derives --resume from the production directory, so a smoke run started
    after a launch would continue that run's weights into the throwaway directory."""
    out = dry(GATES, root=str(tmp_path))
    assert f"{tmp_path}/results_spiderman/040-unet-nexttic/log.jsonl" in out
    assert "resume" in out
    assert 'gate_fail "4 smoke' in source_of(GATES)


def test_the_readback_scores_64_val_windows_at_tic_spacing(tmp_path):
    out = dry(GATES, root=str(tmp_path))
    evals = [ln for ln in out.splitlines() if "eval_tf.py" in ln]
    assert evals, out
    for ln in evals:
        assert "--tic-stride 1" in ln and "--num-windows 64" in ln
        assert "snap_0000300.pt" in ln, "the readback must load the snapshot the smoke wrote"
        assert f"--split {tmp_path}/latents_arnold_dense_pertic_eval" in ln
        assert f"--parquet-dir {tmp_path}/raw_arnold_dense/arenas" in ln, "no raw reference, no honest PSNR"


def test_the_sd35_readback_carries_its_own_latent_contract(tmp_path):
    out = dry(GATES, root=str(tmp_path))
    ln = [l for l in out.splitlines() if "eval_tf.py" in l and "--backbone sd35" in l][0]
    assert "--latent-channels 16" in ln
    assert "--latent-scale 1.5305" in ln and "--latent-shift 0.0609" in ln


def test_every_gate_can_stop_the_run(tmp_path):
    src = source_of(GATES)
    assert "GATE_FAILED" in src, "a failure is never announced"
    for gate in ("1 sidecar audit", "2 alignment", "3 fit check", "4 smoke", "5 readback"):
        assert f'gate_fail "{gate}' in src, f"gate {gate} cannot stop the run"
    assert "GATES_GO" in src, "there is no GO summary"
    assert "DRY summary" in dry(GATES, root=str(tmp_path))


# ---------------------------------------------------------------------------------------
# launch_runs.sh
# ---------------------------------------------------------------------------------------

def test_both_rows_launch_on_their_own_card(tmp_path):
    out = dry(LAUNCH, root=str(tmp_path), UNET_GPU="0", SD35_GPU="1")
    cmds = [ln for ln in out.splitlines() if "train_wm.py" in ln]
    assert len(cmds) == 2, cmds
    unet = [c for c in cmds if "--backbone unet" in c][0]
    sd35 = [c for c in cmds if "--backbone sd35" in c][0]
    assert "CUDA_VISIBLE_DEVICES=0" in unet and "CUDA_VISIBLE_DEVICES=1" in sd35
    assert "040-unet-nexttic" in unet and "042-sd35-nexttic" in sd35
    assert "latents_arnold_dense_pertic/arenas" in unet
    assert "latents_arnold_dense_pertic_sd35/arenas" in sd35


def test_the_recipe_is_untouched_and_nothing_accumulates(tmp_path):
    out = dry(LAUNCH, root=str(tmp_path))
    for c in [ln for ln in out.splitlines() if "train_wm.py" in ln]:
        assert "--global-batch 32" in c and "--per-gpu-batch 32" in c
        assert "--tic-stride 1" in c and "--action-history 32" in c
        assert "--lr 5e-5" in c and "--warmup 2000" in c and "--ema-decay 0.9999" in c
        assert "--episode-ids 0:2000" in c and "--val-episode-ids 6000:6100" in c
    assert "ALLOW_ACCUM" not in out, "the launcher's accumulation override must not be set"


def test_a_single_card_per_row_is_explained_not_assumed(tmp_path):
    """GLOBAL=32 with no accumulation means MB * cards = 32: a second card per row halves the
    micro-batch instead of filling both, so the box's spare cards need a recipe decision."""
    out = dry(LAUNCH, root=str(tmp_path))
    assert "DRY note" in out
    note = " ".join(ln for ln in out.splitlines() if ln.startswith("DRY note"))
    assert "32" in note and ("one card" in note or "single card" in note)


def test_the_tmux_sessions_are_named(tmp_path):
    out = dry(LAUNCH, root=str(tmp_path))
    assert "train-unet-nexttic" in out and "train-sd35-nexttic" in out


def test_overlapping_gpu_sets_are_refused(tmp_path):
    r = dry_fail(LAUNCH, root=str(tmp_path), UNET_GPU="2", SD35_GPU="2")
    assert r.returncode != 0
    assert "same" in (r.stdout + r.stderr).lower() or "overlap" in (r.stdout + r.stderr).lower()


def test_workers_follow_the_node(tmp_path):
    assert "--num-workers 7" in dry(LAUNCH, root=str(tmp_path), WORKERS="7")
    default = dry(LAUNCH, root=str(tmp_path))
    n = int(re.search(r"--num-workers (\d+)", default).group(1))
    assert 4 <= n <= 16, f"default worker count {n} is not a sane per-run share of the cores"


# ---------------------------------------------------------------------------------------
# status.sh
# ---------------------------------------------------------------------------------------

def test_status_reads_both_run_logs(tmp_path):
    out = dry(STATUS, root=str(tmp_path))
    assert f"{tmp_path}/results_spiderman/040-unet-nexttic/log.jsonl" in out
    assert f"{tmp_path}/results_spiderman/042-sd35-nexttic/log.jsonl" in out


def write_log(root, run, rows):
    d = root / "results_spiderman" / run
    d.mkdir(parents=True)
    with open(d / "log.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return d


def test_status_reports_step_rate_loss_and_the_last_validation(tmp_path):
    """The real parse, against a log written here: DOOM_ROOT is a throwaway directory and the
    script only reads."""
    write_log(tmp_path, "040-unet-nexttic", [
        {"event": "train", "step": 1000, "loss": 0.31, "steps_per_s": 1.2, "peak_mem_gb": 40.1,
         "lr": 5e-05, "skipped_updates": 0},
        {"event": "val", "step": 1000, "val_loss": 0.3005},
        {"event": "train", "step": 2000, "loss": 0.2712, "steps_per_s": 1.05, "peak_mem_gb": 41.6,
         "lr": 5e-05, "skipped_updates": 2},
    ])
    e = {**os.environ, "DOOM_ROOT": str(tmp_path)}
    r = subprocess.run(["bash", STATUS], capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stderr
    line = [ln for ln in r.stdout.splitlines() if "040-unet-nexttic" in ln][0]
    assert "step=2000" in line
    assert "loss=0.2712" in line
    assert "1.05" in line, "no updates/s"
    assert "val=0.3005@1000" in line or "val_loss=0.3005" in line
    assert "skipped=2" in line
    assert "042-sd35-nexttic" in r.stdout, "a run that has not started must still be reported"


def test_status_reports_disk_and_gpu_memory(tmp_path):
    e = {**os.environ, "DOOM_ROOT": str(tmp_path)}
    r = subprocess.run(["bash", STATUS], capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stderr
    assert "disk" in r.stdout.lower()
    assert "gpu" in r.stdout.lower()


# ---------------------------------------------------------------------------------------
# README.md
# ---------------------------------------------------------------------------------------

def test_the_readme_gives_the_sequence_in_order():
    text = source_of(README)
    order = [text.index(f"scripts/cluster/{n}")
             for n in ("setup_node.sh", "fetch_dataset.sh", "encode_all.sh", "gates.sh",
                       "launch_runs.sh", "status.sh")]
    assert order == sorted(order), "the command sequence is out of order"


def test_the_readme_marks_the_h100_durations_as_estimates():
    text = source_of(README)
    assert "estimate" in text.lower()
    assert "A6000" in text and "H100" in text
    assert "0.455" in text, "the SD 3.5 A6000 rate the estimates multiply is not stated"


def test_the_readme_says_how_to_stop_resume_and_copy_back():
    text = source_of(README)
    assert "tmux kill-session -t train-unet-nexttic" in text
    assert "pkill" in text, "the pkill warning is missing"
    assert "rsync" in text and "128.2.204.110" in text
    assert "ONLY=unet" in text, "no way to resume one row"


def test_the_readme_states_what_a_human_must_confirm_about_the_node():
    text = source_of(README)
    for claim in ("driver", "3.10", "NVMe", "cores"):
        assert claim in text, f"the node assumption about {claim} is not stated"
