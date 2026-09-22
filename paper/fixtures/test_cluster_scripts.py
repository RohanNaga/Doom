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
SCRIPTS = [SETUP]


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
