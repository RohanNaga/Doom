#!/bin/bash
# Zero to a usable node, step 1 of 5. Takes a freshly rented Ubuntu 22.04/24.04 box with 8x H100
# 80 GB, a CUDA driver and local NVMe, and leaves a virtualenv, a checkout of the public repo at a
# named commit, and a NODE.json describing what it found. Nothing here is specific to our lab: the
# only inputs are the public repo and, later, the public Hugging Face dataset.
#
#   usage: [DOOM_ROOT=/data/doom] [COMMIT=<sha|branch>] [PY_BIN=/usr/bin/python3.11] [REQ=..] \
#          [REPO_URL=..] [DRY=1] setup_node.sh
#
# Idempotent. Re-running keeps the venv, re-runs pip (a no-op when the pins are already installed),
# fetches and moves the checkout to $COMMIT, and rewrites NODE.json. That matters because the node
# is rented: the same script runs again after a reboot or a re-attach of the NVMe.
#
# DOOM_ROOT must be on the local NVMe, not the root disk. Everything this cluster writes lives
# under it: the venv, the checkout, the raw dataset (~130 GB for the subset), both latent corpora
# and the run directories. The root filesystem of a rented node is typically small and shared with
# the image.
#
# Python 3.10 or 3.11. The pins in requirements.txt exist for both; the CUDA wheels are cu124,
# which covers H100 (sm90) and needs driver >= 550.
#
# COMMIT pins the code. Pass a full sha for a reproducible node: `launch_nexttic.sh` runs
# `git pull -q` of its own before launching, and a detached HEAD at a sha makes that pull a no-op
# instead of a silent code change between the smoke gate and the launch.
#
# DRY=1 prints every command this would run and touches nothing; DOOM_ROOT repoints the data root.
set -u
D=${DOOM_ROOT:-/data/doom}
DRY=${DRY:-0}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_URL=${REPO_URL:-https://github.com/RohanNaga/Doom.git}
COMMIT=${COMMIT:-main}
REQ=${REQ:-$HERE/requirements.txt}
VENV=$D/env
REPO=$D/repo
NODE=$D/NODE.json
PY=$VENV/bin/python

die() { echo "SETUP_NODE_FAILED $*" >&2; exit 1; }

# --- the interpreter -------------------------------------------------------------------
# A rented image usually ships one python3; 3.10 is Ubuntu 22.04's and 3.11 is available from the
# distribution. Anything else is refused rather than silently used, because the wheel pins were
# chosen for those two.
usable() { "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] in ((3,10),(3,11)) else 1)' 2>/dev/null; }
PICK=""
for C in ${PY_BIN:-} python3.11 python3.10; do
  [ -n "$C" ] || continue
  command -v "$C" >/dev/null 2>&1 || [ -x "$C" ] || continue
  if usable "$C"; then PICK=$C; break; fi
done

if [ "$DRY" = 1 ]; then
  echo "DRY setup root=$D venv=$VENV repo=$REPO commit=$COMMIT"
  if [ -n "$PICK" ]; then echo "DRY python interpreter $PICK"
  else echo "DRY python none of ${PY_BIN:-} python3.11 python3.10 is 3.10 or 3.11 (set PY_BIN)"; fi
  echo "DRY venv ${PICK:-python3.11} -m venv $VENV"
  echo "DRY pip $PY -m pip install --upgrade pip wheel"
  echo "DRY pip $PY -m pip install -r $REQ"
  echo "DRY clone git clone $REPO_URL $REPO"
  echo "DRY checkout git -C $REPO fetch --prune origin && git -C $REPO checkout --detach $COMMIT"
  echo "DRY verify nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader"
  echo "DRY verify $PY -c 'import torch; assert torch.cuda.is_available(); (torch.randn(1024,1024,device=\"cuda\",dtype=torch.bfloat16) @ ...)'"
  echo "DRY versions $PY -c 'import torch, diffusers, transformers, accelerate, pyarrow, lpips'"
  echo "DRY node $NODE"
  exit 0
fi

[ -n "$PICK" ] || die "no python 3.10 or 3.11 on this node (tried ${PY_BIN:-} python3.11 python3.10); install one or set PY_BIN"
[ -f "$REQ" ] || die "no requirements file at $REQ"
mkdir -p "$D" || die "cannot write under $D (is the NVMe mounted?)"

# --- venv ------------------------------------------------------------------------------
if [ -x "$PY" ]; then
  echo "venv already at $VENV ($("$PY" -V 2>&1))"
else
  "$PICK" -m venv "$VENV" || die "venv creation failed"
fi
"$PY" -m pip install --upgrade pip wheel >/dev/null || die "pip self-upgrade failed"
"$PY" -m pip install -r "$REQ" || die "pip install -r $REQ failed"

# --- checkout --------------------------------------------------------------------------
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" fetch --prune origin || die "git fetch failed in $REPO"
else
  git clone "$REPO_URL" "$REPO" || die "git clone failed"
fi
# a sha checks out directly; a branch name has to go through its remote-tracking ref, or a stale
# local branch would win and pin the node to yesterday's code
if git -C "$REPO" rev-parse --verify --quiet "$COMMIT^{commit}" >/dev/null; then
  git -C "$REPO" checkout --detach "$COMMIT" || die "checkout of $COMMIT failed"
else
  git -C "$REPO" checkout --detach "origin/$COMMIT" || die "checkout of origin/$COMMIT failed"
fi
SHA=$(git -C "$REPO" rev-parse HEAD)

# --- the card --------------------------------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 || die "no nvidia-smi: this node has no usable driver"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || die "nvidia-smi failed"
"$PY" - <<'PY' || die "CUDA tensor op failed: torch cannot use these cards"
import math, torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
n = torch.cuda.device_count()
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = float((x @ x).float().abs().sum())
assert math.isfinite(y) and y > 0, f"bf16 matmul produced {y}"
print(f"cuda ok: {n} device(s), torch {torch.__version__}, capability {torch.cuda.get_device_capability(0)}")
PY

# --- versions and NODE.json ------------------------------------------------------------
"$PY" - "$NODE" "$SHA" "$REPO" <<'PY' || die "NODE.json could not be written"
import json, os, platform, subprocess, sys, torch
node, sha, repo = sys.argv[1], sys.argv[2], sys.argv[3]
def ver(name):
    try:
        return __import__(name).__version__
    except Exception as e:                       # a missing optional package is a fact, not a crash
        return f"missing ({e.__class__.__name__})"
smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout
gpus = [dict(zip(("index", "name", "memory_total", "driver"), [c.strip() for c in ln.split(",")]))
        for ln in smi.splitlines() if ln.strip()]
info = {
    "hostname": platform.node(),
    "kernel": platform.release(),
    "python": sys.version.split()[0],
    "cores": os.cpu_count(),
    "gpus": gpus,
    "driver": gpus[0]["driver"] if gpus else None,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "device_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    "versions": {p: ver(p) for p in ("diffusers", "transformers", "accelerate", "numpy",
                                     "pyarrow", "PIL", "lpips", "scipy", "timm", "huggingface_hub")},
    "repo": repo,
    "commit": sha,
}
tmp = node + ".tmp"
with open(tmp, "w") as f:
    json.dump(info, f, indent=1)
os.replace(tmp, node)                            # atomic: a half-written NODE.json is worse than none
print(json.dumps(info, indent=1))
PY
echo "SETUP_NODE_DONE root=$D commit=$SHA node=$NODE"
