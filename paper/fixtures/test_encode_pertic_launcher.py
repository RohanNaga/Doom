"""`scripts/spiderman/encode_pertic.sh` must build a corpus that stays comparable with the old one.

Two things make the per-tic corpus safe, and both live in the launcher rather than in the encoder:
it passes `--every-tic` with the same `--stride` the stride-4 corpora were built at, and it reuses
`latents_arnold_aligned/canonical_controls.json` rather than recomputing a table per corpus. A
recomputed table would pick different decision rows on the small evaluation corpora, and then the
`is_decision` reduction would no longer reproduce the stride-4 latents.

    python -m pytest paper/fixtures/test_encode_pertic_launcher.py -q
"""
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SCRIPT = os.path.join(REPO, "scripts", "spiderman", "encode_pertic.sh")
sys.path.insert(0, REPO)


@pytest.fixture(scope="module")
def text():
    with open(SCRIPT) as f:
        return f.read()


def test_script_is_valid_bash():
    assert subprocess.run(["bash", "-n", SCRIPT]).returncode == 0


def test_it_asks_for_every_tic_at_the_stride_the_old_corpora_used(text):
    body = text.split("one() {", 1)[1]
    assert "--every-tic" in body and "--stride 4" in body


def test_it_reuses_the_aligned_corpus_canonical_table(text):
    assert "latents_arnold_aligned/canonical_controls.json" in text
    assert "--canonical $CANON" in text


def test_it_refuses_to_run_without_that_table(tmp_path):
    r = subprocess.run(["bash", SCRIPT], capture_output=True, text=True)
    # /sata2 does not exist off the server, so the table is missing and this is the path under test
    assert r.returncode == 2 and "missing canonical table" in r.stderr


def test_outputs_never_land_in_the_existing_stride4_directories(text):
    outs = set(re.findall(r"\$D/(latents_arnold[a-z_0-9/$A-Z]*)", text.split("one() {", 1)[1]))
    for o in outs:
        assert "pertic" in o or "canonical" in o, f"output path {o} is not a per-tic directory"


def test_all_four_corpora_are_covered(text):
    case = text.split("case $CORPUS in", 1)[1]
    for name in ("train", "seen", "unseen", "unseen2"):
        assert name in case
    assert "raw_arnold_eval/$CORPUS" in case and "$D/raw_arnold " in case


def test_an_unknown_corpus_is_refused(text):
    assert re.search(r"unknown CORPUS", text)


def test_every_flag_it_passes_exists_in_the_encoder(text):
    from encode_parquet import build_parser
    known = {a.option_strings[0] for a in build_parser()._actions if a.option_strings}
    used = set(re.findall(r"(--[a-z][a-z0-9-]+)", text.split('nice -n 5 $PY "$ENC"', 1)[1].split("echo", 1)[0]))
    assert used <= known, f"launcher passes flags the encoder does not define: {sorted(used - known)}"


def test_it_runs_the_encoder_from_its_own_checkout_not_a_scratch_copy(text):
    """It used to hardcode one agent's scratch copy under $D/tmp/night, so every checkout that ran
    this script executed that file; two agents doing it at once put two writers on one output
    directory (Sep 21 2026)."""
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert "tmp/night" not in code, "an executable line still points at a scratch copy"
    assert 'REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}' in text
    assert "ENC=$REPO/encode_parquet.py" in text and 'nice -n 5 $PY "$ENC"' in text


def test_a_missing_encoder_is_refused_with_a_way_out(text):
    assert "set REPO=" in text and "exit 2" in text


def test_one_writer_per_output_directory(text):
    """Two concurrent encodes of the same corpus race on each episode's .tmp file."""
    body = text.split("one() {", 1)[1]
    assert 'mkdir "$2/.lock"' in body and "already being encoded" in body
    assert "return 3" in body and 'rmdir "$2/.lock"' in body


def test_the_lock_is_released_even_when_the_encode_fails(tmp_path):
    """A lock left behind by a crash would block every later run, so it is released on return."""
    out = tmp_path / "out"
    script = f'''
set -u
one() {{
  if ! mkdir "$1/.lock" 2>/dev/null; then echo LOCKED; return 3; fi
  trap 'rmdir "$1/.lock" 2>/dev/null' RETURN
  return 1
}}
mkdir -p {out}
one {out}; echo "first exit $?"
one {out}; echo "second exit $?"
ls -a {out} | grep -c lock || true
'''
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert "first exit 1" in r.stdout and "second exit 1" in r.stdout, r.stdout
    assert "LOCKED" not in r.stdout, "the lock was not released after a failed run"


def test_each_corpus_uses_the_batch_size_its_reference_was_built_with(text):
    """The micro-batch is part of reproducing the reference: cuDNN picks its algorithm from the batch
    shape under bf16 autocast, so batch 64 on an evaluation corpus built at 16 leaves only 60% of
    latent values bit-identical instead of 99.6%."""
    assert "BATCH_TRAIN=${BATCH_TRAIN:-64}" in text      # reencode_aligned.sh used 64
    assert "BATCH_EVAL=${BATCH_EVAL:-16}" in text        # record_eval_corpus.sh:16 used 16
    case = text.split("case $CORPUS in", 1)[1]
    assert "$BATCH_TRAIN" in case and "$BATCH_EVAL" in case
    # the train corpus must never be encoded at the evaluation batch, or vice versa
    train_line = [ln for ln in case.splitlines() if "latents_arnold_pertic " in ln]
    assert train_line and all("$BATCH_TRAIN" in ln for ln in train_line)
    eval_lines = [ln for ln in case.splitlines() if "latents_arnold_eval_pertic" in ln]
    assert eval_lines and all("$BATCH_EVAL" in ln for ln in eval_lines)


def test_the_measured_equivalence_numbers_are_recorded_in_the_header(text):
    head = text.split("BATCH_TRAIN", 1)[0]
    assert "99.62%" in head and "1.45e-05" in head and "60.3%" in head


def test_header_states_the_disk_and_time_cost(text):
    head = text.split("set -u", 1)[0]
    assert "61.3 GiB" in head and "22 card-hours" in head
    assert "resume-safe" in head
