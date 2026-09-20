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
    used = set(re.findall(r"(--[a-z][a-z0-9-]+)", text.split("encode_parquet.py", 1)[1].split("echo", 1)[0]))
    assert used <= known, f"launcher passes flags the encoder does not define: {sorted(used - known)}"


def test_header_states_the_disk_and_time_cost(text):
    head = text.split("set -u", 1)[0]
    assert "61.3 GiB" in head and "22 card-hours" in head
    assert "resume-safe" in head
