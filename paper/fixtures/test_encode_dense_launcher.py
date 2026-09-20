"""`scripts/spiderman/encode_dense.sh` must ask `encode_parquet.py` for what it claims to produce.

A launcher is where a corpus silently goes wrong: a stride that does not mean what the comment says,
an alignment flag on a mode that cannot align, a flag the encoder does not have. These read the
script and check it against `encode_parquet.py`'s own parser, plus the size arithmetic the mode
choice rests on, which is the reason the every-tic mode is not the default.

    python -m pytest paper/fixtures/test_encode_dense_launcher.py -q
"""
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SCRIPT = os.path.join(REPO, "scripts", "spiderman", "encode_dense.sh")
sys.path.insert(0, REPO)

BYTES_PER_FRAME = 12451      # measured: latents_arnold_aligned, 11 GB / (850 x 1116 frames)
ARENAS_TICS = 8000 * 5100


@pytest.fixture(scope="module")
def text():
    with open(SCRIPT) as f:
        return f.read()


def test_script_is_valid_bash():
    assert subprocess.run(["bash", "-n", SCRIPT]).returncode == 0


def test_the_two_modes_map_to_the_strides_the_comment_promises(text):
    body = text.split("case $MODE in", 1)[1]
    assert re.search(r"decisions\)\s*STRIDE=4", body)
    assert re.search(r"every-tic\)\s*STRIDE=1", body)


def test_only_the_decision_mode_aligns(text):
    """--align-decisions selects decision rows, so it contradicts keeping every tic."""
    body = text.split("case $MODE in", 1)[1]
    decisions, every = body.split("every-tic)", 1)
    assert "--align-decisions" in decisions
    assert "--align-decisions" not in every.split("*)", 1)[0]


def test_alignment_reuses_the_main_corpus_canonical_table(text):
    """A decision must mean the same thing here as in latents_arnold_aligned, or the two cannot pool."""
    assert "--canonical" in text and "latents_arnold_aligned/canonical_controls.json" in text


def test_every_flag_the_launcher_passes_exists_in_the_encoder(text):
    from encode_parquet import __file__ as enc
    with open(enc) as f:
        parser_src = f.read()
    known = set(re.findall(r'p\.add_argument\("(--[a-z0-9-]+)"', parser_src))
    used = set(re.findall(r"(--[a-z][a-z0-9-]+)", text.split("encode_parquet.py", 1)[1]))
    assert used <= known, f"launcher passes flags the encoder does not define: {sorted(used - known)}"


def test_an_unknown_mode_is_refused(tmp_path):
    r = subprocess.run(["bash", SCRIPT], env={**os.environ, "SEGMENT": "arenas", "MODE": "sideways"},
                       capture_output=True, text=True)
    assert r.returncode == 2 and "unknown MODE" in r.stderr


def test_a_missing_segment_is_refused(text):
    r = subprocess.run(["bash", SCRIPT], env={**os.environ, "SEGMENT": "no_such_segment"},
                       capture_output=True, text=True)
    assert r.returncode == 2 and "no such segment directory" in r.stderr


def test_nothing_is_created_before_the_segment_is_validated(text):
    """A typo must not leave an empty latents directory that a later resume would treat as started."""
    body = text.split("case $MODE in", 1)[1]
    assert body.index("no such segment directory") < body.index("mkdir -p")


def test_segment_is_required():
    env = {k: v for k, v in os.environ.items() if k != "SEGMENT"}
    r = subprocess.run(["bash", SCRIPT], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "SEGMENT" in r.stderr


@pytest.mark.parametrize("stride,gib", [(4, 118), (1, 473)])
def test_the_size_arithmetic_in_the_header_is_right(stride, gib):
    """The header's disk figures are why every-tic is not the default; they have to be true."""
    got = (ARENAS_TICS // stride) * BYTES_PER_FRAME / 2**30
    assert abs(got - gib) < 6, f"stride {stride}: {got:.0f} GiB, header says {gib}"


def test_header_states_both_disk_figures(text):
    head = text.split("set -u", 1)[0]
    assert "118 GiB" in head and "473 GiB" in head
    assert "(4, 32, 40)" in head
