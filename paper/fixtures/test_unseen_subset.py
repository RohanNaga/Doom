"""The unseen scoring subset holds no worker-first episode, and no script carries its own copy of it.

`release/dense_split.json` declared arenas_678 ids 0:60 on 2026-09-21. The 2026-09-22 review measured
51 of those 60 as worker-first episodes (k = 0), the only episodes in which Arnold's weapon-select
requests execute; validation and test hold none. The subset was replaced by 60:120 before any model
was scored on it. These tests pin three things:

  * the json names 60:120 in both places a subset is written, records the replacement in `history`,
    and is disjoint from every measured worker-first id (a static check, no data needed);
  * `make_dense_eval_splits.worker_first_episodes` classifies synthetic recordings by the rule the
    review used (`transitions.button_width_report`'s `inferred_starts`), and the publish step refuses
    a range that contains a k = 0 episode;
  * the launchers read the range from the json, so none of them still says 0:60.

    python -m pytest paper/fixtures/test_unseen_subset.py -q
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import make_dense_eval_splits as mdes  # noqa: E402
from doom_data import parse_episode_ids  # noqa: E402

SPLIT = os.path.join(REPO, "release", "dense_split.json")
DENSE_IDS = os.path.join(REPO, "scripts", "dense_ids.sh")

# measured by the 2026-09-22 review from the raw `buttons` column: ids 0-24 and 26-51 are k = 0,
# 25 has no weapon request, 52-59 are k = 1
MEASURED_WORKER_FIRST = set(range(0, 25)) | set(range(26, 52))


def split():
    with open(SPLIT) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------
# the json
# ---------------------------------------------------------------------------------------

def test_the_unseen_subset_is_60_to_120_everywhere_the_json_names_it():
    s = split()
    assert s["next_tic_runs"]["unseen_ids"] == "60:120"
    assert s["segments"]["arenas_678"]["ranges"]["unseen"] == "60:120"


def test_the_unseen_subset_contains_no_measured_worker_first_episode():
    ids = set(parse_episode_ids(split()["next_tic_runs"]["unseen_ids"]))
    assert not ids & MEASURED_WORKER_FIRST, sorted(ids & MEASURED_WORKER_FIRST)
    # and none of the ids the three recorder starts could have made first episodes (0 to 51)
    assert min(ids) >= 52


def test_the_json_records_the_measured_worker_first_ids_it_avoids():
    s = split()
    recorded = set(parse_episode_ids_multi(s["history"][-1]["worker_first_ids_measured"]))
    assert recorded == MEASURED_WORKER_FIRST
    assert not recorded & set(parse_episode_ids(s["next_tic_runs"]["unseen_ids"]))


def parse_episode_ids_multi(spec):
    """"0:25,26:52" -> the union of the ranges."""
    out = []
    for part in spec.split(","):
        out += parse_episode_ids(part)
    return out


def test_the_history_says_when_0_60_was_declared_and_why_it_was_replaced():
    h = split()["history"]
    declared = [e for e in h if e["date"] == "2026-09-21"]
    replaced = [e for e in h if e["date"] == "2026-09-22"]
    assert declared and "0:60" in declared[0]["what"]
    assert replaced and "60:120" in replaced[0]["what"]
    assert "before any model was scored" in replaced[0]["what"]
    why = replaced[0]["why"]
    for fact in ("worker-first", "k = 0", "button_width_report", "REVIEW_2026-09-22"):
        assert fact in why, fact


def test_the_new_subset_is_twenty_per_map():
    from doom_data import dense_map_counts
    s = split()
    counts = dense_map_counts(s["segments"]["arenas_678"]["maps"],
                              parse_episode_ids(s["next_tic_runs"]["unseen_ids"]))
    assert counts == {6: 20, 7: 20, 8: 20}


# ---------------------------------------------------------------------------------------
# the worker-first checker, on synthetic recordings
# ---------------------------------------------------------------------------------------

def _plain():
    return "100000000"                      # nine characters: no weapon request at all


def _switch(k, j=2):
    """A request for SELECT_WEAPONj after k starts: a 1 at index 9 + 10k + j."""
    s = ["0"] * (9 + 10 * k + j + 1)
    s[0] = "1"
    s[9 + 10 * k + j] = "1"
    return "".join(s)


def _corpus(dirpath, kinds):
    """One parquet per episode id; `kinds[ep]` is k, or None for an episode with no request."""
    from test_decision_only import write_parquet
    os.makedirs(dirpath, exist_ok=True)
    for ep, k in kinds.items():
        rows = [(t, 0, _plain(), 0) for t in range(12)]
        if k is not None:
            rows[5] = (5, 0, _switch(k), 0)
            rows[9] = (9, 0, _switch(k, j=7), 0)
        write_parquet(os.path.join(dirpath, f"ep_{ep:05d}.parquet"), rows, 1)
    return dirpath


def test_the_rule_is_the_reviews_inferred_starts():
    from transitions import button_width_report
    assert button_width_report(np.array([_plain(), _switch(0)]))["inferred_starts"] == 0
    assert button_width_report(np.array([_plain(), _switch(1)]))["inferred_starts"] == 1
    assert button_width_report(np.array([_plain(), _switch(4, j=9)]))["inferred_starts"] == 4
    assert button_width_report(np.array([_plain()]))["inferred_starts"] is None


def test_the_checker_classifies_every_episode(tmp_path):
    d = _corpus(str(tmp_path / "raw"), {60: 1, 61: 0, 62: None, 63: 3})
    r = mdes.worker_first_episodes(d, [60, 61, 62, 63])
    assert r["worker_first"] == [61]
    assert r["unclassifiable"] == [62]
    assert r["k"] == {60: 1, 61: 0, 62: None, 63: 3}


def test_the_checker_rejects_a_range_that_contains_a_k0_episode(tmp_path):
    d = _corpus(str(tmp_path / "raw"), {60: 1, 61: 0, 62: 2})
    bad = mdes.check_no_worker_first(d, [60, 61, 62])
    assert bad and "61" in bad[0] and "worker-first" in bad[0]
    assert mdes.check_no_worker_first(d, [60, 62]) == []


def test_a_missing_recording_is_a_problem_not_a_pass(tmp_path):
    d = _corpus(str(tmp_path / "raw"), {60: 1})
    bad = mdes.check_no_worker_first(d, [60, 61])
    assert bad and "61" in " ".join(bad)


def test_the_publish_step_refuses_a_corpus_holding_a_worker_first_episode(tmp_path):
    """The encoded corpus is valid; only the recording says it is the wrong control regime."""
    from pertic_fixtures import held_actions, write_pertic_episode
    raw = _corpus(str(tmp_path / "raw"), {60: 1, 61: 0})
    lat = str(tmp_path / "eval" / "arenas_678")
    for ep in (60, 61):
        write_pertic_episode(lat, ep, held_actions([0] * 4))
    ok = mdes.validate(lat, [60, 61])
    assert ok["ok"], ok["problems"]
    rep = mdes.validate(lat, [60, 61], refuse_worker_first=raw)
    assert not rep["ok"] and rep["worker_first"] == [61]
    with pytest.raises(SystemExit, match="worker-first"):
        mdes.build(lat, expect_ids=[60, 61], refuse_worker_first=raw)
    assert not os.path.exists(mdes.split_path(lat))


def test_the_cli_takes_the_recording_directory(tmp_path):
    a = mdes.build_parser().parse_args(["--latents-dir", "x", "--refuse-worker-first", "/raw/arenas_678"])
    assert a.refuse_worker_first == "/raw/arenas_678"
    assert mdes.build_parser().parse_args(["--latents-dir", "x"]).refuse_worker_first == ""


# ---------------------------------------------------------------------------------------
# no script carries its own copy of the range
# ---------------------------------------------------------------------------------------

def test_the_shell_parse_of_the_json_agrees_with_json_load(tmp_path):
    nt = split()["next_tic_runs"]
    for key in ("train_ids", "val_ids", "test_ids", "unseen_ids"):
        r = subprocess.run(["bash", "-c", f'. "{DENSE_IDS}"; dense_ids {key}'], capture_output=True,
                           text=True, env={**os.environ, "REPO": REPO})
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == nt[key], key


def test_the_shell_parse_refuses_a_missing_key(tmp_path):
    r = subprocess.run(["bash", "-c", f'. "{DENSE_IDS}"; dense_ids no_such_ids'], capture_output=True,
                       text=True, env={**os.environ, "REPO": REPO})
    assert r.returncode != 0 and "no_such_ids" in r.stderr


def _code_lines(path):
    with open(path) as f:
        return [ln for ln in f.read().splitlines() if not ln.lstrip().startswith("#")]


@pytest.mark.parametrize("rel", ["scripts/spiderman/encode_nexttic.sh", "scripts/spiderman/after_nexttic.sh",
                                 "scripts/cluster/encode_all.sh", "scripts/cluster/fetch_dataset.sh"])
def test_no_launcher_hardcodes_the_old_subset(rel):
    for ln in _code_lines(os.path.join(REPO, rel)):
        assert not re.search(r"(?<![0-9])0:60(?![0-9])", ln), f"{rel}: {ln}"
    assert "dense_ids unseen_ids" in open(os.path.join(REPO, rel)).read(), rel


def test_the_unseen_encode_refuses_worker_first_episodes_at_publish():
    """The publish step runs only after a real encode, so this is read from the source: the unseen
    corpus, and only it, is published with the recording directory to classify against."""
    text = open(os.path.join(REPO, "scripts", "spiderman", "encode_nexttic.sh")).read()
    assert '[ "$3" = unseen ] && WF=(--refuse-worker-first "$1")' in text
    assert '"${WF[@]}"' in text


def test_the_encoder_dry_run_uses_the_new_subset(tmp_path):
    e = {**os.environ, "DRY": "1", "DOOM_ROOT": str(tmp_path), "CORPUS": "unseen"}
    r = subprocess.run(["bash", os.path.join(REPO, "scripts", "spiderman", "encode_nexttic.sh")],
                       capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stderr
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("DRY unseen ")][0]
    assert "--episode-ids 60:120" in line and "raw_arnold_dense/arenas_678" in line


def test_the_cluster_publish_of_the_unseen_corpus_refuses_worker_first(tmp_path):
    e = {**os.environ, "DRY": "1", "DOOM_ROOT": str(tmp_path), "VAES": "sd15,sd35"}
    r = subprocess.run(["bash", os.path.join(REPO, "scripts", "cluster", "encode_all.sh")],
                       capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stderr
    splits = [ln for ln in r.stdout.splitlines() if "make_dense_eval_splits.py" in ln]
    unseen = [ln for ln in splits if "/arenas_678" in ln]
    assert len(unseen) == 2, unseen                       # one per latent space
    for ln in unseen:
        assert "--expect-ids 60:120" in ln
        assert f"--refuse-worker-first {tmp_path}/raw_arnold_dense/arenas_678" in ln
    assert all("--refuse-worker-first" not in ln for ln in splits if ln not in unseen)
