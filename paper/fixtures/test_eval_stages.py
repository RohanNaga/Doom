"""The evaluation wrapper seals test: validation first, a recorded selection, then test once.

The 2026-09-22 review (H2) found four ways `after_nexttic.sh` could produce a number that should not
exist or that named the wrong weights:

  (a) `CORPORA=val` still rolled out and scored test, because the rollout stage ignored CORPORA;
  (b) validation, test and unseen were scored in one pass, with no selection recorded before test;
  (c) every stage was skipped on file existence, so a result computed for one checkpoint was
      stamped with the next checkpoint's name by the provenance step;
  (d) a corpus was scored on whichever of its episodes happened to be encoded.

The launcher runs FOR REAL against a throwaway root with a stub python, as the other launcher tests
do. The stub records every call and fakes the GPU scripts; `select_checkpoint.py` runs for real on
real (tiny) torch checkpoints, because the selection is what is under test.

    python -m pytest paper/fixtures/test_eval_stages.py -q
"""
import json
import os
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import make_dense_eval_splits as mdes  # noqa: E402
import select_checkpoint  # noqa: E402
from test_after_nexttic_stages import AFTER, RUN, STUB_PY, _calls_to, _tf_calls  # noqa: E402

CKPTS = (("snap_0270000.pt", 270000), ("snap_0280000.pt", 280000),
         ("snap_0290000.pt", 290000), ("0300000.pt", 300000))


def root_with(tmp_path, checkpoints=CKPTS, best=True):
    """A data root with a run directory holding real checkpoints that all carry an EMA."""
    root = tmp_path / "root"
    r = root / "results_spiderman" / RUN
    r.mkdir(parents=True)
    (root / "repo").mkdir(parents=True)
    for c in ("val", "test", "arenas_678"):
        (root / "latents_arnold_dense_pertic_eval" / c).mkdir(parents=True)
    for name, step in checkpoints:
        torch.save({"model": {"w": torch.full((4,), float(step))}, "ema": {"w": torch.zeros(4)},
                    "step": step}, str(r / name))
    if best:
        torch.save({"model": {"w": torch.zeros(4)}, "step": 41000, "val_loss": 0.2}, str(r / "best.pt"))
    return root, r


def run(tmp_path, root, r, *args, timeout=120, **env):
    py = tmp_path / "stubpy"
    py.write_text(STUB_PY)
    py.chmod(0o755)
    log = tmp_path / "stub.log"
    if log.exists():
        log.unlink()
    import rollout_eval
    e = {**os.environ, "DOOM_ROOT": str(root), "PY": str(py), "PY_SD35": str(py),
         "STUB_LOG": str(log), "STUB_PICK": str(r / "0300000.pt"),
         "STUB_SCORE_FILE": rollout_eval.SCORE_FILE, "STUB_REAL_PY": sys.executable,
         "STUB_REPO": REPO, "NUM_WINDOWS": "8", "SELECT_WINDOWS": "4", **env}
    proc = subprocess.run(["bash", AFTER, "0", "unet", *args], capture_output=True, text=True,
                          env=e, timeout=timeout)
    calls = log.read_text().splitlines() if log.exists() else []
    return proc, calls


def selected(tmp_path, root, r, scores=None):
    """Run the selection stage and return (proc, calls, selection.json)."""
    env = {"STUB_SCORES": json.dumps(scores or {})}
    proc, calls = run(tmp_path, root, r, "--select", **env)
    sel = r / "selection.json"
    return proc, calls, (json.loads(sel.read_text()) if sel.exists() else None)


def flag(call, name):
    t = call.split()
    return t[t.index(name) + 1] if name in t else None


def corpus_of(call):
    """The corpus a scoring call reads: the basename of its --latents-dir."""
    return os.path.basename(flag(call, "--latents-dir") or "")


# ---------------------------------------------------------------------------------------
# (a) CORPORA=val means val only
# ---------------------------------------------------------------------------------------

def test_corpora_val_scores_validation_and_nothing_sealed(tmp_path):
    root, r = root_with(tmp_path)
    proc, calls = run(tmp_path, root, r, CORPORA="val", CKPT=str(r / "0300000.pt"))
    assert proc.returncode == 0, proc.stderr
    assert _calls_to(calls, "rollout_eval.py") == [], "CORPORA=val rolled out on test"
    assert _calls_to(calls, "fvd.py") == []
    assert _tf_calls(calls) and {corpus_of(c) for c in _tf_calls(calls)} == {"val"}


def test_the_default_is_the_validation_stage(tmp_path):
    root, r = root_with(tmp_path)
    proc, calls = run(tmp_path, root, r, CKPT=str(r / "0300000.pt"))
    assert proc.returncode == 0, proc.stderr
    assert "stage=val" in proc.stdout
    assert {corpus_of(c) for c in _tf_calls(calls)} == {"val"}


def test_mixing_val_with_a_sealed_corpus_is_refused(tmp_path):
    root, r = root_with(tmp_path)
    proc, calls = run(tmp_path, root, r, CORPORA="val test", CKPT=str(r / "0300000.pt"))
    assert proc.returncode == 2 and "mixes val with sealed" in proc.stderr
    assert _tf_calls(calls) == []


def test_an_unknown_corpus_is_refused(tmp_path):
    root, r = root_with(tmp_path)
    proc, _ = run(tmp_path, root, r, CORPORA="valid")
    assert proc.returncode == 2 and "unknown corpus" in proc.stderr


# ---------------------------------------------------------------------------------------
# (b) the selection step, then test once
# ---------------------------------------------------------------------------------------

def test_the_candidates_are_the_last_stable_checkpoint_and_two_snapshots_before_it(tmp_path):
    _, r = root_with(tmp_path)
    got = select_checkpoint.candidates(str(r))
    assert [c["step"] for c in got] == [300000, 290000, 280000]
    assert os.path.basename(got[0]["path"]) == "0300000.pt"
    assert all(len(c["sha256"]) == 64 for c in got)


def test_a_snapshot_without_an_ema_is_not_a_candidate(tmp_path):
    _, r = root_with(tmp_path, checkpoints=(("snap_0290000.pt", 290000),))
    torch.save({"model": {"w": torch.zeros(4)}, "step": 280000}, str(r / "snap_0280000.pt"))
    assert [c["step"] for c in select_checkpoint.candidates(str(r))] == [290000]


def test_the_rule_is_raw_psnr_with_an_lpips_check():
    rows = [{"step": 300000, "variant": "live", "psnr_raw": 22.0, "lpips_raw": 0.300},
            {"step": 300000, "variant": "ema", "psnr_raw": 21.9, "lpips_raw": 0.250},
            {"step": 290000, "variant": "ema", "psnr_raw": 21.5, "lpips_raw": 0.255}]
    chosen, eligible = select_checkpoint.choose(rows)
    # the 22.0 dB row fails the LPIPS check (0.300 > 0.250 + 0.01), so the best PSNR that passes wins
    assert (chosen["step"], chosen["variant"]) == (300000, "ema")
    assert {(r["step"], r["variant"]) for r in eligible} == {(300000, "ema"), (290000, "ema")}


def test_ties_go_to_lower_lpips_then_later_step_then_live():
    a = {"step": 1, "variant": "ema", "psnr_raw": 20.0, "lpips_raw": 0.2}
    b = {"step": 2, "variant": "ema", "psnr_raw": 20.0, "lpips_raw": 0.2}
    c = {"step": 2, "variant": "live", "psnr_raw": 20.0, "lpips_raw": 0.2}
    assert select_checkpoint.choose([a, b, c])[0] is c
    d = {"step": 1, "variant": "live", "psnr_raw": 20.0, "lpips_raw": 0.19}
    assert select_checkpoint.choose([a, b, c, d])[0] is d


def test_select_scores_every_candidate_live_and_ema_on_val_and_writes_selection(tmp_path):
    root, r = root_with(tmp_path)
    scores = {"snap_0290000.pt|ema": [22.5, 0.24], "0300000.pt|live": [23.0, 0.40]}
    proc, calls, sel = selected(tmp_path, root, r, scores)
    assert proc.returncode == 0, proc.stderr
    tf = _tf_calls(calls)
    assert len(tf) == 6, tf                                    # three checkpoints x live, EMA
    assert all("--num-windows 4" in c for c in tf)
    assert {corpus_of(c) for c in tf} == {"val"}
    assert sel["chosen"]["path"].endswith("snap_0290000.pt") and sel["chosen"]["variant"] == "ema"
    assert sel["rule"]["name"] == select_checkpoint.RULE["name"]
    assert len(sel["scores"]) == 6 and sel["meta"]["windows"] == "4"
    assert "AFTER_NEXTTIC_SELECTED" in proc.stdout


def test_a_selection_is_made_once(tmp_path):
    root, r = root_with(tmp_path)
    selected(tmp_path, root, r)
    proc, calls, _ = selected(tmp_path, root, r)
    assert proc.returncode == 4 and "a selection is made once" in proc.stderr
    assert _tf_calls(calls) == []


def test_the_sealed_stage_refuses_to_run_without_a_selection(tmp_path):
    root, r = root_with(tmp_path)
    proc, calls = run(tmp_path, root, r, CORPORA="test")
    assert proc.returncode == 4 and "selection.json" in proc.stderr
    assert _tf_calls(calls) == [] and _calls_to(calls, "rollout_eval.py") == []


def test_the_sealed_stage_scores_only_the_selected_checkpoint_and_variant(tmp_path):
    root, r = root_with(tmp_path)
    selected(tmp_path, root, r, {"snap_0280000.pt|ema": [24.0, 0.20]})
    proc, calls = run(tmp_path, root, r, CORPORA="test arenas_678")
    assert proc.returncode == 0, proc.stderr
    scored = _tf_calls(calls) + [c for c in _calls_to(calls, "rollout_eval.py") if c.split()[1] == "--rollout"]
    assert scored
    for c in scored:
        assert flag(c, "--ckpt") == str(r / "snap_0280000.pt"), c
        assert "--use-ema" in c, c
    assert {flag(c, "--out-dir") for c in _tf_calls(calls)} == {
        str(r / "eval_tf_test"), str(r / "eval_tf_test_h4"),
        str(r / "eval_tf_arenas_678"), str(r / "eval_tf_arenas_678_h4")}
    # test is rolled out, unseen is not
    rolled = [c for c in _calls_to(calls, "rollout_eval.py") if c.split()[1] == "--rollout"]
    assert [corpus_of(c) for c in rolled] == ["test"]


def test_a_sealed_corpus_is_scored_once(tmp_path):
    root, r = root_with(tmp_path)
    selected(tmp_path, root, r)
    first, _ = run(tmp_path, root, r, CORPORA="test")
    assert first.returncode == 0, first.stderr
    record = (r / "test_scored_at").read_text()
    assert "test started" in record and "test done" in record and "selection_sha256=" in record
    again, calls = run(tmp_path, root, r, CORPORA="test", RESCORE="1")
    assert again.returncode != 0 and "already scored once" in again.stderr
    assert _tf_calls(calls) == [] and _calls_to(calls, "rollout_eval.py") == []
    forced, calls = run(tmp_path, root, r, CORPORA="test", RESCORE="1", FORCE_TEST="1")
    assert forced.returncode == 0, forced.stderr
    assert _tf_calls(calls), "FORCE_TEST=1 must allow a second scoring"
    assert "forced=1" in (r / "test_scored_at").read_text()


def test_no_selection_after_a_test_score(tmp_path):
    root, r = root_with(tmp_path)
    selected(tmp_path, root, r)
    run(tmp_path, root, r, CORPORA="test")
    proc, calls, _ = selected(tmp_path, root, r)
    assert proc.returncode == 4 and "selection on test" in proc.stderr
    proc, _ = run(tmp_path, root, r, "--select", FORCE_SELECT="1")
    assert proc.returncode == 4, "FORCE_SELECT must not reopen a selection once test was scored"


def test_ckpt_and_step_are_validation_only(tmp_path):
    root, r = root_with(tmp_path)
    selected(tmp_path, root, r)
    proc, calls = run(tmp_path, root, r, CORPORA="test", CKPT=str(r / "snap_0270000.pt"))
    assert proc.returncode == 2 and "validation stage only" in proc.stderr
    assert _tf_calls(calls) == []


def test_a_selected_file_that_changed_on_disk_is_refused(tmp_path):
    root, r = root_with(tmp_path)
    _, _, sel = selected(tmp_path, root, r)
    path = sel["chosen"]["path"]
    torch.save({"model": {"w": torch.ones(4)}, "ema": {"w": torch.ones(4)}, "step": sel["chosen"]["step"]}, path)
    os.remove(path + ".sha256")
    proc, calls = run(tmp_path, root, r, CORPORA="test")
    assert proc.returncode == 4 and _tf_calls(calls) == []
    with pytest.raises(SystemExit, match="not the file that was selected"):
        select_checkpoint.show(str(r / "selection.json"))


def test_a_score_labelled_for_another_file_is_not_used_to_select(tmp_path):
    _, r = root_with(tmp_path)
    cands = select_checkpoint.candidates(str(r))
    for c in cands:
        for v in ("live", "ema"):
            d = r / "select" / select_checkpoint.score_dir_name(c) / ("eval_tf_val" + ("_ema" if v == "ema" else ""))
            d.mkdir(parents=True)
            (d / "metrics.json").write_text(json.dumps({
                "psnr_raw": {"mean": 20.0}, "lpips_raw": {"mean": 0.3},
                "config": {"ckpt": str(r / "snap_0270000.pt"), "use_ema": v == "ema", "step": c["step"]}}))
    rows, problems = select_checkpoint.load_scores(cands, str(r / "select"))
    assert rows == [] and len(problems) == 6


def test_the_rollout_score_stage_is_gated_on_the_file_it_writes(tmp_path):
    """Moved from test_after_nexttic_stages.py: the score gate tested `metrics.json`, a file the
    stage never writes, so the 256-decode pass reran on every invocation."""
    import rollout_eval
    root, r = root_with(tmp_path)
    selected(tmp_path, root, r)
    first, _ = run(tmp_path, root, r, CORPORA="test")
    assert first.returncode == 0, first.stderr
    out = r / "rollout_metrics_test" / rollout_eval.SCORE_FILE
    assert out.is_file() and json.load(open(out))["idm"] == 0.5
    assert (r / "rollout_metrics_test" / (rollout_eval.SCORE_FILE + ".key")).is_file()
    again, calls = run(tmp_path, root, r, CORPORA="test", FORCE_TEST="1")
    assert again.returncode == 0, again.stderr
    assert [c for c in _calls_to(calls, "rollout_eval.py") if "--score" in c] == []
    redo, calls = run(tmp_path, root, r, CORPORA="test", FORCE_TEST="1", RESCORE="1")
    assert [c for c in _calls_to(calls, "rollout_eval.py") if "--score" in c]


def test_a_failed_rollout_is_not_reported_as_done(tmp_path):
    root, r = root_with(tmp_path)
    selected(tmp_path, root, r)
    proc, _ = run(tmp_path, root, r, CORPORA="test", STUB_ROLL_RC="1")
    assert proc.returncode != 0 and "AFTER_NEXTTIC_DONE" not in proc.stdout
    record = (r / "test_scored_at").read_text()
    assert "test started" in record and "test done" not in record


# ---------------------------------------------------------------------------------------
# (c) caches keyed by checkpoint identity
# ---------------------------------------------------------------------------------------

def test_a_changed_checkpoint_under_the_same_name_is_rescored(tmp_path):
    root, r = root_with(tmp_path)
    ck = str(r / "0300000.pt")
    run(tmp_path, root, r, CKPT=ck, STUB_SHA="aaaa")
    same, calls = run(tmp_path, root, r, CKPT=ck, STUB_SHA="aaaa")
    assert _tf_calls(calls) == [], "an unchanged checkpoint was rescored"
    changed, calls = run(tmp_path, root, r, CKPT=ck, STUB_SHA="bbbb")
    assert changed.returncode == 0
    outs = {flag(c, "--out-dir") for c in _tf_calls(calls)}
    assert str(r / "eval_tf_val") in outs and str(r / "eval_tf_val_ema") in outs
    assert "sha256=bbbb" in (r / "eval_tf_val" / "metrics.json.key").read_text()
    assert "sha256=bbbb" in (r / "eval_tf_val" / "decoder_provenance.txt").read_text()


def test_a_result_with_no_key_is_not_trusted(tmp_path):
    """The old cache: any metrics.json at all counted as this checkpoint's score."""
    root, r = root_with(tmp_path)
    out = r / "eval_tf_val"
    out.mkdir()
    (out / "metrics.json").write_text('{"psnr": 99}')
    _, calls = run(tmp_path, root, r, CKPT=str(r / "0300000.pt"))
    assert [c for c in _tf_calls(calls) if c.endswith("eval_tf_val")]


def test_a_different_step_is_a_different_key(tmp_path):
    root, r = root_with(tmp_path)
    run(tmp_path, root, r, CKPT=str(r / "0300000.pt"), STUB_SHA="same")
    _, calls = run(tmp_path, root, r, CKPT=str(r / "snap_0290000.pt"), STUB_SHA="same")
    assert [c for c in _tf_calls(calls) if c.endswith("eval_tf_val")]
    assert "step=290000" in (r / "eval_tf_val" / "metrics.json.key").read_text()


def test_a_failed_rerun_leaves_no_result_that_looks_current(tmp_path):
    root, r = root_with(tmp_path)
    run(tmp_path, root, r, CKPT=str(r / "0300000.pt"), STUB_SHA="aaaa")
    proc, _ = run(tmp_path, root, r, CKPT=str(r / "0300000.pt"), STUB_SHA="bbbb", STUB_TF_RC="1",
                  STUB_TF_NO_OUTPUT="1")
    assert proc.returncode != 0
    assert not (r / "eval_tf_val" / "metrics.json.key").exists()
    assert not (r / "eval_tf_val" / "metrics.json").exists()


def test_the_launcher_no_longer_caches_on_existence_alone():
    text = open(AFTER).read()
    assert 'should_run() { [ ! -e "$1" ] || [ "$RESCORE" = 1 ]; }' not in text
    assert '[ "$(cat "$1.key" 2>/dev/null)" = "$2" ] && return 1' in text


# ---------------------------------------------------------------------------------------
# (d) a corpus is scored only when it is exactly the expected episodes
# ---------------------------------------------------------------------------------------

def test_a_corpus_that_fails_its_check_is_not_scored(tmp_path):
    root, r = root_with(tmp_path)
    proc, calls = run(tmp_path, root, r, CKPT=str(r / "0300000.pt"), STUB_CHECK_FAIL="val")
    assert proc.returncode != 0 and "corpus val" in proc.stderr
    assert _tf_calls(calls) == []
    check = _calls_to(calls, "make_dense_eval_splits.py")
    assert check and "--expect-ids 6000:6100" in check[0] and "--check-only" in check[0]
    assert "--split-file" in check[0] and "split_val.json" in check[0]


def test_the_unseen_check_uses_the_json_range(tmp_path):
    root, r = root_with(tmp_path)
    selected(tmp_path, root, r)
    _, calls = run(tmp_path, root, r, CORPORA="arenas_678")
    check = [c for c in _calls_to(calls, "make_dense_eval_splits.py") if "arenas_678" in c]
    assert check and "--expect-ids 60:120" in check[0]


def _corpus(tmp_path, ids):
    from pertic_fixtures import held_actions, write_pertic_episode
    lat = str(tmp_path / "eval" / "val")
    for ep in ids:
        write_pertic_episode(lat, ep, held_actions([0] * 4))
    return lat


def test_validate_compares_the_split_file_with_the_expected_ids(tmp_path):
    lat = _corpus(tmp_path, [10, 11, 12])
    path, _ = mdes.build(lat, expect_ids=[10, 11, 12])
    assert mdes.validate(lat, [10, 11, 12], split_file=path)["ok"]
    with open(path) as f:
        s = json.load(f)
    s["val"] = [10, 11]                       # a split file that names fewer episodes than exist
    with open(path, "w") as f:
        json.dump(s, f)
    rep = mdes.validate(lat, [10, 11, 12], split_file=path)
    assert not rep["ok"] and any("split file" in p for p in rep["problems"])


def test_validate_refuses_a_missing_split_file(tmp_path):
    lat = _corpus(tmp_path, [10, 11])
    rep = mdes.validate(lat, [10, 11], split_file=str(tmp_path / "nope.json"))
    assert not rep["ok"] and any("split file" in p for p in rep["problems"])


def test_validate_refuses_a_corpus_holding_the_wrong_range(tmp_path):
    """The case the unseen replacement creates: a directory still holding the old ids."""
    lat = _corpus(tmp_path, [0, 1, 2])
    rep = mdes.validate(lat, [60, 61, 62])
    assert not rep["ok"] and rep["missing"] == [60, 61, 62] and rep["unexpected"] == [0, 1, 2]


# ---------------------------------------------------------------------------------------
# DRY prints the plan of the stage it would run
# ---------------------------------------------------------------------------------------

def dry(tmp_path, *args, **env):
    e = {**os.environ, "DRY": "1", "DOOM_ROOT": str(tmp_path), **env}
    p = subprocess.run(["bash", AFTER, "3", "unet", *args], capture_output=True, text=True, env=e)
    assert p.returncode == 0, p.stderr
    return p.stdout


def test_the_dry_plans_name_their_stage(tmp_path):
    assert "stage=val" in dry(tmp_path)
    sel = dry(tmp_path, "--select")
    assert "stage=select" in sel and "DRY candidates" in sel and "DRY choose" in sel
    sealed = dry(tmp_path, CORPORA="test")
    assert "stage=sealed" in sealed and "DRY require" in sealed and "DRY rollout" in sealed
    assert list(tmp_path.iterdir()) == []
