"""The launch gate: does the control on row t explain the motion between frames t and t+1?

A one-tic error in the action conditioning changes no loss curve and no PSNR number, so this script
is the only thing between a wrong convention and a multi-day run. It has to fail loudly, which means
the tests below are mostly about the ways a naive version passes when it should not. Two such ways
were found by Astra on Sep 20 2026 and both have a regression test here:

* dropping the rows where the *shifted* control is a no-op scores each shift on a different row set,
  and on the pattern Arnold's action set actually produces (no-turn, left, no-turn, right, four tics
  each) all three shifts then scored a perfect 1.000;
* "the best shift wins by a margin" is not a pass: a stale-control injection scored 1.000 at shift 0
  and 1.000 at shift +1 and came back "inconclusive" with exit code 0.

    python -m pytest paper/fixtures/test_action_alignment.py -q
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import check_action_alignment as caa  # noqa: E402

BITS = 9
TEST_FLOOR = dict(min_rows=8, draws=200)   # the real gate wants 1,000 rows; a synthetic episode has fewer
ALTERNATING = [1, -1]                      # turn left, turn right
WITH_NO_TURN = [0, 1, 0, -1]               # the pattern Arnold's action set actually produces


def _episode(pattern, shift=0, reps=40, step=5.0, deaths=None):
    """A recording whose row t carries the control that turns the agent between t and t+1.

    `pattern` is one decision's turn intent, each held for 4 tics (Arnold's action repeat).
    `shift` injects an error of that many tics into the CONTROLS only: the recorded state is
    untouched, so a correct gate must still name shift 0 the truth at shift = 0 and blame the
    injected offset otherwise. shift = +1 is the stale-buffer / record-after-stepping case.
    """
    intent = np.tile(np.repeat(np.asarray(pattern, dtype=int), 4), reps)
    n = len(intent)
    angle = np.zeros(n)
    for t in range(n - 1):
        angle[t + 1] = angle[t] + step * intent[t]
    controls = np.zeros((n, BITS), dtype=np.float32)
    src = np.roll(intent, shift)
    controls[src > 0, caa.TURN_LEFT] = 1.0
    controls[src < 0, caa.TURN_RIGHT] = 1.0
    d = np.zeros(n, dtype=np.int64) if deaths is None else np.asarray(deaths)
    return controls, angle % 360.0, np.arange(n, dtype=np.int64), d


def _yaw(controls, angle, tic, deaths):
    return caa.alignment_scores(controls, angle, tic, deaths)["yaw"]


def _write_latents(d, controls, angle, tic, deaths):
    from pertic_fixtures import write_pertic_episode
    btns = ["".join(str(int(v)) for v in row) for row in controls]
    write_pertic_episode(d, 0, np.zeros(len(btns), dtype=np.int64), buttons=btns,
                         tics=tic, deaths=deaths)
    meta_path = os.path.join(d, "ep_00000_meta.npz")
    meta = dict(np.load(meta_path))
    meta["angle"] = angle
    np.savez(meta_path, **meta)


# ---------------------------------------------------------------------------------------
# it passes a correctly aligned corpus
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("pattern", [ALTERNATING, WITH_NO_TURN], ids=["alternating", "with_no_turn"])
def test_the_unshifted_control_wins_on_a_correctly_aligned_episode(pattern):
    sc = _yaw(*_episode(pattern))
    v = caa.verdict(sc, **TEST_FLOOR)
    assert v["verdict"] == "aligned", v
    assert sc["shifts"]["0"] == 1.0, sc["shifts"]


def test_a_correctly_aligned_corpus_exits_zero(tmp_path):
    d = str(tmp_path / "good")
    _write_latents(d, *_episode(WITH_NO_TURN))
    assert caa.main(caa.build_parser().parse_args(
        ["--latents-dir", d, "--min-rows", "8", "--bootstrap", "200"])) == caa.EXIT_ALIGNED


# ---------------------------------------------------------------------------------------
# the two ways a naive gate passed when it should not have
# ---------------------------------------------------------------------------------------

def test_the_no_turn_pattern_no_longer_scores_perfectly_on_every_shift():
    """Astra's first finding: each shift was being scored on a different row set."""
    sc = _yaw(*_episode(WITH_NO_TURN))
    assert sc["shifts"]["0"] == 1.0
    assert sc["shifts"]["-1"] < 1.0 and sc["shifts"]["1"] < 1.0, sc["shifts"]
    assert sc["per_class"]["0"] > 0, "no-turn must be a class, not a skipped row"


def test_the_stale_control_injection_fails_with_a_nonzero_exit_code(tmp_path):
    """Astra's second finding: the exact bug the gate exists to catch used to exit 0."""
    d = str(tmp_path / "stale")
    _write_latents(d, *_episode(WITH_NO_TURN, shift=1))
    code = caa.main(caa.build_parser().parse_args(
        ["--latents-dir", d, "--min-rows", "8", "--bootstrap", "200"]))
    assert code == caa.EXIT_MISALIGNED != 0


@pytest.mark.parametrize("shift", [-1, 1])
def test_an_injected_off_by_one_is_named_and_signed(shift):
    v = caa.verdict(_yaw(*_episode(WITH_NO_TURN, shift=shift)), **TEST_FLOOR)
    assert v["verdict"] == "misaligned", v
    assert v["best"] == shift, v
    assert "stored" in v["reason"]


def test_one_shift_scoring_a_single_row_is_not_a_pass():
    """`second = -inf` used to produce "aligned at shift 0" from one row."""
    v = caa.verdict(_yaw(*_episode(WITH_NO_TURN, reps=1)), min_rows=1000)
    assert v["verdict"] == "inconclusive", v


# ---------------------------------------------------------------------------------------
# it refuses to certify weak evidence
# ---------------------------------------------------------------------------------------

def test_too_few_rows_is_inconclusive_and_nonzero(tmp_path):
    d = str(tmp_path / "tiny")
    _write_latents(d, *_episode(WITH_NO_TURN, reps=2))
    code = caa.main(caa.build_parser().parse_args(["--latents-dir", d, "--bootstrap", "200"]))
    assert code == caa.EXIT_INCONCLUSIVE != 0
    v = caa.verdict(_yaw(*_episode(WITH_NO_TURN, reps=2)))
    assert v["verdict"] == "inconclusive" and "below the floor" in v["reason"]


def test_a_weak_signal_is_inconclusive_not_aligned():
    """Random controls: no shift explains the motion, and the gate must certify nothing."""
    rng = np.random.RandomState(0)
    _, angle, tic, deaths = _episode(WITH_NO_TURN)
    controls = np.zeros((len(angle), BITS), dtype=np.float32)
    pick = rng.randint(0, 3, len(controls))
    controls[pick == 1, caa.TURN_LEFT] = 1.0
    controls[pick == 2, caa.TURN_RIGHT] = 1.0
    v = caa.verdict(_yaw(controls, angle, tic, deaths), **TEST_FLOOR)
    assert v["verdict"] == "inconclusive", v


def test_the_gate_has_distinct_nonzero_codes():
    assert caa.EXIT_ALIGNED == 0
    assert caa.EXIT_MISALIGNED != 0 and caa.EXIT_INCONCLUSIVE != 0
    assert caa.EXIT_MISALIGNED != caa.EXIT_INCONCLUSIVE


# ---------------------------------------------------------------------------------------
# the row set
# ---------------------------------------------------------------------------------------

def test_all_three_shifts_are_scored_on_one_identical_row_set():
    sc = _yaw(*_episode(WITH_NO_TURN))
    assert sorted(sc["shifts"]) == ["-1", "0", "1"]
    assert all(np.isfinite(v) for v in sc["shifts"].values())
    assert sum(sc["per_class"].values()) == sc["rows"]


def test_rows_across_a_respawn_are_never_scored():
    """A respawn teleports the agent, so no control explains that frame pair."""
    controls, angle, tic, _ = _episode(WITH_NO_TURN)
    n = len(angle)
    deaths = (np.arange(n) >= n // 2).astype(np.int64)
    angle = angle.copy(); angle[n // 2:] += 137.0            # the teleport
    sc = _yaw(controls, angle, tic, deaths)
    assert sc["rows"] < _yaw(*_episode(WITH_NO_TURN))["rows"], "the life boundary removed no rows"
    assert sc["shifts"]["0"] == 1.0, "a respawn row leaked into the score"


def test_rows_across_a_tic_gap_are_never_scored():
    _, _, tic, deaths = _episode(WITH_NO_TURN)
    tic = tic.copy(); tic[len(tic) // 2:] += 5               # an unrendered stretch
    usable = caa.usable_rows(tic, deaths, len(tic))
    assert not usable[len(tic) // 2 - 1] and not usable[len(tic) // 2]


def test_usable_rows_need_both_neighbours():
    """All three shifts must have a defined control, or they are not scored on the same rows."""
    assert caa.usable_rows(np.arange(6), np.zeros(6), 6).tolist() == \
        [False, True, True, True, True, False]


def test_balanced_accuracy_is_not_carried_by_the_majority_class():
    truth = np.array([0] * 90 + [1] * 10)
    assert caa.balanced_accuracy(truth, np.zeros(100, dtype=int)) == 0.5
    assert caa.balanced_accuracy(truth, truth) == 1.0


def test_the_yaw_difference_wraps_at_three_sixty():
    assert abs(caa.wrap_deg(359.0 - 1.0) - (-2.0)) < 1e-9
    assert abs(caa.wrap_deg(1.0 - 359.0) - 2.0) < 1e-9


# ---------------------------------------------------------------------------------------
# the other axis, the sources, and the report
# ---------------------------------------------------------------------------------------

def test_position_alignment_is_scored_the_same_way():
    intent = np.tile(np.repeat(np.array([0, 1, 0, -1]), 4), 10)
    n = len(intent)
    angle, px = np.zeros(n), np.zeros(n)
    for t in range(n - 1):
        px[t + 1] = px[t] + 10.0 * intent[t]
    controls = np.zeros((n, BITS), dtype=np.float32)
    controls[intent > 0, caa.MOVE_FORWARD] = 1.0
    controls[intent < 0, caa.MOVE_BACKWARD] = 1.0
    sc = caa.alignment_scores(controls, angle, np.arange(n), np.zeros(n), px, np.zeros(n))
    assert sc["position"]["shifts"]["0"] == 1.0
    assert caa.verdict(sc["position"], **TEST_FLOOR)["verdict"] == "aligned"


def test_the_checker_runs_off_a_per_tic_latent_directory(tmp_path):
    """So it can be run where the latents are, without the 1.7 TiB of parquet."""
    d = str(tmp_path / "lat")
    _write_latents(d, *_episode(WITH_NO_TURN))
    per = caa.from_latents(d, episodes=1)
    assert len(per) == 1
    assert caa.verdict(caa.merge(per)["yaw"], **TEST_FLOOR)["verdict"] == "aligned"


def test_pooling_episodes_concatenates_rows_rather_than_averaging_accuracies():
    a = _yaw(*_episode(WITH_NO_TURN, reps=4))
    b = _yaw(*_episode(WITH_NO_TURN, reps=4))
    pooled = caa.merge([("a", {"yaw": a}), ("b", {"yaw": b})])["yaw"]
    assert pooled["rows"] == a["rows"] + b["rows"]
    assert pooled["shifts"]["0"] == 1.0


def test_the_report_names_the_sign_convention(tmp_path, capsys):
    d = str(tmp_path / "conv")
    _write_latents(d, *_episode(WITH_NO_TURN))
    caa.main(caa.build_parser().parse_args(["--latents-dir", d, "--min-rows", "8", "--bootstrap", "200"]))
    report = json.loads(capsys.readouterr().out)
    assert "row t+s" in report["sign_convention"]
    assert "reproduction is s = +1" in report["sign_convention"]


def test_the_parser_defaults_are_the_real_gate_thresholds():
    with pytest.raises(SystemExit):
        caa.build_parser().parse_args([])
    with pytest.raises(SystemExit):
        caa.build_parser().parse_args(["--parquet", "a", "--latents-dir", "b"])
    a = caa.build_parser().parse_args(["--latents-dir", "b"])
    assert (a.min_rows, a.min_accuracy, a.margin) == (1000, 0.95, 0.20)


# ---------------------------------------------------------------------------------------
# the sidecar audit
# ---------------------------------------------------------------------------------------

def _write_pair(tmp_path, side_buttons, raw_buttons, actions=None, tics=None):
    """A per-tic latent directory and the recording it claims to come from."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pertic_fixtures import write_pertic_episode
    n = len(side_buttons)
    tics = np.arange(n, dtype=np.int64) if tics is None else np.asarray(tics)
    acts = np.zeros(n, dtype=np.int64) if actions is None else np.asarray(actions)
    lat = str(tmp_path / "lat")
    write_pertic_episode(lat, 0, acts, buttons=side_buttons, tics=tics)
    raw = str(tmp_path / "raw")
    os.makedirs(raw, exist_ok=True)
    pq.write_table(pa.table({"tic": pa.array(tics, pa.int32()),
                             "buttons": pa.array(list(raw_buttons), pa.string()),
                             "action": pa.array(acts, pa.int32())}),
                   os.path.join(raw, "ep_00000.parquet"))
    return lat, raw


def test_the_audit_confirms_the_sidecar_matches_the_recording(tmp_path):
    pytest.importorskip("pyarrow")
    b = ["100000000"] * 8 + ["001000000"] * 8
    lat, raw = _write_pair(tmp_path, b, b)
    r = caa.audit_sidecar(lat, raw, episodes=1, rows=16)
    assert r["ok"] and r["mismatches"] == 0 and r["rows_checked"] == 16


def test_the_audit_catches_a_sidecar_that_does_not_match(tmp_path):
    pytest.importorskip("pyarrow")
    side = ["100000000"] * 16
    raw = ["100000000"] * 8 + ["000100000"] * 8
    lat, rawdir = _write_pair(tmp_path, side, raw)
    r = caa.audit_sidecar(lat, rawdir, episodes=1, rows=16)
    assert not r["ok"] and r["mismatches"] == 8 and r["problems"]


def test_the_audit_counts_the_anti_stuck_override_rows(tmp_path):
    """The rows on which conditioning on the action id instead of the button vector would be wrong."""
    pytest.importorskip("pyarrow")
    b = ["100000000"] * 12 + ["000011000"] * 4          # the last four are not action 0's canonical vector
    lat, raw = _write_pair(tmp_path, b, b, actions=np.zeros(16, dtype=np.int64))
    json.dump({"0": "100000000"}, open(os.path.join(lat, "canonical_controls.json"), "w"))
    r = caa.audit_sidecar(lat, raw, episodes=1, rows=16)
    assert r["ok"] and r["canonical_table"] and r["anti_stuck_override_rows"] == 4
    assert abs(r["override_fraction"] - 0.25) < 1e-9


def test_a_failing_audit_fails_the_gate(tmp_path, capsys):
    pytest.importorskip("pyarrow")
    side = ["100000000"] * 160
    rawb = ["100000000"] * 80 + ["000100000"] * 80
    lat, rawdir = _write_pair(tmp_path, side, rawb)
    meta_path = os.path.join(lat, "ep_00000_meta.npz")
    meta = dict(np.load(meta_path))
    meta["angle"] = np.zeros(160)
    np.savez(meta_path, **meta)
    code = caa.main(caa.build_parser().parse_args(
        ["--latents-dir", lat, "--audit-parquet-dir", rawdir, "--min-rows", "8", "--bootstrap", "100"]))
    assert code == caa.EXIT_MISALIGNED
    assert json.loads(capsys.readouterr().out)["sidecar_audit"]["ok"] is False
