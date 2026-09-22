"""`buttons_report.py` measures the recorded button widths, so the Arnold explanation is checkable.

The claim under test is not the normalisation but its cause: Arnold's `add_buttons` appends the ten
`SELECT_WEAPON%i` names to a SHARED `available_buttons` list on every `Game.start()`
(`src/doom/actions.py:197-199`), and `record_arnold.py:167` starts the game once per recorded
episode, so the widest string in one episode pins that worker's running episode count k and the tail
width is CONSTANT inside an episode. A within-episode change would contradict the explanation, so
the report prints it rather than averaging it away.

    python -m pytest paper/fixtures/test_buttons_report.py -q
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

pytest.importorskip("pyarrow")

import buttons_report  # noqa: E402

FORWARD = "100000000"


def _switch(k, j=0):
    """Arnold's string after k `Game.start()` calls, pressing SELECT_WEAPON j."""
    return FORWARD + "0" * (10 * k + j) + "1"


def _write(d, ep, buttons, tics=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(d, exist_ok=True)
    n = len(buttons)
    tics = np.arange(n) if tics is None else tics
    pq.write_table(pa.table({"tic": pa.array(list(tics), pa.int32()),
                             "buttons": pa.array(list(buttons), pa.string()),
                             "action": pa.array([0] * n, pa.int16()),
                             "deaths": pa.array([0] * n, pa.int16())}),
                   os.path.join(d, f"ep_{ep:05d}.parquet"))


def test_the_index_of_a_weapon_select_press_follows_the_start_count():
    assert _switch(0, 0) == FORWARD + "1" and len(_switch(0, 0)) == 10
    assert len(_switch(1, 0)) == 20            # index 19: the first k that cannot be executed
    assert len(_switch(249, 0)) == 2500


def test_the_report_names_the_widths_and_the_executed_and_unexecuted_switches(tmp_path):
    d = str(tmp_path / "raw")
    _write(d, 0, [FORWARD] * 18 + [_switch(0, 4)] + [_switch(0, 9)])
    r = buttons_report.report(d)
    ep = r["episodes"][0]
    assert ep["episode"] == 0 and ep["rows"] == 20
    assert ep["raw_max_width"] == 19            # SELECT_WEAPON9 at index 18, the last executed button
    assert ep["executed_switch_rows"] == 2 and ep["unexecuted_switch_rows"] == 0
    assert ep["inferred_starts"] == 0


def test_the_report_counts_a_switch_the_engine_never_saw(tmp_path):
    d = str(tmp_path / "raw")
    _write(d, 0, [FORWARD] * 19 + [_switch(3, 0)])
    ep = buttons_report.report(d)["episodes"][0]
    assert ep["raw_max_width"] == 40 and ep["unexecuted_switch_rows"] == 1
    assert ep["executed_switch_rows"] == 0
    assert ep["inferred_starts"] == 3
    assert ep["fraction_over_executed"] == pytest.approx(0.05)


def test_the_report_flags_a_within_episode_width_change(tmp_path):
    """This would contradict "the list grows once per `Game.start()`", so it must be visible."""
    d = str(tmp_path / "raw")
    _write(d, 0, [_switch(2, 0), _switch(7, 0)])
    ep = buttons_report.report(d)["episodes"][0]
    assert ep["within_episode_growth"] is True
    assert ep["raw_widths_over_control_bits"] == [30, 80]


def test_a_clean_episode_shows_no_within_episode_growth(tmp_path):
    d = str(tmp_path / "raw")
    _write(d, 0, [FORWARD] * 10 + [_switch(4, 0), _switch(4, 9)])
    ep = buttons_report.report(d)["episodes"][0]
    assert ep["within_episode_growth"] is False, "one start, so one tail width up to the weapon id"
    assert ep["inferred_starts"] == 4


def test_the_corpus_summary_aggregates_every_episode(tmp_path):
    d = str(tmp_path / "raw")
    _write(d, 0, [FORWARD] * 20)
    _write(d, 1, [FORWARD] * 19 + [_switch(1, 0)])
    r = buttons_report.report(d)
    assert [e["episode"] for e in r["episodes"]] == [0, 1]
    assert r["corpus"]["rows"] == 40
    assert r["corpus"]["raw_max_width"] == 20
    assert r["corpus"]["unexecuted_switch_rows"] == 1
    assert r["corpus"]["episodes_with_within_episode_growth"] == 0
    assert r["corpus"]["max_inferred_starts"] == 1


def test_the_report_reads_the_sidecars_when_asked(tmp_path):
    """After normalisation the parquet may be far away, so the provenance columns answer instead."""
    from pertic_fixtures import held_actions, write_pertic_episode
    pytest.importorskip("torch")
    lat = str(tmp_path / "lat")
    write_pertic_episode(lat, 0, held_actions([1] * 5))
    p = os.path.join(lat, "ep_00000_meta.npz")
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files}
    cols["buttons_raw_len"] = np.array([9] * 19 + [2503], dtype=np.int16)
    cols["switch_requested_index"] = np.array([-1] * 19 + [2502], dtype=np.int32)
    np.savez(p, **cols)
    ep = buttons_report.report(lat, sidecars=True)["episodes"][0]
    assert ep["raw_max_width"] == 2503 and ep["unexecuted_switch_rows"] == 1
    assert ep["inferred_starts"] == 249


def test_the_cli_prints_json_and_limits_the_episode_count(tmp_path, capsys):
    d = str(tmp_path / "raw")
    for ep in range(4):
        _write(d, ep, [FORWARD] * 4)
    buttons_report.main(buttons_report.build_parser().parse_args(["--dir", d, "--episodes", "2"]))
    out = json.loads(capsys.readouterr().out)
    assert len(out["episodes"]) == 2 and out["corpus"]["episodes"] == 2
