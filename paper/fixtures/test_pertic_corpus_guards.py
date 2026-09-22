"""The per-tic window rules are no longer fail-open on a missing or malformed column.

`tic_window_starts` applied the death rule `if "deaths" in meta.files` and the map rule
`if "map_id" in meta.files`, so a sidecar without one silently dropped that rule: a 70-row synthetic
sidecar lacking `deaths` admitted 38 L32 windows, any of which could span a respawn. Nothing in the
window count says so. `control_matrix` converted each character with `float()`, so a '2' from a
mis-imported column became a control value of 2.0.

    python -m pytest paper/fixtures/test_pertic_corpus_guards.py -q
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import doom_data  # noqa: E402
from doom_data import TicWindowDataset, check_control_strings, tic_window_starts  # noqa: E402
from pertic_fixtures import held_actions, write_pertic_episode  # noqa: E402


def _meta(rows=40, **over):
    m = {"tic": np.arange(rows, dtype=np.int64), "deaths": np.zeros(rows, dtype=np.int64),
         "map_id": np.full(rows, 3, dtype=np.int64)}
    m.update(over)
    return m


@pytest.mark.parametrize("column", ["deaths", "map_id"])
def test_a_missing_validity_column_is_refused_not_ignored(column):
    m = _meta()
    m.pop(column)
    with pytest.raises(ValueError, match="missing"):
        tic_window_starts(m, 8)
    assert len(tic_window_starts(m, 8, strict=False)) == 40 - 8, \
        "strict=False must keep the documented permissive behaviour"


def test_the_old_fail_open_rule_admitted_a_window_across_a_death():
    deaths = np.array([0] * 20 + [1] * 20, dtype=np.int64)
    with_col = _meta(deaths=deaths)
    assert len(tic_window_starts(with_col, 8)) == 2 * (20 - 8)
    without = _meta()
    without.pop("deaths")
    assert len(tic_window_starts(without, 8, strict=False)) == 40 - 8      # 12 illegal extras


@pytest.mark.parametrize("column", ["tic", "deaths", "map_id"])
def test_a_column_of_the_wrong_length_is_refused(column):
    m = _meta()
    m[column] = m[column][:30]
    with pytest.raises(ValueError, match="do not describe the same rows|missing"):
        tic_window_starts(m, 8)


def test_the_dataset_requires_every_column_the_contract_names():
    assert set(doom_data.TIC_CORPUS_COLUMNS) == {"tic", "deaths", "map_id", "buttons",
                                                 "is_decision", "action"}


def _corpus(tmp_path, name, rows=40, **over):
    d = str(tmp_path / name)
    write_pertic_episode(d, 0, held_actions([1] * (rows // 4)), **over)
    return d


@pytest.mark.parametrize("column", ["tic", "deaths", "map_id", "buttons", "action", "is_decision"])
def test_a_sidecar_missing_a_required_column_is_refused(tmp_path, column):
    d = _corpus(tmp_path, f"drop_{column}")
    p = os.path.join(d, "ep_00000_meta.npz")
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files if k != column}
    np.savez(p, **cols)
    with pytest.raises(ValueError, match="column"):
        TicWindowDataset(d, None, context_frames=4, action_history=4)


@pytest.mark.parametrize("column", ["tic", "deaths", "map_id", "buttons", "action"])
def test_a_column_shorter_than_the_latents_is_refused(tmp_path, column):
    d = _corpus(tmp_path, f"short_{column}")
    p = os.path.join(d, "ep_00000_meta.npz")
    with np.load(p) as z:
        cols = {k: z[k] for k in z.files}
    cols[column] = cols[column][:-3]
    np.savez(p, **cols)
    with pytest.raises(ValueError, match="does not describe these latents|do not describe the same rows"):
        TicWindowDataset(d, None, context_frames=4, action_history=4)


def test_a_non_binary_control_string_is_refused(tmp_path):
    """`float(c)` accepted any digit, so a mis-imported column became a control value of 2.0."""
    d = _corpus(tmp_path, "notbinary", buttons=np.array(["120000000"] * 40))
    with pytest.raises(ValueError, match="not binary"):
        TicWindowDataset(d, None, context_frames=4, action_history=4)
    with pytest.raises(ValueError, match="not binary"):
        check_control_strings(["101", "1-1"])


def test_control_strings_of_two_raw_widths_are_normalised_not_refused(tmp_path):
    """Arnold appends a weapon-select press on some rows only, so one episode holds two raw widths.

    The old rule demanded one constant raw width and therefore refused every real episode; the
    executed control is 19 entries on all of them.
    """
    b = np.array(["100000000"] * 39 + ["1000000001"])
    d = _corpus(tmp_path, "widths", buttons=b)
    ds = TicWindowDataset(d, None, context_frames=4, action_history=4)
    assert ds.control_bits == 19


def test_two_episodes_of_different_raw_widths_are_accepted(tmp_path):
    """The raw width is Arnold's bookkeeping, not the WAD's button list, so it cannot mix corpora."""
    d = str(tmp_path / "mixedwads")
    write_pertic_episode(d, 0, held_actions([1] * 10))
    write_pertic_episode(d, 1, held_actions([1] * 10), buttons=np.array(["1000000001"] * 40))
    assert TicWindowDataset(d, None, context_frames=4, action_history=4).control_bits == 19


def test_the_control_check_runs_without_action_history(tmp_path):
    """`--action-history 0` used to skip the buttons column entirely, so a broken one went unnoticed
    until an evaluator with history read the same corpus."""
    d = _corpus(tmp_path, "nohistory", buttons=np.array(["120000000"] * 40))
    with pytest.raises(ValueError, match="not binary"):
        TicWindowDataset(d, None, context_frames=4)


def test_a_healthy_corpus_still_loads_unchanged(tmp_path):
    d = _corpus(tmp_path, "healthy")
    ds = TicWindowDataset(d, None, context_frames=4, action_history=4)
    assert len(ds) == 40 - 4
    assert ds.control_bits == 19
    assert check_control_strings(["100000000"]).width == 19
