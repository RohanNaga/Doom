"""The recorder's `buttons` string is Arnold's REQUEST; the engine executes only its first 19 entries.

`record_arnold.py:255` serialises Arnold's control list (`decision_buttons`, line 69) and hands the
same list to `make_action` at line 273. ViZDoom's `setAction` reads `actions[i]` only for
`i < availableButtons.size()` (19) and zero-fills the missing entries (ViZDoom 1.2.4
`src/lib/ViZDoomGame.cpp:151-158`), so the executed control is exactly `s[:19]` right-padded with 0.

The strings are longer than 19 because Arnold's `add_buttons` appends the ten `SELECT_WEAPON%i`
names to a SHARED `available_buttons` list on every `Game.start()` (`src/doom/actions.py:197-199`,
`game.py:485`) while ViZDoom deduplicates its own list, so after k starts in one recorder process
`SELECT_WEAPONj` maps to 9 + 10*k + j. `record_arnold.py:167` starts the game once per recorded
episode, so k is the worker's running episode count and only its first episode can execute a switch.

    python -m pytest paper/fixtures/test_executed_buttons.py -q
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import transitions  # noqa: E402
from transitions import (CONTROL_BITS, EXECUTED_BUTTONS, button_width_report,  # noqa: E402
                         normalize_button_column, normalize_buttons, raw_button_lengths,
                         switch_request_indices)

FORWARD = "100000000"                       # the 95% case: Arnold's nine control entries
SWITCH_IN_RANGE = "100000000" + "0" * 4 + "1"          # 14 chars, a 1 at index 13: an executed switch
SWITCH_BEYOND = "100000000" + "0" * 2493 + "1"         # 2503 chars, a 1 at index 2502: never executed


def test_the_executed_width_is_the_engines_button_count():
    assert EXECUTED_BUTTONS == 19
    assert CONTROL_BITS == 9, "the canonical table and every finished row depend on the 9-bit prefix"


def test_a_nine_character_request_is_right_padded_to_nineteen():
    assert normalize_buttons(FORWARD) == FORWARD + "0" * 10
    assert len(normalize_buttons(FORWARD)) == EXECUTED_BUTTONS


def test_a_switch_inside_the_nineteen_buttons_survives():
    got = normalize_buttons(SWITCH_IN_RANGE)
    assert got == SWITCH_IN_RANGE + "0" * 5
    assert got[13] == "1"


def test_a_switch_beyond_the_nineteen_buttons_is_dropped():
    """The engine never read index 2502, so a trailing 1 out there is not a weapon switch."""
    got = normalize_buttons(SWITCH_BEYOND)
    assert got == FORWARD + "0" * 10
    assert "1" not in got[CONTROL_BITS:]


def test_normalisation_never_changes_the_first_nine_characters():
    for s in (FORWARD, SWITCH_IN_RANGE, SWITCH_BEYOND, "000000100", "1" * 19, "0" * 2506):
        assert normalize_buttons(s)[:CONTROL_BITS] == s[:CONTROL_BITS]


def test_normalisation_is_idempotent():
    for s in (FORWARD, SWITCH_IN_RANGE, SWITCH_BEYOND):
        assert normalize_buttons(normalize_buttons(s)) == normalize_buttons(s)


def test_a_non_binary_character_inside_the_executed_prefix_is_refused():
    with pytest.raises(ValueError, match="not binary"):
        normalize_buttons("120000000")
    with pytest.raises(ValueError, match="not binary"):
        normalize_buttons("10000000-")


def test_a_non_binary_character_beyond_the_executed_prefix_is_ignored():
    """Nothing past index 18 reaches the model, so it cannot make a control value wrong."""
    assert normalize_buttons("1" * 19 + "x") == "1" * 19


def test_an_empty_string_is_refused():
    with pytest.raises(ValueError, match="empty"):
        normalize_buttons("")


def test_the_column_helpers_are_fixed_width_and_small():
    col = normalize_button_column([FORWARD, SWITCH_IN_RANGE, SWITCH_BEYOND])
    assert col.dtype == np.dtype("<U19")
    assert col.tolist() == [normalize_buttons(s) for s in (FORWARD, SWITCH_IN_RANGE, SWITCH_BEYOND)]
    assert col.nbytes == 3 * 19 * 4


def test_the_raw_length_and_requested_switch_index_are_recoverable_per_row():
    rows = [FORWARD, SWITCH_IN_RANGE, SWITCH_BEYOND]
    lens = raw_button_lengths(rows)
    idx = switch_request_indices(rows)
    assert lens.dtype == np.int16 and lens.tolist() == [9, 14, 2503]
    assert idx.dtype == np.int32 and idx.tolist() == [-1, 13, 2502]


def test_the_requested_switch_index_is_the_highest_one_beyond_the_control_bits():
    assert switch_request_indices(["1" * 19]).tolist() == [18]
    assert switch_request_indices(["1" * 9 + "0" * 10]).tolist() == [-1]


def test_the_width_report_names_the_raw_maximum_and_the_unexecuted_fraction():
    r = button_width_report([FORWARD] * 18 + [SWITCH_IN_RANGE, SWITCH_BEYOND])
    assert r["rows"] == 20
    assert r["raw_max_width"] == 2503
    assert r["rows_over_executed"] == 1
    assert r["fraction_over_executed"] == pytest.approx(0.05)
    assert r["executed_switch_rows"] == 1
    assert r["unexecuted_switch_rows"] == 1
    assert r["width"] == EXECUTED_BUTTONS


def test_the_width_report_infers_the_recorder_start_count():
    """k = (max_width - 1 - 9) // 10 from `SELECT_WEAPONj` at index 9 + 10*k + j."""
    assert button_width_report([SWITCH_BEYOND])["inferred_starts"] == 249
    assert button_width_report([SWITCH_IN_RANGE])["inferred_starts"] == 0
    assert button_width_report([FORWARD])["inferred_starts"] is None


def test_the_width_report_flags_a_within_episode_growth():
    """Arnold's list grows per `Game.start()`, so inside one episode the tail width is constant."""
    ok = button_width_report([FORWARD] * 5 + [SWITCH_IN_RANGE, "1" + "0" * 12 + "1"])
    assert ok["within_episode_growth"] is False
    bad = button_width_report([SWITCH_IN_RANGE, SWITCH_BEYOND])
    assert bad["within_episode_growth"] is True
    assert sorted(bad["raw_widths_over_control_bits"]) == [14, 2503]


def test_the_canonical_prefix_of_every_row_is_unchanged_by_normalisation():
    """`canonical_table` counts `str(b)[:9]`; normalising the sidecar must not move a single count."""
    rng = np.random.RandomState(0)
    raw = []
    for _ in range(200):
        head = "".join(str(b) for b in rng.randint(0, 2, CONTROL_BITS))
        tail = "0" * int(rng.randint(0, 40)) + ("1" if rng.rand() < 0.5 else "")
        raw.append(head + tail)
    before = transitions.canonical_table(np.zeros(len(raw), dtype=np.int64), raw)
    after = transitions.canonical_table(np.zeros(len(raw), dtype=np.int64),
                                        normalize_button_column(raw))
    assert before == after


# ---------------------------------------------------------------------------------------
# what the corpus reader does with a normalised, and an un-normalised, sidecar
# ---------------------------------------------------------------------------------------

pytest.importorskip("torch")

from pertic_fixtures import held_actions, write_pertic_episode  # noqa: E402

import doom_data  # noqa: E402
from doom_data import (TicWindowDataset, check_control_strings, control_matrix,  # noqa: E402
                       corpus_control_bits)


def _corpus(tmp_path, name, buttons=None, rows=40):
    d = str(tmp_path / name)
    kw = {} if buttons is None else {"buttons": np.asarray(buttons)}
    write_pertic_episode(d, 0, held_actions([1] * (rows // 4)), **kw)
    return d


def test_the_control_matrix_is_always_the_nineteen_executed_buttons():
    m = control_matrix([FORWARD, SWITCH_IN_RANGE, SWITCH_BEYOND])
    assert m.shape == (3, EXECUTED_BUTTONS) and m.dtype.name == "float32"
    assert m[0].tolist() == [1.0] + [0.0] * 18
    assert m[1][13] == 1.0
    assert m[2].tolist() == m[0].tolist(), "a 1 beyond index 18 was never executed"


def test_control_strings_of_different_raw_widths_are_accepted_and_reported():
    """The recorder emits 9, 12 to 17 and 112 to 2,506 characters; refusing that refused every episode."""
    r = check_control_strings([FORWARD] * 9 + [SWITCH_BEYOND])
    assert r.width == EXECUTED_BUTTONS
    assert r.raw_max_width == 2503
    assert r.fraction_over_executed == pytest.approx(0.1)


def test_a_non_binary_control_string_is_still_refused():
    with pytest.raises(ValueError, match="not binary"):
        check_control_strings(["101", "1-1"])
    with pytest.raises(ValueError, match="not binary"):
        control_matrix(["120000000"])


def test_the_corpus_reports_nineteen_control_bits(tmp_path):
    d = _corpus(tmp_path, "bits")
    assert corpus_control_bits(d) == EXECUTED_BUTTONS
    ds = TicWindowDataset(d, None, context_frames=4, action_history=4)
    assert ds.control_bits == EXECUTED_BUTTONS
    assert ds.summary["control_bits"] == EXECUTED_BUTTONS


def test_two_episodes_of_different_raw_widths_now_share_one_control_width(tmp_path):
    """Both are the same WAD's 19 buttons; the raw width was an Arnold artefact, not a button list."""
    d = str(tmp_path / "mixedwidths")
    write_pertic_episode(d, 0, held_actions([1] * 10))
    write_pertic_episode(d, 1, held_actions([1] * 10), buttons=np.array([SWITCH_IN_RANGE] * 40))
    ds = TicWindowDataset(d, None, context_frames=4, action_history=4)
    assert ds.control_bits == EXECUTED_BUTTONS


def test_an_un_normalised_sidecar_is_refused_and_names_the_repair(tmp_path):
    """A `<U2503` column is the 40 MB sidecar; loading 2,000 of them is what we refuse to do."""
    d = _corpus(tmp_path, "wide", buttons=np.array([SWITCH_BEYOND] * 40))
    with pytest.raises(ValueError, match="normalize-sidecars"):
        TicWindowDataset(d, None, context_frames=4, action_history=4)
    with pytest.raises(ValueError, match="normalize-sidecars"):
        corpus_control_bits(d)


def test_a_sidecar_at_the_executed_width_or_narrower_is_accepted(tmp_path):
    for name, b in (("exact", np.array([FORWARD.ljust(19, "0")] * 40)),
                    ("narrow", np.array([FORWARD] * 40))):
        d = _corpus(tmp_path, name, buttons=b)
        assert TicWindowDataset(d, None, context_frames=4, action_history=4).control_bits == 19


def test_the_sidecar_dtype_guard_is_exported_for_every_reader():
    assert doom_data.check_sidecar_buttons_dtype(np.array(["1" * 19], dtype="<U19"), "x.npz") is None
    with pytest.raises(ValueError, match="normalize-sidecars"):
        doom_data.check_sidecar_buttons_dtype(np.array([SWITCH_BEYOND]), "x.npz")
