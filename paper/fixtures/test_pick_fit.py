"""Choosing the decoder tune's micro-batch and update budget from its fit checks.

The choice has to survive the two things that actually happen on a shared card: a fit check that
ran out of memory and therefore wrote no metrics at all, and a fit check that fitted but used more
memory than is free next to somebody else's job. Both must lose to a slower fit that fits, and the
update budget must be the one whose linear decay ends when the card is handed back.

Pure Python, no torch, so this runs on the Mac:

    python -m pytest paper/fixtures/test_pick_fit.py -q
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from tools.pick_fit import MIN_STEPS, choose, read_fits  # noqa: E402


def fit(tmp_path, micro, steps, seconds, peak_gb=None, eff=None, write=True):
    """One fit check's output directory, or an empty one for a check that ran out of memory."""
    d = tmp_path / f"fit_mb{micro}"
    d.mkdir()
    if not write:
        return d
    json.dump({"steps": steps, "train_seconds": seconds, "effective_batch": eff or micro,
               "peak_mem_gb": peak_gb}, open(d / "metrics.json", "w"))
    return d


def test_read_fits_reports_throughput_per_micro_batch(tmp_path):
    fit(tmp_path, 16, 60, 60.0, peak_gb=12.0)      # 16 frames/s
    fit(tmp_path, 32, 60, 80.0, peak_gb=22.0)      # 24 frames/s
    got = {f["micro"]: f for f in read_fits(str(tmp_path))}

    assert set(got) == {16, 32}
    assert got[16]["frames_per_s"] == pytest.approx(16.0)
    assert got[32]["frames_per_s"] == pytest.approx(24.0)
    assert got[32]["peak_gb"] == 22.0


def test_a_fit_that_wrote_nothing_does_not_compete(tmp_path):
    """An out-of-memory fit check raises before it writes, so its directory is empty."""
    fit(tmp_path, 16, 60, 60.0)
    fit(tmp_path, 64, 0, 0.0, write=False)

    assert [f["micro"] for f in read_fits(str(tmp_path))] == [16]


def test_a_warmup_length_fit_is_not_a_measurement(tmp_path):
    fit(tmp_path, 16, MIN_STEPS - 1, 5.0)
    assert read_fits(str(tmp_path)) == []


def test_choose_takes_the_fastest_fit_and_sizes_the_budget_to_the_time_left(tmp_path):
    fit(tmp_path, 16, 60, 60.0)                    # 16 frames/s
    fit(tmp_path, 48, 60, 60.0)                    # 48 frames/s
    pick = choose(read_fits(str(tmp_path)), seconds_left=3600)

    assert pick["micro"] == 48
    assert pick["max_steps"] == int(48.0 * 3600 / 48)
    assert pick["max_steps"] * pick["effective_batch"] == pytest.approx(48.0 * 3600, rel=1e-6)


def test_choose_rejects_a_fit_that_will_not_fit_beside_another_job(tmp_path):
    """The fastest batch is useless if the card only has 22 GB free."""
    fit(tmp_path, 16, 60, 60.0, peak_gb=12.0)      # 16 frames/s, fits
    fit(tmp_path, 48, 60, 40.0, peak_gb=41.0)      # 72 frames/s, does not
    pick = choose(read_fits(str(tmp_path)), seconds_left=3600, max_gb=22.0)

    assert pick["micro"] == 16 and pick["rejected"] == [48]


def test_choose_keeps_a_fit_whose_peak_was_not_recorded(tmp_path):
    """A missing number is not evidence of a breach, so it must not silently drop the fastest fit."""
    fit(tmp_path, 32, 60, 30.0, peak_gb=None)
    pick = choose(read_fits(str(tmp_path)), seconds_left=600, max_gb=22.0)

    assert pick["micro"] == 32 and pick["rejected"] == []


def test_choose_gives_nothing_when_no_fit_qualifies(tmp_path):
    fit(tmp_path, 48, 60, 60.0, peak_gb=41.0)
    assert choose(read_fits(str(tmp_path)), seconds_left=3600, max_gb=22.0) is None
    assert choose(read_fits(str(tmp_path)), seconds_left=0) is None
    assert choose([], seconds_left=3600) is None


def test_the_budget_never_rounds_down_to_zero(tmp_path):
    fit(tmp_path, 16, 60, 600.0)
    pick = choose(read_fits(str(tmp_path)), seconds_left=1)
    assert pick["max_steps"] == 1
