"""Merging the two ceiling-scoring passes into one table.

`vae_gate_score.py` writes its metrics only after it has scored every decoder it was handed, so the
decoder ceilings are scored in two passes: the headline (stock, incumbent, finished tune) first and
the hourly curve second, because the card has a hand-back deadline and losing the second pass must
not cost the answer. Both passes score the baseline, so the merge has to keep the first reading of
a decoder that appears twice rather than letting a later pass overwrite it.

Pure Python, no torch, so this runs on the Mac:

    python -m pytest paper/fixtures/test_gate_merge.py -q
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "results", "levers_2026-09-20", "e2-decoder"))

from make_curve import hour_of, merge_gates, presentations  # noqa: E402


def gate(decoders, baseline="tuned_sd_lpips"):
    """One scoring pass's output. A decoder's PSNR is its position in this pass, so a merge that
    lets a later pass overwrite an earlier reading shows up as a changed number."""
    def reading(i):
        return {"psnr": {"mean": float(i), "sem": 0.01},
                "lpips": {"mean": 0.05, "sem": 0.001},
                "hud_psnr": {"mean": 30.0, "sem": 0.01}}

    seen = {n: reading(i) for i, n in enumerate(decoders)}
    paired = {n: {"psnr": {"delta": float(i), "ci95": [-0.1, 0.1]}}
              for i, n in enumerate(decoders) if n != baseline}
    return {"baseline": baseline,
            "decoders": {n: f"/path/{n}" for n in decoders},
            "metrics": {"seen": {**seen, "episodes": 60}},
            "paired": {"seen": paired}}


def test_merge_unions_the_decoders_of_both_passes():
    m = merge_gates([gate(["stock_sd", "tuned_sd_lpips", "mse_final"]),
                     gate(["tuned_sd_lpips", "mse_h1", "mse_h2"])])

    assert set(m["decoders"]) == {"stock_sd", "tuned_sd_lpips", "mse_final", "mse_h1", "mse_h2"}
    assert set(m["metrics"]["seen"]) == set(m["decoders"]) | {"episodes"}
    assert m["baseline"] == "tuned_sd_lpips"


def test_the_later_pass_does_not_overwrite_a_shared_decoder():
    """Both passes score the baseline; the headline pass's reading is the one that is kept."""
    first = gate(["stock_sd", "tuned_sd_lpips"])          # tuned_sd_lpips gets psnr 1.0 here
    second = gate(["tuned_sd_lpips", "mse_h1"])           # and 0.0 here
    m = merge_gates([first, second])

    assert m["metrics"]["seen"]["tuned_sd_lpips"]["psnr"]["mean"] == 1.0
    assert m["metrics"]["seen"]["mse_h1"]["psnr"]["mean"] == 1.0   # from the second pass


def test_merge_survives_a_lost_second_pass():
    m = merge_gates([gate(["stock_sd", "tuned_sd_lpips", "mse_final"])])
    assert set(m["decoders"]) == {"stock_sd", "tuned_sd_lpips", "mse_final"}
    assert m["paired"]["seen"]


def test_merge_refuses_to_invent_a_table_from_nothing():
    with pytest.raises(SystemExit):
        merge_gates([])


def test_hour_of_reads_the_checkpoint_names():
    assert hour_of("mse_h1") == 1 and hour_of("mse_h12") == 12
    assert hour_of("mse_final") == 10**6
    assert hour_of("stock_sd") is None and hour_of("tuned_sd_lpips") is None


def test_presentations_comes_from_the_tune_history():
    tune = {"effective_batch": 32, "presentations": 320000,
            "history": [{"step": 100}, {"step": 3000, "hour_ckpt": 1}, {"step": 6000, "hour_ckpt": 2}]}
    assert presentations(tune, 1) == 96000
    assert presentations(tune, 2) == 192000
    assert presentations(tune, 10**6) == 320000
    assert presentations(tune, 9) is None
    assert presentations(None, 1) is None
