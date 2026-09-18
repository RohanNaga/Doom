"""CPU gates for the twelve-cell knob grid's trainer flags.

Each cell of the grid turns exactly one knob relative to the base PixArt-alpha recipe, so every
new flag has to default to the behaviour the five finished rows were trained with, and has to be
readable back out of a checkpoint by the evaluators. These checks pin both halves on the CPU:

* `--train-fraction`: nested seeded subsets of the *training* episode list only.
* `--objective {v,eps}`: the same betas, the same x0 estimate at a fixed (x_t, t), the objective
  recorded in the checkpoint and read back by the samplers.
* `--action-inject {token,adaln}`: additive injection into PixArt's adaLN-single path, zero-init
  so step 0 is still the pretrained model.
* the tiny fixture path: five real training updates and a two-window evaluation for each cell's
  flag combination, with a DiT-S/2 stand-in for the heavy warm-started backbones.

The PixArt and SD 3.5 transformers themselves are not built here (network access and gigabytes of
weights); `verify_sd35.py` and a GPU run gate those.

    python -m pytest paper/fixtures/test_grid_flags.py -q
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from doom_data import select_train_episodes  # noqa: E402

TRAIN_IDS = list(range(100, 164))          # 64 training episodes
VAL_IDS = list(range(200, 208))


# ---- --train-fraction ------------------------------------------------------------------------

def test_full_fraction_is_the_whole_training_list():
    assert select_train_episodes(TRAIN_IDS, 1.0) == sorted(TRAIN_IDS)


def test_fraction_sizes_are_rounded_counts():
    for frac, n in ((0.5, 32), (0.25, 16), (0.125, 8)):
        assert len(select_train_episodes(TRAIN_IDS, frac)) == n


def test_subsets_are_nested_so_the_ladder_varies_only_in_size():
    eighth, quarter, half = (set(select_train_episodes(TRAIN_IDS, f)) for f in (0.125, 0.25, 0.5))
    assert eighth < quarter < half < set(TRAIN_IDS)


def test_subset_is_seeded_and_reproducible():
    assert select_train_episodes(TRAIN_IDS, 0.25, seed=0) == select_train_episodes(TRAIN_IDS, 0.25, seed=0)
    assert select_train_episodes(TRAIN_IDS, 0.25, seed=1) != select_train_episodes(TRAIN_IDS, 0.25, seed=0)


def test_subset_never_reaches_the_validation_episodes():
    # the function only ever sees the train list, which is what keeps validation and evaluation
    # identical across the data cells
    assert not set(select_train_episodes(TRAIN_IDS, 0.125)) & set(VAL_IDS)


def test_at_least_one_episode_survives_a_tiny_fraction():
    assert len(select_train_episodes(TRAIN_IDS, 0.001)) == 1


def test_bad_fractions_are_refused():
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            select_train_episodes(TRAIN_IDS, bad)


def test_ids_come_back_sorted_and_unique():
    got = select_train_episodes([5, 1, 3, 2, 4, 6, 7, 8], 0.5)
    assert got == sorted(set(got))
