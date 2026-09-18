"""`fit_split.py` must hand the sweep a growing prefix of the parent split, never a leak.

Three properties matter and are pinned here: the picked episodes are a subset of the parent
split's own train/val lists (so a validation episode can never arrive in training), the pick is
a prefix of the sorted intersection (so the file can be rewritten as more episodes land without
changing what an already-launched sweep is reading), and an episode with no metadata file is not
counted as encoded.

    python -m pytest paper/fixtures/test_fit_split.py -q
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from fit_split import encoded_episodes, fit_split  # noqa: E402

PARENT = {"train": list(range(0, 100)), "val": list(range(100, 120)), "unseen_map": list(range(120, 140))}


def write_episode(d, ep, meta=True):
    open(os.path.join(d, f"ep_{ep:05d}_latents.npy"), "wb").close()
    if meta:
        open(os.path.join(d, f"ep_{ep:05d}_meta.npz"), "wb").close()


def test_only_complete_pairs_count(tmp_path):
    d = str(tmp_path)
    write_episode(d, 3)
    write_episode(d, 7, meta=False)      # latents landed, metadata still in flight
    assert encoded_episodes(d) == [3]


def test_prefix_is_subset_of_parent_lists():
    present = list(range(0, 60)) + list(range(100, 110)) + list(range(120, 130))
    s = fit_split(PARENT, present, n_train=40, n_val=8)
    assert s["train"] == list(range(0, 40))
    assert s["val"] == list(range(100, 108))
    assert set(s["train"]) <= set(PARENT["train"])
    assert set(s["val"]) <= set(PARENT["val"])
    assert not set(s["train"]) & set(s["val"])
    assert s["unseen_map"] == []           # an unseen-map episode is never pulled in


def test_pick_is_stable_as_more_episodes_land():
    early = fit_split(PARENT, list(range(0, 50)) + list(range(100, 112)), 40, 8)
    late = fit_split(PARENT, list(range(0, 100)) + list(range(100, 120)), 40, 8)
    assert early["train"] == late["train"] and early["val"] == late["val"]


def test_out_of_order_arrival_still_gives_sorted_prefix():
    present = list(reversed(range(0, 45))) + list(range(100, 110))
    s = fit_split(PARENT, present, 40, 8)
    assert s["train"] == list(range(0, 40))


def test_too_few_encoded_raises_unless_allowed():
    present = list(range(0, 10)) + [100, 101]
    with pytest.raises(SystemExit):
        fit_split(PARENT, present, 40, 8)
    s = fit_split(PARENT, present, 40, 8, allow_partial=True)
    assert s["train"] == list(range(0, 10)) and s["val"] == [100, 101]


def test_meta_records_what_the_sweep_is_reading(tmp_path):
    s = fit_split(PARENT, list(range(0, 60)) + list(range(100, 110)), 40, 8)
    assert s["meta"]["requested"] == {"train": 40, "val": 8}
    assert s["meta"]["encoded_episodes"] == 70
    json.dumps(s)   # the file has to be serialisable as written
