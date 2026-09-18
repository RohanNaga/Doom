"""The EMA list is keyed by parameter name, and a persistent buffer must not shift it.

`train_wm.py` keeps the EMA as a plain fp32 CPU list, one tensor per parameter, and has to attach
names to it when it writes a checkpoint. Zipping against `state_dict().keys()` is only correct
while a model has no persistent buffers. SD 3.5's positional table is one, and it sorts *before*
the patch projection inside the same module, so that zip would shift every key by one and save an
EMA whose tensors belong to the wrong weights, which no later strict load would catch.

    python -m pytest paper/fixtures/test_ema_keys.py -q
"""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import backbones  # noqa: E402
from doomdit_utils import load_world_model_state  # noqa: E402
from train_wm import ema_keys  # noqa: E402
from verify_sd35 import TINY  # noqa: E402


def dit():
    return backbones.DiTWorldModel(num_actions=3, context_frames=2, noise_buckets=2, model_name="DiT-S/2",
                                   action_dropout=0.0, grad_ckpt=False)


def sd35():
    from diffusers import SD3Transformer2DModel
    torch.manual_seed(0)
    return backbones.SD35WorldModel(num_actions=3, context_frames=2, noise_buckets=2, grad_ckpt=False,
                                    latent_channels=16, transformer=SD3Transformer2DModel(**TINY))


def checkpoint(model):
    """What train_wm.py writes: bf16 weights plus a parameter-keyed bf16 EMA."""
    ema = [p.detach().float().cpu().clone() for p in model.parameters()]
    return {"model": {k: v.detach().cpu().to(torch.bfloat16) for k, v in model.state_dict().items()},
            "ema": {k: t.to(torch.bfloat16) for k, t in zip(ema_keys(model), ema)}}


def test_the_four_channel_rows_are_unaffected():
    m = dit()
    # no persistent buffers, so the new keying is the old keying: finished runs keep loading
    assert ema_keys(m) == list(m.state_dict().keys())


def test_sd35_has_a_persistent_buffer_that_would_shift_the_old_keying():
    m = sd35()
    keys, names = list(m.state_dict().keys()), ema_keys(m)
    assert set(keys) - set(names) == {"transformer.pos_embed.pos_embed"}
    # the buffer sorts before the patch projection, so from there on the old zip paired every EMA
    # tensor with the next weight's name: a shift, not a missing entry a strict load could catch
    shift = next(i for i, (k, n) in enumerate(zip(keys, names)) if k != n)
    assert keys[shift] == "transformer.pos_embed.pos_embed"
    assert names[shift] == "transformer.pos_embed.proj.weight" == keys[shift + 1]


@pytest.mark.parametrize("build", [dit, sd35], ids=["dit", "sd35"])
def test_ema_round_trip_loads_into_a_fresh_model(build):
    ck = checkpoint(build())
    fresh = build()
    assert load_world_model_state(fresh, ck, use_ema=True) == "ema"
    assert load_world_model_state(fresh, ck, use_ema=False) == "model"


def test_a_mismatched_ema_is_an_error_not_a_silent_skip():
    ck = checkpoint(sd35())
    ck["ema"] = {("wrong." + k): v for k, v in ck["ema"].items()}
    with pytest.raises(SystemExit):
        load_world_model_state(sd35(), ck, use_ema=True)
