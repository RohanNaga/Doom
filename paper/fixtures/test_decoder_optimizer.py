"""The decoder tune's optimizer: AdamW fused into one kernel per step on a GPU.

finetune_decoder.py built a plain AdamW, the one piece of the fill-the-card recipe (CLAUDE.md) the
tune lacked (Astra's review, 2026-09-26, section 5); train_wm.py already fuses with
`fused=(device.type == "cuda")`. On CPU the tune keeps the plain kernel.

    python -m pytest paper/fixtures/test_decoder_optimizer.py -q
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from finetune_decoder import decoder_optimizer  # noqa: E402


def test_a_gpu_tune_asks_for_the_fused_kernel(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, params, **kw):
            seen.update(kw)
    monkeypatch.setattr(torch.optim, "AdamW", Recorder)
    decoder_optimizer([torch.nn.Parameter(torch.zeros(2))], 1e-5, "cuda:0")
    assert seen == {"lr": 1e-5, "weight_decay": 0.0, "fused": True}


def test_a_cpu_tune_keeps_the_plain_kernel():
    opt = decoder_optimizer([torch.nn.Parameter(torch.zeros(2))], 1e-5, "cpu")
    assert isinstance(opt, torch.optim.AdamW) and not opt.defaults.get("fused")
    assert opt.defaults["lr"] == 1e-5 and opt.defaults["weight_decay"] == 0.0


def test_the_tune_builds_its_optimizer_there_and_records_it(tmp_path, monkeypatch):
    from test_decoder_validation import tune
    m, _ = tune(tmp_path, monkeypatch)
    assert m["optimizer"] == {"name": "AdamW", "fused": False}
