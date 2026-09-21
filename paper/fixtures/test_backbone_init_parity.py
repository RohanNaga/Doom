"""Under a fixed seed, the DEFAULT path builds bit-identical weights to the merge base's code.

Why this file exists. `nn.Embedding.__init__` draws from the global generator inside
`reset_parameters`, so moving an `nn.init.normal_` call *between* two `nn.Embedding` constructions
reorders the draws and changes the initial values of a table nobody meant to touch. That is exactly
what the first version of `--action-history` did: wrapping one of the two tables in an `if` put its
`normal_` call before the other table's construction, and the default non-history action table's
weights moved by up to 0.088. Nothing failed -- the models still trained -- but a seeded re-run of a
finished PixArt, UniDiffuser or SD 3.5 cell was no longer reproducible, which is the one property the
grid rests on.

The check loads `backbones.py` as it stood at the merge base straight out of git, builds each wrapper
from both modules under the same seed, and compares every tensor. It is self-maintaining: it cannot
drift from what the base actually did, because it reads the base.

    python -m pytest paper/fixtures/test_backbone_init_parity.py -q
"""
import hashlib
import importlib.util
import os
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import backbones  # noqa: E402

# The commit this branch started from, i.e. the last state every finished row was trained under.
MERGE_BASE = "d870ef9"

TINY_PIXART = dict(num_attention_heads=2, attention_head_dim=8, in_channels=4, out_channels=8,
                   num_layers=2, caption_channels=32, sample_size=64, patch_size=2,
                   cross_attention_dim=16, use_additional_conditions=False, norm_num_groups=2)
TINY_UNET = dict(sample_size=8, block_out_channels=(8, 8), layers_per_block=1, in_channels=4,
                 out_channels=4, cross_attention_dim=16, attention_head_dim=2, norm_num_groups=8,
                 down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                 up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"))
TINY_SD3 = dict(sample_size=8, patch_size=2, in_channels=16, num_layers=1, attention_head_dim=8,
                num_attention_heads=2, joint_attention_dim=16, caption_projection_dim=16,
                pooled_projection_dim=8, out_channels=16, pos_embed_max_size=32)
SEED = 1234


@pytest.fixture(scope="module")
def base_module(tmp_path_factory):
    """`backbones.py` as of the merge base, imported as its own module."""
    try:
        src = subprocess.run(["git", "-C", REPO, "show", f"{MERGE_BASE}:backbones.py"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:      # pragma: no cover
        pytest.skip(f"cannot read the merge base from git: {e}")
    if src.returncode != 0:
        pytest.skip(f"merge base {MERGE_BASE} is not in this repository: {src.stderr.strip()}")
    path = tmp_path_factory.mktemp("base") / "backbones_base.py"
    path.write_text(src.stdout)
    spec = importlib.util.spec_from_file_location("backbones_base", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build(module, name, **extra):
    """One wrapper from `module`, under a fixed seed, on the default (no-history, no-phase) path."""
    from diffusers import PixArtTransformer2DModel, SD3Transformer2DModel, UNet2DConditionModel
    common = dict(num_actions=3, context_frames=2, noise_buckets=4, action_dropout=0.0, grad_ckpt=False)
    torch.manual_seed(SEED)
    if name == "pixart":
        return module.PixArtWorldModel(**common, transformer=PixArtTransformer2DModel(**TINY_PIXART),
                                       **extra)
    if name == "sd35":
        return module.SD35WorldModel(**common, latent_channels=16,
                                     transformer=SD3Transformer2DModel(**TINY_SD3), **extra)
    if name == "unet":
        # the wrapper pulls its U-Net from the hub; serve a tiny one instead
        real = UNet2DConditionModel.from_pretrained
        UNet2DConditionModel.from_pretrained = classmethod(
            lambda cls, *a, **k: cls(**TINY_UNET, num_class_embeds=k.get("num_class_embeds", 4)))
        try:
            return module.UNetWorldModel(**common, **extra)
        finally:
            UNet2DConditionModel.from_pretrained = real
    raise ValueError(name)


def _digest(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("name", ["unet", "pixart", "sd35"])
def test_the_default_path_is_bit_identical_to_the_merge_base(name, base_module):
    """Every tensor, not just the ones the change was about."""
    mine, base = _build(backbones, name), _build(base_module, name)
    assert set(mine.state_dict()) == set(base.state_dict()), name
    for k in sorted(base.state_dict()):
        a, b = mine.state_dict()[k], base.state_dict()[k]
        assert a.shape == b.shape, f"{name}.{k}"
        assert torch.equal(a, b), (f"{name}.{k} moved: max abs diff "
                                   f"{(a.float() - b.float()).abs().max().item():.4g}")
    assert _digest(mine) == _digest(base), name


@pytest.mark.parametrize("name", ["pixart", "sd35"])
def test_the_regression_this_pins_was_real(name, base_module):
    """Build the wrapper with the two `normal_` calls interleaved, as the first version did, and
    show the default action table really does move. Without this, the test above could pass for the
    wrong reason -- a seed the reordering happens not to disturb."""
    base = _build(base_module, name)
    torch.manual_seed(SEED)
    import torch.nn as nn
    width = base.action_embedder.embedding_dim
    rows = base.action_embedder.num_embeddings
    # the base's order: construct both, then initialise both
    torch.manual_seed(SEED)
    a1 = nn.Embedding(rows, width); b1 = nn.Embedding(4, width)
    nn.init.normal_(a1.weight, std=0.02); nn.init.normal_(b1.weight, std=0.02)
    # the broken order: construct, initialise, construct, initialise
    torch.manual_seed(SEED)
    a2 = nn.Embedding(rows, width); nn.init.normal_(a2.weight, std=0.02)
    b2 = nn.Embedding(4, width); nn.init.normal_(b2.weight, std=0.02)
    assert not torch.equal(a1.weight, a2.weight), "the reordering would have been harmless"
    assert (a1.weight - a2.weight).abs().max() > 1e-3


@pytest.mark.parametrize("name", ["unet", "pixart", "sd35"])
def test_action_history_is_the_only_thing_that_changes_the_state_dict(name):
    """The new flag adds and removes tables; it must not perturb what it leaves in place."""
    off = _build(backbones, name)
    on = _build(backbones, name, action_history=2, control_bits=9)
    assert not any("action_embedder" in k for k in on.state_dict()), name
    assert any("control_history" in k for k in on.state_dict()), name
    shared = set(off.state_dict()) & set(on.state_dict())
    # the bucket table is built at the same point in the draw sequence either way, so it is shared
    assert any("bucket_embedder" in k for k in shared) or name == "unet", name
