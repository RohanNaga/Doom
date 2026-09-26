"""CPU gates for the LoRA adapter (`lora.py`) on the three adapted backbones.

What has to hold, or an adaptation curve is unreadable:

* **Step 0 is the frozen model, exactly.** `lora_B` starts at zero, so a wrapped model returns the
  unwrapped model's output bit for bit on every backbone; the zero-shot score and the step-0 score of
  an adaptation are then the same number rather than two numbers that happen to be close.
* **Only the adapter and the selected parts train.** Everything else is frozen, and each fully trained
  part is individually switchable.
* **The counts are the arithmetic.** LoRA is rank x (in + out) per wrapped projection; each part is the
  size of the modules it names.
* **The base keys do not move**, so a source snapshot loads into a wrapped model unchanged, and the
  adapter state round-trips strictly.

The heavy warm starts are replaced by tiny configs of the same diffusers classes, as the other backbone
tests do.

    python -m pytest paper/fixtures/test_lora.py -q
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import backbones  # noqa: E402
import lora  # noqa: E402

BITS, HIST = 6, 2
TINY_UNET = dict(sample_size=8, block_out_channels=(8, 8), layers_per_block=1, in_channels=4,
                 out_channels=4, cross_attention_dim=16, attention_head_dim=2, norm_num_groups=8,
                 down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                 up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"))
TINY_PIXART = dict(num_attention_heads=2, attention_head_dim=8, in_channels=4, out_channels=8,
                   num_layers=2, caption_channels=32, sample_size=64, patch_size=2,
                   cross_attention_dim=16, use_additional_conditions=False, norm_num_groups=2)
TINY_SD3 = dict(sample_size=8, patch_size=2, in_channels=16, num_layers=2, attention_head_dim=8,
                num_attention_heads=2, joint_attention_dim=16, caption_projection_dim=16,
                pooled_projection_dim=8, out_channels=16, pos_embed_max_size=32, dual_attention_layers=(0,))


@pytest.fixture
def tiny_unet_hub(monkeypatch):
    from diffusers import UNet2DConditionModel
    monkeypatch.setattr(UNet2DConditionModel, "from_pretrained",
                        classmethod(lambda cls, *a, **k: cls(**TINY_UNET, num_class_embeds=k.get("num_class_embeds", 4))))


def build(backbone, history=HIST):
    """A tiny world model of one backbone with an executed-control history, as the next-tic rows have."""
    torch.manual_seed(0)
    kw = dict(num_actions=3, context_frames=HIST, noise_buckets=4, action_dropout=0.0, grad_ckpt=False,
              action_history=history, control_bits=BITS if history else 0)
    if backbone == "unet":
        return backbones.UNetWorldModel(**kw)
    if backbone == "pixart":
        from diffusers import PixArtTransformer2DModel
        return backbones.PixArtWorldModel(**kw, transformer=PixArtTransformer2DModel(**TINY_PIXART))
    from diffusers import SD3Transformer2DModel
    return backbones.SD35WorldModel(**kw, latent_channels=16, transformer=SD3Transformer2DModel(**TINY_SD3))


def inputs(model, n=2, seed=7, history=HIST):
    g = torch.Generator().manual_seed(seed)
    c = model.latent_channels
    action = (torch.randint(0, 2, (n, history, BITS), generator=g).float() if history
              else torch.tensor([1, 2])[:n])
    return dict(x=torch.randn(n, c, 32, 40, generator=g), t=torch.tensor([300, 700])[:n], action=action,
                context=torch.randn(n, c * model.context_frames, 32, 40, generator=g),
                noise_bucket=torch.tensor([0, 3])[:n])


BACKBONES = ("unet", "pixart", "sd35")


# ---------------------------------------------------------------------------------------
# the layer
# ---------------------------------------------------------------------------------------

def test_the_wrapped_layer_keeps_shape_keys_and_output_at_init():
    torch.manual_seed(0)
    base = nn.Linear(12, 5)
    x = torch.randn(3, 7, 12)
    ref = base(x)
    w = lora.LoRALinear(base, rank=4, alpha=8)
    assert w.weight is base.weight and w.bias is base.bias
    assert isinstance(w, nn.Linear)
    assert set(w.state_dict()) == {"weight", "bias", "lora_A", "lora_B"}
    assert w.lora_A.shape == (4, 12) and w.lora_B.shape == (5, 4)
    assert w.scaling == 2.0
    out = w(x)
    assert out.shape == ref.shape and torch.equal(out, ref), "B = 0 must return the base output exactly"


def test_the_adapter_path_is_scaled_by_alpha_over_rank():
    torch.manual_seed(0)
    base = nn.Linear(6, 3, bias=False)
    w = lora.LoRALinear(base, rank=2, alpha=6)
    with torch.no_grad():
        w.lora_B.fill_(0.5)
    x = torch.randn(4, 6)
    want = base(x) + 3.0 * (x @ w.lora_A.T @ w.lora_B.T)
    assert torch.allclose(w(x), want, atol=1e-6)


def test_the_initial_a_depends_on_the_seed_only():
    def a_of(seed, noise):
        torch.manual_seed(noise)      # the global generator must not matter
        m = nn.Sequential()
        m.attn = nn.Module()
        m.attn.to_q = nn.Linear(8, 8)
        lora.inject_lora(m, rank=2, alpha=2, seed=seed)
        return m.attn.to_q.lora_A.detach().clone()
    assert torch.equal(a_of(3, 0), a_of(3, 99))
    assert not torch.equal(a_of(3, 0), a_of(4, 0))


def test_a_layer_is_never_wrapped_twice():
    m = nn.Module()
    m.attn = nn.Module()
    m.attn.to_q = nn.Linear(4, 4)
    lora.inject_lora(m, rank=2)
    with pytest.raises(ValueError, match="already"):
        lora.inject_lora(m, rank=2)


# ---------------------------------------------------------------------------------------
# the three backbones
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("backbone", BACKBONES)
def test_step_zero_is_the_frozen_model_exactly(backbone, tiny_unet_hub):
    """Compared the way the evaluators compare: both models built the same way, nothing frozen.

    Freezing is not part of the comparison on purpose: it can change a CPU kernel's choice and move an
    output by one float32 ulp (SD 3.5's `to_add_out`), which is why `lora.apply_adapter` never freezes.
    """
    m = build(backbone).eval()
    kw = inputs(m)
    with torch.no_grad():
        before = m(**kw)
    names = lora.inject_lora(m, rank=4, alpha=4, include_mlp=True, seed=0)
    with torch.no_grad():
        after = m(**kw)
    assert names and torch.equal(before, after)
    lora.configure_trainable(m, backbone)
    with torch.no_grad():
        assert torch.allclose(m(**kw), before, rtol=0, atol=1e-5)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_every_attention_projection_is_wrapped_and_the_mlp_only_on_request(backbone, tiny_unet_hub):
    m = build(backbone)
    attn = lora.lora_target_names(m)
    both = lora.lora_target_names(m, include_mlp=True)
    for tail in (".to_q", ".to_k", ".to_v", ".to_out.0"):
        assert any(n.endswith(tail) for n in attn), tail
    assert not any(".ff." in n or ".ff_context." in n for n in attn)
    assert set(attn) < set(both) and all(".ff" in n for n in set(both) - set(attn))
    if backbone in ("unet", "pixart"):
        # self- and cross-attention both
        assert any(".attn1." in n for n in attn) and any(".attn2." in n for n in attn)
    else:
        # the joint attention's context stream and the dual-attention block's second attention
        assert any(n.endswith("add_q_proj") for n in attn) and any(n.endswith("to_add_out") for n in attn)
        assert any(".attn2." in n for n in attn)
    # nothing outside attention is wrapped: no timestep, caption, control or output layer
    assert not any(k in n for n in attn for k in ("time", "caption", "control", "proj_out", "norm", "context_embedder"))


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("parts", [lora.FULL_PARTS, ("control",), ("input_proj", "noise_emb"), ()])
def test_only_the_adapter_and_the_selected_parts_require_grad(backbone, parts, tiny_unet_hub):
    m = build(backbone)
    lora.inject_lora(m, rank=4, alpha=4, seed=0)
    names = lora.configure_trainable(m, backbone, parts)
    trained = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trained == set(names)
    want = {n for n, _ in lora.lora_parameters(m)}
    for part in parts:
        want |= {n for n, _ in lora.part_parameters(m, backbone, part)}
    assert trained == want
    for part in set(lora.FULL_PARTS) - set(parts):
        assert not any(p.requires_grad for _, p in lora.part_parameters(m, backbone, part))
    # the backbone proper stays frozen: every non-LoRA attention weight, every norm, the output layer
    assert not any(p.requires_grad for n, p in m.named_parameters()
                   if n.endswith((".to_q.weight", ".to_k.weight", ".to_v.weight")))


@pytest.mark.parametrize("backbone", BACKBONES)
def test_the_parts_name_the_modules_they_should(backbone, tiny_unet_hub):
    m = build(backbone)
    lora.inject_lora(m, rank=4, alpha=4)
    control = {n for n, _ in lora.part_parameters(m, backbone, "control")}
    assert {"control_history.pos", "control_history.mlp.0.weight", "control_history.mlp.2.weight"} <= control
    proj = {n for n, _ in lora.part_parameters(m, backbone, "input_proj")}
    noise = {n for n, _ in lora.part_parameters(m, backbone, "noise_emb")}
    if backbone == "unet":
        assert proj == {"unet.conv_in.weight", "unet.conv_in.bias"}
        assert noise == {"unet.class_embedding.weight"}
    else:
        assert proj == {"transformer.pos_embed.proj.weight", "transformer.pos_embed.proj.bias"}
        assert noise == {"bucket_embedder.weight"}
    if backbone == "sd35":
        assert {"pooled_control.weight", "pooled_control.bias", "pooled_base"} <= control
    # a single-action row trains its action table instead of a control MLP
    m1 = build(backbone, history=0)
    lora.inject_lora(m1, rank=4, alpha=4)
    assert any(n.startswith("action_embedder.") for n, _ in lora.part_parameters(m1, backbone, "control"))


@pytest.mark.parametrize("backbone", BACKBONES)
def test_the_parameter_counts_are_the_arithmetic(backbone, tiny_unet_hub):
    m = build(backbone)
    targets = lora.lora_target_names(m, include_mlp=True)
    sizes = {n: (mod.in_features, mod.out_features) for n, mod in m.named_modules() if n in targets}
    lora.inject_lora(m, rank=4, alpha=4, include_mlp=True)
    counts = lora.parameter_counts(m, backbone, ("control", "noise_emb"))
    assert counts["lora"] == sum(4 * (i + o) for i, o in sizes.values())
    ch = m.control_history
    control = ch.pos.numel() + sum(p.numel() for p in ch.mlp.parameters())
    if backbone == "sd35":
        control += sum(p.numel() for p in m.pooled_control.parameters()) + m.pooled_base.numel()
    assert counts["part_control"] == control
    conv = m.unet.conv_in if backbone == "unet" else m.transformer.pos_embed.proj
    assert counts["part_input_proj"] == conv.weight.numel() + conv.bias.numel()
    table = m.unet.class_embedding if backbone == "unet" else m.bucket_embedder
    assert counts["part_noise_emb"] == table.weight.numel()
    assert counts["trainable"] == counts["lora"] + counts["part_control"] + counts["part_noise_emb"]
    assert counts["total"] == sum(p.numel() for p in m.parameters())
    assert counts["backbone_frozen"] == counts["total"] - counts["trainable"]


@pytest.mark.parametrize("spec,want", [("", ()), ("none", ()), ("control", ("control",)),
                                       ("noise_emb,control,control", ("noise_emb", "control")), ("all", ("all",))])
def test_the_parts_flag_parses(spec, want):
    assert lora.parse_parts(spec) == want


def test_an_unknown_part_or_backbone_is_refused():
    with pytest.raises(ValueError, match="unknown full part"):
        lora.parse_parts("control,timestep")
    with pytest.raises(ValueError, match="alone"):
        lora.parse_parts("all,control")
    with pytest.raises(NotImplementedError):
        lora.check_backbone("dit")


@pytest.mark.parametrize("backbone", BACKBONES)
def test_rank_zero_with_every_part_is_the_full_fine_tune_reference(backbone, tiny_unet_hub):
    """Design decision 5 as a flag value: no adapter, every backbone parameter trained."""
    m = build(backbone)
    cfg = lora.adapter_config(backbone, 0, 0, 0.0, False, ("all",), seed=0)
    assert lora.apply_adapter(m, cfg) == [n for n, _ in m.named_parameters()]
    assert not any(isinstance(x, lora.LoRALinear) for x in m.modules())
    names = lora.configure_trainable(m, backbone, ("all",))
    assert all(p.requires_grad for p in m.parameters()) and len(names) == len(list(m.parameters()))
    counts = lora.parameter_counts(m, backbone, ("all",))
    assert counts["lora"] == 0 and counts["trainable"] == counts["total"] and counts["backbone_frozen"] == 0
    with pytest.raises(ValueError, match="nothing to train"):
        lora.trained_names(build(backbone), backbone, ())


# ---------------------------------------------------------------------------------------
# the adapter state
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("backbone", BACKBONES)
def test_the_source_keys_load_into_a_wrapped_model(backbone, tiny_unet_hub):
    src = build(backbone)
    sd = {k: v.clone() for k, v in src.state_dict().items()}
    m = build(backbone)
    lora.inject_lora(m, rank=4, alpha=4)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected and missing and all(k.rsplit(".", 1)[-1] in lora.LORA_PARAM_NAMES for k in missing)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_the_adapter_state_round_trips_strictly(backbone, tiny_unet_hub):
    cfg = lora.adapter_config(backbone, 4, 4, 0.0, True, lora.FULL_PARTS, seed=5)
    a = build(backbone).eval()
    cfg["targets"] = lora.inject_lora(a, 4, 4, 0.0, True, seed=5)
    lora.configure_trainable(a, backbone, lora.FULL_PARTS)
    with torch.no_grad():                    # what training does: move the adapter and every part
        for _, p in lora.trainable_parameters(a):
            p.add_(0.01 * torch.randn_like(p))
    state = lora.adapter_state(a)
    assert all(t.dtype == torch.float32 and t.device.type == "cpu" for t in state.values())
    b = build(backbone).eval()
    names = lora.apply_adapter(b, cfg)
    assert sorted(names) == sorted(state)
    lora.load_adapter_state(b, state, names)
    kw = inputs(a)
    with torch.no_grad():
        assert torch.allclose(a(**kw), b(**kw), rtol=0, atol=1e-5)
    # a rebuild without a part, or at another rank, is refused rather than loaded partially
    c = build(backbone)
    fewer = lora.apply_adapter(c, {**cfg, "parts": ["control"], "targets": None})
    with pytest.raises(ValueError, match="does not match"):
        lora.load_adapter_state(c, state, fewer)
    d = build(backbone)
    other = lora.apply_adapter(d, {**cfg, "rank": 2, "targets": None})
    with pytest.raises(ValueError, match="adapter tensor"):
        lora.load_adapter_state(d, state, other)
