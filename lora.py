"""
Low-rank adapters (LoRA) on a frozen world-model backbone, and the small parts trained in full beside them.

This is the adapter half of the per-map adaptation study (`docs/lora_adaptation_design_2026-09-26.html`):
`adapt_wm.py` trains it, `eval_tf.py` and `directional_check.py` read it back through
`load_adapter_checkpoint`. Nothing here is imported by `train_wm.py`, so pretraining is untouched.

**Why a module of our own and not `peft`.** `environment.yml` pins no `peft`, and neither the laptop env
nor the Superman `Doom` env is known to carry it; a new dependency on the day of a pilot is a risk with no
payoff, because the adapter is one matrix product. Owning it also keeps two properties easy to state:

  * `LoRALinear` subclasses `nn.Linear` and holds the base layer's OWN `weight` and `bias` Parameters, so
    the base keys of the state dict (`...attn1.to_q.weight`) do not move. A source snapshot therefore
    loads into a wrapped model unchanged, and `isinstance(m, nn.Linear)` checks inside diffusers pass.
  * `lora_B` starts at zero, so the wrapped layer returns `base(x) + 0` and step 0 of an adaptation is
    the frozen model exactly (not approximately): the zero-shot score and the step-0 score coincide.

**What LoRA wraps** (`ATTENTION_TARGETS`): every attention projection of the denoiser, query, key, value
and output, in self-attention and cross-attention alike (U-Net and PixArt `attn1` and `attn2`; SD 3.5's
joint attention including the context-stream projections `add_*_proj` and `to_add_out`, and its
dual-attention `attn2`). `include_mlp=True` also wraps the feed-forward projections (`MLP_TARGETS`), the
DiT-recipe precedent (Cosmos, DiffSynth, Flux); it is off by default.

**What is trained in full beside it** (`FULL_PARTS`), each switchable: `control` (the executed-control
MLP and its position table, or the action table of a single-action row, plus SD 3.5's pooled control
projection and base vector), `input_proj` (the inflated input projection, the whole conv: pretrained
target-channel kernel, context-channel kernel and bias) and `noise_emb` (the context-noise bucket table
that noise augmentation conditions on; the diffusion timestep embedding stays frozen).

The scale is `alpha / rank`, the LoRA paper's convention, so rank 16 and alpha 16 is a scale of 1.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_BACKBONES = ("unet", "pixart", "sd35")
ATTENTION_TARGETS = ("to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out")
MLP_TARGETS = ("ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2")
FULL_PARTS = ("control", "input_proj", "noise_emb")
DEFAULT_PARTS = FULL_PARTS

# The modules (or bare Parameters) each fully trained part consists of, per backbone. A name that the
# built model does not carry is skipped (the single-action tables are deleted under --action-history,
# the control MLP exists only under it), but every selected part must resolve to at least one tensor.
PART_MODULES = {
    "unet": {"control": ("control_history", "action_embedder"),
             "input_proj": ("unet.conv_in",),
             "noise_emb": ("unet.class_embedding",)},
    "pixart": {"control": ("control_history", "action_embedder"),
               "input_proj": ("transformer.pos_embed.proj",),
               "noise_emb": ("bucket_embedder",)},
    "sd35": {"control": ("control_history", "action_embedder", "pooled_control", "pooled_action", "pooled_base"),
             "input_proj": ("transformer.pos_embed.proj",),
             "noise_emb": ("bucket_embedder",)},
}

ADAPTER_KEY = "adapter"            # live adapter and part tensors in an adaptation checkpoint
ADAPTER_EMA_KEY = "adapter_ema"    # their fp32 EMA
LORA_PARAM_NAMES = ("lora_A", "lora_B")


class LoRALinear(nn.Linear):
    """`base(x) + (alpha / rank) * B(A(dropout(x)))` around an existing `nn.Linear`, B zero at init.

    The base layer's `weight` and `bias` are this module's own, so its state-dict keys are unchanged and
    the adapter adds exactly two: `lora_A` (rank, in) and `lora_B` (out, rank). `A` is drawn uniform in
    +-1/sqrt(in_features), which is `kaiming_uniform_(a=sqrt(5))`, the LoRA and peft default, from the
    CPU `generator` when one is given, so every rank of a multi-GPU run and every rebuild for evaluation
    draws the same `A`. The draw happens on the CPU and is then moved to the base weight's device.
    """

    def __init__(self, base, rank, alpha, dropout=0.0, generator=None):
        nn.Module.__init__(self)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear wraps an nn.Linear, got {type(base).__name__}")
        if isinstance(base, LoRALinear):
            raise ValueError("this layer already carries a LoRA adapter")
        if int(rank) < 1:
            raise ValueError(f"LoRA rank must be at least 1, got {rank}")
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight = base.weight
        if base.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = base.bias
        self.rank, self.alpha = int(rank), float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()
        bound = 1.0 / math.sqrt(self.in_features)
        a = torch.empty(self.rank, self.in_features, dtype=torch.float32, device="cpu")
        a.uniform_(-bound, bound, generator=generator)
        dev = base.weight.device
        self.lora_A = nn.Parameter(a.to(dev))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank, dtype=torch.float32, device="cpu").to(dev))

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        return out + F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B) * self.scaling

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, rank={self.rank}, alpha={self.alpha:g}")


def check_backbone(backbone):
    """Refuse a backbone this adapter path was not written for (the DiT and UniDiffuser use fused or
    non-diffusers attention modules whose projections these target names do not describe)."""
    if backbone not in SUPPORTED_BACKBONES:
        raise NotImplementedError(f"LoRA adaptation is implemented for {SUPPORTED_BACKBONES}, not {backbone}")


def _ends_with(name, target):
    parts, tail = name.split("."), target.split(".")
    return len(parts) > len(tail) and parts[-len(tail):] == tail


def lora_target_names(model, include_mlp=False):
    """Qualified names of the `nn.Linear` layers LoRA wraps, in module order."""
    targets = ATTENTION_TARGETS + (MLP_TARGETS if include_mlp else ())
    return [n for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and not isinstance(m, LoRALinear) and any(_ends_with(n, t) for t in targets)]


def _parent(model, name):
    head, _, leaf = name.rpartition(".")
    return (model.get_submodule(head) if head else model), leaf


def inject_lora(model, rank=16, alpha=16.0, dropout=0.0, include_mlp=False, seed=0):
    """Wrap every target projection in a `LoRALinear`, in module order; returns the wrapped names.

    `seed` seeds the one CPU generator all `A` matrices are drawn from, so the adapter's initial state
    is a function of the seed alone, independent of the global RNG, the process rank and the device.
    """
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError("this model already carries LoRA adapters; inject once")
    names = lora_target_names(model, include_mlp)
    if not names:
        raise ValueError("no attention projection (to_q/to_k/to_v/to_out.0) found to adapt")
    g = torch.Generator().manual_seed(int(seed))
    for name in names:
        parent, leaf = _parent(model, name)
        setattr(parent, leaf, LoRALinear(getattr(parent, leaf), rank, alpha, dropout, generator=g))
    return names


def lora_parameters(model):
    """[(name, Parameter)] of every LoRA factor in the model."""
    return [(n, p) for n, p in model.named_parameters() if n.rsplit(".", 1)[-1] in LORA_PARAM_NAMES]


def _under(name, prefix):
    return name == prefix or name.startswith(prefix + ".")


def part_parameters(model, backbone, part):
    """[(name, Parameter)] of one fully trained part (`FULL_PARTS`) of this backbone."""
    check_backbone(backbone)
    if part not in FULL_PARTS:
        raise ValueError(f"unknown part {part!r}; the parts are {FULL_PARTS}")
    prefixes = PART_MODULES[backbone][part]
    out = [(n, p) for n, p in model.named_parameters()
           if any(_under(n, pre) for pre in prefixes) and n.rsplit(".", 1)[-1] not in LORA_PARAM_NAMES]
    if not out:
        raise ValueError(f"part {part!r} resolves to no parameter on this {backbone} model (looked for {prefixes})")
    return out


def parse_parts(spec):
    """"control,input_proj" -> ("control", "input_proj"); "" or "none" -> (), LoRA only."""
    s = str(spec or "").strip()
    if s.lower() in ("", "none"):
        return ()
    parts = tuple(dict.fromkeys(x.strip() for x in s.split(",") if x.strip()))
    bad = [p for p in parts if p not in FULL_PARTS]
    if bad:
        raise ValueError(f"unknown full part(s) {bad}; choose from {FULL_PARTS} or 'none'")
    return parts


def parameter_counts(model, backbone, parts=DEFAULT_PARTS):
    """Tensor sizes the adapter path trains or leaves frozen, for the launch line and the log.

    Every part is counted whether it is selected or not (`part_<name>`), so a report can say what
    switching one on would cost; `trainable` is LoRA plus the selected parts.
    """
    check_backbone(backbone)
    lora = sum(p.numel() for _, p in lora_parameters(model))
    counts = {"lora": lora}
    for part in FULL_PARTS:
        counts[f"part_{part}"] = sum(p.numel() for _, p in part_parameters(model, backbone, part))
    counts["trainable"] = lora + sum(counts[f"part_{p}"] for p in parts)
    counts["total"] = sum(p.numel() for p in model.parameters())
    counts["backbone_frozen"] = counts["total"] - counts["trainable"]
    counts["trainable_fraction"] = counts["trainable"] / max(1, counts["total"] - lora)
    return counts


def trained_names(model, backbone, parts=DEFAULT_PARTS):
    """Names of the tensors an adaptation trains: every LoRA factor and the selected parts, in model order.

    Computed from the module structure alone, never from `requires_grad`, so an evaluator can load an
    adapter without freezing anything (see `apply_adapter` for why that matters).
    """
    check_backbone(backbone)
    found = lora_parameters(model)
    if not found:
        raise ValueError("the LoRA adapters have to be injected first")
    names = {n for n, _ in found}
    for part in parts:
        names |= {n for n, _ in part_parameters(model, backbone, part)}
    return [n for n, _ in model.named_parameters() if n in names]


def configure_trainable(model, backbone, parts=DEFAULT_PARTS):
    """Freeze everything, then unfreeze the LoRA factors and the selected parts. Returns the trained names.

    The trained set is the adapter state an adaptation checkpoint saves, in `named_parameters` order.
    """
    names = trained_names(model, backbone, parts)
    keep = set(names)
    model.requires_grad_(False)
    for n, p in model.named_parameters():
        if n in keep:
            p.requires_grad_(True)
    return names


def trainable_parameters(model):
    """[(name, Parameter)] that require grad, in `named_parameters` order."""
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def adapter_state(model, names=None):
    """{name: fp32 CPU copy} of the adapter and the fully trained parts (the trainable set by default)."""
    keep = set(names) if names is not None else {n for n, _ in trainable_parameters(model)}
    return {n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if n in keep}


def load_adapter_state(model, state, names):
    """Copy a saved adapter state into a model adapted the same way; strict on `names` and on shapes."""
    params = dict(model.named_parameters())
    want = set(names)
    missing, unexpected = sorted(want - set(state)), sorted(set(state) - want)
    if missing or unexpected:
        raise ValueError(f"adapter state does not match this adapted model: missing {missing[:6]}, "
                         f"unexpected {unexpected[:6]} (check rank, --lora-mlp and the full parts)")
    with torch.no_grad():
        for n, t in state.items():
            if tuple(params[n].shape) != tuple(t.shape):
                raise ValueError(f"adapter tensor {n}: saved {tuple(t.shape)}, model {tuple(params[n].shape)}")
            params[n].copy_(t.to(params[n].dtype))
    return sorted(state)


def adapter_config(backbone, rank, alpha, dropout, include_mlp, parts, seed, targets=None):
    """What an evaluator needs to rebuild the adapted graph exactly: stored in every adaptation checkpoint."""
    return {"backbone": backbone, "rank": int(rank), "alpha": float(alpha), "dropout": float(dropout),
            "include_mlp": bool(include_mlp), "parts": list(parts), "seed": int(seed),
            "targets": list(targets) if targets is not None else None}


def adapter_checkpoint(model, names, ema, config, source, source_args, step, **extra):
    """An adaptation checkpoint: the trained tensors live and EMA (fp32, CPU), how to rebuild them, and on what.

    `source` is {path, sha256, weights ("ema" or "live"), step} of the frozen snapshot; `source_args` is that
    run's own args, stored under `args` so every evaluator rebuilds the source's graph from it unchanged.
    No backbone weight is stored: the snapshot, checked by hash, supplies them.
    """
    missing = sorted(set(names) - set(ema))
    if missing:
        raise ValueError(f"the EMA does not cover trained tensor(s) {missing[:6]}")
    return {ADAPTER_KEY: adapter_state(model, names),
            ADAPTER_EMA_KEY: {n: ema[n].detach().float().cpu().clone() for n in names},
            "adapter_config": dict(config), "source": dict(source), "args": dict(source_args), "step": int(step),
            **extra}


def is_adapter_checkpoint(ck):
    """True for a checkpoint written by `adapt_wm.py`, which carries an adapter and not a full model."""
    return isinstance(ck, dict) and ck.get(ADAPTER_KEY) is not None and "adapter_config" in ck


def apply_adapter(model, cfg):
    """Inject LoRA exactly as the adaptation run did; returns the names of the tensors it trained.

    Nothing is frozen here. Freezing can change which kernel PyTorch picks for an op (on the CPU, SD 3.5's
    context-stream output projection `to_add_out`, fed a non-contiguous slice, moves by one float32 ulp
    once its weight stops requiring grad), so an evaluator that froze the base would not score the step-0
    adapter identically to the frozen model it is compared with. Evaluators build every model the same
    way and leave `requires_grad` alone; only the trainer calls `configure_trainable`.
    """
    check_backbone(cfg["backbone"])
    targets = inject_lora(model, cfg["rank"], cfg["alpha"], cfg["dropout"], cfg["include_mlp"], cfg["seed"])
    if cfg.get("targets") is not None and list(cfg["targets"]) != targets:
        raise ValueError(f"the adapted layers differ from the run's: {len(targets)} here, {len(cfg['targets'])} "
                         "recorded; the backbone was built differently")
    return trained_names(model, cfg["backbone"], cfg["parts"])


def load_source_weights(model, source_ck, weights):
    """The frozen source snapshot's weights into `model`: its EMA tensors (`weights="ema"`) or its live ones.

    `doomdit_utils.load_world_model_state` is the loader every zero-shot score was produced with, so the
    adapted model starts from exactly the weights those scores measured.
    """
    from doomdit_utils import load_world_model_state
    if weights not in ("ema", "live"):
        raise ValueError(f"source weights must be 'ema' or 'live', got {weights!r}")
    if weights == "ema" and not source_ck.get("ema"):
        raise SystemExit("--source-weights ema, but the source checkpoint carries no EMA tensors "
                         "(best.pt has none; use a snap_*.pt or a recovery checkpoint)")
    return load_world_model_state(model, source_ck, use_ema=(weights == "ema"))


def load_adapter_checkpoint(model, ck, use_ema=False, verify_source=True):
    """Rebuild an adaptation checkpoint on a freshly built backbone: source weights, adapter, trained parts.

    The source snapshot is the one the checkpoint names, checked against its recorded SHA-256 (the
    `<file>.sha256` cache makes the check free after the first time), loaded with the same weights choice
    the run trained from. `use_ema` selects the adapter's fp32 EMA instead of its live tensors; the
    frozen backbone is the same either way.
    """
    src = ck["source"]
    path = src["path"]
    if verify_source and src.get("sha256"):
        from eval_identity import sha256_file
        got = sha256_file(path)
        if got != src["sha256"]:
            raise SystemExit(f"the adaptation's source {path} has SHA-256 {got}, the run trained from "
                             f"{src['sha256']}; the frozen backbone is not the one the adapter was trained on")
    source_ck = torch.load(path, map_location="cpu", weights_only=False)
    load_source_weights(model, source_ck, src["weights"])
    del source_ck
    names = apply_adapter(model, ck["adapter_config"])
    key = ADAPTER_EMA_KEY if use_ema else ADAPTER_KEY
    if not ck.get(key):
        raise SystemExit(f"this adaptation checkpoint carries no {key}")
    load_adapter_state(model, ck[key], names)
    return key
