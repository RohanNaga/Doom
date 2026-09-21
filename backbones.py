"""
The five world-model backbones behind one interface.

    model(x_noisy, t, action, context, noise_bucket[, phase]) -> v prediction, (B, C, 32, 40)

All five take the same C*(L+1)-channel input (L context latents channel-stacked, then the noisy
target), the same discrete action, the same context-noise bucket, and predict the same
C-channel velocity, where C is the autoencoder's latent channel count (`latent_channels`, 4 for
the SD KL-f8 rows, 16 for SD 3.5's autoencoder). Where the conditioning enters is each
backbone's native pathway: adaLN for the DiT, cross-attention (action) plus class embedding
(noise bucket) for the U-Net, GameNGen's construction, cross-attention on two learned caption
tokens for PixArt, two learned text tokens in UniDiffuser's joint token sequence, and two
joint-attention context tokens plus the pooled projection for SD 3.5. All five warm-start
from published weights with the input projection inflated: pretrained weights on the C target
channels, zeros on the context channels, so step 0 equals the pretrained model applied to the
noisy target.

What PixArt-alpha needed on top of that, and why:

* Patch projection. PixArt's patch conv lives at `pos_embed.proj`. Same inflation as the other
  two backbones: a new 132-channel conv holding the pretrained kernel on the last 4 input
  channels, zeros on the 128 context channels, and the pretrained bias.
  `register_to_config(in_channels=132)` keeps the config a checkpoint carries in step with the
  weights.
* Positional embedding. The 512x512 checkpoint is a 64x64 latent, a 32x32 token grid at patch 2,
  and diffusers keeps the sin-cos table as a NON-persistent buffer. Our latent is 32x40, a 16x20
  grid. We keep the checkpoint's `base_size=32` and `interpolation_scale=1` and precompute
  diffusers' own `get_2d_sincos_pos_embed` for grid (16, 20) into that buffer, then set
  `pos_embed.height/width` to 16/20 so `PatchEmbed.forward` takes the cached-buffer branch
  instead of regenerating the identical table on every call. The table is the same one the
  runtime branch would build; only the recomputation goes away. Being non-persistent, the buffer
  never enters the state dict, so save/load is unaffected.
* Conditioning. PixArt has no class-label path; its only non-timestep conditioning is the T5
  caption sequence. Under the default `action_inject="token"` we feed two learned tokens through the
  pretrained caption projection: an action embedding with a trailing null row for CFG dropout (same
  `action_dropout` semantics as the U-Net) and a context-noise-bucket embedding, both of width
  `caption_channels` (4096). The timestep keeps PixArt's own adaLN-single path untouched, and
  `use_additional_conditions` is forced False so no resolution or aspect-ratio embeddings are required.
  `action_inject="adaln"` is the grid's injection cell: the two embeddings, now `inner_dim` wide and
  zero-initialised, are *added* to the embedded timestep that feeds adaLN-single and the final
  shift/scale (`AdaLNActionInjection`), and cross-attention sees one learned action-free caption token
  instead. Nothing is added to the attention pathway, which is the point of the comparison.
* Output head. PixArt is a learn-sigma model: 8 output channels, epsilon first and the learned
  variance second (`PixArtAlphaPipeline` takes `chunk(2, dim=1)[0]`). We return the first 4 after
  unpatchify. Those are the pretrained epsilon channels, retrained here toward velocity, exactly
  as we treat DiT-XL/2's head.

What UniDiffuser v1 needed is documented on `UniDiffuserWorldModel` itself: it is the one backbone whose
positional table is learned (sliced and interpolated rather than regenerated), whose conditioning is a token
sequence rather than adaLN or cross-attention, and whose original forward cannot unpatchify a rectangular grid.

What SD 3.5 Medium needed is documented on `SD35WorldModel`: it is the one backbone in a 16-channel latent
space, so it is the reason `latent_channels` exists as an argument instead of a literal 4.
"""
import torch
import torch.nn as nn

from models import DiT_models, get_2d_sincos_pos_embed

LATENT_HW = (32, 40)
LATENT_CHANNELS = 4          # SD KL-f8; SD 3.5's autoencoder is 16 and passes latent_channels=16
PIXART_DEFAULT = "PixArt-alpha/PixArt-XL-2-512x512"
UNIDIFFUSER_DEFAULT = "thu-ml/unidiffuser-v1"
SD35_DEFAULT = "stabilityai/stable-diffusion-3.5-medium"

# The latent space each warm start was pretrained in. A backbone cannot be built in any other one:
# its patch projection has exactly this many input channels (see `inflate_input_conv`).
BACKBONE_LATENT_CHANNELS = {"dit": 4, "unet": 4, "pixart": 4, "unidiffuser": 4, "sd35": 16}
BACKBONES = tuple(BACKBONE_LATENT_CHANNELS)


WARM_START_NONE = "none"
ACTION_INJECTIONS = ("token", "adaln")


def resolve_latent_channels(backbone, latent_channels=None):
    """Latent channel count for a backbone: the caller's value, else the warm start's native one."""
    return BACKBONE_LATENT_CHANNELS[backbone] if latent_channels in (None, 0) else int(latent_channels)


def resolve_warm_start(backbone, warm_start):
    """(source, scratch) for `build_model`, resolving `--warm-start none`.

    "none" means the same architecture with a random init, which is the grid's scratch cell. The
    DiT is built from local code, so scratch is simply naming no checkpoint. PixArt still needs
    its repo, but only for the config that fixes the shapes; no weights are read from it. The
    other diffusers backbones have no scratch path here, so asking for one is an error rather
    than a silent warm start.
    """
    if not (isinstance(warm_start, str) and warm_start.strip().lower() == WARM_START_NONE):
        return warm_start, False
    if backbone == "dit":
        return None, True
    if backbone == "pixart":
        return PIXART_DEFAULT, True
    raise NotImplementedError(f"--warm-start {WARM_START_NONE} is implemented for dit and pixart, not {backbone}")


class AdaLNActionInjection(nn.Module):
    """Adds a per-sample conditioning vector to PixArt's timestep embedding.

    `AdaLayerNormSingle.emb` produces the embedded timestep that feeds both every block's
    adaLN-single modulation and the final layer's shift and scale, so adding to its output is the
    additive injection Nano World Models prices at zero new attention parameters. diffusers calls
    `emb` itself and takes no extra argument, so the world model's forward stashes the vector here
    and clears it again; the wrapper registers no parameter or buffer of its own.
    """

    def __init__(self, emb):
        super().__init__()
        self.emb = emb
        self.extra = None

    def forward(self, *args, **kwargs):
        out = self.emb(*args, **kwargs)
        return out if self.extra is None else out + self.extra.to(out.dtype)


def add_phase_embedder(module, phase_buckets, width, zero_init=False):
    """Attach a `tics_since_decision` table next to the noise-bucket table, or nothing at all.

    `phase_buckets == 0`, the default everywhere, attaches no module: no parameter, no state-dict
    entry, and no draw from the RNG, so a model built without phase conditioning is bit-identical
    to one built before this argument existed. When it is on, the table is created *after* every
    other embedding in the constructor, so the other tables' initial values do not move either and
    the only difference between the two models is the extra table.

    The phase enters wherever the noise bucket already enters, because both are the same kind of
    signal: a small discrete scalar the network has to read alongside the action.
    """
    if not phase_buckets:
        return None
    emb = nn.Embedding(int(phase_buckets), width)
    if zero_init:
        nn.init.zeros_(emb.weight)
    else:
        nn.init.normal_(emb.weight, std=0.02)
    module.phase_embedder = emb
    return emb


def phase_vec(module, phase):
    """This batch's phase embedding, or None when the model carries no phase table."""
    emb = getattr(module, "phase_embedder", None)
    if emb is None:
        return None
    if phase is None:
        raise ValueError("this model was built with phase conditioning, so forward() needs the "
                         "tics_since_decision tensor as its sixth argument")
    return emb(phase)


class ControlHistoryEmbedder(nn.Module):
    """GameNGen's action conditioning: one token per context tic, from the EXECUTED button vector.

    GameNGen "learn[s] an embedding A_emb from each action into a single token and replace[s] the
    cross attention from the text into this encoded actions sequence", with one token per context
    frame, so a held action appears on each of its four tics. Ours differs in one respect on
    purpose: the token encodes the button vector the engine actually executed on that tic, not
    Arnold's requested action id. A per-tic corpus keeps every tic, including the up-to-40-tic
    anti-stuck overrides during which the executed vector is not the canonical vector of the
    requested id, so the id would be a wrong label on exactly those tics.

    A two-layer MLP lifts the `bits`-wide 0/1 vector to the backbone's conditioning width, and a
    learned position embedding makes the order available to attention. The newest token is `u_t`,
    the control applied from the last context frame into the frame being predicted.
    """

    def __init__(self, bits, width, length):
        super().__init__()
        self.bits, self.width, self.length = int(bits), int(width), int(length)
        self.mlp = nn.Sequential(nn.Linear(self.bits, self.width), nn.SiLU(), nn.Linear(self.width, self.width))
        self.pos = nn.Parameter(torch.zeros(1, self.length, self.width))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, controls):
        """(B, L, bits) -> (B, L, width), oldest first."""
        if controls.ndim != 3 or controls.shape[1] != self.length or controls.shape[2] != self.bits:
            raise ValueError(f"expected controls of shape (B, {self.length}, {self.bits}), got "
                             f"{tuple(controls.shape)}; --action-history must match the dataset's")
        return self.mlp(controls.to(self.pos.dtype)) + self.pos


def add_control_history(module, action_history, control_bits, width):
    """Attach the executed-control token embedder, or nothing at all.

    `action_history == 0`, the default, attaches no module and draws no RNG, so the single-action
    path every finished row trained under is bit-identical.
    """
    if not action_history:
        return None
    if not control_bits:
        raise ValueError("--action-history needs the button-vector width (--control-bits, or read "
                         "from the corpus with doom_data.corpus_control_bits)")
    module.control_history = ControlHistoryEmbedder(control_bits, width, action_history)
    return module.control_history


def token_conditioning(module, action, noise_bucket, phase=None):
    """The conditioning token sequence for a backbone that feeds [action, bucket] through attention.

    Order, oldest control first: the L executed-control tokens (or the single action token when
    `--action-history 0`), then the noise-bucket token, then the phase token if there is one. The
    bucket stays last of the non-phase tokens so the single-action layout is the L=1 case of this
    one and nothing about the existing rows' token order changes.
    """
    ch = getattr(module, "control_history", None)
    if ch is not None:
        parts = [ch(action)]
    else:
        parts = [module.action_embedder(action).unsqueeze(1)]
    parts.append(module.bucket_embedder(noise_bucket).unsqueeze(1))
    pv = phase_vec(module, phase)
    if pv is not None:
        parts.append(pv.unsqueeze(1))
    return torch.cat(parts, dim=1)


def drop_actions(module, action):
    """Classifier-free-guidance dropout of the action id, or a refusal when there is nothing to drop.

    The null row is a row of the action *table*; an executed button vector has no null id, so
    action dropout and `--action-history` are incompatible rather than silently a no-op.
    """
    if getattr(module, "control_history", None) is not None:
        if module.action_dropout > 0:
            raise ValueError("--action-dropout has no meaning with --action-history: the conditioning is "
                             "a button vector, not an id with a null row. Set --action-dropout 0")
        return action
    if module.training and module.action_dropout > 0:
        drop = torch.rand(action.shape[0], device=action.device) < module.action_dropout
        return torch.where(drop, torch.full_like(action, module.num_actions), action)
    return action


def stacked_in_channels(latent_channels, context_frames):
    """Input channel count: L context latents channel-stacked, then the noisy target."""
    return latent_channels * (context_frames + 1)


def inflate_input_conv(old, in_channels, latent_channels):
    """The shared warm-start trick: a conv over `in_channels` inputs holding the pretrained kernel on
    the last `latent_channels` (the noisy target) and zeros on the context channels, plus the
    pretrained bias, so step 0 equals the pretrained model applied to the noisy target alone.

    The pretrained conv must already take exactly `latent_channels` inputs, which is what ties a
    backbone to the latent space it was trained in.
    """
    if old.weight.shape[1] != latent_channels:
        raise ValueError(f"pretrained input projection takes {old.weight.shape[1]} channels, "
                         f"latent_channels={latent_channels}")
    new = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding,
                    bias=old.bias is not None)
    with torch.no_grad():
        new.weight.zero_(); new.weight[:, -latent_channels:] = old.weight
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new


class DiTWorldModel(nn.Module):
    def __init__(self, num_actions=29, context_frames=32, noise_buckets=10, model_name="DiT-XL/2",
                 action_dropout=0.1, grad_ckpt=True, latent_channels=LATENT_CHANNELS, phase_buckets=0,
                 action_history=0, control_bits=0):
        super().__init__()
        self.context_frames = context_frames
        self.latent_channels = latent_channels
        self.num_actions = num_actions
        self.action_dropout = action_dropout
        self.dit = DiT_models[model_name](input_size=LATENT_HW,
                                          in_channels=stacked_in_channels(latent_channels, context_frames),
                                          pred_channels=latent_channels, num_classes=num_actions,
                                          class_dropout_prob=action_dropout, learn_sigma=False)
        self.dit.use_grad_ckpt = grad_ckpt
        hidden = self.dit.t_embedder.mlp[-1].out_features
        self.noise_embedder = nn.Embedding(noise_buckets, hidden)
        nn.init.normal_(self.noise_embedder.weight, std=0.02)
        add_phase_embedder(self, phase_buckets, hidden)
        # The DiT has no cross-attention pathway, so an action *sequence* cannot enter the way it
        # does in the other four backbones; the position-embedded control tokens are averaged into
        # the adaLN vector instead. That is NOT GameNGen's construction, and this backbone is here
        # for fit checks rather than as a next-tic row.
        add_control_history(self, action_history, control_bits, hidden)
        if action_history:
            # the DiT's class table is part of the pretrained DiT module and cannot be skipped, so
            # it is frozen: an unused parameter that still requires grad makes DDP fail on the
            # second iteration unless unused-parameter detection is on, which costs throughput
            self.dit.y_embedder.requires_grad_(False)

    def forward(self, x, t, action, context, noise_bucket, phase=None):
        d = self.dit
        h = d.x_embedder(torch.cat([context, x], dim=1)) + d.pos_embed
        ch = getattr(self, "control_history", None)
        if ch is None:
            act_c = d.y_embedder(action, self.training)   # the DiT table does its own CFG dropout
        else:
            act_c = ch(drop_actions(self, action)).mean(dim=1)
        c = d.t_embedder(t) + act_c + self.noise_embedder(noise_bucket)
        pv = phase_vec(self, phase)
        if pv is not None:
            c = c + pv                       # summed into adaLN, exactly as the noise bucket is
        for block in d.blocks:
            if d.use_grad_ckpt and self.training:
                h = torch.utils.checkpoint.checkpoint(d.ckpt_wrapper(block), h, c, use_reentrant=False)
            else:
                h = block(h, c)
        return d.unpatchify(d.final_layer(h, c))

    def load_imagenet_warm_start(self, path):
        """ImageNet DiT-XL/2-256 checkpoint: inflate the patch projection, rebuild pos_embed for the
        16x20 grid, keep the eps half of the 8-channel output head, skip the class table.

        The checkpoint is a 4-channel SD-KL-f8 model, so its patch projection and output head only
        line up with `latent_channels == 4`; any other latent space has no ImageNet warm start.
        """
        if self.latent_channels != 4:
            raise ValueError(f"the ImageNet DiT-XL/2 checkpoint is 4-channel; latent_channels={self.latent_channels}")
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("ema", sd.get("model", sd))
        own = self.dit.state_dict()
        loaded, skipped = 0, []
        for k, v in sd.items():
            if k not in own:
                skipped.append(k); continue
            if k == "x_embedder.proj.weight":
                w = torch.zeros_like(own[k]); w[:, -4:] = v; own[k] = w
            elif k == "final_layer.linear.weight" or k == "final_layer.linear.bias":
                p = self.dit.patch_size
                rows = [(pi * p + pj) * 8 + ci for pi in range(p) for pj in range(p) for ci in range(4)]
                own[k] = v[rows]
            elif k == "pos_embed":
                continue
            elif own[k].shape != v.shape:
                skipped.append(k); continue
            else:
                own[k] = v
            loaded += 1
        self.dit.load_state_dict(own)
        pe = get_2d_sincos_pos_embed(self.dit.pos_embed.shape[-1], self.dit.x_embedder.grid_size)
        self.dit.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))
        return loaded, skipped


class UNetWorldModel(nn.Module):
    def __init__(self, num_actions=29, context_frames=32, noise_buckets=10, sd_path="CompVis/stable-diffusion-v1-4",
                 action_dropout=0.1, grad_ckpt=True, cache_dir=None, latent_channels=LATENT_CHANNELS,
                 phase_buckets=0, action_history=0, control_bits=0):
        super().__init__()
        from diffusers import UNet2DConditionModel
        self.context_frames = context_frames
        self.num_actions = num_actions
        self.action_dropout = action_dropout
        self.latent_channels = latent_channels
        in_channels = stacked_in_channels(latent_channels, context_frames)
        self.unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet", cache_dir=cache_dir,
                                                         num_class_embeds=noise_buckets, low_cpu_mem_usage=False)
        self.unet.conv_in = inflate_input_conv(self.unet.conv_in, in_channels, latent_channels)
        self.unet.register_to_config(in_channels=in_channels)
        if not action_history:
            # skipped entirely under --action-history: the control tokens replace it, and a
            # trainable-but-unused table would break DDP on iteration two
            self.action_embedder = nn.Embedding(num_actions + 1, self.unet.config.cross_attention_dim)  # last id = null
            nn.init.normal_(self.action_embedder.weight, std=0.02)
        # the U-Net's single class-embedding slot already carries the noise bucket, so the phase
        # goes in the other conditioning pathway it has: a second cross-attention token
        add_phase_embedder(self, phase_buckets, self.unet.config.cross_attention_dim)
        # GameNGen's own backbone and its own construction: the cross-attention sequence that used to
        # carry text carries one token per context tic. The noise bucket stays in the class-embedding
        # slot, where it already is.
        add_control_history(self, action_history, control_bits, self.unet.config.cross_attention_dim)
        if grad_ckpt:
            self.unet.enable_gradient_checkpointing()

    def forward(self, x, t, action, context, noise_bucket, phase=None):
        action = drop_actions(self, action)
        ch = getattr(self, "control_history", None)
        tokens = ch(action) if ch is not None else self.action_embedder(action).unsqueeze(1)   # (B, L or 1, 768)
        pv = phase_vec(self, phase)
        if pv is not None:
            tokens = torch.cat([tokens, pv.unsqueeze(1)], dim=1)
        return self.unet(torch.cat([context, x], dim=1), t, encoder_hidden_states=tokens,
                         class_labels=noise_bucket).sample


class PixArtWorldModel(nn.Module):
    """PixArt-alpha DiT (611M, patch 2, adaLN-single) adapted to the action-conditioned world model.

    See the module docstring for what was changed relative to the published checkpoint.
    """

    def __init__(self, num_actions=29, context_frames=32, noise_buckets=10, pixart_path=PIXART_DEFAULT,
                 action_dropout=0.1, grad_ckpt=True, cache_dir=None, latent_channels=LATENT_CHANNELS,
                 scratch=False, action_inject="token", transformer=None, phase_buckets=0,
                 action_history=0, control_bits=0):
        super().__init__()
        from diffusers import PixArtTransformer2DModel
        from diffusers.models.embeddings import get_2d_sincos_pos_embed as diffusers_pos_embed
        if action_inject not in ACTION_INJECTIONS:
            raise ValueError(f"action injection must be one of {ACTION_INJECTIONS}, got {action_inject}")
        self.context_frames = context_frames
        self.num_actions = num_actions
        self.action_dropout = action_dropout
        self.latent_channels = latent_channels
        self.action_inject = action_inject
        in_channels = stacked_in_channels(latent_channels, context_frames)
        if transformer is not None:
            # a pre-built transformer: how the CPU gate exercises this wrapper without the 611M checkpoint
            self.transformer = transformer
        elif scratch:
            # same architecture, random init: the config fixes every shape, the checkpoint is never read
            cfg = PixArtTransformer2DModel.load_config(pixart_path, subfolder="transformer", cache_dir=cache_dir)
            self.transformer = PixArtTransformer2DModel.from_config({**cfg, "use_additional_conditions": False})
        else:
            self.transformer = PixArtTransformer2DModel.from_pretrained(
                pixart_path, subfolder="transformer", cache_dir=cache_dir, low_cpu_mem_usage=False,
                use_additional_conditions=False)

        patch = self.transformer.pos_embed
        if scratch:
            # nothing pretrained to preserve, so a plain conv over all in_channels under the default
            # init, rather than the pretrained kernel on the target slice and zeros on the context
            old = patch.proj
            patch.proj = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding,
                                   bias=old.bias is not None)
        else:
            patch.proj = inflate_input_conv(patch.proj, in_channels, latent_channels)
        self.transformer.register_to_config(in_channels=in_channels)

        # fixed table for our 16x20 token grid, built the way PatchEmbed would build it at runtime
        grid = (LATENT_HW[0] // patch.patch_size, LATENT_HW[1] // patch.patch_size)
        # diffusers < 0.32 returns numpy and has no output_type argument; >= 0.33 raises unless output_type="pt"
        import inspect
        extra = {"output_type": "pt"} if "output_type" in inspect.signature(diffusers_pos_embed).parameters else {}
        pe = torch.as_tensor(diffusers_pos_embed(patch.pos_embed.shape[-1], grid, base_size=patch.base_size,
                                                 interpolation_scale=patch.interpolation_scale, **extra))
        patch.register_buffer("pos_embed", pe.float().unsqueeze(0), persistent=False)
        patch.height, patch.width = grid

        caption_channels = self.transformer.config.caption_channels
        if action_inject == "token":
            if not action_history:      # the control tokens replace it; see UNetWorldModel
                self.action_embedder = nn.Embedding(num_actions + 1, caption_channels)   # last id = null
                nn.init.normal_(self.action_embedder.weight, std=0.02)
            self.bucket_embedder = nn.Embedding(noise_buckets, caption_channels)
            nn.init.normal_(self.bucket_embedder.weight, std=0.02)
        else:
            inner = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
            self.action_embedder = nn.Embedding(num_actions + 1, inner)
            self.bucket_embedder = nn.Embedding(noise_buckets, inner)
            # zero init keeps step 0 equal to the pretrained model on the noisy target, the property the
            # token path gets from the zeroed context channels. The gradient through adaLN-single's SiLU
            # is nonzero at zero, so both tables still train from the first update.
            nn.init.zeros_(self.action_embedder.weight)
            nn.init.zeros_(self.bucket_embedder.weight)
            # cross-attention stays alive but action-free: one learned caption token, zero at init
            self.null_caption = nn.Parameter(torch.zeros(1, 1, caption_channels))
            self.transformer.adaln_single.emb = AdaLNActionInjection(self.transformer.adaln_single.emb)
        # the phase follows the bucket into whichever pathway this cell puts it: a third caption
        # token under "token", a third summand into adaLN-single (zero-initialised) under "adaln"
        add_phase_embedder(self, phase_buckets,
                           caption_channels if action_inject == "token" else self.bucket_embedder.embedding_dim,
                           zero_init=(action_inject != "token"))
        if action_history and action_inject != "token":
            raise NotImplementedError("--action-history feeds a token sequence through cross-attention, which "
                                      "is exactly the pathway --action-inject adaln removes")
        add_control_history(self, action_history, control_bits, caption_channels)
        if grad_ckpt:
            self.transformer.enable_gradient_checkpointing()

    def forward(self, x, t, action, context, noise_bucket, phase=None):
        action = drop_actions(self, action)
        if self.action_inject == "token":
            tokens = token_conditioning(self, action, noise_bucket, phase)
        else:
            pv = phase_vec(self, phase)
            # adaln_single runs once, before any checkpointed block, so the stash is read before a recompute
            extra = self.action_embedder(action) + self.bucket_embedder(noise_bucket)
            self.transformer.adaln_single.emb.extra = extra if pv is None else extra + pv
            tokens = self.null_caption.expand(x.shape[0], -1, -1)
        try:
            # older diffusers unpack added_cond_kwargs unconditionally; both keys are ignored with use_additional_conditions=False
            out = self.transformer(torch.cat([context, x], dim=1), encoder_hidden_states=tokens, timestep=t,
                                   encoder_attention_mask=None,
                                   added_cond_kwargs={"resolution": None, "aspect_ratio": None}).sample
        finally:
            if self.action_inject == "adaln":
                self.transformer.adaln_single.emb.extra = None
        return out[:, :self.latent_channels]   # epsilon half of the learn-sigma head, retrained as velocity


class UniDiffuserWorldModel(nn.Module):
    """UniDiffuser v1 U-ViT (952M, patch 2, post-LN, joint image/CLIP/text token sequence) adapted to the world model.

    Source facts, read Sep 16 2026 from the Hub repo `thu-ml/unidiffuser-v1` and diffusers 0.31/0.40
    (`pipelines/unidiffuser/modeling_uvit.py`, `pipeline_unidiffuser.py`; identical apart from import paths):

    * Transformer: subfolder `unet` (`unet/config.json`: `_class_name: UniDiffuserModel`,
      `diffusion_pytorch_model.bin`, 3.81 GB fp32). 30 `num_layers` become 15 in-blocks, 1 mid block, 15
      out-blocks with U-Net style skips; width 24 x 64 = 1536; gelu; `pre_layer_norm: false`;
      `cross_attention_dim: null`, so every conditioning signal is a token in one self-attention sequence;
      `text_dim: 64`; `clip_img_dim: 512`; `num_text_tokens: 77`; `sample_size: 64`; `patch_size: 2`;
      `use_data_type_embedding: true`; `use_timestep_embedding: false` (the timestep tokens are the raw
      1536-dim sinusoidal projections, no MLP); `use_patch_pos_embed: false`.
    * Autoencoder: subfolder `vae` is `AutoencoderKL`, `latent_channels 4`, `scaling_factor 0.18215`,
      `block_out_channels [128, 256, 512, 512]`: the Stable Diffusion KL-f8 autoencoder (the card: "a
      pretrained image autoencoder from Stable Diffusion", `autoencoder_kl.pth` "converted from Stable
      Diffusion"). Our latents come from `sd-vae-ft-mse`, whose fine-tune touched only the decoder, so the
      encoder and the latent space are the ones this transformer was trained in. Its schedule is SD's
      scaled_linear 0.00085 to 0.012 (`scheduler/scheduler_config.json`), ours the linear 1e-4 to 0.02, the same
      mismatch the U-Net and PixArt rows carry.
    * Sequence layout (`UniDiffuserModel.forward`, steps 1.4 and 1.5): [t_img (1), t_text (1), data_type (1),
      text (77), clip_img (1), vae_img (1024 = 32x32 at 512 px)]. The learned table `pos_embed` is
      (1, 1104, 1536) laid out as [t_img, t_text, text 77, clip 1, img 1024]; `data_type_pos_embed_token`
      (1, 1, 1536) is spliced in after the first two rows at forward time.
    * Text input width: the pipeline reduces the 768-dim CLIP-L hidden states to 64 with `text_decoder.encode`
      (`reduce_text_emb_dim` is true because `prefix_hidden_dim 64 < hidden_size 768`, `__call__` step 3), so
      the model sees (B, 77, 64) through `text_in: Linear(64, 1536)`.
    * Timesteps (`_get_noise_pred`, mode "text2img"): `timestep_img=t, timestep_text=0, data_type=data_type`
      with `data_type=1` the `__call__` default. The CLIP image embedding is NOT marginalised in text2img: it is
      generated jointly with the VAE latent under the same `timestep_img` (`_split` / `_combine`). The only
      marginalisation convention in the pipeline (`img2text` CFG branch, mode "text") puts BOTH image
      modalities at `max_timestep = 1000` with Gaussian inputs, which cannot apply here because the VAE latent
      must sit at timestep t.
    * Output: `vae_img_out: Linear(1536, 16)` is patch 2x2 x 4 channels, epsilon prediction
      (`prediction_type: epsilon`), no learned variance; `clip_img_out` and `text_out` are the other modalities.

    Adaptation, in the order the tensors are touched:

    * Patch projection. `vae_img_in.proj` (Conv2d 4 -> 1536, kernel 2, stride 2) becomes a 132-channel conv with
      the pretrained kernel on the last 4 input channels, zeros on the 128 context channels, bias kept, so
      step 0 equals the pretrained model applied to the noisy target. `register_to_config(in_channels=132)`.
    * Positional table. Learned, so there is no formula to regenerate it at a new grid as the DiT and PixArt
      sin-cos tables allow. Kept: the two timestep rows, the first `cond_tokens` (2) of the 77 text rows (the
      rows every training caption filled: BOS and the first word), the CLIP row. The 1024 image rows are
      reshaped to a 32x32 map of 1536-vectors, bilinearly interpolated (`align_corners=False`) to our 16x20
      token grid, and flattened in the same row-major order. Interpolation rather than slicing because the
      attention patterns were learned against each token's relative place in the frame; slicing a 16x20
      corner would tell the model our whole frame is the top-left quadrant of a 512 px image. It is the standard
      ViT resolution change (DeiT). Cost: the interpolated rows are blends the model never saw, the row spacing
      halves vertically and shrinks 1.6x horizontally, and the aspect ratio moves from 1:1 to 4:5, so the
      step-0 function at 16x20 is not the pretrained function (no function on that grid is); exactness holds
      only at the native grid, where the interpolation is the identity and the parity gate runs.
    * Conditioning. Action (num_actions + 1 rows, last is the null row for dropout) and noise-bucket tables of
      width `text_dim` (64) feed the pretrained `text_in` as the two text tokens with `timestep_text = 0`, the
      text2img convention for a clean conditioning modality, and `data_type = 1`, the pipeline default.
      Attention is full over the sequence (no mask), so two tokens instead of 77 both shortens the sequence
      (326 instead of 1104 at native) and avoids inventing 75 padding tokens. The diffusion timestep enters as
      UniDiffuser's own image-timestep token, untouched.
    * CLIP-image slot. A learned 512-dim token, zero-initialised, through the pretrained `clip_img_in` (at step 0
      the slot carries `clip_img_in.bias` plus its positional row). The pipeline co-denoises this slot with
      the VAE latent, so no convention leaves the latent at t while marginalising the slot; a fixed learned
      vector is the deterministic choice (fresh noise per call would break eta-0 DDIM reproducibility) and the
      network can learn to ignore or use it. Its output is discarded.
    * Output. The image-token outputs through `vae_img_out`, unpatchified to (B, 4, 32, 40): the pretrained
      epsilon head retrained as velocity, as in the other rows. `clip_img_out`, `text_out`, and the inner
      transformer's own unused `PatchEmbed` (bypassed by `hidden_states_is_embedding=True` in the original
      forward) are replaced by `nn.Identity()` so no parameter sits outside the gradient path (0.9M dropped).
    * Forward. Reimplemented on the pretrained submodules instead of calling `UniDiffuserModel.forward`, which
      unpatchifies with `height = width = int(N ** 0.5)` and cannot express a 16x20 grid; the block loop is
      reimplemented as well so gradient checkpointing can wrap each block (the class does not set
      `_supports_gradient_checkpointing`). `forward_uvit` takes explicit text and CLIP inputs so the parity
      gate can drive the adapted stack with the original model's own conditioning.
    """

    def __init__(self, num_actions=29, context_frames=32, noise_buckets=10, unidiffuser_path=UNIDIFFUSER_DEFAULT,
                 action_dropout=0.1, grad_ckpt=True, cache_dir=None, latent_hw=LATENT_HW, cond_tokens=2,
                 latent_channels=LATENT_CHANNELS, phase_buckets=0, action_history=0, control_bits=0):
        super().__init__()
        from diffusers import UniDiffuserModel
        self.context_frames = context_frames
        self.num_actions = num_actions
        self.action_dropout = action_dropout
        self.grad_ckpt = grad_ckpt
        self.latent_channels = latent_channels
        in_channels = stacked_in_channels(latent_channels, context_frames)
        # the repo ships only diffusion_pytorch_model.bin; saying so skips the safetensors probe, which offline mode would turn into a failed lookup
        m = UniDiffuserModel.from_pretrained(unidiffuser_path, subfolder="unet", cache_dir=cache_dir, low_cpu_mem_usage=False, use_safetensors=False)
        cfg = m.config
        assert cfg.use_data_type_embedding and not cfg.use_timestep_embedding and cfg.cross_attention_dim is None, \
            "UniDiffuserWorldModel was written against the unidiffuser-v1 config"
        self.uvit = m

        m.vae_img_in.proj = inflate_input_conv(m.vae_img_in.proj, in_channels, latent_channels)
        m.register_to_config(in_channels=in_channels)

        p = cfg.patch_size
        native = cfg.sample_size // p
        self.grid = (latent_hw[0] // p, latent_hw[1] // p)
        # the phase is a third conditioning token in the joint sequence, so it takes one more row of
        # the pretrained text positional table (77 are available) than the action/bucket pair does
        cond_tokens = int(cond_tokens) + (1 if phase_buckets else 0) + max(0, int(action_history) - 1)
        if cond_tokens > cfg.num_text_tokens:
            raise ValueError(f"{cond_tokens} conditioning tokens exceed the pretrained text slots "
                             f"({cfg.num_text_tokens}); shorten --action-history")
        self.cond_tokens = cond_tokens
        n_text = cfg.num_text_tokens
        with torch.no_grad():
            pe = m.pos_embed.data
            head, text, clip, img = pe[:, :2], pe[:, 2:2 + cond_tokens], pe[:, 2 + n_text:3 + n_text], pe[:, 3 + n_text:]
            img = img.reshape(1, native, native, -1).permute(0, 3, 1, 2)
            img = torch.nn.functional.interpolate(img, size=self.grid, mode="bilinear", align_corners=False)
            img = img.permute(0, 2, 3, 1).reshape(1, self.grid[0] * self.grid[1], -1)
            m.pos_embed = nn.Parameter(torch.cat([head, text, clip, img], dim=1).contiguous())
        m.num_text_tokens = cond_tokens
        m.num_tokens = m.pos_embed.shape[1]
        m.register_to_config(num_text_tokens=cond_tokens)
        m.clip_img_out = nn.Identity(); m.text_out = nn.Identity(); m.transformer.pos_embed = nn.Identity()

        if not action_history:      # the control tokens replace it; see UNetWorldModel
            self.action_embedder = nn.Embedding(num_actions + 1, cfg.text_dim)   # last id = null
            nn.init.normal_(self.action_embedder.weight, std=0.02)
        self.bucket_embedder = nn.Embedding(noise_buckets, cfg.text_dim)
        nn.init.normal_(self.bucket_embedder.weight, std=0.02)
        self.clip_token = nn.Parameter(torch.zeros(1, 1, cfg.clip_img_dim))
        add_phase_embedder(self, phase_buckets, cfg.text_dim)
        add_control_history(self, action_history, control_bits, cfg.text_dim)

    def forward(self, x, t, action, context, noise_bucket, phase=None):
        action = drop_actions(self, action)
        text = token_conditioning(self, action, noise_bucket, phase)     # (B, L+1 or 2 or 3, 64)
        clip = self.clip_token.expand(x.shape[0], -1, -1)
        return self.forward_uvit(torch.cat([context, x], dim=1), t, text, clip)

    def forward_uvit(self, latent, t, text_embeds, clip_embeds, timestep_text=0, data_type=1):
        """The adapted U-ViT with explicit text (B, n, 64) and CLIP (B, 1, 512) inputs; returns the 4-channel image output."""
        m = self.uvit
        b = latent.shape[0]
        h_img = m.vae_img_in(latent)
        h_clip = m.clip_img_in(clip_embeds)
        h_text = m.text_in(text_embeds)
        t_img = t * torch.ones(b, dtype=t.dtype, device=t.device) if torch.is_tensor(t) else torch.full((b,), t, dtype=torch.long, device=latent.device)
        t_txt = torch.full((b,), timestep_text, dtype=torch.long, device=latent.device)
        tok_t_img = m.timestep_img_embed(m.timestep_img_proj(t_img).to(m.dtype)).unsqueeze(1)
        tok_t_txt = m.timestep_text_embed(m.timestep_text_proj(t_txt).to(m.dtype)).unsqueeze(1)
        tok_dtype = m.data_type_token_embedding(torch.full((b,), data_type, dtype=torch.long, device=latent.device)).unsqueeze(1)
        h = torch.cat([tok_t_img, tok_t_txt, tok_dtype, h_text, h_clip, h_img], dim=1)
        pos = torch.cat([m.pos_embed[:, :2], m.data_type_pos_embed_token, m.pos_embed[:, 2:]], dim=1)
        h = m.pos_embed_drop(h + pos)
        h = self._blocks(h)
        n_img = h_img.shape[1]
        out = m.vae_img_out(h[:, -n_img:])                       # (B, N, p*p*4)
        gh, gw = self.grid if n_img == self.grid[0] * self.grid[1] else (int(n_img ** 0.5), int(n_img ** 0.5))
        p, c = m.patch_size, m.out_channels
        out = out.reshape(b, gh, gw, p, p, c)
        out = torch.einsum("nhwpqc->nchpwq", out)
        return out.reshape(b, c, gh * p, gw * p)

    def _blocks(self, h):
        tr = self.uvit.transformer
        ckpt = self.grad_ckpt and self.training and torch.is_grad_enabled()

        def run(block, x):
            return torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False) if ckpt else block(x)

        skips = []
        for blk in tr.transformer_in_blocks:
            h = run(blk, h); skips.append(h)
        h = run(tr.transformer_mid_block, h)
        for ob in tr.transformer_out_blocks:
            h = ob["skip"](h, skips.pop())
            h = run(ob["block"], h)
        return tr.norm_out(h)


class SD35WorldModel(nn.Module):
    """Stable Diffusion 3.5 Medium's MMDiT-X (2B, patch 2, 16-channel latents) as the world model.

    Source facts, read Sep 17 2026 from `stabilityai/stable-diffusion-3.5-medium` and diffusers 0.40
    (`models/transformers/transformer_sd3.py`, `models/embeddings.py`):

    * Transformer: subfolder `transformer` (`SD3Transformer2DModel`), `in_channels` 16, `out_channels` 16,
      `patch_size` 2, `pos_embed_max_size` 384, `sample_size` 128, `joint_attention_dim` 4096,
      `pooled_projection_dim` 2048, `caption_projection_dim` = the inner width, dual-attention blocks in
      the first 12 layers (MMDiT-X); 2.246B parameters once the patch projection is inflated. Flow
      matching, no learned variance, so the head is 16 channels flat.
    * Autoencoder: 16 channels, f8, `scaling_factor` 1.5305 and `shift_factor` 0.0609, applied as
      `(z - shift) * scale` by `StableDiffusion3Pipeline`. Our 320x256 padded frame gives 16x32x40, so
      context 32 x 16 = 512 channels plus the noisy target 16 = 528 input channels.
    * Conditioning: `encoder_hidden_states` is the T5/CLIP token sequence at width `joint_attention_dim`
      through `context_embedder`; `pooled_projections` is the CLIP pooled vector at width
      `pooled_projection_dim` through `time_text_embed.text_embedder`, summed with the timestep embedding.
      No text encoder is instantiated here: only the two projections inside the transformer are used.

    Adaptation, in the order the tensors are touched:

    * Patch projection. `pos_embed.proj` (Conv2d 16 -> inner, kernel 2, stride 2) is inflated to 528 input
      channels by `inflate_input_conv`, so step 0 equals the pretrained model applied to the noisy target.
      `register_to_config(in_channels=528)` keeps the config a checkpoint carries in step with the weights.
    * Positional embedding. Nothing to do, and this is the one backbone where that is true: SD3's
      `PatchEmbed` stores a 384x384 sin-cos table (`pos_embed_max_size`, read from the checkpoint at
      init rather than assumed here) and `cropped_pos_embed` center-crops it to the incoming token grid
      at every forward, which is how the pipeline itself serves non-square resolutions. Our 16x20 grid
      takes rows 184..199 and columns 182..201 of that table. So the rows we use are pretrained
      rows at their pretrained spacing (unlike UniDiffuser, whose learned table has to be interpolated),
      and the parity gate is exact at our own grid rather than only at the native one. The buffer is
      persistent, so it rides in the state dict and the shape never changes.
    * Conditioning. Action (`num_actions + 1` rows, the last being the null row for dropout, same
      `action_dropout` semantics as the other backbones) and noise bucket become two learned tokens of
      width `joint_attention_dim` through the pretrained `context_embedder`, the PixArt construction. The
      pooled slot, which has no text-free neutral value, is a learned per-action vector added to a learned
      base vector of width `pooled_projection_dim`; the base starts at zeros, so at step 0 the pooled
      contribution is `text_embedder(0)` for every sample. The diffusion timestep keeps the model's own
      `time_text_embed` path.
    * Output. The 16-channel head, flat (no learn-sigma chunk to take), retrained from flow-matching
      velocity to our v-prediction under `VDiffusion`'s linear 1e-4..0.02 DDPM schedule. That schedule
      mismatch against SD 3.5's rectified-flow training is the same mismatch the U-Net, PixArt and
      UniDiffuser rows carry.
    * Forward. The pretrained `forward` already reads height and width off the input and unpatchifies with
      both, so a 16x20 grid needs no reimplementation, unlike UniDiffuser's square-only unpatchify.

    `transformer=` takes an already-constructed `SD3Transformer2DModel` instead of downloading one, which
    is how `verify_sd35.py` runs the CPU gates on a small randomly initialised config of the same class.
    """

    def __init__(self, num_actions=29, context_frames=32, noise_buckets=10, sd35_path=SD35_DEFAULT,
                 action_dropout=0.1, grad_ckpt=True, cache_dir=None, latent_channels=16, transformer=None,
                 phase_buckets=0, action_history=0, control_bits=0):
        super().__init__()
        self.context_frames = context_frames
        self.num_actions = num_actions
        self.action_dropout = action_dropout
        self.latent_channels = latent_channels
        in_channels = stacked_in_channels(latent_channels, context_frames)
        if transformer is None:
            try:
                from diffusers import SD3Transformer2DModel
            except ImportError as e:   # SD3 landed in diffusers 0.29; Spiderman has 0.40, Superman 0.37
                raise ImportError("SD35WorldModel needs a diffusers with SD3Transformer2DModel (>= 0.29)") from e
            transformer = SD3Transformer2DModel.from_pretrained(sd35_path, subfolder="transformer",
                                                                cache_dir=cache_dir, low_cpu_mem_usage=False)
        self.transformer = transformer
        cfg = self.transformer.config
        assert cfg.patch_size == 2 and cfg.pos_embed_max_size, \
            "SD35WorldModel was written against the SD 3.5 config (patch 2, cropped positional table)"

        self.transformer.pos_embed.proj = inflate_input_conv(self.transformer.pos_embed.proj, in_channels, latent_channels)
        self.transformer.register_to_config(in_channels=in_channels)

        if not action_history:      # both action tables are replaced by the control tokens
            self.action_embedder = nn.Embedding(num_actions + 1, cfg.joint_attention_dim)   # last id = null
            self.pooled_action = nn.Embedding(num_actions + 1, cfg.pooled_projection_dim)
            nn.init.normal_(self.action_embedder.weight, std=0.02)
            nn.init.normal_(self.pooled_action.weight, std=0.02)
        self.bucket_embedder = nn.Embedding(noise_buckets, cfg.joint_attention_dim)
        nn.init.normal_(self.bucket_embedder.weight, std=0.02)
        self.pooled_base = nn.Parameter(torch.zeros(cfg.pooled_projection_dim))
        add_phase_embedder(self, phase_buckets, cfg.joint_attention_dim)
        add_control_history(self, action_history, control_bits, cfg.joint_attention_dim)
        if action_history:
            # the pooled slot has no text-free neutral value, so under control-history conditioning it
            # carries the NEWEST control's embedding; zero-init keeps step 0 at text_embedder(0), the
            # same property the single-action path gets from a zeroed pooled_base
            self.pooled_control = nn.Linear(cfg.joint_attention_dim, cfg.pooled_projection_dim)
            nn.init.zeros_(self.pooled_control.weight); nn.init.zeros_(self.pooled_control.bias)
        if grad_ckpt:
            self.transformer.enable_gradient_checkpointing()

    def conditioning(self, action, noise_bucket, phase=None):
        """The joint-attention context tokens and the pooled projection for one batch.

        Two tokens by default (action, bucket), L+1 under `--action-history` (one executed control
        per context tic, then the bucket), plus a phase token when the model carries one. The pooled
        slot carries the action id's own vector, or the newest control's embedding when conditioning
        on controls; it never carries the bucket or the phase.
        """
        tokens = token_conditioning(self, action, noise_bucket, phase)
        if getattr(self, "control_history", None) is not None:
            n = self.control_history.length
            return tokens, self.pooled_control(tokens[:, n - 1]) + self.pooled_base
        return tokens, self.pooled_action(action) + self.pooled_base

    def forward(self, x, t, action, context, noise_bucket, phase=None):
        action = drop_actions(self, action)
        tokens, pooled = self.conditioning(action, noise_bucket, phase)
        return self.transformer(torch.cat([context, x], dim=1), encoder_hidden_states=tokens,
                                pooled_projections=pooled, timestep=t).sample


def build_model(backbone, num_actions, context_frames, noise_buckets=10, grad_ckpt=True, warm_start=None,
                cache_dir=None, action_dropout=0.1, latent_channels=None, action_inject="token",
                phase_buckets=0, action_history=0, control_bits=0):
    latent_channels = resolve_latent_channels(backbone, latent_channels)
    warm_start, scratch = resolve_warm_start(backbone, warm_start)
    if action_inject != "token" and backbone != "pixart":
        raise NotImplementedError(f"--action-inject {action_inject} is a PixArt knob, not a {backbone} one")
    if backbone == "dit":
        m = DiTWorldModel(num_actions, context_frames, noise_buckets, action_dropout=action_dropout, grad_ckpt=grad_ckpt,
                          latent_channels=latent_channels, phase_buckets=phase_buckets,
                          action_history=action_history, control_bits=control_bits)
        if warm_start:
            n, skipped = m.load_imagenet_warm_start(warm_start)
            print(f"DiT warm start: {n} tensors loaded, skipped {skipped}")
    elif backbone == "unet":
        m = UNetWorldModel(num_actions, context_frames, noise_buckets, action_dropout=action_dropout, grad_ckpt=grad_ckpt,
                           sd_path=warm_start or "CompVis/stable-diffusion-v1-4", cache_dir=cache_dir,
                           latent_channels=latent_channels, phase_buckets=phase_buckets,
                           action_history=action_history, control_bits=control_bits)
    elif backbone == "pixart":
        m = PixArtWorldModel(num_actions, context_frames, noise_buckets, action_dropout=action_dropout,
                             grad_ckpt=grad_ckpt, pixart_path=warm_start or PIXART_DEFAULT, cache_dir=cache_dir,
                             latent_channels=latent_channels, scratch=scratch, action_inject=action_inject,
                             phase_buckets=phase_buckets, action_history=action_history,
                             control_bits=control_bits)
    elif backbone == "unidiffuser":
        m = UniDiffuserWorldModel(num_actions, context_frames, noise_buckets, action_dropout=action_dropout, grad_ckpt=grad_ckpt,
                                  unidiffuser_path=warm_start or UNIDIFFUSER_DEFAULT, cache_dir=cache_dir,
                                  latent_channels=latent_channels, phase_buckets=phase_buckets,
                                  action_history=action_history, control_bits=control_bits)
    elif backbone == "sd35":
        m = SD35WorldModel(num_actions, context_frames, noise_buckets, action_dropout=action_dropout, grad_ckpt=grad_ckpt,
                           sd35_path=warm_start or SD35_DEFAULT, cache_dir=cache_dir, latent_channels=latent_channels,
                           phase_buckets=phase_buckets, action_history=action_history, control_bits=control_bits)
    else:
        raise ValueError(backbone)
    return m
