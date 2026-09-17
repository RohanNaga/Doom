"""
The four world-model backbones behind one interface.

    model(x_noisy, t, action, context, noise_bucket) -> v prediction, (B, 4, 32, 40)

All four take the same 4L+4-channel input (L context latents channel-stacked, then the noisy
target), the same discrete action, the same context-noise bucket, and predict the same
4-channel velocity. Where the conditioning enters is each backbone's native pathway: adaLN for
the DiT, cross-attention (action) plus class embedding (noise bucket) for the U-Net, GameNGen's
construction, cross-attention on two learned caption tokens for PixArt, and two learned text tokens in
UniDiffuser's joint token sequence. All four warm-start
from published weights with the input projection inflated: pretrained weights on the 4 target
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
  caption sequence. We feed two learned tokens through the pretrained caption projection: an
  action embedding with a trailing null row for CFG dropout (same `action_dropout` semantics as
  the U-Net) and a context-noise-bucket embedding, both of width `caption_channels` (4096). The
  timestep keeps PixArt's own adaLN-single path untouched, and `use_additional_conditions` is
  forced False so no resolution or aspect-ratio embeddings are required.
* Output head. PixArt is a learn-sigma model: 8 output channels, epsilon first and the learned
  variance second (`PixArtAlphaPipeline` takes `chunk(2, dim=1)[0]`). We return the first 4 after
  unpatchify. Those are the pretrained epsilon channels, retrained here toward velocity, exactly
  as we treat DiT-XL/2's head.

What UniDiffuser v1 needed is documented on `UniDiffuserWorldModel` itself: it is the one backbone whose
positional table is learned (sliced and interpolated rather than regenerated), whose conditioning is a token
sequence rather than adaLN or cross-attention, and whose original forward cannot unpatchify a rectangular grid.
"""
import torch
import torch.nn as nn

from models import DiT_models, get_2d_sincos_pos_embed

LATENT_HW = (32, 40)
PIXART_DEFAULT = "PixArt-alpha/PixArt-XL-2-512x512"
UNIDIFFUSER_DEFAULT = "thu-ml/unidiffuser-v1"


class DiTWorldModel(nn.Module):
    def __init__(self, num_actions=29, context_frames=32, noise_buckets=10, model_name="DiT-XL/2",
                 action_dropout=0.1, grad_ckpt=True):
        super().__init__()
        self.context_frames = context_frames
        self.dit = DiT_models[model_name](input_size=LATENT_HW, in_channels=4 * context_frames + 4,
                                          pred_channels=4, num_classes=num_actions,
                                          class_dropout_prob=action_dropout, learn_sigma=False)
        self.dit.use_grad_ckpt = grad_ckpt
        hidden = self.dit.t_embedder.mlp[-1].out_features
        self.noise_embedder = nn.Embedding(noise_buckets, hidden)
        nn.init.normal_(self.noise_embedder.weight, std=0.02)

    def forward(self, x, t, action, context, noise_bucket):
        d = self.dit
        h = d.x_embedder(torch.cat([context, x], dim=1)) + d.pos_embed
        c = d.t_embedder(t) + d.y_embedder(action, self.training) + self.noise_embedder(noise_bucket)
        for block in d.blocks:
            if d.use_grad_ckpt and self.training:
                h = torch.utils.checkpoint.checkpoint(d.ckpt_wrapper(block), h, c, use_reentrant=False)
            else:
                h = block(h, c)
        return d.unpatchify(d.final_layer(h, c))

    def load_imagenet_warm_start(self, path):
        """ImageNet DiT-XL/2-256 checkpoint: inflate the patch projection, rebuild pos_embed for the
        16x20 grid, keep the eps half of the 8-channel output head, skip the class table."""
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
                 action_dropout=0.1, grad_ckpt=True, cache_dir=None):
        super().__init__()
        from diffusers import UNet2DConditionModel
        self.context_frames = context_frames
        self.num_actions = num_actions
        self.action_dropout = action_dropout
        self.unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet", cache_dir=cache_dir,
                                                         num_class_embeds=noise_buckets, low_cpu_mem_usage=False)
        old = self.unet.conv_in
        new = nn.Conv2d(4 * context_frames + 4, old.out_channels, old.kernel_size, old.stride, old.padding)
        with torch.no_grad():
            new.weight.zero_(); new.weight[:, -4:] = old.weight; new.bias.copy_(old.bias)
        self.unet.conv_in = new
        self.unet.register_to_config(in_channels=4 * context_frames + 4)
        self.action_embedder = nn.Embedding(num_actions + 1, self.unet.config.cross_attention_dim)  # last id = null
        nn.init.normal_(self.action_embedder.weight, std=0.02)
        if grad_ckpt:
            self.unet.enable_gradient_checkpointing()

    def forward(self, x, t, action, context, noise_bucket):
        if self.training and self.action_dropout > 0:
            drop = torch.rand(action.shape[0], device=action.device) < self.action_dropout
            action = torch.where(drop, torch.full_like(action, self.num_actions), action)
        tokens = self.action_embedder(action).unsqueeze(1)          # (B, 1, 768)
        return self.unet(torch.cat([context, x], dim=1), t, encoder_hidden_states=tokens,
                         class_labels=noise_bucket).sample


class PixArtWorldModel(nn.Module):
    """PixArt-alpha DiT (611M, patch 2, adaLN-single) adapted to the action-conditioned world model.

    See the module docstring for what was changed relative to the published checkpoint.
    """

    def __init__(self, num_actions=29, context_frames=32, noise_buckets=10, pixart_path=PIXART_DEFAULT,
                 action_dropout=0.1, grad_ckpt=True, cache_dir=None):
        super().__init__()
        from diffusers import PixArtTransformer2DModel
        from diffusers.models.embeddings import get_2d_sincos_pos_embed as diffusers_pos_embed
        self.context_frames = context_frames
        self.num_actions = num_actions
        self.action_dropout = action_dropout
        in_channels = 4 * context_frames + 4
        self.transformer = PixArtTransformer2DModel.from_pretrained(
            pixart_path, subfolder="transformer", cache_dir=cache_dir, low_cpu_mem_usage=False,
            use_additional_conditions=False)

        patch = self.transformer.pos_embed
        old = patch.proj
        new = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding,
                        bias=old.bias is not None)
        with torch.no_grad():
            new.weight.zero_(); new.weight[:, -4:] = old.weight
            if old.bias is not None:
                new.bias.copy_(old.bias)
        patch.proj = new
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
        self.action_embedder = nn.Embedding(num_actions + 1, caption_channels)   # last id = null
        self.bucket_embedder = nn.Embedding(noise_buckets, caption_channels)
        nn.init.normal_(self.action_embedder.weight, std=0.02)
        nn.init.normal_(self.bucket_embedder.weight, std=0.02)
        if grad_ckpt:
            self.transformer.enable_gradient_checkpointing()

    def forward(self, x, t, action, context, noise_bucket):
        if self.training and self.action_dropout > 0:
            drop = torch.rand(action.shape[0], device=action.device) < self.action_dropout
            action = torch.where(drop, torch.full_like(action, self.num_actions), action)
        tokens = torch.stack([self.action_embedder(action), self.bucket_embedder(noise_bucket)], dim=1)
        # older diffusers unpack added_cond_kwargs unconditionally; both keys are ignored with use_additional_conditions=False
        out = self.transformer(torch.cat([context, x], dim=1), encoder_hidden_states=tokens, timestep=t,
                               encoder_attention_mask=None,
                               added_cond_kwargs={"resolution": None, "aspect_ratio": None}).sample
        return out[:, :4]   # epsilon half of the learn-sigma head, retrained as velocity


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
                 action_dropout=0.1, grad_ckpt=True, cache_dir=None, latent_hw=LATENT_HW, cond_tokens=2):
        super().__init__()
        from diffusers import UniDiffuserModel
        self.context_frames = context_frames
        self.num_actions = num_actions
        self.action_dropout = action_dropout
        self.grad_ckpt = grad_ckpt
        in_channels = 4 * context_frames + 4
        # the repo ships only diffusion_pytorch_model.bin; saying so skips the safetensors probe, which offline mode would turn into a failed lookup
        m = UniDiffuserModel.from_pretrained(unidiffuser_path, subfolder="unet", cache_dir=cache_dir, low_cpu_mem_usage=False, use_safetensors=False)
        cfg = m.config
        assert cfg.use_data_type_embedding and not cfg.use_timestep_embedding and cfg.cross_attention_dim is None, \
            "UniDiffuserWorldModel was written against the unidiffuser-v1 config"
        self.uvit = m

        old = m.vae_img_in.proj
        new = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
        with torch.no_grad():
            new.weight.zero_(); new.weight[:, -4:] = old.weight
            if old.bias is not None:
                new.bias.copy_(old.bias)
        m.vae_img_in.proj = new
        m.register_to_config(in_channels=in_channels)

        p = cfg.patch_size
        native = cfg.sample_size // p
        self.grid = (latent_hw[0] // p, latent_hw[1] // p)
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

        self.action_embedder = nn.Embedding(num_actions + 1, cfg.text_dim)   # last id = null
        self.bucket_embedder = nn.Embedding(noise_buckets, cfg.text_dim)
        nn.init.normal_(self.action_embedder.weight, std=0.02)
        nn.init.normal_(self.bucket_embedder.weight, std=0.02)
        self.clip_token = nn.Parameter(torch.zeros(1, 1, cfg.clip_img_dim))

    def forward(self, x, t, action, context, noise_bucket):
        if self.training and self.action_dropout > 0:
            drop = torch.rand(action.shape[0], device=action.device) < self.action_dropout
            action = torch.where(drop, torch.full_like(action, self.num_actions), action)
        text = torch.stack([self.action_embedder(action), self.bucket_embedder(noise_bucket)], dim=1)   # (B, 2, 64)
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


def build_model(backbone, num_actions, context_frames, noise_buckets=10, grad_ckpt=True, warm_start=None, cache_dir=None, action_dropout=0.1):
    if backbone == "dit":
        m = DiTWorldModel(num_actions, context_frames, noise_buckets, action_dropout=action_dropout, grad_ckpt=grad_ckpt)
        if warm_start:
            n, skipped = m.load_imagenet_warm_start(warm_start)
            print(f"DiT warm start: {n} tensors loaded, skipped {skipped}")
    elif backbone == "unet":
        m = UNetWorldModel(num_actions, context_frames, noise_buckets, action_dropout=action_dropout, grad_ckpt=grad_ckpt,
                           sd_path=warm_start or "CompVis/stable-diffusion-v1-4", cache_dir=cache_dir)
    elif backbone == "pixart":
        m = PixArtWorldModel(num_actions, context_frames, noise_buckets, action_dropout=action_dropout,
                             grad_ckpt=grad_ckpt, pixart_path=warm_start or PIXART_DEFAULT, cache_dir=cache_dir)
    elif backbone == "unidiffuser":
        m = UniDiffuserWorldModel(num_actions, context_frames, noise_buckets, action_dropout=action_dropout, grad_ckpt=grad_ckpt,
                                  unidiffuser_path=warm_start or UNIDIFFUSER_DEFAULT, cache_dir=cache_dir)
    else:
        raise ValueError(backbone)
    return m
