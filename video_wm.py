"""
SkyReels-V2 Diffusion Forcing 1.3B as an action-conditioned Doom world model.

This is the video-pretrained row. Unlike the DiT / U-Net / PixArt rows, which stack L context
latents on the CHANNEL axis of a single image, this backbone keeps them as what they are: the
leading frames of one latent video. A window is

    latents             (B, 16, L+1, 32, 40)   L context frames, then the noisy target
    per_frame_timesteps (B, L+1)               one diffusion time per frame -- the whole point
    action              (B,)                   the action of the transition into the target

and `forward` returns the prediction for the target frame only, (B, 16, 32, 40).

Diffusion forcing IS the per-frame timestep: the context frames sit at a low (or zero) time
while the target sits at the sampled one, so the model reads near-clean history and denoises one
future frame. `SkyReelsV2Transformer3DModel.forward` takes `timestep` of shape (B, F), one entry
per LATENT FRAME, together with `enable_diffusion_forcing=True`; see `SKYREELS_NOTES` for the
quoted line numbers.

What this wrapper adds to the published checkpoint, and why:

* Action. One zero-initialised `Embedding(num_actions + 1, inner_dim)` (1536 wide; the last row
  is the null id for action dropout), added to the TARGET frame's tokens only, right after the
  Conv3d patch embedding. Latest action only, matching the other rows ("Design A"), so the action
  information is identical across the comparison. Zero init means step 0 is exactly the
  pretrained model.
* Text. The checkpoint is a text-to-video model; its only non-timestep conditioning is UMT5
  cross-attention. We precompute ONE null-prompt embedding with `make_null_prompt.py` and load it
  as a non-persistent buffer, so neither training nor evaluation ever instantiates the 22.7 GB
  UMT5-XXL encoder. Non-persistent keeps it out of the state dict, so checkpoints stay strictly
  key-matched to the pretrained transformer plus the action table.
* fps. `inject_sample_info=true` in this config, so the transformer wants an `fps` id into an
  `Embedding(2, inner)` -- id 0 for 16 fps, id 1 otherwise (pipeline line 787). The native branch
  that folds it into the per-frame timestep projection only broadcasts correctly at batch size 1
  (see `SKYREELS_NOTES`), so we turn `inject_sample_info` off in the config and add the identical
  bias ourselves through a hook on `condition_embedder`. At B=1 this is bit-for-bit the native
  computation; at B>1 it is what the native code was trying to do.

Both changes are forward hooks on named submodules, not a copy of the diffusers forward: the
decorated `forward` (`@apply_lora_scale`) and the entire DF timestep path stay untouched.

Two objectives are selectable, because which one a video-pretrained checkpoint prefers is an
empirical question, not a design one:

* `flow` (default) -- the checkpoint's native rectified flow. x_t = (1 - sigma) x0 + sigma eps,
  target eps - x0, timestep 1000 * sigma, optional schedule shift.
* `vp-v` -- the shared velocity target of the other three rows, on `diffusion_v.VDiffusion`'s
  linear-beta VP schedule, so the loss number is comparable with them.

Loss is on the target frame only under either objective.
"""
import os

import torch
import torch.nn as nn

SKYREELS_DEFAULT = "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers"
NULL_PROMPT_DEFAULT = "weights/skyreels_null_prompt.pt"
LATENT_CHANNELS = 16
LATENT_HW = (32, 40)
NULL_PROMPT_TOKENS = 512        # what SkyReelsV2DiffusionForcingPipeline.__call__ defaults to

# Facts read out of diffusers 0.40.0 (the SkyReelsV2 classes first ship in 0.35.0; they do NOT
# exist in the 0.31 that the U-Net and PixArt rows run under). Paths are under site-packages.
SKYREELS_NOTES = """
models/transformers/transformer_skyreels_v2.py
  599  self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
  681-682  patch embed then .flatten(2).transpose(1, 2): tokens run (frame, height, width) row-major,
           so the target frame owns the LAST post_patch_height * post_patch_width tokens.
  685  causal_mask is built only when num_frame_per_block > 1; this checkpoint ships 1, so attention
       over the window is full, not causal. _set_ar_attention(k) is how the pipeline turns it on.
  721-729  enable_diffusion_forcing branch: `b, f = timestep.shape` -- per-frame timesteps are
           (B, F), one per LATENT FRAME, not per token. temb becomes (b, f*pph*ppw, inner) and
           timestep_proj (b, 6, f*pph*ppw, inner) by repeating each frame's time over its tokens.
  710-719  inject_sample_info: fps -> Embedding(2, inner) -> FeedForward -> added to timestep_proj.
           The DF branch does .repeat(timestep.shape[1], 1, 1) on a (B, 6, inner) tensor, giving
           (B*F, 6, inner), which only broadcasts against (B, F, 6, inner) when B == 1.
  347  text: PixArtAlphaTextProjection(text_dim=4096 -> inner) inside the condition embedder;
  505  the projected sequence is the key/value of attn2, i.e. plain cross-attention.
  563  _supports_gradient_checkpointing = True;  732  the blocks are wrapped when it is enabled.
pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing.py
  787  fps_embeds = [0 if i == 16 else 1 for i in fps_embeds]   # the fps id is 0 or 1
  875-885  context ("prefix") frames are held at a small fixed time: the latent is blended with
           noise_factor = 0.001 * addnoise_condition and its timestep set to addnoise_condition.
           That is sigma = t / 1000 under the flow forward process -- our --ctx-stabilize.
  180/618  the null prompt embedding is (1, max_sequence_length, 4096); the DF pipeline calls
           encode_prompt with max_sequence_length=512.
schedulers/scheduling_unipc_multistep.py
  428-441  use_flow_sigmas: sigmas = linspace(1, 1/1000, n+1)[:-1], shifted by
           flow_shift * s / (1 + (flow_shift - 1) * s), and timesteps = 1000 * sigma.
  805-807  prediction_type "flow_prediction": x0 = x_t - sigma * model_output, and
           _sigma_to_alpha_sigma_t gives alpha_t = 1 - sigma, sigma_t = sigma, i.e.
           x_t = (1 - sigma) x0 + sigma eps with the model predicting eps - x0.
"""


class FlowMatching:
    """The checkpoint's native rectified-flow objective.

    x_t = (1 - sigma) x0 + sigma eps, target eps - x0, timestep = num_steps * sigma. `shift` is
    the scheduler's flow shift; the stored scheduler config has 1.0 and the DF pipeline docstring
    recommends 8.0 for T2V at inference. Applied to a uniform draw it biases sigma toward 1
    exactly as `set_timesteps` biases the inference grid.
    """

    def __init__(self, shift=1.0, num_steps=1000):
        self.shift = float(shift)
        self.num_steps = int(num_steps)

    def shift_sigma(self, s):
        return self.shift * s / (1.0 + (self.shift - 1.0) * s)

    def sample_sigma(self, shape, device, generator=None, max_sigma=1.0, u=None):
        if u is None:
            u = torch.rand(shape, device=device, generator=generator)
        return self.shift_sigma(u.to(device) * max_sigma)

    def q_sample(self, x0, sigma, noise):
        s = sigma.view(-1, *([1] * (x0.ndim - 1)))
        return (1.0 - s) * x0 + s * noise

    def target(self, x0, noise):
        return noise - x0

    def timestep(self, sigma):
        return sigma * self.num_steps


def tiny_config(num_layers=2, num_attention_heads=2, attention_head_dim=32, ffn_dim=128):
    """A tiny randomly-initialised transformer with the real config's structure.

    Same class, same patch size, same in/out channels, same `inject_sample_info` and the same DF
    timestep path, so a gate run against it exercises the code path the 1.3B checkpoint uses.
    """
    return dict(patch_size=(1, 2, 2), num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim, in_channels=LATENT_CHANNELS,
                out_channels=LATENT_CHANNELS, text_dim=4096, freq_dim=256, ffn_dim=ffn_dim,
                num_layers=num_layers, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
                eps=1e-6, rope_max_seq_len=1024, inject_sample_info=True, num_frame_per_block=1)


class SkyReelsWorldModel(nn.Module):
    """SkyReels-V2 DF 1.3B with a zero-initialised action embedding on the target frame."""

    def __init__(self, num_actions=29, skyreels_path=SKYREELS_DEFAULT, null_prompt=NULL_PROMPT_DEFAULT,
                 action_dropout=0.0, grad_ckpt=True, cache_dir=None, fps_id=1, tiny=None,
                 torch_dtype=None):
        super().__init__()
        from diffusers import SkyReelsV2Transformer3DModel
        self.num_actions = num_actions
        self.action_dropout = action_dropout
        self.fps_id = int(fps_id)
        if tiny is not None:
            self.transformer = SkyReelsV2Transformer3DModel(**tiny)
        else:
            self.transformer = SkyReelsV2Transformer3DModel.from_pretrained(
                skyreels_path, subfolder="transformer", cache_dir=cache_dir, torch_dtype=torch_dtype)
            # note: low_cpu_mem_usage must stay at its default True -- this class declares
            # _keep_in_fp32_modules, and diffusers refuses the False path when it is set
            # (models/modeling_utils.py:1203).
        cfg = self.transformer.config
        self.inner_dim = cfg.num_attention_heads * cfg.attention_head_dim

        self.action_embedder = nn.Embedding(num_actions + 1, self.inner_dim)   # last id = null
        nn.init.zeros_(self.action_embedder.weight)   # step 0 is exactly the pretrained model

        self.register_buffer("null_prompt", _load_null_prompt(null_prompt, cfg.text_dim), persistent=False)

        # Take the fps injection away from the native branch (correct only at B=1) and do it here.
        self._inject_fps = bool(cfg.inject_sample_info)
        if self._inject_fps:
            self.transformer.register_to_config(inject_sample_info=False)
        self._action_emb = None
        self.transformer.patch_embedding.register_forward_hook(self._hook_action)
        self.transformer.condition_embedder.register_forward_hook(self._hook_fps)
        if grad_ckpt:
            self.transformer.enable_gradient_checkpointing()

    # ---- the two interception points -------------------------------------------------------

    def _hook_action(self, module, args, output):
        """Add the action embedding to the target frame's tokens, after patch embedding.

        `output` is (B, inner, F, H/2, W/2) and the target is the last frame. Out-of-place, so
        autograd never sees a mutated convolution output.
        """
        emb = self._action_emb
        if emb is None:
            return output
        past, tgt = output[:, :, :-1], output[:, :, -1:]
        return torch.cat([past, tgt + emb.to(output.dtype)[:, :, None, None, None]], dim=2)

    def _hook_fps(self, module, args, output):
        """Add the fps bias that the native `inject_sample_info` branch would have added.

        `condition_embedder` returns (temb, timestep_proj, encoder_hidden_states, image_states)
        and `timestep_proj` is (B, F, 6 * inner) here, before the caller's unflatten, so a bias
        of shape (B, 1, 6 * inner) reproduces the native per-frame-constant addition exactly.
        """
        if not self._inject_fps:
            return output
        temb, timestep_proj, ehs, ehs_img = output
        t = self.transformer
        fps = torch.full((timestep_proj.shape[0],), self.fps_id, dtype=torch.long, device=timestep_proj.device)
        bias = t.fps_projection(t.fps_embedding(fps))
        if timestep_proj.dim() == 3:          # (B, F, 6*inner): diffusion forcing
            bias = bias.unsqueeze(1)
        return temb, timestep_proj + bias, ehs, ehs_img

    # ---- forward ---------------------------------------------------------------------------

    def forward(self, latents, per_frame_timesteps, action, return_all=False):
        """latents (B, 16, L+1, 32, 40), per_frame_timesteps (B, L+1), action (B,).

        Returns the target frame's prediction (B, 16, 32, 40), or every frame when `return_all`.
        """
        b = latents.shape[0]
        if self.training and self.action_dropout > 0:
            drop = torch.rand(b, device=action.device) < self.action_dropout
            action = torch.where(drop, torch.full_like(action, self.num_actions), action)
        self._action_emb = self.action_embedder(action)
        try:
            out = self.transformer(hidden_states=latents, timestep=per_frame_timesteps,
                                   encoder_hidden_states=self.null_prompt.expand(b, -1, -1).to(latents.dtype),
                                   enable_diffusion_forcing=True, return_dict=False)[0]
        finally:
            self._action_emb = None
        return out if return_all else out[:, :, -1]

    # ---- sizing ----------------------------------------------------------------------------

    def tokens_per_window(self, context_frames, latent_hw=LATENT_HW):
        p_t, p_h, p_w = self.transformer.config.patch_size
        return ((context_frames + 1) // p_t) * (latent_hw[0] // p_h) * (latent_hw[1] // p_w)


def _load_null_prompt(path, text_dim, max_sequence_length=NULL_PROMPT_TOKENS):
    """The precomputed UMT5 null-prompt embedding, (1, S, text_dim).

    `path` None or "none" gives zeros, which is legitimate only for shape and gradient gates: the
    text projection has biases, so zeros are NOT the same conditioning as the real null prompt.
    """
    if path in (None, "", "none"):
        print("WARNING: no --null-prompt; cross-attention conditioned on ZEROS, not the real null prompt")
        return torch.zeros(1, max_sequence_length, text_dim)
    if not os.path.exists(path):
        raise FileNotFoundError(f"null prompt embedding {path} not found; run make_null_prompt.py "
                                f"(or pass --null-prompt none for a shape-only gate)")
    obj = torch.load(path, map_location="cpu", weights_only=True)   # tensors and primitives only
    emb = obj["embed"] if isinstance(obj, dict) else obj
    emb = emb.float()
    if emb.ndim == 2:
        emb = emb.unsqueeze(0)
    if emb.shape[0] != 1 or emb.shape[-1] != text_dim:
        raise ValueError(f"null prompt {path} has shape {tuple(emb.shape)}, expected (1, S, {text_dim})")
    return emb


def build_video_model(num_actions=29, skyreels_path=SKYREELS_DEFAULT, null_prompt=NULL_PROMPT_DEFAULT,
                      action_dropout=0.0, grad_ckpt=True, cache_dir=None, fps_id=1, tiny=None):
    return SkyReelsWorldModel(num_actions=num_actions, skyreels_path=skyreels_path,
                              null_prompt=null_prompt, action_dropout=action_dropout,
                              grad_ckpt=grad_ckpt, cache_dir=cache_dir, fps_id=fps_id, tiny=tiny)


def full_finetune_bytes(n_params):
    """Parameter, gradient and Adam-state bytes for a full fine-tune of `n_params` parameters.

    bf16 autocast keeps the weights themselves in fp32 (the cast is per-op and transient), so
    autograd produces fp32 gradients and AdamW holds two fp32 moments.
    """
    return {"params": n_params * 4, "grads": n_params * 4, "adam": n_params * 8,
            "total": n_params * 16}


def window_timesteps(target_t, ctx_t, context_frames):
    """(B, L+1) per-frame timesteps from a target time (B,) and a shared context time (B,)."""
    ctx = ctx_t.view(-1, 1).expand(-1, context_frames)
    return torch.cat([ctx, target_t.view(-1, 1)], dim=1)


if __name__ == "__main__":
    print(SKYREELS_NOTES)
