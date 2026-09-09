"""
The two world-model backbones behind one interface.

    model(x_noisy, t, action, context, noise_bucket) -> v prediction, (B, 4, 32, 40)

Both take the same 4L+4-channel input (L context latents channel-stacked, then the noisy
target), the same discrete action, the same context-noise bucket, and predict the same
4-channel velocity. Where the conditioning enters is each backbone's native pathway:
adaLN for the DiT, cross-attention (action) plus class embedding (noise bucket) for the U-Net,
GameNGen's construction. Both warm-start from published weights with the input projection
inflated: pretrained weights on the 4 target channels, zeros on the context channels, so
step 0 equals the pretrained model applied to the noisy target.
"""
import torch
import torch.nn as nn

from models import DiT_models, get_2d_sincos_pos_embed

LATENT_HW = (32, 40)


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


def build_model(backbone, num_actions, context_frames, noise_buckets=10, grad_ckpt=True, warm_start=None, cache_dir=None):
    if backbone == "dit":
        m = DiTWorldModel(num_actions, context_frames, noise_buckets, grad_ckpt=grad_ckpt)
        if warm_start:
            n, skipped = m.load_imagenet_warm_start(warm_start)
            print(f"DiT warm start: {n} tensors loaded, skipped {skipped}")
    elif backbone == "unet":
        m = UNetWorldModel(num_actions, context_frames, noise_buckets, grad_ckpt=grad_ckpt,
                           sd_path=warm_start or "CompVis/stable-diffusion-v1-4", cache_dir=cache_dir)
    else:
        raise ValueError(backbone)
    return m
