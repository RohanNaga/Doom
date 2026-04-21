"""
Pre-flight sanity check for DoomDiT.

Builds the model with DOOM dimensions, optionally loads pretrained warm-start,
runs a single forward and a single training_losses call on dummy data, and
asserts shapes + finite loss. Runs in a few seconds on a single GPU.

Usage:
    python sanity_check.py                       # with warm-start (default)
    python sanity_check.py --ckpt ""             # random init only
    python sanity_check.py --model DiT-XL/2 --num-classes 7
"""
import argparse

import torch

from models import DiT_models
from diffusion import create_diffusion
from download import find_model


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DiT_models[args.model](
        input_size=(16, 20),
        in_channels=20,
        pred_channels=4,
        num_classes=args.num_classes,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}   params: {n_params:,}")
    print(f"pos_embed shape: {tuple(model.pos_embed.shape)} (expect (1, 80, 1152))")

    if args.ckpt:
        state_dict = find_model(args.ckpt)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        pretrained_proj = state_dict.pop("x_embedder.proj.weight")
        pretrained_bias = state_dict.pop("x_embedder.proj.bias", None)
        with torch.no_grad():
            model.x_embedder.proj.weight[:, :4].copy_(pretrained_proj.to(device))
            if pretrained_bias is not None:
                model.x_embedder.proj.bias.copy_(pretrained_bias.to(device))
        for k in list(state_dict.keys()):
            if k.startswith("pos_embed") or k.startswith("y_embedder"):
                state_dict.pop(k)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Warm-start: {len(missing)} missing, {len(unexpected)} unexpected keys")
        if unexpected:
            print(f"  unexpected (first 5): {unexpected[:5]}")

    model.eval()

    # Dummy batch matching CustomDataset output.
    B = 2
    context = torch.randn(B, 16, 16, 20, device=device)
    target = torch.randn(B, 4, 16, 20, device=device)
    action = torch.randint(0, args.num_classes, (B,), device=device)
    t = torch.randint(0, 1000, (B,), device=device)

    with torch.no_grad():
        out = model(target, t, action, context=context)
    expected_shape = (B, 8, 16, 20)  # 8 = pred_channels * 2 (learn_sigma)
    assert tuple(out.shape) == expected_shape, f"bad shape: {tuple(out.shape)} vs {expected_shape}"
    print(f"forward OK: output shape {tuple(out.shape)}")

    diffusion = create_diffusion(timestep_respacing="")
    model.train()
    loss_dict = diffusion.training_losses(model, target, t, dict(context=context, action=action))
    loss = loss_dict["loss"].mean()
    assert torch.isfinite(loss), f"non-finite loss: {loss.item()}"
    print(f"training_losses OK: loss={loss.item():.4f}")

    print("PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--ckpt", type=str, default="DiT-XL-2-256x256.pt",
                        help="Pretrained checkpoint; pass empty string to skip warm-start")
    args = parser.parse_args()
    main(args)
