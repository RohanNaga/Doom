"""
FVD between generated and real clips (I3D features, the StyleGAN-V / common-metrics recipe).

Reads the `clips_u8.npz` written by `rollout_eval.py --score` (pred and gt, uint8, (N, H, 240, 320, 3)
after transpose), takes the first `--frames` predicted frames of each rollout, resizes to 224x224,
runs the TorchScript I3D (downloaded once to --i3d), and reports FVD with N clips. GameNGen reports
FVD at 16 and 32 frames; FVD is biased at small N, so report N alongside.

    python fvd.py --clips results/010-dit-l32/rollout_metrics/clips_u8.npz --frames 16 --i3d weights/i3d_torchscript.pt
"""
import argparse
import json
import os
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

I3D_URL = "https://github.com/JunyaoHu/common_metrics_on_video_quality/raw/main/fvd/styleganv/i3d_torchscript.pt"


def get_i3d(path, device):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(I3D_URL, path)
    return torch.jit.load(path).eval().to(device)


@torch.no_grad()
def features(i3d, clips_u8, device, bs=8):
    """clips_u8: (N, T, H, W, 3) uint8 -> (N, 400) I3D logits-space features."""
    out = []
    for i in range(0, len(clips_u8), bs):
        x = torch.from_numpy(clips_u8[i:i + bs]).to(device).float() / 127.5 - 1.0     # (b, T, H, W, 3)
        x = x.permute(0, 4, 1, 2, 3)                                                    # (b, 3, T, H, W)
        b, c, t, h, w = x.shape
        x = F.interpolate(x.reshape(b, c * t, h, w), size=(224, 224), mode="bilinear", align_corners=False).reshape(b, c, t, 224, 224)
        out.append(i3d(x, rescale=False, resize=False, return_features=True).cpu().double())
    return torch.cat(out).numpy()


def fvd_from_features(a, b):
    from scipy import linalg
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(ca @ cb)          # scipy >= 1.18 returns the array only
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(ca + cb - 2 * covmean))


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(args.clips)
    pred, gt = d["pred"], d["gt"]                       # (N, H, 3, 240, 320) uint8, one clip per rollout
    assert pred.shape[1] >= args.frames, f"rollouts have {pred.shape[1]} frames, need {args.frames}"
    pred = pred[:, :args.frames].transpose(0, 1, 3, 4, 2); gt = gt[:, :args.frames].transpose(0, 1, 3, 4, 2)
    i3d = get_i3d(args.i3d, device)
    fa, fb = features(i3d, pred, device), features(i3d, gt, device)
    val = fvd_from_features(fa, fb)
    res = {"fvd": val, "frames": args.frames, "num_clips": int(len(pred))}
    print(json.dumps(res))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clips", required=True); p.add_argument("--frames", type=int, default=16)
    p.add_argument("--i3d", default="weights/i3d_torchscript.pt"); p.add_argument("--out", default="")
    main(p.parse_args())
