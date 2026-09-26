"""Closed-loop rollouts of one read as GIFs: ground truth | prediction (EMA), every second tic, half size.
usage: rollout_gif2.py <space sd1|sd35> <npz> <out_prefix> <rollout indices comma-separated>"""
import sys, numpy as np, torch
from PIL import Image
from diffusers import AutoencoderKL
D = "/sata2/data/rnagabhi/doom"
space, npz_path, prefix, rolls = sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(",")]
dev = "cuda:3"
if space == "sd35":
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="vae", cache_dir=f"{D}/hf/hub", torch_dtype=torch.float16).to(dev).eval()
    scale, shift = 1.5305, 0.0609
else:
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", cache_dir=f"{D}/hf/hub", torch_dtype=torch.float16).to(dev).eval()
    scale, shift = 0.18215, 0.0
def decode(lat):
    out = []
    for i in range(0, len(lat), 16):
        z = torch.from_numpy(np.asarray(lat[i:i+16], dtype=np.float32)).to(dev).half() / scale + shift
        with torch.no_grad():
            img = vae.decode(z).sample.float().clamp(-1, 1)
        out.append(((img + 1) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[:, :240])
    return np.concatenate(out)
z = np.load(npz_path)
idx = list(range(0, z["pred"].shape[1], 2))
for r in rolls:
    gt, pr = decode(z["gt"][r, idx]), decode(z["pred"][r, idx])
    frames = []
    for k in range(len(idx)):
        row = np.concatenate([gt[k], pr[k]], axis=1)
        im = Image.fromarray(row).resize((row.shape[1] // 2, row.shape[0] // 2), Image.BILINEAR)
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
    out = f"{prefix}_r{r:02d}_map{int(z['map'][r])}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=57, loop=0, optimize=True)
    print("wrote", out, "episode", int(z["episode"][r]), flush=True)
