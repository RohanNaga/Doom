"""One closed-loop rollout as an animated GIF: ground truth | EMA | live, every second tic of the 256, half size."""
import sys, numpy as np, torch
from PIL import Image
from diffusers import AutoencoderKL
D = "/sata2/data/rnagabhi/doom"
step, r, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
dev = "cuda:3"
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="vae",
                                    cache_dir=f"{D}/hf/hub", torch_dtype=torch.float16).to(dev).eval()
scale, shift = 1.5305, 0.0609
def decode(lat):
    out = []
    for i in range(0, len(lat), 16):
        z = torch.from_numpy(np.asarray(lat[i:i+16], dtype=np.float32)).to(dev).half() / scale + shift
        with torch.no_grad():
            img = vae.decode(z).sample.float().clamp(-1, 1)
        out.append(((img + 1) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[:, :240])
    return np.concatenate(out)
base = f"{D}/results_spiderman/042-sd35-nexttic/steward_{step}"
live = np.load(f"{base}/rollout_live.npz"); ema = np.load(f"{base}/rollout_ema.npz")
idx = list(range(0, 256, 2))
gt, e, l = decode(live["gt"][r, idx]), decode(ema["pred"][r, idx]), decode(live["pred"][r, idx])
frames = []
for k in range(len(idx)):
    row = np.concatenate([gt[k], e[k], l[k]], axis=1)
    im = Image.fromarray(row).resize((row.shape[1] // 2, row.shape[0] // 2), Image.BILINEAR)
    frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
frames[0].save(out, save_all=True, append_images=frames[1:], duration=57, loop=0, optimize=True)
print("wrote", out, "rollout", r, "map", int(live["map"][r]), "episode", int(live["episode"][r]), "frames", len(frames))
