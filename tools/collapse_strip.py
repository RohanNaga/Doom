"""Decode a few tics of a collapsed live rollout, the EMA rollout of the same window, and the ground truth
(SD 3.5 16-channel latents from a steward rollout npz) into one PNG strip. CPU/GPU light: one VAE decode."""
import sys, numpy as np, torch
from PIL import Image
from diffusers import AutoencoderKL
D = "/sata2/data/rnagabhi/doom"
step, rolls, out = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
TICS = [1, 4, 8, 16, 32, 64, 96, 128, 192, 256]
dev = "cuda:3" if torch.cuda.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="vae",
                                    cache_dir=f"{D}/hf/hub", torch_dtype=torch.float16).to(dev).eval()
scale, shift = 1.5305, 0.0609
def decode(lat):  # lat: (T,16,32,40) stored as (z - shift) * scale
    z = torch.from_numpy(np.asarray(lat, dtype=np.float32)).to(dev).half() / scale + shift
    with torch.no_grad():
        img = vae.decode(z).sample.float().clamp(-1, 1)
    return ((img + 1) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
live = np.load(f"{D}/results_spiderman/042-sd35-nexttic/steward_{step}/rollout_live.npz")
ema = np.load(f"{D}/results_spiderman/042-sd35-nexttic/steward_{step}/rollout_ema.npz")
rows = []
for r in rolls:
    idx = [t - 1 for t in TICS]
    for name, z in (("gt", live["gt"][r, idx]), ("live", live["pred"][r, idx]), ("ema", ema["pred"][r, idx])):
        frames = decode(z)
        rows.append(np.concatenate(list(frames), axis=1))
        print(f"r{r:02d} map {int(live['map'][r])} {name}: ch13 mean per tic", [round(float(z[i, 13].mean()), 2) for i in range(len(TICS))], flush=True)
strip = np.concatenate(rows, axis=0)
Image.fromarray(strip).save(out)
print("wrote", out, strip.shape, "tics", TICS, "rows per rollout: gt, live, ema")
