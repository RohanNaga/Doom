# Teacher-forced context noise: clean against near-clean

**Some level above 0.0 exceeds its own standard error somewhere, so this is on the record:**

- PixArt-alpha 512 (033) / seen / PSNR vs raw: level 0.07 beats 0.0 by 0.10 (1.73 SE)
- PixArt-alpha 512 (033) / seen / PSNR vs decoded: level 0.07 beats 0.0 by 0.12 (1.70 SE)

**And the same levels are significantly worse elsewhere (above 2 SE):**

- PixArt-alpha 512 (033) / unseen / LPIPS vs raw: level 0.035 is 0.0058 WORSE than 0.0 (2.95 SE)
- PixArt-alpha 512 (033) / unseen / LPIPS vs raw: level 0.07 is 0.0092 WORSE than 0.0 (4.77 SE)
- PixArt-alpha 512 (033) / unseen / LPIPS vs decoded: level 0.035 is 0.0054 WORSE than 0.0 (2.69 SE)
- PixArt-alpha 512 (033) / unseen / LPIPS vs decoded: level 0.07 is 0.0091 WORSE than 0.0 (4.57 SE)
- SD 1.4 U-Net (031) / unseen / LPIPS vs raw: level 0.07 is 0.0063 WORSE than 0.0 (3.33 SE)
- SD 1.4 U-Net (031) / unseen / LPIPS vs decoded: level 0.07 is 0.0066 WORSE than 0.0 (3.37 SE)

**Reading:** nothing here changes a headline number. The gains are under 2 SE and sit on one corpus; the losses are up to 4.8 SE and sit on the held-out corpora, which is where the paper's transfer claim lives. Feeding exactly clean context with bucket 0 is the right default.

Bucket 0 is trained on levels uniform in [0, 0.07), mean 0.035, so 0.0 sits at the edge of its range rather than its centre; that is what these runs test. Note 0.07 itself maps to bucket 1 in training and at inference alike (verified identical at eight levels), so that row is bucket 1's bottom edge, not bucket 0's top.

### PixArt-alpha 512 (033) — seen (512 windows, seed 0)

Stored 2048-window row numbers for orientation: PSNR 21.35, LPIPS 0.272 (different window count, so not a paired comparison).

Persistence floor on these windows 19.14 dB; autoencoder ceiling 28.56 dB.

| metric | level | mean | paired delta vs 0.0 | SE | t |
|---|---|---|---|---|---|
| PSNR vs raw | 0 (reference) | 21.06 | -- | -- | -- |
| PSNR vs raw | 0.035 | 21.12 | +0.06 | 0.06 | +0.95 |
| PSNR vs raw | 0.07 | 21.16 | +0.10 | 0.06 | +1.73 |
| LPIPS vs raw | 0 (reference) | 0.2801 | -- | -- | -- |
| LPIPS vs raw | 0.035 | 0.2780 | -0.0021 | 0.0024 | -0.88 |
| LPIPS vs raw | 0.07 | 0.2790 | -0.0011 | 0.0024 | -0.46 |
| PSNR vs decoded | 0 (reference) | 21.95 | -- | -- | -- |
| PSNR vs decoded | 0.035 | 22.02 | +0.06 | 0.07 | +0.91 |
| PSNR vs decoded | 0.07 | 22.07 | +0.12 | 0.07 | +1.70 |
| LPIPS vs decoded | 0 (reference) | 0.2673 | -- | -- | -- |
| LPIPS vs decoded | 0.035 | 0.2651 | -0.0022 | 0.0024 | -0.90 |
| LPIPS vs decoded | 0.07 | 0.2660 | -0.0013 | 0.0024 | -0.54 |

### PixArt-alpha 512 (033) — unseen (512 windows, seed 0)

Stored 2048-window row numbers for orientation: PSNR 19.39, LPIPS 0.434 (different window count, so not a paired comparison).

Persistence floor on these windows 18.54 dB; autoencoder ceiling 26.43 dB.

| metric | level | mean | paired delta vs 0.0 | SE | t |
|---|---|---|---|---|---|
| PSNR vs raw | 0 (reference) | 19.42 | -- | -- | -- |
| PSNR vs raw | 0.035 | 19.41 | -0.01 | 0.05 | -0.27 |
| PSNR vs raw | 0.07 | 19.46 | +0.04 | 0.05 | +0.79 |
| LPIPS vs raw | 0 (reference) | 0.4305 | -- | -- | -- |
| LPIPS vs raw | 0.035 | 0.4362 | +0.0058 | 0.0020 | +2.95 |
| LPIPS vs raw | 0.07 | 0.4397 | +0.0092 | 0.0019 | +4.77 |
| PSNR vs decoded | 0 (reference) | 20.32 | -- | -- | -- |
| PSNR vs decoded | 0.035 | 20.31 | -0.02 | 0.06 | -0.28 |
| PSNR vs decoded | 0.07 | 20.37 | +0.05 | 0.06 | +0.82 |
| LPIPS vs decoded | 0 (reference) | 0.4155 | -- | -- | -- |
| LPIPS vs decoded | 0.035 | 0.4209 | +0.0054 | 0.0020 | +2.69 |
| LPIPS vs decoded | 0.07 | 0.4246 | +0.0091 | 0.0020 | +4.57 |

### SD 1.4 U-Net (031) — seen (512 windows, seed 0)

Stored 2048-window row numbers for orientation: PSNR 21.36, LPIPS 0.270 (different window count, so not a paired comparison).

Persistence floor on these windows 19.14 dB; autoencoder ceiling 28.56 dB.

| metric | level | mean | paired delta vs 0.0 | SE | t |
|---|---|---|---|---|---|
| PSNR vs raw | 0 (reference) | 21.12 | -- | -- | -- |
| PSNR vs raw | 0.035 | 21.10 | -0.02 | 0.05 | -0.44 |
| PSNR vs raw | 0.07 | 21.12 | -0.00 | 0.05 | -0.01 |
| LPIPS vs raw | 0 (reference) | 0.2776 | -- | -- | -- |
| LPIPS vs raw | 0.035 | 0.2782 | +0.0005 | 0.0021 | +0.26 |
| LPIPS vs raw | 0.07 | 0.2793 | +0.0017 | 0.0021 | +0.82 |
| PSNR vs decoded | 0 (reference) | 22.02 | -- | -- | -- |
| PSNR vs decoded | 0.035 | 21.99 | -0.03 | 0.06 | -0.51 |
| PSNR vs decoded | 0.07 | 22.02 | -0.01 | 0.06 | -0.09 |
| LPIPS vs decoded | 0 (reference) | 0.2647 | -- | -- | -- |
| LPIPS vs decoded | 0.035 | 0.2654 | +0.0007 | 0.0021 | +0.33 |
| LPIPS vs decoded | 0.07 | 0.2664 | +0.0018 | 0.0021 | +0.83 |

### SD 1.4 U-Net (031) — unseen (512 windows, seed 0)

Stored 2048-window row numbers for orientation: PSNR 19.14, LPIPS 0.446 (different window count, so not a paired comparison).

Persistence floor on these windows 18.54 dB; autoencoder ceiling 26.43 dB.

| metric | level | mean | paired delta vs 0.0 | SE | t |
|---|---|---|---|---|---|
| PSNR vs raw | 0 (reference) | 19.18 | -- | -- | -- |
| PSNR vs raw | 0.035 | 19.16 | -0.02 | 0.05 | -0.35 |
| PSNR vs raw | 0.07 | 19.21 | +0.03 | 0.05 | +0.67 |
| LPIPS vs raw | 0 (reference) | 0.4449 | -- | -- | -- |
| LPIPS vs raw | 0.035 | 0.4480 | +0.0031 | 0.0019 | +1.65 |
| LPIPS vs raw | 0.07 | 0.4511 | +0.0063 | 0.0019 | +3.33 |
| PSNR vs decoded | 0 (reference) | 20.02 | -- | -- | -- |
| PSNR vs decoded | 0.035 | 20.00 | -0.02 | 0.06 | -0.30 |
| PSNR vs decoded | 0.07 | 20.07 | +0.05 | 0.06 | +0.75 |
| LPIPS vs decoded | 0 (reference) | 0.4309 | -- | -- | -- |
| LPIPS vs decoded | 0.035 | 0.4342 | +0.0033 | 0.0019 | +1.72 |
| LPIPS vs decoded | 0.07 | 0.4374 | +0.0066 | 0.0019 | +3.37 |

