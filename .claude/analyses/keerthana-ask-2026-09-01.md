# Message to Keerthana (draft, Sep 1, 2026)

Context for the ask: the Doom directory on Superman was deleted some time after
May 2 (probably in a disk cleanup; the drive is at 98% and our home is 94 GB of
LEGO and perseve work). Nothing of the April pipeline survives there except the
conda env and the training logs. The DiT weights are safe on the GitHub release.
The two Google Drive links from the April tmux history (`full_latents.zip`,
5.43 GB, file id 1kWHfGNB7LqTD9AlaB5tRrqRbCoNb9529, and a second file
1ajh_Nl9wE8Ae-u6A67n5hLOartE6-7fi) now redirect to a Google sign-in.

---

Hi Keerthana,

I'm turning the DoomDiT project into a 4-page workshop paper (CoRL 2026 PhysWM
workshop, due Sep 30) and I'm rebuilding the evaluation so the U-Net vs DiT
comparison holds up. The Doom directory on Superman got wiped, so I'm missing
three things that only you have. Could you send whichever of these you still
have, even if messy?

1. **The frame-to-latent encoding script** (the one that wrote
   `ep_XXXX_latents.npy` / `ep_XXXX_actions.npy`). I have rewritten it, but I
   need to match your choices exactly so the released checkpoint evaluates on
   the same latents: resize filter for 320x240 -> 160x120, whether you used
   `latent_dist.mean` or `.sample()`, and whether the stored action at row t is
   the action taken after frame t or the one that produced it.
2. **The U-Net baseline**: model code, training config, and the checkpoint
   that gave 24.60 dB / 0.198 LPIPS. Also: was it warm-started from SD 1.4 or
   trained from scratch, and how many steps / what batch size?
3. **The PSNR/LPIPS script** behind 26.04 / 0.153 and 24.60 / 0.198: which
   frames were scored (the 10x8 training segments?), LPIPS backbone (alex or
   vgg), PSNR against the raw frame or the VAE-decoded latent, and which sampler
   and step count.

If you still have `full_latents.zip` on Drive, re-sharing it as
"anyone with the link" would save me a re-encode of the 48 GB source.

No need to clean anything up, a zip of the folder is perfect. Happy to put you
as co-author on the workshop paper, same as the class project.

Thanks!
Rohan
