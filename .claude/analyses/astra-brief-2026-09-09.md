# Brief for Astra: review the DoomDiT training and evaluation stack before numbers freeze

## The problem and why it matters
We are writing a 4-page workshop paper (CoRL 2026 PhysWM, due Sep 30) whose single claim is a controlled comparison: a DiT-XL/2 versus the Stable Diffusion 1.4 U-Net as the backbone of a GameNGen-style Doom world model, with data, latents, context, conditioning, augmentation, objective, optimizer, and step budget held identical. Both runs are training now on 16 GB A4000s (DiT at step 25k of 90k, U-Net at 7k). If the shared code has a bug that affects one backbone more than the other, or a metric that mis-scores, the comparison is worthless and there is no time to rerun. The work is already built; this is a review, so everything is handed over.

## Constraints
- Runs cannot be restarted: DiT finishes about Sep 10, U-Net about Sep 13. A bug in the training path can only be documented or worked around at evaluation time unless it is fatal.
- Evaluation code can still change freely (nothing is frozen until Sep 26).
- No GPU here; the code paths run on CPU with tiny synthetic tensors (see "how to check").

## Artifacts (absolute paths, main checkout)
- Context: /Users/rohan/Documents/Github/Doom/RESEARCH_CONTEXT.md (sections 0 and 7), /Users/rohan/Documents/Github/Doom/TECHNICAL_PLAN.md (every design decision), /Users/rohan/Documents/Github/Doom/REQUIREMENTS.md (sections 7b, 9, 10).
- Shared diffusion: /Users/rohan/Documents/Github/Doom/diffusion_v.py (velocity DDPM, DDIM, context noise augmentation).
- Backbones: /Users/rohan/Documents/Github/Doom/backbones.py (DiT wrapper with warm-start inflation; U-Net wrapper on diffusers UNet2DConditionModel), /Users/rohan/Documents/Github/Doom/models.py (DiT from fast-DiT, modified).
- Trainer: /Users/rohan/Documents/Github/Doom/train_wm.py. Data: /Users/rohan/Documents/Github/Doom/doom_data.py (LatentWindowDataset, make_split_by_map), /Users/rohan/Documents/Github/Doom/encode_parquet.py.
- Evaluation: /Users/rohan/Documents/Github/Doom/eval_tf.py (teacher-forced), /Users/rohan/Documents/Github/Doom/rollout_eval.py (rollouts, drift, IDM scoring), /Users/rohan/Documents/Github/Doom/train_idm.py, /Users/rohan/Documents/Github/Doom/fvd.py, /Users/rohan/Documents/Github/Doom/finetune_decoder.py.
- Recording (for the row semantics): /Users/rohan/Documents/Github/Doom/record_arnold.py.
- Early numbers to sanity-check against: RESEARCH_CONTEXT.md Sep 9 entries (DiT at 25k: held-out v-loss 0.2299, PSNR 20.4 vs decoded target, 18.7 vs raw frame; copy-last 19.7 / 18.9; U-Net v-loss 0.2445 at 5k vs DiT 0.2552 at 5k).

## Assumptions that could be false (check these first)
1. diffusion_v.py: v = sqrt(abar)*eps - sqrt(1-abar)*x0, x0 = sqrt(abar)*xt - sqrt(1-abar)*v, eps = sqrt(1-abar)*xt + sqrt(abar)*v; the DDIM step uses these with eta=0 correctly, including the last step returning x0.
2. noise_augment: level in [0, 0.7] as a variance fraction, a = sqrt(1-level), s = sqrt(level), bucket = floor(level/max*buckets) clamped; bucket 0 is both "clean" and "small noise" (is that a problem?). At inference we pass bucket 0 and clean context.
3. backbones.DiTWorldModel: the ImageNet DiT checkpoint's final_layer.linear rows are ordered (p, p, c) with c=8 and we keep the eps half (ci < 4) for a 4-channel velocity head; x_embedder.proj inflated with pretrained weights on the LAST 4 input channels (the noisy target) since the input is cat([context, x]). pos_embed rebuilt for 16x20.
4. UNetWorldModel: num_class_embeds=10 makes diffusers create a class embedding added to the time embedding; action goes in as a single cross-attention token; action dropout is done manually (DiT does it inside LabelEmbedder when training). Are the two dropout rates and null-token semantics equivalent?
5. train_wm.py: gradient accumulation with accelerate (`Accelerator(gradient_accumulation_steps=1)` but manual accumulation in the loop) is correct under DDP for the DiT (4 procs, accum 1) and single-process for the U-Net (accum 8); the loss is divided by accum; clipping happens once per optimizer step. The LR scheduler steps per optimizer step. EMA (fp32 on CPU every 8 steps with decay^8) is fine. Checkpoint "best.pt" chosen by held-out v-loss computed with a fixed generator (same noise/timesteps per evaluation?). Is the val loss comparable across backbones and steps?
6. LatentWindowDataset: window i uses latents [i, i+L) as context and i+L as target; the action is act[i+L-1], which record_arnold.py defines as the action applied from that frame to the next. Under stride 4 encoding (encode_parquet keeps tic % 4 == 0) the "next frame" is 4 tics later and the action recorded at tic t is held for 4 tics, so act at the last context frame is the action applied during the transition. Confirm with record_arnold.py's loop (frame recorded before make_action).
7. eval_tf.py: the raw frame for window (episode, start) is at tic (start + L) * stride; the copy-last baseline decodes ctx[:, -4:]; PSNR uses [0,1] images; LPIPS inputs scaled to [-1,1]; HUD crop is the bottom 32 rows of the 240-row frame after cropping the 16-row pad.
8. rollout_eval.py: context shifting `torch.cat([ctx[:, 4:], x], dim=1)` keeps temporal order; actions[:, h] indexes the action at the last context frame for step h; IDM pair (seed[-1], pred[0]) uses actions[0].
9. train_idm.py: PairDataset pairs latent s with s+1 and label act[s] (the action applied from s to s+1). Consistent with 6.

## What was already tried
- Fit checks and a 60-step real-data run, a sanity checkpoint through eval_tf, rollout_eval, train_idm, fvd: everything runs and numbers move in the expected direction. That validates plumbing, not correctness of the math or the symmetry between backbones.

## How to check things yourself
- CPU only: `python -c` with tiny tensors against diffusion_v.VDiffusion (e.g. verify q_sample, v_target, x0_from_v, eps_from_v round-trip; verify DDIM with eta=0 reproduces the closed form on a linear toy model).
- backbones: constructing the DiT needs timm; the U-Net needs diffusers and the SD 1.4 config (network). Reading the code is enough for the inflation logic; you may verify the final_layer row selection against models.DiT.unpatchify by constructing a small DiT (DiT-S/2) with in_channels=4*2+4 and checking that keeping rows ci<4 of a synthetic 8-channel head reproduces channels 0..3 of unpatchify.
- Data: no latents locally; reason from doom_data.py and record_arnold.py.

## The ask
1. List concrete bugs (file:line, what is wrong, what it does to the comparison, the fix), ranked by whether they invalidate the DiT-vs-U-Net comparison, then whether they mis-score a metric.
2. List asymmetries between the two backbones that we have not stated (we know: adaLN vs cross-attention conditioning, parameter count, learning rate, single-GPU accumulation for the U-Net).
3. Say what you would change in the evaluation before numbers freeze, with the cost of each change.
4. If you think something in the training path is wrong in a way that matters, say whether it is fatal (rerun needed) or reportable (state it in the paper).
Show the checks you ran.
