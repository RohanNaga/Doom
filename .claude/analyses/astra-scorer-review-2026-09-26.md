# Astra review of the scorer extensions (Sep 26 2026)

- Reviewer: Astra (gpt-6-astra, reasoning high), sandbox workspace-write, no network.
- Thread: `01a0e028-a42f-7660-bb77-9488870eef81`, 2 turns.
- Scope: commit 9614e27 (merged at 5b8301e), `eval_tf.py` and `paper/fixtures/test_eval_tf_columns.py`, against `.claude/analyses/evaluation-cohesion-decision-2026-09-26.md` item 3 and the frozen scores in `results/distance_study/scores/040-unet-nexttic_snap_0200000_ema_ddim10/`.

## Verdict

**NO-GO** for the Sunday rescore as merged. The three headline differences (scene_psnr_dec minus scene_copy_psnr_dec, scene_lpips_raw minus scene_persist_lpips_raw, scene_vae_psnr minus scene_psnr_raw, and their tuned-decoder versions) are computed correctly. Three gaps block the run; all three fixes leave frozen values unchanged.

## Findings (ranked)

1. **P1: fresh manifests can fail at h4.** `adapt_split.py:162` draws windows with `window_starts(..., 1)`, so starts are valid for one tic only; `eval_tf.py:450` resolves them against the requested horizon and refuses any that are not windows of the h4 dataset. Reproduction: episode 2 with 128 rows and 32 context frames has valid starts 0..95 at h1 and 0..92 at h4; a 32-window draw picked 93, 94, 95 and the h4 run aborts. Spec item 3 requires four-tic rows on the same 32-per-episode set. Fix: draw with horizon 4 (lines 162, 333 `window_horizon`), raise instead of warning at line 328 when an episode has fewer than 32, use one manifest for h1 and h4; leave `legacy_held_out_windows()` alone.
2. **P2: the frozen column order is not preserved.** `eval_tf.py:72` `LEGACY_COLUMNS` places `copy_latent_mse, latent_mse_ratio` after `latent_mse`, but the frozen metrics.json order is `psnr_dec, lpips_dec, copy_psnr_dec, latent_mse, hud_psnr_dec, ...`, so in per_window.csv `hud_psnr_dec` and everything after shifts two columns (index 10 to 12). The fixture `paper/fixtures/test_eval_tf_columns.py:49` pins `60d99c4` as the frozen producer, but the frozen `provenance.txt` files name `6a33311` (h1) and `ca03bad` (h4), both ancestors of `60d99c4`, which is the commit that added `copy_latent_mse`. The test therefore checks the wrong header. Fix: move the two latent columns to the end of `LEGACY_COLUMNS`, repin the fixture to `6a33311`. Readers by name are unaffected; positional readers of per_window.csv are.
3. **P2: decoded copy-last LPIPS against raw truth is missing.** `eval_tf.py:149` calls `pair_metrics(last, raw, lp, "copy_psnr_raw")` with no LPIPS key, so `copy_lpips_raw`, `scene_copy_lpips_raw` and their `_tuned` and `_nodup` variants never exist. Spec item 3 asks for LPIPS of model, copy-last and reconstruction under each decoder. Fix: add `"copy_lpips_raw"` as the fifth argument; update the expected set at `test_eval_tf_columns.py:223`. Not needed for the three named headline differences.
4. **Note: cost estimate.** Two decoders cost 6 decodes per batch (3 per decoder) and 18 LPIPS calls per window (8 per decoder plus 2 persistence), not about 14; with fix 3 it becomes 22. This is a wrong spec estimate, not a bug. GPU memory is batch-bounded; saved latents accumulate on CPU. Wall time and peak memory were not measured.
5. **Note: duplicate flags.** `eval_tf.py:542` derives `dup_raw` from `persist_mse_raw == 0` on the uint8/255 fp32 tensors, not from the uint8 bytes directly. Division by 255 is injective on uint8, so equality is preserved exactly; a direct byte compare would match the spec's wording. Every `_nodup` column, tuned ones included, excludes the same windows.

Passed checks (code reading plus CPU fixtures with an LPIPS stand-in): scene crop is rows 0..207 of the 240-row frame after padding is stripped; crop and full-frame LPIPS use the same [-1, 1] mapping; `copy_lpips_dec` uses the same images as `copy_psnr_dec`; the second decoder decodes the same predicted, true and copy latents with the same scale and shift and no resampling; decoder columns are suffixed consistently while persistence stays unsuffixed; `--save-latents` stores the scored prediction (the fourth tic at h4) in fp16 with window identities.

Not checked: a full replay of frozen numbers with production weights and data, real LPIPS numerics, GPU peak memory and wall time.

## Supervisor verification

- Tests: `/opt/miniconda3/envs/PERSEVE/bin/python -m pytest paper/fixtures/test_eval_tf_columns.py paper/fixtures/test_adapt_eval.py paper/fixtures/test_eval_tf_sunday_review.py -q` gives `3 failed, 26 passed`; the requested two files alone pass (22), and the three failures are Astra's reproductions of findings 1 to 3.
- Finding 2 confirmed: the frozen `val_map02_h1/metrics.json` keys are `psnr_dec, lpips_dec, copy_psnr_dec, latent_mse, hud_psnr_dec, psnr_raw, ...` with no latent-ratio columns; `eval_tf.py:72` inserts them at positions 5 and 6. `provenance.txt` files name `6a33311`/`ca03bad`; `git merge-base --is-ancestor` shows both precede `60d99c4` ("add a copy-last latent mse per window"), so the fixture pin is wrong.
- Finding 3 confirmed: `eval_tf.py:149` reads `m.update(pair_metrics(last, raw, lp, "copy_psnr_raw"))`, no LPIPS key, while the neighbouring model and vae calls pass one.
- Finding 1 confirmed in code: `adapt_split.py:162` passes horizon 1 to `window_starts`, which forwards it to `tic_window_starts`; spec item 3 requires the same windows at four tics.

## New file

- `paper/fixtures/test_eval_tf_sunday_review.py` (3 failing reproductions, 4 passing checks). No existing file edited, nothing committed. Astra's fixes were validated in memory only (29 passed) and not applied.
