# Astra review of the DoomDiT training and evaluation stack, Sep 9, 2026

Thread id (codex-reply): 01a086c2-3770-78b2-ae33-492ca5d174bb. Brief: `astra-brief-2026-09-09.md`.

## Verified against the real runs
- **Warmup asymmetry (confirmed):** under `accelerate.prepare` with 4 processes the LR scheduler advances 4x per optimizer step. DiT log: lr 4.0e-5 at step 50, 8.0e-5 at 100, 1.0e-4 at 150 (warmup effectively 125 steps). U-Net (1 process): 5.1e-6 at 50, 1.0e-5 at 100 (nominal 500). Reportable; fix for future runs is to step the unwrapped scheduler.
- **U-Net resume** reset optimizer state and `best_val` (known; logged Sep 9 08:12). Archived checkpoints allow re-selection; the archiver now also keeps `best_stepNNNNNNN.pt` copies (fixed 12:10).

## Accepted, fixes applied to evaluation code (Sep 9, 12:30)
- `rollout_eval.py --score`: clips were per decode chunk (16 frames each) rather than per rollout, so FVD16 mixed horizons and FVD32 was mislabeled. Fixed: one clip per rollout, `H >= frames` asserted.
- Validation determinism: timesteps and context noise now drawn from the seeded generator so val losses are paired across checkpoints and backbones (affects future evaluations; running processes keep the old behavior, so re-select `best` from archived checkpoints with the fixed evaluator).
- IDM movement classes now come from the most frequent button string per action id (robust to the anti-stuck override) with an assertion of full coverage.
- `--infer-noise`: levels quantized against the training maximum (0.7), per-sample buckets kept, clean context preserved for the persistence baseline.
- Gradient accumulation counter persists across epoch boundaries.
- Provenance: config.json now records the split file hash.

## Accepted, to be measured
- **Action-label corruption in the recording:** (a) the anti-stuck manual control executes a different button vector for 40 tics while `action` keeps the requested id (the `buttons` column holds the executed vector); (b) a death cuts the 4-tic decision short, so later decisions fall off the tic%4 grid used at encoding. Audit running on Spiderman (`audit_actions.py`); result decides how the action-following claim is phrased and whether a clean-transition subset is reported.
- Drift and FVD use decoded-latent references; the teacher-forced raw-frame pass is the primary table. Report both persistence baselines (decoded latent, raw frame).
- Per-episode paired bootstrap for confidence intervals; current SEM ignores within-episode dependence.
- I3D preprocessing not validated against a reference implementation.

## Disagreements or nuance
- Astra: "not a literal architecture-only causal claim." Agreed in wording: the paper claims a controlled comparison of two pretrained backbones under one recipe and nominal budget, and lists the residual asymmetries (pretraining corpus and conditioning history, warmup length, U-Net resume, single-token cross-attention).
- Bucket 0 covers noise fractions [0, 0.07), not clean only; inference with clean context is a shared boundary mismatch, not a bug.
