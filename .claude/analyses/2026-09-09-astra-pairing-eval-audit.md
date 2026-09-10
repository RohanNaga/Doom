# Astra pairing session: evaluation audit and paper framing (2026-09-09)

Second-engineer session with Astra (OpenAI Codex, gpt-6-astra) on the paper's
framing risk. Codex thread `01a086b4-4b55-7c93-811c-59b6b1a08f33`. Brief was
deliberately withheld of any position of mine, to get an independent read.

Astra's recommendation note is **staged but uncommitted** in
`RESEARCH_CONTEXT.md`; Codex's own approval reviewer blocked the commit because
the task authorized analysis, not a history change. Rohan decides whether it
lands.

## Two evaluation bugs, independently confirmed against the code

**1. FVD clips are chunk fragments, not rollouts.** `rollout_eval.py:108-116`
decodes in `decode_batch` chunks and appends each chunk to `clips_pred` inside
that loop. With `H=64` and `decode_batch=16`, each rollout contributes four
16-frame entries, and `np.stack(clips_pred)` at line 142 treats them as
independent clips. So FVD32 receives 16-frame clips, and the nominal 256 clips
are 64 rollouts counted four times. Any FVD number reported from this path is
wrong. Fix the export and assert clip length and provenance before reporting.

**2. The copy-last baseline is a VAE round-trip.** `eval_tf.py:98` computes
`last_img = decode(vae, ctx[:, -4:])`, so persistence is scored from a decoded
latent rather than the raw previous frame. The reported 19.73 dB is therefore
not raw persistence.

## Why bug 2 is material, and why the obvious correction is wrong

I argued the fix must raise the baseline, since a raw previous frame carries no
autoencoder distortion, and that the DiT's 20.29 vs 19.73 dB margin could
vanish. Astra agreed the concern is material but rejected the reasoning:

- The scoring is decoded-vs-decoded, so both the persistence input **and** the
  reference target change under a fix. You cannot lift 19.73 and hold 20.29.
- Writing `d = x_{t-1} - x_t` and `e = x̂_{t-1} - x_{t-1}`, the decoded-copy MSE
  against a raw target is `MSE_raw_copy + MSE_recon + 2⟨d, e⟩`. The cross term
  has either sign. Smoothing a previous frame can move it closer to the next
  one, and with both frames reconstructed the error term `e_{t-1} - e_t` can
  cancel through temporal correlation. Raw persistence is the correct baseline
  but not necessarily the higher-scoring one.
- The scripts average per-frame PSNR, so converting that average back to MSE
  yields a geometric mean, not the arithmetic MSE the subtraction needs.
- The 29.11 dB reconstruction figure comes from a different frame sample, and
  VAE reconstruction is a reference, not a strict ceiling.

Sensitivity calculation, explicitly not an estimate: 19.73 dB is 0.01064
normalized MSE, 29.11 dB is 0.001227. If the baseline were decoded-copy against
a raw target with uncorrelated reconstruction error, removing the reconstruction
term gives 20.26 dB. That hypothetical **+0.53 dB is almost the whole reported
0.56 dB margin**.

## The 48-hour action that settles it

Score the **same fixed windows** four ways, saving per-window PSNR, LPIPS,
episode, and start:

1. raw previous → raw target
2. reconstructed previous → raw target
3. reconstructed previous → reconstructed target
4. generated prediction → raw target

The first three need no sampling at all. This measures the baseline correction
and the true model margin directly. Do it before any further training ablation.

## The control claim is not defensible as written

"Backbone is the only variable" does not survive the details: DiT is
ImageNet-pretrained and the U-Net is SD-pretrained, roughly 675M vs 860M
parameters, different native action-conditioning pathways, the U-Net took an
optimizer reset at 5k, and the paper skeleton lists different learning rates
despite the shared-recipe description. Resolve the learning rates from the saved
run arguments.

The honest frame is a comparison of **two pretrained backbone packages under a
shared adaptation protocol**, which still answers what a researcher can pick
today. Astra's suggested question: under a shared data and adaptation protocol,
how do an ImageNet-pretrained DiT and an SD-pretrained U-Net trade prediction
quality, action responsiveness, autoregressive fidelity, and training cost.

Also: measured cost is roughly 111 GPU-hours for the DiT (90k at 0.9 steps/s on
four GPUs) and 83 to 109 for the U-Net, so the April "72 GPU-hours" line cannot
carry forward.

## If neither backbone beats raw persistence

Astra's position, which I did not push back on:

| Evidence at Sep 26 | Honest paper |
|---|---|
| Loses on next-frame averages but wins on longer-horizon fidelity or shows reliable action response | Study of the gap between next-frame reconstruction and action-conditioned prediction |
| Loses, but a controlled measurement identifies a reproducible evaluation effect or localized failure | Negative-result study of this bounded adaptation protocol |
| Loses everywhere, action response unconvincing, no diagnostic survives | Not a paper. An unsuccessful-run report. |

The rollout figure changes shape under this outcome: compare both models against
**copying the initial seed frame** at every horizon, with no ground-truth
refresh for persistence. A model can lose at one step and win once movement
accumulates.

## Other findings worth acting on

- Nonzero inference-noise evaluation overwrites every sample's bucket with the
  batch maximum. Fix before any inference-noise sweep; clean-context evaluation
  avoids the path.
- `eval_tf.py` does not report copy-last LPIPS, which the proposed main table
  needs.
- The 69.3% IDM top-1 is an empirical reference, not a ceiling. Do not normalize
  generated accuracy by it and call the result recovered controllability.
- Per-window SEM understates dependence. Bootstrap paired differences by
  episode, stratified by map.
- The `val` split has already selected checkpoints and informed development.
  Call it held-out validation, not an untouched test, unless untouched episodes
  are reserved.
- Report horizons in seconds as well as steps: at stride 4, horizon 64 is about
  7.3 s. Mark horizon 32, after which context is entirely generated.

## Astra would cut

From-scratch and warm-start ablations, further context or augmentation sweeps,
unseen-map results, HUD OCR and sampler grids, a DiT-only second seed (replicate
both or neither), and all cross-paper ranking against GameNGen plus the April
headline rows.
