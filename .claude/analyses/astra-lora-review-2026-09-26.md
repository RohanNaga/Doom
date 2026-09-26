# Astra review of the LoRA adaptation path (eaf565a), 2026-09-26

Model gpt-6-astra (reasoning high), sandbox workspace-write, no network. Thread `01a0e01b-dd77-7703-8541-a5c4f7fc0230`, 2 turns. Scope: commits 91eed7b..4ef7673 (lora.py, adapt_split.py, adapt_wm.py, score_adapt.py, eval_tf.py) against `docs/lora_adaptation_design_2026-09-26.html` sections 3 to 5 and the decision ledger, `cost-target-decision-2026-09-26.md`, and the 2026-09-26 20:10 EDT log entry.

## Verdict

**NO-GO for the complete experiment. Training can start Sunday; scoring and cost analysis are NO-GO until fixes 3 to 7 below land.** Astra found no defect in the default training update path. Before launch, three things must happen. First, validate and freeze the split by hand (fix 1). Second, inspect the source snapshot's clipping, weight decay and spike-guard args (fix 2). Third, pass the existing gates: step-0 parity on the real source, an A4000 throughput and memory sweep at global batch 32, and a live W&B curve. Keep every checkpoint so the pilot can be scored after the scorer is fixed.

## What passed (Astra's read, not GPU-verified)

The explicit EMA-to-model source loader works. Default LoRA selection covers attention q/k/v/out only. The control MLP, positions, input projection and noise-bucket embedding train in full, and the timestep embedding stays frozen. A single AdamW group gives every trained tensor the same learning rate and a 100-step warmup. EMA is stored and computed in fp32 and advances once per successful optimizer update, which matches the checkpoint steps. Checkpoints carry live and EMA tensors, the source hash, the source args and the certificate. Split permutations are deterministic and disjoint, and recorded windows are reused. W&B event wiring is present.

Tests: the five existing files give **96 passed, 41 warnings** (rerun by the supervisor). Astra added `paper/fixtures/test_adaptation_review_regressions.py`, and all **12 of its cases fail** on current code (rerun by the supervisor), one or more per finding.

## Findings (condensed from Astra)

1. **P1 `eval_tf.py:354`, `score_adapt.py:127`: the primary outcome is not implemented.** The code computes full-frame raw PSNR/LPIPS gains over persistence. The spec's outcome A is scene-only decoded advantage: PSNR(pred) minus PSNR(copy-last), both against the decoded true frame through the tuned SD 1 decoder, rows 0 to 207. The usage example at `score_adapt.py:10` still targets `heldout_psnr_gain_paired 0.46`, the raw-gain target the cost memo rejected. Fix: save per-window scene-only A (plus B and the C diagnostic), keep full frame as the check, freeze decoder identity and training-map anchors before adapted scoring, and cross at 25/50/100 percent of A.
2. **P1 `score_adapt.py:298`: the cache key omits inputs that change results.** It leaves out latent scale and shift, the decoder subfolder, the latents dir and the decoder contents (`decoder_tag` is only the path, `:288`). Raw recordings appear only as a boolean. Fix: key on one evaluation fingerprint covering all of these.
3. **P1 `score_adapt.py:325`: rescoring overwrites earlier evidence.** `out_dir` is `step{N}_{w}` only. A rescore with a new decoder or new sampler steps appends a row but overwrites the CSVs that older rows point to. Fix: put the fingerprint in the path and treat artifacts as immutable.
4. **P1 `score_adapt.py:378`: `cost` pools incompatible curves.** It groups by (run, map, seed, weights) only, so a stock-decoder curve and a tuned-decoder curve merge into one. Fix: group by the evaluation fingerprint, or reject mixed configurations and conflicting duplicate steps.
5. **P2 `score_adapt.py:399`: the default guards skip intermediate checkpoints.** `--guard-steps` defaults to `0,last`, but the spec asks for directional and forgetting reads at every grid point. The training-map inputs are also optional. Fix: default to `all` and require the training-map inputs.
6. **P2 `adapt_split.py:364`: split reuse ignores provenance.** `write_split` compares everything except `meta`. Identical lists from a different corpus or manifest hash therefore keep the old file silently. Fix: compare the identity-bearing meta fields.
7. **P2 `adapt_split.py:326`: short held-out episodes pass.** Fewer than 32 windows prints a note instead of refusing, which breaks the 8 x 32 = 256 protocol and weights episodes unequally. Fix: raise.
8. **P2 `score_adapt.py:190`: NaN counts as an observation.** A NaN at step 4000 extends `last_step_scored` (the censoring point) to 4000, and +inf can count as a crossing. Fix: reject non-finite values, or censor at the last finite observation.
9. **P2 `score_adapt.py:134`: the arithmetic ratio uses the log-valid mask.** A perfect window (ratio 0, skill +inf) is dropped, so ratios [0, 1] average to 1 instead of 0.5. Fix: `_finite_mean([p["latent_ratio"] for p in paired])`.
10. **P2, conditional, `adapt_wm.py:287`, `:389`: the source optimization policy is not inherited or checked.** Clipping and weight decay come from adapter CLI defaults, and the source trainer's finite-gradient spike guard is absent. Fix: assert these match the source args, or reject. Astra could not inspect the real 200k args.
11. **P3 `adapt_split.py:245`: a one-record JSONL manifest parses as a map-to-episodes dict.** It fails loudly before any GPU work. Use the record-list format for Sunday.

## Minimal fixes, ranked (Astra turn 2)

| # | Fix | Blocks |
|---|-----|--------|
| 1 | Split: refuse short draws, compare meta on reuse. Workaround with no code change: write the split to a new filename, verify provenance, disjointness and exactly 32 unique windows for each of 8 episodes, then freeze it. | training launch |
| 2 | Assert source clip, weight-decay and `skip_grad_norm` compatibility. No patch is needed if inspection confirms they match. | training launch (conditional) |
| 3 | Scene-only decoded advantage A (`[..., :208, :]`, paired PSNR difference), anchors frozen first, drop the 0.46 example | scoring |
| 4 | Evaluation fingerprint in the cache key and the artifact path | scoring |
| 5 | Run with `--guard-steps all --weights live,ema` and all training-map inputs (flag-only for Sunday) | scoring |
| 6 | One fingerprint per curve, reject conflicting duplicates, reject non-finite values | cost |
| 7 | Ratio mask one-liner | latent summary |

## Supervisor checks against the code

- **Finding 9 confirmed.** `score_adapt.py:134` reads `_finite_mean([p["latent_ratio"] for p in paired if math.isfinite(p["latent_skill"])])`. Line 124 sets skill to NaN when the ratio is not positive, and `-10*log10(0)` would be inf anyway. So a zero ratio is excluded from the arithmetic mean.
- **Finding 8 confirmed.** At `score_adapt.py:190`, `pts` filters only `r.get(metric) is not None`, so `float("nan")` rows stay in and `last_step_scored = pts[-1][0]` can be a NaN step.
- **Finding 4 confirmed.** `score_adapt.py:381` groups on `(r["run"], r["map"], r["seed"], r["weights"])`, and no decoder or sampler field is in the key.
- **Finding 5 confirmed.** `score_adapt.py:399` has `default="0,last"`.
- **Finding 3 confirmed.** At `score_adapt.py:325`, `out_dir = os.path.join(a.run_dir, "scores", f"step{step:07d}_{w}")` carries no evaluation identity.
- **Finding 7 confirmed.** At `adapt_split.py:326`, short draws only `print` a note.
- **Finding 6 confirmed.** At `adapt_split.py:364`, the comparison strips `meta` entirely.
- **Finding 1 spec basis confirmed, with a qualification.** The cost memo says "scene-only (rows 0 to 207) primary" (line 3) and "Half the raw gain (+0.46 dB) is already exceeded by four arenas, which kills the raw-gain target" (line 15), and `score_adapt.py:10` still uses `--target 0.46`. Astra flagged that the memo's synthesis is headed "recommended to Rohan", and that HTML decision 7 says "Rohan's yes freezes it". I found no RESEARCH_CONTEXT entry that records Rohan ratifying outcome A. **Open decision for Rohan: confirm A as the headline outcome before fix 3 is built.**

## Not verified

The real U-Net 200k snapshot, its args, the corpus and the decoder were not checked. CUDA step-0 parity, A4000 memory and throughput, multi-process torchrun and live W&B delivery were not checked either. The CPU tests do not certify any of these launch gates.
