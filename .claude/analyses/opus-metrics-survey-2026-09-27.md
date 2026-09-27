# What the field puts on its y axes, and how to label ours (Opus reader, 2026-09-27)

I read the raw arXiv HTML (latest version) or the raw page HTML. Figure axes come from the figure images. Our definitions are from `paper/FIGURES.md` and `paper/main.tex` §2.

## One-line answers

1. **GameNGen** reports absolute PSNR and LPIPS against the raw ground truth: teacher-forced 29.43 dB / 0.249, per-step curves to 64 autoregressive steps, FVD16/32, and a human 2AFC (raters picked the real game 58 and 60 percent of the time). It draws no reference line and gives no numeric decoder result.
2. **No paper draws copy-last.** The closest forms are Genie's ΔₜPSNR (a paired difference against a counterfactual), WorldScore's fixed-camera floor and Vista's "GT video" row.
3. **Name A and S the same thing.** Both are *skill over copy-last* in dB (decoded and latent). Label A's axis "ΔPSNR vs copy-last (dB)", give the error ratio once in the caption, and drop "advantage".
4. **Flip M's sign.** Then above zero means "beats copy-last" on every A and M axis.
5. **Add absolute PSNR and LPIPS** (model and persistence) beside A and M in Table 2.
6. **Borrow GameNGen's Fig 6 with a copy-seed curve.** The rollout JSONs already hold it, so it costs CPU only.
7. **FVD** is possible by Tuesday for the appendix (`fvd.py` exists). SSIM is optional. A human study is not honest by Tuesday.
8. **Biggest pitfall.** Half of A in dB is not half the error removed. The 1.8 dB half-line removes 34 percent of copy-last's error, not 28.

## 1. What each paper reports

TF = one step on true context. AR = autoregressive or open-loop from real frames plus real actions.

| Paper (id, venue) | Results axis / column | TF / AR | Reference |
|---|---|---|---|
| GameNGen (2408.14837, ICLR 2025) | TF PSNR 29.43, LPIPS 0.249, 2,048 held-out trajectories, 5 levels. Fig 6: "PSNR" and "LPIPS" against "Auto-regressive Step" 0 to 63, 512 trajectories; read off the plot, about 28.7 to 19.8 dB and 0.26 to 0.455. FVD 114.02 (16 frames, 0.8 s) and 186.23 (32 frames). Table 2 has the decoder frozen: 22.36 dB | both | none; ablations only. Decoder tune qualitative (Fig 12) in v1 and v2 |
| DIAMOND (2405.12399, NeurIPS 2024) | Atari human-normalised score. Fig 8: "Average pixel distance" against "Timestep" 1 to 1000, log axis. Table 8: FID, FVD16 and LPIPS over 1,024 videos | AR | other world models |
| Genie (2402.15391, ICML 2024, PMLR 235) | FVD; ΔₜPSNR = PSNR(xₜ, x̂ₜ) − PSNR(xₜ, x̂′ₜ), with inferred against random latent actions, t = 4: 1.91 and 2.07 | AR | a counterfactual generation; the BC plot draws random and oracle bounds |
| Oasis (oasis-model.github.io, 2024 blog) | a throughput chart only | – | – |
| Vista (2405.17398, NeurIPS 2024) | FID 6.9 / FVD 89.4; 2AFC preference; per-command FVD; "Average Trajectory Difference", the L2 between the IDM's trajectory and the truth over 2 s | AR | a "GT video" row (0.379) and an action-free row |
| GAIA-1 (2309.17080, tech. report) | validation token cross-entropy against compute, on a geofenced set that includes roads never seen in training | TF, tokens | none |
| Cosmos (2501.03575, tech. report) | 3D: Sampson error, pose success, 3DGS PSNR/SSIM/LPIPS. Physics: PSNR, SSIM, DreamSim and tracked IoU against simulated truth over 33 frames | AR from 1 or 9 frames | a "Real Videos (Reference)" row; VideoLDM |
| XEWorld (2608.05799, arXiv) | PSNR/SSIM/LPIPS averaged over frames, region LPIPS, IoU, PCK; "percent of LPIPS gap closed" by fine-tuning | open-loop, 121-frame chunks | the seen robots; no persistence |
| DreamerV3 (2301.04104; Nature 640, 2025) | task-score bars; Minecraft return against steps; video prediction qualitative | – | expert agents, PPO |
| AdaWorld (2503.18938, ICML 2025) | PSNR/LPIPS on 4 unseen environments; Fig 6: "PSNR" against "Training Steps" 0 to 800 | horizon **VERIFY** | an action-agnostic model |
| WorldScore (2504.00983, ICCV 2025) | each metric normalised between empirical bounds, ×100; camera control's worst bound is "a sequence of fixed cameras" | AR | per-metric floor and ceiling |
| WorldMark (2604.21686, arXiv) | direction accuracy, purity, latency and stability from optical flow, 0 to 100 | AR | none |
| Matrix-Game (2506.18701, tech. report) | GameWorld Score: MUSIQ, aesthetic, CLIP temporal consistency, IDM keyboard and mouse accuracy | AR | Oasis, MineWorld |

## 2. Our quantities against theirs

S = same, R = related (same form or role, different reference or space), – = absent.

| Ours | GameNGen | DIAMOND | Genie | Vista | GAIA-1 | Cosmos | XEWorld | AdaWorld | Benchmarks¹ |
|---|---|---|---|---|---|---|---|---|---|
| **A**: ΔPSNR, model minus copy-last, both decoded | R (absolute) | – | **R** (ΔₜPSNR form) | – | – | – | R (% over baseline) | R (action-agnostic gap) | R (fixed-camera floor) |
| **M**: ΔLPIPS against raw persistence | R (absolute) | R | – | – | – | R | R (+ region) | R | – |
| **G**: reconstruction minus model PSNR | – | – | – | – | – | R (real-video row) | R (% gap closed) | – | R (ceiling) |
| **S**: latent MSE ratio to copy-last (dB) | – | – | – | – | **R** (decoder-free, unseen roads) | – | – | – | – |
| **Directional** (left/right swap) | – | – | **R** (counterfactual) | **R** (IDM, GT row) | – | – | R (PCK) | – | **R** (direction, IDM accuracy) |
| **Rollout vs copy-seed** | R (no reference) | R (log time) | – | R | – | R | R | R | – |

¹ WorldScore, WorldMark, Matrix-Game. No paper has an S entry. The ceiling / copy-last / model triple exists only in forecasting (OccWorld, DINO-Foresight; per `related-work-map-2026-09-26.md`, not re-read here).

## 3. Names, units, captions

- **A.** In the text, "decoded skill over copy-last". On the axis, "ΔPSNR vs copy-last (dB)"; Genie readers parse that at once. Caption once: "10·log₁₀ of copy-last's error over the model's, both decoded; 3 dB = half the error; windows averaged, then maps." Drop "advantage", which a CoRL reader takes for the RL advantage Q − V. "Gain" is already the repo's name for the raw difference.
- **S.** "latent skill over copy-last (dB)". It is the same log ratio as A in another space, so both share one scale. Give the ratio in the caption. For the U-Net, the training maps sit at 2.55 to 3.56 dB (latent error 0.56 to 0.44 of copy-last's) and the unseen arenas at 1.14 to 1.96 dB (0.77 to 0.64). Do not plot the ratio alone. It compresses the spread and inverts the direction.
- **M.** Keep a signed difference, not a ratio. LPIPS has a meaningful zero and a familiar scale. Flip it to LPIPS(copy-last) − LPIPS(model), labelled "LPIPS reduction vs copy-last (↑ better)". Figure 3b stacks A and M on one row with opposite senses today. If the flip is too late, put "(↓ better)" on every M label.
- **G.** "gap to decoder reconstruction (dB)", in a table only, with the reconstruction row printed above the models as Cosmos prints real video.
- **Directional.** "turn-reversal rate", with the true-frame estimator value in the same row as Vista's GT row: 0.892 on the fresh set and 0.885 on the training maps (`main.tex` prints 0.91). Persistence scores 0.
- **Rollout.** "PSNR vs raw frame (dB)" against tic, with the copy-seed curve drawn.
- **Absolute numbers.** Table 1 has them. Table 2 needs a "PSNR model / persistence (dB), scene rows" column, so readers of those papers have an anchor.

## 4. What GameNGen's figures offer

- **Fig 6/7, per-step PSNR and LPIPS.** Borrow them, adding the copy-seed curve and using DIAMOND's log-time x axis. `rollout_eval.py` saves `psnr`, `lpips` and `copy_seed_psnr` for h = 1 to 256. GameNGen's 64 steps at 20 fps are about 112 of our tics; ours reach 7.3 s. One appendix panel covers three backbones plus copy-seed, with no GPU. Keep LPIPS beside PSNR: our dossier (§5.5) found higher rollout PSNR tracking less motion, and copy-seed's LPIPS beat every stride-4 model at h64.
- **Human study.** GameNGen used 10 raters on 130 clips, plus 150 clips after 5 to 10 minutes of play. Not borrowable by Tuesday. Only the authors are at hand, and GameNGen itself says authors can tell the real game. State it as a limitation.
- **The 29.43 dB headline.** That number is at 20 fps, full frame with the HUD, through a fine-tuned decoder, on training levels. Its frozen-decoder ablation (22.36) sits near our 22.49 under different settings. Do not compare them.

## 5. What we lack

| Quantity | Who | Cost by Tuesday | Call |
|---|---|---|---|
| FVD16/32 | GameNGen, DIAMOND, Genie, Vista, AdaWorld | `fvd.py` ran on Sep 19 (`results/night_2026-09-19/q1-infer-noise/fvd*.json`). It needs about 512 AR clips per row; we have 16 rollouts per row | appendix, if a card idles Monday; include copy-seed clips as a row |
| SSIM | XEWorld, Cosmos | one metric in the rescore pass | only if free |
| Human 2AFC | GameNGen, Vista, AdaWorld | recruiting non-author raters | no; limitation |
| IDM accuracy | Vista, Matrix-Game | `train_idm.py` exists | the directional check covers control |

## 6. Pitfalls

1. **dB of differences.** A and S are log error ratios, so they add cleanly: "drops 2.1 dB" means the ratio worsened 1.6×. A percent change of a dB value is meaningless, so never write "A fell 58 percent". The half-gap line ½(A₀ + A_home) is a geometric midpoint in error, so say so in its caption.
2. **Averaging.** A mean of per-window dB is a geometric mean of ratios, not 10·log₁₀ of the ratio of mean errors. Name which one the paper uses.
3. **Three references in one table.** A is scored against the decoded truth, M against the raw truth with raw persistence, and G against the raw truth. Readers of every surveyed paper assume the raw frame. Add a formula footnote under Table 2 and "both decoded" or "against raw persistence" in each caption.
4. **Decoded copy-last is not raw persistence.** Swapping one for the other shifts gains by about 1.4 dB (`RESEARCH_CONTEXT.md` §0).
5. **Reference-free consistency scores reward static output.** This covers Matrix-Game's CLIP consistency and WorldMark's stability. Our paired references exist so that a static model cannot score; say so.
6. **Mixed signs on a shared zero line** (A up is good, M down is good): fixed by §3.
7. **Cross-paper absolute numbers.** Frame rate, decoder, HUD crop and holdout definition all differ.

## Sources

arXiv 2408.14837 (v1, v2), 2405.12399, 2402.15391 (PMLR v235 bruce24a), 2405.17398, 2309.17080, 2501.03575, 2608.05799, 2301.04104 (Nature s41586-025-08744-2), 2503.18938, 2504.00983, 2604.21686, 2506.18701; oasis-model.github.io. **VERIFY**: AdaWorld's adaptation horizon, and the GameNGen Fig 6 endpoints (read from the plot).
