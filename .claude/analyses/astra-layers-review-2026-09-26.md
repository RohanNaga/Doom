# Astra review: where physics and appearance live, and the middle-layer post-training (Sep 26 2026)

Thread: `01a0df4a-8564-7bc2-9ee1-18aa09a9a046` (gpt-6-astra, reasoning high, workspace-write, read-only in practice; 3 turns). Brief: `.claude/analyses/astra-brief-layers-2026-09-26.md`, passed verbatim. Main's design was withheld. Astra read from the worktree copies of RESEARCH_CONTEXT.md, the dossier and directional_check.py because root main (`687a9ff`) predates them. No repository files were changed, and no server or GPU work was done.

## Independent proposals (turn 1)

1. **Localization test.** Establish the behavioral separation per map first. Use 16 to 32 turn windows per direction on each of the 30 maps, estimator calibration on real frames per map, and fidelity against persistence on the same windows. Then run a small localization study on 2 training maps and 4 development maps. It has two parts: (a) equal-capacity readouts of recorded yaw change and next-frame change from early, middle and late activations at 3 noise levels, with controls-only and context-only baselines; (b) causal interventions. For the U-Net and PixArt, substitute the swapped newest control at groups of cross-attention sites. For SD 3.5, run a four-way table: joint-attention tokens recorded or swapped, crossed with the pooled vector recorded or swapped. Activation patching uses a self-patch no-op control. Budget: 6 A6000-hours, as a plan rather than a measurement.
2. **Axis.** The better axis is control pathway crossed with noise level, with depth read per architecture; the LLM layer axis is not the right one. There are two reasons. All 32 history frames enter through one channel-stacked input projection (`backbones.py:236`), so time is not a sequence axis that unfolds through depth. Every block also runs at every noise level. Compare changes in x0_hat, not raw v: `x0 = a*xt - s*v` (`diffusion_v.py:67-69`).
3. **Middle-layer head.** Run it only as a bounded pilot, and only behind a gate: a middle-layer readout has to beat an equal-capacity late-layer readout on held-out motion or residual prediction, and a causal patch has to move the output. Minimal design, on PixArt: one midpoint, a rank-64 bottleneck with timestep conditioning and a zero-initialized output projection, added to v. The backbone stays frozen and the default loss, data and batch are kept. Three arms: untouched, mid-head, and an equal-parameter late-head. Train 500 to 2,000 updates per arm.
4. **Four-page paper.** The question is whether camera-control transfer survives the loss of next-frame accuracy across maps. Contents: a three-backbone table, a per-map figure pairing degradation with turn response (estimator reliability beside it), one mechanistic panel (the SD 3.5 route table or a layer-by-noise heatmap), and one matched rollout panel.

## Reviewer critique (turn 1)

The same engine does not imply appearance-only failure: geometry, depth and collisions differ by map, and yaw is the easiest control to transfer. PSNR, LPIPS and latent MSE do not isolate appearance. The swap counterfactual has no engine-rendered target. A probe shows the information is present, not that the model uses it, and use is not exclusive storage. A skip-head gain needs the late-head control and a closed-loop check. Uncertainty has to be clustered by map and episode (30 maps are 30 units).

## Corrections after pushback (turns 2 and 3)

- **correct_frac.** Astra first claimed that an inverted model "can score perfectly". I pointed to `recorded_expected_sign_frac` (`directional_check.py:359`) and the per-direction `expected_sign_frac` (`:341`). It conceded: the only missing quantity is the per-window conjunction P_turn. The existing marginals bound P_turn at [0.787, 0.867] for the U-Net 200k EMA and [0.705, 0.805] for the SD 3.5 70k EMA (inputs at RESEARCH_CONTEXT.md line 243, verified). No conclusion changes, and P_turn can be computed from the saved JSON.
- **SD 3.5 routes.** The pooled vector comes from `tokens[:, n-1]` (`backbones.py:788-790`), so the mixed cells need an override of the transformer's inputs through a pre-hook. MMDiT evolves the control stream at every block, so localization has to patch the whole evolved context stream at a block boundary. It cannot substitute a single token. Hook points: U-Net and PixArt `...attn2`; SD 3.5 `transformer_blocks[b]` plus `time_text_embed`/`temb` for the pooled route. These were checked against diffusers 0.36 locally; the server runs 0.40 and was not checked.
- **Existing timestep data.** `val_loss_by_t_quartile` (`train_wm.py:765-775,859`; `wandb_log.py` `val_row`) comes from uniform-in-t bins, not log-SNR bins. Reading it costs no GPU time but cannot show causal pathway use.
- **Already running.** Told that the 30-map U-Net directional check and the latent ratio are already running, and that the rendering leg is flat, Astra re-ranked. PixArt 200k through the 30 maps (about 1.5 GPU-h) is the best marginal use of GPU 3, because it is the cross-backbone replication.

## Recommended experiments

| Rank | Experiment | Code | GPU | Destination |
|---|---|---|---|---|
| 0 | P_turn from saved JSON; t-quartile curves from W&B | about 100 lines, CPU | 0 | main text / appendix |
| 1 | PixArt 200k through the 30 maps (already queued) | none | about 1.5 h | main |
| 2 (optional, after 1) | t-sweep of turn-swap sensitivity: U-Net 200k EMA, 4 seen and 8 unseen maps, 32 windows per map, t in {100,300,500,700,900}, 2 seeds; S_t = RMS change in x0_hat, E_t, U_t; forward-only, modelled on `smoke_probe.token_sensitivity` (`:241`), not `gradient_and_sensitivity` (`:128`, which runs train mode and a backward pass, verified); needs EMA loading (the CLI loads live weights, verified) | 120 to 180 lines, sibling tool | 15 to 25 min (cap 30) | appendix; one sentence if there is a clear map-group x noise interaction |
| Monday GPU 2 | SD 3.5 30 maps; confirm the strongest pathway effect | 100 to 200 lines | about 4 h each | main / appendix |

Cut: all-backbone layer sweeps, full grids, unfreezing the backbone, and any head without the gate.

Caveats it raised on the paper's decomposition: a noise-level split measures conditional denoising difficulty on noised ground truth. It does not separate layout from texture; that needs spatial-frequency residuals. The flat reconstruction ceiling does not prove the decoder behaves the same on off-distribution predicted latents, so "at most 0.2 dB" is not an upper bound.

## Middle-layer post-training before Wednesday

**No.** The reason a reviewer would find most convincing: there is no evidence that the failure comes from useful information being out of reach of the final head. A head would add a new architectural hypothesis that needs a late-head control and a closed-loop check. It is also a deliberate departure from the default recipe. The gate that would flip the answer is PixArt mid beating late by at least 10% in residual-prediction MSE across maps and 2 noise bins, plus a causal patch that shifts the output without hurting fidelity. For Keerthana: the U-Net already routes early features to its decoder through skips, and each block serves every noise level, so the LLM prior that middle layers are best does not carry over.

Future-work sentence: "We will test whether timestep-conditioned readouts of intermediate features improve cross-map prediction beyond equal-capacity final-layer readouts, while preserving control fidelity and closed-loop stability."
