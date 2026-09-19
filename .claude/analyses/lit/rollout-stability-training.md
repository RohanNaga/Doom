# Training action-conditioned world models to survive their own errors

**Question.** How do action-conditioned world models handle exposure bias — the gap between training
on ground-truth context and rolling out on their own predictions?

**Written** 2026-09-19. Companion to `frame-spacing-prior-work.md`. Same rules: primary sources
only, a quote or a file:line for every factual cell, "not stated" where the source is silent,
**VERIFY** where I could not confirm.

---

## 0. What we currently do (verified in this worktree)

| | |
|---|---|
| Method | GameNGen-style context-noise augmentation, one noise level per **sample**, applied to **all** context latents |
| Parameterisation | **Variance-preserving**: `a * context + s * eps` with `a = sqrt(1 - level)`, `s = sqrt(level)` — `diffusion_v.py:135-139` |
| Level distribution | `level ~ U(0, 0.7)` — `diffusion_v.py:132`, `--noise-aug-max` default 0.7 |
| Buckets | 10, `bucket = clamp(floor(level / 0.7 * 10), max=9)` — `diffusion_v.py:133` |
| Level given to model? | Yes, as a bucket id embedding, alongside action and diffusion timestep — `train_wm.py:303-305` |
| Validation loss | computed **with** seeded noisy context (`SeededCorruption`) — `train_wm.py:265-272` |
| Teacher-forced eval | `eval_tf.py:177` `--infer-noise` default **0.0** → clean context, `bucket = 0` |
| Rollout eval | `rollout_eval.py:104-106` — with `--infer-noise 0.0` the fed-back context is **clean**, `bucket = 0`; only `> 0` triggers `fixed_noise` |
| Rollout context update | `rollout_eval.py:110` — drop oldest latent, append the model's own prediction; no re-noising |

**Two things to note before reading the literature.**

1. **Bucket 0 is not a "clean" bucket.** `floor(level / 0.7 * 10)` maps every `level ∈ [0, 0.07)` to
   bucket 0, and `level` is drawn from a continuous uniform, so the model has **essentially never
   seen bucket 0 paired with exactly-clean context**. At `--infer-noise 0.0` we feed it clean
   context under that bucket. This is a small self-inflicted train/test mismatch, and it is on the
   same axis as the exposure bias we are trying to fix.
2. **Our 0.7 is not the same amount of corruption as the reproduction's 0.7.** Ours is
   variance-preserving (at level 0.7 the context signal is scaled by `sqrt(0.3) = 0.55`); the public
   reproduction adds noise **additively** without rescaling the signal —
   `utils.py:29-31`, `conditional_noise = torch.randn_like(frames) * noise_level; return frames + conditional_noise`.
   Both cite the same "0.7" from GameNGen. At most one of them matches what GameNGen did. GameNGen
   says only "we corrupt context frames by **adding** a varying amount of Gaussian noise to encoded
   frames" while citing Ho et al. (2021) cascaded-diffusion conditioning augmentation, which is a
   forward-diffusion (variance-preserving) construction. **Our reading is the more likely one, but
   it is a reading** — **VERIFY** against Ho et al. (2021) if the number matters.

---

## 1. Summary table

| Paper | Method for exposure bias | Key hyperparameters | What happens at inference | Ablation evidence |
|---|---|---|---|---|
| **GameNGen** 2408.14837 | context noise augmentation | max level **0.7**, **10** buckets, level **given to model** as embedding | "the added noise level **can be controlled** to maximize quality, although **even with no added noise** the results are significantly improved" | Fig. 7, autoregressive only: without it "quality degrades quickly after 10-20 frames" |
| **gameNgen-repro** (code) | same | `MAX_NOISE_LEVEL = 0.7`, `NUM_BUCKETS = 10`, **additive** noise, `round` not `floor` | not verified in `run_autoregressive.py` | none |
| **MultiGen** 2603.06679 | context noise augmentation **+ history guidance** | "randomly sampled noise scale" | **clean context in the conditional branch, noised context in the unconditional branch** (CFG) | external-memory ablation only |
| **DIAMOND** 2405.12399 | **EDM over DDPM**, as the drift fix | n ≤ 10 denoising steps | n/a | "ddpm ... leads to severe compounding error ... In contrast, the edm-based diffusion world model appears much more stable over long time horizons" (App. K, t = 1000) |
| **Oasis** (Decart/Etched) | Diffusion Forcing + **dynamic noising** | not stated | **injects noise early in the denoising passes, removes it later** | none published |
| **Diffusion Forcing** 2407.01392 | independent per-token noise levels | `k_t ~ U{0..K}` i.i.d. per token | **keeps a small noise level** `0 < k_small ≪ K` on the fed-back latent | 1000-frame stable rollout vs teacher-forcing and full-sequence baselines that "diverge quickly" |
| **DFoT / History Guidance** 2502.06764 | DF on a transformer + CFG over history | per-frame noise levels | unconditional branch = history **masked with complete noise** | 862-frame rollout from one image |
| **Self Forcing** 2506.08009 | **train on own rollouts** + distribution-matching loss | DMD/SiD/GAN loss on the finished video | rolling KV cache | VBench SOTA at 17.0 FPS |
| **CausVid** 2412.07772 | asymmetric distillation from a bidirectional teacher | 4-step causal student | — | "the causal diffusion teacher suffers from significant error accumulation (orange), which is then transferred to the student" |
| **Matrix-Game 2.0** 2508.13009 | **Self-Forcing** distillation | 3 denoising steps | — | qualitative |
| **GameFactory** 2501.08325 | **varying noise levels across frames** in AR fine-tuning | k conditional frames, k randomly selected | "iteratively selects the latest k+1 frames as conditions" | none |
| **PlayGen** 2412.00887 | none stated; relies on RNN-like architecture | — | — | "after 1000 frames, the accuracy of interactive mechanics decreases by only 0.2%" |
| **Genie** 2402.15391 | **none stated** | — | — | none |
| **MineWorld** 2504.08388 | **none stated** | — | — | none |
| **WHAM** (Nature 2025) | **not stated** on the model card | — | — | none |
| **NWM** 2412.03572 | none for context; time-shift training instead | — | — | 1 FPS vs 4 FPS crossover at 8 s |
| **GAIA-1** 2309.17080 | guidance scheduled over frames | — | — | not quantified |
| **Vista** 2405.17398 | triangular guidance schedule over time, `s ∈ [1.0, 2.5]` | — | — | not isolated |
| **IRIS / DreamerV3** | **different regime** — discrete/latent rollouts, no pixel context to corrupt | — | — | see §2 |
| **Rolling Diffusion** 2402.09470 | noise increasing with frame index, sliding window | — | window shifts each step | — |
| **FIFO-Diffusion** 2405.11473 | diagonal denoising queue | — | dequeue clean head, enqueue noise at tail | latent partitioning + lookahead denoising to close the train/test gap |
| **Cosmos** 2501.03575 | diffusion decoder over AR tokens | — | — | failure-mode study, not an ablation |
| **Scheduled sampling** 1506.03099 | curriculum from ground-truth to own tokens | — | n/a | the lineage for all of the above |

---

## 2. Per-paper detail

### GameNGen — arXiv 2408.14837 (§3.2.1, §4.2, §5.2.2)
https://arxiv.org/abs/2408.14837

- Method, §3.2.1 "Mitigating Auto-Regressive Drift Using Noise Augmentation":
  > "The domain shift between training with teacher-forcing and auto-regressive sampling leads to
  > error accumulation and fast degradation in sample quality... we **corrupt context frames by
  > adding a varying amount of Gaussian noise to encoded frames in training time, while providing the
  > noise level as input to the model**, following Ho et al. (2021). To that effect, we sample a noise
  > level α uniformly up to a maximal value, **discretize it and learn an embedding for each bucket**.
  > This allows the network to correct information sampled in previous frames, and is critical for
  > preserving frame quality over time."
- **The inference statement** (same paragraph, this is the quote for Q3):
  > "**During inference, the added noise level can be controlled to maximize quality, although we find
  > that even with no added noise the results are significantly improved.**"

  Read precisely: "even with no added noise the results are significantly improved" compares
  *noise-augmented training* against *no noise augmentation at all*; it does not say zero is optimal
  at inference. The first clause says the level is a tunable inference knob. **GameNGen never states
  the inference level it used for its own headline numbers.**
- Hyperparameters, §4.2: "For noise augmentation (Section 3.2.1), we use a **maximal noise level of
  0.7, with 10 embedding buckets**." Our `--noise-aug-max 0.7` / `--noise-buckets 10` match exactly.
- Ablation, §5.2.2:
  > "To ablate the impact of noise augmentation we train a model without added noise. We evaluate
  > both our standard model with noise augmentation and the model without added noise (after 200k
  > training steps) **auto-regressively** and compute PSNR and LPIPS metrics ... up to a total of 64
  > frames in Figure 7. **Without noise augmentation, LPIPS distance from the ground truth increases
  > rapidly compared to our standard noise-augmented model, while PSNR drops**, indicating a
  > divergence of the simulation from ground truth."

  Figure 7 caption: "When noise augmentation is not used **quality degrades quickly after 10-20
  frames**. This is prevented by noise augmentation."
- **The ablation is a figure, not a table — there are no with/without numbers to cite.** And it is
  **autoregressive only**. GameNGen reports **no teacher-forced comparison** with and without noise
  augmentation.
- For scale, the one temporal ablation that *is* tabulated (Table 2, context length, 200k steps,
  frozen decoder) spans 20.94 dB (1 frame) to 22.36 dB (64 frames) — **1.4 dB for a 64x context
  change**. Useful calibration for how small these effects are at fixed everything-else.

### gameNgen-repro — github.com/arnaudstiegler/gameNgen-repro

- `config_sd.py:17-19`: `# Those values are the same as in the paper` / `MAX_NOISE_LEVEL = 0.7` /
  `NUM_BUCKETS = 10`.
- `utils.py:5-7` — bucketing uses **round**, not floor:
  ```python
  def discretize_noise_level(noise_level: float) -> int:
      size_bucket = torch.tensor(MAX_NOISE_LEVEL / NUM_BUCKETS)
      return torch.min(torch.round(noise_level / size_bucket), torch.tensor(NUM_BUCKETS - 1)).to(torch.int32)
  ```
- `utils.py:21` — `noise_level = torch.rand(...) * MAX_NOISE_LEVEL`, one level per sample (same as
  ours).
- `utils.py:29-31` — **additive, not variance-preserving**:
  ```python
  conditional_noise = torch.randn_like(conditioning_frames) * noise_level.view(-1, 1, 1, 1, 1)
  noisy_conditioning_frames = conditioning_frames + conditional_noise
  ```
- Inference-time noise in `run_autoregressive.py`: **not verified** — the file references a
  `noise_scheduler` but I did not trace whether conditioning noise is applied during rollout.
  **VERIFY.**

### MultiGen — arXiv 2603.06679 (the most directly applicable recipe)
https://arxiv.org/abs/2603.06679

Doom, diffusion game engine, and it does **both** halves:

- §3.2 "Noised-context training for drift robustness":
  > "During training, the observation module conditions on ground-truth context frames, whereas at
  > test time it conditions on its own generated history. To reduce this train–test mismatch, we
  > follow prior work [Diffusion Forcing, GameNGen] and **corrupt all context frames with Gaussian
  > noise at a randomly sampled noise scale during training**. This exposes the model to imperfect
  > histories and improves robustness under long autoregressive rollouts."
- §3.4, at inference:
  > "To stabilize long rollouts, we use **history guidance**: **the conditional branch receives the
  > clean context frames, while the unconditional branch receives a noised version of the context**,
  > encouraging fidelity to recent history."

  This is the cheapest concrete idea in the whole card for us: it needs no retraining, only a second
  forward pass at a nonzero bucket. Max noise scale, guidance weight and the unconditional bucket
  are **not stated**.
- No with/without ablation of either mechanism is reported; the paper's Table 1 ablates external
  memory, and notes "larger gains in later rollout segments where implicit-state baselines are more
  prone to drift".

### DIAMOND — arXiv 2405.12399 (Appendix K)
https://arxiv.org/abs/2405.12399

DIAMOND's answer to drift is the **diffusion parameterisation**, not context corruption:

> "While diamond utilizes edm (Karras et al., 2022) ... ddpm (Ho et al., 2020) would also be a
> natural candidate ... To provide a fair comparison of ddpm with our edm implementation, we train
> both variants with the same network architecture, on a shared static dataset of 100k frames
> collected with an expert policy on the game Breakout."

> "To investigate the stability of the diffusion variants, we display imagined trajectories generated
> autoregressively up to **t = 1000 timesteps** in Figure 3, for different numbers of denoising steps
> n ≤ 10. We see that **using ddpm in this regime leads to severe compounding error, causing the
> world model to quickly drift out of distribution. In contrast, the edm-based diffusion world model
> appears much more stable over long time horizons**, even for [few denoising steps]."

Also motivating, §1: "Discretization of the latent space helps to avoid compounding error over
multi-step time horizons. However, this encoding may lose information" — i.e. the discrete-latent
world models buy stability by throwing away detail. Evidence is qualitative (figures), **no numeric
ablation**.

### Oasis — Decart/Etched
https://oasis-model.github.io/

- "The model was trained using **Diffusion Forcing**, which denoises with **independent per-token
  noise levels**, and allows for novel decoding schemes such as ours."
- The inference recipe, and the clearest statement anywhere that inference noise should be nonzero:
  > "**Dynamic noising**, which **adjusts inference-time noise on a schedule, injecting noise in the
  > first diffusion forward passes to reduce error accumulation, and gradually removing noise in the
  > later passes** so the model can find and persist high-frequency details in previous frames for
  > improved consistency."
- "Since our model saw noise during training, it learned to successfully deal with noisy samples at
  inference."
- "In autoregressive models, errors compound, and small imperfections can quickly snowball into
  glitched frames."
- No levels, schedule shape or ablation numbers are published. **VERIFY** against the open-weights
  code if we adopt this.

### Diffusion Forcing — arXiv 2407.01392
https://arxiv.org/abs/2407.01392

- Training, §2: "each token is associated with a **random, independent noise level**"; "We train the
  model to denoise all tokens of a sequence at once, with an independent noise level per token."
  Appendix B.2: "we choose to sample per-token noise level following **i.i.d uniform distribution
  from [1, 2...K]**."
- Stabilisation, §3.3:
  > "For high-dimensional, continuous sequences such as video, auto-regressive architectures are
  > known to diverge, especially when sampling past the training horizon. In contrast, Diffusion
  > Forcing can stably roll out long sequences even beyond the training sequence length by **updating
  > the latents using the previous latent associated with slightly 'noisy tokens' for some small
  > noise level 0 < k ≪ K**."
- Appendix B.4 spells out the inference procedure and the justification:
  > "At each time t, during the denoising, we maintain a latent z_{t-1}^{k_small} from the previous
  > time step, with **0 < k_small ≪ K corresponding to some small amount of noise**."

  > "It is widely appreciated that **adding noise to data ameliorates long-term compounding error in
  > behavior cloning applications**, and even induces robustness to non-sequential adversarial
  > attacks. In autoregressive video generation, the noised x_t^{k_small} **is in-distribution for
  > training, because Diffusion Forcing trains from noisy past observation in its training
  > objective**."
- Result, §4.1 (Minecraft and DMLab): "**While Diffusion Forcing succeeds at stably rolling out even
  far beyond its training horizon (e.g. 1000 frames), teacher forcing and full-sequence diffusion
  baselines diverge quickly.**" Qualitative (Figure 3); no PSNR-vs-horizon curve.

**This is the strongest single argument against our `--infer-noise 0.0`:** the method whose whole
point is stability keeps a small nonzero level at inference, and argues explicitly that the clean
fed-back frame is the out-of-distribution one.

### DFoT / History Guidance — arXiv 2502.06764
https://arxiv.org/abs/2502.06764

- "Extending the 'noising-as-masking' paradigm in Diffusion Forcing to non-causal transformers, DFoT
  trains video diffusion models by **applying independent noise levels to each frame**."
- §5, Vanilla History Guidance (HG-v):
  > "To perform CFG, we need to estimate the unconditional score ... the unconditional score is a
  > special case of the conditional score with H = ∅ and can be estimated by **masking history frames
  > with complete noise**. **Even this simple form of HG significantly improves generation quality and
  > consistency.**"
- The generalised form (Eq. 5) weights several conditional scores, "each masked with a **possibly
  different noise level** k_{H_i}", with the stated rationale: "By composing scores, each individual
  score component operates on a restricted conditional context, **reducing the likelihood of being
  out-of-distribution**."
- Headline: "Diffusion Forcing Transformer with history guidance enables stable rollout of extremely
  long videos. We visualize 21 frames from an **862-frame long navigation video** generated by our
  DFoT model from a single image."
- Per-metric gains are in tables I did not extract — **VERIFY**.

### Self Forcing — arXiv 2506.08009
https://arxiv.org/abs/2506.08009

- The problem statement is precisely ours:
  > "It addresses the longstanding issue of **exposure bias**, where models trained on ground-truth
  > context must generate sequences conditioned on their own imperfect outputs during inference."
  > "models trained with TF or DF often **suffer from error accumulation** during autoregressive
  > generation, leading to degraded video quality over time ... a model is trained exclusively on
  > ground-truth context but must rely on its own imperfect predictions at inference time, resulting
  > in a distributional mismatch that compounds errors as generation progresses."

  Note that this names **Diffusion Forcing as insufficient**, not just teacher forcing.
- The method: "Our Self Forcing approach performs **autoregressive self-rollout during training**,
  denoising the next frame based on previous context frames **generated by itself**. A
  distribution-matching loss (e.g., SiD, DMD, GAN) is computed on the final output video."
- "our chunk-wise autoregressive model achieves the highest VBench scores across all compared models
  while simultaneously delivering real-time throughput (17.0 FPS) with sub-second latency."
- Cost: a rollout inside every training step. This is the expensive end of the menu.

### CausVid — arXiv 2412.07772
https://arxiv.org/abs/2412.07772

"autoregressive models are prone to error accumulation: each generated frame builds on potentially
flawed previous frames, causing prediction errors to magnify and worsen over time." Their fix is
asymmetric distillation from a bidirectional teacher into a 4-step causal student: "We show that
this asymmetric distillation approach **significantly reduced error accumulation** during
autoregressive inference." Diagnostic worth copying: "the causal diffusion teacher suffers from
significant error accumulation (orange), which is then **transferred to the student** (green)"
(Fig. 8).

### Matrix-Game 2.0 — arXiv 2508.13009
https://arxiv.org/abs/2508.13009

"we develop an auto-regressive diffusion model ... through **Self-Forcing**, which addresses exposure
bias by conditioning [on its own outputs]"; "**Critically, the generator samples previous frames from
its own distribution rather than the ground-truth training data**, mitigating training-inference
[mismatch]". Two phases: student initialisation on ODE trajectories, then DMD-based Self-Forcing.
The current state of the art in interactive game video uses self-rollout, not context noise.

### GameFactory — arXiv 2501.08325
https://arxiv.org/abs/2501.08325

§4.3: "we extend our model to autoregressive long video generation by **allowing varying noise levels
across different frames**, continuously conditioning on previously generated frames". Training: "The
frames from index 0 to k serve as conditional frames, while the remaining N−k frames are for
prediction, **with k randomly selected**... Loss computation and optimization focus only on the noise
of predicted frames." Inference: "The model iteratively selects the latest k+1 frames as conditions
to generate N−k new frames." Effectively diffusion forcing with a randomised split point. No
hyperparameters and no ablation given.

### PlayGen, Genie, MineWorld, WHAM — no stated mechanism

- **PlayGen** (2412.00887): claims the best long-rollout stability of any Doom paper — "after 1000
  frames, the accuracy of interactive mechanics decreases by only 0.2%" — and attributes it to
  architecture, not training: "we employ an **RNN-like model architecture that theoretically
  possesses infinite memory**, ensuring extended gameplay" (§1); "to maintain long-term memory under
  limited computing resources, we employ an autoregressive RNN-like model structure for LDM" (§4).
  No context corruption, scheduled sampling or self-rollout is mentioned. **Their 0.2% is a
  mechanics-accuracy metric, not PSNR**, so it is not comparable to GameNGen's Figure 7.
- **Genie** (2402.15391), **MineWorld** (2504.08388): searched for error accumulation / compounding /
  drift / exposure bias / scheduled sampling / context corruption — **no matches**. Genie's stated
  limitation is memory length ("limited to 16 frames of memory"), not drift.
- **WHAM** model card: no drift-mitigation method stated.

### IRIS and DreamerV3 — a different regime, deliberately

Worth one paragraph in the paper because the comparison is often made sloppily.

- These models roll out in a **compact latent/discrete space** and are trained with a multi-step
  imagination objective; there is no high-dimensional pixel context to corrupt, and the RSSM/
  transformer carries a recurrent state rather than a window of past frames. DIAMOND states the
  tradeoff directly (§1): "**Discretization of the latent space helps to avoid compounding error over
  multi-step time horizons.** However, this encoding may lose information, resulting in a loss of
  generality and reconstruction quality."
- IRIS's only context-hygiene device is burn-in: "Before starting the imagination procedure from a
  given frame, we **burn-in the 20 previous frames** to initialize the hidden state" (Appendix);
  DIAMOND does the same for its actor-critic LSTM. Burn-in is not an exposure-bias fix — it is state
  initialisation.
- Neither reports a context-noise or self-rollout ablation. **Their stability comes from the latent
  bottleneck and from being scored on returns rather than pixels**, so "DreamerV3 rolls out for
  hundreds of steps" is not evidence about pixel-space world models.

### GAIA-1 / GAIA-2 / Vista / NWM / Cosmos

- **GAIA-1** (2309.17080): "We found it was important to **schedule the scale factor used for
  guidance over tokens as well as frames**." Not quantified; no context corruption stated.
- **Vista** (2405.17398): a **triangular guidance schedule over the temporal axis**, "we define
  s_min as 1.0 and s_max as 2.5. This triangle scheme assigns moderate guidance scales to the frames
  that will be used as conditions". So the frames that will become context get *less* aggressive
  guidance — an inference-side stability heuristic. Not ablated in isolation.
- **NWM** (2412.03572): no context-corruption mechanism. Its robustness argument is the time-shift
  training itself, and its rollout result is the 1 FPS / 4 FPS crossover already recorded in
  `frame-spacing-prior-work.md`.
- **Cosmos** (2501.03575): mitigates AR artifacts *after the fact* with a diffusion decoder — "We
  further fine-tune our pre-trained diffusion WFM to arrive at a **diffusion decoder to enhance the
  generation results of the autoregressive model**." Their failure analysis is a 100-input study of
  "objects unexpectedly appearing from below", not a drift ablation.

### The schedule-shaped alternatives (lineage and adjacent)

- **Scheduled sampling** (1506.03099) is the ancestor: "tokens ... are thus replaced by tokens
  generated by the model itself, yielding a **discrepancy between how the model is used at training
  and inference**"; "We propose a **curriculum learning strategy** to gently change the [sampling
  from ground truth to sampling from the model]."
- **Rolling Diffusion** (2402.09470): "a method that **explicitly corrupts data from past to
  future**", with a sliding window where noise level increases with frame index.
- **FIFO-Diffusion** (2405.11473): "**diagonal denoising**, which simultaneously processes a series
  of consecutive frames with **increasing noise levels in a queue**; our method dequeues a fully
  denoised frame at the head while enqueuing a new random noise frame at the tail." Notably it
  admits the same failure mode we would create: "such a strategy **induces the discrepancy between
  training and inference**. Hence, we introduce **latent partitioning** to reduce the
  training-inference gap and **lookahead denoising**."
- Diffusion Forcing positions itself against these two (App. B.2): "AR-diffusion and Rolling
  Diffusion can only achieve the first and third [capability]... any tuning of the sampling scheme
  would require **re-training** the model for AR-diffusion [whereas DF trains once]."

---

## 3. Answers

### Q1. Which methods have evidence of helping long rollouts, and by how much?

Ranked by strength of evidence, and note how weak "strongest" is here — **not one paper in this card
reports a with/without number at a stated horizon**. Everything is figures or qualitative.

1. **Context noise augmentation (GameNGen).** "Without noise augmentation, LPIPS ... increases
   rapidly ... while PSNR drops"; the figure caption localises the failure: "quality degrades
   quickly **after 10-20 frames**". Magnitude: **not reported** (Figure 7 only). Independently
   adopted by MultiGen, which calls it "noised-context training for drift robustness".
2. **Diffusion Forcing (per-frame independent noise) plus small inference noise.** "stably rolling
   out even far beyond its training horizon (e.g. **1000 frames**), [while] teacher forcing and
   full-sequence diffusion baselines diverge quickly". Magnitude: not reported.
3. **Self-rollout training (Self Forcing, and Matrix-Game 2.0 which adopts it).** Directly optimises
   the rollout distribution; "achieves the highest VBench scores ... at 17.0 FPS". Most expensive.
4. **History guidance (DFoT, used by MultiGen).** "Even this simple form of HG significantly improves
   generation quality and consistency"; 862-frame rollout. Inference-only, no retraining.
5. **EDM over DDPM (DIAMOND).** Same architecture, same data, t = 1000: "ddpm ... severe compounding
   error ... edm ... much more stable". Qualitative, but it is a controlled comparison.
6. **Distillation from a bidirectional teacher (CausVid).** "significantly reduced error
   accumulation", with the caveat that a causal teacher's drift is inherited by the student.

### Q2. Does anyone report that noise augmentation hurts teacher-forced single-step PSNR?

**No — and nobody reports the measurement at all.** This matters for us because our headline is
teacher-forced with clean context while training is always noisy.

- GameNGen's ablation (§5.2.2) is explicitly **autoregressive**: "We evaluate both our standard model
  with noise augmentation and the model without added noise ... **auto-regressively**". There is no
  teacher-forced with/without pair anywhere in the paper.
- Nor in MultiGen, DIAMOND, DFoT, Self Forcing or Oasis.
- The theoretical expectation is that it *should* cost something single-step: capacity spent on
  denoising the context is capacity not spent on the target, and Ho et al. (2021) conditioning
  augmentation is understood as trading per-step fidelity for cascade robustness — **VERIFY**, I did
  not open Ho et al. (2021).

**So this is an open, cheap, publishable measurement for us.** Train two runs (max level 0.7 vs 0.0),
report teacher-forced PSNR *and* rollout PSNR-vs-horizon for both, against the copy-last baseline.
Nobody has published that 2x2, and we already have both code paths.

### Q3. What inference-time noise level do papers recommend, and is our 0.0 a deviation?

**Our `--infer-noise 0.0` is a defensible reading of GameNGen and a clear deviation from everyone
else.**

- **GameNGen is the only support for zero**, and it is weak: "During inference, the added noise level
  **can be controlled to maximize quality**, although we find that **even with no added noise the
  results are significantly improved**." The second clause is a comparison against *no noise
  augmentation in training*, not a claim that zero is optimal at inference; the first clause says
  treat it as a tunable. GameNGen never states the level it used.
- **Diffusion Forcing explicitly keeps it nonzero**: "updating the latents using the previous latent
  associated with **slightly 'noisy tokens' for some small noise level 0 < k ≪ K**", justified by the
  fact that the noised frame "**is in-distribution for training**".
- **Oasis schedules it within each frame's denoising**: "injecting noise in the first diffusion
  forward passes to reduce error accumulation, and gradually removing noise in the later passes".
- **MultiGen uses both**: clean context on the conditional branch, noised on the unconditional, and
  guides between them.
- **We never sweep it.** `rollout_eval.py:232` defaults `--infer-noise` to 0.0 and `eval_tf.py:177`
  likewise. An `--infer-noise` sweep costs one rollout pass per value and no training.

Plus our own wrinkle: at 0.0 we hand the model **bucket 0 with exactly-clean context**, a pairing it
never saw in training, because `floor(level / 0.7 * 10)` gives bucket 0 the band `[0, 0.07)` over a
continuous uniform. If a sweep finds the optimum near 0.03-0.05, that is this artefact talking, not
a property of the world model.

### Q4. Cheapest changes to try, ranked by evidence

1. **Sweep `--infer-noise` at rollout: {0.0, 0.02, 0.05, 0.1, 0.2}.** Zero training cost, one flag
   already implemented (`rollout_eval.py:105-106`, `fixed_noise` at line 122). Evidence: Diffusion
   Forcing §3.3 and B.4, Oasis dynamic noising, and GameNGen's own "can be controlled to maximize
   quality". Report PSNR-vs-horizon curves, not a single number — the whole point is that the
   ordering changes with horizon. **Do this first; it may be worth a dB at horizon for free.**
2. **History guidance at rollout, MultiGen-style.** Two forward passes per step: conditional at
   bucket 0 with clean context, unconditional at a high bucket with noised context, combined with a
   CFG weight. No retraining — our model already takes the bucket as an input, which is exactly the
   handle this needs. Evidence: MultiGen §3.4 (a Doom model), DFoT §5 ("even this simple form of HG
   significantly improves generation quality and consistency"). Sweep the weight and the
   unconditional bucket together with item 1.
3. **Reserve a genuine zero bucket in training.** Sample `level = 0` with probability ~0.1 and map it
   to bucket 0, shifting the continuous band to buckets 1-9. Removes the train/test mismatch that
   item 1 would otherwise be confounded by, and makes `--infer-noise 0.0` an honest setting. A
   short fine-tune, not a retrain. Evidence: indirect (DF's "in-distribution for training"
   argument), but it costs almost nothing and it makes our own numbers interpretable.
4. **(Larger, only if 1-3 disappoint.) Per-frame independent noise levels, i.e. real diffusion
   forcing.** Requires one bucket embedding per context frame instead of one per sample — a small
   architecture change in `build_model` plus a change to `noise_augment`'s level shape. Evidence is
   the strongest in the card (DF, Oasis, DFoT, GameFactory), but it is a retrain and it changes the
   thing our GameNGen comparison is anchored on.

Not recommended now: **Self Forcing** (a rollout inside every training step, on a 16 GB card),
**EDM re-parameterisation** (DIAMOND's evidence is real but it would replace the v-prediction
objective our whole results table is built on).

---

## 4. Flags on our own setup

1. **Our 0.7 and the reproduction's 0.7 are different amounts of corruption** (variance-preserving
   vs additive). If we cite "the same value as GameNGen" in the paper, we must say which
   parameterisation, because the repro that also claims "the same values as in the paper" uses the
   other one.
2. **Bucket 0 is never paired with clean context in training** (`diffusion_v.py:133`), but that is
   exactly what `--infer-noise 0.0` feeds it at eval. Fix or measure it before reporting an
   `--infer-noise` sweep.
3. **Validation loss uses noisy context, teacher-forced PSNR uses clean context.** Both are
   defensible, but they are different distributions, so `best_val` (`train_wm.py:296`) is selecting
   checkpoints on a criterion that does not match the reported metric. Worth one line in the paper,
   or worth changing.
4. **We have never run the with/without noise-augmentation ablation.** Per Q2 this is the gap in the
   literature too, so it is cheap novelty rather than housekeeping.

## 5. Could not verify

- **Whether `gameNgen-repro` adds conditioning noise during rollout** (`run_autoregressive.py`) — not
  traced.
- **Masao-Taketani/GameNGen** — not inspected at all, for either card.
- **Ho et al. (2021)** cascaded-diffusion conditioning augmentation — the source GameNGen cites for
  the method, and the arbiter of the VP-vs-additive question. Not opened.
- **GameNGen Figure 7** per-step values — a figure; no numbers extractable.
- **DFoT and Self Forcing ablation tables** — not extracted from HTML.
- **Oasis** noise levels, schedule shape, context length — the project page gives prose only; the
  open-weights repo README was not reachable at the URL tried.
- **Delta-IRIS, TWM, AR-Diffusion (2503.07418), GAIA-2 drift handling, Vista's "dynamic priors"** —
  fetched or identified but not read closely enough to quote.
- **PlayGen's 0.2%-over-1000-frames claim** — it is a mechanics-accuracy metric whose definition I
  did not read; do not compare it to PSNR curves without checking §6.
