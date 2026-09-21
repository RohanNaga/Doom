# Astra on the next-tic runs, Sep 20 2026

**Codex thread id**: `01a0c1d5-f413-72f3-838d-9e28df232189` (model `gpt-6-astra`, reasoning effort high, sandbox `read-only`, cwd `/Users/rohan/Documents/Github/Doom`, 2 turns)
**Prompt**: `.claude/analyses/astra-brief-2026-09-20-nexttic.md`, verbatim. Our own design was deliberately withheld and never shown to it.
**Follow-up round** pushed on three underspecified points: card contention in the budget, the cold-start decision, and a go/no-go rule. The numbers below are the revised ones from round 2 where the two rounds differ.

---

## 1. Its design

### Headline

> **I would put the U-Net on one card and SD 3.5 on the other, keep 32 consecutive tics of stacked context, and prioritize more optimization over encoding all 40M tics.** My forecast is **24.8 dB for the U-Net on held-out episodes from the training arenas**, approximately **3.2 dB above a 21.6 dB persistence reference**. I would not budget this round around reaching 29.43 dB.

Its framing of the research question, since it rejects the "beat 29.43" framing:

> At a fixed five-day compute budget, how much better can next-tic prediction and rollout become when we improve supervision density and conditioning?

A third backbone "would answer less than letting these two reach useful budgets".

### The two cards

| Setting | Card A: SD 1.4 U-Net | Card B: SD 3.5 Medium |
|---|---:|---:|
| Nominal updates | **240,000** | **120,000** |
| Updates if contention costs 20% | **192,000** | **98,000** |
| Global batch = microbatch | **32** | **32** |
| Gradient accumulation | **1** | **1** |
| Learning rate | **5e-5** | **5e-5** |
| Warmup | 2,000 updates | 2,000 updates |
| Schedule after warmup | constant | constant |
| Optimizer | fused AdamW, wd 0, clip 1 | same |
| Precision | bf16 autocast, fp32 master weights | same |
| Gradient checkpointing | **off** (must prove it needs it) | **on** |
| Context | **32 consecutive tics** | **32 consecutive tics** |
| Objective | existing v-prediction, linear betas | same |
| Init | **published SD 1.4 image checkpoint** | **published SD 3.5 Medium image checkpoint** |
| Target presentations (nominal) | 7.68M | 3.84M |

On batch size, explicitly against imitating GameNGen: "I would not accumulate to batch 128 just to imitate GameNGen: that does not manufacture four times the training throughput." It wants a 200-update measurement of peak reserved memory and examples/s under the new conditioning before launch.

### Data: about 5M of the 40M tics, not all of it

Fixed hash permutation chosen independently of results, from the first 2,000 dense episodes already encoding:

- **900 training episodes**, 225 per arena (arenas 2 to 5), about 4.5M tics
- **100 development episodes**, 25 per arena, about 0.5M tics
- reporting episodes come from separate sealed manifests
- both backbones get the same episodes and the same target-selection policy
- shuffle individual valid windows; do not fill a batch with neighbouring windows from one episode

> This deliberately uses about **5M of the 40M recorded tics**. At these presentation budgets, I favor revisiting a diverse 4.5M-window training pool over spending another two days encoding unused targets. Figure 13 supports that choice more than it supports immediately using 40M.

Storage: the two 5M-frame latent sets cost about 267 GB; it reserves 500 GB including caches, bounded checkpoint retention and artifacts.

### Conditioning, which is where it differs from us most

Target frame `x_{t+1}` from observations `x_{t-31:t}` and executed controls `u_{t-31:t}`; `u_t` is the control applied from the last context frame into the target.

- **Replace our single action token with 32 ordered action tokens**, learned relative-position embeddings. "GameNGen explicitly conditions on its action history; this is a cheap, relevant omission in our recipe."
- Each token encodes the **actual button vector**, including weapon-selection controls, not Arnold's requested action id: "Anti-stuck overrides make that distinction real." A two-layer button-vector MLP projected to the backbone's conditioning width. SD 3.5's pooled conditioning takes the newest executed-control embedding.
- **No 4-phase embedding in the main run.** Its rule: "repeat the held buttons on every tic, with no special 'first tic' flag. The engine's next transition depends on its state and current buttons, not on when Arnold intends to reconsider them." If phase is tested later, derive it from the controller's actual decision clock and reset at respawn, "never use global `tic % 4`", because identical successive decisions and 40-tic overrides make action-change boundaries unreliable.
- **Keep channel stacking.** 32 tics is about 0.91 s of history against about 3.66 s at the decision rate, and it names that as a deliberate loss of temporal coverage it accepts: GameNGen's 32-to-64-frame ablation gains only 0.05 dB, and replacing stacking with temporal attention is a much larger experiment.

### Noise augmentation: a mixture, plus real observation CFG

It rejects both keeping our augmentation as-is and removing it entirely, because `noaug` "materially worsened FVD despite improving action fidelity". Proposed distribution, which it labels a compromise and "not an established optimum":

- **50%** exactly clean context, bucket 0
- **40%** variance-preserving corruption with `q ~ U(0, 0.1)`, `c' = sqrt(1-q)·c + sqrt(q)·eps`, using buckets 1 to 9 for that positive interval (note: our max is 0.7, so this is a large narrowing)
- **10%** drop the entire observation context to zero, with an explicit context-present flag
- **action dropout 0**
- **inference context noise 0**

The dropped-context 10% is what buys observation CFG: `v_guided = v_drop + 1.5·(v_context - v_drop)`, with the **same actions in both branches**.

Would **not** do now: per-frame diffusion-forcing noise ("independent context corruption alone would still not reproduce the full diffusion-forcing training setup"), self-forcing, native-flow retraining of SD 3.5, a temporal-backbone rewrite.

### Decoder

Both decoders start from stock and train **only on arenas 2 to 5 training episodes**:

- **100,000 unique frames**, sampled across episodes and all tic phases
- frozen encoder; train decoder and post-quant conv where present
- **batch 32, 6,250 updates** (two passes, 200,000 presentations)
- fused AdamW, **lr 1e-5**, wd 0, clip 1
- warmup 200 updates, then linear decay to **1e-6**
- loss **MSE + 0.1 LPIPS** on the actual 320x240 crop
- validation on 2,048 development frames

In round 1 it added a separate MSE-only continuation of 6,250 updates at lr 5e-6 for the 4-channel decoder (scoring the identical predicted latents through both decoders, LPIPS decoder primary). In round 2 it **cut that continuation from the committed schedule** to make the five days fit. On matching GameNGen's decoder exposure: "Matching GameNGen's nominal decoder exposure — **700k x 2,048 = 1.43 billion presentations** — is not feasible or justified here."

### Sampler

Default **DDIM 10, eta 0, observation CFG 1.5**. On development data compare steps 4, 10, 20, 50 at guidance 1.0 and 1.5 with fixed per-window initial noise. Selection rule: cheapest setting within **0.1 dB PSNR and 0.005 LPIPS** of the best teacher-forced setting, provided rollout LPIPS does not worsen by more than **0.01**. Report the 4-step and 50-step anchors regardless.

Two warnings: "Ten guided steps mean **20 denoiser evaluations**, not ten," and "Next-tic prediction at 35 Hz does not establish 35-FPS generation." Report end-to-end batch-one latency including decoding.

### Schedule (round 2, all GPU work serial on each card)

Assumes launch Sep 21 00:00 EDT, freeze Sep 26 00:00 EDT.

| Hours | EDT | Card A | Card B |
|---|---|---|---|
| 0-24 | Sep 21 00:00 to Sep 22 00:00 | encode 5M tics + eval windows, 4-channel VAE | same selections, 16-channel VAE |
| 24-30 | Sep 22 00:00 to 06:00 | 4-channel decoder tune | 16-channel decoder tune |
| 30-32 | Sep 22 06:00 to 08:00 | U-Net fit check and smoke | SD 3.5 fit check and smoke |
| 32-102 | Sep 22 08:00 to Sep 25 06:00 | U-Net training (with pauses) | SD 3.5 training (with pauses) |
| 102-114 | Sep 25 06:00 to 18:00 | final U-Net evaluation | final SD 3.5 evaluation, or the replacement model |
| 114-120 | Sep 25 18:00 to Sep 26 00:00 | evaluation recovery, artifact verification | same |

Inside each 70-hour denoiser block: 67 hours of optimizer updates, 2 hours validation/save/load, 1 hour for the decision-checkpoint pixel evaluation. "No encoder, decoder-training process, or evaluator shares a card with its running denoiser. Pause the denoiser, release its GPU memory, evaluate, then resume."

Slip rule: encoding and decoder finish first and an overrun delays that card's denoiser start, but **training still stops at hour 102**. Each lost training hour costs 3,600 U-Net or 1,836 SD 3.5 updates at nominal rate (2,880 / ~1,469 under the 20% slowdown). Cut order: decoder continuation, then the public-reproduction comparison, then the expanded sampler sweep, never the primary seen/unseen evaluation. Hours 114-120 are for finishing evaluation, not extending training. It also says to stop unrelated grid jobs before this allocation starts.

Caveat it volunteers: "That is a feasible allocation, not a throughput guarantee. In particular, the 16-channel encoding rate remains unmeasured." The 48-hour encoding reservation is not a measured SD 3.5 rate; the logged 4-channel rate implies about 21 hours for 5M tics. Encoding already done is sunk work that releases budget; do not discard it.

### Cold start (its committed answer)

Start both from the **published image checkpoints**, not from our existing Doom checkpoints. Its quantified cost:

> My point estimate is that **30,000 of the U-Net's updates recover adaptation that an existing Doom checkpoint could largely supply.** That costs 8.3 GPU-hours at 1.0 update/s, 10.4 at 0.8, **12.5% of the original 240k-update allocation**.

Reasons: arenas 6 to 8 are the designated unseen-map test, and it could not verify any existing Doom checkpoint trained exclusively on arenas 2 to 5; creating one now costs the adaptation we are trying to save, and "calling its first 30k updates 'pretraining' does not shorten the schedule." It will **not** add a warm-start row this week. Both denoisers and both decoders start with no prior Doom-map exposure. It notes it is not reusing all 90k historical updates anyway because the prediction interval, the temporal meaning of the stacked channels and the action-conditioning interface all change.

### Go/no-go (one step, one metric, one number)

> **At SD 3.5 update 80,000, kill SD 3.5 if its development PSNR minus the U-Net's development PSNR at update 80,000 is below -0.30 dB.**

Measurement fixed in advance: 512 development windows (128 per training arena), raw 320x240 RGB targets, each model's own training-map-only LPIPS decoder, **live weights** (not EMA), DDIM 10, observation CFG 1.5, clean context, fixed window manifest and per-window seeds. Save and evaluate the U-Net 80k checkpoint when it gets there; decide when SD 3.5 reaches 80k. No earlier quality kill, no second gate, a tie at -0.30 continues. It picked 80k because our own SD 3.5 rows show 40k is too early, and it flags that the rule "could reject a model with better LPIPS".

Replacement on the freed card: **PixArt-alpha for 40,000 updates** from its published image checkpoint, batch 32, lr 5e-5, warmup 2,000, same conditioning and corruption recipe, reusing card A's corpus and decoder. Even at 20% slowdown SD 3.5 clears the gate by about hour 89.5 and PixArt needs 11.0 compute-hours, finishing before hour 102 with about 1.5 hours spare. "That replacement is a bounded third-backbone result, compared with the saved U-Net 40k checkpoint — not a promised new headline winner."

### Implementation prerequisites it found in our code

All five verified against the repo by the supervising session; all five hold.

- `encode_parquet.py:127` writes `chain_id=-1` on non-decision tics. `doom_data.py:275` still asserts four-tic spacing inside a chain, and the endpoint-equality filter at `doom_data.py:281` (`cid[s] == cid[s+context_frames]`) can accept windows whose endpoints are both `-1`, even across a life boundary. Rollout selection has the same issue. **Build a separate per-tic validity contract: consecutive recorded tics and one continuous life throughout the window.**
- `eval_tf.py:139` calls decoded-last-latent versus raw-target PSNR "copy" (`copy_psnr_raw` uses `last_img`, the decoded last latent). Add **raw-last-frame versus raw-target** persistence, independent of the decoder.
- `rollout_eval.py` scores against decoded ground truth. Add raw-frame targets before comparing latent spaces.
- `persistence_stats.py:97` does `np.unique(rng.randint(...))[:anchors]`, which sorts and then truncates, favouring early episode positions. Recompute the final persistence references with uniform sampling without replacement.
- Use per-window RNG streams for sampler comparisons; a shared global seed does not give paired samples when step counts or noise branches consume different random numbers.

Plus: the within-life protocol must explicitly exclude respawn transitions and report the excluded fraction, so it supports claims about within-life simulation, not full death-and-respawn gameplay.

---

## 2. Its evaluation protocol for an honest claim

The core position, verbatim:

> There is **no attainable PSNR threshold on our footage that, by itself, establishes superiority to the original GameNGen**. Its unpublished test distribution and unavailable model prevent that inference. Even 30 dB here would establish a higher cross-paper number, not a controlled win.

Three separately labelled comparisons.

**a. The main released benchmark**
- seen maps: **200 sealed episodes, 50 each on arenas 2 to 5**
- unseen maps: **150 sealed episodes, 50 each on arenas 6 to 8**
- teacher forcing: **2,048 fixed windows per split**, balanced across maps
- report raw RGB PSNR, LPIPS-Alex, HUD PSNR, raw persistence, and each decoder's reconstruction reference
- **paired, map-stratified episode-bootstrap 95% confidence intervals**, not frame-level standard errors
- compare both models at **140k updates** (round 1; round 2 revises SD 3.5 to 120k nominal, so the matched point moves with it), then report the U-Net endpoint and actual GPU-hours separately
- warning: "The existing 'seen' evaluation corpus spans more maps than the new training set; its old label cannot simply carry over."

**b. Temporal evaluation**
- **256 rollouts per primary split, each 256 tics long**
- horizons **1, 4, 16, 32, 64, 128, 256**, stated in tics and seconds
- compare against frozen raw seed-frame persistence
- FVD16/32 at both native tic spacing and every-fourth-tic spacing, identical real/generated preprocessing and sample counts
- for the existing decision-rate models, compare their horizon 64 against the new model's horizon 256: both cover **7.31 seconds**
- action following at decision boundaries on four-tic-spaced observations, keeping the IDM's original temporal interval and a matched real-data reference

Its success criterion for SD 3.5 over our U-Net: **at least +0.5 dB paired teacher-forced PSNR, lower LPIPS, and no material regression in rollout or action metrics**. "That establishes a stronger system on this benchmark, not an architectural cause."

**c. An external open reference**
Evaluate the released Stiegler reproduction checkpoint, with its native decoder and conditioning, against ours on **fresh held-out trajectories from the same single-map setup**, controls translated by button semantics, every system given a small development-only sampler sweep plus its published default. But:

> The public **2.47M-tic corpus is useful for a reproducible train/development/test split**, but the released reproduction checkpoint may already have trained on all of it. It cannot automatically supply an honest held-out test for that checkpoint.

A clean subsequent comparison would train two fresh systems on e.g. a 400/50/50 episode split, which it will not pretend fits this allocation.

Claims it would accept:

> "Our system outperforms the released GameNGen reproduction on a shared held-out benchmark."
> "SD 3.5 outperforms our GameNGen-style U-Net under the specified training budget."
> Neither is equivalent to beating Google's original GameNGen.

---

## 3. Its forecast

- **Central: 24.8 dB** U-Net next-tic on fresh episodes from arenas 2 to 5, subjective plausible interval **23.5 to 26.0 dB**
- margin over the supplied 21.6 dB reference: **+3.2 dB**, about **52% lower geometric-mean frame MSE** (compute pooled MSE reduction separately)
- the arena-specific persistence reference must be remeasured; 21.6 was measured on a different mixture of footage and windows
- **biggest lever: the change to one-tic prediction itself.** "It makes the target easier before any modeling improvement." After that, the next largest source of margin is "learning repeated, dense transitions on four maps with enough target presentations, rather than a decoder-only breakthrough."
- expects only a few tenths of a dB in generated-frame PSNR from the extra MSE decoder continuation at this budget, and expects substantial rollout drift to remain

---

## 4. What it says we have wrong

1. **Figure 13 does not show all curves coinciding below 100k.** From the actual figure (arxiv 2408.14837v2 App. C): "the **1M curve peaks near 20k and is already clearly worse at 50k**. The 5M, 10M, and 70M curves remain much closer. The horizontal axis is logarithmic."
2. **Batch normalisation is a presentation count, not a learning-law equivalence.** Our 90k x 32 = 2.88M presentations, comparable in count to 22.5k x 128. Its proposed U-Net at 7.68M presentations is equivalent in count to **60k GameNGen updates, not 240k**. "Optimizer updates, batch noise, and data correlation still differ."
3. **22.36 to 29.43 dB is not a decoder-only ablation.** Section 5.2.1 states the frozen-decoder context models trained for **200k updates** while the main result uses **700k**, and the evaluation sets also differ. Likewise 29.43 and 32.58 "are different reported evaluations, not interchangeable endpoints of one controlled comparison."
4. **Reconstruction is a reference, not a mathematical ceiling.** `D(E(x))` need not be the best reconstruction attainable by any latent passed through `D`; decoder reconstruction error and prediction error interact and their dB improvements cannot be added. Our "spacing + footage + decoder + remaining training" decomposition is "useful hypothesis generation, but not an identified explanation of the headline gap."
5. **A provenance contradiction to resolve.** The overnight table identifies `raw_stiegler` as a transcode of the public 500-episode corpus, while the Sep 9 log and dataset card identify **1,000 newly recorded episodes containing 4,945,104 tics**. "PNG signatures do not establish JPEG ancestry. Until manifests resolve that contradiction, **23.79 dB belongs to the measured `raw_stiegler` corpus**, not confidently to the public 2.47M-tic dataset — and certainly not to GameNGen's private footage."
6. **It corrected the supervising session on action dropout.** `train_wm.py:449` defaults `--action-dropout` to 0.1, but every launcher in `scripts/spiderman/` passes `--action-dropout 0.0` explicitly (verified: `launch_aligned_spiderman.sh:7`, `launch_sd35.sh:52`, `launch_pixart.sh:10`, `launch_unidiffuser.sh:10`, `launch_cell.sh:60`, `fit_sd35.sh:59`, and others). So no row ever trained with action dropout, its proposal is action dropout 0 with observation-context dropout 0.1, and "Checkpoint arguments are the authoritative record of what ran."

## 5. What it could not do

Read-only sandbox, no GPU. It read the requested notes, result tables and eight code files, checked the GameNGen paper and Figure 13 directly, ran a small CPU check confirming the chain-filter failure and the budget arithmetic, and inspected the launcher overrides. It launched nothing and modified nothing. It could not verify remote dataset provenance, current encoding progress, remote checkpoint arguments, or the new recipe's GPU throughput, and the 16-channel encoding rate is unmeasured; those are launch-time checks.

---

# Astra round 3: copying GameNGen's action conditioning

Same thread `01a0c1d5-f413-72f3-838d-9e28df232189`, turn 3. It was told Rohan's decisions (next tic; U-Net and SD 3.5; public pretrained weights; dense episodes 0-1999 of arenas 2-5, split 0-5999 / 6000-6999 / 7000-7999, unseen = arenas 6-8; context 32 tics; batch 32, fused AdamW lr 5e-5 constant after 2k warmup; context-noise augmentation kept uniform to 0.7 in 10 buckets) and asked to stress-test the action-conditioning copy. Its 0.1 narrowing and 50/40/10 mixture were not taken; it recorded the disagreement and dropped it.

**Its headline finding, verbatim:**

> **The critical finding is that the reproduction's action labels use the opposite row convention from ours.** Its extra action is causal because it records frames *after* executing actions. Our frames are recorded *before* execution. Copying its array slice literally would introduce a one-tic error.

It fetched the reproduction at commit `5564afad59c3f8a9be37ba145163fe98bd742eba`.

## 1. What "copying GameNGen's action conditioning" actually requires

| Detail | GameNGen paper | Open reproduction |
|---|---|---|
| Action embedding | 3.2: learned embedding of each action into one token, replaces text conditioning | `nn.Embedding(max_action_id + 1, 768)`, normal init std 0.02 |
| Passing | action sequence through cross-attention | calls `action_embedding(batch["input_ids"])` and passes it as `encoder_hidden_states` |
| Positional information | **not specified** | **none added**; CLIP's text encoder is bypassed |
| Number of actions | 4.2 explicitly says **64 frames and 64 actions** | **9 context frames, 10 action tokens** |
| Target-producing action | included under the causal notation (predict `o_n` from `a_{<n}`, ending `a_{n-1}`) | included as the final token in the **training** path |
| Conditioning dropout | 4.2: drop frame context with probability 0.1 | zeros context images with probability 0.1, leaves action ids unchanged |
| CFG | 3.3.1: guidance on observations only, weight 1.5; action guidance did not help | duplicates the same action embeddings into both branches, attempts to remove frame context |

Cited sources: paper 3.2 and 4.2 at arxiv.org/html/2408.14837v2; repro `model.py:40`, `dataset.py:34`, `train_text_to_image.py:875`, `data_augmentation.py:21`, `run_inference.py:108`.

Naming trap: the repro's `action_embedding_dim` is misleading — `dataset.py:51` supplies the **maximum action id** and `model.py` adds one, so "that extra arithmetic does not establish a dedicated null-action row."

### The indexing difference (verified in our code by the supervising session)

- **Reproduction**: `ViZDoomPPO/load_model_generate_dataset.py:154` calls `env.step(action)` and then records the resulting screen and that action. Row `i` carries the action **entering** its frame.
- **Ours**: `record_arnold.py:272` stores the frame and action, then calls `make_action`; `record_episodes.py:73` appends the row then calls `env.step(action)`. Row `i` carries the action **leaving** its frame.

Both of ours confirmed by reading the loops. The required contract, where `r` is the target's array row:

```python
context_frames = latents[r-32:r]      # 32 frames
target_frame   = latents[r]
action_ids     = action[r-33:r]       # 33 actions
# action_ids[j] = action[r-33+j], j = 0,...,32
# action_ids[-1] = action[r-1], which produces target_frame
```

The first 32 actions enter the 32 context frames; the final action enters the target. The extra predecessor must exist in the same life or the window is skipped (this costs one window per life at the boundary).

> **Do not use `action[r-32:r+1]`.** Its last token is the action leaving the target frame, potentially selected after observing that frame. Also do not take 32 outgoing context actions and append their final action again: that duplicates the target-producing action.

Our existing scalar convention is already right: `doom_data.py:313` returns `act[start + L - 1]`, which is exactly `action[r-1]` with `r = start + L`. Verified.

Honesty note it insists on: 33 actions matches the repro's incoming-action construction translated to our metadata, and is **not** the paper's stated equal frame/action count. "Describe it precisely rather than calling it an exact replication."

### Reproduction bugs it says not to copy

- `run_inference.py:126` zeros `:BUFFER_SIZE` **after frames are flattened into channels**, so it removes 9 channels, not all 36 context channels.
- Training drops normalised **images before encoding** while inference zeros **latent channels**: two different null contexts.
- `run_autoregressive.py:134` initialises only nine actions although training uses ten, and its loop samples before appending the supplied next action at line 69. "This is not a trustworthy rollout-alignment template."

Copy the paper's semantic rule instead: identical actions in both CFG branches, and remove the complete observation context consistently in training and inference. In our layout that is all `C*32` context channels, never the noisy target.

## 2. What goes wrong in our three wrappers

**Cross-attention is permutation-invariant**: `Attn(Q, PK, PV) = Attn(Q, K, V)` for any common permutation `P` of context tokens, and a tokenwise caption projection does not change that. SD 3.5's joint attention is likewise equivariant to permutations of positionless context tokens. So the repro supplies an action **multiset**, not an ordered history, and the paper does not disclose enough to say what Google did.

Its recommendation, explicitly an addition rather than a replication: a learned **33-position table**, std 0.02 init, added to the action embeddings, positions attached to relative time slots oldest first, and no position on the bucket token.

```python
action_tokens = action_embedder(action_ids) + action_positions[None, :, :]
tokens = torch.cat([action_tokens, bucket_embedder(noise_bucket)[:, None, :]], dim=1)
```

| Path | Required shape and change | Failure modes |
|---|---|---|
| **U-Net** `backbones.py:228` | replace `.unsqueeze(1)` with a `[B,33,768]` sequence; with the bucket token **`[B,34,768]`** | leaving `unsqueeze` produces rank four and crashes. The U-Net has **no bucket token today** — noise enters through `class_labels`, so a new 768-wide bucket table is needed for the 34-token contract |
| **PixArt** `backbones.py:286` | embed at `config.caption_channels` (4096), giving **`[B,34,4096]`** before the existing caption projection; replace the `torch.stack` at line 312 with `cat` | stacking `[B,33,D]` with `[B,D]` crashes; passing already-projected tokens through the caption projection again usually fails dimension checks; an accidental flatten "can silently scramble them if a reshape remains numerically compatible" |
| **SD 3.5** `backbones.py:578` (`conditioning`, stack at 580) | context **`[B,34,4096]`**, pooled separately **`[B,2048]`** | passing the sequence into `pooled_action` yields `[B,33,2048]` and violates the pooled contract; averaging all 34 tokens "mixes noise metadata into action conditioning and loses chronology" |

For the U-Net it prefers the minimal change: keep the existing bucket class embedding **and** append the new bucket token, recording that noise then has two conditioning paths (the repro has only the class-embedding path).

For SD 3.5, keep the pooled mechanism and explicitly select the target-producing action:

```python
pooled = self.pooled_base + self.pooled_action(action_ids[:, -1])
assert pooled.shape == (B, self.transformer.config.pooled_projection_dim)
```

"This is already an action-conditioned replacement for pooled CLIP, so it need not contain actual text. The history stays in context tokens; the pooled vector highlights the current control." Same pooled vector in both observation-CFG branches.

Upstream is not the constraint: "PixArt projects then reshapes with a variable sequence length; SD3 accepts variable-length context and projects it separately from pooled conditioning. **The two-token assumptions are currently in our wrappers.**" Repeat the shape smoke tests under the server's installed diffusers version.

### Additional traps (all verified present)

- **Dropout broadcasting**: `backbones.py:226` (U-Net), 309 (PixArt), 585 (SD 3.5) build a `[B]` mask; applying it to `[B,33]` needs `drop[:, None]`. "Otherwise it generally crashes; at `B=33` it can silently drop temporal columns across the batch." For observation-only CFG, actions should not be dropped at all.
- **PixArt mask**: `encoder_attention_mask=None` at line 320 is correct for 34 valid tokens. If padding appears, use `[B,34]` with 1=keep, 0=discard, including the bucket. "A reversed mask silently removes the useful history."
- **PixArt adaln branch**: `action_inject="adaln"` at line 315 assumes one conditioning vector. Reject sequence conditioning there; these runs use `"token"`.
- **Scalar consumers**: `eval_tf.py:129` does `int(act[i])` and will fail on a sequence. Log the newest action separately.
- **Rollout state**: `rollout_eval.py:99` builds only future scalar actions (`m["action"][s+L-1 : s+L-1+H]`). It needs a rolling 33-action buffer updated **before** the corresponding prediction. Keep saved IDM labels scalar `[N,H]`.
- **Smoke tests validate the old interface**: `train_wm.py:43` and `verify_sd35.py:94` generate scalar actions and must be changed to exercise sequences.
- **Wrong action meaning**: `record_arnold.py:89` can execute an anti-stuck override while keeping the requested action id. Use actual `buttons` for alignment diagnostics. "Correct timing cannot repair incorrect control labels."

**Silent-degradation set** (these do not crash): missing positions, incorrect control labels, wrong-but-valid action slices, stale rollout buffers.

## 3. The alignment test list

Proposed acceptance thresholds, not measurements. "A weak or insufficient signal is 'inconclusive,' not permission to shift labels until a test passes." Fixed validation manifest of 100 episodes, 25 per arena. Read button names from the corpus's `buttons.json` (`docs/cards/arnold/buttons.json`, present), never another recorder's ordering. Every physical test excludes episode boundaries, death/respawn boundaries, non-unit tic gaps and nonfinite poses.

### A. Exact raw-to-latent metadata and window audit (mandatory)

On **10,000 valid windows**, independently locate rows by recorded `tic` in the raw parquet. For target row `r`, verify context tics equal raw tics at `[r-32:r]`; target tic equals raw tic at `r`; all 33 ids equal raw `action[r-33:r]` in order; the newest id equals raw `action[r-1]`; and metadata `buttons`, `angle`, `pos_x`, `pos_y` equal their raw counterparts after dtype matching. **Pass: zero mismatches.** Deliberately include action-change windows, because "unchanged held actions can conceal a shifted slice." The encoding copy is `encode_parquet.py:139`.

Extra certainty: instrument the extraction helper to return source-row indices in an audit mode; they must be exactly `r-33,...,r-1`. "This catches shifts even when all 33 action IDs happen to be equal."

### B. Yaw lag test concentrated on control changes (primary physical test)

`q_i = 1[TURN_LEFT_i] - 1[TURN_RIGHT_i]`, `d_i = wrap_[-180,180)(angle[i+1] - angle[i])`. Treat yaw as zero when `|d_i| <= 0.05 deg`, use its sign when `|d_i| >= 0.25 deg`, exclude the band between. Verify the angle orientation from the engine convention once; "do not choose its sign independently for each lag."

Select turn-change boundaries `j` with stable commands four tics before and after, and score **both `i = j-1` and `i = j`** to separate the last old-control transition from the first new-control one. For offsets `l` in {-1, 0, +1} compute balanced sign accuracy `S_l = BalancedAccuracy(q_{i+l}, sign(d_i))`. Require **1,000 qualifying boundaries across at least 20 episodes**, all three turn classes represented.

**Pass**: `S_0 >= 0.95`; `S_0 - max(S_{-1}, S_{+1}) >= 0.20`; paired episode-bootstrap 95% CI for each advantage excludes zero.

Run first on raw buttons, then on the control the loader's **newest conditioning token** actually emits. With that stream `C[i]` (last context frame at row `i`) scored against `d_i`:

| Actual emitted control | Expected best offset |
|---|---:|
| correct, `C[i] = q[i]` | **0** |
| one tic stale, `C[i] = q[i-1]` | **+1** |
| one tic future, `C[i] = q[i+1]` | **-1** |

"A strong peak at +/-1 diagnoses direction. Low scores at every offset suggest weak pose signals, wrong button mapping, overrides, or discontinuities — not a diagnosed off-by-one."

### C. Translation onset from rest (independent cross-check)

Explicit warning first: "Do **not** require ordinary velocity signs to respond immediately to changed buttons: Doom momentum can make a correctly aligned action lose that test."

Select raw-button events `j` with four preceding tics of no translation command and displacement below 0.02 map units per tic; one pure translation direction started and held eight tics; no turning, firing, health change or death in the window; and at least 8 map units travelled over the next eight tics (excluding blocked starts). With angle in radians, project `position[i+1] - position[i]`:

`v_f = dx cos(theta_i) + dy sin(theta_i)`, `v_l = -dx sin(theta_i) + dy cos(theta_i)`

Find the first transition `k` whose displacement exceeds 0.05 map units and independently locate the control-change row `j_C` in the emitted conditioning stream. With **200 qualifying events across at least 20 episodes**: at least **90%** have `k - j_C == 0`, and at least **95%** of onset displacements project positively onto the commanded direction. Stale conditioning shows onset one tic before its command (`-1`), future conditioning one tic after (`+1`). Report inconclusive if unblocked starts are too few.

### D. Teacher-forced / rollout buffer parity (mandatory, no diffusion needed)

**256 real windows**, 64 future tics each, denoiser replaced by a stub that records its conditioning arguments. At rollout step `h` with initial target row `r`:

```python
received_actions == raw_action[r+h-33:r+h]
received_actions[-1] == raw_action[r+h-1]
```

Compare with the teacher-forced loader on the same row. **Pass: zero mismatches across all 16,384 calls.** This catches appending an action after the prediction, shifting by four rows instead of one, and target-action duplication. Stratify failures by distance to actual action changes, "not global `tic % 4`."

### E. Negative controls and token-order checks

- Inject `C[i] = q[i-1]` and `C[i] = q[i+1]` into the same real-window audit; the yaw test must identify **+1** and **-1** respectively at the same thresholds.
- Shift target rows by one in the extraction test; the exact audit must reject both shifts.
- Wrapper tests at batches **1, 2 and 33**; shapes exactly `[B,34,D]`, SD3 pooled exactly `[B,2048]`.
- **Permute older action ids while keeping positions and the newest action fixed**, otherwise SD 3.5's pooled newest-action path "can falsely make an unordered-history implementation appear order-sensitive."
- Check nonzero gradients reach the position table and the action embeddings, and that CFG branches keep identical action tokens and pooled vectors.

## What it ran and could not do

It ran a NumPy sanity check: positionless attention was permutation-invariant to about 3e-15, fixed positional embeddings broke that invariance, and synthetic four-tic holds produced exactly the expected lag peaks (0 correct, +1 stale, -1 future). It read the paper, the pinned reproduction source, our wrappers and both recorder loops. It has not run the acceptance tests on the recordings (not local) and local PyTorch/diffusers are unavailable.

> The worker's diff should first be judged against the explicit `action[r-33:r]` contract.

**Thread is still open for the implementation-diff review.**

---

# Astra round 4: review of the implementation diff

Same thread, turn 4. Reviewed `f083452..0e6299f` in `/Users/rohan/Documents/Github/Doom/.claude/worktrees/agent-a14a258ed0ea99004` plus that worktree's uncommitted trainer and launcher fixes. Evaluators skipped by instruction.

> **The 32-control training and horizon indexing are correct. The alignment checker is not yet a reliable launch gate.**

## Confirmed defects (all verified against the code by the supervising session)

1. **`check_action_alignment.py:85` scores each shift on a different subset.** `keep = want != 0` drops rows where the *shifted* control is zero, so shifts -1, 0 and +1 see different rows. Reproduced: 4-tic holds cycling no-turn/left/no-turn/right with perfect physics score **1.000 on all three shifts** (rows 23/47/24). The `boundaries` union at line 66 defeats the pure left/right version of this trap but not the version with no-turn runs between turns, and Arnold's action set produces those constantly. Fix: score all three shifts on one identical row set, keep no-turn as a third class, report balanced accuracy.
2. **`check_action_alignment.py:183` exits 0 on inconclusive and on no evidence**, because `bad` only catches `best not in (None, "0")`. Worse than Astra reported, and the supervising session's own finding: **injecting the exact stale off-by-one it exists to catch (`C[i] = buttons[i-1]`) gives shift 0 = 1.000 on 24 rows, shift +1 = 1.000 on 48 rows, verdict "inconclusive", `best = "0"`, exit code 0.** The gate passes misaligned data. Also `:135` sets `second = -inf` when only one shift has rows, so one row at 0% accuracy is verdicted "aligned at shift 0" with infinite margin (reproduced). Fix: require all three shifts present, a minimum row count, an absolute accuracy floor and the margin; distinct nonzero exit for inconclusive.
3. **`check_action_alignment.py:146` scores across deaths and tic gaps.** `from_parquet` reads only `buttons, angle, pos_x, pos_y`; `from_latents` likewise ignores `tic` and `deaths`, so respawn teleports enter the score. Confirmed from source, effect on the recordings unmeasured. Fix: pass continuity metadata in and require the whole neighbourhood of all three shifts to stay inside one life at unit spacing.
4. **`train_wm.py:52` fit check crashes with `--phase-conditioning`.** `SyntheticWindows.__getitem__` always returns three tensors, so `phase=None` reaches `backbones.phase_vec:146`, which raises deliberately. Does not affect the settled phase-disabled launch. Fix: append a synthetic phase tensor only when phase conditioning is on, preserving the draw order.
5. **`train_wm.py:33` `FIT_CHECK_CONTROL_BITS = 15` disagrees with `docs/cards/arnold/buttons.json`, which lists 19 buttons** (9 movement/attack plus SELECT_WEAPON0-9). Training reads its width from the corpus so training controls are not truncated, but the default fit check builds a differently shaped input MLP. Fix: read the corpus width, or pass a verified `--control-bits`.

## Inferred, not executed

History mode leaves the old scalar action tables allocated and trainable but unused (`backbones.py:354`, and SD 3.5's `action_embedder`/`pooled_action` around 729-731) and DDP unused-parameter detection is off, so a multi-process history run should fail on the second iteration. Harmless for the planned one-process-per-card runs. Astra could not run DDP locally.

## Checked and found correct (spot-verified)

- **Training alignment**: `doom_data.py:642` returns `controls[start:start+L]`, exactly `buttons[r-32:r]` with `r = start+L`, newest `buttons[r-1]`.
- **Horizon alignment**: `doom_data.py:652` advances by `k`, so target `r+k` gets `buttons[r+k-32:r+k]`. No target-row control, no duplicated newest control.
- **Window validity**: `doom_data.py:479-485` requires every adjacent pair across context *and* the full horizon to have unit tic spacing, unchanged `deaths` and unchanged `map_id`.
- **Trainer tensors**: `SeededCorruption` preserves the control tensor, validation at `train_wm.py:480` moves it without scalar conversion, synthetic history samples are `[L, bits]`, and `load_init_weights` carries no scalar-action assumption (public pretrained warm starts bypass it).
- **Token layout, phase disabled**: U-Net `[B,32,768]` with the bucket left in `class_labels`; PixArt `[B,33,caption_channels]` controls then bucket; UniDiffuser `[B,33,text_dim]`; SD 3.5 `[B,33,4096]` with pooled `[B,2048]` from control token 31 only. Learned positions added before the backbone projections. PixArt's `encoder_attention_mask=None` valid since all tokens are present.
- **Default path**: no new parameters and no extra RNG draws when history and phase are off (source comparison; no PyTorch parity run).
- **Splits**: `release/dense_split.json` gives arenas train `0:6000` / val `6000:7000` / test `7000:8000` and the next-tic prefix train `0:2000`, val `6000:6100`, test `7000:7100`, unseen `arenas_678` `0:60`, which resolves to 500/25/25 episodes per training arena and 20 per unseen arena under `map_ids[episode_id % len(map_ids)]`.
- **Recipe**: `launch_nexttic.sh:127-129` matches lr 5e-5, warmup 2000 then constant, clip 1.0, noise aug max 0.7 in 10 buckets, action dropout 0, seed 0, EMA 0.9999 every 8 updates, public pretrained warm starts. `MB` now defaults to 32 and the launcher refuses `MB * cards != 32` unless `ALLOW_ACCUM=1`.

## Wanted on real recordings before launch

Audit 10,000 windows including control changes and anti-stuck overrides (raw vs encoded buttons identical, newest control from `r-1`, unit tic spacing and unchanged life on every context and horizon transition, zero mismatches); the corrected yaw test over validation ids 6000-6099 needing 1,000 control-change boundaries, balanced accuracy >= 0.95 and a >= 0.20 lead over both neighbours with bootstrap CIs excluding zero; stale and future injections that must select shift +1 and -1 respectively; and verification of the real button width and order, episode-map counts, and a short real-batch train *and* validation smoke per backbone confirming finite losses and nonzero gradients into the history MLP and position table.

Astra ran 57 launcher and alignment fixtures, reproduced the scorer failures in NumPy, and checked 100 randomised validity cases and 64 horizon slices through the real dataset methods with a NumPy shim. No files edited. Thread open for the evaluator pass.
