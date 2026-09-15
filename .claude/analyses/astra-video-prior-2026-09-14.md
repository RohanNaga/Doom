# Astra (gpt-6-astra) — video-pretrained row for DoomDiT

- **Thread id**: `01a0a160-b1ab-7f93-8bc4-86ddd74a0d07`
- **Date**: 2026-09-14
- **Brief**: `.claude/analyses/astra-brief-2026-09-14-video-prior.md` (round 5 of the pairing)
- **Astra's own longer working file**: `.claude/analyses/astra-video-row-2026-09-14.md`
- **Sandbox**: workspace-write, network access on. No queues or training code changed.

Below is Astra's response verbatim across two turns.

---

## Turn 1 — the design

**I would add a gated Open-Sora 1.1 row, keep PixArt and seed 1, and leave Cosmos as a follow-up.** Open-Sora gives us pretrained temporal attention without re-encoding the corpus or changing action cadence. That makes it the most informative video-pretrained experiment we can plausibly finish.

First, I need to correct the premise attributed to our earlier review: **we have not established that pretraining causes the gap**, or ruled out architecture and optimization effects. Nor is video initialization documented for every strong transformer world model. The defensible motivation is that video pretraining is prominent among successful systems and absent from our current rows.

**1. My proposed design**

Use [`hpcai-tech/OpenSora-STDiT-v2-stage3`](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage3), the released Open-Sora **1.1** checkpoint—not 1.2.

The [official report](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_02.md) states:

> "9.7M videos + 2.6M images" for pretraining; "560k videos + 1.6M images" for fine-tuning.

It starts from `Pixart-alpha-1024` and reports:

> "approximately 9 days on 64 H800 GPUs."

The project calls it "700M"; Hugging Face reports **768,271,300 stored parameters**. Both checkpoint metadata and repository specify Apache-2.0.

| Choice | Proposed value |
|---|---|
| Latents | Existing SD cache, `B×4×9×32×40`; same decoder |
| Context | **8 decision frames + 1 target**, 8.75 fps |
| Tokenization | Native `(1,2,2)` patches: **2,880 tokens** |
| Actions | Zero-initialized `Embedding(30,1152)`, added only to target-frame tokens |
| Text conditioning | Native null-caption embedding; no runtime T5 |
| Context handling | Native clean-prefix mask; clamp history throughout sampling |
| Attention | Retain pretrained spatial and temporal attention |
| Objective | Native **epsilon DDPM**, learned-range variance; both losses on target only |
| Optimizer | AdamW, **lr `2e-5`, warmup `500`, wd `0`, clip `1`** |
| Batch | Microbatch **1**, accumulation **8**, effective batch **8** |
| Precision | bf16 autocast, fp32 master/Adam, gradient checkpointing |
| Duration | **10k updates or 36 training card-hours**, whichever comes first |
| Sampling | **DDIM 50**, eta 0, no CFG |

These are **proposed configuration values**, not existing CLI flags.

Why only add the latest action to target tokens? It preserves the action information available to our present models. Historical action embeddings would introduce another information advantage. The table adds **34,560 parameters**; NanoWM's "zero" means no additional per-block injection machinery, not a parameter-free action encoder.

There is no need for causal attention or diffusion forcing here: the sequence contains known history and exactly one noisy future frame. Native Open-Sora already supports timestep-zero conditioning through `x_mask`. Keeping this mechanism avoids a causal-attention conversion and distillation project.

The [native stage-3 config](https://github.com/hpcaitech/Open-Sora/blob/v1.1.0/configs/opensora-v1-1/train/stage3.py) actually specifies **`sd-vae-ft-ema`**; its encoder is compatible with our SD1.x cache. The [VAE wrapper](https://github.com/hpcaitech/Open-Sora/blob/v1.1.0/opensora/models/vae/vae.py) explicitly uses spatial compression `(1,8,8)` and scale **0.18215**.

**Memory and throughput are not verified.** My planning estimate is **20–32 GiB**, **0.08–0.20 optimizer updates/s**, giving approximately **14–35 hours for 10k updates**. Static fp32 parameters, gradients and Adam moments alone occupy about **12.3 GB**. Shared-card load could invalidate that estimate.

Crucially, reserve evaluation time separately: this model processes nine frame grids per prediction. I propose **up to 36 training hours plus 36 evaluation hours**, subject to measured feasibility—not an assumed allocation.

**2. Engineering and scheduling**

I would spend **two to three engineering days**, while the queued runs proceed:

- **Day 1:** Load pinned STDiT2 weights; preserve native positions, metadata and masks. The released code still calls xformers cross-attention despite its optional import. Either install compatible xformers or implement a tested SDPA equivalent, preserving sample boundaries.
- **Day 2:** Add target-action injection, target-only loss, sampler and evaluator adapter. Verify zero-action parity, action/temporal gradients, checkpoint roundtrip and fixed-noise overfitting on real clips.
- **First available A6000:** Run a one-hour gate covering actual optimizer steps **and a 64-step rollout**. Extrapolate the complete 256×64 evaluation, including decoding. A fit check alone is insufficient.

Keep **033 PixArt and 032 seed 1**. Start the video row after one finishes and its evaluation releases the card, around Sep 17–18. PixArt is central to the current paper; seed 1 remains valuable and should finish before this prototype is ready anyway.

At 10k updates, this proposal sees **80k target frames**, not 2.9M. If it remains clearly undertrained at the time cap, describe it as a pilot.

**Sep 22 fallback:** if the trainer and end-to-end evaluator are not operational, stop this extension for the submission. Finish the existing rows and writing. Do not switch to a second video checkpoint at that point.

**3. Runner-up and why Cosmos loses**

My runner-up is **Wan2.1-T2V-1.3B**: standard diffusers support, Apache-2.0, and a contemporary video prior. It loses because we must build a new encoding/cache path, temporal boundary handling, flow objective and prefix-conditioning adaptation. Budget **3–5 engineering days**, with actual training speed unverified.

| Candidate | Verified facts and decision |
|---|---|
| **Wan2.1-T2V-1.3B** | [Released config](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers): 30 layers, 16 latent channels, `(1,2,2)` patches; Wan VAE temporal factor 4. Advertised **8.19 GB** is inference, not training. |
| **Cosmos-Predict2.5-2B** | **My choice if Cosmos is required.** Official [action-training guide](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/post-training_video2world_action.md) supplies a **single-process**, rectified-flow experiment named `ac_reason_embeddings_rectified_flow_2b_256_320`. This does **not** establish 48 GB feasibility. Full fp32 Adam state is roughly 32 GB before activations; LoRA may be necessary. Exact selected checkpoint access/config was **not verified**: model-card access returned 401. |
| **Cosmos-Predict2-2B** | Official [Bridge action recipe](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/post-training_video2world_action.md) launches **two processes**. Its sample predicts 12 frames from an image plus 12 actions. The action module flattens the action sequence and injects learned MLP outputs into timestep/adaLN conditioning. Official **32.54 GB** requirement is inference. |
| **Cosmos-Predict1-7B** | Decline. Approximately **112 GB** of conventional fp32 Adam training state before activations. Also, **CV8x8x8 is 8× temporal compression**, not 4×; the evidence card is incorrect there. |
| **SkyReels-V2-I2V-1.3B** | Verified Matrix-Game parent and native image conditioning. Additional image-conditioning integration and **Skywork license**, rather than Wan's Apache-2.0. Full license text **not verified**. DF adds another training/sampling protocol; defer. |
| **CogVideoX-2B** | [Official card](https://huggingface.co/THUDM/CogVideoX-2b): **47 GB LoRA, batch 1; 62 GB full fine-tuning, batch 1**, at native settings. It specifies **720×480**, including fine-tuning. Apache-2.0, but resolution adaptation adds work. |

Cosmos source code is **Apache-2.0**; model weights use the **NVIDIA Open Model License**, according to their official repositories.

I verified these imports locally in **diffusers 0.40.0**:

- `WanTransformer3DModel`, `WanPipeline`, `AutoencoderKLWan`
- `CosmosTransformer3DModel`, `CosmosVideoToWorldPipeline`, `Cosmos2VideoToWorldPipeline`
- `Cosmos2_5_PredictBasePipeline`
- `SkyReelsV2DiffusionForcingPipeline`
- `CogVideoXTransformer3DModel`, `CogVideoXPipeline`

Predict2 and Predict2.5 pipelines use `AutoencoderKLWan`; native Predict2's video tokenizer confirms **16 channels, temporal 4, spatial 8**. Preserve each checkpoint's normalization—shared VAE architecture does not establish numerically interchangeable latents. **Successful imports are not verified training pipelines.**

**4. Raw tics versus decision frames**

For a Wan-style temporal factor of four, the aligned layout is:

> **One initial state + K four-tic action intervals → `1 + 4K` raw frames → `1 + K` latent frames.**

Score each decoded interval endpoint against the corresponding existing decision-frame target. Start at a **verified decision boundary**, not an arbitrary tic-modulo boundary.

Encoding **decision frames** instead makes one compressed step span **four actions**. Supplying those future actions together changes the task to action-plan-conditioned chunk prediction.

Even raw-tic alignment needs tests:

- Perturb future pixels and verify encoded prefixes stay unchanged.
- Check decoded endpoint timing and cache/reset behavior.
- Never cross death/respawn chains.
- Count the intermediate raw frames as additional training exposure.

That last point matters: raw-tic video training uses images the paired models never see.

**5. What survives review**

A VAE swap **breaks the controlled-ablation interpretation for that row**, but does not invalidate the existing controlled comparison. Pixel-space system comparisons remain honest if they share targets, cadence, available action information, test episodes and evaluation procedures.

Per-row reconstruction references help interpret errors; they do not restore control or justify subtracting "ceiling PSNR." Keep the current SD decoder's fine-tuning visible. Feed generated pixels through the existing IDM preprocessing rather than substituting video latents.

Report **adaptation GPU-hours, target exposures, preprocessing/tuning cost, NFE and generation latency**. Do not compare latent losses across epsilon, VP-velocity, flow and EDM models.

My proposed wording:

> **Under a shared SD-latent next-frame adaptation recipe and limited compute, ImageNet-initialized DiT-XL/2 shows no advantage over the SD-initialized U-Net. We separately evaluate a video-pretrained temporal transformer under its own adaptation recipe to assess an accessible alternative system.**

An Open-Sora win would support that alternative system's usefulness. It would not establish that video pretraining alone closes the gap.

Full sources, gates and estimates are recorded in the [review](/Users/rohan/Documents/Github/Doom/.claude/analyses/astra-video-row-2026-09-14.md). No queues or training code changed.

---

## Turn 2 — pushback on the diffusers claim, token arithmetic, abandonment gates

It was a **live import in the PERSEVE conda environment**, not a pip-index or source-tree check. Your `python3` resolves to a different environment. I reran this exact command successfully:

```bash
/opt/miniconda3/envs/PERSEVE/bin/python - <<'PY'
import sys
import diffusers
print('interpreter:', sys.executable)
print('diffusers:', diffusers.__version__)
print('module:', diffusers.__file__)
for name in (
    'WanTransformer3DModel', 'WanPipeline', 'AutoencoderKLWan',
    'CosmosTransformer3DModel', 'CosmosVideoToWorldPipeline',
    'Cosmos2VideoToWorldPipeline', 'Cosmos2_5_PredictBasePipeline',
    'SkyReelsV2DiffusionForcingPipeline',
    'CogVideoXTransformer3DModel', 'CogVideoXPipeline',
):
    cls = getattr(diffusers, name)
    print(name + ': OK (' + cls.__module__ + ')')
PY
```

Actual stdout, exit code **0**:

```text
interpreter: /opt/miniconda3/envs/PERSEVE/bin/python
diffusers: 0.40.0
module: /opt/miniconda3/envs/PERSEVE/lib/python3.13/site-packages/diffusers/__init__.py
WanTransformer3DModel: OK (diffusers.models.transformers.transformer_wan)
WanPipeline: OK (diffusers.pipelines.wan.pipeline_wan)
AutoencoderKLWan: OK (diffusers.models.autoencoders.autoencoder_kl_wan)
CosmosTransformer3DModel: OK (diffusers.models.transformers.transformer_cosmos)
CosmosVideoToWorldPipeline: OK (diffusers.pipelines.cosmos.pipeline_cosmos_video2world)
Cosmos2VideoToWorldPipeline: OK (diffusers.pipelines.cosmos.pipeline_cosmos2_video2world)
Cosmos2_5_PredictBasePipeline: OK (diffusers.pipelines.cosmos.pipeline_cosmos2_5_predict)
SkyReelsV2DiffusionForcingPipeline: OK (diffusers.pipelines.skyreels_v2.pipeline_skyreels_v2_diffusion_forcing)
CogVideoXTransformer3DModel: OK (diffusers.models.transformers.cogvideox_transformer_3d)
CogVideoXPipeline: OK (diffusers.pipelines.cogvideo.pipeline_cogvideox)
```

**Tokens:** `(8 context + 1 target) × (32/2) × (40/2) = 9 × 16 × 20 = 2,880` video tokens per sample.

**Memory/throughput:** still estimates, **not measured** on an A6000.

- **Day-1 abandonment:** unresolved pretrained-key/shape mismatch, or failure to reproduce the unmodified checkpoint's output with zero action injection after any attention replacement.
- **A6000 abandonment:** OOM at checkpointed microbatch 1 after optimizer-state allocation, or measured throughput below **0.0772 updates/s** (`10,000 / 36 hours`), or projected full evaluation exceeding its **36-hour** reserve.

---

# Claims Astra marked verified, with the source it quoted

| Claim | Source Astra quoted |
|---|---|
| Open-Sora 1.1 pretraining scale: "9.7M videos + 2.6M images"; fine-tune "560k videos + 1.6M images" | `github.com/hpcaitech/Open-Sora/blob/main/docs/report_02.md` |
| Open-Sora 1.1 starts from `Pixart-alpha-1024`, trained "approximately 9 days on 64 H800 GPUs" | same report |
| Checkpoint size 768,271,300 stored parameters (project calls it "700M") | HF card `hpcai-tech/OpenSora-STDiT-v2-stage3` |
| Open-Sora 1.1 licence Apache-2.0 (checkpoint metadata and repo) | HF checkpoint metadata + repo |
| Open-Sora 1.1 stage-3 config specifies `sd-vae-ft-ema` (SD1.x-compatible encoder) | `configs/opensora-v1-1/train/stage3.py` at tag v1.1.0 |
| VAE wrapper uses spatial compression (1,8,8) and scale 0.18215 | `opensora/models/vae/vae.py` at tag v1.1.0 |
| Open-Sora released code still calls xformers cross-attention despite optional import | Open-Sora v1.1.0 source |
| Native Open-Sora supports timestep-zero clean-prefix conditioning via `x_mask` | Open-Sora v1.1.0 source |
| Wan2.1-T2V-1.3B: 30 layers, 16 latent channels, (1,2,2) patches, Wan VAE temporal factor 4; the 8.19 GB figure is inference not training | HF `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` released config |
| Cosmos-Predict2.5 ships a single-process rectified-flow action post-training experiment `ac_reason_embeddings_rectified_flow_2b_256_320` | `nvidia-cosmos/cosmos-predict2.5/docs/post-training_video2world_action.md` |
| Cosmos-Predict2 Bridge action recipe launches two processes; predicts 12 frames from image + 12 actions; action MLP injects into timestep/adaLN conditioning; 32.54 GB figure is inference | `nvidia-cosmos/cosmos-predict2/documentations/post-training_video2world_action.md` |
| Cosmos CV8x8x8 is 8x temporal compression, not 4x (corrects `lit/` evidence card) | Cosmos tokenizer docs |
| Cosmos source Apache-2.0; Cosmos weights under NVIDIA Open Model License | official NVIDIA Cosmos repos |
| CogVideoX-2B: 47 GB LoRA batch 1, 62 GB full fine-tune batch 1, native 720x480; Apache-2.0 | HF `THUDM/CogVideoX-2b` card |
| SkyReels-V2-I2V-1.3B is Matrix-Game 2.0's parent, has native image conditioning, ships under Skywork licence not Apache-2.0 | SkyReels-V2 repo/card |
| diffusers 0.40.0 exposes WanTransformer3DModel, WanPipeline, AutoencoderKLWan, CosmosTransformer3DModel, CosmosVideoToWorldPipeline, Cosmos2VideoToWorldPipeline, Cosmos2_5_PredictBasePipeline, SkyReelsV2DiffusionForcingPipeline, CogVideoXTransformer3DModel, CogVideoXPipeline | live import, `/opt/miniconda3/envs/PERSEVE/bin/python`, stdout above (reproduced independently by the supervising session) |
| Predict2 / Predict2.5 pipelines use `AutoencoderKLWan`; Predict2 video tokenizer is 16 channels, temporal 4, spatial 8 | diffusers 0.40.0 source + Cosmos native config |

## Claims Astra explicitly marked NOT verified

- Memory footprint 20-32 GiB on an A6000 — planning estimate only, never measured.
- Throughput 0.08-0.20 optimizer updates/s and the resulting 14-35 h for 10k updates — estimate only.
- The 36 h evaluation reserve — proposed, not measured.
- Cosmos-Predict2.5-2B exact checkpoint access and config — HF model-card access returned HTTP 401.
- Whether Cosmos-Predict2.5-2B action post-training fits in 48 GB — the single-process recipe does not establish it; LoRA may be required.
- SkyReels Skywork licence full text.
- Wan2.1-T2V-1.3B actual training speed under our recipe.
- That any of the diffusers imports correspond to working *training* pipelines ("Successful imports are not verified training pipelines").
- The premise that pretraining exposure causes the 030/031 gap — Astra explicitly retracts this as unestablished, and denies that video initialization is documented for *every* strong transformer world model.

## Turn 3 — round 5b

### Reply 3a — the five points

**You changed my mind on the objective and conditional queue priority. I still pick Open-Sora, with SkyReels DF now the runner-up.** My reason is the probability of obtaining an interpretable, fully evaluated result before the deadline—not a belief that Open-Sora has the strongest prior.

**1. Prior strength versus control**

If forced to forecast, conditional on successful adaptation:

| Model | Subjective central guess: PSNR / LPIPS |
|---|---:|
| Open-Sora 1.1 | **20.3 dB / 0.33** |
| SkyReels DF 1.3B | **20.7 dB / 0.30** |

These are **judgment calls, not literature-derived predictions**; uncertainty is at least ±1 dB and ±0.05 LPIPS. I expect the Wan lineage to have a better chance of clearing the cited **20.5 / 0.324** reference, especially on LPIPS, but that reference is an intermediate U-Net checkpoint. Stronger perceptual priors do not reliably translate into higher paired PSNR.

The shared PixArt lineage is substantive, but it does **not isolate video pretraining exactly**: our PixArt is 512, Open-Sora's ancestor is PixArt-alpha-1024, and temporal blocks, masking and conditioning differ. It makes the experiment easier to interpret; it does not settle the scientific question.

I also disagree that an intermediate result says little: "this accessible video-pretrained package did not overcome the U-Net under this adaptation budget" is useful, provided we establish adequate adaptation. An obviously undertrained row would say little.

For your stronger-prior option, I verified the [SkyReels DF 1.3B diffusers config](https://huggingface.co/Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers/blob/main/transformer/config.json): **30 layers, 16 input/output channels, `(1,2,2)` patches**, `inject_sample_info=true`, `num_frame_per_block=1`. It uses **`SkyReelsV2Transformer3DModel`**, so preserve its sample-information and per-frame timestep interface.

In the current [Matrix-Game-2 release](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2), I found inference pipelines and model modules, **no world-model training entry point or training configuration**. That removes it from my ready-to-run ranking.

**2. Objective: you are right**

Use our **shared VP v-loss**, retain the first four channels **after unpatchification**, and disable variance loss.

There is no structural entanglement with learned variance that requires retaining native epsilon training. My earlier recommendation overvalued native-objective fidelity while unnecessarily weakening this comparison. I have **no defensible numerical estimate** of a penalty from epsilon-to-v adaptation.

The PixArt overfit result establishes that this conversion can learn; it does not establish held-out performance. Nevertheless, it is sufficient evidence to choose the already-tested shared objective.

**3. Context: eight was a budget choice**

For one target following \(L\) context frames:

| Context | Video tokens | Approximate projection/MLP cost relative to L8 |
|---|---:|---:|
| L8 | `9 × 16 × 20 = 2,880` | 1× |
| L16 | `17 × 16 × 20 = 5,440` | 1.89× |
| L32 | `33 × 16 × 20 = 10,560` | 3.67× |

**Yes, I would run L16 if timing permits.** Choose L16 versus L8 from resource measurements before training, not test performance.

My unverified L8 throughput estimate scales to approximately **0.042–0.106 updates/s at L16**, implying **26–66 hours for 10k updates**. Therefore L16 is conditional, not a free improvement.

Your Wan L16 has the same **5,440 tokens** when raw-tic compression yields one latent per decision. Temporal compression gives no token-count discount relative to SD decision-frame latents. Additionally, full attention forms **29.6M scores/head**, versus approximately **1.83M** for Open-Sora's factorized spatial-plus-temporal attention at this shape—a **16.1× score-count difference**, not a 16.1× runtime prediction.

**4. Scheduling: a gate-passed video row now outranks the unstarted seed**

Given the expanded question, my priority is:

**Original-pair evaluation → PixArt → gate-passed video row → second seed.**

The video row tests a substantially different explanation; the second seed measures repeatability of an already persistent gap. I would therefore recommend **GPU 3 for video after the U-Net**, and **seed 1 behind PixArt on GPU 2**, provided the prototype is ready for its one-hour GPU gate when GPU 3 releases.

If it is not ready, let seed 1 start. Do not hold the card idle, and do not interrupt an underway seed merely because the prototype becomes ready later. This is a revised recommendation; I have not changed the queue.

**5. Cosmos, in three sentences**

Cosmos-Predict2's verified **32.54 GB** figure is inference memory; conventional full fine-tuning needs roughly **32 GB of parameter/gradient/Adam state before activations**, so one-A6000 training remains unverified. Its published Bridge recipe uses **two processes** and embeds a flattened action trajectory through timestep/adaLN MLPs, so neither changing `nproc_per_node` nor replacing action dimensions demonstrates a working Doom adaptation. It becomes my pick if the intended resolution/context passes a **single-A6000 optimizer-step and rollout benchmark**, its action interface passes the Doom timing/control checks, and measured completion prospects beat the alternatives.

**Cheap causal test**

No cheap test fully identifies the cause of the original U-Net/DiT gap. A **1k-update PixArt pretrained-versus-random-initialization comparison**, holding architecture, batches, noise and optimizer fixed, would isolate the short-budget benefit of its initialization package; reuse the scheduled pretrained run's 1k checkpoint and spend roughly **21 minutes at 0.8 updates/s**, plus overhead, on the random branch. It would not isolate pretraining *quantity* from content/objective or establish the 90k ranking.

My revised design is:

| Item | Revised proposal |
|---|---|
| Checkpoint | Open-Sora 1.1 STDiT-v2-stage3 |
| Data / VAE | Existing decision-frame SD latents and decoder |
| Context | **L16 if timed gate passes; otherwise L8**; one target |
| Conditioning | Latest action only; zero-initialized `30×1152` table on target tokens; native null caption and clean-prefix mask |
| Objective | **Shared VP v-MSE; no variance loss** |
| Optimization | Full fine-tune; lr `2e-5`, warmup `500`, wd `0`, clip `1`; microbatch `1`, accumulation `8`; bf16 autocast/fp32 master; checkpointing |
| Budget | 10k updates or 36 training hours; reserve 36 evaluation hours |
| L16 resource estimate | **24–40 GiB; 0.042–0.106 updates/s—unverified** |
| Evaluation | Shared DDIM50, eta0, no CFG; same raw targets and rollout protocol |
| Placement | GPU3 before unstarted seed **only after gates**; seed behind PixArt |
| Stop | No operational trainer/evaluator by Sep22: omit video row |

**Video candidates:** Open-Sora 1.1 > SkyReels DF 1.3B > Wan2.1-T2V-1.3B > Cosmos-Predict2.5-2B > Cosmos-Predict2-2B; Matrix-Game2 pending a verified training path.
**Video versus seed:** gate-passed video > unstarted second seed; second seed > an ungated video reservation.

### Reply 3b — push on point 1 (expected-value decomposition)

**(a)** My subjective estimates—not calibrated probabilities—are **P(complete | Open-Sora) = 0.80** and **P(complete | SkyReels DF) = 0.60**, including full evaluation by Sep 22.

**(b)** Against the cited U-Net LPIPS **0.324**, conditional on completion: **P(beat | Open-Sora) = 0.40** and **P(beat | SkyReels DF) = 0.60**. These concern that reference, not the unknown final U-Net result.

**(c)** Open-Sora: **0.80 × 0.40 = 0.32**. SkyReels: **0.60 × 0.60 = 0.36**. **SkyReels wins narrowly; I change my pick to SkyReels DF 1.3B.** My previous ranking implicitly required its completion probability to be below **53.3%**; I cannot currently justify that. A failed one-A6000 full-update gate at the intended shape would reverse this preference.

**Lineage:** It is a secondary interpretability benefit, **not load-bearing**; my previous preference rested on a completion advantage that these explicit estimates do not justify.

### Reply 3c — revised design under the new pick

| Item | Revised primary: SkyReels-V2 DF 1.3B |
|---|---|
| Checkpoint | `Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers`; retain its native temporal attention and per-frame timestep/sample-information interface. |
| **VAE decision** | **Re-encode raw tics through the checkpoint's Wan VAE. Do not reuse SD latents.** Encode verified action-aligned chains: one initial frame, then four-tic groups; each subsequent latent step corresponds to one held-action interval. Verify causal encoding, decoder endpoint alignment and reset behavior. |
| Encoding cost before Wednesday | Read **202 GB** of raw data; approximately **43 GB** of fp16 latent payload, excluding metadata/overlap. Stream shards through A4000 workers. Encoding **4.21M frames in 24–48 hours requires 49–25 raw frames/s aggregate**; actual throughput is **unverified**. Encoding and alignment checks must finish before launch. |
| Comparability cost | Different VAE, temporal representation and additional intermediate training frames. This is an **exploratory system comparison**, not a controlled backbone row. Report raw-pixel metrics, per-row reconstruction references and encoding/training costs separately. |
| Context and tokens | **L16 if timed gate passes, otherwise L8**; one target latent. At padded 256×320 resolution: `B×16×17×32×40`, **5,440 tokens**; L8 gives **2,880 tokens**. |
| Action conditioning | **Latest-only for this row:** zero-initialized `Embedding(30,1536)` added to target-frame tokens; zero action injection on history. The latest action controls all four target tics. Per-frame historical-action injection remains a separate information-changing ablation. |
| Objective | **Shared VP v-MSE**, uniform 1,000 timesteps, linear betas `1e-4→0.02`; target-only loss, **no variance loss**. Retain all **16 output channels**. This deliberately converts the native flow predictor to VP-v; convergence is unverified. |
| Context conditioning | Clean prefix at the native zero-noise conditioning setting; target at the VP timestep. Clamp prefix throughout sampling and verify timestep mapping. Cache native null-text conditioning. |
| Optimization | Full fine-tune; AdamW **lr `2e-5`, warmup `500`, wd `0`, clip `1`**; bf16 autocast/fp32 master and optimizer; checkpointing; microbatch **1**, accumulation **8**. |
| Training / evaluation budget | **10k updates, maximum 36 training card-hours**; reserve **36 evaluation card-hours**. Shared DDIM50, eta0, no CFG; decode and score decision endpoints using the existing raw-pixel evaluation protocol. |
| **One-A6000 gate / reversion** | Require finite real-data loss/gradients, action-dependent updates, checkpoint roundtrip, and **no OOM after Adam-state allocation within actual free VRAM**. Require **≥0.0772 optimizer updates/s** and timed-rollout extrapolation placing full evaluation within **36 hours**. Test L16, then L8; **failure at both reverts the recommendation to Open-Sora, subject to its own gate**. Memory and throughput remain unmeasured. |
| Placement | **GPU 3 after U-Net evaluation and successful gates**; move the unstarted seed behind PixArt on GPU 2. If the prototype is not ready when GPU 3 releases, retain the seed schedule rather than reserve an idle card. |
| Stop rule | **Sep 22:** without an operational, validated trainer and evaluator, omit the video row; do not begin another checkpoint migration. |

**Video candidates:** SkyReels DF 1.3B > Open-Sora 1.1 > Wan2.1-T2V-1.3B > Cosmos-Predict2.5-2B > Cosmos-Predict2-2B; Matrix-Game2 pending a verified training path.
**Video versus seed:** gate-passed video > unstarted second seed; second seed > an ungated video reservation.
