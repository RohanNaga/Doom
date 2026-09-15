# Astra: video-pretrained follow-up, Sep 14 night

Status: independent proposal for discussion, not an adopted run or queue change. No remote commands or weight downloads. Read the local evidence cards and inspected primary-source configs/code and installed diffusers 0.40.0.

## Recommendation and interpretation

Keep PixArt 033 and ImageNet seed1 032. Prototype Open-Sora 1.1 while they train, then use a released card after Sep 17–18 if the engineering gate passes. Choose `hpcai-tech/OpenSora-STDiT-v2-stage3` (HF revision `3074d7cc95aca24ae12b685e07fab7d3e2570667`). HF reports 768,271,300 stored F32 parameters; the project calls it 700M. This is an exploratory video-pretrained system row, not a fourth controlled backbone row.

Correct the premise: the pair has not established that pretraining causes the gap or ruled out architecture/optimization effects. Nor is video initialization established for every strong world model: Oasis initialization is undisclosed, GameGen-X is unclear, and NanoWM is not evidence for that universal statement. PixArt tests a pretraining-package intervention, not pretraining in isolation.

## Proposed design (new configuration, not existing CLI flags)

- Same SD decision-rate latent cache, same split and decoder; no new encoding. Open-Sora stage3 config actually names `stabilityai/sd-vae-ft-ema`, and its VAE wrapper uses scale 0.18215 and compression (1,8,8). This is compatible with the incumbent SD1.x encoder; it is not a temporal VAE.
- `context_frames=8`, `target_frames=1`, tensor Bx4x9x32x40, patch (1,2,2), 2880 tokens. FPS conditioning 8.75 (35/4); preserve native spatial positions, temporal RoPE, metadata conditioning and pretrained temporal attention.
- Latest incoming action only: a zero-initialized 30x1152 table added to target-frame patch embeddings, zero action signal on history. Added table has 34,560 parameters. Keep the native null-caption conditioning; no runtime T5. This preserves the information available to the current latest-action baselines. Learned historical-action conditioning is a separate optional change, not included here.
- Clean prefix + native `x_mask` selecting timestep zero on history and sampled timestep on target; clamp history at every reverse step. Native bidirectional attention over prefix plus one noisy target is sufficient: there are no later targets or future actions to leak. This is not diffusion forcing or self-forcing.
- Keep native epsilon DDPM, linear 1000-step schedule and learned-range head, with both noise and variance losses restricted to target. DDIM 50, eta 0, CFG disabled for reporting. Do not pass these outputs to the VP-velocity harness unchanged.
- Full fine-tuning, AdamW lr=2e-5, warmup=500, wd=0, clip=1, bf16 autocast with fp32 master/optimizer, gradient checkpointing on; microbatch=1, accumulation=8, effective batch=8; seed=0; target=10k updates. CPU fp32 EMA optional diagnostic with named-state serialization, not parameter/state_dict zip assumptions.
- Unverified planning estimates: 20–32 GiB peak; 0.08–0.20 optimizer updates/s for batch8 on an available A6000. Static fp32 weights, gradients and Adam moments alone are about 12.3 GB decimal. 10k updates therefore roughly 14–35 hours; sharing and kernels can make it slower. Only measurement authorizes a launch. Cap training at 36 card-hours and reserve up to 36 more for evaluation; proposal is up to 72 additional card-hours, not an assumed allocation.
- Exposure at 10k: 80k next-frame targets, 720k total frame presentations including reused context, not 2.9M independent targets.

## Gates and schedule

Day 1 (Sep15): isolated/pinned STDiT2 implementation; strict weight loading; retain state keys while replacing optional xformers cross-attention with equivalent torch SDPA if necessary. HF code still calls xformers despite a guarded import; not a drop-in diffusers backbone. Verify batch isolation and attention equivalence, metadata and non-square positions. Zero-action parity, action/temporal gradients, masking and target-only loss tests, save/reload, correct sampler output semantics.

Day 2 (Sep16): tiny real-data fixed-noise overfit, future-input exclusion, actual trainer/optimizer benchmark and one 64-decision-step rollout benchmark. Implement adapter into common pixel-space evaluation and EMA loading. Use validation-only action shuffle diagnostic; numerical output dependence alone is not evidence of learned control.

Day 3–4 (Sep17–19): launch only after PixArt or seed1 and its evaluation are finished, with enough measured headroom; no premature cancellation of either run. 10k updates or 36 training hours, whichever first. If substantially undertrained at cap, report as pilot instead of a competitive baseline. Evaluate raw-pixel teacher forcing, same 256x64 decision-rate rollouts and FVD16/32 if measured budget permits; do not silently shrink the evaluation set. Extrapolate all 256 rollouts from a timed 64-frame rollout, using realistic batched inference. Existing original-run evaluation remains higher priority.

If operational trainer/evaluator is not ready by Sep22: stop new-model engineering for this submission, retain the controlled pair, PixArt and seed, finish evaluation/writing. Do not initiate a second VAE migration as fallback. If capacity contracts, a gate-passed video run could replace an unstarted seed only by a new scope decision; never displace PixArt or stop an underway seed merely to free a card.

## Runner-up and candidate screen

Runner-up: Wan-AI/Wan2.1-T2V-1.3B-Diffusers. Apache-2.0, standard diffusers WanTransformer3DModel/AutoencoderKLWan, 30 layers, 12 heads x128, 16 input/output channels, patch(1,2,2), text_dim4096. Stronger contemporary prior is plausible, not demonstrated on Doom. Loses to Open-Sora on new VAE encoding, normalization, temporal boundary/caching, new flow objective and prefix conditioning for a T2V checkpoint. Its advertised 8.19 GB and 4-minute RTX4090 generation are inference facts, not training feasibility. Full Adam static state for 1.3B is ~20.8 GB; 48GB low-resolution microbatch1 training is plausible, unverified. Budget 3–5 engineering days versus 2–3 for Open-Sora.

Cosmos-Predict1-7B: decline. 7B fp32 Adam state is ~112 GB before activations; LoRA/offload would create another restricted-adaptation protocol. CV8x8x8 is 8x temporal, not 4x. Installed AutoencoderKLCosmos default confirms 8; prior evidence card is wrong on this number. Model card advertises 41.1GB inference after offloading prompt components/tokenizer, not full training.

Cosmos-Predict2-2B Video2World: viable low-resolution pilot, not first choice. Official Bridge action post-training exists and launches with two processes. Sample takes first image + 12 future actions to predict 12 frames. Action code flattens the complete action sequence and adds two learned MLP embeddings to timestep/adaLN, rather than per-frame zero-cost addition. Installed Cosmos2VideoToWorldPipeline uses CosmosTransformer3DModel, AutoencoderKLWan, FlowMatchEulerDiscreteScheduler. Native video TokenizerInterface confirms 16ch, temporal4, spatial8; do not confuse it with unused image-tokenizer default8. Native code has normalization details: do not assume numerical interchangeability with Wan-AI latents without checking checkpoint config. Official 32.54GB requirement is inference. Source Apache2, weights NVIDIA Open Model License.

Cosmos-Predict2.5-2B: best Cosmos-specific candidate if Cosmos is required, because current official action training has a single-process 256x320-named experiment, rectified flow and Wan-family VAE. Installed Cosmos2_5_PredictBasePipeline exists and uses CosmosTransformer3DModel + AutoencoderKLWan + UniPCMultistepScheduler. Official docs contain a stale config snippet naming 480x640 data under the 256x320 experiment: inspect the resolved configuration. One process is not proof of <=48GB. Rough full-Adam static state ~32GB before activations; LoRA may be necessary. HF model card access failed (401), so exact chosen checkpoint access/config/size unverified; public source confirms base pre/post-trained, distilled and robot/action-cond variants. Use base non-distilled plus action adaptation; do not import robot action semantics unchanged. Code Apache2, models NVIDIA Open Model License per official README. Do not start distillation.

SkyReels-V2-I2V-1.3B-540P: actual Matrix-Game2 parent; adds native I2V/image conditioning and custom Skywork license, rather than Wan Apache2. Card verifies the model name and license label; detailed license text was not retrieved. More integration work than Wan T2V, but better conditional prior is plausible. SkyReels DF has an installed diffusers pipeline; converting/training arbitrary-noise causal streaming remains a new protocol. Avoid Matrix-Game's 0.5B action-module expansion and self-forcing teacher/distillation before deadline.

CogVideoX-2B: Apache2 and installed CogVideoXTransformer3DModel/CogVideoXPipeline; official card lists 47GB LoRA bs1 and 62GB full fine-tuning bs1 at native settings, and says 720x480 only including fine-tuning. Porting arbitrary Doom resolution is extra work. Its causal VAE changes the interface. Not chosen merely because CPU-offloaded inference has low memory.

## Temporal layout and fairness

For Wan temporal4, use a verified decision boundary and raw frames [s0,s1,...,s4K]: K constant 4-tic action intervals produce 1+K latent frames, with the initial frame exceptional. The target for action a_j is the block ending at s_(4j+4), and score that decoded endpoint against the same decision-state target as the incumbent. Confirm phase using actual transition metadata, not tic modulo. Encoding decision frames instead makes one latent span four decisions; giving those actions together is action-plan-conditioned chunk prediction, not next-action interactive simulation.

Compression is not proof of causal correspondence: test prefix invariance by perturbing future raw frames and future latents, verify decoder endpoint alignment and reset/cache policy, and prevent windows across death/respawn chains. Raw-tic training exposes intermediate images absent from the paired rows; report that data difference. Preserve episode splits before encoding. Full fp16 Wan latents alone would be ~43GB at 1.05M latents x16x32x40x2, before overlap/checkpoints; stream raw shards through Superman rather than copying 202GB under its disk reserve.

Pixel-space comparisons across VAEs are honest system comparisons, not controlled backbone ablations. Match raw targets, decision cadence, information available at prediction, rollout actions, crop, metric implementation and test set. Show each row's reconstruction reference including current fine-tuned SD decoder; do not subtract PSNR ceilings or call reconstruction an absolute bound. IDM must consume decoded generated pixels through the original IDM preprocessing/encoder. Do not compare epsilon, VP-v, flow and EDM loss values. Flow velocity is not VP diffusion velocity.

Report adaptation GPU-hours and measured throughput, target-frame exposures, context presentations, encoder cost, pilot/tuning cost, pretrained provenance, NFE and generation latency. Compare original checkpoints at the same measured adaptation budget only if truly available; equal steps are not equal compute and different objectives/VAEs do not give comparable latent losses.

Surviving wording: Under a shared SD-latent next-frame adaptation recipe and limited compute, ImageNet-initialized DiT-XL/2 shows no advantage over the SD-initialized U-Net. An exploratory video-pretrained temporal transformer is evaluated separately under its own adaptation recipe; it tests an accessible alternative system, not the causal effect of architecture or video pretraining.

## Verified primary sources

- Open-Sora 1.1 report: https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_02.md — “9.7M videos + 2.6M images”; “560k videos + 1.6M images”; “6k” steps from Pixart-alpha-1024; “approximately 9 days on 64 H800 GPUs”; “ema is not applied.”
- Checkpoint metadata/count/license/revision: https://huggingface.co/api/models/hpcai-tech/OpenSora-STDiT-v2-stage3 ; https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage3/blob/main/config.json
- Native masked timestep code: https://github.com/hpcaitech/Open-Sora/blob/v1.1.0/opensora/models/stdit/stdit2.py ; native scheduler: https://github.com/hpcaitech/Open-Sora/blob/v1.1.0/opensora/schedulers/iddpm/__init__.py
- VAE identity and scale: https://github.com/hpcaitech/Open-Sora/blob/v1.1.0/configs/opensora-v1-1/train/stage3.py ; https://github.com/hpcaitech/Open-Sora/blob/v1.1.0/opensora/models/vae/vae.py
- Wan: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers ; transformer/config.json and vae/config.json in that repository.
- Cosmos2 action: https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/post-training_video2world_action.md ; https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/cosmos_predict2/models/video2world_action_dit.py
- Cosmos2 tokenizer: https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/cosmos_predict2/tokenizers/tokenizer.py (TokenizerInterface, not CosmosImageTokenizer).
- Cosmos2 inference requirements: https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/performance.md
- Cosmos2.5 action guide: https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/post-training_video2world_action.md ; public model/license descriptions: https://github.com/nvidia-cosmos/cosmos-predict2.5
- Cosmos1 model card: https://huggingface.co/nvidia/Cosmos-Predict1-7B-Video2World
- CogVideoX2 card: https://huggingface.co/THUDM/CogVideoX-2b
- SkyReels card: https://huggingface.co/Skywork/SkyReels-V2-I2V-1.3B-540P
- NanoWM action embedding is not parameter-free: https://github.com/simchowitzlabpublic/nano-world-model/blob/main/src/models/nanowm.py constructs ActionEmbedder; additive injection avoids additional per-block machinery. The table's zero overhead must not be reported as zero total action parameters.

No A6000 training or generation timings were measured in this review. All memory/throughput/engineering-day projections above are estimates, not benchmark results. Local imports verified diffusers 0.40.0 classes named above; no native STDiT2/OpenSora export is present. Successful import is not a working post-training pipeline.
