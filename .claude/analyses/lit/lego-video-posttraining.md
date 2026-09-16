# Post-training open video models on 50-500 LEGO demonstration clips with 4x RTX A6000 48 GB

Evidence cards, primary sources only (raw GitHub READMEs and docs, Hugging Face model cards, arXiv HTML). Every number is quoted from the page named in the card; "not stated" means the fetched page is silent. Fetched 2026-09-16; raw copies live in the session scratchpad under `lit/`. Nothing here was run on hardware.

Written to the worktree copy of `.claude/analyses/lit/` because the base-repo `.claude/` directory is write-blocked from a worktree session; move it with the branch.

Framing from Rohan (mid-task addition): the demonstrations may be either (a) phone-camera videos of human hands assembling LEGO, or (b) videos of the lab's robot arm doing the placements. Case (a) has no proprioceptive actions, so the recipe is text + first frame (DreamGen style), optionally with a hand-pose control channel. Case (b) has logged actions, so the recipe is action-conditioned post-training (Cosmos Bridge example). The feasibility table and the recommended first experiment give one recipe per case.

Hardware baseline used throughout: 4x RTX A6000, 48 GB each, Ampere (sm_86), bf16 but no native fp8 tensor cores. Whether the four cards are NVLink-bridged is not known to me; assumed not.

---

## 1. Cosmos-Predict2 (2B / 14B Video2World) and Cosmos-Predict2.5 (2B)

**Sources.** `nvidia-cosmos/cosmos-predict2` README, `documentations/performance.md`, `post-training_video2world.md`, `post-training_video2world_action.md`, `post-training_video2world_gr00t.md`, `post-training_video2world_agibot_fisheye.md`, `cosmos_predict2/configs/action_conditioned/defaults/data.py`; `nvidia-cosmos/cosmos-predict2.5` README, `docs/post-training.md`, `docs/post-training_video2world_action.md`, `docs/post-training_cosmos_nemo_assets_lora.md`; HF cards `nvidia/Cosmos-Predict2-2B-Video2World`, `nvidia/Cosmos-Predict2.5-2B`.

**Model and size.** Predict2: "Text2Image: 0.6B, 2B, and 14B", "Video2World: 2B and 14B base models" plus samples "post-trained on GR00T Dreams GR1 dataset", "GR00T Dreams DROID dataset", and "Cosmos-Predict2-2B-Sample-Action-Conditioned: Video + Action based future visual world generation, post-trained on Bridge dataset". Predict2.5: "Base 2B checkpoints and 14B checkpoints", "a flow based model that unifies Text2World, Image2World, and Video2World into a single model", using "Cosmos-Reason1 as its text encoder". Both HF cards: "Diffusion transformer model designed for video denoising in the latent space".

**Latent space.** Not stated on the README, performance, or post-training pages fetched here. The earlier card file `video-warm-starts.md` (lines 17-79) settles the Cosmos tokenizer question from the tokenizer repo; defer to it.

**Conditioning out of the box.** Predict2 2B-V2W: "text descriptions paired with images or videos", six variants "720P at 16FPS (default), 720P at 10FPS, 480P at 16FPS, 480P at 10FPS" plus two NATTEN sparse-attention variants; input "1280x704 (720P) or 832x480 (480P)"; "5-second clip". Predict2.5 model table: base "text + image or video"; `robot/action-cond` input "action"; `robot/policy` "action + image" (post-trained on Libero and RoboCasa). Predict2.5 LoRA doc: "0 conditional frames: Text2World", "1 frame: Image2World", "2+ frames: Video2World", with `conditional_frames_probs={0: 0.333, 1: 0.333, 2: 0.334}` during training.

**Official post-training code and docs.** Predict2: general V2W guide, NeMo-Assets, AgiBot fisheye, GR00T GR1/DROID, and Bridge action-conditioned guides. Predict2.5: NeMo-Assets, DreamGen bench (GR00T), multiview, action-conditioned, plus a LoRA guide ("LoRA post-training for both Video2World and Text2World", 2025-10-21) and a policy recipe in cosmos-cookbook.

**Action-conditioned example (Bridge).** Predict2 doc: "We use the train/validation splits of the Bridge dataset from IRASim". Each JSON has `state` "[x, y, z, roll, pitch, yaw]", `continuous_gripper_state`, and `action`: "The first six dimensions represent displacement in (x, y, z, roll, pitch, yaw) within the gripper coordinate frame. The last (seventh) dimension is a binary value indicating whether the gripper should open (1) or close (0)." Injection: "we create new `conditioner` to support action in `cosmos_predict2/configs/action_conditioned/defaults/conditioner.py`". Data config (`defaults/data.py`): `num_frames=13`, `video_size=[480, 640]`, `cam_ids=[0]`, `batch_size=1`. Launch: `torchrun --nproc_per_node=2 ... experiment="predict2_video2world_2b_action_conditioned_training"` with `fsdp_shard_size=-1`. Inference example loads `iter_000001000.pt` with `--num_conditional_frames 1 --guidance 0`. Predict2.5 version: `torchrun --nproc_per_node=1 ... experiment=ac_reason_embeddings_rectified_flow_2b_256_320`, overrides `/data_train: bridge_13frame_480_640_train`, net `cosmos_v1_2B_action_conditioned`; a DMD2 distillation recipe `..._bridge_13frame_256x320...` also on one GPU. GPU type, memory, iterations: not stated.

**GR00T-Dreams usage.** GR00T-Dreams README: system is "prompted by a single image and language instructions"; step 3 is "Extracting robot actions using a fine-tuned IDM model to LeRobot Format" (franka, gr1, so100, robocasa preprocessors). Predict2 GR00T doc: dataset `nvidia/GR1-100` (videos + `metadata.csv`), `num_frames=93`, `batch_size=1`, 2B launch `--nproc_per_node=8`; 14B launch "4 nodes with 8 GPUs" (`--nnodes=4`). No action conditioning in this path; text + first frame only.

**LoRA support.** Predict2: "If you are interested in training with LoRA, attach `model.config.train_architecture=lora`". Predict2.5 LoRA doc: `use_lora=True, lora_rank=32, lora_alpha=32, lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"`, "Only trains ~1-2% of total model parameters"; launch still `--nproc_per_node=8`, `num_frames=93`.

**Stated GPU memory / count.** Predict2 `performance.md` (inference): "Cosmos-Predict2-2B-Video2World 32.54 GB", "Cosmos-Predict2-14B-Video2World 56.38 GB"; "At least 32GB of GPU VRAM for 2B models", "At least 64GB of GPU VRAM for 14B models"; selection guide "2B ... requires ~26-33GB VRAM", "14B ... requires ~49-57GB VRAM". Post-training memory table: none ("Post-training performance: Review the AgiBot-Fisheye ... example"). AgiBot doc iteration speed: "2B model uses 8 GPUs, while 14B model uses 32 GPUs ... 2B model uses Context-Parallelism of size 2"; per-iteration "NVIDIA A100 22.5 sec (2B) / 22.14 sec (14B)", "NVIDIA H100 NVL 10.07 / 8.72". Predict2.5 HF card: "32.54 GB of GPU VRAM"; L40S inference 2567.1 s vs H100 SXM 228.8 s. Minimum post-training memory, with or without LoRA: not stated anywhere fetched. System requirement: "NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer" (A6000 qualifies).

**Data format.** V2W guide: `metas/*.txt` prompts + `videos/*.mp4`; T5-XXL embeddings precomputed to `t5_xxl/*.pickle` (`scripts.get_t5_embeddings`); example dataset config `num_frames=93, video_size=(704, 1280)  # 720 resolution, 16:9`; `max_iter=1000`, `save_iter=500`, `lr=2 ** (-14.5)`, `batch_size=1`. NeMo-Assets example is "a single caption for 4 long videos". AgiBot: "Split videos into (5-second) windows"; "Expect to use ~100 GB storage in the data preparation". Resolution/fps variants: "post-training can be done from the provided checkpoints with resolution - [480p, 720p] and fps - [10fps, 16fps]".

**Example data sizes and steps.** 4 videos (NeMo assets), GR1-100 (name implies 100 clips; count not stated in the doc), Bridge (IRASim split; size not stated). Checkpoint examples at 1000 iterations.

**Licence.** "NVIDIA Cosmos source code is released under the Apache 2 License." "NVIDIA Cosmos models are released under the NVIDIA Open Model License." HF card: "Models are commercially usable. You are free to create and distribute Derivative Models."

**Diffusers / other frameworks.** Predict2.5 README: "Released Cosmos-Predict2.5-2B Diffusers support via Hugging Face" (`Cosmos2_5_PredictBasePipeline`, 2025-12-19). diffusion-pipe README lists "Cosmos" and "Cosmos-Predict2" for LoRA training. finetrainers, musubi, DiffSynth: not listed.

**What 4x A6000 can realistically do.** 2B inference needs 32.54 GB, so a 48 GB card runs it. The 93-frame 704x1280 post-training examples are all written for 8 GPUs with FSDP and context parallelism, and no per-GPU memory is published, so full 720p post-training on 4x48 GB is unverified. The action-conditioned recipe is far lighter (13 frames at 480x640 or 256x320, batch 1, 1-2 GPUs in the official launch) and is the most plausible Cosmos entry point on this hardware; memory still unmeasured. 14B is out (56.38 GB for inference alone).

---

## 2. Wan2.1 (1.3B T2V, 14B I2V) and Wan2.2 (5B TI2V, A14B)

**Sources.** `Wan-Video/Wan2.1` README, `Wan-Video/Wan2.2` README, HF cards `Wan-AI/Wan2.1-T2V-1.3B`, `Wan-AI/Wan2.2-TI2V-5B`, Wan paper arXiv 2503.20314 HTML, DiffSynth-Studio `docs/en/Model_Details/Wan.md`, musubi-tuner README and `docs/wan.md`, finetrainers README and `docs/models/wan.md`, ai-toolkit README, VideoX-Fun README and `scripts/wan2.1_fun/README_TRAIN_LORA.md` / `README_TRAIN_CONTROL.md`, diffusion-pipe README and issue #152, DreamGen arXiv 2505.12705 HTML.

**Models.** Wan2.1: "T2V-1.3B ... Supports 480P", T2V-14B, I2V-14B-480P, I2V-14B-720P, FLF2V-14B, VACE-1.3B, VACE-14B. Note: the only official 1.3B is T2V; official I2V is 14B. 1.3B I2V variants come from Alibaba PAI (`Wan2.1-Fun-V1.1-1.3B-InP`, inputs `input_image`, `end_image`), VACE-1.3B (`vace_control_video`, `vace_reference_image`), and SkyReels-V2-I2V-1.3B (section 3). Wan2.2: "TI2V-5B: High-compression VAE, T2V+I2V, supports 720P", T2V-A14B and I2V-A14B MoE ("a high-noise expert ... and a low-noise expert", 27B total, 14B active), S2V-14B.

**Latent space.** Wan paper: "Wan-VAE can compress the spatio-temporal dimension of a video by 4x8x8", latent "c=16 denotes the latent channel, t=1+(T-1)/4, h=H/8, w=W/8". Wan2.2 README: "Wan2.2-VAE, which achieves a TxHxW compression ratio of 4x16x16 ... With an additional patchification layer, the total compression ratio of TI2V-5B reaches 4x32x32." 1.3B config table: dim 1536, 30 layers, 12 heads.

**Conditioning.** T2V: text. I2V-14B: first frame + text. FLF2V: first and last frame. VACE: reference images and control video. Fun-InP: start and optional end image. Fun-Control (VideoX-Fun): "control video (e.g., pose control)" plus reference image; README lists "Canny, Pose, Depth, etc. and Trajectory Control" and camera control. TI2V-5B: "natively supports both text-to-video and image-to-video tasks within a single unified framework".

**Format.** Wan2.1: 480P/720P, "81 frames" at 16 fps (inference default). Wan2.2 TI2V-5B: "720P video generation at 24 FPS", `--size 1280*704`. DiffSynth training args: "`height`: must be a multiple of 16", "`num_frames`: default value is 81, must be a multiple of 4 + 1". musubi 1.3B: `--blocks_to_swap` max "29 for 1.3B model", 39 for 14B.

**Official fine-tuning code.** None in the Wan repos; both READMEs point to DiffSynth-Studio: Wan2.1 "LoRA training, and more"; Wan2.2 "low-GPU-memory layer-by-layer offload, FP8 quantization, sequence parallelism, LoRA training, full training".

**Framework support and memory reports.**
- DiffSynth-Studio: table has "Full Training" and "LoRA Training" scripts for Wan2.1-T2V-1.3B, T2V-14B, I2V-14B-480P/720P, FLF2V, VACE-1.3B/14B, Wan2.1-Fun-1.3B-InP / Control / V1.1, and Wan2.2-TI2V-5B (row 87: `input_image`, full and LoRA scripts). Flags `--use_gradient_checkpointing`, `--use_gradient_checkpointing_offload`, `--lora_rank`, `--dataset_metadata_path`. Training VRAM numbers: not stated on the page (inference: "Minimum 8GB VRAM is required to run" 1.3B).
- musubi-tuner: "VRAM: 12GB or more recommended for image training, 24GB or more for video training"; Wan doc: "fp8 support and memory reduction by block swap: Inference of a 720x1280x81frames videos with 24GB VRAM, training with 720x1280 images with 24GB VRAM". Tasks: `t2v-1.3B, t2v-14B, i2v-14B, t2i-14B`, Fun-Control variants, `t2v-A14B, i2v-A14B`; "Support for Wan2.2 model architecture, only for 14B models" (no 5B). I2V: "add `--i2v`". Example 1.3B LoRA: `--learning_rate 2e-4 --max_train_epochs 16 --optimizer_type adamw8bit --gradient_checkpointing --fp8_base`. Main RAM "64GB or more recommended".
- finetrainers: supports Wan2.1-T2V-1.3B-Diffusers, T2V-14B, I2V-14B-480P/720P, FLF2V; `--training_type lora` or `full-finetune`; a Wan "I2V conditioning" control example (`examples/training/control/wan/image_condition/`) that adds first-frame conditioning to the 1.3B T2V by channel-concatenating latents and zero-expanding `patch_embedding`. Memory table row for Wan: "TODO". (LTX/Hunyuan/CogVideoX rows in sections 4-5.)
- ai-toolkit: lists Wan2.1-T2V-1.3B, I2V-14B-480P/720P, T2V-14B, Wan2.2-T2V-A14B, I2V-A14B, TI2V-5B (Diffusers checkpoints); memory numbers: not stated on README.
- VideoX-Fun (Wan2.1-Fun): LoRA guide launches `accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json ... train_lora.py` with `--video_sample_size=640 --video_sample_n_frames=81 --train_batch_size=1 --num_train_epochs=100 --checkpointing_steps=50 --low_vram`; "If you encounter insufficient GPU memory when using multiple GPUs with DeepSpeed-Zero-2, you can switch to FSDP". Token-length rule: "A video with 512x512 resolution and 49 frames has a token length of 13,312". Control guide: `train_control.py`, `metadata.json` with `control_file_path` ("pose videos, edge detection videos"), `--train_mode="control_ref"`. VRAM numbers: not stated.
- diffusion-pipe: lists "Wan2.1 (t2v and i2v)", "Wan2.2"; "hybrid data- and pipeline-parallelism", "blocks_to_swap = 32", "AdamW8BitKahan". Community A6000 datapoint (issue #152): "Wan2.1-I2V-14B-480P" on one A6000 48 GB, 10 videos 512x512, 63 frames at 15 fps, `blocks_to_swap: 2`, lr 6e-5, "around 120 to 180 seconds per step" (60-90 s/step before adding the 63-frame bucket), 10 epochs "several hours".
- Diffusers: `WanPipeline`, `WanImageToVideoPipeline`, `AutoencoderKLWan`; Wan2.2 "integrated into Diffusers". No Wan training script in `diffusers/examples` (listing has `cogvideo`, `cosmos`, `cosmos3`, no `wan`).

**Published few-hundred-clip recipe (DreamGen, arXiv 2505.12705).** "For the majority of our downstream robot experiments, we utilize WAN2.1 as our base video world model." "To mitigate forgetting prior internet video knowledge, we use Low-Rank Adaptation (LoRA) by default." Appendix D: "For all of the WAN 2.1 fine-tuning experiments, we used a learning rate of 1e-4, LoRA rank 4, and LoRA alpha 4. For RoboCasa finetuning, we trained the model for 100 epochs with a batch size of 32. For GR1 finetuning, we trained the model for 75 epochs with a batch size of 64. For DROID fine-tuning, we trained the model for 5 epochs with a batch size of 64." Table 2 train sizes: "RoboCasa 1200, GR1 100" trajectories. Real-world: "we collect 100 trajectories per task"; "we use only 10% of the collected trajectories ... (only 10 real-world trajectories per task)"; SO-100: "10 and 13 videos ... yield 68 and 44 trajectories". Conditioning: "Given an initial frame and a language instruction, the model generates video rollouts"; multi-view data "concatenate the viewpoints into a 2x2 grid". Actions recovered afterwards by "a latent action model or an inverse-dynamics model (IDM)"; "IDM trained only on the 2,884 GR1 pick-and-place data". Which Wan size, GPU count, and time: not stated in the extracted text. Their zero-shot table shows all four base models near 0 instruction-following before fine-tuning, and fine-tuning is what makes them usable ("WAN2.1-sft 77.1 ... Cosmos-sft 79.2").

**Licence.** Wan2.1 and Wan2.2: "The models in this repository are licensed under the Apache 2.0 License." (Fun models are PAI releases; licence not checked here.)

**What 4x A6000 can realistically do.** 1.3B (T2V, Fun-InP, VACE): LoRA and full fine-tune both have official DiffSynth scripts and fit comfortably by every community datapoint (24 GB video LoRA in musubi with block swap). 14B I2V: LoRA on a single 48 GB card is demonstrated (diffusion-pipe issue, 512x512x63, 1-3 min/step, i.e. slow); full fine-tune of 14B is not evidenced on 48 GB anywhere fetched and is implausible from parameter count alone (28 GB bf16 weights before optimizer state; my inference, not a quoted number). Wan2.2 TI2V-5B: DiffSynth has full and LoRA scripts; inference fits "at least 24GB VRAM" with offload; training memory unstated. Wan2.2 A14B: two 14B experts; musubi notes "about 42GB of shared VRAM ... for the two models combined" even in fp8 when training both; out of scope here.

---

## 3. SkyReels-V2 (1.3B / 5B / 14B, I2V and Diffusion Forcing)

**Source.** `SkyworkAI/SkyReels-V2` README and `LICENSE.txt`.

**Model and size.** DF, T2V, I2V at "1.3B-540P 544 * 960 * 97f", "14B-540P", "14B-720P 720 * 1280 * 121f"; 5B rows and "Checkpoints of the 5B Models Series" are unchecked todo items. DF "supports both text-to-video (T2V) and image-to-video (I2V) tasks" and "Infinite-Length videos" (`--num_frames 257` for 10 s, up to 1457 for 60 s). 24 fps.

**Latent space.** Diffusers snippets load `AutoencoderKLWan.from_pretrained(model_id, subfolder="vae")`, i.e. the Wan VAE (4x8x8, 16 ch per section 2). The README thanks "the contributors of Wan 2.1" but does not state the base relationship.

**Conditioning.** Text; first image (I2V); DF: text or image plus autoregressive extension; video-to-video DF pipeline.

**Memory.** "Generating a 540P video using the 1.3B model requires approximately 14.7GB peak VRAM, while the same resolution video using the 14B model demands around 51.2GB peak VRAM" (DF); T2V/I2V "14.7GB" / "43.4GB".

**Training code.** None released in the repo ("inference code and model weights"). Not listed by musubi, finetrainers, DiffSynth, or diffusion-pipe. Because the DiT and VAE are Wan-architecture, generic Wan trainers may load the 1.3B I2V checkpoint, but no page fetched says so.

**Licence.** `LICENSE.txt` front matter reads `license: other`; terms not extracted.

**4x A6000.** Inference of 1.3B and 14B-540P fits. Fine-tuning: no supported path; only worth it if a Wan trainer accepts the checkpoint unchanged (unverified).

---

## 4. HunyuanVideo-I2V and LTX-Video 2B

### HunyuanVideo-I2V
**Source.** `Tencent/HunyuanVideo-I2V` README; finetrainers README; musubi README.
**Model.** Size not stated on README (HunyuanVideo family). Text encoder: "a pre-trained Multimodal Large Language Model (MLLM) with a Decoder-Only architecture"; image tokens "concatenated with the video latent tokens". Output "up to 720P and video length up to 129 frames (5 seconds)".
**Latent.** Not stated on the README fetched.
**Official LoRA training.** "we have released the LoRA training code for customizable special effects"; `sh scripts/run_train_image2video_lora.sh`; trigger word in caption; latents pre-extracted via `hyvideo/hyvae_extract`. Memory table: "LoRA Training | 360p | 79GB"; "Minimum: The minimum GPU memory required is 79GB for 360p"; "tested on a single 80G GPU"; "You can train with 360p data and directly infer 720p videos". Inference: "60GB for 720p".
**Third party.** finetrainers table: "HunyuanVideo | Text-to-Video | 32 GB (LoRA) | OOM (full)" at 49x512x768 with FP8 weights (T2V model, not I2V). musubi lists HunyuanVideo (24 GB video training with block swap); I2V variant support not verified. diffusion-pipe: "HunyuanVideo (t2v)".
**Licence.** Not stated on README; LICENSE file not fetched (GitHub API rate-limited).
**4x A6000.** Official I2V LoRA recipe needs 79 GB: no. Only community fp8/block-swap paths, and those are documented for T2V.

### LTX-Video 2B (and 13B)
**Sources.** `Lightricks/LTX-Video` README, HF card, LTX paper arXiv 2501.00103, `Lightricks/LTX-Video-Trainer` README, `docs/dataset-preparation.md`, `docs/training-modes.md`; finetrainers README.
**Model.** `ltxv-2b-0.9.6-dev`, `ltxv-2b-0.9.6-distilled`, `ltxv-2b-0.9.8-distilled`; 13B 0.9.7/0.9.8 dev, distilled, fp8. Trainer README now banners "LTX-2 is Now Available!" and points to the LTX-2 trainer; LTX-Video trainer remains for 0.9.x.
**Latent.** Paper: "Video-VAE that achieves a high compression ratio of 1:192, with spatiotemporal downscaling of 32x32x8 pixels per token", "increased latent depth of 128 channels".
**Conditioning.** "image-to-video, multi-keyframe conditioning, keyframe-based animation, video extension (both forward and backward), video-to-video transformations, and any combination"; IC-LoRA "Depth Control", "Pose Control", "Canny Control" (13B 0.9.7 releases). Default "1216 x 704 pixels at 30 FPS"; "resolutions that are divisible by 32 and number of frames that are divisible by 8 + 1"; "works best on resolutions under 720 x 1280 and number of frames below 257".
**Official trainer.** "LoRA training, full fine-tuning, and video-to-video transformation workflows"; configs `ltxv_2b_lora.yaml`, "`ltxv_2b_lora_low_vram.yaml` - Optimized for GPUs with 24GB VRAM", `ltxv_2b_full.yaml`, `ltxv_13b_lora_cakeify.yaml`, `ltxv_13b_ic_lora.yaml`. Full fine-tune "Requires significant GPU memory (typically 40GB+ for larger models)". Data: CSV/JSON/JSONL with `caption` and `media_path`; Qwen2.5-VL captioner script; "the trainer only supports using a single resolution bucket"; bucket e.g. `768x768x25`; "For longer, motion-focused videos: use smaller spatial dimensions (512x512) with more frames (121)"; frames "multiple of 8 plus 1". Example datasets: Cakeify, Squish, Canny-Control (sizes not stated on the pages fetched). Multi-GPU: not stated.
**finetrainers.** "LTX-Video | 5 GB (LoRA) | 21 GB (full)" at "49x512x768, rank 128, with pre-computation, using FP8 weights & gradient checkpointing" (LoRA) / BF16 (full). Community fork `eisneim/ltx_lora_training_i2v_t2v` adds I2V training.
**Licence.** README: "New license for commercial use (OpenRail-M)" for 0.9.5; HF card links `LTX-Video-Open-Weights-License-0.X.txt` for 0.9.6+ and 13B.
**4x A6000.** Both LoRA and full fine-tune of 2B fit on one card with margin (21 GB full at 49x512x768). Cheapest iteration loop of any model here; quality ceiling of the 2B at 0.9.6 is the risk, and hand-object detail at 1:192 compression is untested for this use.

---

## 5. CogVideoX-5B-I2V and CogVideoX1.5-5B-I2V

**Sources.** `THUDM/CogVideo` README and `finetune/README.md`, HF card `THUDM/CogVideoX-5b-I2V`, diffusers `examples/cogvideo` listing.

**Model.** CogVideoX-2B (T2V), CogVideoX-5B, CogVideoX-5B-I2V (2024-09-19), CogVideoX1.5-5B, CogVideoX1.5-5B-I2V ("supports video generation at any resolution" per README news; card constraint "Min(W, H) = 768, 768 <= Max(W, H) <= 1360, Max(W, H) % 16 = 0").

**Latent.** README: "3D Causal VAE". Compression ratio: not stated on the fetched pages.

**Format.** CogVideoX-5B-I2V: "720 * 480", frames "8N + 1 where N <= 6 (default 49)", "8 frames / second", "6 seconds"; card: "720 x 480, no support for other resolutions (including fine-tuning)". CogVideoX1.5: "1360 * 768", "16N + 1 where N <= 10 (default 81)", "16 frames / second", "5 seconds or 10 seconds".

**Official finetune docs (diffusers-based) memory table** (`finetune/README.md`, resolution FxHxW):
- `cogvideox-t2v-2b | lora (rank128) | DDP | fp16 | 49x480x720 | 16GB VRAM (NVIDIA 4080)`
- `cogvideox-{t2v, i2v}-5b | lora (rank128) | DDP | bf16 | 49x480x720 | 24GB VRAM (NVIDIA 4090)`
- `cogvideox1.5-{t2v, i2v}-5b | lora (rank128) | DDP | bf16 | 81x768x1360 | 35GB VRAM (NVIDIA A100)`
- `cogvideox-{t2v, i2v}-5b | sft | 1-GPU zero-2 + opt offload | bf16 | 49x480x720 | 42GB VRAM (NVIDIA A100)`
- `cogvideox-{t2v, i2v}-5b | sft | 8-GPU zero-2 | 42GB`; `8-GPU zero-3 | 43GB`; `8-GPU zero-3 + opt and param offload | 28GB VRAM (NVIDIA 5090)`
- `cogvideox1.5-{t2v, i2v}-5b | sft | 1-GPU zero-2 + opt offload | 56GB`; `8-GPU zero-2 | 55GB`; `8-GPU zero-3 | 55GB`; `8-GPU zero-3 + opt and param offload | 40GB VRAM (NVIDIA A100)`
Data: `prompts.txt`, `videos/`, `videos.txt`, optional `images/` + `images.txt` ("if not provided, first frame will be extracted from video as reference"); "The number of frames must be a multiple of 8 plus 1 (i.e., 8N+1), such as 49, 81"; videos are resized to the training resolution; latents cached on disk. Scripts `train_ddp_i2v.sh` (LoRA), `train_zero_i2v.sh` (SFT). Example: "70 training videos with a resolution of 200 x 480 x 720", split "10, 25, and 50 videos"; "Videos with 25 or more frames work best for training new concepts and styles"; `lora_alpha` of 1 "performed poorly". Steps: not stated. SAT-path numbers on the I2V card: "47 GB (bs=1, LORA), 61 GB (bs=2, LORA), 62GB (bs=1, SFT)".

**Diffusers.** `examples/cogvideo/train_cogvideox_image_to_video_lora.py` and `train_cogvideox_lora.py` exist upstream. finetrainers: "CogVideoX-5b | 18 GB (LoRA) | 53 GB (full)" at 49x512x768.

**Licence.** "The CogVideoX-2B model ... is released under the Apache 2.0 License." "The CogVideoX-5B model (Transformers module, include I2V and T2V) is released under the CogVideoX LICENSE" (HF card `license: other`).

**4x A6000.** 5B-I2V LoRA: yes (24 GB). 5B-I2V SFT: yes on one card (42 GB, zero-2 + optimizer offload) with little headroom, or zero-3 + offload across 4 cards (28 GB figure is for 8 GPUs; 4-GPU number not stated). 1.5-5B-I2V LoRA: yes (35 GB). 1.5-5B-I2V SFT: 40 GB with 8-GPU zero-3 offload, 56 GB single: marginal, unverified at 4 GPUs. Costs: fixed 720x480 at 8 fps for the older model; 1.5 gives 16 fps at 768x1360 but 81 frames at that size is heavy.

---

## 6. Open-Sora 2.0 and small (<2B) open I2V models with training code

### Open-Sora 2.0
**Sources.** `hpcaitech/Open-Sora` README and `docs/train.md`, HF card `hpcai-tech/Open-Sora-v2`.
**Model.** "Open-Sora 2.0 (11B)"; "Our 11B model supports 256px and 768px resolution. Both T2V and I2V are supported by one model"; "optimized for image-to-video generation"; frames "4k+1 and less than 129". Fine-tuning from flux-dev also supported (`flux1-dev-fused-rope`).
**Latent.** Not stated on the README (acknowledgements list HunyuanVideo, DC-AE, StabilityAI VAE).
**Training.** `docs/train.md`: CSV with `path,text,num_frames,height,width,aspect_ratio,resolution,fps`; `torchrun --nproc_per_node 8 scripts/diffusion/train.py configs/diffusion/train/stage1.py ... --model.from_pretrained ckpts/Open_Sora_v2.safetensors`; "the batch size is searched on H200 GPUs with 140GB memory"; `stage1_i2v.py` "train t2v and i2v with 256px resolution"; selective gradient checkpointing and CPU offload of checkpoint buffers ("25GB"). Inference: "256x256 ... 52.5GB peak" single GPU, "768x768 ... 60.3GB". Example data: 45k Pexels clips (250 GB).
**Licence.** HF card `license: apache-2.0`; repo LICENSE is Apache 2.0 text.
**4x A6000.** No: 11B, inference alone exceeds 48 GB at 256px, training tuned on 140 GB cards.

### Sub-2B I2V models with training code (from the pages above)
- LTX-Video 2B: official trainer (section 4). Only sub-2B model here with an official full-fine-tune config.
- Wan2.1-Fun-V1.1-1.3B-InP (PAI): I2V (start + optional end frame); VideoX-Fun LoRA and full scripts, DiffSynth full and LoRA scripts.
- Wan2.1-VACE-1.3B: reference image + control video; DiffSynth full and LoRA scripts.
- Wan2.1-Fun-V1.1-1.3B-Control: control video + reference image; VideoX-Fun `train_control.py`.
- SkyReels-V2-I2V-1.3B-540P: no training code (section 3).
- Cosmos-Predict2-2B / 2.5-2B: 2B, not sub-2B, but official post-training (section 1).
- CogVideoX-2B: T2V only; CogVideoX-Fun-V1.1-2b-InP (VideoX-Fun) is the I2V derivative with training code.
- finetrainers "I2V conditioning" control example turns Wan2.1-T2V-1.3B into a first-frame-conditioned model by latent channel concatenation (section 2).

---

## 7. Action and trajectory conditioning add-ons

**Tora (CVPR'25).** `alibaba/Tora` README and arXiv 2407.21705. "Trajectory Extractor (TE), a Spatial-Temporal DiT, and a Motion-guidance Fuser (MGF). The TE encodes arbitrary trajectories into hierarchical spacetime motion patches with a 3D motion compression network. The MGF integrates the motion patches into the DiT blocks." Base: "a CogVideoX version of Tora, built on the CogVideoX-5B model ... meant for academic research purposes only"; T2V and I2V variants; "trajectory is drawn on a 256x256 canvas". Training released ("Text-to-Video training code released"): "requires around 60 GiB GPU memory tested on NVIDIA A100"; paper: "8 NVIDIA A100", "about 630k eligible videos", "2 epochs with dense optical flow and fine-tune for 1 epoch with sparse trajectories", lr 2e-5. Inference "around 30 GiB" (SAT) or "around 5 GiB" (diffusers version). Licence: CogVideoX licence. On 4x A6000: inference yes; retraining the fuser at the published 60 GiB per GPU, no (unless memory tricks not documented).

**MotionCtrl.** `TencentARC/MotionCtrl` README. "Independently control complex camera motion and object motion of generated videos, with only a unified model"; deployments on VideoCrafter, SVD, AnimateDiff; "MotionCtrl deployed on AnimateDiff ... containing both training and inference code"; object trajectories from "ParticleSfM" on WebVid, or drawn with "HandyTrajDrawer". All bases are U-Net era (16-frame class); no DiT port on the page.

**DragAnything.** `showlab/DragAnything` README. "utilizes an entity representation to achieve motion control for any object"; users "only need to draw a line (trajectory)"; built on "svd-temporal-controlnet" (SVD ControlNet); trajectories from Co-Tracker; entity features from a SD1.5 checkpoint (ChilloutMix); train scripts `train_VIPSeg.sh`, `train_youtube_vos.sh`. SVD base, not a DiT.

**Go-with-the-Flow (CVPR'25 oral).** arXiv 2501.08331; repo relocated to `Eyeline-Labs/Go-with-the-Flow` (the `Eyeline-Research` path is empty). "a novel noise warping algorithm ... that replaces random temporal Gaussianity with correlated warped noise derived from optical flow fields"; "agnostic to diffusion model design, requiring no changes to model architectures or training pipelines"; "We use CogVideoX as a base model"; "8 NVIDIA A100 80GB GPUs over the course of 40 GPU days, for 30,000 iterations using a rank-2048 LoRA with a learning rate of 1e-5 and a batch size of 8"; training data >= 720x480, 10-120 s clips captioned by CogVLM2 (count not extracted). Gives object-drag, camera, and motion-transfer control purely through the noise. Bolt-on cost is a LoRA on any base; the published run is 40 A100-days, but the method itself needs no new modules.

**Wan-native control (cheapest on this hardware).** Wan2.1-Fun-Control 1.3B/14B: control video (pose, canny, depth, trajectory, camera) + reference image, official VideoX-Fun training; VACE-1.3B/14B: control video + reference, DiffSynth training; musubi lists `t2v-1.3B-FC`, `i2v-14B-FC`. LTX IC-LoRA: pose/depth/canny control adapters exist for 13B and the trainer supports IC-LoRA training from paired videos.

**Hand-pose conditioning (for human-hand videos).**
- Generated Reality, arXiv 2602.18422: "a human-centric video world model that is conditioned on both tracked head pose and joint-level hand poses"; "we evaluate existing diffusion transformer conditioning strategies and propose an effective mechanism for 3D head and hand control" ("Hybrid 2D-3D Hand" representation); camera encoder "initialized from the FUN model" (Wan-family); distilled to a causal system at "11 FPS ... on a single H100". Training data, GPUs, code: not extracted / not stated in the fetched text.
- LOME, arXiv 2603.27449: egocentric hand-object manipulation "conditioned on the corresponding per-frame human actions"; "rasterized 2D action maps" encoded by the VAE and concatenated with the noisy video latents, plus "A camera adapter encodes per-frame ray maps"; "2,000 steps on 32 NVIDIA A100-80GB GPUs with a global batch size of 32 (one video per GPU), ... learning rate of 1e-5 and a sample resolution of 832x480", 49 frames; dataset "338,234 short videos ... approximately 800 hours" from Apple Vision Pro with 3D hand/arm poses. Baseline "Wan2.1-I2V-14B" compared. Note: one video per A100-80GB at 832x480x49 for a Wan-scale DiT full fine-tune, which is above 48 GB.
- HANDI, arXiv 2412.04189: hand poses are not an input; "Hand Refinement Loss" on detected poses; SD1.5 3D-UNet; "50 epochs with batch size 48 ... on 6 x H100"; no code mentioned. Not a bolt-on.
- Practical route on 4x A6000: render MediaPipe/HaMeR hand keypoints or meshes into a pose video and train Wan2.1-Fun-1.3B-Control (or VACE-1.3B) with `control_file_path`; this is exactly the "pose control" path the VideoX-Fun control guide documents, no new modules.

**How GR00T-Dreams conditions.** Language plus one image only: "prompted by a single image and language instructions"; actions come after generation from an IDM ("Extracting robot actions using a fine-tuned IDM model to LeRobot Format") or LAPA latent actions. The action-conditioned Cosmos path (Bridge) is a separate recipe: per-frame 7-D actions via a conditioner, 13-frame windows.

---

## Feasibility table for 4x RTX A6000 48 GB

"Quoted" = number from a primary page above; "inferred" = my reading, flagged. Human-hand case = text + first frame (+ optional pose control). Robot-arm case = proprioceptive action conditioning.

| Model | Full fine-tune on 48 GB/card | LoRA on 48 GB/card | Quoted memory evidence | Human-hand recipe available | Robot-arm (action) recipe available |
|---|---|---|---|---|---|
| LTX-Video 2B (0.9.6) | Yes | Yes | finetrainers: LoRA 5 GB, full 21 GB at 49x512x768; trainer: 2B low-VRAM LoRA config "for GPUs with 24GB", full "40GB+ for larger models" | Yes: official trainer, I2V via keyframe conditioning (I2V training via community fork) | No action path; would need custom conditioner |
| Wan2.1 1.3B (T2V / Fun-1.3B-InP / VACE-1.3B) | Yes (inferred; DiffSynth and VideoX-Fun ship full-training scripts, memory not stated) | Yes | musubi: "24GB or more for video training", "training with 720x1280 images with 24GB VRAM"; VideoX-Fun `--low_vram` Zero-2 | Yes: Fun-InP first frame + text; Fun-Control pose video for hands; DreamGen hyperparams (LoRA r4, lr 1e-4) | Only via finetrainers-style channel concat of action maps (custom); no official action recipe |
| Wan2.2 TI2V-5B | Probably (inferred; DiffSynth full script exists, memory not stated) | Yes (inferred; DiffSynth and ai-toolkit list it; musubi does not) | Inference "at least 24GB VRAM" with offload | Yes: native TI2V, 720p 24 fps | No |
| Wan2.1 I2V-14B | No (inferred from 14B size; no 48 GB evidence) | Yes, slowly | diffusion-pipe issue #152: one A6000 48 GB, 512x512x63, "120 to 180 seconds per step", blocks_to_swap 2 | Yes: DreamGen used WAN2.1 LoRA (size not stated) | No |
| Cosmos-Predict2-2B V2W | Unverified (official examples: 8 GPUs, FSDP, CP=2, 93 frames 720p; per-GPU memory not published) | Unverified, same launch shape | Inference "32.54 GB"; A100 "22.5 sec" per iteration on 8 GPUs | Yes: text + 1 conditional frame (`--num_conditional_frames 1`), GR00T doc | Yes: Bridge conditioner, 13 frames 480x640, 2 GPUs (2B) |
| Cosmos-Predict2.5-2B | Unverified (8-GPU examples) | Unverified (LoRA r32 recipe, 8-GPU launch) | Inference "32.54 GB"; action-cond launch `--nproc_per_node=1` at 256x320 / 480x640, 13 frames | Yes: unified T2W/I2W/V2W LoRA doc | Yes: action-cond config + DMD2 distillation, single GPU |
| Cosmos-Predict2(.5)-14B | No | No | Inference "56.38 GB"; "At least 64GB of GPU VRAM for 14B models" | - | - |
| CogVideoX-5B-I2V | Yes, marginal | Yes | Official: LoRA "24GB", SFT "42GB" single GPU zero-2 + opt offload, "28GB" 8-GPU zero-3 + offload | Yes: official I2V finetune, diffusers script | No |
| CogVideoX1.5-5B-I2V | Marginal / unverified at 4 GPUs | Yes | LoRA "35GB"; SFT "56GB" single, "40GB" 8-GPU zero-3 + offload | Yes | No |
| HunyuanVideo-I2V | No | No (official) | "minimum GPU memory required is 79GB for 360p" LoRA; finetrainers T2V LoRA 32 GB (fp8), full OOM | Only via community fp8 T2V tooling | No |
| SkyReels-V2 1.3B I2V | No path | No path | Inference "14.7GB peak"; no training code | No official | No |
| Open-Sora 2.0 11B | No | No | Inference "52.5GB peak" at 256px; batch sizes "searched on H200 GPUs with 140GB" | - | - |
| Tora (CogVideoX-5B + TE/MGF) | No | n/a | Training "around 60 GiB ... A100" | trajectory add-on, inference only | - |
| Go-with-the-Flow (noise warping) | LoRA on any base | Yes in principle | Published run: 8x A100 80GB, 40 GPU-days, rank-2048 LoRA | drag/camera/motion-transfer control on top of any base above | - |

A6000 caveat for every row: numbers quoted from 4090/A100/H100 runs assume bf16 or fp8; the A6000 has no fp8 tensor cores, so `--fp8_base`-style savings still cut weight memory but not compute, and musubi/finetrainers fp8 numbers should be treated as memory-only.

---

## Recommended first experiment (every assumption flagged)

### Case A: human-hand phone videos (text + first frame)

- **Model.** `PAI/Wan2.1-Fun-V1.1-1.3B-InP` (first frame + text; optional end frame lets you also condition on the goal state, which suits "place brick k" steps). Fallback with the same data: `Wan-AI/Wan2.1-T2V-1.3B` + finetrainers I2V control example. Step-up once the loop works: `Wan2.2-TI2V-5B` (720p 24 fps, native TI2V, DiffSynth full/LoRA scripts). Wan chosen because it is the base DreamGen used at 100-trajectory scale and the only family with 1.3B, 5B, and 14B on one VAE and one trainer stack.
- **Trainer.** DiffSynth-Studio LoRA script for Fun-1.3B-InP (official for that checkpoint; also VideoX-Fun `train_lora.py` with DeepSpeed Zero-2 across the 4 cards). Diffusers-only alternative: finetrainers (Wan memory row is "TODO", so no quoted number).
- **Data format.** One 5 s clip per assembly step, cut so the first frame shows the pre-placement state; 832x480 (Wan 480P recommended for 1.3B: "we recommend using 480P"), 81 frames at 16 fps (Wan default; DiffSynth requires "multiple of 4 + 1"); captions from Qwen2.5-VL (LTX trainer's captioner or your own) with a fixed trigger phrase plus the step description ("place the 2x4 red brick on the left stud"); `metadata.csv` with video path and prompt. If the phone shoots 30/60 fps, subsample to 16 fps. Multi-view (overhead + side): DreamGen tiled views into a 2x2 grid; single view is simpler.
- **Hyperparameters.** DreamGen: lr 1e-4, LoRA rank 4 / alpha 4 (on their Wan; size not stated); musubi example: lr 2e-4, adamw8bit, 16 epochs; VideoX-Fun default 100 epochs at batch 1. Start: rank 16-32 (assumption; DreamGen's rank 4 was on a larger model), lr 1e-4, batch 1 per GPU x 4 GPUs, gradient checkpointing on, 60-100 epochs over 200 clips (i.e. 3k-5k optimizer steps at global batch 4). Validate on held-out first frames every 500 steps with fixed seeds.
- **Expected wall-clock.** No primary number exists for 1.3B at 832x480x81 on an A6000. The only A6000 datapoint is 14B I2V at 512x512x63 at 60-180 s/step with block swap. Assumption: 1.3B without block swap at ~4x fewer tokens per clip runs roughly 3-8 s/step per GPU; 4k steps at global batch 4 is then ~1-3 hours of pure compute, plus VAE/T5 precompute. Treat as an order-of-magnitude guess to be replaced by a 50-step timing run.
- **Hand control (optional second run).** Extract 2D hand keypoints per frame, render to a skeleton video, train `Wan2.1-Fun-V1.1-1.3B-Control` with VideoX-Fun `train_control.py` (`control_file_path`), following the pose-control example; this is the closest documented analogue of Generated Reality / LOME hand conditioning that fits 48 GB.

### Case B: robot-arm videos with logged actions (action-conditioned)

- **Model and recipe.** `Cosmos-Predict2.5-2B` action-conditioned post-training (`docs/post-training_video2world_action.md`): convert each episode to the Bridge layout, `videos/*.mp4` plus `annotations/*.json` with per-frame `state` [x, y, z, roll, pitch, yaw], `continuous_gripper_state`, and 7-D `action` (6-D end-effector displacement in the gripper frame + binary gripper). If the arm has no parallel gripper, the 7th dim is a placeholder (assumption; the conditioner's action dimension is set in config, not checked here). Launch is the documented single-GPU command (`--nproc_per_node=1`, `ac_reason_embeddings_rectified_flow_2b_256_320`, or the `480_640` data override); scale to 4 GPUs with `--nproc_per_node=4` (assumption: FSDP config accepts it). Start from `Cosmos-Predict2.5-2B/robot/action-cond` (already post-trained on Bridge) rather than the base, to reduce steps.
- **Data format.** 13-frame windows (`num_frames=13`, `sequence_interval=1`) at 480x640 or 256x320, `cam_ids=[0]`, batch 1. Frame rate should match the action logging rate after subsampling (Bridge fps not stated on the page; pick 5-10 Hz so 13 frames cover 1.3-2.6 s of a placement).
- **Steps and wall-clock.** The doc's inference example loads `iter_000001000.pt`, so 1000 iterations is the documented scale. Memory: 2B inference at 93 frames 720p is 32.54 GB; 13 frames at 480x640 is ~14x fewer tokens (inferred), so a 48 GB card should hold training with FSDP over 4 cards, but no published number confirms it. Speed: A100 does a 93-frame 720p iteration in 22.5 s with CP=2 over 8 GPUs; at 13x480x640 on an A6000, assume 3-8 s/iter per GPU, so 1000 iterations is ~1-2 hours. Flagged guess.
- **If actions are not needed as an input** (you only want videos and will label them post hoc), use the Case A recipe on the arm videos and recover actions with an IDM, exactly the GR00T-Dreams pipeline; the Cosmos GR00T doc gives the text + first-frame post-training config (93 frames, 8 GPUs) and the DreamGen paper the Wan LoRA hyperparameters.

### Shared first-week plan
1. Timing/memory probe: 50 steps of each candidate on one A6000 at the target shape; record peak VRAM and s/step. This replaces every guess above.
2. Split 20% of clips by assembly (not by clip) for held-out first frames.
3. Metrics: DreamGen's two axes, instruction following (VLM-judged) and physics following; plus per-step brick-placement correctness judged on the last frame.

---

## Open questions
1. Cosmos-Predict2/2.5 post-training memory on 48 GB cards: never published; the 8-GPU FSDP+CP launches may or may not shard down to 4x48 GB at 93 frames. Only the 13-frame action config is plausibly single-card.
2. Which Wan2.1 size DreamGen fine-tuned (1.3B or 14B) and on how many GPUs: not in the extracted text; check the paper PDF appendix F/G.
3. Wan2.2 TI2V-5B training memory: no number anywhere fetched (DiffSynth has scripts; musubi excludes 5B).
4. SkyReels-V2 1.3B I2V as a Wan-trainer drop-in: architecture match is implied by `AutoencoderKLWan` usage but untested.
5. HunyuanVideo-I2V licence and any fp8 I2V LoRA path under 48 GB: not verified.
6. Cosmos action conditioner dimensionality and whether it accepts non-gripper action vectors (joint-space arm actions): need `cosmos_predict2/_src/predict2/configs/conditioner.py`.
7. NVLink between the A6000s: affects whether FSDP/Zero-3 across 4 cards is practical for the 5B/14B options.
8. Phone-camera domain gap: all cited robot recipes use fixed cameras; DreamGen's initial-frame prompting assumes a stable viewpoint. Handheld motion may need a camera-control channel (Fun-Control-Camera) or stabilisation at preprocessing.
