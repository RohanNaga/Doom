# Video-pretrained warm starts for DoomDiT — verified source notes

Compiled 2026-09-14. Every number below is quoted from a primary source that was downloaded raw
(`curl` of arXiv LaTeXML HTML, `raw.githubusercontent.com` files, `huggingface.co/<repo>/raw/main/...`,
`huggingface.co/api/models/...`). No WebFetch summariser was used. Where a source is silent the field
says **not stated**; where a fetch failed the field says **not retrieved**.

Two corrections to the brief, up front:

- **arXiv 2505.13211 is not SkyReels-V2.** `https://arxiv.org/abs/2505.13211` returns
  `<title>[2505.13211] MAGI-1: Autoregressive Video Generation at Scale</title>`. The SkyReels-V2
  paper is **arXiv 2504.13074** (the ID linked from the SkyReels-V2 README), title
  "SkyReels-V2: Infinite-length Film Generative model". Card 5 uses 2504.13074.
- **The NVIDIA Cosmos HF model cards are gated.** `huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World/raw/main/README.md`
  returns HTTP 401 with the body `Access to model nvidia/Cosmos-Predict2-2B-Video2World is restricted.
  You must have access to it and be authenticated to access it. Please log in.` Same for the 14B, the
  action-conditioned sample model, `nvidia/Cosmos-Predict2.5-2B`, and `nvidia/Cosmos-1.0-Tokenizer-CV8x8x8`.
  Card *text* for those is therefore **not retrieved**; licence/date/file-list metadata came from the
  public `huggingface.co/api/models/...` endpoint, and the technical facts came from the Apache-2.0
  GitHub repos and from the diffusers source, which are not gated.

---

## 1. NVIDIA Cosmos-Predict1 — Diffusion 7B / 14B Video2World

**What it is.** From `README.md` of `nvidia-cosmos/cosmos-predict1` (repo created 2025-03-02, licence
Apache-2.0 per the GitHub API): "Cosmos-Predict1 is a key branch of Cosmos World Foundation Models
(WFMs) specialized for future state prediction, often referred to as world models." Paper:
arXiv 2501.03575, "Cosmos World Foundation Model Platform for Physical AI". The repo carries a banner:
"This repository is no longer under active development and will receive only limited maintenance
updates."

**Parameter count of the diffusion network.** The paper names the models 7B and 14B and gives Table 11
("Configuration details of Cosmos-1.0-Diffusion models"), quoted verbatim:

| | 7B-Text2World | 14B-Text2World | 7B-Video2World | 14B-Video2World |
|---|---|---|---|---|
| Number of Layers | 28 | 36 | 28 | 36 |
| Model Dimension | 4,096 | 5,120 | 4,096 | 5,120 |
| FFN Hidden Dimension | 16,384 | 20,480 | 16,384 | 20,480 |
| AdaLN-LoRA Dimension | 256 | 256 | 256 | 256 |
| Number of Attention Heads | 32 | 40 | 32 | 40 |
| Conditional Information | Text; FPS | Text; FPS | Text; FPS; Frames | Text; FPS; Frames |
| Base Learning Rate | 2^-15 | 2^-16 | 2^-15 | 2^-16 |
| Weight decay | 0.1 | 0.2 | 0.1 | 0.2 |

Also quoted: "For Cosmos-1.0-Diffusion-7B, this architectural optimization achieves a 36% reduction in
parameter count (from 11B to 7B parameters)". No exact integer parameter count is printed.

**VAE / tokenizer.** Paper: "These diffusion models are latent diffusion models that take continuous
tokens. We use Cosmos-1.0-Tokenizer-CV8x8x8 to produce the visual tokens."

*Temporal compression — settled.* The `NVIDIA/Cosmos-Tokenizer` README says: "Cosmos Tokenizer achieves
spatial compression rates of 8x or 16x and temporal compression factors of 4x or 8x", and for this
specific checkpoint: "To instantiate a `Cosmos-CV` with a temporal factor of 8 and a spatial
compression factor of 8, append the following command line arguments: … `--temporal_compression=8`
… `--spatial_compression=8`", with the example headed `# Autoencoding videos using Cosmos-CV with a
compression rate of 8x8x8.` and `model_name="Cosmos-1.0-Tokenizer-CV8x8x8"`. **CV8x8x8 is 8× temporal,
8×8 spatial.** (The 4× temporal option belongs to the `CV4x8x8` / `DV4x8x8` variants.) The Cosmos paper's
own context-length footnote agrees: "10,240 (the context length) is computed as: 640 (width) ÷ 8
(tokenize) ÷ 2 (patchify) × 512 (height) ÷ 8 (tokenize) ÷ 2 (patchify) × [(57−1)÷8+1] (tokenize frames)."

*Latent channels: 16.* From `cosmos_predict1/diffusion/training/config/video2world/experiment.py`,
every experiment block reads:

```python
latent_shape=[
    16,  # Latent channel dim
    16,  # Latent temporal dim
    88,  # Latent height dim
    160, # Latent width dim
],
```

Independently corroborated in diffusers `v0.34.0`,
`src/diffusers/models/autoencoders/autoencoder_kl_cosmos.py`: "Autoencoder used in
[Cosmos](https://huggingface.co/papers/2501.03575). … `latent_channels` (`int`, defaults to `16`)".

*Causal?* The tokenizer API class is `CausalVideoTokenizer` (`from cosmos_tokenizer.video_lib import
CausalVideoTokenizer`). *Licence:* Cosmos-Tokenizer README — "**Models**: The models are licensed under
[NVIDIA Open Model License] … **GitHub Code**: This repository is licensed under the Apache 2.0 license."

**Objective — EDM, not flow matching.** Paper §5.1: "To train our diffusion WFMs, we adopt the approach
outlined in EDM (Karras et al., 2022; Karras et al., 2024). The denoising score matching loss for the
denoiser D_θ, evaluated at a noise level σ, is defined as …". The overall loss is "a weighted expectation
of L(D_θ;σ) over the noise levels", "where the distribution of noise levels σ is controlled by
hyperparameters P_mean and P_std". They add uncertainty weighting: "We use a simple MLP to parameterize
u(σ) and minimize the overall loss L(D_θ) during training." They explicitly contrast this with flow
matching: "Compared to recent video generative models that adopt the Gaussian flow matching
formulation …, our work is derived from the diffusion score matching perspective … these frameworks are
theoretically equivalent".

**How conditioning frames enter — concat along time + binary mask channel.** Paper: "the conditional
frame(s) are concatenated with the generated frames along the temporal dimension. To improve robustness
against variations in input frame(s) during inference, we introduce augmented noise to the conditional
frames during training. The sigma value for this augmented noise is sampled with P_mean = −3.0,
P_std = 2.0. Additionally, the input to the diffusion model is concatenated along the channel dimension
with a binary mask that distinguishes conditional frames from generated frames. The loss function
excludes contributions from the locations of conditional frames … During inference, the model can
flexibly operate with either a single conditional frame (image) or multiple previous frames as input."
Code confirms the extra channel: `general_dit_action.py` — `def __init__(self, *args, in_channels=16 + 1,
...)` with the comment `# extra channel for video condition mask`.

**Text encoder, and whether it can be bypassed.** Paper: "For our Text2World models, we employ T5-XXL
(Raffel et al., 2020) as the text encoder. We zero-pad T5 embeddings to maintain a fixed sequence length
of 512." Injection is cross-attention: "cross-attention integrates semantic context using T5-XXL …
embeddings as keys and values". **Bypassable in the action-conditioned recipe:** the post-training doc
`examples/post-training_diffusion_video2world_action.md` states plainly "Action control does not require
T5-XXL embeddings. No preprocessing is necessary.", and the dataset config sets
`load_t5_embeddings=False`.

**Native resolution and frame count.** Table 12 "Stages of progressive training and their specifications":
Low-resolution Pre-training "512p (640 × 512)", 57 frames, context length 10,240, FSDP size 64, CP size 2;
High-resolution Pre-training and High-quality Fine-tuning "720p (1280 × 704)", 121 frames, context length
56,320, FSDP 64, CP 8. Multi-aspect training uses "five distinct buckets corresponding to ratios of 1:1,
3:4, 4:3, 9:16, and 16:9".

**Training data scale as stated.** "We use the pipeline to extract about 100M clips of videos ranging from
2 to 60 seconds from a 20M hour-long video collection." And: "In total, we accumulate about 20M hours of
raw videos with resolutions from 720p to 4k. … We generate about 10^8 video clips for pre-training and
about 10^7 for fine-tuning."

**Published post-training recipe and memory.** `examples/post-training_diffusion_video2world.md` support
matrix: `| Cosmos-Predict1-7B-Video2World | **Supported** | 8 NVIDIA GPUs* |` with the footnote
"`H100-80GB` or `A100-80GB` GPUs are recommended." The base example uses `num_frames = 121`,
`video_size=(720, 1280)`, `batch_size=1`, `torchrun --nproc_per_node=8`.

There is a dedicated low-memory doc, `examples/post-training_diffusion_video2world_lowmemory.md`, which
is the most relevant thing in this whole survey for a 48 GB card. It says: "Optionally, reducing the
video resolution and the number of frames can facilitate training with less number (4) of GPUs or with
GPUs with lower memory (40GB)." Its configurations are 4×80GB, 8×40GB, 4×40GB. The 4×40GB one:
"To run with 4 GPUs with A100 40GB, run experiment
`video2world_7b_example_cosmos_nemo_assets_4gpu_40gb`. It trains with `cosmos_nemo_assets` data at
384x384 resolution, video length of 25 frames." with config

```python
n_length_4gpu_40gb = 2
num_frames_4gpu_40gb = 8 * n_length_4gpu_40gb + 1  # 17
video_size=(192, 192),  # a low-res example for lower VRAM utilization without considering aspect ratio.
ema=dict(enabled=False),  # turn off to save memory
```

(Note the internal inconsistency in NVIDIA's own doc: the prose says "384x384 resolution, video length
of 25 frames" while the config immediately below says `video_size=(192, 192)` and `num_frames = 17`.
Reported as-is.)

The paper's own pre-training memory accounting: "Model parameters: 10 bytes per parameter … Gradients:
2 bytes per parameter … Optimizer states: 8 bytes per parameter", and "our 14B model
(Cosmos-1.0-Diffusion-14B-Text2World) requires approximately 280 GB for model parameters, gradients, and
optimizer states, alongside 310 GB for activations during high-resolution pre-training."

**Published action-conditioned variant and its recipe.** Yes —
`examples/post-training_diffusion_video2world_action.md` plus
`cosmos_predict1/diffusion/training/config/video2world_action/experiment.py`.

- Dataset: Bridge from IRASim (`bridge_train_data.tar.gz` from `lf-robot-opensource.bytetos.com`).
- Dataloader: `num_frames=2`, `video_size=[256, 320]`, `cam_ids=[0]`, `accumulate_action=False`,
  `load_action=True`, `load_t5_embeddings=False`, `batch_size=8`, `num_workers=8`.
- Optimiser/schedule: `lr=4e-4`, `weight_decay=0.1`, `betas=[0.9, 0.99]`, `eps=1e-10`,
  `max_iter=100_000`, `warm_up_steps=[2500]`, `distributed_parallelism="fsdp"`,
  `sharding_group_size=32`, `sharding_strategy="hybrid"`.
- Latent shape: `# Use 16x2x32x40 latent shape for training` → `[16, 2, 32, 40]`.
- Warm start: `load_path="checkpoints/Cosmos-Predict1-7B-Video2World/model.pt"`,
  `load_training_state=False`, `strict_resume=False`.
- Launch: `torchrun --nproc_per_node=8`, same 8×(H100/A100-80GB) requirement table.

**How the action enters — this is the part worth copying.** `general_dit_action.py`, class
`ActionConditionalVideoExtendGeneralDIT`, docstring: "Action embedding is would be added to timestep
embedding." Two MLPs on a **7-dimensional** action:

```python
self.action_embedder_B_D = Mlp(in_features=7, hidden_features=self.model_channels * 4,
                               out_features=self.model_channels, act_layer=lambda: nn.GELU(approximate="tanh"))
self.action_embedder_B_3D = Mlp(in_features=7, hidden_features=self.model_channels * 4,
                                out_features=self.model_channels * 3, ...)
```

and in the forward:

```python
assert action is not None, "Action is required for action-conditional training"
action = action[:, 0, :]  # Since we are now training on 1 frame, we only need the first frame action.
action_embedding_B_D = self.action_embedder_B_D(action)
action_embedding_B_3D = self.action_embedder_B_3D(action)
timesteps_B_D = timesteps_B_D + action_embedding_B_D
adaln_lora_B_3D = adaln_lora_B_3D + action_embedding_B_3D
```

So: one action vector per sample, added to the timestep embedding and to the AdaLN-LoRA modulation —
structurally the same "one action conditions the step" design as DoomDiT's Design A.

**diffusers support.** `CosmosTransformer3DModel`, `CosmosTextToWorldPipeline`,
`CosmosVideoToWorldPipeline` (and the Cosmos2 pair) first appear in **diffusers v0.34.0**
(published 2025-06-24 per the GitHub releases API — `src/diffusers/models/transformers/transformer_cosmos.py`
is 404 at `v0.33.0` and 200 at `v0.34.0`). `pipeline_cosmos_video2world.py` at v0.34.0 imports
`from ...models import AutoencoderKLCosmos, CosmosTransformer3DModel` and declares
`vae: AutoencoderKLCosmos`.

**Weights licence.** NVIDIA Open Model License (Cosmos-Tokenizer README, and the HF API `cardData` for
the Cosmos-Predict2 repos gives `license: other`, `license_name: nvidia-open-model-license`,
`license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license`).
Code licence Apache-2.0.

**URLs / dates.** Repo `https://github.com/nvidia-cosmos/cosmos-predict1`, created 2025-03-02 (GitHub API).
Paper `https://arxiv.org/abs/2501.03575`. HF collection
`https://huggingface.co/collections/nvidia/cosmos-predict1-67c9d1b97678dbf7669c89a7`.

---

## 2. NVIDIA Cosmos-Predict2 — 2B / 14B Video2World, and the action-conditioned sample model

**What it is.** Repo `https://github.com/nvidia-cosmos/cosmos-predict2`, created 2025-06-11, code
Apache-2.0 (GitHub API). HF `nvidia/Cosmos-Predict2-2B-Video2World` and `-14B-Video2World` (both gated).

**Parameter count.** Named 2B / 14B; no integer count stated in any retrieved source. Net configs from
`cosmos_predict2/configs/base/config_video2world.py`:

| | 2B | 14B |
|---|---|---|
| `model_channels` | 2048 | 5120 |
| `num_blocks` | 28 | 36 |
| `num_heads` | 16 | 40 |
| `in_channels` / `out_channels` | 16 / 16 | 16 / 16 |
| `patch_spatial` / `patch_temporal` | 2 / 1 | 2 / 1 |
| `max_img_h` / `max_img_w` / `max_frames` | 240 / 240 / 128 | 240 / 240 / 128 |
| `pos_emb_cls` | `rope3d` | `rope3d` |
| `use_adaln_lora` / `adaln_lora_dim` | True / 256 | True / 256 |

`cosmos_predict2/models/video2world_dit.py`: `kwargs["in_channels"] += 1  # Add 1 for the condition mask`.

**VAE — is it Wan2.1's VAE?** The repo does not say so in prose; the checkpoint is shipped per-model as
`tokenizer/tokenizer.pth` (`imaginaire/constants.py`:
`return f"{model_dir}/tokenizer/tokenizer.pth"`, and the HF API file list for the action model contains
`tokenizer/LICENSE.txt`, `tokenizer/config.json`, `tokenizer/tokenizer.pth`). **The diffusers port
settles it**: `src/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py` at `v0.34.0` reads

```python
from ...models import AutoencoderKLWan, CosmosTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
...
    vae ([`AutoencoderKLWan`]):
...
        vae: AutoencoderKLWan,
```

i.e. HF implements Cosmos-Predict2's tokenizer as the **Wan VAE class**, whose docstring says
"A VAE model with KL loss for encoding videos into latents … Introduced in [Wan 2.1]" with
`z_dim: int = 16` and `temperal_downsample: List[bool] = [False, True, True]`. Cosmos-Predict2's own
`cosmos_predict2/tokenizers/tokenizer.py` is structurally the same code (`CausalConv3d`, `RMS_norm`,
`temperal_downsample=[True, True, False]`, `z_dim=16`, `class VAE: def __init__(..., z_dim=16, ...)`),
including the same misspelling `temperal_downsample`. **NVIDIA never states "we use the Wan VAE"** in
any source I retrieved, so treat "same architecture and latent spec as Wan-VAE, and diffusers loads it
as `AutoencoderKLWan`" as the verified claim, and "it is literally Wan's checkpoint" as **not stated**.

Pipeline-level tokenizer settings (`config_video2world.py`): `state_ch=16`, `state_t=24` (16 for the
10 fps variants), `tokenizer=L(TokenizerInterface)(chunk_duration=81, load_mean_std=False, ...)`.

**Text encoder, and bypassing it.** `imaginaire/constants.py`: `T5_MODEL_DIR = f"{CHECKPOINTS_DIR}/google-t5/t5-11b"`,
and `TextEncoderClass` has members `COSMOS_REASON1 = "cosmos_reason1"` and `T5 = "t5"` with
`default=TextEncoderClass.T5`. **Explicitly bypassable for action conditioning** — in
`cosmos_predict2/configs/action_conditioned/defaults/model.py`:

```python
model_manager_config=L(Predict2ModelManagerConfig)(
    dit_path=get_cosmos_predict2_action_conditioned_checkpoint(model_size="2B", resolution="480", fps=4),
    text_encoder_path="",  # Do not load text encoder for training.
),
```

The action pipeline still keeps a `text=L(TextAttr)(dropout_rate=0.2, input_key=["t5_text_embeddings"])`
slot in the conditioner, and disables the prompt refiner and guardrail:
`# disable prompt refiner and guardrail for action conditional`.

**Objective — rectified flow.** `Video2WorldPipelineConfig` fields include
`rectified_flow_t_scaling_factor=1.0`, `rectified_flow_loss_weight_uniform=True`, `sigma_data=1.0`,
`sigma_conditional=0.0001`, `precision="bfloat16"`. The diffusers port uses
`FlowMatchEulerDiscreteScheduler`. The loss is not written out in the repo docs — the equation itself is
**not stated** in anything I retrieved for Predict2 (it is written out for Predict2.5, see card 3).

**How conditioning frames enter.** `ConditioningStrategy` enum, quoted:

```python
FRAME_REPLACE = "frame_replace"   # First few frames of the video are replaced with the conditional frames
CHANNEL_CONCAT = "channel_concat" # First few frames of the video are concatenated in the channel dimension
```

Base video2world uses `conditioning_strategy=str(ConditioningStrategy.FRAME_REPLACE)`,
`min_num_conditional_frames=1`, `max_num_conditional_frames=2`. The condition mask is concatenated on
channels inside the DiT (`torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W...], dim=1)`).
The inference doc says: "**Multi-frame conditioning**: Uses the last 5 consecutive frames from a video
for enhanced temporal consistency" and `--num_conditional_frames: Number of frames to condition on
(choices: 1, 5, default: 1)`.

**Native resolution and frame count.** `--resolution: … (choices: 480, 720, default: 720)`,
`--fps: … (choices: 10, 16, default: 16)`. "By default the checkpoint you downloaded … are for 720P and
16FPS." Long video is autoregressive: "we only generate one chunk of video. To generate longer videos of
multiple chunks, we support long video generation in an auto-regressive inference manner … iteratively
taking the last `num_conditional_frames` frames of the previous chunk as input condition of next chunk."

**Training data scale.** **Not stated** in the repo. (No Predict2 tech report is linked from the repo
README; the only paper link in the Cosmos-Predict2.5 README is arXiv 2511.00062.)

**Inference memory table** (`documentations/performance.md`), quoted:

| Model | Required GPU VRAM |
|---|---|
| Cosmos-Predict2-2B-Text2Image | 26.02 GB |
| Cosmos-Predict2-14B-Text2Image | 48.93 GB |
| Cosmos-Predict2-2B-Video2World | 32.54 GB |
| Cosmos-Predict2-14B-Video2World | 56.38 GB |

plus "At least 32GB of GPU VRAM for 2B models / At least 64GB of GPU VRAM for 14B models" and
"NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer". Note: "Video2World was run with
480p resolution and at 16 FPS." An L40S (48 GB) row exists: 2B-Video2World 127.49 sec, 14B-Video2World
1036.24 sec; 14B-Text2Image is "(OOM)".

**Post-training memory.** There is **no post-training VRAM table** in `cosmos-predict2`. `performance.md`
only says "Review the [AgiBot-Fisheye](post-training_video2world_agibot_fisheye.md) post-training example,
which contains performance numbers on different GPUs." That doc gives iteration speed, not memory:
"Note that 2B model uses 8 GPUs, while 14B model uses 32 GPUs. 14B model also has 4x lower global batch
size, as it uses Context-Parallelism of size 8, while 2B model uses Context-Parallelism of size 2."

| GPU Hardware | 2B-Video2World | 14B-Video2World |
|---|---|---|
| NVIDIA H100 NVL | 10.07 sec | 8.72 sec |
| NVIDIA A100 | 22.5 sec | 22.14 sec |

The generic video2world post-training doc launches with `torchrun --nproc_per_node=8`.

**Action-conditioned sample model — `nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned`.**
HF API metadata: created `2025-06-12T22:28:00.000Z`, last modified `2025-06-17T17:35:07.000Z`,
`gated: auto`, `license: other` / `nvidia-open-model-license`, files
`['.gitattributes', 'README.md', 'config.json', 'model-480p-4fps.pt', 'tokenizer/LICENSE.txt',
'tokenizer/config.json', 'tokenizer/tokenizer.pth']` — note the single checkpoint is literally named
**`model-480p-4fps.pt`**.

- **Dataset: Bridge.** "We use the train/validation splits of the Bridge dataset from IRASim for
  action-conditioned post-training."
- **Action definition (7-dim):** "`action`: The gripper's displacement at each timestep. The first six
  dimensions represent displacement in (x, y, z, roll, pitch, yaw) within the gripper coordinate frame.
  The last (seventh) dimension is a binary value indicating whether the gripper should open (1) or
  close (0)."
- **Resolution / fps:** `get_cosmos_predict2_action_conditioned_pipeline(model_size="2B",
  resolution="480", fps=4)`.
- **Frames / conditioning:** `min_num_conditional_frames=1`, `max_num_conditional_frames=1`,
  `state_ch=16`, `state_t=4`, `chunk_duration=81`. Inference is run with `--num_conditional_frames 1
  --guidance 0 --disable_guardrail --disable_prompt_refiner`.
- **How actions are injected — per chunk, not per frame.** `cosmos_predict2/configs/action_conditioned/config.py`
  sets `action_dim=7 * 12` (comment: `# NOTE: add action dimension`), i.e. 12 timesteps × 7 dims flattened.
  `cosmos_predict2/models/video2world_action_dit.py`:

  ```python
  # NOTE: project action to action embedding
  assert action is not None, "action must be provided"
  action = rearrange(action, "b t d -> b 1 (t d)")
  action_emb_B_D = self.action_embedder_B_D(action)
  action_emb_B_3D = self.action_embedder_B_3D(action)
  ...
  # NOTE: add action embedding to the timestep embedding and adaln_lora
  t_embedding_B_T_D = t_embedding_B_T_D + action_emb_B_D
  adaln_lora_B_T_3D = adaln_lora_B_T_3D + action_emb_B_3D
  ```

  Same recipe as Predict1: MLP → added to timestep embedding and AdaLN-LoRA. The whole 12-step action
  chunk becomes one embedding.
- **GPUs and steps:** the launch command is `torchrun --nproc_per_node=2 --master_port=12341 -m
  scripts.train ... experiment="predict2_video2world_2b_action_conditioned_training"` with
  `fsdp_shard_size=-1` and `{"override /optimizer": "fusedadamw"}`. **Two GPUs.** Steps, batch size, and
  learning rate for this experiment are **not stated** in the doc (the doc shows only the `defaults`
  block); the inference example references `iter_000001000.pt`, i.e. a 1000-iteration checkpoint.
- **Post-training memory for the action model: not stated.**

**diffusers support.** `Cosmos2VideoToWorldPipeline` + `CosmosTransformer3DModel` + `AutoencoderKLWan`,
first in **diffusers v0.34.0** (2025-06-24).

---

## 3. NVIDIA Cosmos-Predict2.5 — 2B

**What it is / what changed vs Predict2.** Repo `https://github.com/nvidia-cosmos/cosmos-predict2.5`,
created 2025-09-25, code Apache-2.0. README, quoted:

> "We introduce Cosmos-Predict2.5, the latest version of the Cosmos World Foundation Models (WFMs)
> family, specialized for simulating and predicting the future state of the world in the form of video.
> Cosmos-Predict2.5 is a flow based model that **unifies Text2World, Image2World, and Video2World into a
> single model** and utilizes **Cosmos-Reason1**, a Physical AI reasoning vision language model (VLM),
> as the text encoder. Cosmos-Predict2.5 significantly improves upon Cosmos-Predict1 in both quality and
> prompt alignment."

Paper: arXiv 2511.00062 (linked from README; **not retrieved** — I did not download it). The repo carries
the same end-of-life banner pointing at Cosmos 3.

**Parameter count.** Named 2B and 14B. No integer count stated.

**VAE and text encoder — verified through diffusers.** `src/diffusers/pipelines/cosmos/pipeline_cosmos2_5_predict.py`
at `v0.37.0`:

```python
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from ...models import AutoencoderKLWan, CosmosTransformer3DModel
...
class Cosmos2_5_PredictBasePipeline(DiffusionPipeline):
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        vae: AutoencoderKLWan,
        scheduler: UniPCMultistepScheduler,
```

So: **Wan-class VAE** (same as Predict2), **Cosmos-Reason1 is a Qwen2.5-VL model** used as the text
encoder, and the sampler is UniPC.

**Objective — rectified flow, with the loss written out.** `docs/rectified-flow.md` (added
"[November 8, 2025] Added a new pedagogical README in docs/ detailing the Rectified Flow formulation and
its integration with the UniPC solver"), quoted:

> "Rectified flow describes a deterministic, straight-line evolution between a clean signal $x_0$ and a
> noise sample $\epsilon$. The sample at any intermediate time $t$ is given by
> `x(t) = \alpha(t) x_{0} + \sigma(t) \epsilon` … `v(t) \equiv \frac{dx}{dt} = \dot{\alpha}(t) x_{0} +
> \dot{\sigma}(t) \epsilon` … For rectified flow, the signal and noise amplitudes satisfy a
> straight-line constraint: `\alpha(t) + \sigma(t) = 1`".

(The doc derives the velocity target and the UniPC solver; it does not print the training loss as an
explicit expectation. The loss-as-written is therefore **partially stated**: velocity prediction under a
linear interpolant.)

**Native resolution and frame count.** `docs/diffusers_inference.md`: "`--num_output_frames` Sets output
length. Use `1` for image output and `>1` (default & recommended: `93`) for world (video) generation."
Model variants listed: `2B/pre-trained`, `2B/post-trained`, `2B/distilled` (text2world only),
`14B/pre-trained`, `14B/post-trained`; plus `2B/auto/multiview` (7-camera), `2B/robot/action-cond`,
`2B/robot/multiview-agibot` (3-camera), `2B/robot/policy`.

**Training data scale.** **Not stated** in the repo.

**Post-training docs and memory.** `docs/post-training.md` covers HF auth, output dirs, W&B, and DCP vs
consolidated checkpoints. Its example launch is `torchrun --nproc_per_node=8`. **There is no memory
table anywhere in `cosmos-predict2.5` docs.** `docs/setup.md` says only "NVIDIA GPUs with Ampere
architecture (RTX 30 Series, A100) or newer" and a note about `--ipc=host` shared memory. Memory
requirements: **not stated.**

**Action-conditioned post-training — and it is single-GPU.** `docs/post-training_video2world_action.md`,
same Bridge/IRASim dataset and same 7-dim action definition as Predict2. The launch command, quoted verbatim:

```bash
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320 ~dataloader_train.dataloaders
```

Config snippet:

```python
AC_REASON_EMBEDDINGS_RECTIFIED_FLOW_2B_256_320 = LazyDict(dict(
    defaults=[ DEFAULT_CHECKPOINT.experiment,
        {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
        {"override /net": "cosmos_v1_2B_action_conditioned"},
        {"override /conditioner": "action_conditioned_video_conditioner"},
        {"override /data_train": "bridge_13frame_480_640_train"},
        {"override /data_val": "bridge_13frame_480_640_val"}, "_self_"],
    job=dict(group="cosmos_predict_v2p5", name="2b_bridge_action_conditioned",
             project="cosmos_predict2_action_conditioned"),
    optimizer=dict(lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
                   weight_decay=0.1),
))
```

So: 13 frames, data bucket named `480_640`, experiment named `..._2b_256_320`, lr 3.05e-5, FSDP,
**`--nproc_per_node=1`**. Batch size and total steps are **not stated**. A DMD2 distillation recipe is
also published, also `--nproc_per_node=1`, experiment
`dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_256x320_no_s3`, and the doc
says "The figure below compares the student model produced by the above procedure (after 4,000 training
steps) with the teacher model it was distilled from."

**diffusers support.** `Cosmos2_5_PredictBasePipeline` and `Cosmos2_5_TransferPipeline` are absent at
`v0.36.0` and present at **`v0.37.0`** (published 2026-03-05). The repo's own news line dates the feature
earlier: "[December 19, 2025] Released Cosmos-Predict2.5-2B Diffusers support via Hugging Face" — i.e.
merged to `main` in Dec 2025, shipped in the v0.37.0 release.

**Licence.** README: "NVIDIA Cosmos source code is released under the Apache 2 License. NVIDIA Cosmos
models are released under the NVIDIA Open Model License."

---

## 4. Wan2.1 — T2V-1.3B

**What it is.** Paper arXiv 2503.20314 ("Wan: Open and Advanced Large-Scale Video Generative Models");
repo `https://github.com/Wan-Video/Wan2.1` created 2025-02-25, licence Apache-2.0 (GitHub API);
HF `Wan-AI/Wan2.1-T2V-1.3B` created `2025-02-25T11:13:48.000Z`, `license: apache-2.0`, not gated.

**Parameter count / architecture.** The 1.3B is named, not counted. Exact config from
`https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/raw/main/config.json`:

```json
{"_class_name": "WanModel", "_diffusers_version": "0.30.0", "dim": 1536, "eps": 1e-06,
 "ffn_dim": 8960, "freq_dim": 256, "in_dim": 16, "model_type": "t2v", "num_heads": 12,
 "num_layers": 30, "out_dim": 16, "text_len": 512}
```

and from `wan/configs/wan_t2v_1_3B.py`:

```python
t2v_1_3B.vae_stride = (4, 8, 8)
t2v_1_3B.patch_size = (1, 2, 2)
t2v_1_3B.dim = 1536 ; t2v_1_3B.ffn_dim = 8960 ; t2v_1_3B.freq_dim = 256
t2v_1_3B.num_heads = 12 ; t2v_1_3B.num_layers = 30
t2v_1_3B.qk_norm = True ; t2v_1_3B.cross_attn_norm = True ; t2v_1_3B.eps = 1e-6
t2v_1_3B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
t2v_1_3B.t5_tokenizer = 'google/umt5-xxl'
```

`wan/configs/shared_config.py`: `t5_model = 'umt5_xxl'`, `t5_dtype = torch.bfloat16`, `text_len = 512`,
`param_dtype = torch.bfloat16`, `num_train_timesteps = 1000`, `sample_fps = 16`. Layers 30, dim 1536,
heads 12, **patch size (1,2,2)**, **in_channels 16**.

**Wan-VAE.** Paper Fig. 5 caption: "Our Wan-VAE Framework. Wan-VAE can compress the spatio-temporal
dimension of a video by 4 × 8 × 8 times." §: "we design a 3D causal VAE"; "For a given video
V ∈ ℝ^{(1+T)×H×W×3}, the encoder of Wan-VAE encodes it from pixel space into the latent space
x ∈ ℝ^{1+T/4 × H/8 × W/8}"; "most models … adopt the same compression rate and latent feature dimension
as our approach, i.e., a compression rate of 4 × 8 × 8 with a **latent dimension of 16**". Chunking:
"the number of video sequence frames follows the 1 + T input format, so we divide the video into
1 + T/4 chunks … the number of frames in each processing chunk is limited to at most 4, effectively
preventing memory overflow." Diffusers `AutoencoderKLWan` defaults confirm: `z_dim: int = 16`,
`temperal_downsample: List[bool] = [False, True, True]` (two temporal halvings = 4×), `base_dim: int = 96`,
`dim_mult: Tuple[int] = [1, 2, 4, 4]`, plus hard-coded `latents_mean` / `latents_std` of length 16.

**Patchify.** Paper: "In the patchifying module, we use a 3D convolution with a kernel size of (1, 2, 2)
and apply a flattening operation to convert x into a sequence of features with the shape of (B, L, D),
where … L = (1 + T/4) × H/16 × W/16".

**Objective — flow matching.** "We leverage the flow matching framework (Lipman et al., 2022; Esser et
al., 2024) to model a unified denoising diffusion process across both image and video domains. During
training, given an image or video latent x₁, a random noise x₀ ∼ 𝒩(0, I), and a timestep t ∈ [0,1]
sampled from a **logit-normal distribution**, an intermediate latent x_t is obtained as the training
input", with "u(x_t, c_txt, t; θ) denotes the output velocity predicted by the model" and "c_txt is the
umT5 text embedding sequence of 512 tokens long".

**Text encoder, and bypassing it.** "Wan's architecture uses the umT5 (Chung et al., 2023) to encode
input text." Ablation: "We select three text encoders that can handle bilingual inputs: umT5
(Chung et al., 2023) (**5.3B**), Qwen2.5-7B-Instruct, and GLM-4-9B". There is **no statement** in the
paper or repo about running with a fixed/null text embedding; the CLI offers `--t5_cpu` (offload to CPU)
only. Bypassability: **not stated** (though `sample_neg_prompt` shows the pipeline already handles a
fixed negative embedding).

**Conditioning frames.** T2V-1.3B is text-only. I2V is a 14B-only product (see below), and the paper's
I2V description is not part of the 1.3B card.

**Native resolution and frame count.** README model table: `| T2V-1.3B | … | Supports 480P`. Caveat,
quoted: "The 1.3B model is capable of generating videos at 720P resolution. However, due to limited
training at this resolution, the results are generally less stable compared to 480P. For optimal
performance, we recommend using 480P resolution." Example command uses `--size 832*480`. `sample_fps = 16`.

**Training data scale.** "the 14B model of Wan, trained on a vast dataset comprising billions of images
and videos"; "Wan has seen large-scale data comprising billions of images and videos, amounting to
𝒪(1) trillions of tokens in total". No per-model breakdown for the 1.3B.

**Inference memory.** "The T2V-1.3B model requires only 8.19 GB VRAM, making it compatible with almost
all consumer-grade GPUs. It can generate a 5-second 480P video on an RTX 4090 in about 4 minutes
(without optimization techniques like quantization)."

**Published fine-tuning recipe.** **None.** The official repo tree (59 files) contains `generate.py`,
`wan/text2video.py`, `wan/image2video.py`, `wan/first_last_frame2video.py`, `wan/vace.py` and gradio
demos — **no training script**. Training-memory requirements: **not stated.**

**Action-conditioned variant.** None official. (Third-party derivatives are listed in the README —
DriVerse, Wan-Move, ATI, Phantom, etc. — but those are other people's repos.)

**diffusers support.** `WanTransformer3DModel` (in
`src/diffusers/models/transformers/transformer_wan.py`) and `AutoencoderKLWan` (in
`src/diffusers/models/autoencoders/autoencoder_kl_wan.py`) are **404 at `v0.32.2` and 200 at `v0.33.0`**
→ added in **diffusers v0.33.0**, published **2025-04-09**. The v0.33.0 release notes carry a "### Wan
2.1" section listing `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, `Wan-AI/Wan2.1-T2V-14B-Diffusers`,
`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`, `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`. The README also dates it:
"Mar 3, 2025: Wan2.1's T2V and I2V have been integrated into Diffusers".

**Is there an official 1.3B image-to-video?** **No.** The README "Model Download" table lists exactly:
T2V-14B (480P+720P), I2V-14B-720P, I2V-14B-480P, **T2V-1.3B (480P only)**, FLF2V-14B (720P),
VACE-1.3B (480P), VACE-14B. The Diffusers model list above only names I2V at 14B. Every I2V checkpoint
is 14B; the only other 1.3B is VACE-1.3B.

**Licence.** "The models in this repository are licensed under the Apache 2.0 License. We claim no rights
over the your generated contents".

---

## 5. SkyReels-V2 — DF-1.3B-540P and I2V-1.3B-540P

**arXiv ID correction:** **2504.13074**, not 2505.13211. Title from the HTML: "SkyReels-V2:
Infinite-length Film Generative model". Repo `https://github.com/SkyworkAI/SkyReels-V2` created
2025-04-15.

**What it is.** Abstract: "we propose SkyReels-V2, an Infinite-length Film Generative Model, that
synergizes Multi-modal Large Language Model (MLLM), Multi-stage Pretraining, Reinforcement Learning, and
Diffusion Forcing Framework."

**Base architecture — it is a Wan2.1 clone.** "We adopt the model architecture from Wan2.1 [5] and only
train the DiT from scratch while retaining the pretrained weight of other components including VAE and
text encoder. Then, we also use the **Flow Matching** framework [35, 24] to train our video generation
model." And: "Given a latent representation x₁ (image or video), we sample a timestep t ∈ [0,1] from a
**logit-normal distribution** [24]. Then, initialize noise x₀ ∼ 𝒩(0, I). and construct the intermediate
latent x_t via linear interpolation". So VAE = Wan-VAE (4×8×8, 16 channels), text encoder = umT5 — by
inheritance, stated in the paper, not re-specified.

**Parameter count.** "SkyCaptioner-V1 and SkyReels-V2 series models including diffusion-forcing,
text2video, image2video, camera director and elements2video models with various sizes (1.3B, 5B, 14B)
are open-sourced." No integer count.

**How diffusion forcing is trained — per-frame timesteps under a non-decreasing constraint.** §3.4.2:

> "Diffusion Forcing [50] is a training and sampling strategy where **each token is assigned an
> independent noise level**. This allows tokens to be denoised according to arbitrary, per-token schedules
> using the trained model. Conceptually, this approach functions as a form of partial masking: a token
> with zero noise is fully unmasked, while complete noise fully masks it. … Note that the synchronous
> full sequence diffusion is a special case of Diffusion Forcing, where all tokens share the same noise
> level. **This relationship allows us to fine-tune the Diffusion Forcing Transformer from a full-sequence
> diffusion model.**"

The timestep sampler is FoPP: "Inspired by AR-Diffusion [11], we utilize the **Frame-oriented Probability
Propagation (FoPP)** timestep scheduler for Diffusion Forcing Training. The process involves the
following steps: 1. **Uniform Sampling**: First, we uniformly sample a frame index f ∼ U(1,F) and a
corresponding timestep t ∼ U(1,T). … 2. **Dynamic Programming for Probability Propagation** … 3. …
d_{i,j} = d_{i,j−1} + d_{i−1,j} with boundary conditions d_{*,T}=1 and d_{F,*}=1. … 5. **Timestep
Sampling**: Finally, timesteps for previous or subsequent frames are sampled one by one based on the
calculated probabilities." The abstract frames the point: "Our diffusion forcing framework with
**non-decreasing noise schedules** enables long-video synthesis in an efficient search space", and the
intro quantifies it: "we use the non-decreasing noise schedule in the consecutive frames, which
significantly reduce the Composition Space Size from O(1e48) to O(1e32)."

Also: "our post training consists of four stages: high-quality SFT in 540p, Reinforcement Learning,
Diffusion Forcing Training, and high-quality SFT in 720p. … we propose the diffusion forcing training
stage, in which we transform the full-sequence diffusion model into a diffusion forcing model that
applies **frame-specific noise levels** such that enables a various length video generation ability."

**How inference extends video.** "During inference, we adapt an **Adaptive Difference (AD)** timestep
scheduler [11] … The AD scheduler treats the timestep difference between neighboring frames as an
adaptive variable s. … Notably, synchronous diffusion (s = 0) and auto-regressive generation (s = T) are
special cases. A smaller s yields more similar neighboring frames, while a larger s increases content
variability." And: "our Diffusion Forcing Transformer can extend video generation indefinitely based on
the last frames of the previous segment." Causal attention + KV cache: "bidirectional attention is
unnecessary and can be replaced with more efficient causal attention. After training the Diffusion
Forcing Transformer with bidirectional attention, one can fine-tune the model with context-causal
attention for enhanced efficiency. During inference, this architecture enables caching of K, V features
from historical samples".

Concrete inference flags from the README: `--resolution 540P --ar_step 0 --base_num_frames 97
--overlap_history 17` (synchronous) or `--ar_step 5` with `--causal_block_size 5` (asynchronous).
"You can use `--ar_step 5` to enable asynchronous inference. … Asynchronous inference will take more
steps to diffuse the whole sequence which means it will be SLOWER than synchronous mode. In our
experiments, asynchronous inference may improve the instruction following and visual consistent
performance." Diffusers snippet uses `base_num_frames=97,  # 121 for 720P` and `overlap_history=None, #
… 17 for long video generations`.

**I2V-1.3B exists, and how it conditions.** Model tables list **SkyReels-V2-DF-1.3B-540P** and
**SkyReels-V2-I2V-1.3B-540P** (HF API: both `Skywork/...`, `gated: False`, created 2025-04-18 and
2025-04-20). Two I2V routes, quoted: "1) **Fine-Tuning full-sequence Text-to-Video (T2V) diffusion
Models (SkyReels-V2-I2V)**: Following Wan 2.1's I2V implementation, we extend T2V architectures by
injecting the first reference frame as an image condition. The input image is padded to match the target
video length, then processed through a VAE encoder to obtain image latents. These latents are
concatenated with noise latents and **4 binary mask channels** (1 for the reference frame, 0 for
subsequent frames) … we apply zero-initialization to newly added convolutional layers … **this approach
achieves competitive results with only 10,000 training iterations on 384 GPUs.** 2) **Text-to-Video (T2V)
Diffusion Forcing model with first-frame Conditioning (SkyReels-V2-DF)**: Our alternative method directly
utilizes the diffusion framework's conditioning mechanism by feeding the first frame as a clean reference
condition. This bypasses explicit model retraining".

**Resolution.** 540P (and 720P for 14B DF). "We further scale up the resolution to 540p in this final
pretraining stage, focusing exclusively on video objectives."

**Optimiser.** "we employ the AdamW optimizer throughout all pretraining stages. In Stage 1, we initialize
the learning rate at 1e-4 with weight decay set to 0. Once the loss converges to a stable range, we adjust
the learning rate to 5e-5 and introduce weight decay at 1e-4. In Stages 2 and 3, we further reduce the
learning rate to 2e-5."

**Inference memory.** README: "Generating a 540P video using the **1.3B model requires approximately
14.7GB peak VRAM**, while the same resolution video using the 14B model demands around 51.2GB peak
VRAM." And: "To reduce peak VRAM, just lower the `--base_num_frames`, e.g., to 77 or 57, while keeping
the same generative length `--num_frames` you want to generate."

**Training code.** **None released.** The repo tree (42 files) is `generate_video.py`,
`generate_video_df.py`, `skyreels_v2_infer/...` (note the `_infer` package name), `skycaptioner_v1/...`.
Training memory requirements: **not stated.**

**diffusers support.** `src/diffusers/models/transformers/transformer_skyreels_v2.py` and
`src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing.py` are **404 at `v0.34.0`,
200 at `v0.35.0`** → added in **diffusers v0.35.0**, published **2025-08-19**. Exact exported names from
`src/diffusers/pipelines/skyreels_v2/__init__.py` at v0.35.0:
`SkyReelsV2Pipeline`, `SkyReelsV2DiffusionForcingPipeline`, `SkyReelsV2DiffusionForcingImageToVideoPipeline`,
`SkyReelsV2DiffusionForcingVideoToVideoPipeline`, `SkyReelsV2ImageToVideoPipeline`; model class
`SkyReelsV2Transformer3DModel`.

**Licence.** HF API `cardData` for both 1.3B repos: `license: other`, `license_name: skywork-license`.
GitHub API reports the repo licence as `NOASSERTION`; `LICENSE.txt` begins with a YAML header
`license: other`. **Not Apache.**

---

## 6. Matrix-Game 2.0

**What it is.** arXiv 2508.13009 (title confirmed from the HTML `<title>`: "Matrix-Game 2.0: An
Open-Source, Real-Time, and Streaming Interactive World Model"). Repo
`https://github.com/SkyworkAI/Matrix-Game` (MIT per GitHub API), code under `Matrix-Game-2/`.
HF `Skywork/Matrix-Game-2.0` created `2025-08-08T05:48:53.000Z`, `license: mit`, not gated.

**Is training code released?** **No — inference only.** The full `Matrix-Game-2/` tree contains
`inference.py`, `inference_streaming.py`, `pipeline/causal_inference.py`, `wan/...`, `demo_utils/...`,
`utils/...`, configs and demo images. There is no train script, no optimiser config, no dataset loader.
The README's Quick Start shows only the two inference entry points. So the action-conditioning *design*
is reusable; the *recipe* is not runnable from the repo.

**Exact base checkpoint.** The HF model card front-matter says it outright:

```yaml
license: mit
pipeline_tag: image-to-video
library_name: diffusers
base_model:
- Skywork/SkyReels-V2-I2V-1.3B-540P
```

The paper agrees: "For training the foundation model, we initialize our model with
**SkyReels-V2-I2V-1.3B [8]**, which follows Wan 2.1 [44] architecture. The 1.3B variant provides an
optimal balance between generation quality and computational efficiency … **We remove the text injection
modules from the released checkpoint.** To stabilize the whole training process, we firstly fine-tune the
model for 5k steps. After that, **action modules are added into each DiT block, leading to the total
model size as 1.8B**. We train the foundation model for **120k steps with learning rate=2e-5, batch
size=256**." Model card: "Matrix-Game-2.0(1.8B) is derived from the Wan. By removing the text branch and
adding action modules, the model predicts next frames only from visual contents and corresponding
actions." Acknowledgements name "[SkyReels-V2] for their strong base model" and "[GameFactory] for their
idea of action control module".

**Distillation recipe.** "For distillation, we firstly collect **40k ODE pairs** and fine-tune the causal
student model for **6k steps**, with subsequent **4k training steps** via DMD-based Self-Forcing. The
learning rate is **6e-6**. The chunk size of latent frames and attention local size are set as **3 and 6**,
respectively."

**Action modules in the released code.** File `Matrix-Game-2/wan/modules/action_module.py` (531 lines),
class `ActionModule`, instantiated per DiT block in `Matrix-Game-2/wan/modules/causal_model.py`:
`self.action_model = ActionModule(**action_config, local_attn_size=self.local_attn_size)`, called inside
`cross_attn_ffn` as
`x = self.action_model(x.to(context.dtype), grid_sizes[0], grid_sizes[1], grid_sizes[2], mouse_cond,
keyboard_cond, ..., is_causal=True, kv_cache_mouse=kv_cache_mouse, kv_cache_keyboard=kv_cache_keyboard,
start_frame=start_frame, ..., num_frame_per_block=num_frame_per_block)`.

`Matrix-Game-2/configs/foundation_model/config.json`, verbatim:

```json
{"_class_name": "CausalWanModel", "_diffusers_version": "0.35.0.dev0",
 "action_config": {"blocks": [0,1,...,29], "enable_keyboard": true, "enable_mouse": true,
   "heads_num": 16, "hidden_size": 128, "img_hidden_size": 1536,
   "keyboard_dim_in": 4, "keyboard_hidden_dim": 1024,
   "mouse_dim_in": 2, "mouse_hidden_dim": 1024,
   "mouse_qk_dim_list": [8,28,28], "patch_size": [1,2,2], "qk_norm": true, "qkv_bias": false,
   "rope_dim_list": [8,28,28], "rope_theta": 256,
   "vae_time_compression_ratio": 4, "windows_size": 3},
 "dim": 1536, "eps": 1e-06, "ffn_dim": 8960, "freq_dim": 256, "in_dim": 36,
 "inject_sample_info": false, "local_attn_size": -1, "model_type": "i2v",
 "num_heads": 12, "num_layers": 30, "out_dim": 16, "sink_size": 0, "text_len": 512}
```

So the trunk is Wan-1.3B-shaped (`dim 1536`, `num_layers 30`, `num_heads 12`, `ffn_dim 8960`) with
`in_dim 36` for I2V, and **action modules in all 30 blocks**.

**Mouse and keyboard, per frame or per chunk.** Both conditions arrive **per raw frame** and are grouped
into per-latent-frame windows. `ActionModule.forward` signature and docstring:

```python
def forward(self, x, tt, th, tw, mouse_condition=None, keyboard_condition=None, ...,
            is_causal=False, kv_cache_mouse=None, kv_cache_keyboard=None, start_frame=0,
            use_rope_keyboard=True, num_frame_per_block=3):
    """
    mouse_condition: B, N_frames, C1
    keyboard_condition: B, N_frames, C2
    """
```

- **Mouse (2-dim per frame)** is concatenated with the image hidden state and run through an MLP, then
  self-attended over time: `self.mouse_mlp = torch.nn.Sequential(torch.nn.Linear(mouse_dim_in *
  vae_time_compression_ratio * windows_size + img_hidden_size, c, bias=True), ...)`, with grouping
  `group_mouse = [mouse_condition[:, self.vae_time_compression_ratio*(i - self.windows_size) + pad_t : i
  * self.vae_time_compression_ratio + pad_t, :] for i in range(num_frame_per_block)]`, then
  `mouse_qkv = self.t_qkv(group_mouse)` and `self.proj_mouse = nn.Linear(c, img_hidden_size, ...)`.
  So each latent frame sees `windows_size(3) × vae_time_compression_ratio(4) = 12` raw mouse samples.
- **Keyboard (4-dim per frame)** is embedded and enters as cross-attention keys/values with the image as
  query: `self.keyboard_embed = nn.Sequential(nn.Linear(keyboard_dim_in, hidden_size, bias=True),
  nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))`,
  `self.mouse_attn_q = nn.Linear(img_hidden_size, keyboard_hidden_dim, bias=qkv_bias)`,
  `self.keyboard_attn_kv = nn.Linear(hidden_size * windows_size * vae_time_compression_ratio,
  keyboard_hidden_dim * 2, bias=qkv_bias)`, `self.proj_keyboard = nn.Linear(keyboard_hidden_dim,
  img_hidden_size, bias=qkv_bias)`.

The conditioning is sliced per autoregressive block in `pipeline/causal_inference.py`:
`new_cond["keyboard_cond"] = conditional_dict["keyboard_cond"][:, : 1 + 4 * (current_start_frame +
num_frame_per_block - 1)]`.

**Resolution and frames per chunk.** Paper: "All the videos are resized to **352 × 640** resolution" and
"the current 352×640 resolution output falls short of state-of-the-art video generation models". The
inference config `configs/inference_yaml/inference_universal.yaml`:

```yaml
image_or_video_shape: [1, 16, 15, 44, 80]
num_frame_per_block: 3
denoising_step_list: [1000, 666, 333]
warp_denoising_step: true
causal: true
model_kwargs: {timestep_shift: 5.0, model_config: configs/distilled_model/universal}
```

i.e. batch 1, 16 latent channels, **15 latent frames**, 44×80 latent spatial, **3 latent frames per
autoregressive block**, **3 denoising steps**.

**KV cache size.** `pipeline/causal_inference.py`: `self.frame_seq_length = 880`, and

```python
if self.local_attn_size != -1:
    kv_cache_size = self.local_attn_size * self.frame_seq_length
else:
    kv_cache_size = 15 * 1 * self.frame_seq_length # 32760
```

with per-block tensors, and for the action caches `kv_cache_size = 15 * 1` with
`torch.zeros([batch_size, kv_cache_size, 16, 64], ...)`. In `wan/modules/causal_model.py`:
`self.max_attention_size = 15 * 1 * 880 if local_attn_size == -1 else local_attn_size * 880`.
(**Source inconsistency flagged:** the inline comment says `# 32760` but 15 × 880 = 13200. The comment is
stale; the computed value is 13200 tokens.) The distilled configs set a finite `local_attn_size`, and the
paper says "The chunk size of latent frames and attention local size are set as 3 and 6" and "we
constrain the KV-cache window size. This forces the model to rely more on its learned priors".

**Training data.** "about **800-hour** action-annotated video at **360p** resolution in total. The data
includes 153-hour Minecraft video data and 615-hour Unreal Engine data, arranged into **57 frames** for
each video clip. … we utilize the open-source Sekai dataset [24], obtaining an additional 85 hours …
we further collect 574-hour GTA-driver data and 560-hour Temple Run game data". The abstract instead says
"~1200 hours"; both figures are in the same paper.

**VAE.** "A 3D Causal VAE [55, 20] is first employed to compress raw video data along both spatial and
temporal dimensions — by a factor of **8 × 8 in space and 4 in time**"; "we integrated the efficient
**Wan2.1-VAE** architecture with caching mechanism". Image also encoded by CLIP: "The image input is
encoded by 3D VAE encoder and the CLIP image encoder [33] as condition input."

**Hardware stated.** Repo README "Requirements": "Nvidia GPU with at least **24 GB memory** (A100, and
H100 are tested). Linux operating system. 64 GB RAM." This is an **inference** figure. Real-time claim:
"achieving 25 FPS generation on a single H100 GPU".

**Dependencies.** "install apex and FlashAttention … Our project also depends on FlashAttention".

---

## 7. Hunyuan-GameCraft-1.0

**What it is.** Repo `https://github.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0`, created 2025-08-13,
GitHub licence `NOASSERTION`. HF `tencent/Hunyuan-GameCraft-1.0` created `2025-08-13T07:10:08.000Z`,
HF API `cardData` has **no `license` field** (`license: None`), though a `LICENSE` and `Notice.txt` file
exist in the repo file list. Paper arXiv 2506.17201, "Hunyuan-GameCraft: High-dynamic Interactive Game
Video Generation with Hybrid History Condition".

**Base model size as stated on the model card.** **Not stated.** The model card
(`huggingface.co/tencent/Hunyuan-GameCraft-1.0/raw/main/README.md`) contains no parameter count and no
"13B"/"B parameters" string anywhere; its only base-model reference is the acknowledgement line
"We would like to thank the contributors to the [HunyuanVideo] …". The **paper** names the base without
sizing it: "Built upon a text-to-video foundation model, **HunyuanVideo [18]**", and "a high-dynamic
interactive game video generation model based on a previously open-sourced **MM-DiT [9] based
text-to-video model, HunyuanVideo [18]**". A parameter count is **not stated** in either source.

**VAE.** Paper: the model uses a **causal VAE** — "the causal VAE's uneven encoding of initial versus
subsequent frames fundamentally limits efficiency and scalability"; "The chunk latent, serving as a
global representation by causal VAE, is subsequently decoded into a temporally consistent video segment".
The shipped weights confirm HunyuanVideo's own VAE: `weights/README.md` file tree shows
`stdmodels/vae_3d/hyvae/pytorch_model.pt` + `config.json`, alongside
`stdmodels/llava-llama-3-8b-v1_1-transformers/` and `stdmodels/openai_clip-vit-large-patch14/`. Exact
channel/compression numbers: **not stated** in the repo or the card.

**Text encoder.** Not named in prose, but the weights tree is decisive: **LLaVA-Llama-3-8B** plus
**openai/clip-vit-large-patch14** (the HunyuanVideo encoder pair).

**Objective.** Paper: "the preceding head condition remains noise-free as clean latent, which guides
subsequent noisy chunk latents through **flow matching** to progressively denoise". The repo ships
`hymm_sp/diffusion/schedulers/scheduling_flow_match_discrete.py`.

**How conditioning frames enter — hybrid history condition.** "we define each autoregressive step as a
chunk latent denoising process guided by head latent and interactive signals. … **Head condition can be
different forms, including (i) a single image frame latent, (ii) the final latent from the previous clip,
or (iii) a longer latent clip segment.** Hunyuan-GameCraft achieves high-fidelity denoising of chunk
latents through **concatenation at both condition and noise levels. An additional binary mask assigns
value 1 to head latent regions and 0 to chunk segments**, enabling precise control over the denoising
part." Training mixes them: "The hybrid history condition maintains specific ratios: 0.7 for single
historical clip, 0.05 for multiple historical clips, and 0.25 for single frame."

**Actions.** "we unify standard keyboard and mouse inputs into a **shared camera representation space**";
"we eliminate the degree of freedom in the roll dimension while incorporating velocity control … this
representation can be seamlessly converted into standard camera trajectory parameters and **Plücker
embeddings**." The module is `hymm_sp/modules/cameranet.py`.

**Resolution / frames / fps.** "The system operates at **25 fps**, with each video chunk comprising
**33-frame clips at 720p resolution**." README inference examples use `704px1216p`.

**Minimum GPU memory for inference.** Model card and repo README, quoted: "An NVIDIA GPU with CUDA support
is required. The model is tested on a machine with 8GPUs. **Minimum: The minimum GPU memory required is
24GB but very slow. Recommended: We recommend using a GPU with 80GB of memory for better generation
quality.**" And in the usage section: "to generate a video with 1 GPU with Low-VRAM (**minimum GPU memory
required is 24GB for 704px1216p but very slow**)". There is a `scripts/run_sample_batch_4090.sh`.

**Training / fine-tuning code.** **None.** The repo's Python tree is entirely inference:
`hymm_sp/inference.py`, `hymm_sp/sample_batch.py`, `hymm_sp/sample_inference.py`,
`hymm_sp/diffusion/pipelines/pipeline_hunyuan_video_game.py`, gradio apps, and four `scripts/*.sh` that
are all sampling scripts. Training memory requirements: **not stated.**

**Published training recipe (paper only, not runnable).** "The experiments employ **full-parameter
training on 192 NVIDIA H20 GPUs**, conducted in two phases with a **batch size of 48**. The first phase
trains the model for **30k iterations at a learning rate of 3 × 10⁻⁵** using all collected game data and
synthetic data at their original proportions. The second phase introduces data augmentation techniques …
while reducing the **learning rate to 1 × 10⁻⁵ for an additional 20,000 iterations**." Data: "over one
million gameplay recordings across over 100 AAA games". Distillation: "we adopt the Phased Consistency
Model (PCM) [28], which distills the original diffusion process and classifier-free guidance into a
compact **eight-step** consistency model", giving "a 10–20× acceleration in inference speed, reducing
latency to less than 5s per action".

**diffusers support.** Not claimed by the repo; **not stated** (the repo ships its own pipeline class).

---

## 8. CogVideoX-2B

**What it is.** arXiv 2408.06072, "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer".
HF `THUDM/CogVideoX-2b`, created `2024-08-05T14:13:31.000Z`, `license: apache-2.0`, `library_name:
diffusers`, not gated. Code repo `THUDM/CogVideo` (now `zai-org/CogVideo`, Apache-2.0).

**Licence — 2B vs 5B differ, confirmed.** HF API: `THUDM/CogVideoX-2b` → `license: apache-2.0`;
`THUDM/CogVideoX-5b` → `license: other`. The 2B card says: "This model is released under the
[Apache 2.0 License]" while for the 5B "please refer to the [CogVideoX LICENSE]".

**VAE specs.** Paper §2.1: "we design a **3D causal VAE** to compress the video into the latent space";
"This enables the 3D VAE to achieve a **4× compression in the temporal dimension and an 8 × 8 compression
in the spatial dimension. In total, this achieves a 4 × 8 × 8 compression from pixels to the latents.**"
`huggingface.co/THUDM/CogVideoX-2b/raw/main/vae/config.json`:

```json
{"_class_name": "AutoencoderKLCogVideoX", "block_out_channels": [128, 256, 256, 512],
 "in_channels": 3, "latent_channels": 16, "out_channels": 3,
 "scaling_factor": 1.15258426, "temporal_compression_ratio": 4}
```

**Latent channels = 16, temporal compression = 4.**

**Transformer config** (`transformer/config.json`): `"_class_name": "CogVideoXTransformer3DModel"`,
`"_diffusers_version": "0.30.0.dev0"`, `num_layers: 30`, `num_attention_heads: 30`,
`attention_head_dim: 64`, `in_channels: 16`, `out_channels: 16`, `patch_size: 2`,
`sample_frames: 49`, `sample_height: 60`, `sample_width: 90`, `text_embed_dim: 4096`,
`max_text_seq_length: 226`, `use_rotary_positional_embeddings: false`.

**Text encoder.** "we encode the textual input into text embeddings z_text using T5 (Raffel et al.,
2020)". Bypass with a fixed embedding: **not stated**.

**Native resolution and frame count.** Card: "Video Resolution: **720 x 480, no support for other
resolutions (including fine-tuning)**"; "**8 Frames per Second**"; diffusers example uses
`num_frames=49`.

**Inference VRAM (card table).** 2B column: "SAT FP16: 18GB / diffusers FP16: starting from 4GB* /
diffusers INT8(torchao): starting from 3.6GB*"; multi-GPU "FP16: 10GB* using diffusers". Caveat quoted:
"this solution has not been tested for actual VRAM/memory usage on devices other than NVIDIA A100 / H100
… with [optimizations] disabled, VRAM usage will increase significantly, with peak VRAM usage being about
3 times higher than the table".

**Fine-tuning memory — two conflicting published tables.** The model card says, for the 2B:

| Fine-tuning VRAM Consumption (per GPU) — CogVideoX-2B |
|---|
| 47 GB (bs=1, LORA) |
| 61 GB (bs=2, LORA) |
| 62 GB (bs=1, SFT) |

The repo's own finetune doc (`finetune/README.md` in `zai-org/CogVideo`) gives much lower numbers:

| Model | Type | Strategy | Precision | Resolution | Memory |
|---|---|---|---|---|---|
| cogvideox-t2v-2b | lora (rank128) | DDP | fp16 | 49x480x720 | **16GB VRAM (NVIDIA 4080)** |
| cogvideox-t2v-2b | sft | DDP | fp16 | 49x480x720 | **36GB VRAM (NVIDIA A100)** |
| cogvideox-t2v-2b | sft | 1-GPU zero-2 + opt offload | fp16 | 49x480x720 | **17GB VRAM (NVIDIA 4090)** |
| cogvideox-t2v-2b | sft | 8-GPU zero-2 | fp16 | 49x480x720 | 17GB VRAM (NVIDIA 4090) |
| cogvideox-t2v-2b | sft | 8-GPU zero-3 | fp16 | 49x480x720 | 19GB VRAM (NVIDIA 4090) |
| cogvideox-t2v-2b | sft | 8-GPU zero-3 + opt and param offload | bf16 | 49x480x720 | 14GB VRAM (NVIDIA 4080) |

**These two sources disagree by ~3× for the same model.** Both are quoted verbatim; I did not resolve
which is current. The finetune doc adds: "For SFT training, model offload is not used during validation,
so the peak VRAM usage may exceed 24GB. For GPUs with less than 24GB VRAM, it's recommended to disable
validation."

**Conditioning frames.** CogVideoX-2b is text-to-video only; an image-to-video variant exists only at 5B
(`cogvideox-{t2v, i2v}-5b` rows in the finetune table). Frame conditioning for the 2B: **not stated**.

**Training data scale.** **Not retrieved** (I grepped the paper for data-scale statements and did not
capture one; the paper has a data pipeline section I did not read in full).

**diffusers support.** `src/diffusers/models/transformers/cogvideox_transformer_3d.py` is **404 at
`v0.29.2`, 200 at `v0.30.0`** → added in **diffusers v0.30.0**, published **2024-08-07**. Classes:
`CogVideoXTransformer3DModel`, `AutoencoderKLCogVideoX`.

---

## 9. LTX-Video

**What it is.** arXiv 2501.00103, "LTX-Video: Realtime Video Latent Diffusion". Repo
`https://github.com/Lightricks/LTX-Video`, created 2024-11-20, GitHub API reports the repo licence as
Apache-2.0 (that is the *code*). HF `Lightricks/LTX-Video` created `2024-10-31T12:36:00.000Z`,
`license: other`, `library_name: diffusers`, not gated.

**VAE compression ratios — 32 × 32 × 8, confirmed.** Abstract: "At its core is a carefully designed
Video-VAE that achieves a **high compression ratio of 1:192, with spatiotemporal downscaling of
32 × 32 × 8 pixels per token**, enabled by relocating the patchifying operation from the transformer's
input to the VAE's input." §: "our Video-VAE applies a spatio-temporal compression of 32 × 32 × 8 with
**128 channels**, resulting in a total compression of 1:192 (twice the typical compression) and a
pixels-to-tokens ratio of 1:8192 (four times the typical ratio), **without requiring a patchifier**."
And: "our approach features a carefully designed VAE architecture that achieves higher spatial
compression while maintaining video quality through an **increased latent depth of 128 channels**".

`huggingface.co/Lightricks/LTX-Video/raw/main/vae/config.json`:

```json
{"_class_name": "AutoencoderKLLTXVideo", "latent_channels": 128, "patch_size": 4, "patch_size_t": 1,
 "block_out_channels": [128, 256, 512, 512], "encoder_causal": true, "decoder_causal": false,
 "in_channels": 3, "out_channels": 3, "scaling_factor": 1.0}
```

**Latent channels = 128.** Encoder is causal, decoder is not. The decoder also denoises: "we propose
tasking the VAE decoder with performing the last denoising step in conjunction with converting latents to
pixels."

**2B model.** The README model table lists `ltxv-2b-0.9.6`, `ltxv-2b-0.9.6-distilled`,
`ltxv-2b-0.9.8-distilled`, `ltxv-2b-0.9.8-distilled-fp8`, alongside 13B variants. Note the 2B 0.9.8 is
described as distilled *from the 13B*: "Both models are distilled from the same base model
[ltxv-13b-0.9.8-dev] and are compatible for use together in the same multiscale pipeline." Parameter
counts beyond the "2B"/"13B" names: **not stated**.

**Licence — per-checkpoint, and it changed over versions.** The card lists a separate licence link per
version: "2B version 0.9 [license]", "2B version 0.9.1", "2B version 0.9.5", then from 0.9.6 onward
everything points at `LTX-Video-Open-Weights-License-0.X.txt`. The repo README notes for 0.9.5: "New
license for commercial use (**OpenRail-M**)". HF `license: other`. **Weights are not Apache-2.0**;
they are under Lightricks' own open-weights licence, version-dependent.

**Resolution / frame constraints.** Card: "The model works on resolutions that are divisible by 32 and
number of frames that are divisible by 8 + 1 (e.g. 257). In case the resolution or number of frames are
not divisible by 32 or 8 + 1, the input will be padded with -1 and then cropped". The diffusers example
uses `expected_height, expected_width = 480, 832`. README: "It can generate up to 50 FPS videos at native
4K resolution with synchronized audio in one pass."

**Objective / text encoder / conditioning.** **Not retrieved** in the depth the other cards got — I read
the abstract, intro, and §on the VAE, not the training objective section. README says the model supports
"image-to-video, multi-keyframe conditioning, keyframe-based animation, video extension (both forward and
backward), video-to-video transformations".

**Fine-tuning.** The README has a "Training" section in its table of contents; I did not retrieve its
contents. Training memory requirements: **not stated** in what I read. Inference VRAM is mentioned only
for the distilled LoRA: "Requires only 1GB of VRAM".

**diffusers support.** `src/diffusers/models/transformers/transformer_ltx.py` is **404 at `v0.31.0`,
200 at `v0.32.0`** → added in **diffusers v0.32.0**, published **2024-12-23**. Classes:
`LTXVideoTransformer3DModel`, `AutoencoderKLLTXVideo`, `LTXPipeline` / `LTXImageToVideoPipeline`. The
repo also says: "Support loading checkpoints of LTX-Video in Diffusers format (conversion is done
on-the-fly)".

---

## 10. Open-Sora 1.1 — STDiT-2

**What it is.** `docs/report_02.md` in `hpcaitech/Open-Sora` (repo created 2024-02-20, Apache-2.0;
`v1.1.0` tag exists). Opening line: "In Open-Sora 1.1 release, we train a **700M** models on **10M** data
(Open-Sora 1.0 trained on 400K data) with a better STDiT architecture." Announced "**[2024.04.25]** We
released **Open-Sora 1.1**, which supports **2s~15s, 144p to 720p, any aspect ratio** text-to-image,
text-to-video, image-to-video, video-to-video, infinite time generation."

**Architecture changes (ST-DiT-2).** Quoted list: "**Rope embedding for temporal attention** …
**AdaIN and Layernorm for temporal attention** … **QK-normalization with RMSNorm**: Following SD3, we
apply QK-normalization to the all attention for better training stability in half-precision. …
**Dynamic input size support and video infomation condition**: … Extending PixArt-alpha's idea, we
conditioned on video's height, width, aspect ratio, frame length, and fps. **Extending T5 tokens from
120 to 200**."

**Masked-frame conditioning — the mask strategies.** From `report_02.md`:

> "Typically, we unmask the frames to be conditioned on for image/video-to-video condition. During the
> ST-DiT forward, **unmasked frames will have timestep 0, while others remain the same (t)**. We find
> directly apply the strategy to trained model yield poor results as the diffusion model did not learn to
> handle different timesteps in one sample during training. Inspired by UL2, we introduce **random mask
> strategy during training**. Specifically, we randomly unmask the frames during training, including
> unmask the first frame, the first k frames, the last frame, the last k frames, the first and last k
> frames, random frames, etc. Based on Open-Sora 1.0, with **50% probability of applying masking**, we see
> the model can learn to handle image conditioning (while 30% yields worse ability) for 10k steps, with a
> little text-to-video performance drop. Thus, for Open-Sora 1.1, we pretrain the model from scratch with
> masking strategy."

The training-config mask ratios (`docs/config.md` at `v1.1.0`), verbatim:

```python
mask_ratios = {
    "mask_no": 0.75,                   # 75% no mask
    "mask_quarter_random": 0.025,      # 2.5% random mask with 1 frame to 1/4 #frames
    "mask_quarter_head": 0.025,        # 2.5% mask at the beginning with 1 frame to 1/4 #frames
    "mask_quarter_tail": 0.025,        # 2.5% mask at the end with 1 frame to 1/4 #frames
    "mask_quarter_head_tail": 0.05,    # 5% mask at the beginning and end with 1 frame to 1/4 #frames
    "mask_image_random": 0.025,        # 2.5% random mask with 1 image to 1/4 #images
    "mask_image_head": 0.025,          # 2.5% mask at the beginning with 1 image to 1/4 #images
    "mask_image_tail": 0.025,          # 2.5% mask at the end with 1 image to 1/4 #images
    "mask_image_head_tail": 0.05,      # 5% mask at the beginning and end with 1 image to 1/4 #images
}
```

At inference the `mask_strategy` is a **6-number tuple** (the report says "five number tuple"; the config
doc says six and enumerates six — the config doc is the operative one): "It is 6 number tuples separated
by `;`. Each tuple indicate an insertion of the condition image or video to the target generation. …
**First number**: the loop index … **Second number**: the index of the condition image or video in the
`reference_path`. **Third number**: the start frame … **Fourth number**: the location to insert. (0 means
insert at the beginning, 1 means insert at the end, and -1 means insert at the end of the video)
**Fifth number**: the number of frames to insert … **Sixth number**: the edit rate of the condition image
or video. (0 means no edit, 1 means full edit)." Long video: "the total length of the video is
`loop * (num_frames - condition_frame_length) + condition_frame_length`."

**Dependencies needed to run the model.** `v1.1.0` README install block, verbatim:

```bash
# install torch
pip install torch torchvision
# install flash attention (optional)
pip install flash-attn --no-build-isolation
# install apex (optional)
# set enable_layernorm_kernel=False in config to avoid using apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" \
  git+https://github.com/NVIDIA/apex.git
# install xformers
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

**apex and flash-attn are both marked optional**, and apex is avoidable via
`enable_layernorm_kernel=False`. The shipped stage-3 config has
`"enable_flash_attn": false, "enable_layernorm_kernel": false`.

**Checkpoint name and size.** README model-weights table:

| Resolution | Model Size | Data | #iterations | Batch Size | URL |
|---|---|---|---|---|---|
| mainly 144p & 240p | 700M | 10M videos + 2M images | 100k | dynamic | `hpcai-tech/OpenSora-STDiT-v2-stage2` |
| 144p to 720p | 700M | 500K HQ videos + 1M images | 4k | dynamic | `hpcai-tech/OpenSora-STDiT-v2-stage3` |

`hpcai-tech/OpenSora-STDiT-v2-stage3` (HF API: created `2024-04-24T14:38:46.000Z`, `license:
apache-2.0`, not gated) ships `model.safetensors` of **3,073,157,592 bytes** (~3.07 GB; the config says
`"torch_dtype": "float32"`). `config.json`:

```json
{"architectures": ["STDiT2"], "caption_channels": 4096, "depth": 28, "hidden_size": 1152,
 "in_channels": 4, "input_size": [32, 60, 106], "input_sq_size": 512, "mlp_ratio": 4.0,
 "model_max_length": 200, "num_heads": 16, "patch_size": [1, 2, 2], "pred_sigma": true,
 "qk_norm": true}
```

Note **`in_channels: 4`** — Open-Sora 1.1 uses a 2D SD VAE, not a temporal one. The report says so
explicitly in its limitations: "we find the generated model is sometimes noisy and not fluent, especially
for long videos. We think the problem is **due to not using a temporal VAE**. … we plan to develop a
temporal VAE for the model in the next version."

**Training cost as published.** "we pretrain the model with gradient-checkpointing for **24k** steps,
which takes **4 days** on **64 H800 GPUs**" … "To summarize, the training of Open-Sora 1.1 requires
approximately **9 days on 64 H800 GPUs**." Stage 1 total "81k" steps; stage 2 "22k steps for one day";
stage 3 "4k with one day on high-quality data". Also: "for Open-Sora 1.1's training includes multiple
changes, and as a result, **ema is not applied**." Mask ratio was raised mid-run: "increase the mask
ratio to 25% as we find image-condition not learning well".

**Data.** "we use a dataset with **9.7M videos + 2.6M images** for pre-training, and **560k videos +
1.6M images** for fine-tuning."

**Licence.** Repo `LICENSE` at `v1.1.0` is the **Apache License Version 2.0**; the HF checkpoint repo
also reports `license: apache-2.0`. This is the only candidate here whose code *and* weights are both
plainly Apache-2.0 alongside Wan2.1.

**Per-GPU training memory.** **Not stated** anywhere in the report or config docs. Training config shows
`dtype = "bf16"`, `grad_checkpoint = True`, `plugin = "zero2"`, `sp_size = 1` and per-bucket batch sizes
(e.g. `"144p": {1: (1.0, 48), 16: (1.0, 17), 32: (1.0, 9), 64: (1.0, 4), 128: (1.0, 1)}`), but no VRAM
figure.

**diffusers support.** None. **Not stated** — Open-Sora ships its own `scripts/train.py` /
`scripts/inference.py`.

---

## Feasibility table on one 48 GB A6000

Only what the sources state. No estimates of mine. "—" means the source is silent.

| Candidate | Stated **training / post-training** memory or hardware | Stated **inference** memory | Training code public? |
|---|---|---|---|
| Cosmos-Predict1 7B Video2World | "8 NVIDIA GPUs*" with "`H100-80GB` or `A100-80GB` GPUs are recommended". Low-memory doc: "reducing the video resolution and the number of frames can facilitate training with less number (4) of GPUs or with GPUs with **lower memory (40GB)**"; named configs `4gpu_80gb`, `8gpu_40gb`, `4gpu_40gb` (last one at `video_size=(192,192)`, `num_frames=17`, `ema=dict(enabled=False)  # turn off to save memory`). Action recipe: same 8-GPU table. Paper: 14B needs "approximately 280 GB for model parameters, gradients, and optimizer states, alongside 310 GB for activations" | — (repo gives no single-GPU inference VRAM) | Yes (Apache-2.0) |
| Cosmos-Predict2 2B Video2World | No post-training memory table exists in the repo. Iteration speed only: "2B model uses 8 GPUs … Context-Parallelism of size 2"; A100 22.5 sec/iter, H100 NVL 10.07 sec/iter | "Cosmos-Predict2-2B-Video2World — **32.54 GB**"; "At least 32GB of GPU VRAM for 2B models"; L40S (48 GB) runs it in 127.49 sec | Yes (Apache-2.0) |
| Cosmos-Predict2 2B action-conditioned (Bridge) | `torchrun --nproc_per_node=2` (**2 GPUs**), `fsdp_shard_size=-1`, `fusedadamw`, 480p/4fps. Memory: — | inherits the 32.54 GB row | Yes |
| Cosmos-Predict2 14B Video2World | "14B model uses 32 GPUs … Context-Parallelism of size 8" | "**56.38 GB**"; "At least 64GB of GPU VRAM for 14B models" | Yes |
| Cosmos-Predict2.5 2B (base) | `torchrun --nproc_per_node=8` in the generic post-training doc. Memory: — ; setup doc says only "Ampere … or newer" | — | Yes (Apache-2.0) |
| Cosmos-Predict2.5 2B action-cond (Bridge, 13 frames) | **`torchrun --nproc_per_node=1`** — one GPU, FSDP, `lr=2**(-14.5)`. DMD2 distillation also `--nproc_per_node=1`, "after 4,000 training steps". Memory: — | — | Yes |
| Wan2.1 T2V-1.3B | **No training code in the official repo.** — | "The T2V-1.3B model requires only **8.19 GB VRAM** … It can generate a 5-second 480P video on an RTX 4090 in about 4 minutes" | No |
| SkyReels-V2 DF-1.3B-540P / I2V-1.3B-540P | **No training code released** (`skyreels_v2_infer` only). Paper's I2V recipe: "only 10,000 training iterations on **384 GPUs**". Pretrain lr 1e-4 → 5e-5 → 2e-5, AdamW | "Generating a 540P video using the **1.3B model requires approximately 14.7GB peak VRAM**"; "To reduce peak VRAM, just lower the `--base_num_frames`, e.g., to 77 or 57" | No |
| Matrix-Game 2.0 (1.8B) | **No training code released.** Paper: foundation model "120k steps with learning rate=2e-5, **batch size=256**"; distillation "6k steps" + "4k training steps", lr 6e-6. Memory: — | "Nvidia GPU with at least **24 GB memory** (A100, and H100 are tested). … 64 GB RAM"; "25 FPS generation on a single H100 GPU" | No |
| Hunyuan-GameCraft-1.0 | **No training code released.** Paper: "**full-parameter training on 192 NVIDIA H20 GPUs** … batch size of 48 … 30k iterations at lr 3e-5 … 20,000 iterations" at lr 1e-5 | "**Minimum: The minimum GPU memory required is 24GB but very slow. Recommended: … 80GB**"; "24GB for 704px1216p but very slow" | No |
| CogVideoX-2B | Model card: "**47 GB (bs=1, LORA) / 61 GB (bs=2, LORA) / 62GB (bs=1, SFT)**" per GPU. Repo finetune README (conflicting, lower): lora rank128 DDP fp16 49x480x720 "**16GB VRAM (NVIDIA 4080)**"; sft DDP fp16 "**36GB VRAM (NVIDIA A100)**"; sft 1-GPU zero-2 + opt offload "**17GB VRAM (NVIDIA 4090)**" | "SAT FP16: 18GB / diffusers FP16: starting from 4GB* / INT8: 3.6GB*" | Yes (`finetune/` in `zai-org/CogVideo`) |
| LTX-Video 2B | — (Training section not retrieved) | distilled LoRA "Requires only 1GB of VRAM"; otherwise — | Training section exists in README; contents **not retrieved** |
| Open-Sora 1.1 (700M STDiT-2) | No VRAM figure stated. Published cost: "approximately **9 days on 64 H800 GPUs**"; config uses `dtype="bf16"`, `grad_checkpoint=True`, `plugin="zero2"`, `sp_size=1` | — | Yes (Apache-2.0) |

**The only two rows where a published recipe is stated to run on one GPU:** Cosmos-Predict2.5's
action-conditioned Bridge post-training (`--nproc_per_node=1`, but *no memory figure is given*), and
CogVideoX-2B fine-tuning (where the two official numbers disagree by ~3×, and the lower set is
explicitly single-GPU: "sft, 1-GPU zero-2 + opt offload, fp16, 49x480x720, 17GB VRAM (NVIDIA 4090)").

---

## Open questions

1. **Is Cosmos-Predict2's tokenizer literally Wan2.1's VAE checkpoint, or a re-trained model of the same
   architecture?** Verified: diffusers loads it as `AutoencoderKLWan` (whose docstring says "Introduced in
   [Wan 2.1]"), and NVIDIA's `cosmos_predict2/tokenizers/tokenizer.py` is the same code down to the
   `temperal_downsample` misspelling. Not verified: whether the *weights* are Wan's. NVIDIA never says so
   in any source I retrieved, and the gated model cards may.
2. **Cosmos-Predict2 / 2.5 post-training VRAM.** Neither repo publishes a training-memory table. The
   feasibility of the `--nproc_per_node=1` Predict2.5 action recipe on 48 GB is therefore unknown from
   sources; it needs a rendered run.
3. **Cosmos-Predict2 action-conditioned training steps / batch size / lr.** The doc shows only the hydra
   `defaults` block. The numbers live in
   `cosmos_predict2/configs/action_conditioned/experiment/exp.py`, which I downloaded but did not read.
4. **CogVideoX-2B fine-tuning memory: which table is right?** The model card (47 GB LoRA bs=1) and the
   repo finetune README (16 GB LoRA rank128) cannot both describe the same configuration. Likely
   different ranks/precision/offload, but neither source says.
5. **LTX-Video**: objective, text encoder, conditioning mechanism, licence text of
   `LTX-Video-Open-Weights-License-0.X.txt`, and the README's Training section — all **not retrieved**.
   Also whether a 2B *base* (non-distilled) 0.9.8 exists, given "Both models are distilled from the same
   base model [ltxv-13b-0.9.8-dev]".
6. **Cosmos-Predict2.5 paper (arXiv 2511.00062)** was not downloaded. Parameter counts, training data
   scale, and the exact rectified-flow loss for 2.5 would come from there.
7. **Hunyuan-GameCraft parameter count** is stated nowhere I could find — not the model card, not the
   paper. "HunyuanVideo-based MM-DiT" is all that is asserted.
8. **Cosmos-Predict2 training data scale** is stated nowhere in the repo; there is no linked Predict2
   tech report.
9. **CogVideoX training data scale** — the paper has a data section I did not read.
10. **Two internal contradictions to be aware of when citing:** NVIDIA's own low-memory doc says
    "384x384 resolution, video length of 25 frames" directly above a config reading
    `video_size=(192, 192)` and `num_frames = 17`; and Matrix-Game's `kv_cache_size = 15 * 1 *
    self.frame_seq_length # 32760` comment contradicts its own arithmetic (15 × 880 = 13200).
    Matrix-Game's paper also states both "~1200 hours" (abstract) and "about 800-hour … in total" (§5).
