# Generated video as robot training data for LEGO assembly: evidence cards

Date: 2026-09-16. Question: can a video generation model, fine-tuned on a small set of demonstrations of LEGO assembly (either phone-camera human-hand videos or robot-executed videos with proprioception), generate text- or action-conditioned assembly videos from a starting frame that are good enough to serve as synthetic demonstrations for downstream robot learning?

Method: primary sources only (arXiv HTML full text, GitHub READMEs, NVIDIA docs), downloaded raw and grepped. Quotes are verbatim. "not stated" means the source text I read does not say it; "not extracted" means the paper has it in a table or figure I did not fully parse. Nothing is inferred. Two settings are tracked throughout: (a) human-hand phone videos, no action labels; (b) robot-executed videos with recorded proprioception (the DreamGen setting).

Written to the worktree fallback path because the main checkout's `.claude/` is write-blocked from this worktree session.

Sources read: arXiv 2505.12705 (DreamGen), 2503.14734 (GR00T N1), 2409.16283 (Gen2Act), 2406.16862 (Dreamitate), 2302.00111 (UniPi), 2310.10625 (VLP), 2310.08576 (AVDC), 2412.14803 (VPP), 2503.00200 (UVA), 2410.11758 (LAPA), 2402.15391 (Genie), 2407.15208 (Im2Flow2Act), 2405.01527 (Track2Act), 2401.00025 (ATM), 2503.14492 (Cosmos-Transfer1), 2604.09330 (VAG), 2603.17808 (EVA), 2602.15922 (DreamZero), 2606.12604 (EgoEngine), 2509.22578 (EgoDemoGen), 2608.18948 (RoboEdit), 2512.09406 (H2R-Grounder), 2607.08436 (EgoWAM), 2601.16163 (Cosmos Policy), 2512.06628 (MIND-V), 2601.15282 (RBench/RoVid-X), 2608.13049 (H2R-Bench), 2606.08828 (Video2Sim2Real); GitHub NVIDIA/GR00T-Dreams README; nvidia-cosmos/cosmos-predict2.5 docs/post-training_video2world_gr00t.md; Cosmos Cookbook GR00T-Dreams post-training recipe.

---

## Card 1. DreamGen (NVIDIA GEAR, May 2025) and the GR00T-Dreams pipeline

**What it is.** "a simple yet highly effective 4-stage pipeline for training robot policies that generalize across behaviors and environments through neural trajectories—synthetic robot data generated from video world models." Steps: "(1) We fine-tune video world models on a target robot to capture the dynamics and kinematics of the specific embodiment; (2) we prompt the model with pairs of initial frames and language instructions to generate large volumes of robot videos" ; (3) pseudo-actions via IDM or latent action model; (4) train visuomotor policies on neural trajectories.

**Base video model and size.** Paper: "For the majority of our downstream robot experiments, we utilize WAN2.1 [9] as our base video world model." WAN2.1 variant size (1.3B vs 14B): not stated in the paper text. DreamGen Bench also evaluates Hunyuan, CogVideoX, Cosmos. Released pipeline (GR00T-Dreams README): "We provide the full pipeline for DreamGen, as Cosmos-Predict2 as the video world model in the repository." The cosmos-predict2.5 doc has configs for both sizes: 2B ("torchrun --nproc_per_node=1 ... predict2_video2world_training_2b_groot_gr1_480") and 14B ("torchrun --nproc_per_node=8 ... predict2_video2world_training_14b_groot_gr1_480"). Multi-view: "we concatenate the viewpoints into a 2×2 grid (with one grid with black pixels) and fine-tune the video world models."

**Conditioning.** Text + first frame: "Given an initial frame and a language instruction, the model generates video rollouts depicting the intended behavior." No action conditioning of the video model. Initial frames are collected manually: "For real-world experiments, we manually take new initial frames while randomizing the location of the target object."

**Fine-tuning data size.**
- GR1 humanoid generalization experiments: "2,884 trajectories of the GR1 Humanoid performing diverse pick-and-place motions" "collected in a single lab environment".
- GR1 data-augmentation tasks: "For each task, we collect 100 trajectories, but only utilize 10 trajectories for Hammering, Wiping, Stacking, and 25 trajectories for Folding to test data efficiency."
- Franka: "49,895 DROID data examples, and further fine-tune the model on the low data trajectories for each task ... 11, 10, and 8 trajectories for putting milk in bowl, cube stacking, and scooping M&Ms".
- SO-100: "we sample 10 and 13 videos for the two tasks, which yield 68 and 44 trajectories, respectively, after trimming."
- RoboCasa (sim): "we train our video world model on 1,200 original human demonstrations".
- DreamGen Bench training sets: RoboCasa 1200 trajs, GR1 100 trajs (Table 2). The released GR1 set is `nvidia/GR1-100` on Hugging Face.
- Hours: not stated for the video-model fine-tuning sets.

**Compute and steps.** "For all of the WAN 2.1 fine-tuning experiments, we used a learning rate of 1e-4, LoRA rank 4, and LoRA alpha 4. For RoboCasa finetuning, we trained the model for 100 epochs with a batch size of 32. For GR1 finetuning, we trained the model for 75 epochs with a batch size of 64. For DROID fine-tuning, we trained the model for 5 epochs with a batch size of 64. For both of the two tasks in SO-100 finetuning, we trained the model for 200 epochs with batch size 8." GPUs and wall-clock for fine-tuning: not stated. LoRA rationale: "To mitigate forgetting prior internet video knowledge, we use Low-Rank Adaptation (LoRA) [21] by default". Generation compute: "generating the 240k-sample RoboCasa dataset took 54 hours on 1500 NVIDIA L40 GPUs." Per-video time: not stated. Resolution/frames for training: not stated in the paper; the cosmos-predict2.5 GR1 config is named `_480` and the cookbook inference JSON uses `"num_output_frames": 93, "resolution": "432,768"`.

**How videos become robot-usable data.**
- IDM: "we use diffusion transformers with SigLIP-2 vision encoder and train with a flow matching objective. IDM is conditioned on two image frames and is trained to predict action chunks between the image frames ... We do not explicitly use any language or proprioception as input ... For the IDM training data, we use the same dataset used to train the video world models for each setup ... After training, we employ a sliding window approach". IDM accuracy on generated video: not stated numerically; "we also replay the IDM actions in simulation to empirically see the quality of the IDM actions".
- LAPA: "transformer encoder-decoder architecture and is trained on diverse robot and human videos ... VQ-VAE objective ... we condition the latent action model on the current frame and the future frame (1 second ahead) ... We use the pre-quantized continuous embedding as the latent action following GR00T N1". "We use a codebook size of 8 and a sequence length of 16 for vector quantization. We train 100K steps with a batch size of 1024." Training mixture Table 3 totals "438.1M" frames, "5,721.3" hours across GR-1 teleop (88.4 h), DexMG, DROID (428.3 h), RT-1, Language Table, Bridge-v2, RoboCasa (268 h), Agibot-Alpha (1,979.4 h), Sth-v2 (105.7 h), Ego4D (2,144.7 h).
- Policy training: "We condition state information with zero values, since neural trajectories do not contain state information." Co-training "with a sampling ratio of 1:1"; "For GR00T N1, we treat the two types of trajectories as separate embodiments by using separate action encoder and decoder." "For behavior and environment generalization experiments, we only use neural trajectories for policy training."
- Filtering of generated videos: not stated in the paper. The Cosmos Cookbook GR00T-Dreams recipe (March 2026) adds it: "Best-of-N rejection sampling: Generate several candidates per prompt and keep only high-scoring (4.0 or 5.0) videos" using Cosmos Reason 2 as a zero-shot physics critic on a 1–5 scale; "you can filter your synthetic videos by removing those that received scores of 1.0 or 2.0". Cookbook caveat: "The Reason 2 video critic is a proxy for physical plausibility, not a ground-truth oracle."

**Downstream results (numbers).**
- RoboCasa scaling: "scaling synthetic data up to 333× relative to the original human demonstrations. This yields log-linear improvements in policy performance as the number of neural trajectories increases"; ground-truth regimes "low data (720), mid-data (2.4k), and high-data (7.2k)". "solely training on neural trajectories with IDM actions enables us to reach a non-trivial performance (20.6% average success rate across 24 tasks)". Example Table 4 row (PnPCounterToCab, GR00T N1): 30/100/300 traj = 1.85/6.86/36.27; +240k NT = 3.85/19.23/50.96; ONLY NT = 16.67. Table 4 averages: not extracted.
- Real-world low-data co-training (Table 5, GR00T N1, success %): GR1 Hammering 60.0→65.0, Wiping 36.6→49.0, Folding 27.0→37.0, Stacking 25.0→35.0, average 37.0→46.0; Franka Pick&Place 40.0→60.0, Cube Stacking 10.0→20.0, Tool Usage 20.0→30.0; SO-100 Pick&Place 17.0→26.0, Tic-Tac-Toe 25.0→65.0. Diffusion Policy: GR1 average 22.0→27.0 but Hammering 35.0→15.0 (a regression). High-data (100%) GR00T N1 reference: GR1 average 69.0. Neural trajectories generated: "300 neural trajectories for each GR1 task, 100 neural trajectories for each Franka task, and 40 and 50 neural trajectories for the two SO-100 tasks".
- Behavior generalization (Table 1): "we generate 50 neural trajectories for each of the 14 novel behavior tasks and train our downstream visuomotor robot policy only on the neural trajectories ... (11.2% → 43.2%)". Environment generalization (13 tasks in novel environments): 0.0 → 28.5. "we only capture initial frames, effectively implementing a zero-shot transfer methodology". Evaluation: "10 rollouts per checkpoint"; partial credit ("we give 0.5 success for picking up the bottle for the 'Pour Water' task").
- DreamGen Bench (Table 2, instruction following by GPT-4o, %): zero-shot models are near zero (Cosmos-zero 4.2 RoboCasa, 0.0 GR1-object, 6.4 GR1-behavior, 3.5 GR1-env; WAN2.1-zero 0.0/0.0/0.0/0.0). After fine-tuning on 100 GR1 trajs: WAN2.1-sft 77.1/72.0/72.3/48.3; Cosmos-sft 79.2/90.0/59.6/69.0. Benchmark-to-policy correlation: "we generate 7k neural trajectories for each of the video world models ... and show that benchmark numbers directly correlate to downstream robot policy performances."

**Failure modes reported.** "Empirically, we observe that most of the bottleneck is from the quality of the neural trajectories, which indicates that future video models that can generate videos with better language following and physics alignment could lead to a significant boost". "Enabling zero-shot generalization to novel behaviors and novel environments with robot embodiments with zero ground-truth data still remains an open research question." DROID-only model "produced trajectories that made mistakes on fine-graed details (e.g. grasping)". "The method also relies on manually providing initial frames". Evaluator: "These models can occasionally hallucinate, especially when evaluating physical realism in videos". SO-100 side effect: "the policy augmented with neural trajectories is less likely to get stuck at the initial home position".

**Code and weights.** github.com/NVIDIA/GR00T-Dreams: fine-tuning via cosmos-predict2 docs, IDM preprocessing for "franka, gr1, so100, robocasa", "Training Custom IDM model ... Given a few ground-truth trajectories of a specific embodiment, we can train an IDM model" (`scripts/idm_training.py`; new embodiments need `modality.json`, `stats.json`, and a data config), GR00T N1 fine-tuning ("boost your batch size to the max, and train for 20k steps"), DreamGen Bench evaluators (Qwen2.5-VL, GPT-4o). Datasets: `nvidia/GR1-100`, `nvidia/EVAL-175`. Released IDM checkpoints: not stated in the README. Video-model weights post-trained on GR1: not stated.

**GR00T N1 context (2503.14734).** "we fine-tune image-to-video generation models ... on all of our 88 hours of in-house collected teleoperation data and generate 827 hours of video data given the existing initial frames with novel language prompts, augmenting it by around 10×." Prompt diversity: "we first use a commercial-grade multimodal LLM to detect the objects given initial frames and generate many more possible combinations of 'pick up {object} from {location A} to {location B}', while instructing the model to only consider the physically feasible combinations."

**What transfers to 4×A6000 48 GB and ~100 videos.** The released code path is Cosmos-Predict2.5 2B/14B; the 2B GR1 config runs on `--nproc_per_node=1`. 100 videos is exactly the DreamGen Bench GR1 training size and above the SO-100 sizes (10–13 videos). Setting (b) reproduces the recipe directly: fine-tune video model (LoRA rank 4), train IDM on the same 100 trajectories, generate, label, co-train 1:1. Setting (a) lacks the IDM; only the LAPA path applies, and LAPA labels are not executable without a robot-labelled fine-tuning set (see Card 6). Not transferable: 1500-L40 generation budgets; the paper's largest gains used 240k generated videos.

---

## Card 2. Gen2Act (Google DeepMind, Sep 2024)

**What it is.** "Gen2Act casts language-conditioned manipulation as zero-shot human video generation followed by execution with a single policy conditioned on the generated video." The video model is not a training-data source; it is a test-time conditioning signal.

**Base video model.** "We use a pre-trained VideoPoet model [20] directly without any adaptation or fine-tuning ... trained on diverse large-scale video datasets (>270M videos)". Size: not stated. "current video generation models cannot generate robot videos zero-shot and require robot-specific fine-tuning data ... Such fine-tuning often subtracts the benefits of generalization to novel scenes".

**Conditioning.** Text + first frame: prompt "A person task-name, static camera"; "the image of the scene input to the model doesn't have the robot in the frame". Generation "takes less than 10 seconds to generate a new video after generating the very first video".

**Fine-tuning data.** Video model: none. Policy: "an existing offline dataset of robot demonstrations collected by a prior work [1] [RT-1] and augment this with some paired demonstrations of human videos collected by another prior work [46] [Vid2Robot]"; generated-human-video/robot-demo pairs created "by generating videos conditioned on the first frame of the robot trajectories and the language instruction". Co-training adds "∼400 trajectories".

**How generated video becomes robot-usable.** Closed-loop policy conditioned on generated video tokens (ViT + Perceiver-Resampler, 64 tokens) with a track-prediction auxiliary loss: "we extract point tracks from the generated human video and the video of robot observations (through an off-the-shelf tracker [21]) and optimize a track prediction auxiliary loss ... not used at test-time". "Instead of attempting to explicitly extract waypoints from the generated video based on heuristics, we adopt a more end-to-end approach".

**Downstream numbers (Table I, success %, levels MG/G/OTG/MTG/avg).** RT1 68/18/0/0/22; RT1-GC 75/24/5/0/26; Vid2Robot 83/38/25/0/37; Gen2Act w/o track 83/58/50/5/49; Gen2Act 83/67/58/30/60. Co-training with ~400 extra teleop trajectories (Table III): 85/75/62/35/64. "Gen2Act achieves on average ∼30% higher absolute success rate over the most competitive baseline" on OTG/MTG.

**Failure modes.** "for the higher levels of generalization, object type (OTG) and motion type (MTG), if video generation yields implausible videos, then the policy doesn't succeed"; "generating a video plausibly ... is not a guarantee of the policy succeeding because there might be issues with grasping the object correctly and following the trajectory of the object post grasp". "limited by the current capabilities of video generation models, like inability to generate realistic hands and thereby limited ability to perform very dexterous tasks."

**Code/weights.** Not stated (VideoPoet is not public).

**Transfer.** Relevant to (a): shows human-hand video generation is usable as a motion cue without pairing hands to actions, but the policy still needed a large offline robot dataset plus ~400 diverse teleop trajectories. Its finding that realistic hands are the bottleneck applies directly to LEGO fine-grained placement.

---

## Card 3. Dreamitate (Columbia, CoRL 2024)

**What it is.** "fine-tunes a video diffusion model on human demonstrations of a given task. At test time, we generate a video showing an execution of the task conditioned on images of a novel scene, and use this synthesized video to directly control the robot." Embodiment gap bridged by a tool: "using common tools allows us to effortlessly bridge the embodiment gap between the human hand and the robot manipulator."

**Base model.** "We initialize θ to the pre-trained weights learned from large-scale internet video datasets (Stable Video Diffusion [29]). We train a separate f_θ for each task"; "the encoder and decoder are frozen such that only the spatial/temporal attention layers are fine-tuned." SVD size: not stated in the paper.

**Conditioning.** Stereo image pair, no text: "the first 13 frames correspond to the stereo view 1, and the last 12 frames correspond to the stereo view 2"; "During inference, we use 30 denoising steps with a constant classifier-free guidance of 1.0."

**Fine-tuning data (Table 1, demonstrations per task).** Rotation 371 (31 training objects), Scooping 368, Sweeping 356, Push-Shape 727. Recorded at 1280×720 with two RealSense D435i "spaced approximately 660 mm apart at a 45° angle". Human demonstrators, no teleoperation.

**Compute and steps (Table 3).** Res 768×448, lr 1e-5, batch size 4 (3 for the 1/3-data run), train steps 16384 (15360; 17408 for Push-Shape), clip duration 2.0–3.0 s, fps 5–6. GPUs and wall-clock: not stated.

**How generated video becomes actions.** "we 3D track the tool in the synthesized video, and simply transfer the trajectory into explicit robot actions." "Actions a_t ∈ SE(3) are represented as the 6D pose of the tool relative to the camera. By using a known CAD model, we can efficiently and accurately track the tool"; "we use Megapose [39] and operate on 768×448 resolution video frames"; stereo triangulation of the tool center, rotation averaged across views. Open-loop execution. Tracking init: "background subtraction and hand removal based on skin color were employed. In rare cases where this process failed, human corrections were applied".

**Downstream numbers (vs Diffusion Policy on the same demos, 40 trials).** Rotation "92.5% vs. 55%"; Scooping "85% vs. 55%"; Sweeping "92.5% success rate ... Diffusion Policy ... only a 12.5% success rate"; Push-Shape reported as mIoU and rotation error (values not extracted). Data scaling: "our model remains stable and maintains high success rates even with only one-third of the data" (about 124 demos for rotation).

**Failure modes.** "limited to generating visually trackable robot actions ... our approach can fail when the end-effector is heavily occluded. Additionally, reliance on rigid tools limits the applicability of our approach to the task requiring fine-grained control. Finally, video models have higher computational costs, making real-time closed-loop control infeasible". Scooping "particularly challenging ... due to the small target (the scooper)".

**Code/weights.** "We will release the code and data for reproducing our results." GitHub cvlab-columbia/dreamitate exists (not read).

**Transfer.** Closest precedent for setting (a) with a small human-video set and consumer-class compute (SVD at 768×448, batch 4, ~16k steps). The trick that made it work, a rigid CAD-known tool tracked with MegaPose, does not exist for bare-hand brick placement; a LEGO variant would need a hand-held tool, a tracked gripper, or a hand-pose-plus-brick-pose tracker.

---

## Card 4a. UniPi (Du et al., 2023)

**What it is.** "we cast the sequential decision making problem as a text-conditioned video generation problem"; "A set of underlying actions are then regressed from the synthesized frames".

**Base model and size.** Imagen-Video-style U-Net: "first-frame conditioned video diffusion models on 10x48x64 videos (skipping every 8 frames) with 1.7B parameters and a temporal super resolution of 20x48x64 ... with 1.7B parameters"; text via "T5-XXL ... 4.6 billion parameters".

**Conditioning.** Text + first frame, with "Trajectory Consistency through Tiling" and temporal super-resolution.

**Data and compute.** Pretraining "14 million video-text pairs, 60 million image-text pairs, and the publicly available LAION-400M"; robot fine-tuning "Bridge dataset [29] with 7.2k video-text pairs"; sim "200k example videos". "We train each of our video diffusion models for 2M steps using batch size 2048 with learning rate 1e-4 ... We use 256 TPU-v4 chips".

**Actions.** "We train a small model to estimate actions given input images ... on a separate, smaller and potentially suboptimal dataset generated by a simulator"; "3x3 convolutional layer, 3 layers of 3x3 convolutions with residual connection, a mean-pooling layer ... MLP layer of (128, 7)".

**Downstream numbers.** Real-world Table 4 (surrogate video-plan success, not robot execution): "No Pretrain 72.6%, Pretrain 77.1%". "UniPi without pretraining often synthesizes plans that fail to complete the task". Sim tables: not extracted.

**Failure modes.** "the underlying video diffusion process can be slow, it can take a minute to generate highly photorealistic videos"; "In partially observable environments, video diffusion models might make hallucination of objects or movements that are unfaithful or not in the physical world."

**Code.** Not stated (UVA later: "the official implementation is not available").

## Card 4b. Video Language Planning (Du et al., Oct 2023)

**What it is.** "tree search procedure, where we train (i) vision-language models to serve as both policies and value functions, and (ii) text-to-video models as dynamics models."

**Video model.** UniPi family; size not stated. Data: "approximately 10000 long horizon trajectories in both simulation and real ... roughly 20000 trajectories and had approximately 400000 short-horizon text labels" (Language Table); "approximately 1200 teleoped demonstrations on a kitchen stacking task, where operators were first asked to stack bowls on top of each other, then cups on top of bowls" (14-DoF ALOHA); mobile manipulator on RT-1 data plus "Bridge, RT-2, Ego4D, EPIC-KITCHEN, and LAION-400M".

**Actions.** "short-horizon goal-conditioned policy π_control(x, x_g), which given the current image observation x and next frame in a video plan x_g outputs a low level control action"; "using a goal-conditioned policy conditioned on each intermediate frame in a synthesized video leads to the best overall performance". Goal policy trained "using 16 TPUv3 pods for 1 day".

**Numbers.** Table 2 execution on Language Table long-horizon tasks: VLP rows reach 64/92/16 % where baselines are mostly 0–26 % (row labels not extracted). Cost: "VLP took approximately 1 hour per environment".

**Failure modes.** "the video model would incorrectly interpret tasks ... where it interprets the manipulator as an octopus"; "synthesized videos with inconsistent physics, where objects would infrequently appear or disappear"; "images as a world state representation ... does not capture the full 3D state and cannot encode latent factors such as physics or mass."

## Card 4c. AVDC (Ko et al., Oct 2023)

**What it is.** "infer actions from video prediction without the need of any action labels by leveraging dense correspondences in a video."

**Model.** Pixel-space video U-Net "modified version of the image diffusion model proposed by Dhariwal & Nichol (2021)", resolution 128×128, CLIP text encoder (63M), trained from scratch per domain. "We train all models on 4 V100 GPUs with 32GB memory each."

**Data and compute.** "Meta-World: about 24 hours of training (165 videos)"; "iTHOR: about 24 hours of training (240 videos)"; "Real world experiment: 48 hours of pre-training on Bridge and 4 hours of fine-tuning on human data. (40000 Bridge + 20 Human videos)"; Visual Pusher "only actionless human pushing data (198 videos) ... trained the model for 10k steps".

**Actions.** GMFlow optical flow between generated frames plus the initial depth map, "reconstruct a sequence of 3D rigid transformations for each object"; "if the object is graspable, we randomly sample a grasp on the object"; object mask from GT or Language-SAM; replanning up to 5 times.

**Numbers.** Meta-World average 43.1% (with predicted masks 34.5%); iTHOR 31.3% vs BC baselines 0.4/2.1%; Visual Pusher "90% zero-shot success rate out of 40 runs"; real Franka pick-and-place after 20 human videos: "our approach failed in 8 of the 10 tested trials. We found that 75% of the failures were caused by the wrong plan from the video diffusion model. It either picked the wrong object or placed it at the wrong target. The other 25% of the failures were caused by the discontinuity of video generation." Planning "about 18 seconds for each round" on an RTX 3080Ti; 10-step DDIM 37.5% vs 43.1%.

**Failure modes.** "errors made by the optical flow tracking model ... small pixel-level errors in tracking small objects would result in large errors in the 3D space"; "when the majority of an object is occluded by the robot arm, our algorithm may lose track"; "force information, crucial for manipulation, is unobtainable from RGB videos."

**Code.** "we provide an open-source codebase for training video policy models".

**Transfer (4a–c).** UniPi/VLP compute is out of reach; AVDC's 4×V100 recipe is in reach but its real-world result (2/10) with 20 human videos and small objects is a direct warning for bricks: flow on small objects plus a 128×128 generator was the failure source.

---

## Card 5a. Video Prediction Policy (VPP, Dec 2024)

**What it is.** "learns implicit inverse dynamics model conditioned on predicted future representations inside VDMs"; the video model is used "primarily as a 'vision encoder' rather than a 'denoiser' by performing only a single forward step".

**Base model.** "the open-sourced Stable Video Diffusion (SVD) model (Blattmann et al., 2023a) with 1.5 billion parameters"; output "16×256×256"; text conditioning added.

**Data and compute.** Stage 1: "193,690 human manipulation trajectories (Goyal et al., 2017) [SSv2] and 179,074 robotic manipulation trajectories (O'Neill et al., 2023) [OXE], along with downstream task videos ... Fine-tuning the video model takes 2-3 days on eight NVIDIA A100 GPUs." Stage 2 policy: "approximately 6-12 hours on four NVIDIA A100 GPUs." Real data: Franka "2,000 trajectories for over 30 tasks"; Xhand "4,000 trajectories over 100+ tasks".

**Numbers.** CALVIN ABC→D "from an average task completion length of 3.35 to 4.33"; "with only 10% of the annotated Calvin ABC data ... 3.25". Ablation: "removing the co-trained Internet manipulation data resulted in a performance decrease from 4.33 to 3.97"; from scratch without SVD: "substantial performance drop". Real world (Table 5, success rate): Franka seen 0.85 vs DP 0.42/Susie 0.56/GR-1 0.52; unseen 0.73 vs 0.25/0.46/0.38; dexterous hand seen 0.75, unseen 0.60, tool-use 0.68 vs DP 0.28/0.11/0.05. Control at "7-10 Hz with consumer-level NVIDIA RTX 4090".

**Failure modes.** Not stated as a section; the design exists because full denoising is "time-cosuming and lead to low control frequency."

**Code.** "Code can be found at supplementary materiel."

## Card 5b. Unified Video Action model (UVA, Mar 2025)

**What it is.** "jointly optimizes video and action predictions ... learning a joint video-action latent representation and decoupling video-action decoding"; masked training makes one model a policy, forward model, inverse dynamics model, or video generator.

**Base model.** "Our model builds on the pretrained model (MAR-B) released by [27]"; VAE "kl-f16"; UVA is "0.5B". Not built on a large video foundation model.

**Data.** Sim: PushT, Toolhang, PushT-M, Libero10; real UMI: "The three public datasets contain a total of 6,764 episodes. We randomly selected 500 episodes from each dataset and combined them into a dataset with 1500 episodes". Action-free video: "pretrained for video generation using the Human Video dataset from [12], which contains 3,175 human-only videos ... the accuracy can be further improved".

**Numbers.** "UVA surpasses the best baseline by 20% on the PushT-M task and by 5% on the Libero10 benchmark"; visual disturbance "UniPi achieves a success rate of 40%, UVA achieves 64%, while OpenVLA only reaches 32%"; forward-dynamics guidance "DP-C alone achieves an 38% success rate, while incorporating our model ... increases the success rate to 60%"; real multi-task "15% higher success rate on the Cup task and a 40% higher success rate on the Mouse task compared to DP-UMI". Inference 95 ms per 16-action chunk (16 steps).

**Failure modes.** "fails when the background color is the same as the objects"; "it does not currently leverage large amounts of actionless video data ... our method occasionally achieves only comparable performance to the DP-UMI". UniPi baseline "may fail to generate some objects entirely".

**Code.** "Please check out our code for details" (released).

**Transfer (5a–b).** VPP uses a fine-tuned video model as a representation, not a data generator; it needs action-labelled data for the policy head (setting (b) only). UVA's inverse-dynamics mode is a candidate IDM for labelling generated videos in setting (b), trained on the same ~100 robot episodes.

---

## Card 6a. LAPA (Latent Action Pretraining from Videos, Oct 2024)

**What it is.** "the first unsupervised method for pretraining Vision-Language-Action (VLA) models without ground-truth robot action labels."

**Latent action model.** "encoder-decoder architecture where the encoder takes the current frame x_t and the future frame x_{t+H} ... VQ-VAE objective"; "we use cross attention to attend z_t given x_t instead of additive embedding"; NSVQ; "The encoder can be seen as the inverse dynamics model and the decoder can be seen as the world model." Codebook and sequence sizes are swept in Figure 5 (values not extracted).

**VLM backbone.** "Large World Model (Liu et al., 2024) as the backbone VLM model"; "the underlying policy model (7B)".

**Data.** Pretraining on Bridgev2, Open-X, Something-Something v2 ("human manipulation videos"); Language Table "181k trajectories", "440k real-world trajectories". Fine-tuning: real tasks "Each task involves 150 trajectories across 15 objects"; SIMPLER "100 multi-task trajectories".

**Compute.** "For pretraining LAPA (Open-X) ... we use 8 H100 GPUs for 34 hours with a batch size of 128 (total of 272 H100-hours). In contrast, OpenVLA required a total of 21,500 A100-hours".

**Numbers.** "outperforms the current state-of-the-art VLA model trained with ground-truth actions (by +6.22%)"; "LAPA trained with human videos outperforms OpenVLA (Bridge) on average"; pairwise "LAPA outperforms OpenVLA in 65.4% when disregarding the ties"; "LAPA outperforms OpenVLA in reaching performance (83.33% vs 66.67%)".

**Failure modes.** "LAPA underperforms compared to action pretraining when it comes to fine-grained motion generation tasks like grasping"; "most failures of LAPA are due to early grasping"; on SSv2 "latent actions capture not only hand movements but also camera movements".

**Code/weights.** "We will open-source the model checkpoints and code at latentactionpretraining.github.io."

## Card 6b. Genie (Feb 2024), background on latent actions

"we learn latent actions in a fully unsupervised manner"; "We limit the vocabulary size |A| of the VQ codebook ... (we use |A|=8 in our experiments)"; "The latent action model has 300M parameters, a patch size of 16, and a codebook with embedding size 32 and 8 unique codes"; Platformers data "6.8M 16s video clips (30k hours)"; 11B model. Robotics: "We trained a 2.5B-parameter model on the Robotics dataset" (RT1 ∼130k demos + 209k episodes, actions discarded); latent actions were "consistent across varied prompt frames and have semantic meaning: down, up and left". Imitation: "The LAM-based policy achieves the same score as the oracle given as few as 200 expert samples to adapt". Weights: "We have chosen not to release the trained model checkpoints".

**Transfer (6a–b).** Latent actions are the only labelling route for setting (a) that needs no paired data, but every published use (Genie, LAPA, DreamGen, GR00T N1) still maps latents to real actions with a robot-labelled fine-tuning set (LAPA: 100–150 trajectories per task), and LAPA reports grasp-timing errors, which is the failure mode that matters for brick seating. Phone video adds the camera-motion leakage LAPA saw on SSv2.

---

## Card 7. Human-video-to-action bridges via point tracks and flow

### 7a. Im2Flow2Act (Jul 2024)

"use object flow as the manipulation interface, bridging domain gaps between different embodiments (i.e., human and robot) and training environments (i.e., real-world and simulated)." Flow generator: "built on top of the video generation model Animatediff [18]" with the Stable Diffusion autoencoder; "we insert the LoRA ... with a rank of 128 into the Unet from StableDiffusion and train the motion module layer from scratch with learning rate of 1×10−4 for 4000 epochs"; flow image H=W=32, T=32 (1024 keypoints). Conditioning: text + initial frame + initial keypoints (Grounding DINO box, TAPIR tracks). Policy: "trained on simulated robot play data" ("In total, we collect 4800 random exploration trajectories", UR5e in MuJoCo). Human videos: "We collect in-domain human demonstration videos for four tasks" at 30 FPS with a RealSense; count per task: not stated. Result: "average success rate of 81% across four real-world tasks ... without any real-world robot data"; "the performance only drops 15% on average in realworld compare to in simulation"; "Im2Flow2Act outperforms ATM by an average of 30% across four tasks" under cross-embodiment demos. Failures: "2D flow ... inherently ambiguous for representing 3D actions ... z-axis movements ... lack precision ... struggles with tasks involving out-of-plane rotation (e.g., screwing)"; "significant performance drop in the folding task"; assumes calibrated camera between sim and test. Inference on "a 24 GB NVIDIA RTX 4090". Code: not stated in text read (project site exists).

### 7b. Track2Act (May 2024, ECCV)

"predicts tracks of how points in an image should move in future time-steps based on a goal"; DiT diffusion model over point tracks, conditioned on first frame and goal image; trained on "around 400,000 videos clips" from EpicKitchens, Something-Something-v2, RT1, Bridge with "Co-Tracker [29]" pseudo-ground truth; "choose a dense grid of 400 points". Actions: rigid transforms from tracks plus depth ("Algorithm 1"), then "a residual policy learned with limited robot data" ("∼400 trajectories obtained by tele-operating the Spot" over 10 tasks). Table 2 (success %, MG/G/CG/TG, 20 rollouts per level, 25 tasks), rows in the order the paper lists methods: Goal-Conditioned BC 60/20/0/0; Affordance-Conditioned BC 65/30/10/5; Video-Conditioned BC 60/25/0/0; Hand-Object Mask BC 70/40/25/20; Ours (Open Loop) 35/25/30/25; Ours (actions, not residuals) 70/45/30/30; Ours 70/60/55/40. On video generation as a bridge: "predicting an RGB video followed by tracking suffers due to issues of implausible generation because video generation is a much more complex task than predicting the tracks". Failures: "inability to grasp the object at the right location, and inability to recover from intermediate failures"; "tasks are still of short-horizon and involve manipulating a single object". Code: not stated in text read.

### 7c. ATM (Any-point Trajectory Modeling, Dec 2023)

"pre-training a trajectory model to predict future trajectories of arbitrary points within a video frame" then "a track-guided policy ... from only a few action-labeled demonstrations." Track transformer trained on CoTracker tracks, 50% image-patch masking, track length 16; "We train all models on 4 A100 GPUs". LIBERO: "10 action-labeled demonstration trajectories and 50 action-free video demonstration trajectories of the robot for each task"; "average success rate of 63% compared to the highest success rate of 37% by previous methods". Human-to-robot (Table I, three tasks): "We collect 10 robot teleportation trajectories (action-labeled) and 100 human manipulation videos (action-free)"; BC-only 0/10/30 %, ATM with 10 robot videos only 0/0/13 %, ATM with human videos 63/63/60 %. On competing bridges: "VPT performs poorly as we empirically observed that the pseudo-action labels predicted by VPT generally show large errors on the video dataset. UniPi fails on more complex as video prediction models are not physically grounded and often generate future frames that are not physically feasible, such as cases where robots disappear from the image." Limitations: "still relies on a set of action-labeled demonstration trajectories"; "the video dataset we use in this paper only contains small domain gaps." Code released (project page).

**Transfer (7a–c).** These are the strongest evidence that 100 human videos plus ~10 robot demos can train a policy, but they use the human videos to train a track/flow predictor, not a pixel video generator, and they explicitly rank pixel generation as a worse intermediate. For LEGO, all three assume rigid-object motion and 2D tracks, which will be ambiguous for a brick seated by a vertical press (out-of-plane, small pixel motion, occlusion by fingers).

---

## Card 8. Cosmos-Transfer1 (NVIDIA, Mar 2025), the sim-to-real alternative

"a conditional world generation model that can generate world simulations based on multiple spatial control inputs of various modalities such as segmentation, depth, and edge"; control branches for blur ("Vis"), Edge (Canny), Depth (DepthAnything2), Segmentation (GroundingDINO+SAM2), each "post-trained Cosmos-Predict1-7B-Video2World". "We train each control branch with 1024 NVIDIA H100 GPUs for a period of 2 to 4 weeks depending on the modality"; one call "generates a 5-second 1280x704p video under 24 fps ... 56K tokens". Robotics case study: "a small simulated dataset of 20 robot manipulation scenarios in a basic kitchen scene using NVIDIA Omniverse and Isaac Lab ... six different text prompts" (120 videos); evaluated with "Blur SSIM, Edge F1, Mask mIoU, Diversity-LPIPS, and Quality Score"; recommended Setting 1 "preserve both the shape and appearance of the foreground robot while modifying the background" (Edge+Vis on robot, Seg on background). Policy success with augmented data: not stated (no policy experiment). Real-time only on "64 B200 GPUs, generating 5 seconds of video in only 4.2 seconds". Open source: "our code, model weights, and example scripts are open-sourced". Transfer: applies only if a LEGO simulator with brick physics exists; it changes appearance, not dynamics, so it does not solve brick-state correctness, and 7B inference is heavy on A6000s.

---

## Card 9. 2025–2026 work on generating manipulation videos as data

### 9a. VAG: Dual-Stream Video-Action Generation (Apr 2026)
"a unified flow-matching-based dual-stream framework that jointly generates video and action under visual and language conditioning"; "Our video model is post-trained on Cosmos-Predict2 (2B-Video2World) [1], which produces 480P videos with 10 Hz. The number T of video-action frames to be predicted is 93 ... approximately the next 10 seconds ... height H of 432 and the width W of 768"; action branch is a 1D U-Net conditioned on adaptive-3D-pooled video latents; "35 denoising steps". Data: AgiBot G1 "1794 video-action pairs for training and 200 for testing, where the action dimension D is 16"; LIBERO; "trained for 40,000 iterations" (AgiBot), 20,000 (LIBERO). Downstream: "when VAG-generated data is used for pretraining, the downstream VLA success rate increases from 35% to 55% (+20% absolute)"; action-prediction success defined as "the error in each dimension is below 0.2". Compared against two-stage "video generation followed by action regression" (ResNet50 IDM, AnyPos) and found better. GPUs: not stated. Code: not stated. Relevance: this is the exact setting (b) model shape (Cosmos-Predict2 2B, ~1.8k paired episodes) with a reported policy gain.

### 9b. EVA: IDM-reward alignment of video world models (Mar 2026)
Identifies "the executability gap: visually coherent rollouts may still violate rigid-body and kinematic consistency, producing unstable or infeasible control commands when decoded by an IDM." Base: "Wan2.1-14B backbone [42] and incorporate diffusion forcing", initialized from the Large Video Planner; IDM "a convolutional backbone extracts spatial features, a spatial softmax layer converts each channel into a 2D coordinate, and an MLP maps these coordinates to actions"; reward penalizes joint velocity/acceleration/jerk and out-of-bound actions; "GRPO ... G=8 rollouts per prompt ... LoRA with rank 32 ... 8 NVIDIA A800 GPUs with a total batch size of 32". Real data: "50 human-teleoperated demonstrations for each of five tasks (250 trajectories total) and use them to SFT both the embodiment-specific video generator and the IDM." Results: "improves Kinematic plausibility by +20.9% ... EVA achieves an 83.8% Perfect execution rate"; RoboTwin 2.0 Table 2 (successes/20, subset shown): EVA w/o RL 18/0/6/4/8/4/18/5, average 8/20; EVA with RL 20/3/12/6/9/5/20/4, average 13/20; π0 average 5/20. Real Table 3: 20 trials per task; numbers not extracted. Relevance: video-plus-IDM on 250 robot trajectories works with a 14B model on 8×A800; the physics-executability gap is the named failure and needed RL alignment.

### 9c. DreamZero / World Action Models are Zero-shot Policies (Feb 2026)
"a 14B robot foundation model built upon a pretrained image-to-video diffusion backbone (Team Wan, 2025) ... jointly generate future frames and actions"; "shifts action learning from dense state–action imitation to inverse dynamics". Data: "7.2K episodes (∼500 hours)" AgiBot G1. Claims: "over 2× improvement in generalization to new tasks and environments compared to state-of-the-art VLAs"; "video-only data from humans (12 minutes) or other robots (20 minutes) yields a relative improvement of over 42% on unseen tasks"; "adapts to an entirely new robot (YAM) with only 30 minutes of play data". Cost: "A naive implementation of DreamZero on a single GPU requires approximately 5.7 seconds per action chunk"; ~7 Hz after multi-GPU optimizations. "policy performance is fundamentally tied to video generation quality." "We open-source our model weights, inference code". Relevance: video-only human clips (12 minutes) help a WAM already trained on 500 robot hours; not evidence for a 100-video-only regime.

### 9d. EgoEngine (Jun 2026)
"From Egocentric Human Videos to High-Fidelity Dexterous Robot Demonstrations": Aria Gen2 glasses ("synchronized RGB frames and per-frame 3D hand poses with 21 hand keypoints"), digital-twin reconstruction, "retargets human motion into a robot reference trajectory, then refines it in simulation to produce an executable robot trajectory supervised by reference object motion"; visual branch renders the robot into the video. Results: "the first zero-shot visuomotor dexterous policy learning from egocentric human videos without real-robot demonstrations"; "matches or exceeds real robot demonstrations on 2 of 4 tasks"; "Human videos and Phantom videos achieve near-zero success and mostly fail due to grasping pose, indicating that visual conversion alone is insufficient for dexterous policy learning"; throughput "2.88 demos/hour on a single RTX 4090". Baseline video editors: "VACE [42], built on WAN2.1 [62]". Failures: "unstable pinch grasps and contact-timing mismatch". Relevance: strongest setting-(a) result, but it uses simulation refinement with object-motion supervision, not a video generator. That is the pattern to copy if a LEGO simulator is available.

### 9e. EgoDemoGen (Sep 2025)
Generates "paired observation–action demonstrations under novel egocentric viewpoints" by transferring the trajectory geometrically (IK-filtered) and synthesizing the view with "a conditional video generation model that fuses a novel-viewpoint reprojected scene video and a robot motion video rendered from the transferred trajectory ... trained with a self-supervised double reprojection strategy without requiring multi-viewpoint data." Gains: "absolute gains of +24.6% and +16.9% in simulation and +16.0%" real (sentence truncated in source). Base model, data size, compute: not extracted. Relevance: setting (b) augmentation where actions stay ground-truth and only pixels are generated; this sidesteps action-labelling error entirely.

### 9f. RoboEdit (Aug 2026) and H2R-Grounder (Dec 2025): human-to-robot video translation
RoboEdit: "transforms human manipulation videos into action-consistent, physically plausible robot videos with aligned 3D hand states"; "RoboEdit-Trans builds on NovaEdit's Wan2.1-VACE-1.3B backbone"; automatic pairing pipeline (HaMeR, SAM 2, TRELLIS, FoundationPose, VGGT) yields "174,547 aligned human/robot video pairs totaling over 14.1M paired frames, constructed from 24,197 human-interaction clips"; 3D Robot-State Decoder; downstream "trajectory-reproduction success rates of 71% with the Panda gripper". H2R-Grounder: "does not require any paired human–robot videos for training – only a set of unpaired robot videos"; "We fine-tune the Wan 2.2 TI2V-5B model"; LoRA on Q/K/V only; "fine-tune our in-context model for 200 steps with a mini-batch size of 4, using 8 NVIDIA H200 GPUs"; trained "only on the DROID indoor dataset"; evaluated by VLM and human preference (first-rank visual quality 61.4%, physical plausibility 63.6%); limitations "supports only single-hand to single-arm translation" and "produces only Franka-style outputs". Relevance: converts setting (a) footage into robot-looking video, but neither paper trains a policy on the translated videos and reports task success; RoboEdit's 71% is trajectory reproduction.

### 9g. EgoWAM (Jul 2026)
Controlled co-training of a policy on robot data plus in-the-wild egocentric human data with a world-prediction head: "Pixel-based prediction transfers weakly, while DINO and 3D flow yield substantial gains: DINO improves out-of-distribution object and scene generalization by up to 4×, and 3D flow improves in-domain performance by 20–30%"; "human data often degrades BC yet yields large WAM gains"; 1800 real rollouts. Relevance: direct evidence that pixel-level targets are the wrong interface for human-to-robot transfer; a LEGO project aimed at policy learning should predict tracks/flow or features, not RGB.

### 9h. Cosmos Policy (Jan 2026)
"adapting a large pretrained video model (Cosmos-Predict2) into an effective robot policy through a single stage of post-training on the robot demonstration data collected on the target platform, with no architectural modifications"; base "Cosmos-Predict2-2B-Video2World"; actions injected as latent frames. LIBERO "98.5%", RoboCasa "67.1%"; real ALOHA tasks with "80 demos", "15 demos", "45 demos" and "93.6%" average. Relevance: with ~100 robot episodes (setting b), fine-tuning the 2B video model directly as a policy is a documented alternative to generating data at all.

### 9i. MIND-V (Dec 2025)
Hierarchical long-horizon video world model: "MVG is initialized from the pretrained CogVideoX-5B"; Bridge V2 at "480×640 pixels and 37 frames per subtask"; "SFT for 30,000 steps ... followed by GRPO post-training for 1,500 iterations" with a "Physical Foresight Coherence (PFC) reward, which employs V-JEPA2 as a physics referee". Downstream: OpenVLA-OFT fine-tuned on "300 expert trajectories per task", then augmented with generated visual goals (gain numbers not extracted). Baseline failures: "logical hallucinations, physical implausibility (e.g., spontaneous object disappearance), and inaccurate semantic grounding".

### 9j. Benchmarks: RBench/RoVid-X (Jan 2026) and H2R-Bench (Aug 2026)
RBench evaluates 25 video models on robot videos: "from Wan 2.1 (Rank 14, 0.399) to Wan 2.6 (Rank 1, 0.607)"; "models fine-tuned on specific robot entities (e.g., Vidar, UnifoLM) struggle significantly, ranking at the bottom of the benchmark ... domain-specific data ... cannot fully compensate for the deficit in 'World Knowledge'"; "models consistently score higher on coarse-grained locomotion tasks ... than on fine-grained manipulation"; DreamGen(gr1) and DreamGen(droid) are on the leaderboard (rank not extracted). RoVid-X: "4 million annotated video clips". H2R-Bench evaluates human-video-to-robot-video transfer over six families including "insertion and assembly"; "Seedance 2.0 ranks first with H2RCore scores of 77.3 for the Parallel-Jaw Gripper and 84.6 for the Dexterous Hand"; open models weaker; "most of the separation comes from contact transfer and embodiment correctness". Relevance: the only public benchmark family containing assembly; both say fine-grained contact is where current generators fail, and narrow fine-tuning hurts world knowledge.

### 9k. Video2Sim2Real (Jun 2026) and EgoDex (May 2025)
Video2Sim2Real: single RGB-D human video to digital twin, keyframe (contact/interaction/detachment) refinement, IL for geometry plus residual RL for contacts; Kinova + 16-DoF Leap Hand. No generative video. EgoDex: Apple Vision Pro egocentric dataset with 3D hand pose; a data source, not a method. Cited as the non-generative alternatives for setting (a).

---

## Comparison table

| Work | Base model (size) | Fine-tune data | Conditioning | Video → robot data | Best downstream number | Setting (a) human hands | Setting (b) robot + proprio | Code |
|---|---|---|---|---|---|---|---|---|
| DreamGen 2025 | WAN2.1 (variant n.s.); released path Cosmos-Predict2(.5) 2B/14B | 100 GR1 trajs (bench); 44–68 SO-100 trajs; 2,884 GR1; LoRA r4, lr 1e-4 | text + first frame | IDM (DiT+SigLIP-2, 2 frames→chunk) or LAPA latents | GR1 low-data avg 37→46; novel behaviors 11.2→43.2 (NT only); RoboCasa NT-only 20.6 | no (needs IDM in robot action space) | yes, exactly | pipeline yes; GR1 weights n.s. |
| GR00T N1 2025 | image-to-video models | 88 h teleop → 827 h generated | text + first frame | LAPA + IDM | n.s. here | no | yes | model yes |
| Gen2Act 2024 | VideoPoet, zero-shot (>270M videos) | none for video; RT-1 + ~400 teleop for policy | text + first frame | generated human video conditions policy; point-track aux loss | OTG 58, MTG 30 vs 25/0 (Vid2Robot) | yes, but needs large robot set | n/a | no |
| Dreamitate 2024 | SVD, attention layers only | 356–727 human demos/task, 768×448, 16k steps, batch 4 | stereo first frames | MegaPose 6D tool track → open-loop SE(3) | 92.5 vs 55 (rotation); 92.5 vs 12.5 (sweep) | yes, with a tracked rigid tool | n/a | "will release" |
| UniPi 2023 | 1.7B video U-Net + T5-XXL | 7.2k Bridge; 256 TPU-v4, 2M steps | text + first frame | small CNN IDM | plan success 77.1 vs 72.6 (surrogate) | no | in principle; compute out of reach | no |
| VLP 2023 | UniPi-class | ~20k LT trajs; 1,200 ALOHA demos | text + first frame | goal-conditioned policy per frame | LT long-horizon up to 92 vs ≤26 | no | compute out of reach | no |
| AVDC 2023 | pixel U-Net from scratch, 128² | 165–240 videos (sim); 40k Bridge + 20 human; 4×V100, 1–2 days | text + first frame | GMFlow + depth → rigid transform + grasp sampler | Meta-World 43.1; real 2/10 | partly; 8/10 real failures | yes | yes |
| VPP 2024 | SVD 1.5B, 16×256² | 193k SSv2 + 179k OXE + own; 8×A100 2–3 d | text + first frame | video-model features → diffusion policy | CALVIN 3.35→4.33; Franka unseen 0.73 vs 0.46 | video prior only | yes | supp. |
| UVA 2025 | MAR-B (0.5B) | 1,500 UMI episodes; 3,175 human videos help | history obs | joint latent; IDM mode | PushT-M +20; real +15/+40 vs DP-UMI | no | yes; usable as IDM | yes |
| LAPA 2024 | LWM 7B + VQ-VAE LAM | Bridge/OXE/SSv2; 100–150 trajs/task fine-tune; 8×H100 34 h | frame pairs | latent labels → fine-tune to real actions | +6.22 vs OpenVLA; SSv2-only > OpenVLA(Bridge) | latent labels yes; executable only after robot fine-tune | yes | yes |
| Genie 2024 | 11B; LAM 300M, 8 codes | 30k h platformers; RT1 videos | latent action | LAM as IDM | CoinRun oracle-equivalent with 200 expert samples | background | background | no weights |
| Im2Flow2Act 2024 | AnimateDiff + SD (LoRA r128) for flow | human videos (count n.s.) + 4,800 sim play | text + first frame + keypoints | object flow → sim-trained flow policy | 81% avg real, no real robot data | yes (flow, not pixels) | n/a | site |
| Track2Act 2024 | DiT over point tracks | 400k web clips; ~400 Spot demos | first + goal image | tracks → rigid transform + residual policy | CG 55, TG 40 vs 0/0 (GC-BC) | yes (tracks) + ~400 robot demos | n/a | n.s. |
| ATM 2024 | track transformer; 4×A100 | 100 human videos + 10 robot demos/task | first frame + text + points | tracks → track-guided policy | human-to-robot 63/63/60 vs 0/10/30 | yes (best small-data evidence) | yes | yes |
| Cosmos-Transfer1 2025 | Cosmos-Predict1-7B + ControlNets; 1024 H100 2–4 wk/branch | sim renders | depth/seg/edge/blur | appearance transfer only | no policy number | no | only with a simulator | yes |
| VAG 2026 | Cosmos-Predict2 2B, 93 frames 432×768 | 1,794 AgiBot pairs, 40k iters | text + first frame | joint video+action generation | VLA 35→55 with generated pretraining data | no | yes (closest single-model recipe) | n.s. |
| EVA 2026 | Wan2.1-14B + diffusion forcing; 8×A800 | 250 real trajs (50×5) | text + obs history | CNN IDM + GRPO executability reward | RoboTwin avg 8/20→13/20 | no | yes; names the executability gap | n.s. |
| DreamZero 2026 | Wan 14B WAM | 500 h AgiBot; 12 min human video helps | text + obs + proprio | joint | >2× VLA generalization | only on top of 500 robot hours | yes | yes |
| EgoEngine 2026 | no generator; digital twin + sim refine | Aria Gen2 human videos | n/a | retarget + object-motion-supervised refinement | zero-shot real dexterous policy; ≥ real demos on 2/4 tasks | yes (needs sim) | n/a | n.s. |
| EgoDemoGen 2025 | conditional video model (n.s.) | single-view robot demos | rendered robot motion + reprojected scene | actions transferred geometrically; pixels generated | +24.6/+16.9 sim, +16.0 real | no | yes (actions stay GT) | n.s. |
| RoboEdit / H2R-Grounder | Wan2.1-VACE-1.3B / Wan2.2 TI2V-5B (8×H200, 200 steps) | 174k pairs / DROID only | masked human video + cues | robot-look video + 3D state decoder | 71% trajectory reproduction / preference study | translates (a) into (b)-looking video; no policy success | n/a | n.s. |
| EgoWAM 2026 | HPT trunk | robot + in-the-wild ego human | obs | world head (pixel/DINO/3D flow) | pixel weak; DINO 4× OOD; flow +20–30 ID | yes (co-training) | yes | n.s. |
| Cosmos Policy 2026 | Cosmos-Predict2-2B | 15–80 demos/task real | obs + text + proprio | model is the policy | LIBERO 98.5; ALOHA 93.6 | no | yes (skip data generation) | site |

n.s. = not stated.

---

## The two settings, explicitly

### (a) Human-hand phone videos (~100 clips), no actions

Applicable precedents: Dreamitate (fine-tune SVD on 356–727 human demos per task, execute by tracking a rigid tool), Gen2Act (human video generation as a policy cue, but with a large robot dataset behind it), ATM / Im2Flow2Act / Track2Act (human videos train a track or flow predictor; 100 human videos + 10 robot demos is ATM's exact regime), LAPA (latent action labels from human video, then a robot-labelled mapping), EgoEngine / Video2Sim2Real (no generator; digital twin plus simulation refinement), RoboEdit / H2R-Grounder (translate hand video into robot-looking video).

What the human case gives for free: the video prior already generates hands (Gen2Act) and phone footage is cheap to collect at 100+ clips.

What it does not give: any action label. Every published path from generated human video to executable actions goes through one of: a tracked rigid tool with a CAD model (Dreamitate), a track/flow predictor plus ≥10 robot demos (ATM) or ~400 robot demos (Track2Act) or a sim-trained flow policy (Im2Flow2Act), a latent-action model plus 100–150 robot-labelled trajectories (LAPA), or hand-pose retargeting with simulation refinement (EgoEngine). Gen2Act, LAPA, EgoEngine and EgoWAM each report that hand realism, grasp timing, or pixel-level targets are the failure point, which is exactly the brick-seating moment. No paper trains a policy purely on generated human-hand video and reports task success.

### (b) Robot-executed videos with proprioception (~100 episodes)

Applicable precedents: DreamGen (100 GR1 trajs is its benchmark training size; 10–25 real trajectories per task in the low-data experiments; 44–68 SO-100 trajectories), GR00T N1 (88 h → 827 h), VAG (Cosmos-Predict2 2B on 1,794 pairs, joint video+action, VLA 35→55), EVA (250 trajectories, IDM + executability reward), EgoDemoGen (generate pixels, keep ground-truth actions), Cosmos Policy (skip generation, fine-tune the 2B video model as the policy on 15–80 demos), UVA (IDM mode).

What the robot case simplifies: action labels exist, so an IDM in the robot's own action space can be trained on the same 100 episodes (DreamGen: "IDM is conditioned on two image frames and is trained to predict action chunks"; the repo ships `idm_training.py`); the IDM can be validated by replaying its actions on the real robot or in sim (DreamGen Figure 7); co-training is a known 1:1 recipe; executability of decoded actions can be scored (EVA's velocity/acceleration/jerk penalty) without human labelling; alternatively actions can be kept ground-truth and only pixels generated (EgoDemoGen).

What it does not simplify: (1) brick-state correctness. No IDM or filter in these papers checks object state; DreamGen filters nothing, the cookbook filters on a 1–5 VLM physics score, and RBench/H2R-Bench report that fine-grained contact is where generators fail. A stud that lands one row off is physically plausible and instruction-following by every metric above. (2) Physical plausibility of generated frames. EVA names the executability gap on a 14B model after SFT on 250 trajectories; DreamGen: "most of the bottleneck is from the quality of the neural trajectories"; Diffusion Policy regressed on Hammering (35→15) with neural trajectories (DreamGen Table 5). (3) Scale. DreamGen's headline numbers used 240k videos on 1500 L40s for 54 h; the low-data real-world gains (+9 points average on GR1 with 300 generated per task) are the realistic reference.

---

## Three most transferable recipes

### Recipe 1 (setting b, start here): DreamGen-small on Cosmos-Predict2.5-2B
1. Collect ~100 robot episodes with a fixed camera, proprioception, and a per-episode brick-placement label (target stud position, success).
2. Post-train Cosmos-Predict2.5-2B Video2World with the shipped GR1 config (`predict2_video2world_training_2b_groot_gr1_480`, single GPU) or Wan2.1 with LoRA rank 4, lr 1e-4, batch ≤8, ~200 epochs (the SO-100 setting). Measure instruction following and physics alignment on 20 held-out episodes with the DreamGen Bench scripts before generating anything.
3. Train the IDM on the same episodes (`scripts/idm_training.py`; new embodiment needs `modality.json`/`stats.json`) or a UVA-style masked model. Validate the IDM by replaying decoded actions on the real robot from real videos; this is the one number that must be high before generation.
4. Generate 300–1,000 videos from new initial frames (DreamGen's per-task count) with prompts enumerated by an LLM over feasible brick-target combinations (GR00T N1), score with Cosmos Reason 2 best-of-N (cookbook), and add a brick-state check (risk 4).
5. Co-train Diffusion Policy or GR00T N1 1:1 on real + neural trajectories; compare against real-only and against EgoDemoGen-style geometric augmentation with ground-truth actions. Expected effect from precedents: high single digits to ~+10 points at 10–25 real demos per task; regression is possible.
Compute: 2B post-training runs on one GPU per the NVIDIA docs; 14B recipes (EVA, DreamZero) need 8×A800/H100 and are out.

### Recipe 2 (setting a): ATM-style track model, not a pixel generator
Use the ~100 phone videos to train a track transformer (CoTracker pseudo-labels, 50% patch masking, track length 16, 4 GPUs) and collect ≥10 robot demos per task for the track-guided policy. This is the only published configuration matching "100 human videos + 10 robot demos" with a real-robot result (63/63/60% vs 0/0/13% without human videos). Add Im2Flow2Act's object-centric keypoints (brick and hand boxes from Grounding DINO) so the predicted motion is of the brick, not the hand (+30 points over grid tracks under cross-embodiment in Im2Flow2Act). Known gap: 2D tracks are ambiguous for the vertical press; a depth camera at execution time (AVDC/Track2Act) is required.

### Recipe 3 (setting a, if a generator is the research object): Dreamitate-style with a trackable end-effector
Fine-tune SVD or Wan (attention layers only or LoRA) on the phone videos at ~768×448, batch 4, ~16k steps, first-frame conditioned, then recover actions by tracking a rigid, CAD-known hand-held tool or a marked gripper with MegaPose or FoundationPose (RoboEdit's pipeline uses FoundationPose + HaMeR). Dreamitate needed 356–727 demos per task; with ~100 the 1/3-data ablation (~124 demos, "remains stable") is the closest evidence and it is for one task. Without a rigid tool, replace tracking by HaMeR hand pose + brick pose and expect Gen2Act's and EgoEngine's grasp-pose failures.

Not recommended first: RoboEdit/H2R-Grounder translation of hand videos into robot videos (no policy-success evidence), Cosmos-Transfer1 (needs a simulator, 7B), any 14B video model (EVA, DreamZero) on 4×A6000.

---

## Open risks

1. **Physical plausibility of generated frames.** Named by UniPi ("hallucination"), VLP ("objects would infrequently appear or disappear"), ATM ("robots disappear from the image"), AVDC ("25% of the failures were caused by the discontinuity of video generation"), EVA ("executability gap"), RBench ("Manipulation Gap ... fine-grained contact dynamics"), DreamGen ("most of the bottleneck is from the quality of the neural trajectories"). Mitigations in the literature: LoRA to keep the prior (DreamGen), best-of-N with a VLM physics critic (cookbook; "a proxy ... not a ground-truth oracle"), RL alignment with an IDM smoothness reward (EVA) or V-JEPA2 latent-consistency reward (MIND-V). None checks discrete object state.

2. **Action-labelling accuracy.** IDM: DreamGen gives no accuracy number; VPT-style pseudo-labels "show large errors" (ATM); an IDM trained on 100 episodes is narrow. Latent actions: LAPA "underperforms ... fine-grained motion generation tasks like grasping", "early grasping", and absorbs camera motion from egocentric/phone footage. Tool tracking: fails "when the end-effector is heavily occluded" (Dreamitate); small objects give "large errors in the 3D space" (AVDC). For LEGO the critical label is the final press, which is low-motion and occluded.

3. **Distribution collapse and narrow priors.** RBench: entity-specific fine-tunes rank "at the bottom" because they lose world knowledge; DreamGen's DROID-only model generalized scenes but "made mistakes on fine-graed details (e.g. grasping)"; Gen2Act argues robot-specific fine-tuning "subtracts the benefits of generalization". With 100 videos of one table and one brick set, generated data will cover the training distribution and can regress a policy (DreamGen DP Hammering 35→15). Mitigation: new initial frames and LLM-enumerated prompts, which DreamGen calls manual and "operational overhead."

4. **Evaluation.** Instruction-following/physics scores from Qwen2.5-VL/GPT-4o correlate with humans at >0.9 (DreamGen) but the evaluator "can occasionally hallucinate, especially when evaluating physical realism"; H2R-Bench and RBench score contact and embodiment, not stud-level placement. A LEGO-specific automatic check (final-frame brick pose vs target stud, e.g. FoundationPose on a known brick CAD, or an overhead stud-grid detector) is in no paper and must be built before generated videos can be trusted as data. Downstream, only real-robot success on held-out placements counts; DreamGen's numbers come from 10 rollouts per checkpoint with partial credit, so variance is large.

5. **Compute and scale.** Gains scale log-linearly with generated count (DreamGen) and the large gains used 240k videos; per-video generation time is not stated by DreamGen; VLP ~1 hour per environment; AVDC 18 s per plan on a 3080Ti; DreamZero 5.7 s per chunk naive. On 4×A6000, plan for hundreds to low thousands of clips, the regime where DreamGen saw +9 to +40 points on low-data real tasks, not the 333× regime.

## What was not verified
DreamGen Table 4 averages and π0 rows of Table 5; Dreamitate Push-Shape metrics; EVA Table 3 real-robot numbers; EgoDemoGen base model and data sizes; VLP row labels in Table 2; Im2Flow2Act human-demo counts and Tables 1–2; RBench rank of DreamGen(gr1)/(droid); GR00T-Dreams release of GR1-post-trained video weights and IDM checkpoints. None of these change the recommendations above.
