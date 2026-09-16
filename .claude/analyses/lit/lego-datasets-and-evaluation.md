# LEGO assembly video: datasets to train on / compare against, and how to evaluate generated demonstrations

Evidence cards, primary sources only (arXiv abstract/HTML, project pages, GitHub READMEs, dataset cards). "Not stated" means the fetched source is silent; "unverified" means a secondary source or memory. Compiled 2026-09-16 for the decision on fine-tuning a video generator on phone videos of LEGO assembly (human hands, or possibly the lab's robot arm) to produce synthetic demonstrations.

Sections: A. datasets and benchmarks (A1 LEGO-specific, A2 human assembly video, A3 hand-object pose, A4 robot datasets, A5 sim/real assembly benchmarks, A6 robot LEGO works). B. evaluating generated video as training data. C. what to use in a month, minimal protocol (human and robot cases), honest risks.

---

## A. Datasets and benchmarks

### A1. LEGO-specific

#### Break and Make / LTRON (2022, ECCV)
- Source: https://arxiv.org/abs/2207.13738 ; https://github.com/aaronwalsman/ltron
- What: "an agent is given a LEGO model and attempts to understand its structure by interactively inspecting and disassembling it", then "must prove its understanding by rebuilding the model from scratch using low-level action primitives" (abstract).
- Size: "1727 ethically-sourced fan-made LEGO models" from the Open Model Repository, "from 5 to 7302 individual bricks", "1790 distinct brick shapes"; sliced-model splits "2 Brick Slices: 136072 train, 2000 test scenes; 4 Brick Slices: 61514 train, 2000 test; 8 Brick Slices: 28094 train, 2000 test scenes" (arXiv HTML). Hours/clips: not applicable, it is a simulator, not video.
- Views and cameras: synthetic renders; "The table is rendered at 256x256 pixels and the hand is rendered at 96x96 pixels"; a Rotate Camera action exists. No real camera.
- Annotations: brick identity and 6D pose from the LDraw scene state; action space "Disassemble, Assemble, Pick, Rotate Brick, Rotate Camera, Switch Phase"; connection edges in the scene graph. No human hand. Text: not stated.
- Licence: "The ltron software package is provided under the MIT license." LDraw parts library and OMR files: "Creative Commons Attribution 2.0"; LDCad: "free for personal/educational use".
- Access: `pip install ltron` then `ltron_asset_installer` (~3 GB); github.com/aaronwalsman/ltron ; training code github.com/aaronwalsman/ltron-torch-eccv22. Follow-up "Learning to Build by Building Your Own Instructions" (https://arxiv.org/abs/2410.01111): "procedurally built LEGO vehicles that contain an average of 31 bricks each and require over one hundred steps to disassemble and reassemble".
- Relevance: a source of ground-truth brick connectivity graphs and rendered assembly sequences; can serve as a checker for brick connectivity of a predicted assembly state, not as video training data. Follow-up "Learning to Build by Building Your Own Instructions" (ECCV 2024, Springer link above) is the same group.

#### StableLego (2024, RA-L) and StableText2Lego / BrickGPT (2025, ICCV, "LegoGPT")
- Source: https://github.com/intelligent-control-lab/StableLego ; https://arxiv.org/abs/2505.05469 ; https://arxiv.org/html/2505.05469v1 ; https://avalovelace1.github.io/BrickGPT/ ; https://huggingface.co/datasets/AvaLovelace/StableText2Brick
- What: StableLego is "the implementation of estimating the structural stability of block stacking assembly" with a dataset of LEGO layouts and stability scores; it uses an optimisation with the Gurobi solver over contact forces (GitHub README). LegoGPT/BrickGPT generates text-conditioned brick structures with "an efficient validity check and physics-aware rollback during autoregressive inference, which prunes infeasible token predictions using physics laws and assembly constraints" (abstract).
- Size: "StableText2Brick, containing over 47,000 brick structures of over 28,000 unique 3D objects accompanied by detailed captions" (project page). Brick library: "1x1, 1x2, 1x4, 1x6, 1x8, 2x2, 2x4, and 2x6" bricks, axis-aligned, 1 unit tall (paper). StableLego dataset counts: not stated on the README.
- Views and cameras: no video; voxel/brick layouts plus captions.
- Annotations: brick layout, stability score per brick, caption. No hands, no video.
- Key numbers: LegoGPT Table 1: without rollback "100% valid, 24.0% stable"; with rejection sampling plus rollback "100% valid, 98.8% stable" (arXiv HTML v1). The stability method "builds on ... StableLego: Stability Analysis of Block Stacking Assembly" with "a structural force model F which consists of a set of candidate forces (e.g., pulling, pressing, supporting, dragging, normal, etc.)".
- StableLego paper numbers (https://arxiv.org/abs/2402.10711): "more than 50k of different objects from 55 common object categories with their Lego layouts"; "20x20x20 grid world"; "a mix of valid and invalid brick layouts". StableText2Brick dataset card: 47,389 rows, "No images are included".
- Licence: StableLego "MIT License" (README); StableText2Brick dataset card "MIT", paper CC BY 4.0.
- Access: public GitHub and Hugging Face.
- Relevance: this is the lab's own stability checker (Changliu Liu group). It is the natural "brick connectivity / stability" oracle for judging the end state of a generated assembly video once the state is lifted to a brick layout. It says nothing about the video itself.

#### Prompt-to-Product / BrickMatic (2025, arXiv 2508.21063)
- Source: https://arxiv.org/abs/2508.21063 ; https://arxiv.org/html/2508.21063
- What: "an automated pipeline that generates real-world assembly products from natural language prompts", using "LEGO bricks as the assembly platform" and "a bimanual robotic system" (abstract). Authors include Ruixuan Liu, Deva Ramanan, Jun-Yan Zhu, Jiaoyang Li, Changliu Liu.
- Stability check: "we solve its force distribution F by formulating the problem into a nonlinear program"; a brick with score si = 0 "fail[s] equilibrium or exceed[s] friction capacity FT" (HTML).
- Robot: "two Yaskawa GP4 robot arms" with force-torque sensors and an "Eye-in-Finger (EiF)" end-effector with "an endoscope camera inside the tooltip" (HTML).
- Numbers: Table II, BrickMatic "1/1" on four designs (Faucet, Fish, Vessel, Guitar) vs baseline dual-arm 0-1/5; survival length 14-36 bricks vs 7.8-33.7 (HTML).
- Data/code: no release statement found in the HTML.
- Relevance: the lab's robot LEGO stack; the downstream "real robot" test for generated robot-arm videos would run on this hardware.

#### Simulation-aided LfD for Robotic LEGO Construction (2023, arXiv 2309.11010) and Robotic LEGO Assembly and Disassembly from Human Demonstration (2023, arXiv 2305.15667)
- Source: https://arxiv.org/html/2309.11010v2 ; https://arxiv.org/abs/2305.15667
- What: the closest precedent to "learn LEGO assembly from human video": a FANUC LR-mate 200id/7L learns brick placement sequences from a human demonstration captured by "A Realsense L515 ... integrated into the EOAT" (RGB-D, third-person, single camera). Extracted per brick operation: brick type, discrete 3D position, binary orientation; hand pose is not extracted.
- Verification: a ROS Gazebo digital twin "executes the highest ranked brick candidate and compares the resulting LEGO state with the observations (i.e., color and depth) in the real environment".
- Numbers: LfD only "63.15%-76.7%"; with simulation verification "86.3%-100%" across structures (AI, RI, Human, Chair, Spiral, Bridge, Pyramid, Temple; 10 trials each) (HTML v2).
- Data/code: no release statement found.
- Relevance: establishes (a) the lab already parses human LEGO video to brick placement sequences and (b) a digital twin is the accepted guard against perception errors. Both transfer directly to checking generated videos.

#### LEGO Co-builder (2025, arXiv 2507.05515)
- Source: https://arxiv.org/abs/2507.05515 ; https://arxiv.org/html/2507.05515v2
- What: "a hybrid benchmark combining real-world LEGO assembly logic with programmatically generated multimodal scenes" for VLM assistants.
- Size: Table 1: manuals "65", sessions "397", scenes "5,612", objects "4,784", states "2,716", instruction steps "10,428" (4,814 identification, 5,614 assembly), "35,612" samples.
- Views: rendered images, not photographs; no video.
- Annotations: per-step image plus textual instruction; object detection and state detection labels.
- Licence: CC BY-NC-SA 4.0 (arXiv badge). Access: "We release the benchmark, codebase, and generation pipeline" (URL not given in the HTML).
- Numbers: GPT-4o object detection 98.16% accuracy, theme identification 37.52% F1, state detection 40.54% F1.
- Relevance: a source of step text and state labels for LEGO sets; the state-detection weakness (40.54% F1) is a warning about using an off-the-shelf VLM as the sole judge of assembly state.

#### AssemblyHands (2023, CVPR)
- Source: https://assemblyhands.github.io/ ; https://arxiv.org/abs/2304.12301
- What: "3.0M annotated images, including 490K egocentric images" sampled from Assembly101 (take-apart toys, not LEGO), with 3D hand pose at "an average keypoint error of 4.20 mm".
- Views: "synchronized egocentric and exocentric images".
- Annotations: 3D hand keypoints (count not stated on the page); action classification task.
- Licence: "Creative Commons Attribution-NonCommercial 4.0 International License". Access: data and code released 2023-05-25 via the toolkit (github.com/facebookresearch/assemblyhands-toolkit).
- Relevance: the largest egocentric 3D hand-pose set on an assembly activity; candidate for training/validating a hand-pose consistency scorer.

#### MEPNet: Translating a Visual LEGO Manual to a Machine-Executable Plan (2022, ECCV), image-only
- Source: https://arxiv.org/abs/2207.12572 ; https://github.com/Relento/lego_release
- What: parses rendered manual-step images into 3D component poses; releases synthetic and re-rendered real-set manual datasets.
- Size: synthetic "8000 manuals for training, 10 sets for validation, and 20 sets for testing ... 200K individual steps in training"; real manuals: 11 Classics and 5 Architecture sets. Rendered via LPub3D; no cameras, no hands.
- Annotations: 3D pose (translation and rotation) of added components per step; 2D keypoints and masks; step structure with submodules.
- Licence: repo "MIT license". Access: SharePoint folder linked from the README.
- Relevance: step-conditioned 3D pose targets for LEGO manual steps, usable as structured conditioning; no video.

#### BrickNet (2026, CVPR 2026), LDraw only
- Source: https://arxiv.org/abs/2604.22984 ; https://kulits.github.io/BrickNet
- What: graph-backed autoregressive generation over LDraw build sequences with "a typed connector system with precise positions" annotated on the LDraw part library.
- Size: "over 100,000 human-designed LDraw brick objects and scenes". No video. Licence: "available for research purposes"; exact licence not stated.
- Relevance: a connectivity-typed part library and generator; a possible connectivity checker for arbitrary parts beyond the 8 axis-aligned bricks of LegoGPT.

#### MobileBrick (2023, CVPR), real RGB-D video of finished LEGO models, not assembly
- Source: https://arxiv.org/abs/2303.01932 ; http://code.active.vision/MobileBrick/
- What: 3D reconstruction benchmark using LEGO models with CAD-exact ground truth; "153 LEGO models"; handheld mobile RGB-D sequences; annotations are camera poses, masks, exact geometry; no hands, no steps.
- Licence: "CC BY-NC-ND 4.0".
- Relevance: real LEGO appearance and geometry with exact CAD alignment; static models only.

#### Other LEGO video datasets
- No public dataset of real human LEGO assembly VIDEO with hand or brick annotations was found (two independent search passes: "LEGO assembly dataset video", "brick assembly benchmark", "LEGO building video dataset hand", "Brick-by-Brick", "LEGO-Net", "LEGO human demonstration video dataset"). "Brick-by-Brick" (NeurIPS 2021) is an RL construction task and "LEGO-Net" is a room-rearrangement paper. Neural Assembler (https://arxiv.org/abs/2404.16423) uses synthetic multi-view images of block models. The LEGO items above are simulators (LTRON), layout datasets (StableLego, StableText2Brick, BrickNet), rendered step images (MEPNet, LEGO Co-builder), static-model scans (MobileBrick), or unreleased lab captures (SaLfD, Prompt-to-Product). A phone-video LEGO assembly set with hand and brick-state labels would be the first of its kind; state this as a contribution and as a gap.

### A2. Human assembly video datasets (non-LEGO)

#### IKEA-ASM (2021, WACV)
- Source: https://arxiv.org/abs/2007.00394 ; https://github.com/IkeaASM/IKEA_ASM_Dataset
- What: "a three million frame, multi-view, furniture assembly video dataset" of people assembling four IKEA items.
- Size: "371 unique assemblies", "1113 RGB videos and 371 depth videos (top view)", "3,046,977 frames (~35.27h)", "48 human subjects", "five different environments".
- Views and cameras: third-person, "three Kinect V2 cameras" at "~24 fps", "three RGB views, one depth stream".
- Annotations: "33 atomic actions" with "16,764 annotated actions"; "2D human joint annotations in the COCO format" plus "pseudo-ground-truth 3D annotations"; instance segmentation of parts on "1% of the video frames"; part tracking. Hand pose: not stated (body pose only). Object 6D pose: not stated. Text: not stated.
- Licence: README states MIT. Access: Google Drive link in README (~240 GB).
- Relevance: closest analogue for multi-view third-person assembly video with action segments; no hand or part 6D pose.

#### IKEA Video Manuals / IKEA Manuals at Work (2024, NeurIPS D&B)
- Source: https://arxiv.org/abs/2411.11409 ; https://github.com/yunongLiu1/IKEA-Manuals-at-Work ; https://doi.org/10.5281/zenodo.11623997
- What: internet assembly videos aligned in space and time to manual steps and 3D part models ("4D grounding").
- Size: 98 videos, 34,441 annotated frames, 36 furniture models in 6 categories, 137 high-level steps and 1,120 substeps (HTML).
- Views and cameras: third-person internet videos, varied viewpoints; no egocentric.
- Annotations: 6-DoF poses for all furniture parts on annotated frames, part and sub-assembly masks, camera parameters, temporal alignment of manual steps to video segments. Hand pose: not stated. Text: manual steps.
- Licence: "Available for public access under the CC-BY-4.0 license". Access: GitHub and Zenodo.
- Relevance: best existing template for step-aligned assembly video with part 6D pose and a permissive licence; small (98 videos).

#### IKEA-FA (2017) and IKEA Assembly in the Wild (2023, CVPR), brief
- Source: https://arxiv.org/abs/1709.06391 ; https://tengdahan.github.io/ikea ; https://arxiv.org/abs/2303.13800
- IKEA-FA: "101 videos, 1920x1080, 30fps, each 2-4 minutes long"; 14 actors; stationary GoPro third-person; "12 action classes" with frame timestamps; licence not stated.
- IAW: "1007" YouTube videos, "183h:07m:05s", "420" furniture pieces, "8,568" diagrams, "15,683" diagram-to-segment alignments; mostly third-person; licence not stated.
- Relevance: IAW is the closest "in-the-wild internet video aligned to instruction diagrams" precedent for a LEGO version.

#### Assembly101 (2022, CVPR)
- Source: https://arxiv.org/abs/2203.14712 ; https://assembly-101.github.io/ ; https://huggingface.co/datasets/cvml-nus/assembly101
- What: "4321 videos of people assembling and disassembling 101 'take-apart' toy vehicles".
- Size: "513 hours of footage", 4,321 videos, 362 sequences, 53 participants.
- Views and cameras: "simultaneous static (8) and egocentric (4) recordings"; 8 static RGB at 1920x1080, 4 egocentric monochrome at 640x480, synchronised and calibrated.
- Annotations: "more than 100K coarse and 1M fine-grained action segments" (1,380 fine and 202 coarse classes); "18M 3D hand poses" as "21 keypoint locations (21 per hand) in world coordinates"; coarse segments labelled correct/mistake/correction. Object pose: not stated. Text: not stated.
- Licence: "Creative Commons Attribution-NonCommercial 4.0 International License". Access: Hugging Face (migrated May 2026) and download scripts on GitHub.
- Relevance: the largest multi-view hand-centric assembly corpus with 3D hands and fine action steps; the nearest public stand-in for LEGO assembly video (small toy parts, two hands, bench-top). NC licence.

#### MECCANO (2021, WACV; extended 2023, CVIU)
- Source: https://arxiv.org/abs/2010.05654 ; https://arxiv.org/abs/2209.08691 ; https://iplab.dmi.unict.it/MECCANO/
- What: egocentric videos of 20 people building a toy motorbike in an "industrial-like" setting.
- Size: 20 videos from 20 subjects; hours not stated.
- Views and cameras: egocentric headset; "RGB: 1920x1080 12.00 fps, Depth: 640x480 12.00 fps", gaze at 200 Hz.
- Annotations: 8,857 temporal action segments; 64,349 active-object boxes; 89,628 hand boxes; 48,024 next-active-object annotations; 12 verbs, 20 objects, 61 actions. Hand pose: boxes only. Object 6D pose: not stated.
- Licence: not stated. Access: direct links on the project page.
- Relevance: small egocentric small-parts assembly with depth and gaze; no 3D hands or part poses.

#### HA-ViD (2023, NeurIPS D&B)
- Source: https://arxiv.org/abs/2307.05721 ; https://iai-hrc.github.io/ha-vid
- What: human assembly video dataset on a generic assembly box, industrial setting.
- Size: "3222 multi-view, multi-modality videos (each video contains one assembly task), 1.5M frames".
- Annotations: "96K temporal labels and 2M spatial labels"; actions as "subject, action verb, manipulated object, target object, and tool"; benchmarks action recognition, segmentation, object detection, multi-object tracking. Hand pose: not stated.
- Licence: CC BY-NC-SA 4.0 (arXiv badge). Access: project site.
- Relevance: multi-view assembly with tool and part labels; no hand pose.

#### HoloAssist (2023, ICCV)
- Source: https://arxiv.org/abs/2309.17024 ; https://holoassist.github.io/
- What: egocentric instructor-guided manipulation tasks on HoloLens 2.
- Size: "166 hours of data captured by 350 unique instructor-performer pairs" (project page says 169 hours).
- Views and cameras: egocentric HoloLens 2; RGB, depth, head pose, 3D hand pose, eye gaze, audio, IMU.
- Annotations: action labels, mistake and intervention labels, conversational transcripts and summaries; native 3D hand pose. Object pose: not stated.
- Licence: "CDLAv2" (CDLA-Permissive-2.0). Access: project page (video 184 GB, depth 560 GB, hand pose 219 GB).
- Relevance: largest egocentric set with native 3D hand pose and spoken instruction; general manipulation, not LEGO.

#### EPIC-KITCHENS-100 (2021, IJCV), scale reference
- Source: https://arxiv.org/abs/2006.13256 ; https://epic-kitchens.github.io/2025
- Size: "100 hours, 20M frames, 90K actions in 700 variable-length videos"; 37 participants. Egocentric GoPro Hero7 at 50 fps. 89,977 action segments, 97 verbs, 300 nouns; no hand or object pose.
- Licence: "Creative Commons Attribution-NonCommercial 4.0 International". Access: Data.Bris or Academic Torrents (740 GB to 1.1 TB).

#### Ego4D (2022, CVPR), scale reference
- Source: https://arxiv.org/abs/2110.07058 ; https://ego4d-data.org/docs/
- Size: "3,670 hours of daily-life activity video", 931 camera wearers. Egocentric; subsets with audio, 3D meshes, gaze, stereo, multi-camera.
- Licence: custom Ego4D agreement; permits use "for academic research, commercial or noncommercial product development"; no resale/sublicensing. Access: sign agreement, AWS CLI; primary set ~7.1 TB.

### A3. Hand-object pose datasets (human)

#### EgoDex (2025 arXiv; ICLR 2026)
- Source: https://arxiv.org/abs/2505.11709 ; https://github.com/apple/ml-egodex
- What: Apple's egocentric dexterous-manipulation dataset recorded on Apple Vision Pro with on-device 3D hand tracking, plus an imitation-learning benchmark.
- Size: "829 hours of 30 Hz 1080p egocentric video", "194 diverse tasks"; split "about 725 hours, 7 hours, and 97 hours" (train / test / extra) (README). Download: train parts 1-5 at "300 GB each", test "16 GB", extra "200 GB".
- Views and cameras: egocentric; "Apple Vision Pro" with "multiple calibrated cameras and on-device SLAM" (arXiv). Single egocentric RGB stream per episode.
- Annotations: "3D pose annotations for the head, upper body, and hands"; 68 skeletal joints as 4x4 transforms plus confidences and camera intrinsics in HDF5 (ARKit skeleton, not MANO); "natural language annotation" per episode. Object pose: not stated. Temporal segments: not stated.
- Licence: "The dataset is available under CC-by-NC-ND terms." (README)
- Access: public direct download from ml-site.cdn-apple.com, no form.
- Relevance: largest public egocentric hand-pose corpus of tabletop manipulation; NC-ND forbids derivative redistribution, and there is no object pose, so it is a hand/video prior, not assembly supervision. Whether any of the 194 tasks is a brick or toy assembly: not checked.

#### HOI4D (2022, CVPR)
- Source: https://hoi4d.github.io/
- What: category-level egocentric RGB-D human-object interaction dataset with 4D annotations.
- Size: "2.4M RGB-D egocentric video frames", "4000 sequences", "800 different object instances from 16 categories", "9 participants", "610 different indoor rooms". Hours: not stated.
- Views and cameras: egocentric RGB-D; device not stated on the page.
- Annotations: "3D hand pose" (format not stated on page), "category-level object pose", "hand action" and "egocentric action segmentation", "panoptic segmentation, motion segmentation", "reconstructed object meshes and scene point clouds". Language: not stated.
- Licence: "HOI4D is licensed under CC BY-NC 4.0". Access: direct OneDrive / Baidu links, no form.
- Relevance: egocentric hand plus object pose with action segments on articulated household objects; no assembly.

#### ARCTIC (2023, CVPR)
- Source: https://arctic.is.tue.mpg.de/ ; https://arxiv.org/abs/2204.13662 ; https://github.com/zc-alexfan/arctic
- What: bimanual dexterous manipulation of articulated objects with dense hand-object contact annotation.
- Size: "2.1M video frames paired with accurate 3D hand and object meshes and detailed, dynamic contact information" (arXiv). Subject/object counts: not stated on fetched pages (unverified: 10 subjects, 11 objects).
- Views and cameras: "Images are from 8x 3rd-person views and 1x egocentric view (for mixed-reality setting)"; "captured in a MoCap setup using 54 high-end Vicon cameras" (GitHub).
- Annotations: "3D groundtruth for SMPL-X, MANO, articulated objects" (GitHub); "dynamic contact information" (arXiv). Language, action segments: not stated.
- Licence: GitHub: "ARCTIC data and software are not available for commercial use." Access: "Register ARCTIC Account" at arctic.is.tue.mpg.de/register.php; login required.
- Relevance: closest public analogue to two-handed part manipulation (MANO both hands, articulated object pose, contact); ideal for validating a bimanual hand-object pose pipeline; small object set, no assembly.

#### DexYCB (2021, CVPR)
- Source: https://dex-ycb.github.io/ ; https://arxiv.org/abs/2104.04631 ; https://github.com/NVlabs/dex-ycb-toolkit
- What: multi-view RGB-D captures of humans grasping YCB objects; benchmarks 6D object pose, 3D hand pose, and handover grasps.
- Size: 10 subjects, "1,000 in total" sequences, "21" YCB objects (toolkit); s0_train split "465504" samples. Total frames: not stated on the fetched page (unverified: 582K).
- Views and cameras: "8 RealSense" RGB-D cameras, third-person studio setup; no egocentric view.
- Annotations: "MANO shape parameter" and "MANO pose coefficients in PCA representation"; "6D object pose estimation" ground truth; "2D object and keypoint detection". No language, no action segments.
- Licence: "DexYCB is licensed under CC BY-NC 4.0."; toolkit GPLv3. Access: Google Drive, "single 119GB compressed file or 13 separate files", no form.
- Relevance: clean 6D object plus MANO ground truth for single-object grasps; a pose-estimator validation set, not policy data.

#### HOT3D (2024 arXiv; CVPR 2025)
- Source: https://facebookresearch.github.io/hot3d/
- What: egocentric multi-view hand-object tracking dataset from Aria and Quest 3.
- Size: "833 minutes of multi-view image streams", "1.5M multi-view frames (3.7M+ images)", "3832 clips", "19 subjects interacting with 33 diverse rigid objects".
- Views and cameras: Aria "one RGB 1408x1408 and two monochrome 640x480"; Quest 3 "two monochrome 1280x1024"; egocentric.
- Annotations: "Hand annotations are provided in the UmeTrack and MANO formats"; "ground-truth 3D poses" for hands and objects (6DoF rigid). Language, action segments: not stated.
- Licence: "HOT3D license agreement" at projectaria.com/datasets/hot3d/license/ (terms not fetched). Access: projectaria.com/datasets/hot3D/ and HOT3D-Clips on Hugging Face.
- Relevance: egocentric MANO plus 6DoF object pose on rigid objects is exactly the annotation shape a LEGO assembly video model would need; objects are household, not bricks.

#### Ego-Exo4D (2023 arXiv; CVPR 2024)
- Source: https://ego-exo4d-data.org/ ; https://arxiv.org/abs/2311.18259
- What: paired egocentric plus exocentric video of skilled activities with language and 3D pose.
- Size: "1,286 hours of video", "740 participants from 13 cities", "123 different natural scene contexts"; "5035" takes (site).
- Views and cameras: Aria glasses ("8 MP RGB camera and two SLAM cameras") plus "4-5 (stationary) GoPros".
- Annotations: "3D hand/body pose" benchmark, keystep recognition, "expert commentary", "narrate-and-act descriptions", "atomic action descriptions". Object pose: not stated.
- Licence: "Ego-Exo4D Licenses" signed via form at ego4d.dev; CLI download after approval.
- Relevance: ego plus exo pairing and keystep segments on procedural tasks (e.g. bike repair) is the nearest large public analogue to step-labelled assembly video.

#### Smaller 2024-2026 human-to-robot video sets (one line each)
- DexWild (2025, https://arxiv.org/abs/2505.07813 ; https://dexwild.github.io/): "9,290 demos across 93 environments" from "palm mounted cameras"; hand pose as "a 17 dimensional joint angle space"; dataset licence not stated; no assembly tasks.
- EgoZero (2025, https://arxiv.org/abs/2505.20290): Aria glasses, "7 manipulation tasks", "20 minutes of data collection per task"; dataset release unverified.
- EgoMimic (2024, https://arxiv.org/abs/2410.24221 ; https://huggingface.co/datasets/gatech/EgoMimic): Aria egocentric human video with "3D hand tracking" for bimanual co-training; HF repo "243 GB"; licence "No dataset card yet".
- UMI / UMI-Data (2024, https://arxiv.org/abs/2402.10329 ; https://umi-data.github.io/): hand-held gripper on "GoPro (or iPhone)"; the site "only provide[s] a list of references to existing data"; gripper, not hand.
- EgoVerse (2026, https://arxiv.org/abs/2604.07607): "1,362 hours (80k episodes)", "1,965 tasks" over "240 scenes", "manipulation-relevant annotations"; hand-pose format, licence, and any assembly task: not stated in the abstract; access via egoverse.ai.

### A4. Robot manipulation datasets

#### RH20T (2023 arXiv; ICRA 2024)
- Source: https://rh20t.github.io/ ; https://arxiv.org/abs/2307.00595
- What: multi-robot contact-rich manipulation dataset with paired human demonstration videos.
- Size: "over 110,000 contact-rich robot manipulation sequences"; 147 tasks ("48 tasks from RLBench, 29 tasks from MetaWorld, and introduce 70 self-proposed tasks"); 7 robot configurations. Hours: not stated.
- Views and cameras: "8-10 global RGBD cameras" per configuration, "RGB image 1280 x 720 x 3", depth at 10 Hz; third-person; human demo video per sequence.
- Annotations: "Robot joint angle", "Robot joint torque", "Gripper Cartesian pose", "Gripper width", "6-DoF Force/Torque"; "a corresponding human demonstration video and a language description for each robot sequence". Hand pose, object pose: not stated.
- Licence: "RH20T-C ... CC BY-SA 4.0"; "RH20T-NC ... CC BY-NC 4.0". Access: Google Drive and Baidu links.
- Relevance: paired human-robot videos plus force on contact-rich tasks; whether the RLBench-derived subset includes peg/insertion tasks is unverified.

#### DROID (2024)
- Source: https://droid-dataset.github.io/ ; https://arxiv.org/abs/2403.12945
- Size: "76k demonstration trajectories or 350 hours of interaction data", "564 scenes and 84 tasks by 50 data collectors" (arXiv; project page says 86 tasks).
- Views and cameras: "two adjustable Zed 2 stereo cameras, a wrist-mounted Zed Mini stereo camera"; "Franka Panda 7DoF robot arm"; teleop via "Oculus Quest 2".
- Annotations: robot actions and proprioception; "3 natural language annotations for 95% of all successful DROID episodes". No hand or object pose.
- Licence: CC BY 4.0 (arXiv listing; dataset-page licence statement unverified). Access: "gs://gresearch/robotics" via TFDS (RLDS).
- Relevance: broad-scene robot video prior; no assembly subset stated.

#### BridgeData V2 (2023, CoRL)
- Source: https://rail-berkeley.github.io/bridgedata/
- Size: "60,096 trajectories" = "50,365 teleoperated demonstrations" + "9,731 rollouts from a scripted pick-and-place policy"; "24 environments", "13 skills"; "average trajectory length is 38 timesteps" at "5 Hz".
- Views and cameras: "RGBD camera ... over-the-shoulder view, two RGB cameras with poses that are randomized ... and RGB camera attached to the robot's wrist", "640x480"; WidowX 250.
- Annotations: "Each trajectory is labeled with a natural langauge instruction"; robot actions.
- Licence: "Creative Commons Attribution 4.0 International License". Access: rail.eecs.berkeley.edu/datasets/bridge_release/data/.
- Relevance: pick-and-place dominant, no assembly.

#### Open X-Embodiment (2023 arXiv; ICRA 2024)
- Source: https://robotics-transformer-x.github.io/ ; https://arxiv.org/html/2310.08864v9 ; https://github.com/google-deepmind/open_x_embodiment
- Size: "1M+ real robot trajectories", "22 robot embodiments", "60 existing robot datasets from 34 robotic research labs", "527 skills (160266 tasks)".
- Annotations: robot actions in RLDS with varied action spaces; language in many subsets. No hand or object pose stated.
- Licence: "All software is licensed under the Apache License, Version 2.0"; "All other materials are licensed under ... CC-BY" (GitHub); component datasets carry their own licences.
- Access: "gs://gdm-robotics-open-x-embodiment/" via gsutil or tfds.
- Assembly subsets: the component list includes "Furniture Bench Dataset" and "CMU Franka Pick-Insert Data"; no LEGO or NIST-board subset located on the fetched pages (unverified absence).
- Relevance: FurnitureBench and Pick-Insert are the only assembly-flavoured subsets; no dexterous-hand or LEGO data.

### A5. Simulated and real assembly benchmarks (downstream policy targets)

#### FurnitureBench (2023, RSS)
- Source: https://arxiv.org/abs/2305.12821 ; https://clvrai.github.io/furniture-bench/ ; https://clvrai.github.io/furniture-bench/docs/tutorials/dataset.html ; https://github.com/clvrai/furniture-bench
- What: "Reproducible Real-World Benchmark for Long-Horizon Complex Manipulation" for furniture assembly with a Franka Panda, plus "FurnitureSim".
- Tasks: "8 tasks" (lamp, square table, desk, drawer, cabinet, round table, stool, chair) plus "one-leg assembly"; furniture "designed inspired by IKEA furniture and modified to enable a single robotic arm to carry out the assembly".
- Size: "5000+ demonstrations", "200+ hours" (abstract); docs "219.6 hours", "5,100 successful demonstrations", raw "1,179 GB" across low/medium/high randomness.
- Observations: "Wrist camera image (224, 224, 3)" and "Front camera image (224, 224, 3)"; EE pose/velocities, "Joint positions (7,), joint velocities (7,), joint torques (7,)", "Gripper width (1,)"; 10 Hz delta-EE actions. Success: "rewards (1 if a furniture part is assembled; otherwise, 0)" and skill completion flags. Part poses in the released data: not stated.
- Licence: code "MIT License"; dataset licence not stated. Access: Google Drive via gdown/rclone.
- Relevance: closest public real-robot "insert part into part" benchmark with wrist plus front camera and part-level success reward; a non-LEGO transfer target.

#### BEHAVIOR-1K (2024) and BEHAVIOR Challenge 2025/2026
- Source: https://arxiv.org/abs/2403.09227 ; https://behavior.stanford.edu/knowledgebase/tasks/index.html ; https://behavior.stanford.edu/challenge/
- Assembly-like activities among 1,016 tasks: "assembling_furniture-0" (goal "attached ?desk_bracket.n.01_1 ?desk_top.n.01_1" and legs attached to the bracket), "attach_a_camera_to_a_tripod-0", "assembling_gift_baskets-0" (packing). Assembly is the symbolic "attached" predicate, not contact physics. No LEGO or brick task.
- Challenge 2026: "100 full-length household tasks", "20,000 human teleoperation demos, 1,950 hours in total", RGB-D, proprioception, skill annotations; success "Average task success score with BDDL partial credit". Whether the assembly-named tasks are in the challenge set: unverified.
- Licence: paper CC BY 4.0; dataset/simulator licence not stated on fetched pages.
- Relevance: symbolic attach only; not a contact-rich insertion target.

#### RLBench (2020, RA-L)
- Source: https://arxiv.org/abs/1909.12271 ; https://github.com/stepjam/RLBench/tree/master/rlbench/tasks
- Insertion-type tasks: "insert_onto_square_peg", "insert_usb_in_computer", "plug_charger_in_power_supply", "screw_nail"; about ten more stacking/placing tasks (stack_blocks, block_pyramid, put_shoes_in_box, ...). Demos generated on demand by motion planner.
- Observations: "rgb, depth, and segmentation masks from an over-the-shoulder stereo camera and an eye-in-hand monocular camera" plus proprioception.
- Licence: custom Imperial College licence, "non-exclusive, temporary, fully paid-up, royalty free, non-transferable, non-sub-licensable"; academic non-commercial.
- Relevance: cheap peg-in-hole sim proxy; no snap-fit; restrictive licence.

#### ManiSkill3 (2024 arXiv; RSS 2025)
- Source: https://arxiv.org/abs/2410.00425 ; https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html ; https://github.com/haosulab/ManiSkill
- Assembly-like tasks: "PegInsertionSide-v1" (success "The white end of the peg is within 0.015m of the center of the box (inserted mid way)", demos yes); "PlugCharger-v1" (success "The charger is inserted into the receptacle", demos yes); "AssemblingKits-v1" ("insert it into the correct empty slot", demos no).
- Observations: state, "RGBD + Segmentation data", point clouds; GPU-parallel.
- Licence: code "Apache License, Version 2.0"; assets "CC BY-NC 4.0".
- Relevance: fastest sim for peg/charger insertion with demos; no brick physics.

#### Factory / IndustReal / AutoMate / Isaac Lab (NVIDIA, 2022-2025)
- Source: https://arxiv.org/abs/2205.03532 ; https://arxiv.org/abs/2305.17110 ; https://arxiv.org/abs/2407.08028 ; https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html
- Envs: "Isaac-Factory-PegInsert-Direct-v0", "Isaac-Factory-GearMesh-Direct-v0", "Isaac-Factory-NutThread-Direct-v0", "Isaac-AutoMate-Assembly-Direct-v0" ("Insert a plug into its corresponding socket"). AutoMate: "a dataset of 100 assemblies compatible with simulation and the real world"; specialists "individually solve 80 assemblies", generalist "jointly solves 20 assemblies with an 80%+ success rate".
- Real-world results: IndustReal "600 trials": peg "76.7%", gear "92.5%", connector "85%", pick-place-insert "89.16%". AutoMate "500 real-world trials": specialists "86.50 +/- 16.52%", generalist "84.50 +/- 21.79%".
- Observations: Franka with wrist "Intel RealSense D435"; policy inputs "joint angles, gripper/object poses, and/or target poses".
- Licence: Isaac Lab BSD-3 (unverified); AutoMate assets not stated.
- Relevance: strongest public sim-to-real insertion envs; 100 plug/socket geometries are the nearest analogue to "many brick geometries"; none snap-fit.

### A6. Robot LEGO and brick works, 2023-2026 (mostly the lab's own)

#### BrickSim (2026, arXiv 2603.16853; Wen, Liu, Piao, Li, C. Liu)
- Source: https://arxiv.org/abs/2603.16853 ; https://github.com/intelligent-control-lab/BrickSim
- What: Isaac Sim based simulator with "a compact force-based mechanics model for snap-fit connections" solved by "a structured convex quadratic program"; handles "assembly, disassembly, and structural collapse".
- Results: "100% accuracy in static stability prediction with an average solve time of 5 ms" on "150 real-world assemblies"; drop tests reproduce collapse; Franka single- and two-arm demos. No policy learning or sim-to-real shown.
- Code: "open-source"; licence not stated.
- Relevance: the only simulator with snap-fit brick physics; the place to (a) replay pseudo-labelled actions from generated videos and (b) check the stability of lifted end states, before real hardware.

#### BrickCraft (2026, arXiv 2605.07605; Yu, Li, Tang, Lu, Hu, Liu, C. Liu)
- Source: https://arxiv.org/abs/2605.07605 ; https://intelligent-control-lab.github.io/BrickCraft
- What: "Visuomotor Skill Composition with Situated Manual Guidance for Long-Horizon Interlocking Brick Assembly"; three primitive skills learned "from a limited set of demonstrations", composed from a manual; tasks "Pyramid," "Stairs," "House".
- Success rates and demo count: not stated on fetched pages. Code: "Coming Soon". Licence CC BY 4.0 (paper).
- Relevance: the group's first learned visuomotor LEGO policy; the natural baseline and the natural consumer of generated robot videos.

#### Lightweight and Transferable EOAT for LEGO (2023, arXiv 2309.02354), Eye-in-Finger (2025, arXiv 2503.06848), APEX-MR (2025, RSS), Prompt-to-Product (2025)
- EOAT: insert-then-twist tool; "FANUC LR-mate 200id/7L" and "Yaskawa GP4"; bricks "1x2, 1x4, 2x2, 2x4"; Table 1 e.g. 1x2 on solid support "100% / 96%" assemble/disassemble, "[100%/100%]" with safe learning.
- Eye-in-Finger: endoscope in the fingertip; tolerance to calibration error "from 0.4mm to up to 2.0mm"; "success remained at 100% for errors up to 2.0 mm" (12 trials per condition). This is the alignment tolerance any learned policy must meet.
- APEX-MR (https://arxiv.org/abs/2503.15836): asynchronous dual-arm planning with "commercial LEGO bricks"; speed-up "48% compared to sequential planning".
- Prompt-to-Product: see A1.
- Code/data: not stated for any of these.

#### WorkBenchMark (2026, arXiv 2606.19358; RoboCup 2026 Symposium)
- Source: https://arxiv.org/abs/2606.19358 ; https://workbenchmark.github.io
- What: "LEGO Duplo-based robotic assembly benchmark" with "400 tasks across four complexity tiers" (two-brick stacking; "3-5 bricks"; 3D shapes "3-12 bricks"; overhangs); sim Franka in MuJoCo/LIBERO, real "Universal Robots UR5 cobot" with wrist RGB-D; perception GroundingDINO + SAM + FoundationPose.
- Success: "If every required brick v in V is placed at its target pose in the assembly area, attached to the structure." Results: planning pipeline 94/87/67/62% by tier; fine-tuned VLA 82/63/23/2%.
- Code/data: "will be released openly"; paper CC BY-SA 4.0.
- Relevance: a ready-made tiered brick benchmark with a VLA baseline; Duplo tolerances are looser than LEGO System.

#### Others, one line each
- Brick-Composer / BC-Bench (2026, https://arxiv.org/abs/2606.05445): MLLM brick-selection and step composition; "strict step-level assembly success from less than 1% to around 15%"; no robot.
- Learning to Build (2026, https://arxiv.org/abs/2602.23934): ABB CRB 15000 with suction, non-interlocking blocks, sim "93.3%", real "80%"; stacking not insertion.
- RoCo Challenge at AAAI 2026 (https://arxiv.org/abs/2603.15469): dual-arm planetary gearbox assembly, Isaac Sim plus real, step-wise scoring.
- BiAssemble (ICML 2025, https://arxiv.org/abs/2506.06221): bimanual fracture reassembly; not fastened joints.
- BricksRL (2024, https://arxiv.org/abs/2406.17490): robots built from LEGO, not assembling LEGO; listed to avoid confusion.
- "ASAP", "AssemblyBench", "IKEA-Bench", "RoboCasa assembly": no primary page found; unverified.

## B. How the field evaluates generated video as training data

### B1. DreamGen (2025, arXiv 2505.12705, NVIDIA GEAR)
- Source: https://arxiv.org/html/2505.12705 (v2) ; https://research.nvidia.com/labs/gear/dreamgen
- What it evaluates: whether robot videos generated by fine-tuned video world models ("neural trajectories", pseudo-labelled with actions) are usable as policy training data, via (a) VLM instruction-following and physics scores and (b) downstream policy success.
- Metrics:
  - Instruction Following (IF): "The generated videos are fed into Qwen-VL-2.5 with specific prompts to give a binary score (0 or 1) for quantifying the consistency between the video content and the task instructions" (Sec. 4); also scored by GPT-4o and humans (Table 2). Prompt (App. H.1): "The video shows a robot arm completing a specific task. Please evaluate: if the video follows the instruction to finish the task '{prompt}', give a positive score. Reply only '0' for No or '1' for Yes." A stricter variant adds "If you see HUMAN HANDS instead of robot arms, IMMEDIATELY ANSWER 0".
  - Physics Alignment (PA): "we first employ the VideoCon-Physics [26], a VLM specifically trained to give scores for physics adherence of generated videos ... we use a general VLM: Qwen-VL-2.5 to also score each video ... and then calculate the average score of these two scores". Qwen prompt (App. H.2): "Does the video show good physics dynamics that is aligned with the physical world? Answer 0 for No or 1 for Yes."
  - Human: IF = did "the video complete the task specified by the language"; PA = "humans rank the model's output, given the same initial frame" (App. H.3).
- Filtering: the pipeline text (Sec. 2.2, 2.3) describes no per-video VLM rejection gate; the word "filter" does not occur in the HTML. IF/PA rank world models; they are not a stated data filter. Cheap sanity check (App. H.4): "we replay the IDM actions in simulation, where we have access to the digital twin of the Fourier GR1."
- Action labelling: IDM = "diffusion transformers with SigLIP-2 vision encoder ... flow matching objective ... conditioned on two image frames and is trained to predict action chunks between the image frames"; LAPA latent actions as the alternative.
- Protocol: DreamGen Bench, 4 world models (Hunyuan, CogVideoX, WAN 2.1, Cosmos), zero-shot and fine-tuned, RoboCasa (Franka, sim; 1200 train trajs, 48 eval frames) and Fourier GR1 (real; 100 train trajs, 50/47/30 eval frames). Binary 0/1 per video, reported as %.
- Headline numbers (Table 2, RoboCasa): WAN2.1-sft IF(GPT) 77.1, IF(Qwen) 18.8, IF(human) 91.7, PA 55.3; Cosmos-sft 79.2 / 29.2 / 93.8 / 61.5; zero-shot models score about 0. Human alignment: "average Pearson correlation of > 90%" between IF(GPT-4o) and human; Table 6 r = 0.94 (RoboCasa), 0.93, 0.96, 1.00 (GR1 splits). Qwen-7B IF is far below GPT-4o and humans.
- Downstream: "only 10 real-world trajectories per task"; "We generate 300 neural trajectories for each GR1 task, 100 neural trajectories for each Franka task ... co-train with real-world trajectories with a 1:1 sampling ratio." Generalization (Table 1): "GR00T N1 trained on pick-and-place alone achieves 0% success rates on most novel behavior and environment experiments, while DreamGen enables 43.2% success rates on new behaviors in seen environments and 28.5% in completely unseen environments" (14 novel-behaviour tasks 11.2% to 43.2%; 13 novel-environment tasks 0.0% to 28.5%; 50 neural trajectories per novel task; teleop base set 2,885 pick-and-place trajectories). Sim (RoboCasa, Fig. 4): real regimes 720 / 2.4k / 7.2k demos, "co-training with neural trajectories yields a performance boost ... across all data regime scenarios" (curve values are figure-only).
- Metric-to-policy link: yes. "we measure the performance of the RoboCasa benchmark by only training on neural trajectories generated from the different video world models ... 7K neural trajectories per model. For DreamGen Bench score, we use the average of IF (GPT) and PA from Table 2 ... the correlation between DreamGen Bench and RoboCasa shows a positive correlation" (Fig. 6; coefficient not stated in text).
- Reuse for LEGO: copy the two-axis binary VLM scoring (task completion, physics) averaged with VideoCon-Physics, validate on a small human-labelled set with Pearson r, and replay IDM actions in a digital twin as the cheap proxy before policy training.

### B2. Cosmos World Foundation Model Platform (2025, arXiv 2501.03575, NVIDIA)
- Source: https://arxiv.org/html/2501.03575 (Sec. 5.3.1 "3D Consistency", 5.3.2 "Physics Alignment", 5.2.7 "Limitations")
- What it evaluates: whether WFM rollouts are geometrically consistent 3D worlds and whether predicted dynamics match a physics simulator.
- Metrics:
  - 3D consistency: "Sampson error ... the first-order approximation of the distance from one interest point to its corresponding epipolar line in another view", keypoints from "SuperPoint and LightGlue", F via "OpenCV's 8-point RANSAC"; "success rate of camera pose estimation algorithms" (COLMAP); view synthesis by holding out "every 8 frames" and fitting 3D Gaussian splatting (Nerfstudio), PSNR/SSIM/LPIPS. Classical geometry, no VLM.
  - Physics alignment: "Pixel-level metrics ... PSNR and SSIM"; "Feature-level metrics ... DreamSim"; "Object-level metrics ... Using SAMURAI, we propagate the ground-truth instance masks in the first frame through the rest of the predicted video frames ... compute the intersection-over-union (IoU) between ground truth and predicted object masks for each frame and object of interest." Averaged over frames, videos, and "four random seeds".
  - Failure audit: "evaluation set of 100 Physical AI inputs ... we manually inspect the failure cases"; 9-frame conditioning "failure rate lower than 2%".
- Protocol: 3D on "500 videos randomly chosen from the test set of the RealEstate10K dataset". Physics: "Using PhysX and Isaac Sim, we design eight 3D scenarios" (free fall, tilted slope, U-slope, stable stack, unstable stack, dominoes, seesaw, gyroscope), "4 different static camera views", "800 1080p videos of 100 frames", conditioning "either 1 or 9 frames" plus a kinematic caption, "Metrics are calculated over 33 frames".
- Headline numbers: Table 19: VideoLDM Sampson 0.841, pose success 4.4%; Cosmos-Predict1-7B-Text2World 0.355 / 62.6%; 7B-Video2World 0.473 / 68.4%; real videos 0.431 / 56.4%. Table 20: 7B-Video2World prompt+9 frames PSNR 21.06, SSIM 0.691, DreamSim 0.859, Avg IoU 0.592; prompt+1 frame IoU 0.332; 14B 9 frames IoU 0.598. Authors: "our results do not suggest that the larger model performs better on our physics alignment ... all the WFMs equally struggle with physics adherence".
- Metric-to-policy link: not stated ("We leave a more comprehensive evaluation as future work").
- Reuse for LEGO: the object-level IoU against a simulated ground truth transfers directly: render LEGO steps in sim or capture real multi-view, condition on the first frames, track brick masks, and score IoU instead of pixel PSNR.

### B3. VBench (CVPR 2024, arXiv 2311.17982) and VBench-2.0 (2025, arXiv 2503.21755)
- Source: https://arxiv.org/html/2311.17982 ; https://github.com/Vchitect/VBench ; https://arxiv.org/html/2503.21755
- What: VBench measures "superficial faithfulness" (per-frame quality, temporal consistency, prompt adherence); VBench-2.0 measures "intrinsic faithfulness ... physical laws, commonsense reasoning, anatomical correctness, and compositional integrity".
- VBench (16 dims): Subject Consistency (DINO), Background Consistency (CLIP), Temporal Flickering, Motion Smoothness (AMT), Dynamic Degree (RAFT), Aesthetic Quality (LAION), Imaging Quality (MUSIQ); Object Class, Multiple Objects (GRiT), Human Action (UMT), Color, Spatial Relationship, Scene (Tag2Text), Appearance Style (CLIP), Temporal Style, Overall Consistency (ViCLIP). Nothing on physics.
- VBench-2.0 (18 dims): Human Fidelity (Human Anatomy, Temporal Consistency-Clothes/-Identity); Creativity; Controllability (Dynamic Attribute, Dynamic Spatial Relationship, Motion Order Understanding, Human Interaction, Complex Plot, Complex Landscape, Camera Motion); Physics (State Change-Mechanics/-Thermotics/-Material; Geometry-Multi-View Consistency); Commonsense (Motion Rationality, Instance Preservation). "Unless otherwise specified, we adopt LLaVA-Video-7B as our VLM model." Mechanics: "video-based multi-question answering ... we prompt GPT-4o to generate explicit visual descriptions of expected physical behavior ... based solely on the text prompt"; with "pre-filtering step to exclude cases where the initial state of the generated video does not align with the prompt". Multi-View Consistency: SIFT + FLANN + RANSAC matching stability. Motion Rationality: "whether the generated motion leads to the correct real-world consequences", VQA with "Redundant questioning ... helps filter out false positives". Instance Preservation: "We tune a clip-level entity abnormal detector based on Qwen2.5-VL-3B-Instruct ... Only [if] all of the clips in a video [are] normal will [it] be considered as normal (score 1), otherwise 0". Human Anatomy: ViT-base anomaly detectors for body, hands, faces.
- Protocol: pairwise human preference, 4 models, 5 groups per prompt, 6 pairs per group, win ratio with ties 0.5, Spearman between model-level human and metric win ratios. VBench-2.0: "284 hours across 18 annotators"; "We randomly sample 20% of the annotated pairs for verification, with a required success rate of 95%".
- Headline numbers (VBench-2.0 Table II): Mechanics HunyuanVideo 76.09%, CogVideoX-1.5 80.80%, Sora 62.22%, Kling 1.6 65.55%; Multi-view 43.80 / 21.79 / 58.22 / 64.38. Human alignment per dimension 81.70% to 99.46% (Table IV).
- Metric-to-policy link: not stated (human preference only).
- Reuse for LEGO: Motion Rationality's "consequence check" pattern (ask a VLM "after the press, is the brick attached?" with redundant questions) and the Instance Preservation detector (bricks must not merge, split, appear, or vanish) are the two transferable pieces.

### B4. PhyGenBench / PhyGenEval (2024, arXiv 2410.05363)
- Source: https://arxiv.org/html/2410.05363
- What: whether T2V outputs obey a named physical law ("physical commonsense alignment", PCA) separately from semantic alignment (SA).
- Metrics: three stages. "Key Physical Phenomena Detection": keyframe VQA with "VQAScore", "adjacent 5 frames near the keyframe", questions generated by GPT-4o. "Physics Order Verification": "verify whether key physical phenomena occur in the correct order" (multi-image VLM). "Overall Naturalness": video VLM. PCA on a 0-3 scale.
- Protocol (Table 3): 27 physical laws, 4 domains (Optics 50, Mechanics 40, Thermal 30, Material 40), 160 captions, 8 T2V models, 1280 videos. Human study: "64 prompts ... 512 videos ... three annotators ... integer score of 0-3".
- Headline numbers: "even the best-performing model, Gen-3, only attains a PCA score of 0.51". Human alignment: PCA Kendall tau 0.78, Spearman rho 0.81 vs VideoPhy 0.03/0.04, VideoScore 0.17/0.19, DEVIL 0.17/0.18.
- Metric-to-policy link: not stated.
- Reuse for LEGO: the "expected event order" verifier maps onto assembly: encode the step sequence (grasp, align, press, release) as ordered keyframe questions and score order violations.

### B5. Physics-IQ (2025, Google DeepMind, arXiv 2501.09038)
- Source: https://arxiv.org/html/2501.09038 ; https://github.com/google-deepmind/physics-IQ-benchmark
- What: whether a video model predicts the real physical continuation of a real video, and whether visual realism tracks that ability.
- Metrics: "Spatial IoU; ... Spatiotemporal IoU; ... Weighted spatial IoU; ... MSE", all against the real continuation using motion masks; "combined into a single score, the Physics-IQ score ... normalized such that physical variance -- the upper limit of what we can reasonably expect a model to capture -- is at 100%." Gemini 1.5 Pro only for the realism side-study.
- Protocol: "396 videos (66 scenarios x 3 perspectives x 2 takes)", static camera, 3 s conditioning then predict 5 s; physical variance = difference between the two takes.
- Headline numbers (Table 1): VideoPoet (multiframe) 29.5, Lumiere (multiframe) 23.0, Runway Gen 3 22.8, Stable Video Diffusion 14.8, Sora 10.0 (physical variance = 100). Realism 2AFC: "Sora achieved the best MLLM score of 55.6%" (closest to chance). "no significant correlation" between realism and Physics-IQ (r = -0.46, p = .249, unverified against raw text). Quoted conclusion: "visual realism does not imply physical understanding."
- Metric-to-policy link: not stated.
- Reuse for LEGO: the two-take physical-variance ceiling is the cleanest normalisation to copy: record each assembly step twice and report generated-vs-real motion-mask IoU relative to real-vs-real.

### B6. VideoPhy (2024, arXiv 2406.03520) and VideoPhy-2 (2025, arXiv 2503.06800)
- Source: https://arxiv.org/html/2406.03520 ; https://arxiv.org/html/2503.06800 ; https://github.com/Hritikbansal/videophy
- What: human-rated semantic adherence (SA) and physical commonsense (PC) of generated videos, plus a trained 7B auto-rater.
- Metrics: VideoPhy: SA = "the text caption is semantically grounded in the frames", PC = "depicted actions and object states follow physics laws in the real-world", binary. VideoPhy-2: 5-point Likert for SA and PC, per-rule grounding "violated (0), followed (1), or cannot be determined (2)"; joint success = "SA>=4 and PC>=4". Auto-rater: VideoCon-Physics (VideoCon 7B finetuned "to maximize the log likelihood of Yes/No"); VideoPhy-2-Autoeval trained on "~50K human annotations".
- Protocol: VideoPhy 688 captions, 12 models, 11,330 videos, 36,500 human annotations, "14 workers who have studied high-school physics". VideoPhy-2: 197 actions, 3,940 prompts, 7 models, "12 human annotators ... after passing a qualification test", three annotators per video.
- Headline numbers: VideoPhy: "the best performing model, CogVideoX-5B, generates videos that adhere to the caption and physical laws for 39.6% of the instances"; auto-rater ROC-AUC SA/PC: GPT-4-Vision 53/53, Gemini-1.5-Pro 73/58, VideoCon-Physics 82/73. VideoPhy-2 joint % (All / Hard): Wan2.1-14B 32.6 / 21.9, CogVideoX-5B 25.0 / 0.0, Cosmos-Diff-7B 24.1 / 10.9; "models struggle the most with conservation [laws]". Auto-rater Pearson x100 (unseen prompts, Avg/SA/PC): VideoPhy-2-Autoeval 42.0 / 47.0 / 37.0 vs VideoCon-Physics 28.5 / 32.0 / 25.0 vs Gemini-2.0-Flash-Exp 18.5 / 26.0 / 11.0.
- Metric-to-policy link: not stated in either paper; the only link is external (DreamGen uses VideoCon-Physics as one PA judge, B1).
- Reuse for LEGO: the SA>=4 and PC>=4 joint criterion with three annotators per clip is a cheap, defensible human protocol; VideoCon-Physics is an open 7B rater we can run, but its PC Pearson is 0.25-0.37 even in-domain, so use it as a filter, not a headline metric.

### B7. Metric pitfalls the field has documented (blur, static content, per-frame bias)
- FVD content bias (Ge et al., CVPR 2024, https://arxiv.org/abs/2404.12391): FVD "increases only slightly with large temporal corruption" and can be drastically reduced by sampling videos without motion; cause "attributed to the features extracted from a supervised video classifier trained on the content-biased dataset"; fix: "FVD with features extracted from the recent large-scale self-supervised video models is less biased toward image quality."
- FVD and blur (Beyond FVD / JEDi, https://arxiv.org/html/2410.05203v2): "FVD fails to detect low blur noise and incorrectly suggests an improvement in video quality"; "I3D and VideoMAE are not ideal feature spaces ... as they do not capture blur distortion well". JEDi = V-JEPA features (fine-tuned on SSv2) with polynomial-kernel MMD; "Requires only 16% of samples needed by FVD to reach convergence"; "Increases alignment with human evaluation by 34% on average".
- Motion-based assessment (Direct Motion Models / TRAJAN, https://arxiv.org/abs/2505.00209): auto-encodes point tracks; "markedly more sensitive to temporal distortions in synthetic data" than FVD; generators "generate plausible looking frames, but poor motion"; can "spatiotemporally localize generative video inconsistencies".
- Physics-IQ (B5): "visual realism does not imply physical understanding"; realism 2AFC and physics score are uncorrelated across models.
- Grounding generated plans with a world model (GVP-WM, https://arxiv.org/abs/2602.01960): "video-generated plans often violate temporal consistency and physical constraints, leading to failures when mapped to executable actions"; fix is "a goal-conditioned latent-space trajectory optimization problem that jointly optimizes latent states and actions under world-model dynamics"; demonstrated on "motion-blurred videos that violate physical constraints" in sim navigation and manipulation; numbers not in the abstract.

### B8. Synthetic-vs-real mixing results outside DreamGen (my own fetches; more in B9-B12 from the second pass)
- AnchorDream (2025, https://arxiv.org/html/2512.11797): repurposes "Cosmos-Predict2 2B" with LoRA; conditions on a robot-only render so the arm is embodiment-consistent. Real robot, 50 human demos vs +500 generated: SweepCoffeeBeans 35% to 95%, PourToBowl 0% to 35%, OpenDrawer 0% to 25%, CloseDrawer 30% to 75%, ToyToPlate 85% to 100%, BookToShelf 20% to 45%, average 28% to 63%. Sim (24 RoboCasa tasks): 22.5% (Human50) to 30.7% (+300 generated). Filtering: none described; "Generated demonstrations are used directly after synthesis."
- Efficient Sim-to-Real Transfer of World-Action Models from Synthetic Priors (2026, https://arxiv.org/html/2606.31101): Cosmos Policy trained on "~800 demonstrations per task (~3,200 total)" from GPU simulation with domain randomisation, zero real demos. Real robot (10 trials each): Ours (800 sim) Banana 5/10, Brick 5/10, Drawer 2/10, Strawberry 2/10, avg 35%; Diffusion Policy with 10 real demos avg 5%; with 50 real demos avg 25%. A "Brick" task exists (details not quoted). Video quality: only qualitative ("visually consistent with the synchronized real camera observations").
- Video Generators are Robot Policies (Columbia, 2025, https://arxiv.org/html/2508.00795): RoboCasa 50 demos, 0.63 average success over 24 tasks; action decoder trained on 12 of 24 tasks with action-free video for all 24 generalises to the unseen 12 while "DP-ResNet ... only exhibits a minimal degree of generalization". Ablation: freezing the pretrained SVD without task fine-tuning drops success to 0.09. Failures attributed to "unrealistic video predictions -- e.g., failing to generate upright placements or gripper-induced toppling -- likely due to limited real-world physics priors in the pretrained SVD model"; no filtering mechanism.
- DemoGen (2025, https://demo-generation.github.io/): 3D point-cloud editing, not video; "2,214 synthetic demonstrations generated in 22 seconds" from one demo per task; 8 real tasks; the "800 generated ~ 200 real" and "94.7%" claims surfaced in a search summary were not found on the project page (unverified; do not cite).

### B9. WorldModelBench (2025, arXiv 2502.20694)
- Source: https://arxiv.org/html/2502.20694
- What: instruction following, physics adherence, and commonsense of video models used as world models, on 350 image+text conditions across 7 domains.
- Metrics: instruction following "Four levels ... (scores 0-3)" from subject absent/stationary to task fully completed; physics "Five fundamental physical laws ... Each law is assigned a binary score of 0 or 1, totaling scores from 0 to 5" (Newton's first law, conservation of mass / solid mechanics, fluid mechanics, impenetrability, gravitation); commonsense "Frame-wise quality: Whether there is visually unappealing frames or low-quality content. Temporal quality: whether there is noticeable flickering, choppy motion, or abrupt appearance." Judge: VILA-2B fine-tuned on "8336 complete votes, translating into 67K human labels" from 65 annotators.
- Headline numbers: judge "4.1% averaged prediction error on all 350 instances", 9.9% lower than GPT-4o; human inter-rater "70% pairwise agreement". Leaderboard: KLING 8.82, Minimax 8.59, Mochi 7.62.
- Metric-to-policy link: not stated.
- Reuse for LEGO: the three-axis rubric (0-3 instruction, 0-5 physics laws, frame/temporal quality) plus a small fine-tuned VLM judge; impenetrability and mass conservation map directly onto "bricks must not interpenetrate, appear, or vanish".

### B10. VLM-as-judge protocols for instruction following
- VideoScore (2024, https://arxiv.org/html/2406.15252): five 1-4 axes (Visual Quality, Temporal Consistency, Dynamic Degree, Text-to-Video Alignment, Factual Consistency); Mantis-Idefics2-8B fine-tuned on VideoFeedback (37.6K videos, "20 expert raters"); Spearman with humans 77.1 vs GPT-4o 23.0 on VideoFeedback-test. Factual Consistency checklist includes "spontaneous upward movement against gravity". Pairing Temporal Consistency with Dynamic Degree stops a frozen clip from winning on consistency alone.
- VideoAgent (ICML 2025, https://arxiv.org/html/2410.10076): GPT-4 Turbo asked whether a generated plan video is acceptable on "trajectory smoothness, physical stability, and achieving the goal"; binary accept/reject or descriptive feedback; VLM is "a proxy for the environment's reward". Judge accuracy 69% binary (75% with a weighted prompt that rejects "when the VLM is uncertain"); descriptive feedback 73.5% on problem identification. Meta-World success AVDC 19.6% to VideoAgent-Online 38.2% (53.7% with replanning); Bridge human-judged 42.0% to 64.0%. Only environment-verified successful rollouts are fed back into training.
- GR00T N1 (2025, https://arxiv.org/html/2503.14734): generated "827 hours of video data ... augmenting it by around 10x"; "we use a commercial-grade multimodal LLM as a judge and feed the downsampled 8 frames to filter out neural trajectories that do not follow the language instruction precisely"; prompts constrained to "physically feasible combinations". Neural-trajectory post-training "+5.8% on average across the 8 tasks with the GR-1 Humanoid". This is the only production-scale per-clip VLM filter found.
- GEVRM (2025, https://arxiv.org/html/2502.09268): scores generated goal frames with FID/FVD/SSIM/PSNR/LPIPS and reports CALVIN success (0.92/0.70/0.54 for 1/2/3 chains); no gate, the policy consumes generated goals directly. A negative example for filtering.
- T2V-CompBench (2024, https://arxiv.org/abs/2407.14505) and EvalCrafter (2023, https://arxiv.org/abs/2310.11440): "MLLM-based metrics", "detection-based metrics", "tracking-based metrics" for action binding and object interactions; correlation coefficients not in the abstracts. Tracking-based action binding is the right shape for "brick X attaches to brick Y".
- Genie Envisioner / EWMBench (2025, https://arxiv.org/html/2508.05635): trajectory metrics (inverse symmetric Hausdorff, "Normalized Dynamic Time Warping (NDTW)", Wasserstein distance of velocity/acceleration profiles) plus VLM-detected predefined error types; "EWMBench rankings exhibit strong concordance with human judgments" (numbers not stated). Velocity-profile distance penalises implausible motion.

### B11. Sim-to-real and validity checks for generated tasks/data (GenSim, RoboGen, Gen2Sim)
- GenSim (2024, https://arxiv.org/html/2310.01361): staged pass rates "syntax-correct", "runtime-verified", "task completed" (scripted oracle), then an LLM critic and a human check ("average human time is around 10 seconds" per task). 120 generated tasks, 100 kept. Sim-to-real: pretraining on 70 GPT-4 tasks reaches "62.5%" real success vs "27.5%" CLIPort-only; "40% zero-shot transfer to new tasks in simulation". Ablation is over the number of generated tasks, not a real/generated ratio.
- RoboGen (ICML 2024, https://arxiv.org/html/2311.01455): "Gemini-Pro ... to verify the retrieved assets and filter out the undesired ones"; scene validity by BLIP-2 similarity plus manual inspection; "Over all 69 benchmarked tasks, RoboGen achieves an average success rate of 0.774"; audit "Out of 155 generated tasks ... we found 13 failures due to incorrect scene generation" and "6 failure cases ... for supervision". Reports validity as counts, which is the honest format.
- Gen2Sim (2023, https://arxiv.org/html/2310.18308): asset quality by visual comparison; validity implicit in "successfully train RL policies"; "No explicit filtering of generated assets or tasks mentioned"; per-task success not stated. Negative example.

### B12. Policy success vs number or fraction of generated demonstrations (the table that matters)
| Paper | Data type | Real vs generated | Result (quoted) |
|---|---|---|---|
| DreamGen (B1; https://arxiv.org/html/2505.12705v2 Table 4, RoboCasa, GR00T N1, 1:1 co-training) | generated video + IDM actions | 30 real: 17.44% vs 30 real + NT 23.32%; 100 real: 32.07% vs 39.94%; 300 real: 49.59% vs 57.61%; "ONLY NT" 20.55% | project page: "log-linear slope between the total number of neural trajectories and the downstream robot policy performance", 0 to 240k NT; the NT count per Table 4 row is unverified (one extraction says 240k) |
| DreamGen real robots (about 10 real demos per task + NT) | same | GR1 average 37.0 to 46.4 (hammering 60.0 to 65.0, wiping 36.6 to 49.0, folding 27.0 to 37.0, stacking 25.0 to 35.0); DROID Franka 23 to 37; SO-100 21 to 45.5 | cost: 240k RoboCasa NT took "54 hours on 1500 NVIDIA L40 GPUs" |
| AnchorDream (B8) | generated video (Cosmos-Predict2 2B) | real 50 vs real 50 + 500 generated: 28% to 63% average over 6 real tasks; sim 22.5% to 30.7% with 300 generated | no filter |
| RoboTransfer (https://arxiv.org/html/2505.23171) | geometry-conditioned multi-view video, ACT policy, 100 real ALOHA demos per task | Fig. 10 sweeps synthetic proportion "0% to 100%"; "Both metrics peak at a 50/50 ratio"; spoon pick-and-place 13.3% (real only) to 46.7% (50/50); towel 12% to 28% | intermediate ratio values not in text; "no mention of filtering" |
| MimicGen (https://arxiv.org/html/2310.17596) | replayed segments in sim, success-filtered | Square D0: 10 source demos 11.3%, 1000 generated 90.7%, diminishing at 5000; "performance is comparable on 200 MimicGen demos and 200 human demos" | generation acceptance 8.2% (Factory Gear Assembly D1) to 82.3%; real policy 36% Stack, 14% Coffee |
| DexMimicGen (https://arxiv.org/html/2410.24185) | same, bimanual/dexterous | Drawer Cleanup 0.7% (source) to 76.0% (1000 generated); Threading 1.3% to 69.3%; Piece Assembly 3.3% to 80.7%; real can-sorting 0% (4 source) to 90% (40 generated) | success-function rejection |
| RoboCasa (https://arxiv.org/html/2406.02523) | MimicGen in sim | Human-50 (1250) 28.8%; Generated-100 (2,400) 26.3%; Generated-300 (7,200) 35.0%; Generated-3000 (72,000) 47.6%; real: real only 13.6% vs real + sim 24.4% (seen), 2.6% vs 9.3% (unseen) | admits generated demos "exhibited undesirable effects, such as jerky motions and collisions" despite the success filter |
| CACTI (https://arxiv.org/html/2212.05711) | inpainting augmentation | sim held-out: 10 layouts 14.1%, 50 layouts 31.6%, 100 layouts "47.2% +/- 4.5" | masked region only |
| GenAug (https://arxiv.org/html/2302.06671) | depth-guided inpainting of 10 real demos per task | unseen envs 38% to 80%, unseen place objects 8% to 54%, unseen pick objects 10% to 46% | bounding-box overlap check |
| ROSIE (https://arxiv.org/html/2302.11550) | Imagen-Editor inpainting | e.g. place into unseen sink 0.0 to 0.60; new backgrounds 0.33 to 0.71 | no filter; "InstructionAug" text-only control |
| Gen2Act (https://arxiv.org/html/2409.16283) | zero-shot generated human video as conditioning | MG/G/OTG/MTG 83/67/58/30 (avg 60) vs Vid2Robot avg 37; +400 teleop demos avg 64 | no filter |
| Video2Policy (https://arxiv.org/html/2502.09886) | sim tasks from internet video | generalist about 50% (10 tasks) to 75% (100 tasks) on unseen tasks; real "47%" | GPT-4o picks best code candidate |
| Cosmos Policy sim-to-real (B8) | sim-only WAM | 800 sim demos per task, 0 real: 35% avg vs DP 10 real 5%, 50 real 25% | no video metric |
Not usable: GEVRM, EnerVerse, Cosmos-Transfer1, This&That, AVDC, UniPi, Vid2Robot report no real/generated sweep.

---

## C. Decision summary

### C1. What we could realistically use in a month, and what each buys

Human-hands case (phone videos of people assembling LEGO):
1. Our own phone captures (no substitute exists). Every LEGO-specific public asset is a simulator, a layout set, rendered manual images, or a static scan (A1). The nearest real human assembly video is Assembly101 (513 h, 8 static + 4 ego views, 18M 3D hand poses, CC BY-NC) and its AssemblyHands subset (3.0M images with 4.20 mm hand keypoints); both are take-apart toy vehicles, not bricks. Use Assembly101 to (a) sanity-check a hand-pose scorer on small-part assembly and (b) as an out-of-domain fine-tuning control if we want to show the video prior needs LEGO data specifically.
2. IKEA Video Manuals (98 internet videos, part 6-DoF pose per frame, step alignment, CC BY 4.0) is the annotation template to copy: step text, per-frame part pose, sub-assembly masks. Cheap to imitate on 20-50 of our own clips and it gives an assembly-state ground truth for generated-video scoring.
3. Brick-state oracles, all public and permissive: LTRON (MIT; LDraw scene graph with connection edges), StableLego (MIT; Gurobi force-based stability), StableText2Brick (MIT/CC BY; 47k stable layouts with captions), BrickNet (typed connectors on the LDraw library). These are how we test whether the final state of a generated clip is a legal, stable brick structure once the state is lifted from pixels.
4. LEGO Co-builder (CC BY-NC-SA; 65 manuals, 10,428 step texts, rendered states) supplies step-level instruction text and shows that a stock VLM scores only 40.54% F1 on LEGO state detection, so a VLM judge needs a task-specific prompt or fine-tune.
5. Hand-pose scoring: HOT3D or ARCTIC (MANO ground truth, egocentric and multi-view) to validate whatever off-the-shelf hand tracker we run on generated frames; EgoDex (829 h, CC BY-NC-ND) only as a hand-motion prior if we fine-tune a hand-consistency model, and it cannot be redistributed.

Robot-arm case (the lab's arm executes the demonstrations):
1. The lab's own robot captures on the BrickMatic / Yaskawa GP4 stack (A1, Prompt-to-Product) or the FANUC LR-mate setup (SaLfD). Nothing public contains robot LEGO video.
2. FurnitureBench and the OXE "Furniture Bench Dataset" and "CMU Franka Pick-Insert Data" subsets are the only public real-robot assembly-flavoured video; see A5 for sizes. They are transfer targets or co-training data for a robot video prior, not LEGO.
3. RH20T (110k contact-rich sequences with paired human videos and force/torque, CC BY-SA / CC BY-NC split) if we want a human-to-robot paired prior for contact-rich insertion.
4. BrickSim (A6, 2026, the lab's own Isaac Sim brick simulator with snap-fit mechanics, "100% accuracy in static stability prediction" on 150 real assemblies, open source) is the digital twin for replaying pseudo-labelled actions and for stability checks. WorkBenchMark (A6, 400 Duplo tasks in four tiers, VLA baseline 82/63/23/2%) is the closest external benchmark to report against once it is released. BrickCraft (A6) is the lab's learned visuomotor LEGO policy and the natural consumer of generated robot videos.
5. Generic sim assembly benchmarks (A5: RLBench, ManiSkill peg insertion, Isaac Lab Factory / IndustReal / AutoMate) give cheap insertion tasks; none has studs, so they are a proxy for "press-fit insertion", not the target.

### C2. Minimal evaluation protocol, in order of cost

Each tier gates the next; report the pass rate at every tier so reviewers can see where generated data dies.

Tier 0 (free, every clip): distribution metrics that are not blur-blind. Report JEDi (V-JEPA MMD; "FVD fails to detect low blur noise", B7) or FVD with self-supervised features (Ge et al., B7) plus TRAJAN-style point-track motion score (B7). Precedent: Beyond FVD, Ge et al. 2024, Direct Motion Models 2025. Do not headline plain FVD.

Tier 1 (cheap, every clip): VLM instruction-following judge. GR00T N1's production filter is the simplest precedent: downsample to 8 frames and ask a strong multimodal LLM whether the clip follows the instruction "precisely" (B10). Copy DreamGen's binary prompt with a strong VLM (GPT-4o class; DreamGen reports r = 0.93-1.00 with humans for GPT-4o and much worse for Qwen-7B, B1) and PhyGenBench's ordered-keyframe questions for the step sequence (grasp, align, press, release; B4). Add VBench-2.0 Motion Rationality "consequence" questions ("after the press, is the brick seated on the studs?") with redundant phrasing to cut false positives (B3). Use VideoAgent's abstain rule (reject when the VLM is uncertain; precision rose with a weighted prompt, B10). Validate on 50-100 human-labelled clips with three raters and the VideoPhy-2 joint criterion SA>=4 and PC>=4 (B6), or WorldModelBench's 0-3 instruction / 0-5 physics-law rubric (B9). Precedent: DreamGen Sec. 4 and App. H, GR00T N1, VideoAgent, VBench-2.0, VideoPhy-2, WorldModelBench.

Tier 2 (cheap to moderate, every clip): physics plausibility. Run VideoCon-Physics as a filter, not a metric (its PC Pearson with humans is 0.25-0.37 even in-domain, B6), plus VBench-2.0 Instance Preservation (bricks must not appear, vanish, merge, or split; B3). For the LEGO-specific check: lift the final frame (or every step boundary) to a brick layout with the lab's brick detector and run the StableLego / LegoGPT stability and connectivity check (A1; LegoGPT went from 24.0% to 98.8% stable structures with this check, which shows how often unconstrained generators violate it). Report the fraction of generated clips whose end state is a legal, stable structure that matches the instruction's target layout; in the robot case, run the same check in BrickSim (A6), which also catches interpenetration during the press. Precedent: Cosmos object-level IoU against simulated ground truth (B2), SaLfD digital-twin verification (A1), LegoGPT rollback (A1), MimicGen / RoboCasa success-function rejection (B12).

Tier 3 (moderate, human case only): hand-pose consistency. Run a MANO-based tracker validated on HOT3D or ARCTIC (A3) over generated frames; score per-frame anatomy validity (VBench-2.0 Human Anatomy detector, B3), temporal joint jitter, and contact consistency (hand must be in contact with the brick that moves). Precedent: VBench-2.0 Human Anatomy, ARCTIC contact annotations, AssemblyHands keypoint protocol. In the robot case, replace with an arm-mask consistency check against the known arm render (AnchorDream's robot-only conditioning, B8) and IDM action replay in the digital twin (DreamGen App. H.4, B1).

Tier 4 (expensive, the only one that answers the question): downstream policy success.
- Human-hands case: there is no direct action label. Precedent chain: pseudo-label with an IDM or latent actions (DreamGen), or retarget hand pose to the arm (EgoMimic / EgoZero / DexWild recipe, A3), then co-train with a small real-robot set and measure real-robot success on held-out LEGO placements. Report success vs number of generated clips at fixed real demos (DreamGen: 10 real per task, 1:1 sampling; AnchorDream: 50 real + 500 generated, 28% to 63% average; B1, B8). A sim proxy is possible only for the insertion primitive (A5 peg tasks), not for stud press-fit.
- Robot-arm case: this is exactly DreamGen's setting and is cheaper. Fine-tune the video model on robot LEGO clips, generate neural trajectories from new initial frames and new instructions, pseudo-label with an IDM trained on the same robot's real clips (DreamGen: "conditioned on two image frames and is trained to predict action chunks between the image frames"), then measure (i) IDM-replay success in BrickSim (A6) or the Gazebo twin (SaLfD, A1) as the gate and (ii) real-robot success on the BrickMatic / GP4 or FANUC stack for seen structures, new structures, and new backgrounds, mirroring DreamGen Table 1 (new behaviours 11.2% to 43.2%, new environments 0.0% to 28.5%). Report the three-row table DreamGen uses (real N, real N + generated, generated only) at two or three N: DreamGen Table 4 gives 30/100/300 real with +5.9/+7.9/+8.0 points from co-training and 20.55% from generated only (B12). Real-demo budget: 10-50 per task with 100-500 generated per task is the range precedents use (DreamGen 10 real + 100-300 NT; AnchorDream 50 real + 500 generated; B1, B8). AnchorDream's robot-only render conditioning is the trick to keep the arm embodiment-consistent in generated clips.
- Which datasets matter for which case: robot brick/peg assembly targets are BrickSim, WorkBenchMark, FurnitureBench, AutoMate/IndustReal, ManiSkill PegInsertionSide, RLBench insert tasks (A5, A6); robot video priors are DROID, BridgeData V2, OXE (incl. the Furniture Bench and CMU Pick-Insert subsets), RH20T (A4). Human assembly targets and priors are Assembly101/AssemblyHands, IKEA Video Manuals, IKEA-ASM, HoloAssist, HA-ViD (A2) and the hand-pose sets HOT3D, ARCTIC, DexYCB, EgoDex (A3). Nothing public covers robot LEGO video; nothing public covers human LEGO video.
- Ablation both cases must include: real-only, real+generated, and real+generated-after-filtering, with the filtered-out fraction reported per tier, and a synthetic-fraction sweep because RoboTransfer found success "peak[s] at a 50/50 ratio" and falls beyond it (B12). Precedent for the shape of the curve: DreamGen Table 4 and Fig. 4, RoboCasa's Human-50 vs Generated-100/300/3000 rows (28.8 / 26.3 / 35.0 / 47.6%), AnchorDream sim scaling (22.5% to 30.7%), MimicGen 10 source to 1000 generated (11.3% to 90.7%).

### C3. Honest risks and how prior papers guarded against them

1. Generated LEGO videos that look right but violate brick connectivity or stability. LegoGPT measured this directly: an unconstrained generator produced "100% valid, 24.0% stable" structures; only rejection sampling plus physics-aware rollback reached 98.8% (A1). Video models have no such constraint, so expect a large fraction of clips whose end state is not a legal build. Guard: lift end states to layouts and run StableLego / LTRON connectivity checks (Tier 2); reject clips that fail; report the rejection rate as a result, not a footnote. SaLfD is the in-lab precedent for a simulator gate (63-77% to 86-100%, A1). Cosmos's object-mask IoU against a simulated ground truth is the external precedent (B2).
2. Evaluation that rewards blur or stillness. FVD "fails to detect low blur noise" and "incorrectly suggests an improvement" (Beyond FVD), and drops when motion is removed (Ge et al.). A model that smears hands during the press will score well on FVD and on per-frame VLM checks. Guard: JEDi or self-supervised-feature FVD, a point-track motion metric (TRAJAN), Dynamic Degree from VBench, and ordered keyframe questions that require the press event to be visible (PhyGenBench order verification). Physics-IQ's two-take physical-variance ceiling gives a normalised motion-IoU reference if we record each step twice.
3. VLM judges that pass wrong assembly states. LEGO Co-builder shows GPT-4o at 40.54% F1 on LEGO state detection; DreamGen shows a 7B Qwen judge at 18.8% IF where humans give 91.7% (B1). Guard: use the strongest VLM, task-specific ordered questions, and always report human agreement (Pearson or Spearman) on a labelled subset before trusting the judge; PhyGenEval's tau 0.78 and DreamGen's r > 0.9 are the bar.
4. Hallucinated or duplicated objects between frames (extra bricks, bricks that vanish). Guard: VBench-2.0 Instance Preservation detector; brick-count consistency between first frame, last frame, and the instruction's parts list.
5. Filtered data that is still useless, or a success filter that passes bad motion. Physics and IF scores rank models but no paper except DreamGen ties them to policy success, and DreamGen only reports "a positive correlation" without a coefficient (B1). AnchorDream, Video Policy, RoboTransfer, and the Cosmos Policy sim-to-real paper use no filter at all and attribute failures to "unrealistic video predictions" (B8, B12). RoboCasa warns that even success-filtered generated demos "exhibited undesirable effects, such as jerky motions and collisions" (B12). Guard: the Tier 4 ablation with and without filtering, plus a motion-quality metric (velocity-profile distance as in EWMBench, or TRAJAN) reported next to the success filter.
6. Licence and redistribution. Assembly101, AssemblyHands, EPIC-KITCHENS, HOI4D, DexYCB are CC BY-NC; EgoDex is CC BY-NC-ND (no derivatives); ARCTIC is non-commercial with registration; HOT3D and Ego4D have custom agreements. Only IKEA Video Manuals (CC BY 4.0), LTRON / StableLego / StableText2Brick / MEPNet (MIT), BridgeData V2 and DROID (CC BY 4.0), and OXE (CC BY) are permissive. Plan any released model or dataset accordingly.
