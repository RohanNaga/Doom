# Turning LEGO-assembly video into robot supervision — verified source notes

Compiled 2026-09-16 for a same-night decision. Every number below is quoted from a primary source
(arXiv abstract or LaTeXML HTML, GitHub README, official project page) fetched during this session.
Where a source is silent the field says **not stated**; where a claim could not be checked against the
repo or PDF it says **unverified**. Lines marked *(inference)* are this file's own reasoning and are
not sourced. Nothing was invented to fill a gap.

Written to the worktree copy of `.claude/analyses/lit/` because the base-repo `.claude/` directory is
write-guarded from a worktree session. Not committed.

**Question.** Given phone-camera videos of LEGO assembly, real or generated, how do we turn each video
into robot-usable supervision: hand and finger poses, brick 6-DoF poses and assembly state, contact and
grasp events, and an action sequence a policy could imitate? And when the "video as robot data" papers
label generated videos, how accurate is that labelling reported to be?

**Addition from Rohan (mid-task).** The demonstrator may be the lab's own robot arm (the Intelligent
Control Lab's FANUC / Kinova LEGO stack: Liu, Sun, Liu 2023; BrickCraft 2026) rather than a human.
Section 7 treats that route in parallel; Section 8 compares the two.

## Headline

1. **Hand pose** from monocular video reaches 5-8 mm PA-MPJPE on benchmarks, but every method drops
   sharply on occluded joints (Hamba HInt occluded PCK@0.1: 59-67 vs 84-90 visible), and a hand
   pinching a 2x2 brick puts most fingertips in the occluded bucket. Licences bind harder than accuracy:
   MANO is non-commercial; WiLoR and HaWoR are CC-BY-NC-ND; only HaMeR and Dyn-HaMR code is MIT.
2. **Brick pose** from a phone: no method in the set targets small textureless repeated parts. FoundationPose
   and SAM-6D need depth; MegaPose is RGB-only with a mm CAD mesh (LEGO CAD is known, which is the one
   advantage); the RGB feed-forward 3D models (DUSt3R, VGGT, MonST3R) are explicitly not metric. Eye-in-Finger
   (2025) states LEGO insertion tolerates only 0.4-2.0 mm calibration error and that external cameras are
   occluded at insertion. A phone pipeline should target discrete assembly state and connectivity, not
   continuous 6-DoF at insertion time.
3. **Label accuracy of generated video** is essentially unmeasured. Only VPT reports it (90.6% keypress,
   R² 0.97 mouse, 1,962 h labelled Minecraft, non-causal). DreamGen, LAPA, UniPi, Genie, UniVLA, Moto report
   only downstream success; DreamGen's IDM-only neural trajectories give 20.55% on RoboCasa, about what 30
   real demos give. The one insertion-specific number is AVDC's 6.7% on Meta-World `assembly`.
4. **Human-to-robot retargeting** has never been evaluated on bimanual fine assembly. EgoZero states the
   precision floor of the Aria+HaMeR hand pipeline as "1-2cm error", "preventing high-precision tasks", and
   monocular depth at ">5cm error". Phantom gives the only explicit hand-to-gripper rule (thumb-index midpoint,
   plane, and distance) but needs RGB-D, is single-arm and quasi-static.
5. **If the demonstrator is the robot**, the hand problem disappears and the action label is free
   (proprioception). BrickCraft already sidesteps brick pose with a rendered "situated manual" aligned by
   ECC and reaches 86.25% per-skill success over 240 trials. That is the tractable route; Sections 7-9.

---

## 1. Hand pose from monocular video

### HaMeR (CVPR 2024)
- Source: https://arxiv.org/abs/2312.05251, https://geopavlakos.github.io/hamer/, https://github.com/geopavlakos/hamer
- What: transformer hand-mesh recovery from one RGB crop; "follows a fully transformer-based architecture".
- Input: monocular RGB, per-frame, no temporal model. Needs a box and handedness: "Given a bounding box of the hand and the hand side (left/right), we use a deep network to reconstruct the hand in 3D in the form of a MANO mesh". Assumed intrinsics, regressed translation; no calibration. Egocentric and third-person (HInt covers Ego4D, EPIC-VISOR).
- Outputs: MANO, hence 3D joints and mesh, camera space. Confidence: not stated.
- Accuracy: FreiHAND PA-MPJPE 6.0 mm, PA-MPVPE 5.7 mm, F@15 0.990; HO3Dv2 PA-MPJPE 7.7 mm. HInt PCK@0.05/@0.1: New Days 48.0/78.0, VISOR 43.0/76.9, Ego4D 38.9/71.3.
- Runtime: not stated.
- Code, licence: MIT; ships ViTPose; MANO not included ("Please visit the MANO website and register").
- Failure modes: none stated; Ego4D is its worst split. *(inference)* Per-frame, so brick-pinch poses jitter and depth wanders.

### WiLoR (CVPR 2025)
- Source: https://arxiv.org/abs/2409.12259, https://github.com/rolpotamias/WiLoR
- What: "a real-time fully convolutional hand localization and a high-fidelity transformer-based 3D hand reconstruction model", trained on "over than 2M in-the-wild hand images".
- Input: monocular RGB, weak perspective, "no full intrinsic camera matrix is required"; per-frame, "smooth 3D hand tracking from monocular videos, without utilizing any temporal components"; multi-hand.
- Outputs: MANO and joints, camera space; detector "predicts both bounding boxes and 'hand side label'".
- Accuracy: FreiHAND PA-MPJPE 5.5 mm, PA-MPVPE 5.1 mm, F@15 0.993; HO3D PA-MPJPE 7.5 mm. HInt: not stated.
- Runtime: detector "more than 130 FPS" (medium) / "up to 175 FPS" (small) on RTX 4090; reconstruction fps not stated.
- Code, licence: "CC-BY-NC--ND License"; Ultralytics and MANO dependencies.
- Failure modes (stated): "still fails to recover challenging cases": extreme finger poses, crowded scenes with small hands, no inter-hand contact constraints. *(inference)* No contact term, fingers will pass through the brick.

### HaWoR (CVPR 2025 Highlight)
- Source: https://arxiv.org/abs/2501.02973, https://github.com/ThunderVVV/HaWoR
- What: world-space egocentric hand motion: "reconstructing the hand motion in the camera space and estimating the camera trajectory in the world coordinate system", "an adaptive egocentric SLAM framework", "a novel motion infiller network that effectively completes the missing frames".
- Input: monocular egocentric video; masked DROID-SLAM plus Metric3D weights; intrinsics for SLAM; off-the-shelf hand tracker upstream.
- Outputs: "MANO pose {Θ∈R^15x3}, shape {β∈R^10}, orientation {Φ∈R^3}, root translation {Γ∈R^3} expressed in world-coordinate system", plus camera trajectory; both hands independently.
- Accuracy: DexYCB PA-MPJPE 4.76 mm (WiLoR 5.01), 5.07 mm under 75-100% occlusion; HOT3D W-MPJPE 33.20 mm (HMP-SLAM 119.41), WA-MPJPE 11.27 mm; camera ATE 3.36 m aligned.
- Runtime: "40 milliseconds per frame" for the hand network; full pipeline "still far from real-time".
- Code, licence: "CC-BY-NC-ND License"; MANO separate.
- Failure modes (stated): "reliance on hand-tracking outputs from an off-the-shelf method... can propagate erroneous detections"; "without any inter-penetration constrains". *(inference)* Tripod phone gives the SLAM stage no parallax; the useful part is the infiller over brick-occluded frames.

### Hamba (NeurIPS 2024)
- Source: https://arxiv.org/abs/2407.09646, https://github.com/humansensinglab/Hamba
- What: single-image hand mesh with "a Graph-guided State Space (GSS) block... uses 88.5% fewer tokens than attention-based methods".
- Input: monocular RGB, single frame; ViTPose dependency.
- Accuracy: FreiHAND PA-MPVPE 5.3 mm, F@15 0.992; HO3Dv3 PA-MPJPE 6.9 mm. HInt PCK@0.1 visible: 88.4 / 89.6 / 84.3 (New Days / VISOR / Ego4D); **occluded: 62.8 / 66.6 / 59.2**. The most LEGO-relevant number in this section.
- Runtime: not stated.
- Code, licence: CC BY-NC 4.0; MANO separate.
- Failure modes (stated): "fails to robustly predict the palm orientation... or when it misses the finger due to motion blur"; "complex finger gestures". *(inference)* Expect the ~25-point occluded drop for fingers wrapped around a brick, uncorrelated frame to frame.

### HaPTIC, "Predicting 4D Hand Trajectory from Monocular Videos" (2025, arXiv)
- Source: https://arxiv.org/abs/2501.08329, https://judyye.github.io/4dhands
- What: "the first feed-forward method that estimates consistent 4D hand trajectories directly from monocular video"; M=8 frame window with cross-view attention.
- Input: monocular video with hand boxes; "egocentric and allocentric"; no intrinsics required.
- Outputs: MANO θ, β, wrist trajectory in camera coordinates.
- Accuracy: ARCTIC-EXO GA-MPJPE 1.88, FA-MPJPE 4.49; DexYCB GA-MPJPE 1.80; ARCTIC-EGO 2.39. **Units as rendered look implausibly small for a global metric; ranking reliable, unit unverified.** HInt PCK@0.1 79.0 / 79.5, "margin is even more significant on occlusion subsplit" over HaMeR.
- Code, licence: "Model will be publicly released"; **unverified**.
- Failure modes (stated): "assumes reliable 2D hand tracking system"; severe object or inter-hand occlusion.

### Dyn-HaMR (CVPR 2025 Highlight)
- Source: https://arxiv.org/abs/2412.12861, https://github.com/ZhengdiYu/Dyn-HaMR
- What: "the first approach to reconstruct 4D global hand motion from monocular videos recorded by dynamic cameras in the wild"; multi-stage optimisation with SLAM, an interacting-hand generative prior for infilling under occlusion, HaMeR initialisation with "robust hallucination prevention and handedness correction".
- Input: monocular RGB(-D) video, moving or static camera, two hands; camera parameters optional; no masks, no object template.
- Outputs: two-hand MANO in world space, per frame, plus camera motion. No object.
- Accuracy: InterHand2.6M MPJPE 7.94 (HaMeR 9.84); H2O dynamic G-MPJPE 45.6 / MPJPE 22.5 (HaMeR 96.9 / 32.9); HOI4D dynamic G-MPJPE 58.5 (HaMeR 201.6); FPHA PA-MPJPE 12.5.
- Runtime: per 128 frames on A100: DPVO 1.49 min, HaMeR 3.18 min, 2D keypoints 1.45 min, stage II 2.5 min, stage III 1.69 min (~8-10 min).
- Code, licence: MIT; MANO separate.
- Failure modes (stated): "works for limited time-horizons"; "We will also work on improving the hand priors and incorporate object interactions." *(inference)* Best-fitting hands component for phone video (moving camera, two hands, MIT), but the SLAM front end wants a textured static background, which a close-up over a bare table lacks.

### PCIE_EgoPose / HP-ViT+ (EgoExo4D challenge winner, 2025)
- Source: https://arxiv.org/abs/2505.24411
- Outputs "21 3D hand joints in millimeters within ego-centric coordinate system"; "8.31 PA-MPJPE in the Hand Pose Challenge"; joints only, no MANO; repo not stated. Useful as an egocentric-occlusion accuracy reference only.

### MANO and 2D front ends
- Source: https://mano.is.tue.mpg.de/; https://arxiv.org/abs/2006.10214; https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
- MANO: "learned from around 1000 high-resolution 3D scans of hands of 31 subjects"; pose 15x3 = 45 axis-angle plus 10 shape, root orientation and translation separate, 21 joints. "freely available for research purposes", requires "signing up and agreeing with the license": non-commercial. Every MANO method inherits this.
- MediaPipe Hand Landmarker: 21 landmarks, handedness, image and "world coordinates"; Pixel 6 latency 17.12 ms CPU / 12.27 ms GPU; Apache 2.0. No MANO.
- *(inference)* Chain: cheap 2D stage for boxes and handedness gating which frames have two visible hands, then a MANO regressor.

**Verdict 1.** Camera-space MANO at 5-8 mm PA-MPJPE is available; Dyn-HaMR is the right bimanual world-space component for phone video and is MIT. No source reports small-part assembly numbers; nearest proxies are Hamba's occluded PCK (59-67) and HaWoR's 5.07 mm under 75-100% occlusion on DexYCB (large objects). Only a pilot on the lab's own footage settles it.

---

## 2. Hand-object interaction reconstruction

### HOLD (CVPR 2024)
- Source: https://arxiv.org/abs/2311.18448, https://github.com/zc-alexfan/hold
- What: "the first category-agnostic method that reconstructs an articulated hand and object jointly from a monocular interaction video"; per-sequence neural-field optimisation.
- Input: monocular RGB video, no template; masks from "SAM-track... point-prompting for the first frame"; pose init from an off-the-shelf hand estimator plus detector-based SfM (HLoc); "We downsample all sequences every 5 frames" (lab), "every 10 frames" (wild). Two-hand mode: "Bimanual category-agnostic reconstruction" (ARCTIC).
- Outputs: MANO-based hand, object implicit surface, per-frame poses.
- Accuracy: HO3D MPJPE 24.2 mm, object CD 0.4 cm², F10 96.5% (iHOI 38.4 / 3.8 / 75.8); unseen categories CD 0.3 cm², F10 96.9%; in-hand scanning CD 0.5 cm².
- Runtime: "initial training for 100 epochs, which requires around 10 hours using an A100 GPU. The final training takes 200 epochs."
- Code, licence: MIT (MANO separate).
- Failure modes (stated): "The reconstruction of thin or textureless objects is limited by our use of detector-based Structure from Motion for pose initialization." *(inference)* A brick has no repeatable keypoints and a repeated stud lattice; SfM init fails or locks onto a wrong match, and 10+ h per clip is impractical for a corpus.

### MCC-HO + RAR (2024; 3DV 2026 per page)
- Source: https://arxiv.org/abs/2404.06507
- What: transformer predicts hand and held-object geometry from one RGB image plus a HaMeR hand; RAR uses GPT-4(V) to retrieve or generate a template mesh and rigidly aligns it across frames.
- Input: "does not require an input hand-object segmentation mask, depth image, or hand joint locations"; single hand.
- Accuracy: single-image F-5/F-10/CD: DexYCB 0.36/0.60/3.74; MOW 0.15/0.31/15.2; HOI4D 0.52/0.78/1.36. Video HOI4D with RAR 0.74/0.91/0.64 (G-HOP 0.61/0.89/0.8).
- Runtime, licence: not stated.
- Failure modes (stated): "relies on the ability to obtain accurate hand and object masks which are tracked across frames". *(inference)* Retrieval is the wrong prior: a brick is defined by stud count and exact dimensions, and a generated "brick-like" mesh is metrically wrong.

### EasyHOI (CVPR 2025)
- Source: https://arxiv.org/abs/2411.14280
- What: single-image HOI by chaining large models (HOI reasoning, hand recon, inpainting, InstantMesh) then contact-aligned optimisation. Single view, not video.
- Accuracy: ARCTIC F5 0.155, F10 0.272, CD 1.089 (MOHO 0.072 / 0.136 / 12.878); OakInk CD 1.035; DexYCB CD 1.628.
- Runtime: reconstruction "average of 58.97 seconds", optimisation "57.03 seconds" per image.
- Licence: not stated.
- Failure modes (stated): "the inpainting model occasionally introduces artifacts, causing the object being held to blend with the background"; "ambiguous grasping poses and motion blur". *(inference)* An LRM asked to hallucinate an occluded brick invents smooth geometry, not a stud lattice.

### "Hands-on-Everything": not found
No paper with that title surfaced in arXiv-targeted or general search; nearest hits were HOLD, MCC-HO, "Follow My Hold" (2508.18213), HORT (2503.21313). No card.

### InterWild (CVPR 2023)
- Source: https://arxiv.org/abs/2303.13652, https://github.com/facebookresearch/InterWild
- What: two interacting hands from an in-the-wild image, hands mapped into "a shared 2D scale space", hand-to-hand translation from "a geometric feature without an image as an input". Hands only.
- Input: cropped single RGB image; MANO files. Outputs: MANO for both hands plus relative translation.
- Accuracy: not stated on the pages fetched (tables in the PDF).
- Licence: "CC-BY-NC 4.0".
- *(inference)* Right bimanual component but supplies no object, contact, or temporal smoothing.

### Datasets: HOI4D, ARCTIC, DexYCB
- HOI4D (CVPR 2022; https://hoi4d.github.io/): "2.4M RGB-D egocentric video frames over 4000 sequences collected by 9 participants interacting with 800 different object instances from 16 categories over 610 different indoor rooms"; "3D hand pose, category-level object pose and hand action"; CC BY-NC 4.0. Participant count rendered as 4 on the arXiv page vs 9 on the site; check the PDF. Nothing textureless at brick scale.
- ARCTIC (CVPR 2023; https://arctic.is.tue.mpg.de): "339 sequences of dexterous manipulation of 11 articulated objects by 10 subjects", "2.1M RGB images from 8 static views and 1 egocentric view", ground truth from "a Vicon MoCap system with 54 infrared Vantage-16 cameras" with "Small hemispherical markers with 1.5mm radius". Baseline ArcticNet-LSTM allocentric hand MPJPE 21.5 mm, MRRPE right-left 49.2 mm, object success 73.5% (egocentric 53.5%). Licence not stated. The closest bimanual-contact analogue; large textured objects, marker rig, not reproducible with a phone.
- DexYCB (CVPR 2021; https://dex-ycb.github.io/): CC BY-NC 4.0, 119 GB; frame/camera counts not fetched (PDF 403). Standard single-hand known-CAD benchmark; opposite of the LEGO regime.

**Verdict 2.** Nothing here targets small textureless repeated parts; HOLD names the failure explicitly. The decomposition that survives is hands-first (Dyn-HaMR or InterWild) plus a brick track that exploits known CAD, which turns object reconstruction into pose + part ID rather than shape recovery.

---

## 3. Object pose for small textureless parts; feed-forward 3D

### FoundationPose (CVPR 2024)
- Source: https://arxiv.org/abs/2312.08344, https://github.com/NVlabs/FoundationPose
- What: "a unified foundation model for 6D object pose estimation and tracking, supporting both model-based and model-free setups".
- Input: **RGB-D**; CAD mesh or ~16 reference views; external first-frame detector ("CNOS or Mask-RCNN"); intrinsics.
- Accuracy: YCB-Video model-free ADD-S 97.4, ADD 91.5; BOP model-based AR: LM-O 78.8, **T-LESS 83.0**, YCB-V 88.0, mean 83.3 (MegaPose-RGBD 58.6).
- Runtime: "pose estimation takes about 1.3 s for one object... Tracking runs much faster at ∼32 Hz."
- Code, licence: "NVIDIA Source Code License"; released weights omit diffusion-augmented data ("Slight performance degradation is expected").
- Failure modes (stated): "false or missing detection frequently bottlenecks the 6D pose estimation"; failures from "texture-less, severe occlusion, and limited edge cues". *(inference)* An 8 mm stud pitch is at or below consumer depth-sensor resolution, so the refiner renders against noise; identical bricks confuse instance association.

### SAM-6D (CVPR 2024)
- Source: https://arxiv.org/abs/2311.15707, https://github.com/JiehongLin/SAM-6D
- Input: RGB-D plus CAD "(mm)" plus intrinsics; 42 rendered templates; single frame.
- Accuracy: BOP mean AR 70.4; **T-LESS 51.5**, **IC-BIN 58.8** (repeated identical instances), ITODD 60.2, TUD-L 90.4.
- Runtime: 4.37 s/image on RTX 3090 with SAM, 1.43 s with FastSAM.
- Licence: not stated in README fetched.
- *(inference)* Partial-to-partial point matching has nothing to match on a few-dozen-point brick; SAM merges adjacent same-colour bricks.

### MegaPose (CoRL 2022)
- Source: https://arxiv.org/abs/2212.06870, https://github.com/megapose6d/megapose6d
- What: render-and-compare coarse classifier plus refiner for novel objects; "only assumes knowledge of (i) a region of interest displaying the object in the image and (ii) a CAD model".
- Input: "RGB image (depth can also be used but is optional)"; separate RGB and RGBD checkpoints; "3x3 camera intrinsic matrix K"; "Mesh units are expected to be in millimeters".
- Accuracy: no per-dataset numbers on the pages fetched; FoundationPose's table reports MegaPose-RGBD mean AR 58.6.
- Runtime: not stated. Licence: Apache 2.0.
- *(inference)* Most attractive for LEGO: RGB-only supported and brick meshes exactly known. The symmetric brick gives a multimodal posterior; without depth the refiner converges to whichever mode the coarse stage ranked first.

### Gen6D (ECCV 2022)
- Source: https://arxiv.org/abs/2204.10776, https://github.com/liuyuan-pal/Gen6D
- Input: RGB only; posed reference images via COLMAP; scale via `align.pkl`. Accuracy: LINEMOD ADD-0.1d 94.11%; GenMOP 50.39%; 8 refs -> 27.55%. Runtime ~0.64 s on 2080Ti. Licence GPL-3.0.
- Failure modes (stated): "is not specially designed to handle occlusions"; "the test sequence for evaluation need be captured in a static scene". *(inference)* Disqualified: COLMAP on a textureless 8 mm brick yields too few correspondences.

### DUSt3R (CVPR 2024), MonST3R (ICLR 2025), CUT3R (CVPR 2025), VGGT (CVPR 2025)
- DUSt3R (https://github.com/naver/dust3r): "without prior information about camera calibration nor viewpoint poses"; pointmaps "regressed up to an unknown scale factor"; DTU overall 1.741 mm; "≈40ms on a H100 GPU" per pair; CC BY-NC-SA 4.0.
- MonST3R (https://github.com/Junyi42/monst3r): per-timestep pointmaps for dynamic scenes; outputs cameras and intrinsics; scale-and-shift-aligned metrics (Sintel Abs Rel 0.335, Bonn 0.063); "about 33G VRAM... 65 frames"; stated "vulnerable to long-term occlusion". Licence unverified, inherits DUSt3R NC.
- CUT3R (https://cut3r.github.io/): recurrent, online, "metric-scale pointmaps" ("When the groundtruth pointmaps are metric, we set ŝ:=s"); 16.58 FPS on A100 at 512x144; "may eventually drift over very long sequences". Code licence not stated. *(inference)* Metric prior is room-scale; untested at 8 mm.
- VGGT (https://github.com/facebookresearch/vggt): cameras, point maps, depth, tracks from 1-hundreds of views; canonical (not metric) scale; Co3Dv2 AUC@30 88.2; 100 frames 3.12 s / 21.15 GB on H100 at 336x518; original checkpoint non-commercial, gated "VGGT-1B-Commercial" exists; "fails in scenarios involving substantial non-rigid deformation" (hands).

### Any6D (CVPR 2025)
- Source: https://arxiv.org/abs/2503.18673, https://github.com/taeyeopl/Any6D
- What: model-free 6D pose and **metric size** from one RGB-D anchor via InstantMesh shape plus "metric scale estimation" in a joint alignment; co-authored by the FoundationPose first author.
- Accuracy: HO3D ADD-S 98.7, ADD 40.4, AR 40.4 (the one number in this set under hand occlusion); REAL275 ADD(-S) 53.5%.
- Failure modes (stated): "limitations when the initial 3D shape is inaccurate, as our approach does not incorporate shape updating". *(inference)* InstantMesh will not produce a correct stud lattice, so shape error becomes unrecoverable scale error.

**Metric scale from phone video without depth.** Not available from any RGB-only model here on what they state: DUSt3R and VGGT are explicitly scale-free, MonST3R is scale-aligned at evaluation, CUT3R's metric claim is learned from room-scale data and drifts. Every pose method gets scale from an input you supply: mm CAD mesh plus intrinsics (MegaPose, SAM-6D) or depth (FoundationPose, SAM-6D, Any6D). Known brick dimensions plus CAD is the supported route. ARKit/LiDAR depth would satisfy the RGB-D inputs but no source reports phone-depth results at 8 mm. A calibration board fixes intrinsics, not scale.

**Verdict 3.** For a phone: MegaPose-RGB with the exact brick mesh and a colour-plus-SAM instance detector is the only defensible 6-DoF candidate, and symmetry will leave rotation ambiguous. Eye-in-Finger (Section 6) argues external cameras are occluded at the insertion moment anyway. Estimate the discrete assembly state (which brick, which studs) rather than continuous pose.

---

## 4. Latent actions and inverse dynamics on actionless video

### VPT (NeurIPS 2022)
- Source: https://arxiv.org/abs/2206.11795, https://github.com/openai/Video-Pre-Training
- What: non-causal IDM trained on contractor gameplay "labels a huge unlabeled source of online data — here, online videos of people playing Minecraft".
- Input: 1,962 h labelled for the IDM; ablation 10/50/100/1,962 h, gains "plateau after 100 hours"; ~70,000 clean web hours; 128x128 at 20 Hz, 128-frame non-causal window.
- Reported labelling accuracy: **keypress accuracy 90.6%, mouse R² 0.97** on held-out data; IDM "two orders of magnitude more data efficient than a BC model trained on the same data".
- Licence: MIT; IDM weights released.
- *(inference)* Rests on actions with a visible effect within a few frames; a force-controlled insertion phase has almost none, so expect near-chance on the frames that matter.

### DreamGen (2025, NVIDIA GEAR)
- Source: https://arxiv.org/abs/2505.12705, https://arxiv.org/html/2505.12705v2, https://github.com/NVIDIA/GR00T-Dreams
- What: fine-tunes a video world model (WAN2.1) on the robot's own teleop data, prompts for new behaviours, and labels the generated videos: "We recover pseudo-action sequences using either a latent action model or an inverse-dynamics model (IDM)".
- Labeller: IDM is "diffusion transformers with SigLIP-2 vision encoder", "flow matching objective", "conditioned on two image frames and is trained to predict action chunks between the image frames", applied by "sliding window approach"; trained on the same teleop set as the video model (GR1: "2,884 GR1 trajectories of pick-and-place collected in a single lab environment"). LAM is LAPA-style on 438.1M frames, codebook 8, sequence length 16. Horizon H and action DoF: not stated.
- Reported labelling accuracy: **no number against ground truth.** "Since both approaches have similar effects, we use IDM as the default". RoboCasa (24 tasks, Table 4): GT-only 30/100/300 traj = 17.44 / 32.07 / 49.59%; IDM co-training 23.32 / 39.94 / 57.61%; **neural trajectories only = 20.55%**. GR1 real: "43.2% success rates on new behaviors in seen environments" (from 11.2%), "28.5% in completely unseen environments" (from 0%), "10 new environments", "50 neural trajectories for each of the 14 novel behavior tasks". Co-training deltas: GR1 37 -> 46.4%, Franka 23 -> 37%, SO-100 21 -> 45.5%.
- Video filtering: DreamGen Bench, Instruction Following via "Qwen-VL-2.5 with specific prompts to give a binary score (0 or 1)", Physics Alignment via "VideoCon-Physics... to get a 0 to 1 score"; human agreement ">90%".
- Runtime: 240k neural trajectories took "54 hours on 1500 NVIDIA L40 GPUs".
- Licence: repo licence **unverified**.
- Failure modes (stated): tasks "relatively simple and cover a limited portion of the robot's full kinematic capabilities"; initial frames collected manually; evaluators "risk hallucination on physics assessment". *(inference)* The IDM knows only the source robot's teleop distribution; on a drifted dream it emits confident in-distribution chunks exactly at the contact frames where the video model's physics is weakest.

### EVA, "Aligning Video World Models with Executable Robot Actions via Inverse Dynamics Rewards" (2026, CUHK-Shenzhen / DexForce)
- Source: https://arxiv.org/html/2603.17808v1
- What: IDM trained on real robot data used as an RL reward to penalise generated videos "containing kinematic violations".
- Labeller data: 1,050 RoboTwin 2.0 trajectories (sim), 250 teleop demos (real); joint-level actions for bimanual 6-7 DoF arms.
- Reported: the IDM reaches "89.52% average execution success rate across 21 diverse bimanual manipulation tasks" **on ground-truth videos**; alignment improves "kinematic plausibility by +20.9%"; real-world 64% seen / 60% OOD vs 52% unaligned.
- Licence: project page only; **unverified**.
- *(inference)* The closest measurement of "IDM labels from generated video are executable"; the paper's premise is that unaligned generations violate kinematics often.

### LAPA (ICLR 2025)
- Source: https://arxiv.org/abs/2410.11758, https://github.com/LatentActionPretraining/LAPA
- "a VQ-VAE-based objective to learn discrete latent actions between image frames"; BridgeV2 60k, OXE ~970k, SSv2 ~220k; frame gap 0.6 s robot / 2.4 s human; latents "size of 8^4 (4,096 discrete actions)"; ~272 H100-hours; MIT.
- Labelling accuracy: none direct. Downstream Language Table 62.0% vs 77.0% action-pretrained vs 15.6% scratch; SIMPLER 57.3 vs 63.5; real 50.1 vs OpenVLA 43.9.
- Failure modes (stated): "underperforms compared to action pretraining when it comes to fine-grained motion generation tasks like grasping".

### Genie (ICML 2024)
- Source: https://arxiv.org/abs/2402.15391
- LAM on ~30,000 h platformer video at 160x90, 10 FPS; codebook |A| = 8 "to permit human playability"; "the entire LAM is discarded at inference time". No accuracy vs ground truth; controllability Δ_t PSNR 1.91 / 2.07 (RT-1). Not released.

### UniPi (NeurIPS 2023)
- Source: https://arxiv.org/abs/2302.00111
- "Control actions are extracted from the generated video" by a small conv IDM (MLP 128 -> 7) trained on 20k-200k sim videos at 48x64. No label accuracy; real robot 72.6 -> 77.1%. "it can take a minute to generate". *(inference)* A brick is a few pixels at 48x64.

### AVDC (ICLR 2024)
- Source: https://arxiv.org/abs/2310.08576, https://flow-diffusion.github.io/
- No IDM/LAM: GMFlow between generated frames, per-object rigid transforms from first-frame depth and intrinsics, IK execution. Meta-World average 43.1%, **6.7% on `assembly` vs 89.3% `door-close`**; real Franka failures "75% of failures from incorrect video plans, 25% from video discontinuity".
- Failure modes (stated): "when the majority of an object is occluded by the robot arm, our algorithm may lose track"; "small pixel-level errors in tracking small objects would result in large errors in the 3D space". Licence unverified.

### villa-X (2025), UniVLA (RSS 2025), Moto (2024), AdaWorld (ICML 2025)
- villa-X (https://github.com/microsoft/villa-x): codebook 32, proprio forward-dynamics grounding. **The one direct probe**: MLP decodes actions from frozen latents on LIBERO, "maximum L1 error across all action dimensions", reported as a histogram (grounded variant has "more samples with smaller errors"). LIBERO avg 90.1%. Licence unverified.
- UniVLA (https://github.com/OpenDriveLab/UniVLA): task-centric latents on DINOv2 patches, vocab 16 x length 4; ablation LIBERO-Spatial 91.2% vs 68.0% naive; real 68.9% vs LAPA 28.9%. No direct label accuracy.
- Moto (https://arxiv.org/html/2412.04445v2): 8 tokens per frame pair, codebook 128; video-classification proxy 79.7% vs 82.8% with real frames; real FANUC 23.33 -> 60%. Code URL unverified.
- AdaWorld (https://arxiv.org/html/2503.18938v1): continuous 32-dim latent; ground-truth comparison "unavailable by design".

### Not covered
Cosmos (2501.03575) says only that WFMs are fine-tuned on "video-action sequences": label source **not stated**. Genie 3: no primary technical page found. GR-2 (2410.06158), Unified Video Action (2503.00200), IGOR: not read.

**Verdict 4.** Only VPT measures label accuracy (90.6% / R² 0.97, discrete game actions, 1,962 h labelled, non-causal). villa-X gives a histogram, Moto a semantics proxy. DreamGen's IDM-only trajectories are worth about 30 real demos on RoboCasa. The only insertion-specific number, AVDC's 6.7% on `assembly`, is bad. Codebooks are tiny everywhere (8 to 128). Nothing here has been evaluated on fine bimanual insertion.

---

## 5. Human-to-robot retargeting for manipulation

### EgoMimic (2024, arXiv 2410.24221)
- Source: https://arxiv.org/abs/2410.24221, https://github.com/SimarKareer/EgoMimic
- What: "a full-stack framework which scales manipulation via human embodiment data, specifically egocentric human videos paired with 3D hand tracking"; co-trains one policy on human and robot data.
- Input: Project Aria glasses ("egocentric video, 3D hand tracking, and device SLAM"); hands from "Aria Machine Perception Services (MPS)" as "a timestamped CSV of cartesian positions", not HaMeR; SAM2 masks. Paired robot data: 270-430 demos per task on "Two 6-DoF ViperX 300 S arms", robot wears a second Aria.
- Outputs: bimanual joint-space control. Gripper is **not** derived from fingers: "the grasping action is supervised only via the robot joint prediction loss... where the gripper is represented as another joint."
- Results: Laundry (t-shirt folding, bimanual) 88% vs ACT 55%, MimicPlay 50%; Groceries 30% (70% bag grasp); unseen cloth "ACT=25% SR; EgoMimic=85% SR". Human demos 160-1400 per task (60-100 min). No insertion.
- Licence: MIT.
- Failure modes (stated): "failure to correctly align with the toy, failure to grasp the bag's handle, or policy only grabs 1 side of the shirt". *(inference)* Human video cannot teach press-and-release timing because gripper comes only from robot demos; depends on MPS SLAM, which a phone lacks.

### MimicPlay (CoRL 2023)
- Source: https://arxiv.org/abs/2302.12422, https://github.com/j96w/MimicPlay
- What: "learns latent plans from human play data to guide low-level visuomotor control trained on a small number of teleoperated demonstrations."
- Input: "two calibrated cameras to track 3D hand trajectories from human play data", "10 minutes of human play video" per environment, "an off-the-shelf hand detector" (not named) triangulated; 20 robot demos per task (Franka, OSC, 17-20 Hz arm / 2 Hz gripper).
- Results: 14 tasks; Kitchen 0.7/0.6/0.8; Study Desk 0.6/0.7/0.4/0.5; generalisation Easy 0.7, Medium 0.5, Hard 0.2. Single-arm; no insertion.
- Licence: not stated on pages fetched.
- Failure modes (stated): "high-level latent plan is learned from scene-specific human play data"; "limited to table-top settings". *(inference)* Plan is a coarse wrist path; one uncalibrated phone removes the triangulation it relies on.

### Vid2Robot (RSS 2024)
- Source: https://arxiv.org/abs/2403.12943
- What: "an end-to-end video-conditioned policy that takes human videos demonstrating manipulation tasks as input and produces robot actions". No hand extractor; raw frames. "~100k robot videos" and "~10k human videos" paired.
- Outputs: "11-dim action vector", each "discretized into 256 bins".
- Results: overall "52.8% (human prompts), 54.9% (robot prompts)"; Pick 100%, Place upright 12.5%. No bimanual, no insertion. Code: none stated.
- Failure modes (stated): "Grasping failures happen, particularly with small and deformable objects"; gripper self-occlusion; distractors. *(inference)* 256-bin discretisation is far coarser than stud tolerance.

### OKAMI (CoRL 2024 oral)
- Source: https://arxiv.org/abs/2410.11792, https://github.com/UT-Austin-RPL/OKAMI
- What: from one RGB-D human video, "object-aware retargeting, which enables the humanoid robot to mimic the human motions in an RGB-D video while adjusting to different object locations during deployment"; body and hand retargeted separately.
- Input: single RGB-D video, camera "static throughout the recording"; GPT-4V object identification, Grounded-SAM, Cutie tracking, SLAHMR + SMPL-H, HaMeR hands, CoTracker. No teleop.
- Outputs: Fourier GR1 with "two 6-DoF Inspire dexterous hands", "joint position commands at 40Hz"; fingers "retarget the human hand poses to the robot's finger joints" via "inverse kinematics and angle mapping" then dex-retarget.
- Results: abstract "79.2%" over 6 tasks x 12 trials; per-task rows as extracted (75.0 / 75.0 / 83.3 / 83.3 / 75.0 / 58.3) average ~75%, which does not reconcile with 79.2%: **per-task row unverified**. Bagging is bimanual. No insertion.
- Licence: not stated on README fetched.
- Failure modes (stated): "Limited robustness against large variations in object shapes"; RGB-only unsupported. *(inference)* Open-loop trajectory warp has no correction once a brick is a millimetre off; RGB-D and static camera exclude handheld phone.

### HumanPlus (CoRL 2024)
- Source: https://arxiv.org/abs/2406.10454
- What: sim-RL shadowing policy so a humanoid "follow[s] human body and hand motion in real time using only a RGB camera", then BC from egocentric shadowed demos.
- Input: WHAM body pose ("25 fps on an NVIDIA RTX4090"), "HaMeR, a transformer-based hand pose estimator using a single RGB camera" at "10 fps"; AMASS "40 hours"; up to 40 shadowed demos per task.
- Outputs: Unitree H1, "33 degrees of freedom", Inspire RH56DFX hands; "19-dimensional joint position setpoints"; hand mapping joint-to-joint from MANO.
- Results: Fold Clothes (40 demos) 100%, Warehouse 90%, Rearrange 90%, Type "AI" 80%, Wear Shoe and Walk 60%. Bimanual yes; no insertion.
- Code: not stated.
- Failure modes (stated): "Each arm has only 5 DoFs"; cannot handle "large areas of occlusion". *(inference)* A 5-DoF arm on a balancing base cannot hold a stiff sub-millimetre wrist pose for a press.

### AnyTeleop / dex-retargeting (RSS 2023)
- Source: https://arxiv.org/abs/2307.04577, https://github.com/dexsuite/dex-retargeting
- What: vision teleop that "support[s] multiple different arms, hands, realities, and camera configurations within a single system"; the retargeting core is released as dex-retargeting.
- Input: "can consume data from both RGB and RGB-D cameras, and from either single or multiple cameras. Most importantly, it does not require extrinsic calibration." Hands via "MediaPipe, a lightweight, RGB-based hand detection tool that can operate in real-time on a CPU."
- Outputs: joint targets for Allegro, Shadow, Schunk SVH, DLR hands; optimisation "‖αvti−fi(qt)‖2+β‖qt−qt−1‖2" over keypoint vectors with a smoothness term; cross-morphology mapping is manual ("we need to specify the keypoint vectors mapping between the robot and human fingers manually"). Vector, position, and DexPilot optimisers.
- Results: 10 real tasks at 0.6-1.0 (single trials); 3 sim IL tasks 36-79%. Bimanual only as two operators. No insertion.
- Licence: MIT.
- Failure modes (stated): "Loss of tracking during fast human hand motion", "unreliable hand pose when the hand is in self-occlusion". *(inference)* Self-occlusion is the normal case around a held brick; MediaPipe gives no metric depth.

### Phantom (CoRL 2025)
- Source: https://arxiv.org/abs/2503.00779, https://github.com/MarionLepert/phantom
- What: converts human videos into robot demonstrations by extracting actions from HaMeR and rendering a robot over the inpainted arm; zero robot data.
- Input: third-person **RGB-D**, single view; deployment camera "height and angle... similar to that of the camera used to deploy"; "We apply HaMeR [28] to each frame" for "21 keypoints" and "778 vertices".
- Outputs: "ar,t=(𝐩t,𝐑t,gt)": position is the "midpoint between the keypoints at the tips of the thumb and index finger", orientation from "a plane through all the keypoints of the thumb and index finger", width is the "distance between the keypoints corresponding to the fingertips of the thumb and index finger". The only fully specified hand-to-gripper rule in this set.
- Results (Franka, human demos only): Pick/Place Book 313 demos 92%; Stack Cups 268, 72%; Tie Rope 307, 64%; Rotate Box 283, 72%; Sweep Trash 279 (88% grasp). Kinova OOD from 950 demos: 64-84%. "We limit our demonstrations to pinch grasps because our robots are limited to using parallel jaw grippers"; single-arm; no insertion task named in the results read (**flag**).
- Licence: repo exists; licence not stated on pages fetched.
- Failure modes (stated): "Hand pose estimators currently still struggle with occlusions, our method does too"; "Differences in the surface properties of a human fingertip and robot gripper may lead to different object motions"; "We only assess quasi-static tasks". *(inference)* The thumb-index midpoint is exactly what HaMeR estimates worst on a pinched brick, and quasi-static excludes the press.

### Shadow (2025, arXiv 2503.00774)
- Source: https://arxiv.org/abs/2503.00774
- What: robot-to-robot transfer by replacing the robot in images with a composite segmentation mask from "robot proprioception, known kinematics and geometry of the robot, and camera parameters". Not human-to-robot; included as the negative control.
- Results: sim Coffee ("inserting a coffee pod into a narrow slot") 0.88-0.95; real Mug 0.70, Hexagon 0.76, Cups 0.92, Blocks 0.76. No code URL.
- Failure modes (stated): "relies on accurate camera calibration parameters and accurate proprioception"; "primary failure mode is lack of precision, where it sometimes slightly misses a grasp".

### Humanoid Policy ~ Human Policy / PH2D (2025, arXiv 2503.13441)
- Source: https://arxiv.org/abs/2503.13441
- What: PH2D egocentric human dataset plus "Human Action Transformer (HAT)" with "unified state-action spaces for humans and humanoids".
- Input: "Apple Vision Pro + Built-in Camera" (ARKit hands) or "Meta Quest 3 / Apple Vision Pro + ZED Camera"; hand pose from device SDKs. "3.02M frames from 26,824 human demonstrations and ~668k frames from 1,552 robot demonstrations".
- Outputs: Unitree H1 with Inspire hands; "a 54-dimensional vector"; fingertips as "3D x/y/z keypoints" with "a bijective mapping between 10 finger tips of robot dexterous hands and common human hands".
- Results: Cup passing 20/20 ID, 52/60 OOD; Horizontal grasp 8/10, 12/30; Vertical grasp 13/20, 29/70; Pouring 8/10, 8/10. Bimanual platform; no insertion. No code URL.
- *(inference)* Not reachable from third-person phone video; depends on headset hand SDKs.

### EgoZero (2025, arXiv 2505.20290)
- Source: https://arxiv.org/abs/2505.20290, https://egozero-robot.github.io
- What: "a minimal system that learns robust manipulation policies from human demonstrations captured with Project Aria smart glasses, and zero robot data".
- Input: Aria with MPS giving "accurate online 6DoF hand poses, camera intrinsics, and camera extrinsics"; HaMeR 21-keypoint fused with Aria hands; objects triangulated from tracked 2D points; iPhone egocentric at inference.
- Outputs: Franka parallel gripper; actions from "thumb and index coordinates and gripper closure", grasp by thresholding "the Euclidean distance between the thumb and index coordinates".
- Results: 7 tasks, "100 demonstrations per task", 15 trials each; "70% zero-shot success rate"; "Insert book in shelf" 9/15. Single-arm.
- Failure modes (stated): the most useful numbers in the section: Aria and HaMeR introduce **"1-2cm error"**, "preventing high-precision tasks"; hand models "brittle to occlusions, temporally inconsistent"; monocular depth "&gt;5cm error" and "nonviable"; triangulation needs "stationary objects". *(inference)* 1-2 cm is one to two orders above stud clearance; ">5cm" monocular depth is the strongest argument in the corpus against third-person phone capture for metric actions.

### Motion Tracks / MT-pi (2025, arXiv 2501.06994)
- Source: https://arxiv.org/abs/2501.06994
- What: actions as "short-horizon 2D trajectories on an image" shared between human hands and robot end-effectors; "we predict motion tracks from two camera views, recovering 6DoF trajectories via multi-view synthesis."
- Results: "an average success rate of 86.5% across 4 real-world tasks". Task names, demo counts, platform, hand tracker: not stated on the abstract page (abstract-only card).
- *(inference)* 2D tracks discard wrist roll and grip force; two views required.

**Verdict 5.** No paper evaluates bimanual fine assembly: bimanual exists (EgoMimic, OKAMI, HumanPlus, PH2D) and insertion exists (Shadow coffee pod, EgoZero book) but never together, and every insertion is large-clearance. The stated precision floor for the hand-pose-to-end-effector family is EgoZero's 1-2 cm. No primary source supports uncalibrated single-phone capture for metric actions: Phantom and OKAMI need RGB-D, MimicPlay two calibrated cameras, EgoMimic/EgoZero/PH2D headset SLAM; AnyTeleop is calibration-free but MediaPipe-based with stated self-occlusion failure. Phantom's thumb-index rule is the template if the human route is pursued; it is single-arm and quasi-static by construction. Unverified: OKAMI per-task table; Phantom insertion task; MimicPlay/OKAMI/Phantom licences.

---

## 6. Assembly-specific perception and LEGO datasets

### Break and Make / LTRON (ECCV 2022)
- Source: https://arxiv.org/abs/2207.13738, https://github.com/aaronwalsman/ltron
- What: agent inspects and disassembles an unseen LEGO assembly then must "prove its understanding by rebuilding the model from scratch"; LTRON is "An environment for interactive machine learning assembly problems using Lego bricks" on LDraw / Open Model Repository, rendered with splendor-render.
- Data: "a dataset of fan-made LEGO creations" with "over a thousand unique brick shapes"; synthetic renders only; per-brick `shape_ids` and `class_ids` (colours). No real imagery, hands, or calibration. Accuracy: not stated in abstract. Licence: not stated in README text fetched.
- Relevance: the only public brick-level ontology and step-ordered assembly supervision, plus a renderer for labelled synthetic data. *(inference)* Appearance gap to phone video is large; use for pretraining and label schema.

### Learning to Build by Building Your Own Instructions (ECCV 2024)
- Source: https://arxiv.org/abs/2410.01111
- Agent "make[s] its own visual instruction book" while disassembling; new dataset of "procedurally built LEGO vehicles that contain an average of 31 bricks each and require over one hundred steps". Synthetic. Accuracy, code: not stated on the abstract page. *(inference)* The image-per-step formulation maps directly onto keyframes from a human build video.

### MEPNet, "Translating a Visual LEGO Manual to a Machine-Executable Plan" (ECCV 2022)
- Source: https://arxiv.org/abs/2207.12572, https://github.com/Relento/lego_release
- Per manual step the model "reads the manual, locates the components to be added to the current shape, and infers their 3D poses" via "neural 2D keypoint detection modules and 2D-3D projection algorithms". Synthetic train/val released. MIT. Numbers not stated in abstract. Project page failed (TLS). *(inference)* Keypoint-to-brick-pose head is reusable; its clean-manual input assumption is what breaks on phone video.

### IKEA ASM (2021)
- Source: https://arxiv.org/abs/2007.00394, https://ikeaasm.github.io/
- "a three million frame, multi-view, furniture assembly video dataset that includes depth, atomic actions, object segmentation, and human pose"; "371 samples", "3 RGB views", one depth stream, "48 unique assemblers", "33 action classes", part instance masks, "extrinsic camera calibration" provided. No part 6-DoF, no assembly-state labels stated. Dataset CC BY-NC 4.0, code MIT.

### Assembly101 (CVPR 2022)
- Source: https://arxiv.org/abs/2203.14712, https://assembly-101.github.io/
- "4321 videos of people assembling and disassembling 101 'take-apart' toy vehicles"; "simultaneous static (8) and egocentric (4) recordings"; "more than 100K coarse and 1M fine-grained action segments, and 18M 3D hand poses"; mistake detection task. CC BY-NC 4.0; hosted on Hugging Face since May 2026. No part pose or assembly graph. *(inference)* Most plausible pretraining source for the hand branch: small multi-part toys, similar grasps and occlusions.

### MECCANO (2020)
- Source: https://arxiv.org/abs/2010.05654, https://iplab.dmi.unict.it/MECCANO/
- Egocentric toy-motorbike assembly, "20 different subjects", "RGB: 1920x1080 12.00 fps, Depth: 640x480 12.00 fps", gaze at 200 Hz; "64349 active objects annotated with bounding boxes", "89628 hands annotated", "61 action classes". Licence not stated. *(inference)* "Active object / next-active object" is a cheap substitute for brick pose: which brick is placed next without 6-DoF.

### HoloAssist (ICCV 2023)
- Source: https://arxiv.org/abs/2309.17024, https://holoassist.github.io/
- "166 hours of data captured by 350 unique instructor-performer pairs"; headset with "seven synchronized data streams"; mistake detection, intervention prediction, hand forecasting. Least transferable for perception; a template for the guidance layer.

### ASDF, assembly state detection with 6D pose (ISMAR 2024)
- Source: https://arxiv.org/abs/2403.16400, https://github.com/roth-hex-lab/IEEE-ISMAR-2024-ASDF
- YOLOv8 extended to jointly estimate 6D pose and assembly state; "our Pose2State module predicts the final assembly state with precision"; "on the GBOT dataset, we outperform the pure deep learning-based network". Synthetic-training claim from a search snippet only (unverified). *(inference)* The transferable idea is joint pose-and-state coupling, not the discrete classification head, which does not scale to combinatorial brick states.

### Supervised representation learning for assembly state recognition (RA-L 2024)
- Source: https://arxiv.org/abs/2408.11700, https://timschoonbeek.github.io/state_rec
- "intermediate-state informed loss function modification (ISIL)" that "leverages unlabeled transitions between states"; "trained exclusively on images without execution errors... accurately distinguishes between correct states and states with various types of execution errors". Numbers not in abstract. *(inference)* Directly applicable to video, where intermediate frames are free supervision between cheap before/after state labels.

### StableLego (RA-L 2024)
- Source: https://arxiv.org/abs/2402.10711, https://github.com/intelligent-control-lab/StableLego
- Force-balance stability of block assemblies; dataset "50k+ 3D objects with their Lego layouts" with `task_graph.json` ("the Lego assembly design of the object") and `stability_score.npy`. MIT. No images. *(inference)* The task-graph format is a ready-made connectivity target for a perception system, and a physics prior for rejecting implausible brick graphs; same lab as the robot executor.

### A Lightweight and Transferable Design for Robust LEGO Manipulation (2023/2024)
- Source: https://arxiv.org/abs/2309.02354
- EOAT "reduces the problem dimension and allows large industrial robots to manipulate small Lego bricks"; evolution-strategy motion optimisation "to a 100% success rate"; FANUC LR-mate 200id/7L and Yaskawa GP4. No vision component in the abstract; code not stated.

### Eye-in-Finger (2025)
- Source: https://arxiv.org/abs/2503.06848
- "Conventional global or wrist-mounted cameras often suffer from occlusions when either assembling or disassembling from an existing structure"; tool-tip perception "increasing the tolerance of calibration error from 0.4mm to up to 2.0mm for the LEGO manipulation robot". *(inference)* The strongest primary-source argument that a single external phone camera cannot recover stud-level placement at insertion time.

### Gaps
LEGO-Net (2301.09629) is room rearrangement, not bricks. No verified "6-DoF pose of LEGO bricks" paper found. 2026 titles seen but not fetched in this section: Brick-Composer (2606.05445), BrickNet, "A Synthetic-Driven Vision System for Assembly Step Recognition". BrickSim and BrickCraft are covered in Section 7.

**Verdict 6.** Label schema exists (LTRON brick ids and step order; StableLego task graph). Hand-plus-action supervision at small-part scale exists (Assembly101, 18M hand poses). No dataset pairs real video with brick-level poses or connectivity; that label would have to be created here.

---

## 7. The robot-executed route

When the demonstrator is the lab's own arm, the label problem changes shape.

### What the lab already has (primary sources)
- Liu, Sun, Liu, "Robotic LEGO Assembly and Disassembly from Human Demonstration" (2023; https://arxiv.org/abs/2305.15667): "A depth camera (e.g., Realsense) is used to capture the evolving workspace"; "a pre-trained pixel-to-grid model generates the task information"; learned representation is a temporal task graph with nodes "g_i={id_i, p^s_i, ω^s_i, p^a_i, ω^a_i}" (brick type, storage pose, assembly pose); FANUC LR-mate 200id/7L with a custom EOT; digital twin verification. Hand tracking, success rates, failure modes: **not stated**. This is the lab's existing human-demo-to-plan path, and it works at the level of plate-grid positions, not continuous 6-DoF.
- BrickCraft (2026; https://arxiv.org/html/2605.07605): Kinova Gen3 + Robotiq 2F-85; "692 demonstration trajectories for 3 assembly skills across more than 60 distinct assembly configurations"; demos collected by gamepad waypoints replayed with a Cartesian planner "while synchronously recording multimodal observations"; two RealSense cameras (wrist and table side), 256x256 inputs; Diffusion Policy with "a CNN-based U-Net"; the "situated manual" is a Blender render of the intended state, aligned to the observation with "the enhanced correlation coefficient (ECC) algorithm... to estimate an optimal affine transformation matrix"; "86.25% across a total of 240 trials" for primitive skills over 8 seen + 8 unseen structures; long-horizon Pyramid/Stairs/House/Castle with "6 independent trials per structure". Named failures: "Misalignment: downward pressure applied before perfect alignment", "Collision", "Structural Deformation: partially supported configurations". Code: not stated.
- BrickSim (2026; https://arxiv.org/abs/2603.16853): "a compact force-based mechanics model for snap-fit connections" solving internal force distribution as a convex QP; handles "assembly, disassembly, and structural collapse"; demonstrates "robotic construction of brick assemblies"; code at intelligent-control-lab/BrickSim; datasets with pose labels **not stated**.
- Eye-in-Finger (2025) and the 2023 EOAT paper: insertion tolerance 0.4-2.0 mm, 100% placement success after ES optimisation.

### How labels arise on the robot route
- **Real robot videos**: the action label is the recorded joint or end-effector trajectory plus gripper state, exact and synchronous with the frames; no hand pose, no IDM. BrickCraft's recorder is this.
- **Generated robot videos** (DreamGen recipe): fine-tune a video model on the robot's own recordings, generate new placements, then label with an IDM trained on the same recordings. DreamGen's IDM is "conditioned on two image frames and is trained to predict action chunks between the image frames"; it was trained on 2,884 GR1 trajectories and, on RoboCasa, on the same 1,200 demos as the video model. The IDM's output space is the robot's own (joint or EE chunk), so labels are directly executable. EVA reports the ceiling: an IDM trained on 250 real bimanual demos reaches "89.52% average execution success rate" **on real videos**, and generated videos need alignment to stop "kinematic violations".
- **Brick state** still has to be perceived, on both real and generated video, if the goal is a state-labelled dataset. On the robot route the plate is fixed and calibrated, the camera is fixed, and the brick is either in the gripper (pose known from kinematics) or on the plate (pose on a known grid). BrickCraft and the 2023 paper both exploit this: grid cells, not 6-DoF. *(inference)* This turns brick-state estimation into a per-cell occupancy and colour classification against a rendered expectation, which is the ECC-aligned situated manual already in BrickCraft.

### What generated robot videos are good for on this route
- DreamGen's evidence: co-training real + neural trajectories lifts RoboCasa from 49.59% to 57.61% at 300 real demos, and enables "22 new behaviors" from one pick-and-place teleop set. The gain is in behaviour and environment diversity, not precision.
- *(inference)* For LEGO the useful "new behaviours" are new structures and new brick sequences, not new motor skills; the motor skill (attach, press, twist) is already at 100% and 86% and lives in the EOAT plus force controller, not in the video. So generated videos plausibly help the planning / state-recognition layers (which brick, where, in what order) and are unlikely to improve the insertion primitive.

---

## 8. Human hands vs robot executor: comparison

| Axis | Human hands, phone video | Robot executor |
|---|---|---|
| Action label source | Hand pose regressor (MANO) then retargeting; 5-8 mm PA-MPJPE on benchmarks, occluded PCK@0.1 59-67 (Hamba); no assembly-scale numbers | Proprioception on real runs (exact); IDM on generated runs (DreamGen: no accuracy number reported; EVA: 89.5% executable on real bimanual video) |
| Contact / grasp events | Must be inferred from hand-object proximity; no MANO method here models contact with a brick (HOLD/EasyHOI target large objects, 10 h or 2 min per clip) | Gripper state and force/torque are logged; BrickCraft names the contact failure modes explicitly |
| Brick state | 6-DoF from phone: no supported method for 8 mm textureless symmetric parts; metric scale must be injected via known CAD; insertion occluded by fingers | Grid-cell state on a calibrated plate against a rendered expectation (BrickCraft ECC alignment); brick in gripper known from kinematics |
| Data cost per demo | Seconds of phone video, no rig; labelling is the cost (Dyn-HaMR ~8-10 min per 128 frames on A100, plus a brick tracker that does not exist yet) | Minutes per demo with gamepad waypoints and replay (BrickCraft: 692 demos for 3 skills); labels free |
| Embodiment gap | Human hand to 2F-85 or EOAT via Phantom's thumb-index rule; stated pipeline error 1-2 cm (EgoZero); press-fit and twist have no human analogue with the EOAT; no Section 5 paper evaluates bimanual insertion | None |
| What generated video adds | Diversity of structures, hands, viewpoints; but labels inherit hand-pose plus retargeting error on top of video-model physics error | Diversity of structures and sequences with executable labels; DreamGen +8 points on RoboCasa co-training; insertion precision does not come from video |
| Licence | MANO non-commercial on every hand method; WiLoR/HaWoR ND | Robot data owned by the lab; video model licence only (WAN2.1 Apache-2.0 per earlier notes in this directory) |
| Verdict | Feasible for *which brick, where, in what order* (assembly state and sequence); not for insertion actions | Feasible end to end today; the only open perception problem is brick state, and the lab already has a working approximation |

*(inference)* If the goal is a labelled dataset for training a world model or policy, the robot route gives exact labels now. The human route is worth pursuing only for what the robot cannot produce: variety of human strategies and natural mistakes (Assembly101's framing), consumed as a plan-level signal (task graph), not as actions.

---

## 9. Proposed labelling pipeline for a phone video of LEGO assembly (human hands)

Steps, tools, expected accuracy, where it breaks. Accuracy expectations are the benchmark numbers above; none is measured on LEGO.

1. **Capture and intrinsics.** Phone on a tripod, 4K at 30-60 fps, a printed ChArUco board on the table in the first seconds. Intrinsics from OpenCV; scale from the board and from the baseplate's known 8 mm pitch. Breaks: rolling shutter on fast hand motion; auto-exposure and autofocus (lock both).
2. **Hand detection and handedness, every frame.** MediaPipe Hand Landmarker (Apache 2.0) or WiLoR's detector (NC-ND) for boxes and side labels; gate frames with two visible hands. Expected: near-perfect on unoccluded frames; misses when a hand is behind the model.
3. **Bimanual MANO in a world frame.** Dyn-HaMR (MIT, ~8-10 min per 128 frames on A100) for two-hand world-space MANO with occlusion infilling; HaMeR-level per-frame accuracy (PA-MPJPE 6-8 mm) on visible joints, Hamba-style 25-point PCK drop on occluded fingertips. Breaks: fingertips inside a pinch, SLAM drift on a bare table (mitigate with the board and static clutter).
4. **Brick instance segmentation.** SAM2 with colour priors and known brick catalogue; instance IDs tracked through occlusion. Breaks: adjacent same-colour bricks merge (SAM-6D's IC-BIN 58.8 AR is the analogue).
5. **Brick pose where needed.** MegaPose-RGB (Apache 2.0) with the exact LDraw mesh in mm and the intrinsics from step 1, on frames where the brick is in flight or resting; symmetry leaves yaw ambiguous by 90 or 180 degrees; resolve with the plate grid once placed. Skip pose during insertion (occluded; Eye-in-Finger).
6. **Assembly state as a brick graph.** Per placement, snap the placed brick to the plate lattice and emit a StableLego-style `task_graph.json` node (brick id, grid position, orientation), validated by StableLego stability and by BrickSim if forces matter. Confirm each step with the before/after image pair (ISIL-style transition supervision; MEPNet keypoint head as a candidate estimator). Expected: this discrete label is the one the pipeline can get right; treat it as the primary output.
7. **Contact and grasp events.** Threshold fingertip-to-brick distance from steps 3 and 5 plus segmentation overlap; label grasp start, release, and press events. Breaks: 5-8 mm joint error is the same order as brick features, so use temporal consistency (velocity zero-crossings) rather than distance alone.
8. **Action sequence for a robot.** Emit the task graph plus per-step approach and release poses in the plate frame; hand the primitive execution to the EOAT/force controller (100% placement, BrickCraft 86% skills). Do not attempt to retarget finger trajectories to the EOAT.
9. **Where it will break.** Insertion frames (occluded, sub-millimetre); identical bricks; same-colour adjacency; any commercial use (MANO). Expect the continuous labels to be usable for coarse hand trajectory only and the discrete labels (graph, events, sequence) to be the deliverable.

## 10. Minimum capture setup that makes this tractable

- **Cameras:** two phones on tripods, one oblique third-person, one near-overhead; a third egocentric view (chest-mounted phone or Aria) only if hand pose is a first-class output. One phone is enough for the discrete pipeline; two are needed for occlusion at insertion and for triangulating brick pose.
- **Calibration:** ChArUco board visible at the start of every take for intrinsics and extrinsics; the baseplate itself is the metric reference (8 mm pitch).
- **Resolution and fps:** 4K at 30 fps minimum; 60 fps preferred for press events; locked exposure and focus.
- **Markers:** none on hands (would break the hand regressors' training distribution); optional small AprilTags on the plate corners for pose; no markers on bricks.
- **Robot route:** the existing BrickCraft rig (two RealSense, calibrated plate, gamepad teleop, synchronised recording) already satisfies all of this and gives exact actions.

## 11. Open questions

1. Does any MANO regressor hold 5-8 mm on a hand pinching a 2x2 brick? No source reports it. One afternoon pilot on lab footage with HaMeR, WiLoR, Dyn-HaMR decides.
2. What is MegaPose-RGB's rotation accuracy on an 8 mm symmetric brick with a mm mesh? Not in any source. Needs a render-to-real test with the plate grid as ground truth.
3. Is DreamGen-style IDM labelling accurate on the lab's robot? DreamGen reports no label accuracy; the test is to hold out real trajectories, run the IDM on their videos, and report joint-space error and executability (EVA's 89.5% is the only comparable number).
4. What do generated videos buy on the robot route beyond structure diversity? DreamGen's +8 points is on kitchen pick-and-place; nothing exists for insertion.
5. Licensing: every hand path is non-commercial (MANO). If the work leaves the lab, the hand branch needs replacing (MediaPipe is Apache 2.0 but joint-only).
6. Unverified in this pass: HaPTIC's GA-MPJPE units and release; DexYCB counts; ARCTIC licence; MegaPose per-dataset AR; MonST3R/CUT3R/DreamGen/AVDC/villa-X repo licences; ASDF synthetic-training claim; "Hands-on-Everything" (not found).
