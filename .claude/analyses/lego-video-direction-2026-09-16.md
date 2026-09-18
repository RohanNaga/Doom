# LEGO assembly video generation as synthetic demonstration data (direction opened Sep 16, 2026)

Origin: Rohan and Changliu, Sep 16. Fall goal 3 (world models into LEGO manipulation). Not a September project; the Doom paper owns compute until Sep 30.

## The idea
Fine-tune a video generation model on demonstrations of LEGO assembly so that, from a first frame plus a text instruction (and later a target-state image or a trajectory), it generates plausible assembly videos; use them as synthetic demonstrations for policy learning or for scaling data. Demonstrator undecided: human hands on phone video (example clip IMG_3110.MOV: 46 s, 1280x720, 30 fps, fixed top-down camera, white table, two hands stacking four coloured bricks; bricks are a few percent of the frame and fingers occlude the studs at every snap) or the lab's robot arm with proprioceptive actions (the DreamGen / GR00T-Dreams setting).

## Precedent
DreamGen (NVIDIA 2025): Cosmos post-trained on small robot demo sets, generates "neural trajectories" for unseen tasks, labels them with an inverse-dynamics or latent-action model, reports policy generalization gains. Also Gen2Act, Dreamitate, UniPi/VLP, AVDC, VPP, UVA, LAPA. Evidence cards being written to `.claude/analyses/lit/lego-*.md` (generated video as data; post-training recipes and memory on 4x A6000; action and brick-state extraction, human vs robot executor; datasets and evaluation protocols).

## Claude's read (Sep 16)
- The video model is the easy half: a few hundred clips and LoRA on a 2B video model produce convincing hands and bricks.
- The hard half is after the pixels: actions are not in a human-hand video (fingers hide the decisive moment), LEGO connectivity is discrete on a stud lattice and pixel metrics miss half-stud errors, and studs are a few pixels at 480p.
- A robot executor simplifies labelling (actions from proprioception, an IDM in robot action space) but not brick-state correctness or the physical plausibility of generated frames.
- Contribution to aim at: a labeller and an evaluation that grade generated assembly videos against the stud lattice (instruction followed, valid assembly state), not "fine-tune Cosmos on LEGO".
- Capture decides tractability: two calibrated cameras (top and oblique), distinct brick colours, known build sequences per clip, 50 to 200 clips; label the real clips first (a LEGO-with-hands dataset is publishable on its own).
- Order: capture protocol and labeller, then fine-tune (Cosmos-Predict2 or a Wan-family model, LoRA), then grade generated videos with the same labeller, only then downstream policy tests.

## Open decisions
Human vs robot demonstrator; capture setup; which video model; what downstream test counts as success. To be discussed with Changliu after the evidence cards land.
