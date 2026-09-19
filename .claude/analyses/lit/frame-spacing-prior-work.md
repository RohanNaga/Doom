# Frame spacing in action-conditioned world models

**Question.** At what temporal spacing do action-conditioned world models predict, relative to the
simulator's native tick and to the agent's action-repeat?

**Written** 2026-09-19 by an evidence-gathering agent for R1.3. Every factual cell below carries a
quote from the paper text or a file:line from the repository. Cells marked "not stated" were
searched for and not found in the source. Cells marked "inferred" show the arithmetic. Cells marked
**VERIFY** could not be confirmed from a primary source.

Method: arXiv HTML (`arxiv.org/html/<id>`) downloaded and grepped locally, plus repository source
fetched from `raw.githubusercontent.com`. No fact below is from memory of a paper.

---

## 0. Definitions used in every row

Three different rates get conflated in this literature; the table keeps them apart.

- **Native sim rate** — how often the engine advances world state and renders. Doom: 35 tics/s.
- **Action repeat / frame skip** — how many native ticks the agent holds one action for. This is a
  property of the *data-collection policy*, not of the model.
- **Model frame spacing** — the game time between two consecutive frames the model predicts. This
  is the quantity R1.3 is about. It equals (native tick) x (training stride), and the training
  stride is *independent* of the action repeat: you can store every tic while the agent acts every
  4th tic.
- **Playback / generation fps** — how fast the model emits frames in wall-clock time. If it differs
  from the model frame spacing, the simulation runs in slow motion or fast forward.

---

## 1. Summary table

| Paper | Native sim rate | Action repeat during collection | Model frame spacing | Context (frames / game seconds) | Playback fps | = game rate? |
|---|---|---|---|---|---|---|
| GameNGen (2408.14837) | ViZDoom 35 tics/s | **4 tics** (stated) | **1 tic, 28.6 ms** (best-supported; paper never states it directly) | 64 frames / **1.83 s** inferred at 35 Hz; paper says "a little over 3 seconds" | 20 FPS (4-step), 50 FPS distilled | **No** — and the paper's own second-conversions assume 20 fps, contradicting its 35 FPS data |
| gameNgen-repro (arnaudstiegler) | 35 tics/s | 4 tics (`ACTION_REPEAT = 4`) | **1 tic, 28.6 ms** (verified in code; `skipframe-4` datasets store *every* tic) | 9 + 1 frames / 0.286 s | not real-time | n/a |
| MultiGen (2603.06679) | ViZDoom 35 tics/s | not stated | not stated | ablates L ∈ {2,4,8,16,32} frames / seconds not stated | ~20 FPS/player | not stated |
| PlayGen (2412.00887) | Doom + SMB | not stated | not stated | not stated | 20 FPS | not stated |
| DIAMOND (2405.12399) Atari | 60 Hz | **4** (Frameskip 4) | 4 raw frames, 66.7 ms | 4 frames / 0.27 s inferred | n/a (training) | n/a |
| DIAMOND CS:GO | server tick not stated | n/a (human play) | **62.5 ms** (16 Hz data) | 4 frames / 0.25 s inferred | **10 Hz** | **No** — 10 Hz playback of 16 Hz data |
| Ha & Schmidhuber (1803.10122) | ViZDoom 35 tics/s | **none** (inferred) | 1 tic, 28.6 ms (inferred) | RNN, unbounded | n/a | n/a |
| SimPLe (1903.00374) | 60 Hz | **4** ("every action is repeated 4 times") | 4 raw frames, 66.7 ms | 4 stacked frames / 0.27 s inferred | n/a | n/a |
| IRIS (2209.00588) | 60 Hz | not stated in text (Atari 100k default 4) | 66.7 ms (inferred) | 20 burn-in frames | n/a | n/a |
| STORM (2310.09615) | 60 Hz | **4** (states the benchmark convention) | 66.7 ms | not extracted | n/a | n/a |
| DreamerV3 (2301.04104) | varies | not stated in Nature text **VERIFY** | not stated | not stated | n/a | n/a |
| TD-MPC2 (2310.16828) | varies | **2** (DMControl/Meta-World/ManiSkill2), **1** (MyoSuite) | = action repeat | n/a (latent) | n/a | n/a |
| Genie (2402.15391) | internet video | **"no action repeats"** on CoinRun | 100 ms (10 FPS data) | 16 frames / 1.6 s | ~1 FPS | **No** |
| MineWorld (2504.08388) | Minecraft 20 ticks/s | not stated | not stated (VPT is 20 Hz; not stated *in this paper*) | 16 state-action pairs / not stated | 4–7 FPS | not stated |
| Oasis (Decart/Etched) | Minecraft 20 ticks/s | not stated | not stated | not stated | 20 FPS | not stated |
| WHAM / Muse (Nature 2025) | Bleeding Edge, not stated | not stated | **100 ms (10 Hz)** | **10 (obs, action) pairs / 1.0 s** inferred | ~1 img/s (WHAM-1.6B); 10+ fps (WHAMM) | **No** |
| GameFactory (2501.08325) | Minecraft | randomized per atomic action | 1 latent = **4 video frames** (r = 4) | window w = 3 latents = 12 actions | not stated | n/a |
| Matrix-Game 2.0 (2508.13009) | UE / GTA5 | not stated | not stated | not stated | 25 FPS | not stated |
| The Matrix (2412.03568) | AAA games | n/a | not stated (data is 60 FPS) | not stated | 8–16 FPS | **No** (60 FPS data) |
| GameGen-X (2411.00769) | AAA game footage | n/a | 24 fps clips | up to 480 frames | 20 FPS | ~no |
| GAIA-1 (2309.17080) | camera 25 Hz | n/a | **160 ms** — "temporally subsample videos from 25Hz to 6.25Hz" | not stated | n/a | separate temporal super-resolution decoder |
| GAIA-2 (2503.20523) | cameras 20/25/30 Hz | n/a | native rate, **timestamp-conditioned** | 24 frames → 3 latents | n/a | n/a |
| Vista (2405.17398) | datasets 2–12 Hz | n/a | **100 ms (10 Hz)** | not extracted | n/a | n/a |
| NWM (2412.03572) | varies | n/a | **variable: time shift k, up to ±16 s** | 4 context frames | 2–10 Hz est. | deliberately decoupled |
| Vid2World (2505.14357) | RECON 4 fps | n/a | 250 ms (RECON native), also runs baselines at **1 fps** | 16 frames | n/a | n/a |
| V-JEPA 2-AC (2506.09985) | Droid | n/a | **250 ms (4 fps)** | 16 frames / 4 s | n/a | n/a |

---

## 2. Per-paper detail with quotes

### GameNGen — arXiv 2408.14837 (v2 HTML)
https://arxiv.org/abs/2408.14837 · https://arxiv.org/html/2408.14837v2

- **Environment and native rate.** "We train the agent to play the game using the ViZDoom
  environment (Wydmuch et al., 2019)." (§4.1). ViZDoom's default tick rate is 35/s: "In **ASYNC**
  modes the game progress with constant speed (default 35 tics per second, this can be set)"
  (ViZDoom docs, `docs/configurationFiles.md` line 15,
  https://github.com/Farama-Foundation/ViZDoom/blob/master/docs/configurationFiles.md).
- **Action repeat during collection.** Appendix A.5 (Reward Function), final paragraph:
  > "Further, to encourage the agent to simulate smooth human play, we apply each agent action for
  > 4 frames and additionally artificially increase the probability of repeating the previous
  > action."

  This is the *only* statement of 4 in the paper, and it is about the agent's behaviour.
- **Which frames the model predicts.** The paper never states a training stride. The strongest
  evidence is Appendix A.6 (Reducing Inference Steps):
  > "We evaluated the performance of a GameNGen model with varying amounts of sampling steps when
  > generating 2048 frames using teacher-forced trajectories on **35FPS data (the maximal sampling
  > rate allowed by ViZDoom**, lower than the maximal rate our model achieves with distillation, see
  > below)."

  Calling the trajectories "35FPS data" and "the maximal sampling rate allowed by ViZDoom" only
  makes sense if consecutive stored frames are consecutive tics. Had they kept one frame per
  decision the data would be 8.75 FPS. **Model frame spacing = 1 tic = 28.6 ms** is the
  best-supported reading, but it is *inferred*, not stated.

  Supporting arithmetic (inferred): §4.1 "We perform a total of 50M environment steps", §4.2 "we use
  a random subset of **70M examples**". 70M > 50M, so the example count exceeds the decision count;
  consistent with ~4 frames stored per decision, inconsistent with one frame per decision unless
  evaluation contributed >20M decisions.
- **Action feeding.** §4.2: "We use a context length of 64 (i.e. the model is provided its own last
  64 predictions as well as **the last 64 actions**)." One action token per frame, so a held action
  appears on 4 consecutive frames. §3.2: "to condition on actions (i.e. key presses), we simply
  learn an embedding A_emb from each action into a single token and replace the cross attention from
  the text into this encoded actions sequence." The paper never discusses that the world changes
  during the 4 held tics.
- **Context in game seconds — the paper is internally inconsistent.** Every seconds figure in the
  paper is computed at 20 fps, not 35 Hz:
  - §5.1: "simulations of length **16 frames (0.8 seconds)** and **32 frames (1.6 seconds)**".
    16/0.8 = 20, 32/1.6 = 20. At 35 Hz these would be 0.46 s and 0.91 s.
  - §5.2.1: "even with our maximal context length, the model only has access to **a little over 3
    seconds of history**". 64/3.2 = 20. At 35 Hz, 64 frames = **1.83 s**.
  - §7: repeats "The model only has access to a little over 3 seconds of history".

  So either the training frames are ~20 Hz apart (impossible to obtain exactly from ViZDoom's 35 Hz
  tick, and contradicted by A.6's "35FPS data"), or the authors converted frame counts to seconds
  using the *playback* rate. The second is far more likely, and it means **every "seconds" figure in
  GameNGen is 1.75x too long in game time**.
- **Playback fps.** §3.3.2: "Using just 4 denoising steps leads to a total U-Net cost of 40ms (and
  total inference cost of 50ms, including the auto encoder) or **20 frames per second**." This is a
  latency budget, i.e. generation speed. Consequence (inferred): a human playing at 20 FPS on 35 Hz
  spaced data experiences Doom at 20/35 = 0.57x real time.
- **Ablation on frame rate / spacing / action repeat.** **None.** The only temporal ablation is
  context *length* in frames (§5.2.1, Table 2, N ∈ {1,2,4,8,16,32,64}): "we observe that while the
  improvement is large at first (e.g. between 1 and 2 frames), we quickly approach an asymptote and
  further increasing the context size provides only small improvements in quality." Spacing is never
  varied.
- **Persistence / copy-frame baseline.** **Not reported.** The headline is §5.1: "When evaluated
  over a random holdout of 2048 trajectories taken in 5 different levels, our model achieves a PSNR
  of 29.43 and an LPIPS of 0.249", under "the teacher-forcing setup ... where we sample an initial
  state and **predict a single frame** based on a trajectory of ground-truth past observations".
  **This 29.43 dB is a single next-frame prediction at (most likely) 28.6 ms spacing with 64 ground
  truth context frames.** There is no copy-last-frame number to normalise it against.

### gameNgen-repro — github.com/arnaudstiegler/gameNgen-repro (decisive)

This settles what `skipframe-4` means, and our note about it is wrong.

- `ViZDoomPPO/load_model_generate_dataset.py:47` — `ACTION_REPEAT = 4`, commented
  `# To replicate frame_skip in the environment`.
- `ViZDoomPPO/load_model_generate_dataset.py:132-133` — in `make_pkls_dataset`, the function that
  writes the released datasets:
  ```python
  # Set frame_skip to 1 to capture all frames
  eval_env_args['frame_skip'] = 1
  ```
- `ViZDoomPPO/load_model_generate_dataset.py:150-165` — the collection loop steps the env **one tic
  at a time**, re-picks an action only every 4th tic, and appends a frame **on every tic**:
  ```python
  while not done:
      if frame_counter % ACTION_REPEAT == 0:
          current_action, _ = agent.predict(obs)
      obs, _, done, _ = env.step(current_action)
      screen = env.venv.envs[0].game.get_state().screen_buffer
      frames.append(screen)
      actions.append(int(current_action.item()))
  ```
  So the HF datasets `vizdoom-{5,500}-episodes-skipframe-4-lvl5` store **every rendered tic** with
  the held action repeated on each of 4 consecutive rows. **"skipframe-4" names the agent's action
  repeat, not a frame stride. No frames are dropped.**
- `dataset.py:61` — the training dataloader takes **consecutive rows**:
  `return self.dataset[idx-BUFFER_SIZE:idx+1]`, with `BUFFER_SIZE = 9` (`config_sd.py:3`). So one
  training example is 9 context tics + 1 target tic = **10 consecutive tics = 0.286 s of game
  time**, and the target is **28.6 ms** after the last context frame.
- `ViZDoomPPO/common/envs.py:47,80` — the PPO training env does use `frame_skip: int = 4` via
  `self.game.make_action(self.possible_actions[action], self.frame_skip)`; that is the agent, not
  the dataset.
- Persistence baseline: **not reported**.
- The second reproduction, github.com/Masao-Taketani/GameNGen, was **not inspected** — **VERIFY**.

### MultiGen — arXiv 2603.06679
https://arxiv.org/abs/2603.06679

Directly comparable (Doom, diffusion game engine, procedural maps), and **it does not state its
frame spacing or action repeat anywhere**.

- Data: "we generate gameplay sequences on 100 procedurally generated maps with randomized
  structure ... We then deploy a pre-trained Doom agent to explore the resulting maps, collecting
  over 10 million gameplay frames" (§4.1); "We build on ViZDoom and collect simulated Doom
  deathmatch sequences in which one pre-trained agent plays against four identical agents" (§5.2).
- Context: "S_t = (M, p_t, o_{t-L+1:t}) contains the static map M, the current pose p_t, and an
  L-frame context window" (§3); ablation "varying L ∈ 2, 4, 8, 16, 32 while keeping all other
  settings fixed ... increasing the context length consistently improves fidelity" (§6). Context in
  *seconds* is never given, because the spacing is never given.
- Playback: "the full system runs at approximately **20 FPS** using a single NVIDIA A100 per player"
  (§5). Whether that equals game time: not stated.
- Persistence baseline: not reported.

### PlayGen — arXiv 2412.00887
https://arxiv.org/abs/2412.00887

- Doom data collection (Appendix A.2) states episode length and action *preference* but no rate:
  "We also collect **200 timesteps per sample** in Doom ... at the start of each episode ... we
  randomly select an action as the preferred action for the current episode. This preferred action
  will have a higher probability of being chosen during the episode." Native rate, action repeat and
  stride: **not stated**.
- Playback: "our game-generative model is able to simulate both video games Super Mario Bros and
  Doom at **20 FPS** on NVIDIA RTX 2060" (§4). Doom PSNR "23.81" (§6) — again with no copy-frame
  baseline.
- Note: PlayGen's related work asserts GameNGen "manages real-time interaction at 20 frames per
  second (FPS) within the Doom game", i.e. it also reads 20 FPS as generation speed.

### DIAMOND — arXiv 2405.12399
https://arxiv.org/abs/2405.12399

- Atari: hyperparameter table, "Frameskip **4**", "Max noop 30", "Termination on life loss True".
  World model input: "The diffusion model D_θ is a standard U-Net 2D, conditioned on the **last 4
  frames and actions** ... We use frame stacking for observation conditioning" (Appendix E). So the
  context is 4 model frames = 16 raw Atari frames ≈ **0.27 s** at 60 Hz (inferred). The paper does
  not say whether the 4 skipped frames are max-pooled — **VERIFY** (the brief's guess of
  max-pooling is not in the text I read).
- CS:GO: "We use the Online dataset of 5.5M frames (95 hours) of online human gameplay **captured at
  16Hz** from the map Dust II by Pearce and Zhu, (2022)" (§4.3). The **64-tick server rate is not
  mentioned anywhere** — **VERIFY** against Pearce & Zhu (2022) if needed.
- Playback ≠ game rate: "This enables a reasonable tradeoff between visual quality and inference
  cost, with the model **running at 10Hz** on an RTX 3090" (§4.3) — 10 Hz playback of 16 Hz data.
- Also: the comma.ai driving experiment "We **downsample the dataset to 10Hz**" (Appendix M.1).
- Ablation on spacing: none. Persistence baseline: not reported.

### Ha & Schmidhuber, World Models — arXiv 1803.10122
https://arxiv.org/abs/1803.10122

- VizDoom take-cover: "Each rollout of the environment runs for a maximum of **2100 time steps (~60
  seconds)**, and the task is considered solved if the average survival time over 100 consecutive
  rollouts is greater than **750 time steps (~20 seconds)**" (§ Doom experiment).
- **Inferred:** 2100 / 60 = 35 time steps per second, and 750 / 20 = 37.5. So one model time step =
  one 35 Hz tic, i.e. **no action repeat and no stride**; the MDN-RNN predicts every rendered frame.
  The words "frame skip" and "action repeat" do not appear in the paper.
- Persistence baseline: not reported.

### SimPLe — arXiv 1903.00374
https://arxiv.org/abs/1903.00374

> "We apply a standard pre-processing for Atari games: a **frame skip equal to 4, that is every
> action is repeated 4 times**. The frames are down-scaled by a factor of 2." (§ Experiments)

> "altogether 6400 · 16 = 102,400 interactions with the Atari environment are used during training.
> This is equivalent to **409,600 frames** from the Atari game (114 minutes at 60 FPS)."

Here action repeat and model spacing are the same thing: the model sees one frame per decision, 4
raw frames = 66.7 ms apart. Input is "four stacked frames (as well as the action selected by the
agent)" (Figure 2 caption) — so 4 model frames ≈ 0.27 s of context.

### STORM — arXiv 2310.09615 (cited for the benchmark convention)
https://arxiv.org/abs/2310.09615

> "The 100k sample constraint corresponds to **400k actual game frames, taking into account frame
> skipping (4 frames skipped) and repeated actions within those frames**. This constraint
> corresponds to approximately 1.85 hours of real-time gameplay." (§ Experiments)

This is the clearest statement in the RL literature that *the model's timestep is the decision, and
the skipped frames are discarded entirely*.

### IRIS — arXiv 2209.00588
https://arxiv.org/abs/2209.00588

Frame skip is **not stated in the paper text** (Atari 100k's default of 4 applies). Burn-in context:
"Before starting the imagination procedure from a given frame, we burn-in the **20 previous
frames**" (Appendix). Spacing ablation: none.

### DreamerV3 — arXiv 2301.04104 (Nature version)
https://arxiv.org/abs/2301.04104

The word "repeat" does not occur in the v2 HTML at all. Atari setup is described only as "a budget
of 200M frames" with "the sticky action simulator setting". **Action repeat per domain: VERIFY
against the released config files (`dreamerv3/configs.yaml`), not the paper.**

### TD-MPC2 — arXiv 2310.16828
https://arxiv.org/abs/2310.16828

Table 6 ("Environment details ... We list the episode length and **action repeat** used for each
task domain"): **DMControl 2, Meta-World 2, ManiSkill2 2, MyoSuite 1**, giving "Effective length"
500/100/100/100. Action repeat is a per-domain constant chosen for control, not ablated for model
quality.

### Genie — arXiv 2402.15391
https://arxiv.org/abs/2402.15391

- Data: "yielding 55M 16s video clips at **10FPS**, with 160x90 resolution" (§4); "We then split
  each video into 16s clips at 10 FPS, which corresponds to **160 frames per clip**" (Appendix B.1).
- Context: "we are still limited to **16 frames of memory**" (§7) = **1.6 s** at 10 FPS (inferred).
- Playback: "Genie currently operates around **1FPS**" (§7) — 10x slower than game time.
- **Explicit action-repeat statement**, for the CoinRun analysis: "Using the 'hard' mode, we collect
  data using a random policy **with no action repeats**." (Appendix) — one of the very few papers
  that says so.
- Persistence baseline: not reported.

### MineWorld — arXiv 2504.08388
https://arxiv.org/abs/2504.08388

- Data: "We utilize the VPT dataset (Baker et al., 2022) ... where each frame is accompanied with
  the keyboard and mouse actions taken at the same time" — **one action per frame**. "We split long
  videos into short clips with 16 frames considering the maximum context length of the model"
  (§3.1). The frame rate of VPT is **not stated in this paper** (VPT records Minecraft at 20 Hz —
  **VERIFY** in 2206.11795).
- Playback target is argued from human action rate, not from game time: "to develop a world model
  that achieves real-time interactions with users, it should generate more than 2 Frames Per Second
  (FPS) to keep up with amateur game players and 5 FPS for professional ones" (§3). Achieved: "4 to
  7 frames per second" (abstract). **This is an explicit argument that the model's frame rate should
  track the player's decision rate (APM), not the renderer's** — the closest thing in the literature
  to a principled defence of one-frame-per-decision.
- Context in seconds: "The maximum input length of the model is set to 5.5k tokens, corresponding to
  16 state-action pairs" — seconds not given.

### Oasis — Decart/Etched
https://oasis-model.github.io/

- "Oasis generates real-time output in **20 frames per second**." Minecraft's native tick is 20/s,
  so 20 FPS *may* coincide with game time, but the page does not say so, and the open-source
  `open-oasis` README was not reachable at the URL tried. Data rate, action repeat, context length:
  **not stated** on the project page. **VERIFY**.

### WHAM / Muse — Nature 2025, model card microsoft/wham
https://huggingface.co/microsoft/wham · https://www.nature.com/articles/s41586-025-08600-3

- "Dataset size: The model was trained on data from approximately 500,000 Bleeding Edge games from
  all seven game maps (over **1 billion observation, action pairs 10Hz**, equivalent to over 7 years
  of continuous human gameplay)."
- "Context length: **10 (observation, action) pairs** / 5560 tokens" → **1.0 s of game time**
  (inferred). The limitations section says "Limited context length (10s)", which conflicts with 10
  pairs at 10 Hz = 1 s — **flagged as ambiguous in the source**.
- Playback: "WHAMM ... is able to generate images at 10+ frames a second ... whereas WHAM-1.6B can
  generate about 1 image a second" (Microsoft Research blog). So WHAM-1.6B runs 10x slower than
  game time; WHAMM roughly at game time.
- Native rate of Bleeding Edge: **not stated** in the card. A 60 FPS console game subsampled to 10
  Hz means a **6x stride** — the most aggressive downsampling in this table.
- The Nature paper itself is paywalled and was **not read** — **VERIFY** the 10 Hz figure against
  the paper if it becomes load-bearing.

### GameFactory — arXiv 2501.08325 (the reference answer for action/frame rate mismatch)
https://arxiv.org/abs/2501.08325

The latent video model compresses time by r = 4, so **4 per-frame actions land inside one model
frame**. Their solution, §4.2 "Grouping Actions with a Sliding Window":

> "Due to the temporal compression ratio r, the number of actions (rn) differs from the number of
> features (n+1), creating a granularity mismatch for action-feature fusion. As shown in Fig. 4, we
> address this by **grouping actions using a sliding window of size w**. For the i-th feature f^i, we
> consider actions within [a^{r(i-w+1)}, ..., a^{ri}]. **This window design captures delayed action
> effects, such as how a jump command influences multiple subsequent frames.**"

Figure 4 caption: "Due to temporal compression (compression ratio r = 4), the number of latent
features differs from the number of actions, causing granularity mismatch during fusion. Grouping
aligns these sequences for fusion. Additionally, the i-th latent feature can fuse with action groups
within a previous window (window size w = 3), accounting for delayed action effects."

Mouse actions are concatenated after grouping; keyboard actions are cross-attended
(K_group as key/value, features as query). Data: "we collected 70 hours of gameplay video"; "We also
**randomize the frame duration of each atomic action** to avoid temporal bias" (§4.1) — i.e. the
action repeat is deliberately varied rather than fixed. Frame rate of GF-Minecraft: not stated.

### Matrix-Game 2.0 — arXiv 2508.13009
https://arxiv.org/abs/2508.13009

- "(2) An action injection module that enables **frame-level mouse and keyboard inputs** as
  interactive conditions" (abstract) — one action per model frame.
- "achieving **25 FPS** generation on a single H100 GPU" (§1). Training-data frame rate, native sim
  rate and any stride: **not stated** in the text searched.

### The Matrix — arXiv 2412.03568
https://arxiv.org/abs/2412.03568

- Data: "the data is segmented into 6-second clips of continuous scenes and captioned using GPT-4o,
  resulting in a dataset of 750k labeled samples and 1.2 million unlabeled samples, **all with 60
  FPS**" (§4). Actions: "This status data is aligned with recorded video frames to create
  **per-frame action-video pairs**".
- Playback: "speeds of up to **16 FPS**", "real-time (8 - 16 FPS) interactive exploration". So
  playback is ~4-8x slower than the 60 FPS source. Whether the model predicts at 60 FPS spacing or a
  stride: **not stated**.

### GameGen-X — arXiv 2411.00769
https://arxiv.org/abs/2411.00769

"each containing 4-16 seconds of content at **24 frames per second**"; bucket sampling over
"durations (from single frames to 480 frames at 24 fps)"; "the model can achieve **20 FPS** on a
single H800 GPU". Native game rate and action repeat: not applicable / not stated.

### GAIA-1 — arXiv 2309.17080 (the keyframe-then-interpolate precedent)
https://arxiv.org/abs/2309.17080

> "To further reduce the sequence length of our world model we **temporally subsample videos from
> 25Hz to 6.25Hz**. This allows the world model to reason over longer periods without leading to
> intractable sequence lengths. **To recover video predictions at full frame rate we perform
> temporal super-resolution using the video decoder** described in Section 2.4." (§2.3)

Training corpus: "4,700 hours at 25Hz of proprietary driving data". This is exactly a **stride-4
world model with a separate temporal upsampler**, and the stated motivation is sequence length /
horizon, not fidelity. No ablation comparing 25 Hz against 6.25 Hz is given.

### GAIA-2 — arXiv 2503.20523 (variable-rate conditioning)
https://arxiv.org/abs/2503.20523

> "To account for variable video frame rates, GAIA-2 uses **timestamp conditioning**. Each timestamp
> is: (i) Normalized relative to the present time and scaled to the range [-1, 1], (ii) Transformed
> using sinusoidal functions (Fourier feature encoding), and (iii) Passed through an MLP ... **this
> encoding ... enables the model to reason effectively over videos recorded at different rates.**"
> (§3)

> "The camera systems varied in their capture frequencies — **20 Hz, 25 Hz, and 30 Hz** — introducing
> a range of temporal resolutions ... supports GAIA-2's ability to generalize across different input
> rates" (§4)

Tokenizer: "Input sequences consisted of T_v = 24 video frames sampled at their native capture
frequencies (20, 25, or 30 Hz)", 8x temporal compression to 3 latents.

### Vista — arXiv 2405.17398
https://arxiv.org/abs/2405.17398

"Vista acquires the ability to anticipate realistic futures at **10 Hz** and 576x1024 pixels" (§1);
Table 1 lists prior driving world models' frame rates as 5 Hz, 8 Hz, 12 Hz, 2 Hz, 2 Hz, 2 Hz, 2 Hz,
and criticises them: "they are also often **confined to low frame rates and resolutions, resulting
in a loss of critical details**." This is a claim that higher rate is better, asserted rather than
ablated.

### NWM (Navigation World Models) — arXiv 2412.03572 (variable spacing, and a real spacing result)
https://arxiv.org/abs/2412.03572

- **Variable spacing by construction.** §3.1:
  > "The formulation in Equation 1 models action but does not allow control over the temporal
  > dynamics. We extend this formulation with a **time shift input k ∈ [T_min, T_max]**, setting
  > a_τ = (u, φ, k), thus now a_τ specifies the time change k, used to determine how many steps
  > should the model move into the future (or past). Hence, given a current state s_τ, we can
  > **randomly choose a timeshift k** and use the corresponding time shifted video frame as our next
  > state s_{τ+1}."

  > "**The navigation actions can then be approximated to be a summation from time τ to m = τ+k-1**"
  — i.e. when one model step spans several control steps, the actions are **summed**.

  > "In practice, we allow time shifts of up to **±16 seconds**."
- **Spacing vs rollout horizon, measured.** §4.2:
  > "We compare predictions to ground truth images at 1, 2, 4, 8, and 16 seconds, reporting FID and
  > LPIPS on the RECON dataset. Figure 4 shows performance over time compared to DIAMOND at 4 FPS and
  > 1 FPS, showing that NWM predictions are significantly more accurate than DIAMOND. **Initially,
  > the NWM 1 FPS variant performs better, but after 8 seconds, predictions degrade due to
  > accumulated errors and loss of context and the 4 FPS becomes superior.**"

  Read carefully: coarse spacing (1 FPS, 1 s per step) wins at short horizons; fine spacing (4 FPS,
  0.25 s per step) wins beyond 8 s. The attribution of "accumulated errors" to the 1 FPS variant is
  counter-intuitive (it takes 4x fewer steps) and the paper does not explain it further. Figure 4 is
  a figure, so the per-point numbers were **not extracted** — **VERIFY** if the exact crossover
  matters.
- Planning uses the fine setting: "We generate trajectories of length 8, with **temporal shift of
  k = 0.25**" (§4.3) — i.e. 4 FPS for downstream control.
- Table 1 ablates "the use of **action and time conditioning**" at a 4 s prediction horizon; numbers
  not extracted (table markup) — **VERIFY**.

### Vid2World — arXiv 2505.14357 (runs baselines at two spacings)
https://arxiv.org/abs/2505.14357

- "Unlike our model, which is restricted to sequential, autoregressive generation, **NWM explicitly
  conditions on the prediction timestep t**, allowing single-step prediction of a distant future
  frame ... we evaluate against both baseline setups: single-step prediction and autoregressive
  rollout, at the dataset's **native 4 fps rate**." (§5.3)
- Appendix C.6: "except for the normal version of predicting future frames at 4 fps, there is also a
  **downsampled version, which predicts future frames at 1 fps. This results in fewer autoregressive
  rollout steps, potentially leading to less error accumulation.**" And: "we include **DIAMOND
  (1fps), which is a model trained using observations and actions at intervals of 1 second**."
- So there exist trained DIAMOND and NWM variants at 1 s and 0.25 s spacing evaluated head to head.
  **The per-metric numbers are in Table 1/Table 2 and were not extractable from the HTML dump —
  VERIFY by reading the PDF tables.** This is the single most directly relevant table found.
- CS:GO long-rollout protocol holds actions: "The action sequence is sampled uniformly from
  {W,A,S,D}, with **each action held constant for 10 consecutive frames** to induce significant
  movement" (Appendix D.2) — action repeat at *evaluation* time, on a model that predicts every
  frame.
- Training: "During training, we use a context length of 16, **with no downsampling in the temporal
  fps**. Since the dataset is collected at 4 fps ... the model is provided with a history of 4
  frames ... and predicts a sequence of 16 frames".

### V-JEPA 2 / V-JEPA 2-AC — arXiv 2506.09985 (frame-rate ablation at fixed context budget)
https://arxiv.org/abs/2506.09985

- Action-conditioned model, §3.1: "The video clips are sampled with resolution 256x256 and a
  **frame-rate of 4 frames-per-second (fps)**, yielding 16 frame clips" → **4 s of context at 250 ms
  spacing**. Actions are per-interval deltas: "We construct a sequence of actions (a_k) by computing
  **the change in end-effector state between adjacent frames**."
- The understanding model has the ablation the field otherwise lacks, Appendix (Figure 17):
  "(Middle) Performance with respect to the **frame rate (fps) used for inference; number of context
  frames fixed at 32**." and
  > "We report in Figure 17, the impact of input resolution and frame sampling parameters. In
  > summary, **V-JEPA 2 benefits from a longer context, a higher frame rate, and higher resolution,
  > up to a point where the performance saturates or slightly decreases.** The optimal performance is
  > obtained by training with a 32-frames context length, a frame rate of 8 and a resolution of
  > 384x384."

  This is a *representation-quality* result (action anticipation on EK100), not a world-model
  rollout result, but it is the clearest "rate at fixed frame budget" curve found: quality rises
  with rate, then saturates.

### IRASim — arXiv 2406.14540
https://arxiv.org/abs/2406.14540 — searched for fps / Hz / frame rate / subsample; **no matches**.
Frame spacing **not stated** in the HTML text.

---

## 3. Answers to the four questions

### Q1. Does anyone ablate prediction spacing / action repeat / frame rate for a world model?

**Almost nobody, and no game world model at all.** Four near-misses, in descending order of
relevance:

1. **Vid2World (2505.14357)** trains and evaluates **DIAMOND and NWM at 1 fps and 4 fps** on RECON —
   the same model family at two spacings, measured on rollout accuracy (PSNR/LPIPS/FVD) over a
   horizon. Quote: "there is also a downsampled version, which predicts future frames at 1 fps. This
   results in fewer autoregressive rollout steps, potentially leading to less error accumulation";
   "DIAMOND (1fps), which is a model trained using observations and actions at intervals of 1
   second". **The numbers are in tables I could not extract — this is the first thing to read next.**
2. **NWM (2412.03572)** measures the same two spacings itself: "Initially, the NWM 1 FPS variant
   performs better, but after 8 seconds, predictions degrade due to accumulated errors and loss of
   context and the 4 FPS becomes superior." That is a direct (b) long-horizon-stability result:
   coarse spacing wins short-horizon fidelity, fine spacing wins long-horizon. For (c) downstream
   control, NWM's own planner uses k = 0.25 s (4 FPS).
3. **V-JEPA 2 (2506.09985)** ablates inference frame rate at a **fixed 32-frame context**, finding
   quality rises with rate then saturates or slightly decreases. Closest to (a) fidelity per unit
   compute, but for representation quality, not generative rollout.
4. **GAIA-1 (2309.17080)** is the alternative architecture rather than an ablation: predict at
   6.25 Hz (stride 4 of 25 Hz) and recover the missing frames with a **temporal super-resolution
   decoder**. Motivation given is horizon length, and no comparison against 25 Hz prediction is
   reported.

Related mechanisms rather than ablations: **GAIA-2** conditions on normalized timestamps so one
model handles 20/25/30 Hz; **NWM** conditions on k directly, making spacing an input; **GameFactory**
randomizes how long each atomic action is held. Diffusion-forcing style per-token noise levels
(Chen et al. 2024, cited in GameNGen §6 as future work) are about horizon, not spacing.

**Nothing in the game world-model literature reports (a), (b) or (c) as a function of spacing.**

### Q2. How do models that predict every rendered frame represent an action held over k frames?

The universal answer is **repeat the action on every frame; do not mark the first frame**:

- GameNGen: "the last 64 actions" for 64 frames, one learned token per action, with the agent's
  action applied for 4 frames. The repetition is implicit and never discussed.
- gameNgen-repro: `actions.append(int(current_action.item()))` inside the per-tic loop
  (`load_model_generate_dataset.py:160`) — literally the same action id written 4 times.
- MineWorld: "each frame is accompanied with the keyboard and mouse actions taken at the same time".
- Matrix-Game 2.0: "frame-level mouse and keyboard inputs"; The Matrix: "per-frame action-video
  pairs".

The **converse** case (several actions inside one model frame) does get engineered:

- **GameFactory** groups the r = 4 actions per latent plus a w = 3 latent window and cross-attends,
  explicitly "accounting for delayed action effects (e.g., 'jump' key affects several subsequent
  frames)".
- **NWM** sums the actions over the interval: "the navigation actions can then be approximated to be
  a summation from time τ to m = τ+k-1".
- **V-JEPA 2-AC** uses the end-effector delta across the interval, which is the same summation idea.

**No paper found discusses what the world does during a held action**, i.e. Rohan's concern that
enemies, projectiles and the player's own momentum evolve across the 4 held tics. The closest thing
is GameFactory's "delayed action effects" framing, which is about an action's consequences extending
beyond one latent, not about lost supervision within one.

### Q3. GameNGen specifically

(i) **Agent frame skip: 4, stated.** Appendix A.5: "we apply each agent action for 4 frames and
additionally artificially increase the probability of repeating the previous action."

(ii) **Training-frame spacing: never stated.** The only quantitative handle is Appendix A.6's
"teacher-forced trajectories on **35FPS data (the maximal sampling rate allowed by ViZDoom**...)",
which is only coherent if stored frames are consecutive 35 Hz tics — i.e. **consecutive rendered
frames, not every 4th**. The public reproduction, whose author read the paper, implemented exactly
that (`frame_skip = 1`, "capture all frames", new action every 4th tic). Secondary support: 70M
training examples exceed 50M environment steps. **Confidence: high but not certain; the paper is
silent.**

(iii) **"20 FPS" is generation speed.** §3.3.2 derives it from latency: "Using just 4 denoising steps
leads to a total U-Net cost of 40ms (and total inference cost of 50ms, including the auto encoder)
or 20 frames per second"; the distilled 1-step model reaches 50 FPS. Our Sep 10 claim is correct on
this point.

(iv) **The 64-frame context spans 1.83 s of game time (inferred at 35 Hz), not the "little over 3
seconds" the paper claims.** The paper converts frames to seconds at 20 fps everywhere ("16 frames
(0.8 seconds) and 32 frames (1.6 seconds)"), which is the *playback* rate. Either the paper's
seconds are wrong by 1.75x, or its data is not 35 Hz and A.6 is wrong. **The paper is internally
inconsistent and cannot be used as a source for context-in-seconds.** A consequence worth stating in
our write-up: GameNGen's human raters watched 20 fps playback of (probably) 35 Hz footage, i.e.
0.57x-speed Doom.

**And the number that matters for us:** the 29.43 dB / 0.249 LPIPS headline is **single-frame
teacher-forced prediction with 64 ground-truth context frames**, at a spacing of ~28.6 ms. On our
footage, copy-the-last-frame at a 1-tic gap scores 21.3 dB (per the brief) — so GameNGen's task is
the easy end of the spacing axis, and its number is not comparable to a stride-4 number at all.

### Q4. Synthesis (under 250 words)

**What papers did.** Two camps. RL world models make the model timestep equal the *decision*: SimPLe
and DIAMOND on Atari use frame skip 4 and simply throw the intermediate frames away; STORM states
the convention outright; TD-MPC2 sets action repeat 2 per domain. Neural game engines instead make
the model timestep equal the *rendered frame*: GameNGen (most likely), its reproduction (certainly),
MineWorld, Matrix-Game 2.0 and The Matrix all carry one action per frame and repeat a held action.
Anything trained on video rather than an engine subsamples hard for sequence-length reasons —
GAIA-1 25 → 6.25 Hz plus a temporal super-resolution decoder, WHAM to 10 Hz, Genie 10 FPS — and
nobody claims that helped fidelity.

**What papers showed to be better.** Almost nothing. NWM is the only action-conditioned model that
measures two spacings of one architecture: 1 FPS wins early, 4 FPS wins past 8 s. Vid2World runs
DIAMOND/NWM at 1 and 4 fps but I could not extract its tables. V-JEPA 2 shows quality rising with
rate at a fixed 32-frame budget, then saturating. No game paper reports a copy-frame baseline at
all, so no reported PSNR in this literature is normalised for spacing.

**For a 35 Hz game, 4-tic action repeat, 32-frame budget.** Evidence supports stride 1 for fidelity
per step and stride 4 for horizon coverage (3.66 s vs 0.91 s). The settling experiment: train the
same DiT at stride 1, 2 and 4 with the 32-frame budget fixed, and report PSNR/LPIPS **against the
copy-last baseline at that stride**, plus 5 s rollout error per unit *game time*, not per step.

---

## 4. Corrections to our existing notes

1. **`skipframe-4` does not mean stride 4.** Our Sep 10 claim that "GameNGen and its open
   reproduction use frame skip 4" is wrong for the reproduction and unsupported for GameNGen. The
   repro stores **every** tic (`load_model_generate_dataset.py:132-133`, `frame_skip = 1`, "capture
   all frames") and its dataloader takes **consecutive** rows (`dataset.py:61`). The "4" is the
   agent's action repeat. Anywhere we say the repro trains at 8.75 fps, that is wrong: it trains at
   35 fps with 9 context frames = 0.26 s.
2. **"GameNGen's 20 FPS is generation speed" — correct, but incomplete.** The paper's own
   frames-to-seconds conversions (0.8 s / 1.6 s / "a little over 3 seconds") are computed at 20 fps
   and contradict its "35FPS data". We should say the paper is internally inconsistent rather than
   citing its "3 seconds of history".
3. **GameNGen 29.43 dB is a next-*tic* number, not a next-decision number.** Any table that puts our
   26.04 dB (stride 4, 114 ms) next to GameNGen's 29.43 dB (~28.6 ms, 64 GT context frames) is
   comparing different tasks. If we cite it, we must state the spacing and the absence of a
   copy-frame baseline on their side.
4. **Our own persistence numbers disagree across documents.** The brief states 21.3 dB at a 1-tic
   gap and 18.8 dB at 4 tics; `REQUIREMENTS.md` R1.3 states "copy-last is 26.2 dB per-tic vs 19.3 dB
   at stride 4". The stride-4 figures (18.8 vs 19.3) are close, the 1-tic figures (21.3 vs 26.2) are
   not. One of these is measured on different footage or a different mask. **This must be resolved
   before either number appears in the paper**, because the whole framing of R1.3 rests on it.
5. **`doom-baselines-2026-09-17.md` lists Oasis as "latent, 20 fps"** in a column that reads as a
   data/model property. The 20 FPS on the Oasis page is generation speed. Minecraft's tick is also
   20/s so the two may coincide, but the source does not say so; the cell should be annotated.
6. **`REQUIREMENTS.md` line 18 is consistent** with everything found here: the 35 Hz tick, the
   29 ms next-frame gap, and the observation that this inflates PSNR. No change needed there beyond
   adding that GameNGen sits at exactly that 29 ms gap.

---

## 5. Could not verify

- **Vid2World's RECON tables (Table 1, Table 2)** — the 1 fps vs 4 fps numbers for NWM and DIAMOND.
  HTML table extraction failed; read the PDF. **This is the highest-value remaining item.**
- **NWM Figure 4** per-point FID/LPIPS for the 1 FPS / 4 FPS crossover, and **NWM Table 1**
  (action/time conditioning ablation).
- **GameNGen's actual training stride** — only inferrable. Nothing in the paper states it, and the
  authors' code is not public.
- **The second GameNGen reproduction**, github.com/Masao-Taketani/GameNGen — not inspected.
- **DIAMOND Atari max-pooling** over the 4 skipped frames — the paper states "Frameskip 4" but not
  the pooling; the brief's assumption is unconfirmed.
- **CS:GO 64-tick server rate** — DIAMOND says only "captured at 16Hz"; the 64-tick figure would
  have to come from Pearce & Zhu (2022), unread.
- **DreamerV3 per-domain action repeat** — absent from the Nature text; it lives in the released
  configs.
- **VPT's 20 Hz recording rate** — not stated in MineWorld; would need 2206.11795.
- **WHAM's 10 Hz** — from the HF model card, not the paywalled Nature paper. The card's "Limited
  context length (10s)" conflicts with "10 (observation, action) pairs" at 10 Hz = 1 s.
- **Oasis** data rate, context length, action conditioning — the project page states only 20 FPS
  generation, and `open-oasis/README.md` was not reachable.
- **Matrix-Game 2.0, MultiGen, PlayGen** training-frame spacing — searched, genuinely not stated in
  any of the three.
- **Not covered at all** (time): AVID (2410.11471, no arXiv HTML), Genie 2 / Genie 3 blog posts,
  Lucid / MineDreamer, Cosmos action-conditioned post-training, 1X world model, Genie Envisioner,
  DriveDreamer, UniSim, TWM, MuZero, Delta-IRIS, and the BAIR / RoboNet / KTH benchmark rates.
