# Research Ideas

Speculative directions for DoomDiT. Nothing here is implemented; each entry is a
sketch with enough detail to be picked up later.

---

## Scenario latents as portable "skill files"

**One line.** Learn a compact continuous vector that summarizes *what kind of
situation the model is in*, then treat that vector as a saveable, nameable,
composable file — a skill — that steers rollouts the way a prompt steers an LLM.

### Where this comes from

Today the world model has exactly two conditioning signals:

| signal | shape | how it enters |
|---|---|---|
| frame history | `(N, 16, H, W)` — 4 past latents, channel-stacked | concatenated onto the noisy target ([models.py:256](models.py:256)) |
| action | `(N,)` — one of 18 VizDoom actions | `LabelEmbedder` → `c = t + y` ([models.py:261](models.py:261)) |

Both are *local*. The history covers ~4 frames and the action covers exactly one
step. Nothing tells the model "you are in a tight corridor fight" versus "you are
strafing an open courtyard" beyond whatever leaks through those 4 frames. That
missing signal is the slow-varying, episode-level context — call it the
**scenario latent** `s ∈ R^d` (d ≈ 64–256).

### The idea

Add `s` as a third additive conditioning term:

```python
c = t + y + self.s_embedder(s)      # s_embedder: small MLP, R^d -> hidden_size
```

Give `s_embedder` the same dropout-to-null treatment `LabelEmbedder` already uses
for CFG, so at sample time you get a knob: guidance scale = *how hard to push
toward this scenario*.

Once `s` exists and conditions generation, it stops being an internal activation
and becomes an **artifact you can serialize**. That's the actual proposal.

### Three ways to obtain `s`

1. **Learned per-episode embedding.** An embedding table indexed by episode ID,
   trained jointly. Cheapest, but only gives vectors for episodes in the training
   set — no way to make new ones.
2. **Clip encoder.** A small set-transformer over K frame latents (K ≈ 16–32),
   mean-pooled to `s`, trained end-to-end with a KL penalty toward `N(0, I)` to
   keep the space smooth enough for interpolation. Gives you `s` for any clip,
   including unseen ones.
3. **Latent inversion (the interesting one).** Freeze the trained world model.
   For a new demo clip, optimize *only* `s` by gradient descent on the standard
   diffusion loss for that clip. This is textual inversion, transplanted. It means
   authoring a new skill costs ~5 seconds of gameplay and a few minutes of
   optimization — no retraining, no new data pipeline. This is what makes a skill
   *library* practical rather than a research curiosity.

(2) and (3) compose: use the encoder for a fast initialization, then invert to
sharpen it.

### Skill files

A skill file is `s` plus the metadata needed to use it safely:

```yaml
# skills/corridor_pushforward.skill.yaml
name: corridor_pushforward
description: tight corridor, forward advance, enemies ahead
vector: corridor_pushforward.npy       # (d,) float32
derived_from:
  method: inversion                     # embedding | encoder | inversion
  source_clip: ep_0143 frames 210-242
  steps: 400
compatible_with:
  model: DiT-XL/2 in=20 pred=4 classes=18
  ckpt_sha256: <hash of best.pt>        # a vector is meaningless against other weights
recommended:
  guidance_scale: 2.0
  action_prior: [3, 3, 3, 10]           # actions this skill co-occurs with
```

The `compatible_with` block matters more than it looks. A latent vector is
defined purely by the weights that interpret it — ship one against the wrong
checkpoint and you get plausible-looking garbage with no error. The hash makes
that a loud failure instead of a silent one.

### Workflow files

A skill is a single point in latent space. A **workflow** is a schedule over
them — the thing you'd actually hand to a long rollout:

```yaml
# workflows/clear_the_level.workflow.yaml
timeline:
  - skill: corridor_pushforward
    frames: 48
  - blend: [corridor_pushforward, open_arena_strafe]
    over_frames: 12                     # transition, not a cut
  - skill: open_arena_strafe
    frames: 64
    guidance_scale: 1.5                 # per-segment override
```

[rollout_video.py](rollout_video.py) already walks a fixed action sequence frame
by frame ([rollout_video.py:98](rollout_video.py:98)); a workflow file slots in
as the *other* input to that same loop — actions drive the step, the scenario
vector drives the regime. Blending across a transition window is what keeps the
seam from producing a hard visual cut.

### Why this is worth doing

- **Long-horizon control.** Autoregressive rollout drifts. A held-constant `s`
  is a low-frequency anchor that the per-frame prediction can't wander away from —
  a plausible, cheap fix for a known failure mode of this model class.
- **Composition is testable.** Does `s_corridor + s_combat - s_empty` produce a
  corridor fight? Vector arithmetic gives a falsifiable claim, not a vibes demo.
- **Retrieval.** Encode the live 4-frame context, nearest-neighbor it against the
  skill library, and auto-apply the match. The world model picks its own skill —
  the same on-demand-loading pattern as agent skill files, but the index is a
  latent space instead of a description string.
- **Few-shot scenarios.** Inversion means a new scenario needs a handful of
  seconds of footage, not a training run.

### Prerequisites and honest caveats

- **Needs scenario diversity in the data.** If the training set is effectively one
  VizDoom scenario, `s` has nothing to separate and will collapse. Multi-scenario
  data collection comes first; this is the real blocker.
- **Entanglement risk.** `s` may capture *appearance* (this level's textures)
  rather than *behavior* (how the agent moves here). Worth probing early: hold `s`
  fixed and vary the action sequence — if the motion character doesn't change,
  the vector is a texture code, not a skill.
- **Evaluation gap.** "Does the skill reduce drift" needs autoregressive metrics,
  which don't exist in this repo yet. That harness is a dependency, not a detail.

### Rough sequencing

1. Multi-scenario data collection.
2. Encoder path (2) + `s_embedder` in `DiT.forward` — smallest change that tests
   whether `s` is learnable at all.
3. Probe entanglement; verify `s` carries behavior, not just appearance.
4. Inversion path (3) → the first real skill files.
5. Workflow scheduling in `rollout_video.py`.
6. Retrieval-based auto-selection.

### Open design question

<!-- TODO(rohan): composition semantics — see note below -->
How multiple skill vectors combine is unresolved, and the choice has real
consequences. Sketch the intended semantics here.
