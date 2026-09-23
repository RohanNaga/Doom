# Next-tic training and evaluation: the rules as implemented

Written 2026-09-20/21 alongside the code. Every rule below is what the code does, with the test that
pins it. Read this before changing anything in `doom_data.TicWindowDataset`, `train_wm.py`'s data
path, or the two evaluators: several of these rules exist because a plausible alternative is wrong
in a way no loss curve would reveal.

The decision (Rohan, Sep 20 2026): the next training runs predict the next **tic**, not the next
agent decision, on the dense corpus, for SD 1.4 U-Net and SD 3.5 (PixArt available as a third row).

---

## 1. Which action conditions which target

**The rule.** For target row `r` and context length `L`:

```
context  = latents[r-L : r]        L consecutive tics, oldest first, channel-stacked
target   = latents[r]              one tic (28.6 ms) later
controls = buttons[r-L : r]        exactly L executed button vectors, oldest first
```

`controls[-1] == buttons[r-1]` is the control that produces the target frame. `buttons[r]` is
**never** included.

**Why that row.** `record_arnold.py` stores a row and *then* steps the engine (`:226` stores
`(tic, action_id, bstr)`, `:265` calls `make_action`), so the value on row `t` is the control applied
from frame `t` to frame `t+1` — the module docstring of `transitions.py` states this as the row
semantics. The control carrying the last context frame into the target therefore sits on row `r-1`,
the newest of the `L` tokens. Including `buttons[r]` would condition the target on a decision made
after the target frame was observed.

This generalises, rather than replaces, the scalar the stride-4 path already used:
`LatentWindowDataset.__getitem__` reads `act[start + L - 1]`, which is `action[r-1]`. A test asserts
the newest control token's source row equals that scalar's row, so a next-tic row and a
next-decision row are conditioned by one rule and differ only in spacing.

**The open GameNGen reproduction uses the opposite convention** (`load_model_generate_dataset.py`
steps and *then* appends, so its row `i` carries the control that produced frame `i`). Copying its
slice literally would be a one-tic error. `check_action_alignment.py` calls that convention shift +1.

**Executed controls, not the requested action id.** Each token is the `buttons` column — the 19-bit
0/1 vector the engine actually executed, weapon-selection bits included (width read from the corpus,
`docs/cards/arnold/buttons.json` says 19). Arnold's anti-stuck override holds a different vector for
up to 40 tics, and the stride-4 corpus hid that by keeping only transitions whose executed vector
matched the canonical vector of the requested id. A per-tic corpus keeps every tic, including those,
so the id would be a wrong label on exactly the rows where it matters. The sidecar already carries
`buttons` per tic, so **no re-encoding is needed**; `check_action_alignment.py --audit-parquet-dir`
checks the sidecar against the recording row for row and reports the override fraction.

`--action-history 0` keeps the single action-id token, bit-identically: no control embedder is
built, nothing is drawn from the RNG, and the state dict is unchanged.

## 2. Window validity

A window spans rows `start .. start + L + horizon - 1` of one episode file and is valid only when,
for **every** adjacent pair it covers:

| | rule | why |
|---|---|---|
| R1 | one episode file | true by construction: one file per episode |
| R2 | `tic[i+1] - tic[i] == 1` | the target really is one tic later, and no unrendered tic hides inside |
| R3 | `deaths[i+1] == deaths[i]` | `deaths` increments on the respawn row, so a constant value means one continuous life and the pixels never teleport to a spawn point |
| R4 | `map_id[i+1] == map_id[i]` | a future recording that walks maps inside one file cannot put a map load mid-window |

**`chain_id` is deliberately not used.** `encode_parquet.py:127` writes `-1` on every non-decision
tic, so the stride-4 endpoint test `cid[s] == cid[s + L]` is satisfied by two `-1` endpoints and
accepts a window that runs straight through a death. That is tested directly: a synthetic episode
with a respawn mid-window and every `chain_id == -1` is accepted by the old test and rejected by the
per-tic contract. `LatentWindowDataset` now refuses a corpus that carries an `is_decision` column
rather than sampling it through a test that means nothing at tic spacing, and
`rollout_eval.collect_rollout_windows` uses the same per-tic contract over the whole seed and horizon.

`TicWindowDataset.summary` reports `candidate_windows`, `windows` and `excluded_fraction`, and the
trainer writes it into `config.json` and the start log, so a "within-life simulation" claim can
state what it threw away.

## 3. `tics_since_decision` (phase), off by default

`PHASE_BUCKETS = 5`: 0, 1, 2, 3 are the positions inside a 4-tic held-action run, and bucket 4 means
"no verified decision row within a control interval before this row". The extra bucket exists
because `is_decision` marks only the rows whose whole run was clean and canonical, so an interrupted
decision, an override, or the rows before an episode's first accepted decision leave a gap in the
grid. Reserving a bucket keeps the **window set identical** whether the flag is on or off, instead of
folding an off-grid tic into phase 3 — otherwise the ablation would be confounded by a different
dataset.

Astra's position, adopted: no phase in the main run. The engine's next transition depends on its
state and the current buttons, not on when Arnold intends to reconsider them, and held buttons simply
repeat on every tic. If phase is ever tested it must come from the controller's decision clock and
reset at respawn — which is what `is_decision` gives — and never from a global `tic % 4`.

## 4. Equal game time

A next-tic model predicts 28.6 ms ahead; a stride-4 model predicts 114 ms ahead. The two numbers are
not comparable, so:

* `eval_tf.py --horizon-tics K` rolls the model K tics forward from **real** context, feeding its own
  outputs back with the recorded controls, and scores the Kth frame. `K = 4` is the same 114 ms as one
  step of a stride-4 model. The floor is recomputed at gap K (the raw frame `K * tic_stride` tics
  earlier), not at gap 1.
* `K = 1` collapses to exactly the old single-step loop, and `K > 1` is refused for a stride-4
  checkpoint.
* `rollout_eval.py` horizons are stated in tics: 4, 32, 64, 128, 256 tics is the same game time as
  1, 8, 16, 32, 64 decision steps. `after_nexttic.sh` rolls out 256 tics = 7.31 s.
* FVD clips are written at **both** spacings: `clips_u8.npz` every tic and `clips_u8_stride4.npz`
  every fourth tic, because a 16-frame clip of consecutive tics is 0.46 s of game time while a
  stride-4 row's 16-frame clip is 1.83 s.

## 5. The floor is decoder-independent

`eval_tf.py`'s existing `copy_psnr_raw` decodes the last context *latent* and scores it against the
raw target, so it carries the autoencoder's own reconstruction error and is not a persistence
reference. Added: `persist_psnr_raw`, `persist_lpips_raw`, `persist_hud_psnr_raw` — the **raw** frame
`K * tic_stride` tics earlier against the **raw** target. The old keys stay for continuity with the
stride-4 rows; every next-tic number is read against the new ones. The autoencoder ceiling
(`vae_psnr`) is unchanged.

## 6. The IDM is scored only on decision tics

`train_idm.WindowDataset` was trained on the stride-4 corpus: 8 consecutive **decision-frame** latents
inside one chain, with a learned positional table for that spacing. Consecutive tics would show it a
quarter of the motion it ever saw. So under `tic_stride == 1` the rollout is subsampled to its
decision tics (the `decision` / `seed_decision` masks are stored in the npz), the seed contributes its
last `K-1` decision frames, and the label for each judged transition is `actions[n][k]` at the
position of the later decision frame — which on the grid is the action held over the whole interval.
`idm_spacing_tics`, `idm_rollouts_scored` and `idm_rollouts_skipped` are reported; rollouts without a
uniform decision subsequence are skipped and counted rather than silently averaged in.

## 7. What is identical to the stride-4 path

Everything except the frame spacing and (optionally) the conditioning interface:

* sample shapes: `(C*L, 32, 40)` context, `(C, 32, 40)` target — so **every backbone is unmodified**;
* the recipe: global batch 32, lr 5e-5 constant after 2,000 warmup, clip 1.0, fused AdamW, bf16
  autocast, fp32 CPU EMA at 0.9999, context noise max 0.7 with 10 buckets, no action dropout,
  v-prediction, seed 0;
* the seeded validation corruption: `SeededCorruption` keeps its seven-element layout and appends the
  phase last, so an existing run's validation loss cannot move;
* `--tic-stride 4` is the default everywhere, and the evaluators read the spacing from the
  **checkpoint** (`eval_tf.checkpoint_interface`), so a pre-flag checkpoint scores exactly as before;
* `SyntheticWindows` keeps its draw order (context, target, action), so fit-check windows do not move.

`config.json` and the start log record `tic_stride`, `dataset_class`, `resolved_phase_buckets`,
`resolved_control_bits`, `dataset_summary`, `init_from` and `init_from_step`.

## 8. The split, fixed before anything was scored

`release/dense_split.json`, over `raw_arnold_dense/arenas` (8,000 episodes, maps 2-5):

| range | ids | per arena |
|---|---|---|
| train | 0:6000 | 1,500 |
| val | 6000:7000 | 250 |
| test | 7000:8000 | 250 |
| unseen (`arenas_678`) | 60:120 (0:60 until 2026-09-22: 51 of those 60 are worker-first episodes; `release/dense_split.json` history) | 20 |

Ranges are half-open, like a Python slice. `record_arnold.py:322-323` assigns
`map_ids[episode_id % len(map_ids)]` independent of the worker count, so any contiguous range whose
length is a multiple of the map count is exactly map-balanced; a test proves it from that rule rather
than from the corpus. The first next-tic runs train on `0:2000` and validate on `6000:6100`.
`check_dense_training_ids` refuses a training range that reaches into val or test at startup, which
is the one guard between a mistyped `TRAIN_IDS` and a headline measured on training data.

A run resumed after more episodes finished encoding would silently train on a larger set, so the
first launch writes the resolved train and val lists to `train_episodes.json` and every `--resume`
reads them back and refuses if any of those episodes has disappeared. Starting on a partly encoded
corpus needs `--allow-partial`.

## 9. Encoder throughput: the measured answer is no

**Question asked:** is the measured 66 frames/s on an A6000 plausibly bound by PNG decoding or
parquet reads rather than the VAE?

**Answer: no, and by three orders of magnitude.** Measured on a 10-core M4 laptop CPU, 256 synthetic
320x240 PNGs at compress level 6 (6.4 KiB each, the size real frames compress to), running the
encoder's own `decode`:

| path | frames/s |
|---|---|
| serial decode, 1 thread | 5,043 |
| thread pool 4 / 8 / 16 / 24 | 14,695 / 15,341 / 15,064 / 13,948 |
| process pool 4 / 8 | 11,449 / 11,213 |
| `pq.read_table` + per-row `as_py()` | 405,786 |

At 66 frames/s the encoder spends about 15 ms per frame. One thread decodes a frame in 0.2 ms and the
parquet read costs 0.0025 ms per frame from a warm local file. **Decoding cannot be the bottleneck**,
and the comment in `scripts/spiderman/encode_pertic.sh` ("the encoder is PNG-decode bound, so threads
matter for speed") is wrong.

Note also that the process pool is *slower* than the thread pool here: PIL releases the GIL inside the
PNG decoder, and pickling a 230 KB uint8 array back to the parent costs more than the GIL contention
it removes.

`--decode-workers N` was added as specified and is byte-identical to the thread path (`decode` is a
pure function of the PNG bytes and both pools' `map` preserves order), but **it should be left at 0**.
The remaining suspects, in the order I would test them on the server, are: the VAE forward itself at
the chosen batch size (a `--decode-check`-style timing of `encode_batch` alone would settle it in
minutes); reading 1.7 TiB off `/sata2` rather than a warm page cache; and the GPU being shared. This
needs a measurement on the card, which I could not make from here.

---

## 10. What I think is wrong or risky in this design

**a. Context is 0.91 s of game time, down from 3.66 s.** 32 tics at 35 Hz is a quarter of the history
32 decision frames gave. GameNGen's own 32-vs-64-frame ablation is 0.05 dB, but GameNGen's frames are
1 tic apart, so that ablation prices 0.91 s against 1.83 s, not against 3.66 s. Nothing in the
literature prices 0.91 s against 3.66 s at tic spacing. The alternative is context 64 tics (1.83 s,
GameNGen's own figure) at double the input channels and roughly double the patch-projection cost, or
keeping 32 tics but subsampling the context non-uniformly (recent tics dense, older tics at stride 4)
so 32 channels cover 3.66 s. The second is cheap and, as far as I can tell, untried; it would be my
first ablation after the main rows.

**b. Per-tic windows include the anti-stuck-override tics that the stride-4 corpus excluded.** The
`--align-decisions` filter dropped transitions whose executed vector was not canonical; the per-tic
contract keeps them. Conditioning on the executed vector makes the label correct, so this is a
deliberate widening rather than a bug, but it means the per-tic and stride-4 corpora are not the same
footage and a "same data, different spacing" claim would be false. The audit reports the override
fraction so the difference can be stated.

**c. Next-tic prediction makes the task easier, and the headline will move for that reason alone.**
Astra says it plainly and the floor arithmetic agrees: the per-tic persistence floor is about 2.7 dB
above the stride-4 floor. Every next-tic number must be reported as floor-relative or it is not a
result. `persist_psnr_raw` exists for exactly this.

**d. The DiT cannot take an action sequence.** It has no cross-attention pathway, so
`--action-history` averages the position-embedded control tokens into the adaLN vector. That is not
GameNGen's construction and the DiT should not be a next-tic row on that basis; it is there for fit
checks. Stated in the code at the injection site.

**e. The IDM comparison across spacings is weaker than it looks.** Subsampling to decision tics gives
the IDM the interval it was trained on, but a per-tic model's decision-tic subsequence is the product
of four autoregressive steps where a stride-4 model's is one. `idm_top1` is therefore not a like-for-
like comparison even after subsampling; it compares action fidelity at equal game time under
different numbers of model calls. Worth saying in the paper rather than fixing.

**f. The alignment gate's thresholds are now tightened, but still uncalibrated on real data.** As of
Sep 21 the gate needs, on the YAW axis only: at least 1,000 scored boundary rows, at least 20
episodes, at least 100 rows in each motion class that occurs, at least two classes occurring,
balanced accuracy at shift 0 of at least 0.95, a lead of at least 0.20 over *both* neighbours, and a
95% bootstrap interval on that lead — resampling EPISODES, not rows — that excludes zero. The yaw
deadband is 0.25 degrees, a seventh of the engine's smallest turn (about 1.758 degrees per tic), so
a real turn is never filtered out as noise.

Translation is a printed diagnostic and never a veto. Displacement between t and t+1 reflects
momentum built before t, so the control one tic earlier legitimately explains it better; a correct
system scores about 0.83 at shift −1 against 0.50 at shift 0 on that axis, and letting it vote would
report a correctly aligned corpus as misaligned. A missing yaw score is `EXIT_INCONCLUSIVE`, never 0.

What is still uncalibrated: every threshold above was chosen by argument, not measured, and the
numbers have only been exercised on synthetic episodes. Run the gate on a real encoded corpus and
read `rows`, `episodes` and `per_class` before trusting the verdict. If a threshold is wrong the gate
says "inconclusive", which is a refusal rather than a false pass — but an inconclusive verdict on
good data would block a launch for no reason, so read the numbers rather than only the exit code.

**g. `--init-from` is available but unused by default.** Rohan's decision is the public pretrained
weights for every next-tic row, for purity of the warm-start comparison. Astra's estimate is that
about 30,000 U-Net updates recover adaptation an existing Doom checkpoint would supply, i.e. roughly
12.5% of a 240k-update allocation. That is a real cost, knowingly paid.
