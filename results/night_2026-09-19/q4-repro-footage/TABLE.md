# How easy is the open GameNGen reproduction's footage?

Measured Sep 19 to 20 2026 with `persistence_stats.py`, 20,000 sampled frame pairs per corpus, 40
episodes each, seed 0. Both corpora go through the identical instrument and the identical **stock**
`sd-vae-ft-mse` decoder, not either of our LPIPS-tuned ones, so the two columns are comparable.

- **Theirs**: `arnaudstiegler/vizdoom-500-episodes-skipframe-4-lvl5`, already on Spiderman as
  `$D/raw_stiegler` (1,000 episodes, 4.95M tics). No download was needed.
- **Ours**: `$D/raw_arnold_eval/seen`, the seen split of the Arnold evaluation corpus
  (`arnold-eval-seen-v1`, 60 episodes).

## Floor and ceiling

| | theirs | ours |
|---|---|---|
| persistence PSNR, gap 1 tic | **23.79** +/- 0.05 | **21.59** +/- 0.06 |
| persistence PSNR, gap 2 tics | 22.65 +/- 0.04 | 20.20 +/- 0.04 |
| persistence PSNR, gap 4 tics (the decision spacing) | **21.23** +/- 0.02 | **18.94** +/- 0.03 |
| persistence PSNR, gap 8 tics | 20.25 +/- 0.02 | 18.12 +/- 0.03 |
| near-static frames (1-tic pair above 35 dB) | 0.81% | 1.47% |
| SD 1.x autoencoder ceiling, PSNR | **24.375** +/- 0.004 | **23.643** +/- 0.007 |
| SD 1.x autoencoder ceiling, LPIPS | 0.0952 | 0.0974 |
| **headroom at the decision spacing** (ceiling − floor) | **3.14 dB** | **4.71 dB** |

## What it says

**Their footage is easier, by 2.3 dB at the spacing both corpora are predicted at.** Copying the
previous decision frame already scores 21.23 dB on their data and 18.94 dB on ours. That is a
difficulty gap, not a modelling result, and it transfers straight into any headline PSNR.

**Both autoencoder ceilings are low, and theirs is only 0.7 dB above ours.** The floors differ much
more than the ceilings, which is why the *headroom* runs the other way: a model on their footage has
3.15 dB to win between persistence and a perfect autoencoder, and one on ours has 4.70 dB. Easier
footage is not more room; it is less.

**GameNGen's 29.43 dB is above the stock ceiling of this latent space.** We measure the stock SD 1.x
autoencoder at 24.38 dB on their own footage, so 29.4 dB is not reachable in the stock space at all,
by any dynamics model. It requires a fine-tuned decoder, which GameNGen used and we use too (ours
lifts 23.7 to 29.1 dB on our footage). The decoder, not the dynamics model, decides whether a
29 dB headline is even available — which is worth stating plainly wherever our numbers sit next to
GameNGen's.

**Caveat, and it is the important one.** GameNGen's own footage has never been published. Everything
above is the open reproduction's dataset, which may be easier or harder than what GameNGen trained
on. Nothing here measures GameNGen.

## Row semantics, checked from the data rather than the card

Their dataset card and our Sep 10 note both say rows are every tic with the action repeated for 4
tics. That is exactly what the columns show, and more strictly than claimed:

| | theirs | ours |
|---|---|---|
| consecutive rows 1 tic apart | 197,634 of 197,634 (100%) | 197,419 of 197,419 (100%) |
| constant-action runs, total | 44,616 | 30,841 |
| of length 4 | 40,329 | 21,888 |
| of length 8 (two equal consecutive decisions) | 3,759 | 5,084 |
| of length 10 or more | 489 | 3,629 |
| of some other length under 10 | 39 | 240 |
| run starts on tic % 4 == 0 | **44,616 of 44,616 (100%)** | **11,427 of 30,841 (37%)** |

**Theirs is a strict 4-tic decision grid with no exceptions.** Every action change without exception
lands on a tic divisible by 4. The 39 runs whose length is not a multiple of 4 are the truncated
final run of each of the 40 sampled episodes, which is the only way such a run can arise once the
phase never slips.

**Ours is per tic too, but its decision phase drifts.** Run starts spread almost evenly over the four
phases (11,427 / 6,586 / 6,584 / 6,244), and 3,629 runs last 10 tics or more, because deaths and
respawns restart the clock and the anti-stuck override runs 40 tics instead of 4. That is precisely
why `encode_parquet.py --align-decisions` exists, and why a plain `tic % 4 == 0` filter is right on
their corpus and wrong on ours.

## Provenance

Their frames are JPEG on Hugging Face; `$D/raw_stiegler` holds a lossless PNG transcode of the
decoded JPEG (both corpora verified PNG, magic `89504e47`, 320x240). The compression artifacts are
therefore baked into the pixels we measured, which is correct: it is what a model trained on that
dataset actually sees. Nothing further was lost in the transcode.

Raw reports: `stats_repro.json`, `stats_ours_seen.json`.
