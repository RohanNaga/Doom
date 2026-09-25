# Weekly update to Changliu, Friday Sep 25 2026 (draft)

Hi Changliu,

Weekly update on goal 1, the Doom world-model paper. Everything below is tagged to that goal; goal 3 (LEGO) had no work this week.

**Data.** The dense corpus is done and public. We record every game tic (35 Hz) instead of every fourth, which was the frame-spacing mismatch with GameNGen we found last Friday. 6,000 training episodes on four arena maps are encoded in both latent spaces (SD 1.4, 4 channels, and SD 3.5, 16 channels) and verified on Hugging Face. We also recorded a seeded evaluation corpus of 210 episodes on 30 distinct maps: held-out episodes of the four training maps, 13 other arenas from the same WAD, and 13 campaign maps. Map selection was fixed before any model result on it.

**Three rows train with one recipe.** 32 tics of context, the executed control history (what the engine actually applied, not what the agent requested), v-prediction, and a persistence baseline computed on the same windows. The rows differ only in backbone: SD 1.4 U-Net, SD 3.5 medium (DiT), and PixArt-alpha (DiT). All three stop at 200k updates so the comparison is matched; the U-Net is done, SD 3.5 finishes Sunday evening, PixArt Saturday. Every run streams to Weights & Biases and is evaluated every 5k steps on a separate card.

**Results so far, one tic ahead, PSNR / LPIPS on held-out episodes of the training maps** (copy-last-frame persistence is 21.57 / 0.203):
- U-Net at 200k: 22.49 / 0.177, +0.9 dB over persistence; +2.1 dB at four tics.
- SD 3.5 at 90k: 23.00 / 0.144, +1.4 dB; +2.1 dB at four tics. It passed the U-Net's final numbers at 40k.
- PixArt at 75k: +0.6 dB, tracking the U-Net's curve.
256-tic autoregressive rollouts of the SD 3.5 EMA weights stay above the copy-seed baseline at every horizon.

**Two findings about training.** (1) The live weights of the SD 3.5 row have an absorbing failure under autoregression that recurs intermittently (50k and 70k): one latent channel gets captured at a fixed value and the frames go blank, while teacher-forced quality keeps improving. The fp32 EMA weights never fall in. So EMA is a stability mechanism for closed-loop generation, not just a smoother. (2) A directional check on real turning windows: swapping TURN_LEFT and TURN_RIGHT in the newest control reverses the predicted camera motion in 87% of windows for the U-Net and 83% for SD 3.5, with the right magnitude, and neither model learned persistence.

**The generalization study is in.** Following the NVIDIA latent-distance method from my internship, each map's footage is a motion-weighted cloud of frame latents and its distance is the sliced Wasserstein distance to the nearest training map. The outcome is the model's PSNR gain over persistence on that map. Over the 30 maps, the U-Net's gain falls with distance: partial Spearman -0.73, 95% CI [-0.84, -0.33], controlling for each map's motion level; leave-one-map-out keeps the sign on all 30 maps, and the relation lives between maps, not between episodes of one map. The honest part: at one tic the U-Net beats persistence only on the training maps and their nearest neighbours. On most unseen maps it is below copy-last-frame, by up to 4 dB on the campaign maps. The pre-registered test and the distances were frozen before any map was scored. SD 3.5 and PixArt get the same scoring at their final checkpoints.

**Literature.** A survey this week found no prior work correlating training-to-test distribution distance with a generative world model's quality across held-out scenes; the closest templates are transfer-learning papers (OTDD, Deng and Zheng 2021). Copy-last-frame baselines were routine in 2014 to 2019 video prediction and are absent from GameNGen, DIAMOND, Genie, Oasis and their successors. Reporting gain over persistence is the paper's methodological point.

**Plan.** Saturday: score the SD 3.5 and PixArt rows, finish the four-tic and rollout tables, and write the draft. Sunday evening: the full draft to you, with the SD 3.5 numbers marked provisional where its run has not yet reached 200k. Monday and Tuesday: final tables and figures. Wednesday Sep 30: CoRL PhysWM workshop submission.

Rohan
