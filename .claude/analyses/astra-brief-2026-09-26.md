# Brief for Astra, Sep 26 2026 (Codex credits return 13:28 EDT)

Repo: /Users/rohan/Documents/Github/Doom (main). Start with RESEARCH_CONTEXT.md section 0 (as of Sep 25), then the "what changed" entries from 2026-09-24 22:20 EDT to now. Deadline: CoRL 2026 PhysWM workshop, Sep 30 AoE, 4 pages; draft to the advisor Sunday Sep 27 evening.

You are the second engineer. Propose before you critique; argue back with lines, files and numbers; recompute from the frozen files rather than trusting the prose.

## 1. The map-distance generalization study (design, result, and one ruling)

Design memo: .claude/analyses/distance-study-design-2026-09-24.md (sections 1 to 6; section 7 is the run order). Code: distance_study.py, paper/make_distance_figure.py, scripts/spiderman/score_distance_maps.sh. Frozen results: results/distance_study/ (distances_{sd1,pixels,sd35}.json, per-episode CSVs, bootstrap npz, splits/, scores/040-unet-nexttic_snap_0200000_ema_ddim10/*_h{1,4}/metrics.json, figure_unet_h1/stats.json, distance_table.md, distance_gain.png). Literature: .claude/analyses/distance-study-literature-2026-09-25.md.

Result: U-Net 200k EMA over 30 maps, outcome = mean(psnr_raw - persist_psnr_raw), covariate = persistence PSNR; partial Spearman -0.734 [-0.840, -0.320] at one tic, -0.795 [-0.875, -0.458] at four tics, permutation p 1e-4, leave-one-map-out all same sign, within arenas -0.57 (p 0.017), within campaign -0.38 (n.s.), LPIPS gain -0.59, episode-within-map rho 0.09 (no relation); replicated on pixel distances (-0.66) and SD 3.5 latent distances (-0.73); the three spaces rank the maps alike (Spearman 0.86 to 0.90).

The ruling we need: the pre-declared verdict gate (make_distance_figure.py, `verdict`) requires the distances file's checks.all_pass, and two of the five checks fail in every space: (iv) same-WAD arenas 6 to 8 are not all nearer than every campaign map (arena 7 at 0.28 sits above most campaign maps), and (ii) 4-episode disjoint subsets of eight campaign maps disagree by a median 11 to 29 percent (the 10-subset mean and the 200-redraw bootstrap give attenuation ratio 0.08 to 0.09, so map-level D is precise). Main's position: (iv) is an assumption about the maps, not a check of the measurement; (ii) is a subset-level tolerance stricter than the precision the test relies on; (i), (iii), (v) validate the measurement. The gate was not changed after seeing the data. Questions: Should the gate be amended (state the principled criterion, not a post-hoc one)? Is the partial Spearman on persistence PSNR the right confound control, or would you add lives or valid_fraction? Is anything in the design a threat to the claim "gain over persistence falls with distance from the nearest training map"? What would a reviewer attack first?

## 2. The closed-loop stability finding and how to state it

Entries 2026-09-24 15:15 through 2026-09-26 09:00. The live SD 3.5 weights collapse intermittently under autoregression (50k and 70k: 8 to 11 of 16 rollouts blank; mechanism: one latent channel's per-frame mean captured at a fixed value, RMS in scale); the EMA has a lower rate but is not clear: rates 50k to 125k, live 17.6 / 13.6 / 8.8 percent (frame under 10 dB / ends under 12 dB / channel-13 end state) over 272 rollouts, EMA 6.2 / 2.0 / 1.6 over 304; the EMA rate roughly doubled from 105k on (4.4 to 8.3 percent) while teacher-forced PSNR improves every read and the directional check is steady; events concentrate on validation maps 4 and 5. Data: $D/tmp/steward/collapse_rates.md on Spiderman (ask main to relay), per-read entries in the log. Questions: how to state this for the paper (a rate per window? per checkpoint?); should the released SD 3.5 weights be chosen by validation-rollout stability rather than the 200k step (the rollouts are on validation windows), with the matched table staying at 200k; what would make the claim "EMA reduces closed-loop failure" robust to a reviewer (seeds, windows, confidence intervals on a rate difference)?

## 3. The paper spine (4 pages)

Section 0 lists it. Question: given the numbers as they stand, what is the strongest honest four-page paper, and what would you cut? Rows differ by about 0.7 dB; the U-Net is below persistence on most unseen maps at one tic; the distance result is the strongest quantitative finding; the persistence reference is the methodological point.

## 4. Smaller

- smoke_probe.py's max-abs sensitivity to the all-bits-flipped token oscillates with a 10k period (high at 65k, 75k, 85k, 95k, 115k, 125k; low at 60k, 70k, 90k, 100k, 110k, 120k; exceptions 80k, 105k); both probed on recovery files. Any mechanism you can think of worth checking after the runs?
- The requested-action ablation was dropped (executed and requested differ on 11 percent of tics, 0.15 percent true overrides). Agree?

Do not touch any server. Report: your independent proposal for 1 and 2 first, then the critique, then what you would change in the code or memo, with file and line.
