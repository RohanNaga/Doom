# Kickoff for the Fall 2026 working chat (paste as the first message: "Follow .claude/KICKOFF.md")

This is the working chat for DoomDiT for Fall 2026. Before anything else, read `RESEARCH_CONTEXT.md` (section 0: a workshop paper in three to four weeks, leaning toward the CoRL 2026 PhysWM workshop, Sep 30 AoE, 4 pages), then `.claude/analyses/DOOMDIT_AGENT_CONTEXT.md` and `.claude/analyses/doomdit-full-recap-2026-07.md`. Do not re-derive history; those files are authoritative.

Then, discussion-first:
1. Confirm the deadline (Sep 30 CoRL PhysWM vs the alternatives in RESEARCH_CONTEXT.md section 5) and write the experiment list backwards from it into RESEARCH_CONTEXT.md.
2. Recover the missing artifacts: inventory `/home/rohan/Doom/Doom/` on Superman (through `/run-server`) for runs 004/012, any U-Net baseline directories, and any PSNR/LPIPS scripts; draft the message to Keerthana asking for the frame-encoding script, U-Net code, and harness. If the baseline cannot be recovered within a few days, plan a matched-params U-Net retrain from the documented recipe.
3. Fix `eval_checkpoint.py` (undefined `ema_state`), define a held-out episode split, rebuild the PSNR/LPIPS harness, and reproduce the 26.04 dB / 0.153 LPIPS headline on the current checkpoint before anything else is trained.
4. Design the autoregressive evaluation suite (FVD vs rollout length, drift-vs-horizon curves, a small VizDoom inverse-dynamics model for action-following accuracy). Present the design before coding.
5. Size the final larger run from the gap analysis and launch it only if it finishes before the deadline; otherwise the paper uses the current checkpoint plus the new metrics.

Superman rules: leave 2 GPUs free, highest-numbered free GPUs first, keep 250 GB disk free, EMA math in fp32, respect the 16 GB VRAM budget. 1-on-1 with Changliu Wednesdays 1:00 to 1:30 PM.
