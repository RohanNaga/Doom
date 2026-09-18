# Overnight plan, Thu Sep 17 to Fri Sep 18 2026 (report at 09:30 EDT)

Read cold by the dashboard-feeder monitor. Everything in `overnight-plan-2026-09-16.md` still applies (ssh hygiene, one ssh per tick, dashboard procedure, record files) except as changed here. D=/sata2/data/rnagabhi/doom, S=the scratchpad usage directory named in that plan. Ticks every 30 minutes with the existing clock; final report at the STOP line or at 09:30 EDT, whichever first (end the clock at 09:30 with TaskStop if it is still running).

## GPU 2 lane
1. 034-unidiffuser-l32-aligned trains to 90k (ends about 01:00 EDT). On its end event: write `$S/reported_034-unidiffuser-l32-aligned.txt`, record final and best val in the tick log, then launch its evaluation:
   `tmux new-session -d -s after-unidiffuser "PY=/home/rnagabhi/wanenc/bin/python bash /sata2/data/rnagabhi/doom/after_run2.sh 034-unidiffuser-l32-aligned unidiffuser 2"`
   (the wanenc python is required: the doom env has no transformers and UniDiffuser cannot build there). Verify on the next tick that the session exists and `$D/logs/034-unidiffuser-l32-aligned_eval_tf_seen.log` grows.
2. Watch `$D/logs/034-unidiffuser-l32-aligned_eval_tf_seen.log`, `_eval_tf_unseen.log`, `_eval_tf_seen_ema.log`, `_rollout.log` for `Traceback`. A traceback is critical: record the last 30 lines of that log in the incident file and end the shift with the report; launch nothing else.
3. When `AFTER_RUN_DONE` appears in `$D/logs/034-unidiffuser-l32-aligned_rollout.log`, launch the map-transfer scoring:
   `tmux new-session -d -s unseen2-034 "PY=/home/rnagabhi/wanenc/bin/python bash /sata2/data/rnagabhi/doom/eval_unseen2.sh 2"`
   Then GPU 2 stays idle until the morning decision; start nothing else on it.

## GPU 3 lane
1. The `vae-gate` agent owns GPU 3 while tmux `vae_gate_flux` or `vae_gate_sd35` exists or until the marker `$D/logs/VAE_GATE_DONE` appears. Never touch those sessions.
2. Once the marker exists and no vae_gate session remains, launch the video row's rollout evaluation (sampler matched to the SD rows: 50 steps, clean context):
   `tmux new-session -d -s rollout-050 "cd /sata2/data/rnagabhi/doom/repo && export TMPDIR=/sata2/data/rnagabhi/doom/tmp/tmpdir HF_HUB_OFFLINE=0 CUDA_VISIBLE_DEVICES=3 && R=/sata2/data/rnagabhi/doom/results_spiderman/050-skyreels-l8-flow && D=/sata2/data/rnagabhi/doom && /home/rnagabhi/wanenc/bin/python eval_video.py --rollout --ckpt \$R/best.pt --hf-cache \$D/hf/hub --null-prompt \$D/weights/skyreels_null_prompt.pt --latents-dir \$D/latents_arnold_wan_eval/seen --parquet-dir \$D/raw_arnold_eval/seen --split \$D/latents_arnold_eval/split_seen.json --subset val --num-rollouts 256 --horizon 64 --steps 50 --ctx-stabilize 0 --batch-size 32 --decode-batch 8 --idm \$D/results_spiderman/idm_aligned/idm.pt --out-dir \$R/rollout_metrics_seen > \$D/logs/050_rollout.log 2>&1; for F in 16 32; do /home/rnagabhi/wanenc/bin/python fvd.py --clips \$R/rollout_metrics_seen/clips_u8.npz --frames \$F --i3d \$D/weights/i3d_torchscript.pt --out \$R/rollout_metrics_seen/fvd\$F.json >> \$D/logs/050_rollout.log 2>&1; done; echo ROLLOUT_DONE >> \$D/logs/050_rollout.log"`
   Watch `$D/logs/050_rollout.log` for `Traceback` (critical, as above) and record its progress lines in the tick log. It may still be running at 09:30; that is fine.
3. If the gate marker never appears by 09:30, say so in the report; do not launch the rollouts on a busy card.

## Records and report
Tick log and incident file as before. Final report under 250 words: 034's final and best val; what was launched, when, and whether its logs grew; the last progress line of every launched job; every incident; anything left for the morning decision (the gate result belongs to the vae-gate agent; just report whether its marker exists).
