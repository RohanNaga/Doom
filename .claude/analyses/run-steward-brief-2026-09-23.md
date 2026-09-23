# Run steward brief, Sep 23 2026: the next-tic rows on Spiderman

For the opus-monitor that watches `040-unet-nexttic` (GPU 1, tmux `train-unet-nexttic`) and, once launched, `042-sd35-nexttic` (GPU 2, tmux `train-sd35-nexttic`). `$D=/sata2/data/rnagabhi/doom`; code `$D/repo_launch` (main, the commit in `$D/GATES_CERT.json`); run dirs `$D/results_spiderman/<run>/` (`log.jsonl`, recovery checkpoints `NNNNNNN.pt` every 5k keeping two, snapshots `snap_NNNNNNN.pt` every 10k never pruned, `best.pt`); train logs `$D/logs/train_<run>.log`; launches and resumes `$D/logs/resumes.log`. Interpreters: `~/miniconda3/envs/doom/bin/python` for the U-Net, `~/wanenc/bin/python` for SD 3.5. Server clock is UTC.

## Rules that override everything
- Stop a job only with `tmux kill-session -t <session>`; never `pkill`/`pgrep -f` on a pattern (the tmux server carries the command string and a pattern match kills every session).
- Never `git pull`, checkout or edit anything under `$D/repo_launch` or `$D/repo`; the certificate pins the commit and a resume refuses a changed checkout.
- Never write into `$D/latents_arnold_dense_pertic*/` (the certificate pins every shard log there). Superman's 2000:6000 encode goes to `$D/ext_2000_6000/`.
- Never launch with `ALLOW_UNGATED=1`, and never change MB, WORKERS, STEPS, TRAIN_IDS, VAL_IDS or EXTRA: the only valid launch is the `GATES_LAUNCH <backbone>:` line printed at GATES_GO (also in `$D/GATES.txt`), which resumes from the newest recovery checkpoint by itself.
- Relaunch (with that exact line) only for an environment failure: the process died with the corpus, code and certificate intact (host reboot, another user's job took the memory, a transient I/O error). Relaunch at most once per failure and report. Anything that looks like a code or data defect: stop, keep the checkpoint, report to main; do not diagnose code.
- One server-touching session at a time besides main: batch your ssh into few commands per tick, no retry loops; on a connect timeout stop and report.
- Password only via `$PERSEVE_SERVER_PASSWORD` after `source ~/.zshrc`; never print it.

## Each tick (every 30 min for the first 3 hours, then hourly)
From `log.jsonl` (`train` events every 100 steps, `val` every 1k):
- validation loss overall and by t quartile against the planning bands (forecasts, not thresholds): U-Net about 0.30 / 0.24 / 0.225 / 0.21 at 1k / 5k / 10k / 20k; SD 3.5 about 0.19 / 0.14 / 0.128 / 0.118; flag an `excursion: true` or a rise over two validations;
- `grad_norm`, `grad_norm_max`, `clip_frac`, `nonfinite_loss`, `skipped_updates` (finite and non-finite separately), `steps_per_s` (U-Net measured 1.55 at the smoke; SD 3.5 about 0.5), `peak_mem_gb`, data-wait fraction if logged;
- `nvidia-smi` for both cards, `df -h /sata2/data` (report if under 400 GB; other users are consuming about 65 GB/h), host memory, tmux sessions, the tail of the train log for tracebacks, `resumes.log` for unexpected relaunches;
- Superman: `bash /home/rohan/Doom/sd35enc/status.sh` and the tail of `/home/rohan/Doom/logs/enc-orch2.log` (phase 2 writes under `$D/ext_2000_6000/`).
Write the tick to `$D/tmp/steward/run_status.md` (overwrite; keep a one-line history at the end).

## At 5k (recovery checkpoint), 10k and 20k (snapshots)
Run these on the run's own card from `$D/repo_launch` with the run's interpreter, live and EMA from the same saved file, into `$D/results_spiderman/<run>/steward_<step>/`:
1. `eval_tf.py --tic-stride 1 --horizon-tics 1` (and 4) on 512 val windows with the stock decoder, exactly the gate-5 readback command in `$D/GATES.txt` with `--num-windows 512` and `--out-dir` changed; report raw PSNR and LPIPS beside `persist_psnr_raw` (the copy floor) and the reconstruction number in the same metrics file.
2. `smoke_probe.py` on the recovery checkpoint (control sensitivity under fixed noise; the gate-4b command in `$D/GATES.txt` with the checkpoint path changed).
3. The directional check (swap TURN_LEFT and TURN_RIGHT on turning windows; the predicted frame must shift the opposite way) and the 16-rollout x 256-tic val rollout with copy-seed and motion ratio (`rollout_eval.py`): run them only if a script in `$D/repo_launch` implements them (look before running; `grep -l TURN_LEFT *.py`); if none does, say so in the report rather than improvising.
4. At 10k: the paired 4/8/16/50-step sweep with both spacings (`--sampler ddim`, review M1) if `eval_tf.py` exposes it; otherwise report that it is pending.
Then mirror the 10k and 20k snapshots (`snap_*.pt`) and `config.json` to the Hugging Face model repo `RohanNaga/doomdit-nexttic` under `<run>/` with `hf upload` from Spiderman (the CLI is logged in), and verify with `hf` that the files are listed.

## Stop and hold rules
- Contract fault (a traceback naming the corpus, the certificate refusing a resume, NaN loss with skips not recovering within 200 steps): stop the session, keep everything, report.
- Still at least 0.3 dB below persistence PSNR at 20k with under 0.1 dB improvement between 10k and 20k, after the evaluator and decoder checks above: hold (do not stop) and report with the numbers.
- Do not stop a healthy run for any other reason; the run is stopped at the numbers freeze by main.

## Reporting
Message main only on: a crash or relaunch, a validation excursion, skip bursts, throughput down more than 20 percent, disk under 400 GB, each 5k/10k/20k read, and a summary at 09:00 EDT and then every 3 hours. Keep `$D/tmp/steward/run_status.md` current for Rohan.
