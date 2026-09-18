# Steward plan, Fri Sep 18 evening to Sat Sep 19 09:30 EDT

Read cold by the monitor agent. Everything in `overnight-plan-2026-09-16.md` still applies (ssh hygiene, one-command tick, dashboard procedure, record files, the in-turn wait method with the until-loop) except as changed here. D=/sata2/data/rnagabhi/doom. Ticks every 45 minutes inside one continuous turn; final report at 09:30 EDT or on a critical event.

## New ssh rules (CMU's edge blocked our IP twice for connection bursts)
- The lab servers are multiplexed via `~/.ssh/config`: keep the plain `sshpass ... ssh` pattern, it reuses one open connection. One ssh per tick plus the dashboard collector, never more. Never retry a failed connection inside a tick; a connect timeout is "link down": log it, skip the dashboard, wait for the next tick, and only after THREE consecutive ticks of connect timeouts end the shift with the report.

## Jobs to keep alive (all in tmux on Spiderman)
| job | tmux | log.jsonl / log | GPU | relaunch if dead |
|---|---|---|---|---|
| 035-sd35-l32-aligned (SD 3.5 row, 90k updates, ends about Sun 19:30 EDT) | train-sd35 | $D/results_spiderman/035-sd35-l32-aligned/log.jsonl; stdout $D/logs/train_sd35.log | 3 | `cd $D/repo && MB=32 bash scripts/spiderman/launch_sd35.sh 3` (resumes from the last recovery checkpoint; spike guard is built in) |
| knob grid queue (12 cells at 30k updates, about 10 h each) | grid-2 | $D/logs/grid_queue_2.log; per cell $D/logs/grid_<cell>.log and $D/results_spiderman/grid/<cell>/log.jsonl | 2 | `cd $D/repo && bash scripts/spiderman/grid_queue.sh 2 base30k scratch noaug data-1_4 ctx8 ctx16 eps lr1e-4 adaln data-1_2 lr2.5e-5 data-1_8` (skips finished cells, resumes the current one) |
| SD 3.5 evaluation waiter | after-sd35 | $D/logs/after_sd35_waiter.log | 3 (after the row ends) | `tmux new-session -d -s after-sd35 "bash $D/repo/scripts/spiderman/after_sd35.sh 3 > $D/logs/after_sd35_waiter.log 2>&1"` |

- Dead = tmux session gone while the job's log has no end event (for the grid: while `CELL_DONE` for the last cell is absent from $D/logs/grid_queue_2.log) or the log.jsonl mtime is older than 45 minutes. Read the last 30 lines of the stdout log, relaunch with the command above, verify on the next tick, record the cause. A relaunch that fails or a log that does not grow is critical.
- The row's log.jsonl currently reads: resumed at 4000, about 0.46 updates/s, val every 1k. A `skipped_update` event is the spike guard working (note, no action). Excursion (a val line flagged, or val more than 15% above the previous three) is critical: record the val lines and the surrounding train lines, intervene in nothing, end the shift.
- Grid cells: each cell's `train` lines and `val` lines live in its own log.jsonl; a cell that fails is logged in grid_queue_2.log and the queue moves on; note it, no action. Report each completed cell's final val loss.

## One-command tick
Remote: `date +%s`; tmux sessions; for the row and the current grid cell: end-event count, log.jsonl mtime, last step, last 3 val lines; `tail -2 $D/logs/grid_queue_2.log`; `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`; `df -BG /sata2/data /home | tail -2`. Flags: /sata2/data under 300 GB or /home under 15 GB free (critical, delete nothing).

## Dashboard
As in the older plan (server_usage.py, usage_compact.py, one write_db batch with `usage/latest` and `history/<ts>`, then roll snapshot_prev). If a write is refused for a version, read the document back and resend once with the returned version; otherwise skip that tick's dashboard and note it.

## Records and report
Tick log `<S>/steward_night.log`, incidents `<S>/steward_incidents.md` (S = the scratchpad usage directory named in the older plan). Final report under 250 words: the row's last step and val trend, grid cells completed with their final val, every incident and action, anything for a human. Never commit, never edit the repo, never launch evaluations or new runs beyond the relaunch commands above, never touch other tmux sessions.
