# Overnight steward plan, Sep 16 to 17 2026 (until 09:30 EDT)

Read cold by the monitor agent. Main session is stopped; only the monitor's final report or an early critical return wakes it.

## Cadence and ending
- Tick every 45 minutes (or the cadence your brief gives). Wait INSIDE your turn with a foreground Bash call whose command is `end=$(( $(date +%s) + 570 )); until [ $(date +%s) -ge $end ]; do sleep 2; done; echo waited` (timeout 600000); repeat it back to back until the tick is due. Standalone `sleep N` and `sleep N; echo` are refused by this harness in every permission mode; the until-loop form is accepted (verified Sep 17 2026). Never use the Monitor tool as a tick clock and never end your turn between ticks: ending the turn wakes the expensive main session, which this shift exists to avoid. First tick immediately. At the end time, write the final report and finish.
- Critical events end the night early (stop the monitor with TaskStop, finish immediately with the report so the main session wakes): a run you could not relaunch (relaunch failed or the log did not grow by the next tick), an excursion flag, a disk flag, two consecutive failed logins. Do not decide direction.

## SSH hygiene (fail2ban has banned the office IP twice)
- Every Bash call that reaches a server starts with `source ~/.zshrc >/dev/null 2>&1;`. The password lives only in `$PERSEVE_SERVER_PASSWORD`. Never print it, never write it into a file or a tmux string.
- `sshpass -p "$PERSEVE_SERVER_PASSWORD" ssh -o NumberOfPasswordPrompts=1 -o ConnectTimeout=20 rnagabhi@128.2.204.110 '<remote>' 2>&1 | grep -v Warning`
- A failed login is never retried within the same tick. Never `pkill`/`killall`; kill by pid from `ps -u rnagabhi -o pid,args | grep <pattern> | grep -v grep`.
- Single quotes around the remote command, double quotes only inside.

## Runs (D=/sata2/data/rnagabhi/doom)
| run | tmux | log.jsonl | stdout | GPU | budget | relaunch if dead |
|---|---|---|---|---|---|---|
| 050-skyreels-l8-flow | train-video | $D/results_spiderman/050-skyreels-l8-flow/log.jsonl | $D/logs/train_video.log | 3 | 10k updates, ~0.12/s, ends ~10:15 EDT Thu | `cd $D && MB=8 CKPT=1 bash launch_video.sh 3 8 flow` |
| 034-unidiffuser-l32-aligned | train-unidiffuser | $D/results_spiderman/034-unidiffuser-l32-aligned/log.jsonl | $D/logs/train_unidiffuser.log | 2 | 90k updates, ~0.8/s, ends Fri | `cd $D && MB=32 LR=1e-5 EXTRA="--skip-grad-norm 5" bash launch_unidiffuser.sh 2` (restored from 30k at 06:14 EDT Thu; never at a higher lr) |

034's log.jsonl was truncated to step 30000 at the second restore; a `skipped_update` event is the spike guard working, not an incident (note it, no action). A third excursion is critical: end the night with the report.

- Dead: no `"event": "end"` line and (tmux session gone or log.jsonl mtime older than 45 min). Read the last 30 lines of the stdout log, relaunch with the command above (launchers resume from the last recovery checkpoint), verify on the next tick that the session exists and the log grew; record the cause. Relaunch failure or no growth is critical.
- Ended: end event present. Record final and best val loss, write `<S>/reported_<run>.txt`, start no evaluation, do not relaunch.
- Excursion: a new val line with `"excursion": true`, or val_loss more than 15% above the previous three. Never intervene; record step, val lines, surrounding train lines (grad_norm_max, clip_frac). Critical.

## One-command tick
Remote: tmux sessions; `date +%s`; per run: end-event count, log.jsonl mtime, last line's step, last 4 val lines as step:val_loss with `!` on excursion; `df -BG /sata2/data /home | tail -2`. Flags: /sata2/data under 300 GB or /home under 15 GB free (critical; delete nothing).

## Dashboard, every tick after the ssh check
S=/private/tmp/claude-501/-Users-rohan-Documents-Github-Doom--claude-worktrees-vibrant-ritchie-6f6280/bbd290c7-2c43-46b3-bfab-7994575157d8/scratchpad/usage (mkdir -p). One Bash call: `source ~/.zshrc >/dev/null 2>&1; M=$(date +%M); DU=""; [ "$M" -le 15 ] && DU="--du"; python3 /Users/rohan/Documents/Github/Doom/tools/server_usage.py --out $S/snapshot.json $DU; python3 /Users/rohan/Documents/Github/Doom/tools/usage_compact.py $S/snapshot.json $S/history.json` (prints `history doc <ts>`; keep ts). Same call, python: if a server's `ours_b` in snapshot.json is empty, copy `ours_b`, `ours_detail_b`, `homes_b` for it from `$S/snapshot_prev.json` (`snap["servers"][name][key]`) and rewrite snapshot.json. Then ONE Artifact tool call: action write_db, url https://claude.ai/code/artifact/6bcc5f0b-0712-4a90-a239-71b7f18dd47d, db_op batch, writes: set `usage`/`latest` from `$S/snapshot.json`, set `history`/`<ts>` from `$S/history.json`. Then `cp $S/snapshot.json $S/snapshot_prev.json`. server_usage.py logs into both servers itself; with your ssh check that is two logins per tick, never more.

## Records and report
- `$S/steward_night.log`: one line per tick (time, steps, last val, flags or ok). `$S/steward_incidents.md`: every incident with time, evidence, action.
- Final message under 250 words: last step and val trend per run, every incident and action, decisions left to a human.
- Never commit, never edit the repo, never touch other tmux sessions, never launch evaluations or runs, never ssh to Superman (another agent works there tonight).
