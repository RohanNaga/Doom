"""The trainer's periodic evaluation: one detached read of the checkpoint just written, never a stall.

`train_wm.py --eval-every N` calls `PeriodicEval.launch(step)` at every multiple of N, right after
the checkpoint for that step is on disk. The read is ONE child process, detached from the trainer
(`start_new_session=True`, so it also survives a `tmux kill-session` of the training session), that
runs, in `<results>/eval_<step>/`:

    eval_tf.py     live and EMA weights, horizons 1 and 4 tics   tf_<live|ema>_h<H>/metrics.json
    smoke_probe.py the conditioning pathways                      probe.json

each with `--wandb-run <run>-eval --wandb-step <step>` when the trainer streams to W&B. The commands
are written to `eval_<step>/run.sh` (rerun it by hand with `bash run.sh`) and the child's output
goes to `eval_<step>/launch.log`.

The trainer never waits for the child and never fails because of it: every problem (no checkpoint
file, no split file, a spawn that fails) becomes an `eval_skipped` event with the reason, and a
second read is refused while the previous one of this run is alive, so reads never pile up on the
evaluation card. `eval_launched` records the child's pid. The wrapper writes each command's exit
code to `eval_<step>/status.json` and exits non-zero if any failed; the trainer records
`eval_finished` (its exit code, `ok`, the per-command codes) the next time a read is due, by
polling, never by waiting.

The checkpoint read is the compact snapshot (`snap_<step>.pt`, bf16 live and EMA, never pruned)
when the step is also a snapshot step, else the recovery checkpoint (`<step>.pt`). The read opens
a hard link to it in `eval_<step>/` (a copy where no link can be made), because the trainer prunes
recovery checkpoints while a slow read may still be on one; see `pin_checkpoint`.
"""
import json
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PID_FILE = "eval_running.pid"
HORIZONS = (1, 4)
DEVICE = re.compile(r"^(?:cuda:(\d+)|cpu)$")
# eval_tf.py's flag for where each backbone's architecture is rebuilt from (its `backbone_source`)
SOURCE_FLAG = {"unet": "--sd-path", "pixart": "--pixart-path", "unidiffuser": "--unidiffuser-path",
               "sd35": "--sd35-path"}


def _popen(argv, **kw):
    return subprocess.Popen(argv, **kw)


def cadence_problem(args):
    """Why `--eval-every` / `--eval-device` cannot work with this run's checkpoints, or None.

    A read needs a checkpoint of that exact step on THIS disk: a recovery checkpoint (a multiple of
    `--ckpt-every`, kept locally when `--keep-last` > 0) or a local snapshot (a multiple of
    `--snapshot-every` that is also a validation step, with `--local-snapshots`).
    """
    n = int(getattr(args, "eval_every", 0) or 0)
    dev = getattr(args, "eval_device", None)
    if dev and not DEVICE.match(dev):
        return f"--eval-device {dev!r} must be cuda:<K> (the physical card K) or cpu"
    if n < 0:
        return f"--eval-every {n}: must be 0 (off) or a positive number of training steps"
    if n == 0:
        return None
    on_recovery = n % args.ckpt_every == 0 and args.keep_last > 0
    on_snapshot = args.local_snapshots and n % args.snapshot_every == 0 and n % args.val_every == 0
    if on_recovery or on_snapshot:
        return None
    if n % args.ckpt_every == 0:
        return (f"--eval-every {n} lands on recovery checkpoints, but --keep-last 0 keeps none on this disk; "
                "keep at least one, or use a multiple of --snapshot-every with --local-snapshots")
    if n % args.snapshot_every == 0:
        return (f"--eval-every {n} lands on snapshot steps, but snapshots are written here only with "
                f"--local-snapshots and at validation steps (--val-every {args.val_every})")
    return (f"--eval-every {n} must be a positive multiple of --ckpt-every {args.ckpt_every} or of "
            f"--snapshot-every {args.snapshot_every}: the read evaluates the checkpoint written at that step")


def default_split(latents_dir):
    """The canonical split file of an evaluation corpus: `<parent>/split_<corpus>.json`."""
    from make_dense_eval_splits import split_path
    return split_path(latents_dir)


def eval_latents(args):
    return args.eval_latents_dir or args.val_latents_dir or args.latents_dir


def eval_split(args):
    return args.eval_split or default_split(eval_latents(args))


def eval_dir(results_dir, step):
    return os.path.join(results_dir, f"eval_{step:07d}")


def checkpoint_for(args, step):
    """The file a read of `step` evaluates: the snapshot on a snapshot step, else the recovery checkpoint."""
    snap = os.path.join(args.results_dir, f"snap_{step:07d}.pt")
    if args.local_snapshots and step % args.snapshot_every == 0 and os.path.isfile(snap):
        return snap
    rec = os.path.join(args.results_dir, f"{step:07d}.pt")
    return rec if os.path.isfile(rec) else None


def child_device(spec, train_device, env):
    """(CUDA_VISIBLE_DEVICES for the child, the device it addresses).

    `cuda:K` is the PHYSICAL card K, whatever the trainer's own CUDA_VISIBLE_DEVICES says; the child
    sees only that card, as cuda:0. The default is the trainer's own card.
    """
    if spec == "cpu":
        return "", "cpu"
    if spec:
        return DEVICE.match(spec).group(1), "cuda:0"
    if getattr(train_device, "type", "cpu") != "cuda":
        return "", "cpu"
    idx = train_device.index or 0
    visible = [v.strip() for v in (env.get("CUDA_VISIBLE_DEVICES") or "").split(",") if v.strip()]
    return (visible[idx] if idx < len(visible) else str(idx)), "cuda:0"


def decoder_flags(latent_channels):
    """eval_tf.py's decoder flags for the latent space the run trained in (check_latent_alignment's contract)."""
    from check_latent_alignment import space_contracts
    c = space_contracts()["sd35" if int(latent_channels) == 16 else "sd15"]
    out = ["--vae-path", c["vae_id"]]
    if c["vae_subfolder"]:
        out += ["--vae-subfolder", c["vae_subfolder"]]
    out += ["--latent-scale", str(c["scale"])]
    if c["shift"] is not None:
        out += ["--latent-shift", str(c["shift"])]
    return out


def probe_episodes(args, split):
    if args.val_episode_ids:
        return args.val_episode_ids
    with open(split) as f:
        return ",".join(str(int(e)) for e in json.load(f)["val"])


def commands(args, step, ckpt, latent_channels, wandb_run, device, python=sys.executable):
    """[(label, argv)] of one read: eval_tf.py at each horizon, live then EMA, then smoke_probe.py."""
    out = eval_dir(args.results_dir, step)
    split = eval_split(args)
    model = ["--backbone", args.backbone, "--latent-channels", str(latent_channels),
             "--context-frames", str(args.context_frames), "--num-actions", str(args.num_actions),
             "--noise-buckets", str(args.noise_buckets)]
    if args.backbone in SOURCE_FLAG and args.warm_start and args.warm_start != "none":
        model += [SOURCE_FLAG[args.backbone], args.warm_start]
    if args.hf_cache:
        model += ["--hf-cache", args.hf_cache]
    wb = []
    if wandb_run:
        wb = ["--wandb-run", f"{wandb_run}-eval", "--wandb-project", args.wandb_project, "--wandb-step", str(step)]
        if args.wandb_entity:
            wb += ["--wandb-entity", args.wandb_entity]
    # horizon 4 rolls a next-tic model forward four tics; a stride-4 model has only the one step
    horizons = HORIZONS if args.tic_stride == 1 else (1,)
    cmds = []
    for h in horizons:
        for mode in ("live", "ema"):
            argv = [python, os.path.join(HERE, "eval_tf.py"), "--ckpt", ckpt, *model, "--horizon-tics", str(h),
                    "--latents-dir", eval_latents(args), "--split", split, "--subset", "val",
                    "--num-windows", str(args.eval_windows), "--steps", str(args.eval_steps),
                    *decoder_flags(latent_channels), "--out-dir", os.path.join(out, f"tf_{mode}_h{h}"), *wb]
            if mode == "ema":
                argv.insert(4, "--use-ema")
            if args.eval_parquet_dir:
                argv += ["--parquet-dir", args.eval_parquet_dir]
            cmds.append((f"tf_{mode}_h{h}", argv))
    if args.tic_stride == 1:        # the probe reads per-tic windows
        cmds.append(("probe", [python, os.path.join(HERE, "smoke_probe.py"), "--ckpt", ckpt, *model,
                               "--latents-dir", eval_latents(args), "--episodes", probe_episodes(args, split),
                               "--device", device, "--out", os.path.join(out, "probe.json"), *wb]))
    return cmds


def pin_checkpoint(ckpt, out, snapshot):
    """(the path the read opens, the file to copy there first or None, whether the wrapper removes it).

    The trainer prunes old recovery checkpoints while a read may still be on one (a FileNotFoundError
    in the read), so the read opens a HARD LINK in its own directory: the same inode, no bytes
    copied, and invisible to the pruning, which lists only the top-level `NNNNNNN.pt` files. The
    wrapper removes the link when the read is done, and the disk space goes when both names are gone.
    Where a link cannot be made, a recovery checkpoint is COPIED by the wrapper before the first read,
    not here: the SD 3.5 row's is 35 GB, and the pruning comes two checkpoints later, hours after
    the copy is done. A snapshot is never pruned, so it is then read in place.
    """
    dst = os.path.join(out, os.path.basename(ckpt))
    if os.path.lexists(dst):
        os.remove(dst)                  # a link left by an earlier, interrupted read of this step
    try:
        os.link(ckpt, dst)
        return dst, None, True
    except OSError:
        if snapshot:
            return ckpt, None, False
        return dst, ckpt, True


STATUS_FILE = "status.json"


def write_script(path, run, step, cmds, copy_from=None, pinned=None, remove_pinned=False):
    """`run.sh`: the copy of the checkpoint if one is needed, every read in order (one failing does
    not stop the rest), the removal of the read's own checkpoint link, then `status.json`.

    `status.json` holds each command's exit code and `ok`, and the wrapper exits 1 if any command
    failed, so the trainer can report a failed read (`eval_finished`) instead of assuming it worked.
    """
    status = os.path.join(os.path.dirname(path), STATUS_FILE)
    lines = ["#!/bin/bash", f"# periodic evaluation of {run} at step {step}, launched by train_wm.py --eval-every",
             'CODES=""', "FAILED=0",
             'note() { CODES="$CODES${CODES:+, }\\"$1\\": $2"; [ "$2" -eq 0 ] || FAILED=1; '
             'echo "$(date -Iseconds) exit $2 $1"; }',
             f"cd {shlex.quote(HERE)} || FAILED=1"]
    if copy_from:
        tmp = pinned + ".tmp"
        lines += ['echo "$(date -Iseconds) copying the checkpoint (no hard link possible)"',
                  f"cp {shlex.quote(copy_from)} {shlex.quote(tmp)} && mv {shlex.quote(tmp)} {shlex.quote(pinned)}",
                  "note copy $?"]
    for label, argv in cmds:
        lines += [f'echo "$(date -Iseconds) start {label}"', shlex.join(argv), f"note {label} $?"]
    if remove_pinned:
        lines.append(f"rm -f {shlex.quote(pinned)}")
    ok = '$([ "$FAILED" -eq 0 ] && echo true || echo false)'
    lines += [f"printf '{{\"step\": {int(step)}, \"commands\": {{%s}}, \"ok\": %s}}\\n' \"$CODES\" \"{ok}\" "
              f"> {shlex.quote(status + '.tmp')} && mv {shlex.quote(status + '.tmp')} {shlex.quote(status)}",
              'exit "$FAILED"']
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _alive(pid, script):
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    proc = f"/proc/{int(pid)}/cmdline"
    if os.path.exists(proc):        # guard against a reused pid: it must still be running our script
        try:
            with open(proc, "rb") as f:
                return script.encode() in f.read()
        except OSError:
            return False
    return True


class PeriodicEval:
    """Launches the read of each `--eval-every` multiple; `log` is the trainer's event writer."""

    def __init__(self, args, latent_channels, train_device, log, wandb_run=None, python=sys.executable):
        self.args, self.latent_channels, self.train_device = args, latent_channels, train_device
        self.log, self.wandb_run, self.python = log, wandb_run, python
        self.every = int(args.eval_every or 0)
        self.proc = self.current = None      # the read this process started, and its pid-file record
        self.pid_file = os.path.join(args.results_dir, PID_FILE)

    def due(self, step):
        return self.every > 0 and step > 0 and step % self.every == 0

    def running(self):
        """The pid of this run's read that is still alive, or None. Never waits: a read found ended is
        reported (`eval_finished`) here, which is the next time a read is due."""
        if self.proc is not None:
            code = self.proc.poll()            # also reaps the finished child
            if code is None:
                return self.proc.pid
            self._finished(self.current, code)
            self.proc = self.current = None
            return None
        # no read started by this process: the pid file names one a previous trainer process started
        try:
            with open(self.pid_file) as f:
                rec = json.load(f)
        except (OSError, ValueError):
            return None
        pid = rec.get("pid")
        if pid and _alive(pid, rec.get("script") or ""):
            return pid
        self._finished(rec, None)              # not our child, so its exit code is only in status.json
        return None

    def _finished(self, rec, returncode):
        """Record `eval_finished` for an ended read: its exit code when this process started it, and
        each command's from the wrapper's status.json; a read without one was killed or cut short."""
        rec = rec or {}
        step = rec.get("step")
        out = rec.get("out") or (eval_dir(self.args.results_dir, step) if isinstance(step, int) else None)
        try:
            with open(os.path.join(out, STATUS_FILE)) as f:
                status = json.load(f)
        except (OSError, ValueError, TypeError):
            status = None
        if status is None:
            self.log(event="eval_finished", step=step, returncode=returncode, ok=False, commands=None, out=out,
                     reason=f"no {STATUS_FILE} in {out}: the read was killed, or ended before writing it")
        else:
            self.log(event="eval_finished", step=step, returncode=returncode, out=out,
                     ok=bool(status.get("ok")) and returncode in (None, 0), commands=status.get("commands"))
        try:
            os.remove(self.pid_file)
        except OSError:
            pass

    def launch(self, step):
        """Start the read of `step` and return its pid, or record why not and return None. Never raises."""
        try:
            return self._launch(step)
        except Exception as exc:
            self.log(event="eval_skipped", step=step, reason=f"launch failed: {type(exc).__name__}: {exc}")
            return None

    def _launch(self, step):
        pid = self.running()
        if pid is not None:
            self.log(event="eval_skipped", step=step, reason=f"the previous read (pid {pid}) is still running")
            return None
        ckpt = checkpoint_for(self.args, step)
        if ckpt is None:
            self.log(event="eval_skipped", step=step, reason=f"no snapshot or recovery checkpoint of step {step} "
                                                             f"in {self.args.results_dir}")
            return None
        split = eval_split(self.args)
        if not os.path.isfile(split):
            self.log(event="eval_skipped", step=step, reason=f"no evaluation split file at {split} (--eval-split)")
            return None
        env = {k: v for k, v in os.environ.items() if k != "WANDB_SERVICE"}   # the child opens its own
        visible, device = child_device(self.args.eval_device, self.train_device, env)
        env["CUDA_VISIBLE_DEVICES"] = visible
        out = eval_dir(self.args.results_dir, step)
        os.makedirs(out, exist_ok=True)
        run = os.path.basename(os.path.normpath(self.args.results_dir))
        pinned, copy_from, remove = pin_checkpoint(ckpt, out, os.path.basename(ckpt).startswith("snap_"))
        script = write_script(os.path.join(out, "run.sh"), run, step,
                              commands(self.args, step, pinned, self.latent_channels, self.wandb_run, device,
                                       self.python),
                              copy_from=copy_from, pinned=pinned, remove_pinned=remove)
        with open(os.path.join(out, "launch.log"), "ab") as logf:
            self.proc = _popen(["bash", script], stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               env=env, cwd=HERE, start_new_session=True, close_fds=True)
        self.current = {"pid": self.proc.pid, "step": step, "script": script, "out": out}
        with open(self.pid_file, "w") as f:
            json.dump(self.current, f)
        self.log(event="eval_launched", step=step, pid=self.proc.pid, ckpt=ckpt,
                 device=self.args.eval_device or "the training card", out=out)
        return self.proc.pid
