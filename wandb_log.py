"""Native Weights & Biases logging for train_wm.py and the evaluators, at zero cost to training.

The trainer writes every event to `results/<run>/log.jsonl` through one function, and
`RunLogger.log_event` is called right after that write, on the main process only. It re-logs the
event under the SAME series names `tools/wandb_tail.py` (the sidecar) uses, so a native run and a
sidecar-tailed run plot on one set of panels:

  train   train/loss, lr, grad_norm, grad_norm_max, clip_frac, steps_per_s, peak_mem_gb,
          nonfinite_loss, skipped_updates (and data_wait_frac when a trainer reports it)
  derived time/wall_hours (since the start event), time/steps_per_hour, data/examples_seen
          (step x global batch), data/game_hours_seen (at 35 tics/s), data/epochs (over the
          start event's window count)
  val     val/loss, val/loss_q1..q4 (by t-quartile), val/excursion, val/loss_minus_train
  eval    eval/<live|ema>_h<H>/psnr, lpips, persist_psnr, persist_lpips and psnr_over_persistence
          from an eval_tf.py summary; any other evaluator's summary flattened under eval/<tag>/

Rules the logger keeps:

  * **A no-op unless enabled.** `wandb` is imported only when a run is opened, so a checkout or a
    test without the package never touches it. A missing package prints one line to stderr.
  * **A custom x axis.** Every row carries `step`, and `define_metric("*", step_metric="step")`
    plots every series against it. `run.log` is never given `step=`: W&B drops a row whose `step=`
    is below the last one, and an evaluation of an earlier checkpoint arrives after later steps.
  * **Never raises into the caller.** Any W&B failure (import, init, log, finish) is reported once
    on stderr and the logger switches itself off, so a W&B outage cannot stop a run. `log.jsonl`
    stays the record; the W&B run is a live view of it.
  * **Never blocks the caller.** `wandb.init`, `run.log` and `run.finish` all run on the logger's
    own daemon thread. The caller builds the row (pure Python on host floats) and appends it to a
    bounded queue under a lock held for microseconds; when the queue is full the OLDEST row is
    dropped, and the drops are counted on stderr. Shutdown is bounded: W&B's own `finish_timeout`
    is set, `close()` waits at most `FINISH_TIMEOUT + CLOSE_MARGIN`, and on a timeout it says so
    once and unregisters the SDK's exit-time teardown, so the process can exit and release its GPU.
    Only `import wandb` runs on the caller's thread (local work, before the timed loop): importing
    on a thread while DataLoader workers fork could leave a module import lock held in the child.
  * **One writer per run id.** W&B's resume docs: "Unexpected results will occur if multiple
    processes use the same `id` concurrently." The trainer owns `<run>`; evaluators write to their
    own run `<run>-eval` in group `<run>`, which W&B overlays with the trainer's run in one panel.
    Evaluations of one training run should therefore not run concurrently either.
"""
import collections
import os
import re
import sys
import threading

DEFAULT_PROJECT = "doomdit-nexttic"
TICS_PER_S = 35.0
TRAIN_KEYS = ("loss", "lr", "grad_norm", "grad_norm_max", "clip_frac", "steps_per_s", "peak_mem_gb",
              "nonfinite_loss", "skipped_updates", "data_wait_frac")
EVAL_TF_TAG = re.compile(r"^(?:live|ema)_h\d+$")
EVAL_TF_KEYS = (("psnr_raw", "psnr"), ("lpips_raw", "lpips"), ("persist_psnr_raw", "persist_psnr"),
                ("persist_lpips_raw", "persist_lpips"), ("recon_psnr_raw", "recon_psnr"),
                ("recon_lpips_raw", "recon_lpips"))
CKPT_STEP = re.compile(r"^(?:snap_)?(\d+)\.pt$")
EVAL_SUFFIX = "-eval"
QUEUE_ROWS = 10000        # rows waiting for W&B; a train row every 100 steps makes this days of backlog
FINISH_TIMEOUT = 30.0     # seconds W&B may spend finishing a run (its own `finish_timeout`)
CLOSE_MARGIN = 10.0       # close() waits FINISH_TIMEOUT plus this, then gives the run up


def _num(v):
    return isinstance(v, (int, float))


def train_row(e, t0=None, global_batch=32, windows=None, tics_per_s=TICS_PER_S):
    """A `train` event's own numbers plus the derived time and data series, keyed as the sidecar keys them."""
    row = {f"train/{k}": e[k] for k in TRAIN_KEYS if _num(e.get(k))}
    step = e.get("step")
    if t0 is not None and _num(e.get("time")):
        row["time/wall_hours"] = (e["time"] - t0) / 3600.0
    if _num(e.get("steps_per_s")):
        row["time/steps_per_hour"] = e["steps_per_s"] * 3600.0
    if isinstance(step, int):
        seen = step * global_batch
        row["data/examples_seen"] = seen
        row["data/game_hours_seen"] = seen / tics_per_s / 3600.0
        if windows:
            row["data/epochs"] = seen / windows
    return row


def val_row(e):
    """A `val` event: overall held-out loss, its four t-quartiles and the excursion flag."""
    row = {}
    if _num(e.get("val_loss")):
        row["val/loss"] = e["val_loss"]
    q = e.get("val_loss_by_t_quartile")
    if isinstance(q, list):
        for i, v in enumerate(q):
            if _num(v):
                row[f"val/loss_q{i + 1}"] = v
    if "excursion" in e:
        row["val/excursion"] = int(bool(e["excursion"]))
    return row


def mean_of(m, key):
    """`m[key]`, or its `mean` when the evaluator wrote a {mean, sem, n} record; None if not numeric."""
    v = m.get(key)
    if isinstance(v, dict):
        v = v.get("mean")
    return v if _num(v) else None


def flatten(prefix, obj, out, depth=0):
    """Every numeric leaf of a JSON object as `prefix/key`; dicts with a `mean` collapse to it."""
    if isinstance(obj, dict) and _num(obj.get("mean")):
        out[prefix] = obj["mean"]
        return
    if isinstance(obj, dict) and depth < 3:
        for k, v in obj.items():
            flatten(f"{prefix}/{k}", v, out, depth + 1)
    elif _num(obj) and not isinstance(obj, bool):
        out[prefix] = obj
    elif isinstance(obj, list) and obj and all(_num(v) for v in obj) and len(obj) <= 16:
        for i, v in enumerate(obj):
            out[f"{prefix}/{i}"] = v


def eval_row(tag, metrics):
    """One evaluator summary as `eval/<tag>/...`.

    An eval_tf.py summary (tag `live_h<H>` or `ema_h<H>`) gives the raw-frame PSNR and LPIPS with
    the persistence floor beside them and PSNR minus persistence; anything else is flattened
    generically, so a new evaluator is plotted without a code change.
    """
    row = {}
    if EVAL_TF_TAG.match(tag):
        for src, dst in EVAL_TF_KEYS:
            v = mean_of(metrics, src)
            if v is not None:
                row[f"eval/{tag}/{dst}"] = v
        pf = row.get(f"eval/{tag}/persist_psnr")
        if f"eval/{tag}/psnr" in row and pf is not None:
            row[f"eval/{tag}/psnr_over_persistence"] = row[f"eval/{tag}/psnr"] - pf
    else:
        flatten(f"eval/{tag}", metrics, row)
    return row


def step_from_checkpoint(path):
    """The training step in a checkpoint's filename (`snap_0010000.pt`, `0010000.pt`), or None (`best.pt`)."""
    m = CKPT_STEP.match(os.path.basename(str(path or "")))
    return int(m.group(1)) if m else None


def resolve_eval_step(explicit, ckpt=None, recorded=None):
    """The step an evaluation is logged at: `--wandb-step`, else the checkpoint's filename, else the
    step the checkpoint records (for `best.pt`), else None."""
    if explicit is not None:
        return int(explicit)
    step = step_from_checkpoint(ckpt)
    if step is not None:
        return step
    return int(recorded) if isinstance(recorded, int) and not isinstance(recorded, bool) else None


def eval_run_names(run):
    """(W&B id and name, group) an evaluation of training run `run` logs to: `<run>-eval` in group `<run>`.

    Accepts the evaluation run's own name too, so `--wandb-run 040-unet-nexttic-eval` and
    `--wandb-run 040-unet-nexttic` land in the same place.
    """
    base = run[:-len(EVAL_SUFFIX)] if run.endswith(EVAL_SUFFIX) else run
    return base + EVAL_SUFFIX, base


def _settings(wandb, finish_timeout):
    """No console capture (W&B must not wrap the trainer's stdout), no system sampling, and a
    bounded finish: `finish_timeout` also bounds the SDK's own exit-time teardown."""
    for kw in ({"console": "off", "_disable_stats": True, "finish_timeout": finish_timeout},
               {"console": "off", "_disable_stats": True}):     # an SDK older than `finish_timeout`
        try:
            return wandb.Settings(**kw)
        except Exception:
            continue
    return None


def open_run(wandb, project, entity, run_id, name, group, config, wandb_dir, finish_timeout=None):
    """The one place a W&B run is opened: resumable by id, every series against the custom `step` axis.

    `resume="allow"` makes a restarted trainer (or a later evaluation) continue the same run. Each
    run id has exactly one writer at a time (see the module docstring); W&B's "shared mode"
    (`Settings(mode="shared", x_primary=..., x_label=...)`, SDK >= 0.19.9) would let the evaluators
    write into the trainer's own run, and can replace this function once it is verified on a
    rendered run.
    """
    ft = FINISH_TIMEOUT if finish_timeout is None else finish_timeout
    run = wandb.init(project=project, entity=entity, id=run_id, name=name, group=group, resume="allow",
                     config=config or {}, dir=wandb_dir, settings=_settings(wandb, ft))
    run.define_metric("step")
    run.define_metric("*", step_metric="step")
    return run


def _release_exit_hook(wandb):
    """Unregister the SDK's exit-time teardown, which waits for W&B's service process without a bound.

    SDK 0.30 registers it in `service_connection._start_and_connect_service` and hands its
    unregistration to the connection as `_cleanup` (`wandb.sdk.wandb_setup._singleton._connection`).
    This is SDK-internal, so it is best effort: on any other layout nothing happens, and W&B's own
    `finish_timeout` still bounds the teardown. The service process then finishes, or not, on its
    own; what is not uploaded stays under `.wandb/` for `wandb sync`.
    """
    try:
        conn = wandb.sdk.wandb_setup._singleton._connection
        cleanup = getattr(conn, "_cleanup", None)
        if cleanup:
            cleanup()
            conn._cleanup = None
            return True
    except Exception:
        pass
    return False


class RunLogger:
    """A W&B run fed from the trainer's events or an evaluator's summary; a no-op unless enabled.

    `name` is the run's W&B id and display name (the trainer uses its results directory's name) and
    `group` defaults to it. W&B's local files go to `<results_dir>/.wandb`. `log_event` and
    `log_eval` return the row they queued (without `step`) or None; nothing here raises, and only
    `close()` ever waits, for at most `FINISH_TIMEOUT + CLOSE_MARGIN` seconds.
    """

    def __init__(self, enabled=False, name=None, project=DEFAULT_PROJECT, entity=None, config=None,
                 results_dir=None, group=None, queue_rows=None):
        self.name = name or None
        self._wandb = self._thread = None
        self._items, self._rows, self._cap = collections.deque(), 0, int(queue_rows or QUEUE_ROWS)
        self._cv = threading.Condition()
        self._failed, self._closed = threading.Event(), False
        self._report_lock, self._reported, self._dropped = threading.Lock(), False, 0
        self._t0, self._global_batch, self._windows, self._last_train_loss = None, 32, None, None
        if not (enabled and self.name):
            return
        try:
            import wandb          # on this thread, on purpose: see "Never blocks the caller"
        except Exception as exc:
            self._fail("importing wandb", exc)
            return
        self._wandb = wandb
        self._thread = threading.Thread(target=self._work, name=f"wandb-{self.name}", daemon=True,
                                        args=(project, entity, group or self.name, config, results_dir))
        self._thread.start()

    @property
    def active(self):
        """True while rows are still accepted: enabled, not failed, not closed."""
        return self._thread is not None and not self._closed and not self._failed.is_set()

    @property
    def failed(self):
        """True once any W&B call has failed (reported once on stderr)."""
        return self._failed.is_set()

    # --- the caller's side: build the row, queue it, never wait -----------------------------------

    def _say(self, msg):
        print(f"wandb: {msg}", file=sys.stderr, flush=True)

    def _fail(self, what, exc):
        self._failed.set()
        with self._report_lock:
            first, self._reported = not self._reported, True
        if first:
            self._say(f"{what} failed ({type(exc).__name__}: {exc}); W&B logging is off for the rest of this "
                      "process, log.jsonl stays the record")

    def _put(self, kind, payload=None):
        dropped_first = False
        with self._cv:
            if kind == "log":
                if self._rows >= self._cap:
                    for i, item in enumerate(self._items):      # the oldest ROW, never a control item
                        if item[0] == "log":
                            del self._items[i]
                            break
                    self._rows -= 1
                    self._dropped += 1
                    dropped_first = self._dropped == 1
                self._rows += 1
            self._items.append((kind, payload))
            self._cv.notify()
        if dropped_first:
            self._say(f"the queue of {self._cap} rows is full (W&B is slower than training); dropping the "
                      "oldest rows, the total is reported when the run closes")

    def log_event(self, e):
        """Queue one trainer event (a `log.jsonl` line as a dict) under the sidecar's series; `end` closes the run."""
        if self._thread is None or self._closed:
            return None
        if self._failed.is_set():
            if e.get("event") == "end":
                self.close()
            return None
        try:
            row = self._row_of(e)
        except Exception as exc:
            self._fail(f"mapping a {e.get('event')} event", exc)
            return None
        if row:
            self._put("log", {"step": e["step"], **row})
        if e.get("event") == "end":
            self.close()
        return row or None

    def _row_of(self, e):
        kind, step = e.get("event"), e.get("step")
        if kind == "start":
            if _num(e.get("time")):
                self._t0 = e["time"]
            if isinstance(e.get("global_batch"), int):
                self._global_batch = e["global_batch"]
            ds = e.get("dataset_summary") or {}
            if isinstance(ds.get("windows"), int):
                self._windows = ds["windows"]
            return {}
        row = {}
        if isinstance(step, int):
            if kind == "train":
                row = train_row(e, self._t0, self._global_batch, self._windows)
                if _num(e.get("loss")):
                    self._last_train_loss = e["loss"]
            elif kind == "val":
                row = val_row(e)
                if _num(e.get("val_loss")) and self._last_train_loss is not None:
                    row["val/loss_minus_train"] = e["val_loss"] - self._last_train_loss
        return row

    def log_eval(self, step, tag, metrics):
        """Queue one evaluator summary at training step `step` under `eval/<tag>/...`."""
        if not self.active:
            return None
        try:
            row = eval_row(tag, metrics)
        except Exception as exc:
            self._fail(f"mapping the {tag} evaluation", exc)
            return None
        if row:
            self._put("log", {"step": int(step), **row})
        return row or None

    def flush(self, timeout=10.0):
        """Wait until everything queued so far has reached W&B (or been dropped); False on a timeout."""
        if not self.active:
            return False
        done = threading.Event()
        self._put("flush", done)
        return done.wait(timeout)

    def close(self, timeout=None):
        """Finish the run, waiting at most `timeout` (default FINISH_TIMEOUT + CLOSE_MARGIN) seconds.

        True when the run finished in time (or there was none). On a timeout it says so once,
        unregisters the SDK's exit-time teardown and returns False: the daemon thread keeps trying,
        but nothing holds the process at exit. Safe to call more than once.
        """
        if self._thread is None or self._closed:
            return True
        self._closed = True
        self._put("close")
        wait = FINISH_TIMEOUT + CLOSE_MARGIN if timeout is None else timeout
        self._thread.join(wait)
        finished = not self._thread.is_alive()
        if self._dropped:
            self._say(f"dropped {self._dropped} rows in total while the queue was full; log.jsonl has all of them")
        if not finished:
            _release_exit_hook(self._wandb)
            self._say(f"the run did not finish within {wait:.0f}s; it is left to W&B's background thread and "
                      "this process is free to exit (unsent rows stay under .wandb/ for `wandb sync`)")
        return finished

    # --- the logger's own thread: every W&B call ----------------------------------------------------

    def _get(self):
        with self._cv:
            while not self._items:
                self._cv.wait()
            kind, payload = self._items.popleft()
            if kind == "log":
                self._rows -= 1
            return kind, payload

    def _work(self, project, entity, group, config, results_dir):
        run = None
        try:
            wandb_dir = os.path.join(results_dir or ".", ".wandb")
            os.makedirs(wandb_dir, exist_ok=True)
            run = open_run(self._wandb, project, entity, self.name, self.name, group, config, wandb_dir,
                           FINISH_TIMEOUT)
        except Exception as exc:
            self._fail("opening the W&B run", exc)
        while True:
            kind, payload = self._get()
            if kind == "flush":
                payload.set()
            elif kind == "close":
                if run is not None and not self._failed.is_set():
                    try:
                        run.finish()
                    except Exception as exc:
                        self._fail("finishing the W&B run", exc)
                return
            elif run is not None and not self._failed.is_set():
                try:
                    run.log(payload)
                except Exception as exc:
                    self._fail("logging a row", exc)


def add_eval_args(p):
    """The evaluators' W&B flags. Logging is off unless `--wandb-run` names the training run."""
    p.add_argument("--wandb-run", dest="wandb_run", default="",
                   help="append this evaluation's summary metrics to W&B. The name of the TRAINING run (its "
                        "results directory, e.g. 040-unet-nexttic); the metrics go to the run <name>-eval in "
                        "group <name>, which W&B overlays with the training curves. Empty (default): no W&B")
    p.add_argument("--wandb-project", dest="wandb_project", default=DEFAULT_PROJECT)
    p.add_argument("--wandb-entity", dest="wandb_entity", default=None)
    p.add_argument("--wandb-step", dest="wandb_step", type=int, default=None,
                   help="the training step to log at; default: from the checkpoint's filename (snap_0010000.pt, "
                        "0010000.pt), else the step the checkpoint records")
    return p


def log_evaluation(args, tag, metrics, ckpt=None, recorded_step=None, out_dir=None):
    """Append one evaluator summary to `<run>-eval` at its checkpoint's step, then close the run.

    A no-op without `--wandb-run`. Returns the row logged (without `step`) or None, and never raises:
    an evaluation's metrics file is its record, and W&B is only the live view of it.
    """
    run = getattr(args, "wandb_run", "") or ""
    if not run:
        return None
    step = resolve_eval_step(getattr(args, "wandb_step", None), ckpt, recorded_step)
    if step is None:
        print(f"wandb: cannot tell which training step {ckpt!r} holds; pass --wandb-step. Nothing logged",
              file=sys.stderr, flush=True)
        return None
    run_id, group = eval_run_names(run)
    lg = RunLogger(enabled=True, name=run_id, group=group, project=getattr(args, "wandb_project", DEFAULT_PROJECT),
                   entity=getattr(args, "wandb_entity", None), config={"trainer_run": group}, results_dir=out_dir)
    row = lg.log_eval(step, tag, metrics)
    finished = lg.close()
    return row if finished and not lg.failed else None
