"""
Shared loader for the paper builders: reads the result directories, never types a number.

Every accessor returns None when the artifact is missing or lacks the field, and records a
warning; `fmt` turns None into "n/a". Schemas (all produced by scripts in this repo):

    <run>/config.json                       train_wm.py / train_video.py: params, backbone, steps, seed
    <run>/log.jsonl                         val events with val_loss (the held-out v-loss)
    <run>/eval_tf_<corpus>/metrics.json     eval_tf.py: {metric: {mean, sem, n}}, per_map {map: {psnr_dec, lpips_dec}}
    <run>/eval_tf_<corpus>/per_window.csv   eval_tf.py per-window rows
    <run>/rollout_metrics_seen/drift.json   rollout_eval.py: psnr / lpips / copy_seed_psnr lists over horizons,
                                            psnr@h, lpips@h, idm_top1 list, idm_*_mean
    <run>/rollout_metrics_seen/fvd{16,32}.json    fvd.py: {fvd, frames, num_clips}
    <run>/audit/audit.json                  rollout_audit.py: blur_sweep, late_drop, bootstrap
    <run>/audit/per_rollout.npz             rollout_audit.py: episode + per-rollout values at horizons 8/32/64
    tmp/audit/summary.json                  rollout_audit.py: identity, per_run, paired differences
    idm_aligned*/metrics.json               train_idm.py: top1, macro_recall, movement, majority_baseline
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS_JSON = os.path.join(HERE, "rows.json")
NA = "n/a"


class Warn:
    """Collects warnings and prints each once to stderr."""

    def __init__(self):
        self.seen = []

    def __call__(self, msg):
        if msg not in self.seen:
            self.seen.append(msg)
            print(f"[paper] warning: {msg}", file=sys.stderr)


warn = Warn()


def read_json(path, what=None):
    if not os.path.exists(path):
        warn(f"missing {what or path} ({path})")
        return None
    try:
        return json.load(open(path))
    except (ValueError, OSError) as e:
        warn(f"unreadable {path}: {e}")
        return None


def load_registry(path=ROWS_JSON):
    return json.load(open(path))


def fmt(x, nd=2, na=NA):
    """Fixed-point string, or n/a for None / NaN."""
    if x is None:
        return na
    try:
        x = float(x)
    except (TypeError, ValueError):
        return na
    if np.isnan(x):
        return na
    return f"{x:.{nd}f}"


def fmt_ci(ci, nd=2):
    """'[lo, hi]' from a two-element list, or '' when absent (the value stands alone)."""
    if not ci or len(ci) != 2 or any(v is None for v in ci):
        return ""
    return f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def fmt_params(n):
    return NA if n is None else f"{n / 1e6:.0f}M"


def get(d, *keys):
    """Nested lookup that returns None on any missing key."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


class Run:
    """One result directory, read lazily; every attribute is None when the artifact is absent."""

    def __init__(self, root, spec, reg):
        self.spec, self.run = spec, spec["run"]
        self.key, self.label, self.short = spec["key"], spec["label"], spec.get("short", spec["label"])
        self.placeholder = bool(spec.get("placeholder"))
        self.dir = os.path.join(root, self.run)
        self.exists = os.path.isdir(self.dir)
        self.corpora = reg["corpora"]
        self.rollout_dir = os.path.join(self.dir, reg.get("rollout_dir", "rollout_metrics_seen"))
        self.audit_dir = os.path.join(self.dir, reg.get("audit_dir", "audit"))
        self._cache = {}
        if not self.exists:
            warn(f"run directory missing for row '{self.key}': {self.run}" + (" (placeholder row)" if self.placeholder else ""))

    # ---- raw artifacts ---------------------------------------------------------------
    def _json(self, name, path, what):
        if name not in self._cache:
            self._cache[name] = read_json(path, what) if self.exists else None
        return self._cache[name]

    @property
    def config(self):
        return self._json("config", os.path.join(self.dir, "config.json"), f"{self.run}/config.json")

    def tf(self, corpus):
        sub = self.corpora.get(corpus, corpus)
        return self._json(f"tf_{corpus}", os.path.join(self.dir, sub, "metrics.json"), f"{self.run}/{sub}/metrics.json")

    @property
    def drift(self):
        return self._json("drift", os.path.join(self.rollout_dir, "drift.json"), f"{self.run} drift.json")

    def fvd(self, frames):
        d = self._json(f"fvd{frames}", os.path.join(self.rollout_dir, f"fvd{frames}.json"), f"{self.run} fvd{frames}.json")
        return get(d, "fvd")

    @property
    def audit(self):
        if "audit" not in self._cache:
            p = os.path.join(self.audit_dir, "audit.json")
            if self.exists and not os.path.exists(p):
                warn(f"no audit for {self.run} (blur sweep, late drop and bootstrap CIs will be n/a)")
                self._cache["audit"] = None
            else:
                self._cache["audit"] = self._json("audit_json", p, f"{self.run} audit.json")
        return self._cache["audit"]

    @property
    def per_rollout(self):
        """Per-rollout arrays from the audit (horizons 8/32/64) or, if present, a per-horizon
        `rollout_metrics_seen/per_rollout.npz` with (N, H) psnr and lpips plus episode."""
        if "per_rollout" not in self._cache:
            out = None
            for p in (os.path.join(self.rollout_dir, "per_rollout.npz"), os.path.join(self.audit_dir, "per_rollout.npz")):
                if self.exists and os.path.exists(p):
                    with np.load(p) as z:
                        out = {k: z[k] for k in z.files}
                    break
            self._cache["per_rollout"] = out
        return self._cache["per_rollout"]

    @property
    def vloss_final(self):
        """Last validation loss in log.jsonl (the held-out v-loss at the end of training)."""
        if "vloss" not in self._cache:
            v = None
            p = os.path.join(self.dir, "log.jsonl")
            if self.exists and os.path.exists(p):
                for line in open(p):
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        continue
                    if ev.get("event") == "val" and "val_loss" in ev:
                        v = float(ev["val_loss"])
            self._cache["vloss"] = v
        return self._cache["vloss"]

    # ---- derived scalars ----------------------------------------------------------------
    @property
    def params(self):
        return get(self.config, "params")

    def tf_mean(self, corpus, key):
        return get(self.tf(corpus), key, "mean")

    def tf_ci(self, corpus, key):
        """Episode-bootstrap 95% CI from audit.json, or None."""
        return get(self.audit, "bootstrap", f"tf_{corpus}", key, "ci95")

    def per_map(self, corpus, key):
        pm = get(self.tf(corpus), "per_map")
        if not pm:
            return {}
        return {int(m): v.get(key) for m, v in pm.items()}

    def rollout_at(self, metric, h):
        return get(self.drift, f"{metric}@{h}")

    def rollout_curve(self, metric):
        v = get(self.drift, metric)
        return None if v is None else np.asarray(v, dtype=np.float64)

    def rollout_ci(self, metric, h, n_boot=2000, seed=0):
        """95% CI at horizon h: the audit's own 10k-resample interval when it has one (horizon 64),
        else an episode bootstrap over per-rollout data (audit horizons 8/32/64, or every horizon
        when a per-horizon per_rollout.npz exists), else None."""
        ci = get(self.audit, "bootstrap", "rollout", f"{metric}_h{h}_s0.0", "ci95")
        if ci:
            return ci
        pr = self.per_rollout
        if pr is not None:
            vals = pr.get(f"{metric}_h{h}_s0.0")
            if vals is None and metric in pr and np.ndim(pr[metric]) == 2 and pr[metric].shape[1] >= h:
                vals = pr[metric][:, h - 1]
            if vals is not None and "episode" in pr:
                return list(bootstrap_ci(vals, pr["episode"], n_boot, seed)[1:])
        return None

    def idm_top1(self):
        return get(self.drift, "idm_top1_mean")

    def idm_real_top1(self):
        return get(self.drift, "idm_real_top1_mean")

    def idm_ci(self):
        return get(self.audit, "bootstrap", "rollout", "idm_top1", "ci95")

    def copy_seed_psnr(self, h):
        c = self.rollout_curve("copy_seed_psnr")
        return None if c is None or len(c) < h else float(c[h - 1])

    def copy_seed_lpips(self, h):
        return get(self.audit, "blur_sweep", str(h), "0.0", "copy_seed_lpips")

    def blur(self, h, sigma, key):
        return get(self.audit, "blur_sweep", str(h), str(float(sigma)), key)


def load_rows(root, reg, group=None):
    rows = [Run(root, s, reg) for s in reg["rows"] if group is None or group in s.get("groups", [])]
    return rows


# ---- bootstrap (same convention as rollout_audit.episode_resamples) ------------------------

def episode_resamples(episode_ids, n_boot, seed):
    """Cluster bootstrap index draws: whole episodes resampled with replacement, RandomState(seed)."""
    episode_ids = np.asarray(episode_ids)
    uniq = np.unique(episode_ids)
    members = {e: np.flatnonzero(episode_ids == e) for e in uniq}
    rng = np.random.RandomState(seed)
    return [np.concatenate([members[e] for e in uniq[rng.randint(len(uniq), size=len(uniq))]]).astype(np.int32)
            for _ in range(n_boot)]


def bootstrap_ci(values, episode_ids, n_boot=2000, seed=0, alpha=0.05):
    """(mean, lo, hi) percentile CI over episode resamples."""
    values = np.asarray(values, dtype=np.float64)
    draws = episode_resamples(episode_ids, n_boot, seed)
    samples = np.array([values[d].mean() for d in draws])
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


# ---- shared references -----------------------------------------------------------------------

def reference(rows, corpus, key, tol=0.02):
    """A corpus-level reference (copy-last, VAE ceiling) that every run's eval_tf reports identically.

    Takes the first run that has it and warns if another run disagrees beyond `tol`, which would
    mean the runs were not scored on the same windows.
    """
    vals = [(r.run, r.tf_mean(corpus, key)) for r in rows if r.tf_mean(corpus, key) is not None]
    if not vals:
        warn(f"no run provides the {corpus} reference '{key}'")
        return None
    first = vals[0][1]
    for run, v in vals[1:]:
        if abs(v - first) > tol:
            warn(f"{corpus} reference '{key}' differs between {vals[0][0]} ({first:.3f}) and {run} ({v:.3f})")
    return first


def read_per_window(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
