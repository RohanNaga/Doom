"""Transition-level distances between evaluation footage and the training footage (Sep 26 2026).

The frozen frame distance D (`distance_study.py distances`) compares clouds of single frames. It separates the
training maps from the unseen ones but, within the 13 unseen arenas, orders nothing (RESEARCH_CONTEXT section 0,
"the distance problem"). The world model predicts TRANSITIONS: the next latent from a 32-tic context and the
executed controls. This module measures the shift at that level, two ways, each to be frozen before any
adaptation curve is read:

  * candidate 1, DIRECTED TRANSITION COVERAGE (Astra, `.claude/analyses/astra-distance-rethink-2026-09-26.md`
    section 2): does the training footage contain transitions like the ones this map requires? Each window is
    described by its state, its context dynamics, its next-tic innovation and its control history; the score is
    the mean, over a map's windows, of the mean distance to the k = 3 nearest training windows, each from a
    different training episode (`directed_coverage`);
  * candidate 2, the kNN TRANSFER GAP G (Opus, `.claude/analyses/opus-distance-view-2026-09-26.md` section 2): a
    nearest-neighbour predictor of the next pooled latent change from (state, previous change) within the same
    control class, once with training-map memory and once with the map's own adaptation episodes as memory;
    G = mean log10 error ratio to copy-last with training memory minus the same with own-map memory
    (`transfer_gap`). Subtracting the own-map predictor removes how hard the map is to predict at all.

**The window.** A scored one-tic window of `eval_tf.py`: context rows r-32 .. r-1 of one life with consecutive
tics, target row r (`distance_study.eligible_rows`). With t = r - 1 the newest context frame and pool = the
192-dimensional `distance_study.pool_latents` (SD 1.x; 768 for SD 3.5):

    state        pool(z_t)
    dynamics     pool(z_t) - pool(z_(t-l)) for l in (1, 4, 16, 31); lag 31 is the oldest context frame
    innovation   pool(z_(t+1)) - pool(z_t)
    controls     newest: the executed 19-bit vector on row t, the control applied from t into the target
                 (record_arnold.py stores a row THEN steps, so row t carries the control leaving frame t);
                 freq: each button's share of the 32 context rows r-32 .. r-1, the model's control history;
                 switch: each button's share of the 31 adjacent context pairs on which it changed
    class        turn {none, L, R} x move {none, fwd, back} x attack of the newest control (18 classes;
                 pressing both directions of a pair counts as neither)

`transitions` draws N windows per episode (250) stratified by motion decile exactly as the frame clouds are
drawn (`stratified_draw` seeded by (seed, episode) on the target rows' motion ||z_r - z_(r-1)||), so with the
same seed the windows sit on the frame clouds' own frames. Each episode is cached as
`<out>/transitions/<space>/<set>/ep_XXXXX.npz` with its settings, and each set gets an `index.json`.

**Candidate 1 metric** (Astra's recipe; design choices, not fitted values). Fixed Gaussian projections (seed 0)
take state, dynamics (4 x 192 flattened) and innovation to 32, 64 and 32 dimensions; the 57 control features
stay raw. Each block is centred on the memory and scaled so its RMS pairwise distance over the memory is
sqrt(1/4): the four blocks carry equal weight and the whole vector has RMS pairwise distance 1, the unit every
coverage value is in. Neighbours never come from the query's own episode. Ablations, all on the same metric:
`state_only` (the state block alone: its share of the full score says how much is still appearance),
`without_state` (the other three blocks), `shuffled_transitions` (each target window's innovation replaced by
that of a random other window of the same map: must rise if the score sees transitions), `shuffled_controls`
(the control block permuted the same way: must rise if controls carry information).

**Candidate 2 metric** (Opus's recipe). Search space (state, previous change = the lag-1 dynamics), each
192-d block z-scored per dimension on the whole training memory and divided by the square root of its number of
non-constant dimensions (unit total variance). The prediction is the mean innovation of the k = 16 nearest memory
windows of the same control class, or of all classes when the memory holds fewer than 500 of that class. The
per-window error is r = ||v_hat - v||^2 / ||v||^2 in pooled latent units: copy-last predicts zero change, so
r < 1 beats it. The training memory is subsampled (seeded) to the own-map memory's size, so both predictors
search equally large memories. Scored on the held-out episodes of the map's adaptation split
(`adapt_split.py`), with the split's `adapt` list as own-map memory.

**Aggregation.** A map's value is the weighted mean over its windows, each weighted by its episode's eligible
windows over its drawn windows (Astra's population weighting; `--weighting episode` makes equal-per-episode
primary instead, and both are always reported). Uncertainty is a bootstrap over the map's target episodes.

**Ambiguities resolved here** (reported with the change): Astra's "dynamics plus controls only" ablation is
`without_state` (every block but the state), so it is the exact complement of `state_only`; the ablations keep
the full metric's block scales, so `state_only <= coverage` and their ratio is a share; own-map memory is the
split's whole `adapt` list (16 episodes on the fresh set; the Opus memo's 12 predates the 16/8 split); the
training memory for G is one seeded subsample per memory size, shared by every map of that size (common random
numbers); windows whose true innovation is exactly zero have no copy-last error and are excluded from G, and
counted.

    from transition_distance import directed_coverage, transfer_gap
"""
import concurrent.futures
import csv
import glob
import hashlib
import json
import os
import subprocess
import time
import zlib

import numpy as np

import distance_study as ds

LAGS = (1, 4, 16, 31)                 # context dynamics lags; 31 reaches the oldest of 32 context frames
K_COVERAGE = 3
K_GAP = 16
MIN_CLASS = 500                       # a control class with fewer memory windows searches every class
PROJECTION_DIMS = {"state": 32, "dynamics": 64, "innovation": 32}
CONTROL_FEATURES = ("newest", "freq", "switch")
BLOCKS = ("state", "dynamics", "innovation", "controls")
BOOTSTRAP = 1000
QUERY_BLOCK = 256                     # queries per block of the exact search: 256 x 50k memory is 100 MB
WEIGHTINGS = ("eligible", "episode")
LATENT_SPACES = ("sd1", "sd35")
CONTROL_BITS = 19                     # the executed button vector (transitions.EXECUTED_BUTTONS)
MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, ATTACK = 0, 1, 2, 3, 6   # transitions.CONTROL_BITS order
N_CLASSES = 18
WINDOW_ARRAYS = ("state", "dyn", "innov", "newest", "freq", "switch", "klass", "row", "tic", "life", "decile",
                 "motion")
SETTING_KEYS = ("space", "context_frames", "lags", "frames_per_episode", "bins")
RATIO_FLOOR = 1e-12                   # a perfect kNN prediction would give log10(0)
DEGENERATE_TOL = 1e-9                 # a block whose memory spread is below this share of its size is constant
COVERAGE_VARIANTS = ("coverage", "state_only", "without_state", "shuffled_transitions", "shuffled_controls")


# =============================================================================================
# the window: features of one transition, the control encoding, the control class
# =============================================================================================

def control_bits(buttons):
    """(T, 19) uint8 executed button vectors from a sidecar's `buttons` strings (raw or normalised)."""
    from transitions import normalize_button_column
    b = normalize_button_column(buttons)
    return (np.ascontiguousarray(b).view("<U1").reshape(len(b), CONTROL_BITS) == "1").astype(np.uint8)


def control_features(bits, targets, context_frames=ds.CONTEXT_FRAMES):
    """(newest, freq, switch) of the windows whose target rows are `targets`.

    newest (N, 19) uint8 is row r-1, the control applied from the newest context frame into the target. freq
    (N, 19) is each button's share of the context rows r-L .. r-1; switch (N, 19) is each button's share of
    the L - 1 adjacent context pairs on which it changed.
    """
    b = np.asarray(bits, dtype=np.float64)
    r = np.asarray(targets, dtype=np.int64)
    L = int(context_frames)
    if L < 2:
        raise ValueError("the control history needs at least two context rows")
    if len(r) and (r.min() < L or r.max() >= len(b)):
        raise ValueError("a target row lacks its full context of controls")
    csum = np.concatenate([np.zeros((1, b.shape[1])), np.cumsum(b, axis=0)])
    changed = np.concatenate([np.zeros((1, b.shape[1])), np.cumsum(b[1:] != b[:-1], axis=0)])
    freq = (csum[r] - csum[r - L]) / L
    switch = (changed[r - 1] - changed[r - L]) / (L - 1)
    return np.asarray(bits)[r - 1].astype(np.uint8), freq.astype(np.float32), switch.astype(np.float32)


def control_class(newest):
    """Class id in [0, 18) of executed controls: 6 * turn + 2 * move + attack.

    turn is 0 none, 1 left, 2 right; move is 0 none, 1 forward, 2 backward; both directions of a pair
    pressed at once cancel and count as none. Strafe, speed, crouch and weapon selects are not in the class.
    """
    n = np.asarray(newest).astype(bool)
    turn = np.where(n[:, TURN_LEFT] & ~n[:, TURN_RIGHT], 1, np.where(n[:, TURN_RIGHT] & ~n[:, TURN_LEFT], 2, 0))
    move = np.where(n[:, MOVE_FORWARD] & ~n[:, MOVE_BACKWARD], 1,
                    np.where(n[:, MOVE_BACKWARD] & ~n[:, MOVE_FORWARD], 2, 0))
    return (6 * turn + 2 * move + n[:, ATTACK]).astype(np.int8)


def window_features(lat, targets, lags=LAGS):
    """(state (N, d), dyn (N, len(lags), d), innov (N, d)) pooled features of the windows ending at `targets`.

    Reads only the rows the features need from `lat` (a memory map is fine). Pooling is linear, so the
    pooled difference of two frames is the difference of their pooled features.
    """
    r = np.asarray(targets, dtype=np.int64)
    t = r - 1
    need = np.unique(np.concatenate([r, t] + [t - int(lag) for lag in lags]))
    if len(need) and need.min() < 0:
        raise ValueError("a lag reaches before the first row")
    pooled = ds.pool_latents(lat[need])

    def at(rows):
        return pooled[np.searchsorted(need, rows)]
    state = at(t)
    dyn = np.stack([state - at(t - int(lag)) for lag in lags], axis=1)
    return state, dyn, at(r) - state


# =============================================================================================
# the exact search: k nearest memory windows, at most one per memory episode
# =============================================================================================

def _sq_dists(q, m, m_norm):
    d = q @ m.T
    d *= -2.0
    d += np.einsum("ij,ij->i", q, q)[:, None]
    d += m_norm[None, :]
    return np.maximum(d, 0.0, out=d)


def knn(queries, query_keys, memory, memory_keys, k, distinct=True, block=QUERY_BLOCK, neighbours=False):
    """(distances (n, k) ascending, memory indices (n, k) or None) of each query's k nearest memory windows.

    Exact Euclidean search in blocks of `block` queries (float64 throughout). A memory window whose key (the
    episode it came from) equals the query's is never a neighbour. With `distinct`, at most one window per
    memory episode counts: each episode contributes its nearest window, and the k nearest episodes are kept.
    """
    q = np.asarray(queries, dtype=np.float64)
    m = np.asarray(memory, dtype=np.float64)
    qk = np.asarray(query_keys, dtype=np.int64).reshape(-1)
    mk0 = np.asarray(memory_keys, dtype=np.int64).reshape(-1)
    if q.ndim != 2 or m.ndim != 2 or q.shape[1] != m.shape[1]:
        raise ValueError(f"queries {q.shape} and memory {m.shape} must share one feature space")
    if len(qk) != len(q) or len(mk0) != len(m):
        raise ValueError("one key per query and per memory window")
    k = int(k)
    order = np.argsort(mk0, kind="stable")
    m, mk = m[order], mk0[order]
    m_norm = np.einsum("ij,ij->i", m, m)
    M = len(m)
    if distinct:
        starts = np.flatnonzero(np.concatenate([[True], mk[1:] != mk[:-1]])) if M else np.zeros(0, np.int64)
        seg_keys = mk[starts]
        seg_of = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, M)))
        available = len(starts)
    else:
        available = M
    own_possible = np.isin(qk, mk)
    if available - int(own_possible.any()) < k:
        raise ValueError(f"{available} memory {'episodes' if distinct else 'windows'} cannot give {k} neighbours "
                         "outside the query's own episode")
    out_d = np.empty((len(q), k))
    out_i = np.empty((len(q), k), dtype=np.int64) if neighbours else None
    cols = np.arange(M)
    for s in range(0, len(q), max(1, int(block))):
        e = min(len(q), s + max(1, int(block)))
        d = _sq_dists(q[s:e], m, m_norm)
        own = qk[s:e]
        if distinct:
            emin = np.minimum.reduceat(d, starts, axis=1)
            if neighbours:
                first = np.minimum.reduceat(np.where(d <= emin[:, seg_of], cols[None, :], M), starts, axis=1)
            pos = np.clip(np.searchsorted(seg_keys, own), 0, len(seg_keys) - 1)
            hit = np.flatnonzero(seg_keys[pos] == own)
            emin[hit, pos[hit]] = np.inf
            table = emin
        else:
            if own_possible[s:e].any():
                d[mk[None, :] == own[:, None]] = np.inf
            table = d
        sel = np.argpartition(table, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(table, sel, 1)
        o = np.argsort(vals, axis=1, kind="stable")
        sel, vals = np.take_along_axis(sel, o, 1), np.take_along_axis(vals, o, 1)
        if not np.all(np.isfinite(vals)):
            raise ValueError("fewer than k neighbours outside the query's own episode")
        out_d[s:e] = np.sqrt(vals)
        if neighbours:
            idx = np.take_along_axis(first, sel, 1) if distinct else sel
            out_i[s:e] = order[idx]
    return out_d, out_i


def directed_coverage(queries, query_keys, memory, memory_keys, k=K_COVERAGE, distinct=True, block=QUERY_BLOCK,
                      neighbours=False):
    """Per query window, the mean distance to its k nearest memory windows (distinct episodes, never its own).

    Returns (values (n,), neighbour memory indices (n, k) or None). A map's coverage is the weighted mean of
    these over its windows: directed, so the map is never charged for training modes it does not use.
    """
    d, i = knn(queries, query_keys, memory, memory_keys, k, distinct, block, neighbours)
    return d.mean(axis=1), i


def gaussian_projection(dim_in, dim_out, seed):
    """(dim_in, dim_out) Gaussian matrix, N(0, 1 / dim_out) entries, reproducible from `seed`."""
    return np.random.default_rng(seed).standard_normal((int(dim_in), int(dim_out))) / np.sqrt(int(dim_out))


def raw_blocks(w, controls=CONTROL_FEATURES):
    """{block: (N, d) float64} of a window table, before projection and scaling."""
    n = len(w["state"])
    return {"state": np.asarray(w["state"], np.float64),
            "dynamics": np.asarray(w["dyn"], np.float64).reshape(n, -1),
            "innovation": np.asarray(w["innov"], np.float64),
            "controls": np.concatenate([np.asarray(w[c], np.float64) for c in controls], axis=1)}


class CoverageSpace:
    """Candidate 1's fixed metric: blocks projected once (seeded), centred and scaled on the memory.

    Every block is scaled so its RMS pairwise distance over the memory is sqrt(1 / 4); the RMS pairwise
    distance of a cloud is sqrt(2 x its total variance), so no pair is ever enumerated. A block that is
    constant over the memory keeps scale 1 and is listed in `degenerate`: it adds nothing between memory
    windows and still charges a target that leaves the constant.
    """

    def __init__(self, memory_blocks, dims=None, seed=0):
        dims = dict(PROJECTION_DIMS if dims is None else dims)
        self.proj, self.mean, self.scale, self.degenerate = {}, {}, {}, []
        for i, b in enumerate(BLOCKS):
            x = memory_blocks[b]
            if b in dims:
                self.proj[b] = gaussian_projection(x.shape[1], dims[b], [int(seed), i])
                x = x @ self.proj[b]
            self.mean[b] = x.mean(axis=0)
            rms_pair = float(np.sqrt(2.0 * np.mean(np.sum((x - self.mean[b]) ** 2, axis=1))))
            # a constant block still shows rounding after projection and centring; scaling that up would
            # turn float noise into distance, so it counts as constant below a relative tolerance
            if rms_pair > DEGENERATE_TOL * max(1.0, float(np.sqrt(np.mean(np.sum(x * x, axis=1))))):
                self.scale[b] = np.sqrt(1.0 / len(BLOCKS)) / rms_pair
            else:
                self.scale[b] = 1.0
                self.degenerate.append(b)

    def transform(self, blocks, which=BLOCKS):
        """(N, sum of the chosen blocks' dimensions) features in the memory-fitted metric."""
        out = []
        for b in which:
            x = blocks[b] @ self.proj[b] if b in self.proj else blocks[b]
            out.append((x - self.mean[b]) * self.scale[b])
        return np.concatenate(out, axis=1)

    def describe(self):
        return {"dims": {b: int(len(self.mean[b])) for b in BLOCKS}, "scale": {b: float(self.scale[b]) for b in BLOCKS},
                "degenerate_blocks": list(self.degenerate)}


# =============================================================================================
# candidate 2: the kNN transfer gap
# =============================================================================================

class GapSpace:
    """Candidate 2's search space (state, previous change), z-scored per dimension on the training memory.

    Each 192-d block is then divided by the square root of its number of non-constant dimensions, so each
    block has unit total variance over the training memory and the two weigh the same.
    """

    def __init__(self, state, prev):
        self.stats = []
        for x in (np.asarray(state, np.float64), np.asarray(prev, np.float64)):
            mu, sd = x.mean(axis=0), x.std(axis=0)
            live = sd > 0
            self.stats.append((mu, np.where(live, sd, 1.0), 1.0 / np.sqrt(max(1, int(live.sum())))))

    def transform(self, state, prev):
        return np.concatenate([(np.asarray(x, np.float64) - mu) / sd * c
                               for x, (mu, sd, c) in zip((state, prev), self.stats)], axis=1)


def knn_predict(queries, query_class, query_keys, memory, memory_class, memory_keys, memory_innov, k=K_GAP,
                min_class=MIN_CLASS, block=QUERY_BLOCK):
    """(predicted innovation (n, d), searched-all-classes flag (n,)): mean innovation of the k nearest memory
    windows of the query's control class, or of all memory windows when that class has fewer than `min_class`."""
    qc = np.asarray(query_class).reshape(-1)
    mc = np.asarray(memory_class).reshape(-1)
    innov = np.asarray(memory_innov, dtype=np.float64)
    out = np.empty((len(qc), innov.shape[1]))
    fallback = np.zeros(len(qc), dtype=bool)
    counts = np.bincount(mc.astype(np.int64), minlength=N_CLASSES)
    for c in np.unique(qc):
        qi = np.flatnonzero(qc == c)
        pool = np.flatnonzero(mc == c) if counts[int(c)] >= max(int(min_class), int(k)) else None
        if pool is None:
            pool = np.arange(len(mc))
            fallback[qi] = True
        _, nb = knn(queries[qi], np.asarray(query_keys)[qi], memory[pool], np.asarray(memory_keys)[pool], k,
                    distinct=False, block=block, neighbours=True)
        out[qi] = innov[pool][nb].mean(axis=1)
    return out, fallback


def log_error_ratio(predicted, true):
    """log10(||predicted - true||^2 / ||true||^2) per window: below 0 beats copy-last (zero change)."""
    p, t = np.asarray(predicted, np.float64), np.asarray(true, np.float64)
    num = np.sum((p - t) ** 2, axis=1)
    den = np.sum(t ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(np.maximum(num / den, RATIO_FLOOR))


def transfer_gap(target, own, train, k=K_GAP, min_class=MIN_CLASS, lag_index=0, block=QUERY_BLOCK,
                 min_innovation=0.0):
    """Per target window: (log ratio with training memory, log ratio with own-map memory, usable mask, info).

    Each of `target`, `own`, `train` is a window table with `state`, `dyn`, `innov`, `klass` and `key`. The
    search space is fitted on `train`. A window whose true innovation has squared norm <= `min_innovation` has no
    copy-last error to compare against and is not usable.
    """
    space = GapSpace(train["state"], train["dyn"][:, lag_index])

    def feats(w):
        return space.transform(w["state"], w["dyn"][:, lag_index])
    q = feats(target)
    v = np.asarray(target["innov"], np.float64)
    lr, fb = {}, {}
    for name, mem in (("train", train), ("own", own)):
        pred, fb[name] = knn_predict(q, target["klass"], target["key"], feats(mem), mem["klass"], mem["key"],
                                     mem["innov"], k, min_class, block)
        lr[name] = log_error_ratio(pred, v)
    usable = np.sum(v ** 2, axis=1) > float(min_innovation)
    info = {"fallback_train": float(np.mean(fb["train"])) if len(v) else None,
            "fallback_own": float(np.mean(fb["own"])) if len(v) else None,
            "excluded_static": int(np.sum(~usable))}
    return lr["train"], lr["own"], usable, info


# =============================================================================================
# weighted means and the episode bootstrap
# =============================================================================================

def weighted_mean(values, weights):
    w = np.asarray(weights, np.float64)
    return float(np.sum(w * np.asarray(values, np.float64)) / np.sum(w))


def episode_mean(values, episodes):
    """Mean of per-episode means: every episode weighs the same."""
    v, e = np.asarray(values, np.float64), np.asarray(episodes)
    return float(np.mean([v[e == x].mean() for x in np.unique(e)]))


def aggregate(values, weights, episodes, weighting):
    return episode_mean(values, episodes) if weighting == "episode" else weighted_mean(values, weights)


def episode_bootstrap(columns, weights, episodes, draws, rng, weighting="eligible"):
    """(draws, len(columns)) bootstrap of each column's map value, target episodes redrawn with replacement.

    One redraw per row is shared by every column, so differences between columns (G = train - own) are paired.
    """
    e = np.asarray(episodes)
    uniq = np.unique(e)
    w = np.asarray(weights, np.float64)
    cols = [np.asarray(c, np.float64) for c in columns]
    if weighting == "episode":
        per = np.array([[c[e == x].mean() for x in uniq] for c in cols])          # (C, E)
        den = np.ones(len(uniq))
    else:
        per = np.array([[np.sum(w[e == x] * c[e == x]) for x in uniq] for c in cols])
        den = np.array([np.sum(w[e == x]) for x in uniq])
    pick = rng.integers(0, len(uniq), size=(int(draws), len(uniq)))
    return (per[:, pick].sum(axis=2) / den[pick].sum(axis=1)[None, :]).T


def boot_summary(x):
    x = np.asarray(x, np.float64)
    if len(x) == 0:
        return None
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "p2.5": float(np.percentile(x, 2.5)), "p97.5": float(np.percentile(x, 97.5))}


# =============================================================================================
# files: the per-episode cache, set loading, provenance
# =============================================================================================

def code_identity():
    """Commit and content hashes of this module and of `distance_study.py`, written beside every output."""
    here = os.path.abspath(__file__)
    root = os.path.dirname(here)
    out = {}
    for name in ("transition_distance.py", "distance_study.py"):
        with open(os.path.join(root, name), "rb") as f:
            out[name.replace(".py", "_sha256")] = hashlib.sha256(f.read()).hexdigest()
    try:
        r = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        commit = r.stdout.strip() if r.returncode == 0 else None
        s = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", "transition_distance.py",
                            "distance_study.py"], capture_output=True, text=True, timeout=20)
        modified = bool(s.stdout.strip()) if s.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        commit, modified = None, None
    return {"git_commit": commit or "unversioned", "modified": modified, **out}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(ds._jsonable(obj), f, indent=1)
    os.replace(tmp, path)


def _atomic_npz(path, arrays):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.npz"
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _atomic_csv(path, cols, rows):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for row in rows:
            wr.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in cols})
    os.replace(tmp, path)


def set_dir(out, space, name):
    return os.path.join(out, "transitions", space, name)


def episode_file(out, space, name, ep):
    return os.path.join(set_dir(out, space, name), f"ep_{int(ep):05d}.npz")


def read_episode_meta(path):
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["meta"]))


def extract_episode(job):
    """Draw, featurise and cache one episode's windows; returns its index record. A top-level function, so a
    process pool can run it."""
    ep, lat_path, meta_path, path, settings, identity, force, chunk, code = job
    if os.path.isfile(path):
        old = read_episode_meta(path)
        if old.get("settings") == settings and old.get("identity") == identity:
            return {**old["episode"], "file": os.path.basename(path), "reused": True}
        if not force:
            raise SystemExit(f"{path} was drawn with other settings or from another source ({old.get('settings')}, "
                             f"{old.get('identity')}); pass --force to redraw it")
    L, fpe, bins = settings["context_frames"], settings["frames_per_episode"], settings["bins"]
    meta = ds._sidecar(meta_path, columns=("tic", "deaths", "map_id", "buttons"))
    lat = np.load(lat_path, mmap_mode="r")
    ds._check_latents(lat, settings["space"], lat_path)
    if lat.shape[0] != len(meta["tic"]):
        raise SystemExit(f"{meta_path}: {len(meta['tic'])} sidecar rows for {lat.shape[0]} latents")
    rows = ds.eligible_rows(meta, L)
    info = {"id": int(ep), **ds.episode_info(meta, L, rows), "drawn": 0}
    if len(rows) < fpe:
        return {**info, "file": None, "skipped": f"{len(rows)} eligible windows, fewer than {fpe}"}
    motion = ds.episode_motion(lat, chunk)[rows]
    idx, dec = ds.stratified_draw(motion, fpe // bins, bins, np.random.default_rng([int(settings["seed"]), int(ep)]))
    r = rows[idx]
    state, dyn, innov = window_features(lat, r, settings["lags"])
    newest, freq, switch = control_features(control_bits(meta["buttons"]), r, L)
    info["drawn"] = int(len(r))
    arrays = {"state": state, "dyn": dyn, "innov": innov, "newest": newest, "freq": freq, "switch": switch,
              "klass": control_class(newest), "row": r.astype(np.int64), "tic": meta["tic"][r].astype(np.int64),
              "life": ds.life_index(meta["deaths"])[r], "decile": dec.astype(np.int64), "motion": motion[idx]}
    record = {"settings": settings, "identity": identity, "episode": info, "code": code,
              "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "latents": os.path.abspath(lat_path)}
    arrays["meta"] = np.array(json.dumps(ds._jsonable(record)))
    _atomic_npz(path, arrays)
    return {**info, "file": os.path.basename(path), "reused": False}


def split_files(dirs, set_name, seed):
    """{map: (path, split)} of the adaptation split files of one set and seed under `dirs`."""
    out = {}
    for d in dirs or []:
        for path in sorted(glob.glob(os.path.join(d, f"split_adapt_{set_name}_map*_seed{int(seed)}*.json"))):
            with open(path) as f:
                sp = json.load(f)
            meta = sp.get("meta", {})
            if (meta.get("kind") != "adapt_split" or meta.get("set") != set_name
                    or int(meta.get("seed", -1)) != int(seed)):
                continue
            m = int(meta["map"])
            if m in out and out[m][0] != path:
                raise SystemExit(f"two adaptation splits of {set_name} map {m} seed {seed}: {out[m][0]} and {path}; "
                                 "point --splits at a directory holding one per map")
            out[m] = (path, sp)
    return out


def select_episodes(a):
    """[(ep, latents, sidecar)] of the set `transitions` draws, and a description of how they were chosen."""
    from doom_data import list_latent_episodes, parse_episode_ids
    listing = {ep: (lp, mp) for ep, lp, mp in list_latent_episodes(a.latents)}
    how = {"latents": os.path.abspath(a.latents)}
    if a.episodes_per_map:
        def usable(ep):
            return len(ds.eligible_rows(ds._sidecar(listing[ep][1]), a.context_frames)) >= a.frames_per_episode
        ids = parse_episode_ids(a.ids) if a.ids else None
        per_map = min(a.episodes_per_map, a.limit) if a.limit else a.episodes_per_map
        chosen = ds.reference_episodes(a.latents, ids, per_map, a.draw, a.seed, usable)
        how.update(rule="seeded map-balanced draw (distance_study.reference_episodes)", ids=a.ids or None,
                   episodes_per_map=per_map, draw=a.draw)
        return [(ep, *listing[ep]) for ep in chosen], how
    eps, published = ds.listed_episodes(a.latents)
    how.update(rule="every listed episode", published_split=published)
    if a.ids:
        keep = set(parse_episode_ids(a.ids))
        eps = [t for t in eps if t[0] in keep]
        how["ids"] = a.ids
    if a.splits:
        sp = split_files(a.splits, a.set, a.split_seed)
        if not sp:
            raise SystemExit(f"--splits: no adaptation split of set {a.set!r} seed {a.split_seed} under {a.splits}")
        keep = {int(e) for _, s in sp.values() for e in s["meta"]["episodes"]}
        missing = sorted(keep - {t[0] for t in eps})
        if missing:
            raise SystemExit(f"the splits list episodes {missing[:8]} that {a.latents} does not hold")
        eps = [t for t in eps if t[0] in keep]
        how.update(splits=[os.path.abspath(p) for p, _ in sp.values()], split_seed=a.split_seed)
    if a.limit:
        keep = ds._first_per_map([(ep, ds._map_of(mp)) for ep, _, mp in eps], a.limit)
        eps = [t for t in eps if t[0] in keep]
        how["limit"] = a.limit
    return eps, how


def cmd_transitions(a):
    if a.space not in LATENT_SPACES:
        raise SystemExit(f"--space {a.space}: transitions need a latent space {LATENT_SPACES}")
    if not ds.NAME_RE.match(a.set):
        raise SystemExit(f"set name {a.set!r}: letters, digits and underscores only")
    if a.frames_per_episode % a.bins:
        raise SystemExit(f"--frames-per-episode {a.frames_per_episode} is not a multiple of {a.bins} motion bins")
    lags = tuple(sorted({int(x) for x in a.lags.split(",")}))
    if not lags or lags[0] < 1 or lags[-1] > a.context_frames - 1:
        raise SystemExit(f"--lags {a.lags}: every lag must lie in 1 .. {a.context_frames - 1} (inside the context)")
    settings = {"space": a.space, "context_frames": a.context_frames, "lags": list(lags),
                "frames_per_episode": a.frames_per_episode, "bins": a.bins, "seed": a.seed}
    identity = {"set": a.set, "source": os.path.abspath(a.latents)}
    eps, how = select_episodes(a)
    if not eps:
        raise SystemExit(f"{a.latents}: no episode selected")
    code = code_identity()
    jobs = [(ep, lp, mp, episode_file(a.out, a.space, a.set, ep), settings, identity, a.force, a.chunk, code)
            for ep, lp, mp in eps]
    t0 = time.time()
    if a.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
            records = list(pool.map(extract_episode, jobs, chunksize=1))
    else:
        records = [extract_episode(j) for j in jobs]
    index = {"kind": "transition_set", "set": a.set, "space": a.space, "source": identity["source"],
             "settings": settings, "selection": how, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "code": code, "episodes": sorted(records, key=lambda r: r["id"])}
    _atomic_json(os.path.join(set_dir(a.out, a.space, a.set), "index.json"), index)
    print(json.dumps({"wrote": set_dir(a.out, a.space, a.set), "episodes": sum(r["drawn"] > 0 for r in records),
                      "skipped": sum(r["drawn"] == 0 for r in records),
                      "reused": sum(bool(r.get("reused")) for r in records),
                      "windows": int(sum(r["drawn"] for r in records)), "seconds": round(time.time() - t0, 1)}),
          flush=True)
    return 0


def load_set(out, space, name):
    """One cached set as a window table: every `WINDOW_ARRAYS` column plus episode, map and weight per window."""
    d = set_dir(out, space, name)
    path = os.path.join(d, "index.json")
    if not os.path.isfile(path):
        raise SystemExit(f"no transition set {name!r} at {d}; run `distance_study.py transitions --set {name} ...`")
    with open(path) as f:
        index = json.load(f)
    parts = []
    for rec in index["episodes"]:
        if not rec.get("file"):
            continue
        with np.load(os.path.join(d, rec["file"]), allow_pickle=False) as z:
            p = {k: z[k] for k in WINDOW_ARRAYS}
        n = len(p["row"])
        p["episode"] = np.full(n, rec["id"], np.int64)
        p["map"] = np.full(n, rec["map"], np.int64)
        p["weight"] = np.full(n, rec["eligible"] / rec["drawn"], np.float64)
        parts.append(p)
    if not parts:
        raise SystemExit(f"transition set {name!r} holds no drawn episode")
    table = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return {"name": name, "index": index, "windows": table,
            "episodes": {int(r["id"]): r for r in index["episodes"]}}


def check_compatible(sets):
    """Every set scored together must describe windows the same way (space, context, lags, draw size)."""
    ref = None
    for s in sets:
        cfg = {k: s["index"]["settings"][k] for k in SETTING_KEYS}
        if ref is None:
            ref = (s["name"], cfg)
        elif cfg != ref[1]:
            raise SystemExit(f"sets {ref[0]} and {s['name']} were drawn differently: {ref[1]} against {cfg}")
    return ref[1]


def assign_keys(sets):
    """Give every window an integer key naming its episode: (source directory, episode id) pairs, globally."""
    codes = {}
    for s in sets:
        src = s["index"]["source"]
        s["windows"]["key"] = np.array([codes.setdefault((src, int(e)), len(codes)) for e in s["windows"]["episode"]],
                                       dtype=np.int64)
    return codes


def take(table, idx):
    return {k: v[idx] for k, v in table.items()}


def concat(tables):
    return {k: np.concatenate([t[k] for t in tables]) for k in tables[0]}


def target_points(sets, splits_dirs, split_seed, episodes_rule, memory_maps):
    """[(point record, window indices into its set)] of every (set, map) scored as a target."""
    points = []
    for s in sets:
        w = s["windows"]
        sp = split_files(splits_dirs, s["name"], split_seed) if splits_dirs else {}
        for m in sorted(int(x) for x in np.unique(w["map"])):
            rec = {"key": f"{s['name']}/{m}", "set": s["name"], "map": m, "cluster": ds.cluster_of(m),
                   "training": m in memory_maps, "split": None}
            present = sorted(int(e) for e in np.unique(w["episode"][w["map"] == m]))
            if m in sp:
                path, split = sp[m]
                want = split["held_out"] if episodes_rule == "held_out" else split["meta"]["episodes"]
                rec["split"] = os.path.abspath(path)
                rec["missing_episodes"] = sorted({int(e) for e in want} - set(present))
                eps = sorted({int(e) for e in want} & set(present))
            else:
                eps = present
            idx = np.flatnonzero(np.isin(w["episode"], eps) & (w["map"] == m))
            if len(idx):
                rec["episodes"] = eps
                points.append((rec, idx))
    return points


def load_memory(a, sets_needed):
    """Memory sets concatenated, and the target sets, all keyed; refuses sets drawn differently."""
    names = list(dict.fromkeys(list(a.memory) + list(sets_needed)))
    loaded = {n: load_set(a.out, a.space, n) for n in names}
    settings = check_compatible(loaded.values())
    assign_keys(loaded.values())
    mem = concat([loaded[n]["windows"] for n in a.memory])
    return loaded, mem, settings


def default_targets(a):
    root = os.path.join(a.out, "transitions", a.space)
    found = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(root, "*", "index.json")))
    return [n for n in found if n not in set(a.memory)]


def subsample(n, size, rng):
    if not size or size >= n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=int(size), replace=False))


# =============================================================================================
# coverage
# =============================================================================================

def _shuffle_within(idx_groups, n, rng):
    """A permutation of range(n) that permutes indices only within each group."""
    perm = np.arange(n)
    for g in idx_groups:
        perm[g] = g[rng.permutation(len(g))]
    return perm


def cmd_coverage(a):
    targets = a.targets or default_targets(a)
    if not targets:
        raise SystemExit("no target set: pass --targets or draw one with `transitions`")
    loaded, mem_all, settings = load_memory(a, targets)
    controls = tuple(c for c in a.controls.split(",") if c)
    if not set(controls) <= set(CONTROL_FEATURES) or not controls:
        raise SystemExit(f"--controls {a.controls}: pick from {CONTROL_FEATURES}")
    dims = {"state": a.state_dim, "dynamics": a.dynamics_dim, "innovation": a.innovation_dim}
    raw_all = raw_blocks(mem_all, controls)
    space = CoverageSpace(raw_all, dims, a.projection_seed)          # fitted on the whole memory, fixed after
    mem_idx = subsample(len(mem_all["row"]), a.memory_size, np.random.default_rng([a.memory_seed, 0]))
    mem = take(mem_all, mem_idx)
    memory_maps = sorted(int(m) for m in np.unique(mem["map"]))
    mem_blocks = raw_all if len(mem_idx) == len(mem_all["row"]) else {b: v[mem_idx] for b, v in raw_all.items()}
    del raw_all
    feats = {"full": space.transform(mem_blocks), "state": space.transform(mem_blocks, ("state",)),
             "nostate": space.transform(mem_blocks, ("dynamics", "innovation", "controls"))}
    points = target_points([loaded[n] for n in targets], a.splits, a.split_seed, a.target_episodes, memory_maps)
    if not points:
        raise SystemExit("no target windows")
    tw = concat([take(loaded[p["set"]]["windows"], idx) for p, idx in points])
    owner = np.concatenate([np.full(len(idx), i) for i, (_, idx) in enumerate(points)])
    groups = [np.flatnonzero(owner == i) for i in range(len(points))]
    tb = raw_blocks(tw, controls)
    rng_t = np.random.default_rng([a.shuffle_seed, 1])
    rng_c = np.random.default_rng([a.shuffle_seed, 2])
    shuf_t = dict(tb, innovation=tb["innovation"][_shuffle_within(groups, len(owner), rng_t)])
    shuf_c = dict(tb, controls=tb["controls"][_shuffle_within(groups, len(owner), rng_c)])
    dist = not a.no_distinct_episodes
    t0 = time.time()

    def cov(q, m, keys_m, neighbours=False):
        return directed_coverage(q, tw["key"], m, keys_m, a.k, dist, a.query_block, neighbours)
    values = {}
    values["coverage"], nb = cov(space.transform(tb), feats["full"], mem["key"], neighbours=True)
    values["state_only"], _ = cov(space.transform(tb, ("state",)), feats["state"], mem["key"])
    values["without_state"], _ = cov(space.transform(tb, ("dynamics", "innovation", "controls")), feats["nostate"],
                                     mem["key"])
    values["shuffled_transitions"], _ = cov(space.transform(shuf_t), feats["full"], mem["key"])
    values["shuffled_controls"], _ = cov(space.transform(shuf_c), feats["full"], mem["key"])

    # sampling stability: the same score against two disjoint halves of the memory's episodes, split per map
    halves = None
    if not a.no_halves:
        half = np.zeros(len(mem["row"]), dtype=np.int64)
        for m in memory_maps:
            eps = np.unique(mem["key"][mem["map"] == m])
            eps = eps[np.random.default_rng([a.memory_seed, 1, m]).permutation(len(eps))]
            half[np.isin(mem["key"], eps[len(eps) // 2:])] = 1
        if min(len(np.unique(mem["key"][half == h])) for h in (0, 1)) > a.k:
            halves = [cov(space.transform(tb), feats["full"][half == h], mem["key"][half == h])[0] for h in (0, 1)]

    # the train-versus-train floor: every memory window scored against the rest of the memory, own episode out
    floor = None
    if not a.no_floor:
        fsel = subsample(len(mem["row"]), a.floor_queries, np.random.default_rng([a.memory_seed, 2]))
        fv, _ = directed_coverage(feats["full"][fsel], mem["key"][fsel], feats["full"], mem["key"], a.k, dist,
                                  a.query_block)
        fkey, fmap, fw = mem["key"][fsel], mem["map"][fsel], mem["weight"][fsel]
        per_ep = {int(k_): float(fv[fkey == k_].mean()) for k_ in np.unique(fkey)}
        floor = {"windows": int(len(fsel)), "episodes": len(per_ep),
                 "per_map": {str(m): weighted_mean(fv[fmap == m], fw[fmap == m]) for m in memory_maps},
                 "episode_band": [float(min(per_ep.values())), float(max(per_ep.values()))],
                 "what": "each sampled memory window's coverage by the rest of the memory, its own episode excluded"}
    seconds = round(time.time() - t0, 1)

    # per point
    out_points, csv_rows = [], []
    for i, (rec, _) in enumerate(points):
        g = groups[i]
        w, e = tw["weight"][g], tw["episode"][g]
        rec = dict(rec)
        rec.update(n_episodes=len(rec["episodes"]), windows=int(len(g)))
        for name in COVERAGE_VARIANTS:
            rec[name] = aggregate(values[name][g], w, e, a.weighting)
        rec["coverage_eligible_weighted"] = weighted_mean(values["coverage"][g], w)
        rec["coverage_episode_weighted"] = episode_mean(values["coverage"][g], e)
        rec["coverage_motion_weighted"] = weighted_mean(values["coverage"][g], ds.motion_weights(tw["motion"][g]))
        rec["state_share"] = rec["state_only"] / rec["coverage"] if rec["coverage"] > 0 else None
        rec["rise_shuffled_transitions"] = rec["shuffled_transitions"] - rec["coverage"]
        rec["rise_shuffled_controls"] = rec["shuffled_controls"] - rec["coverage"]
        rec["by_decile"] = strata(values["coverage"][g], tw["decile"][g], w)
        rec["by_class"] = strata(values["coverage"][g], tw["klass"][g], w)
        if halves is not None:
            rec["half_a"], rec["half_b"] = (aggregate(h[g], w, e, a.weighting) for h in halves)
        if a.bootstrap > 0:
            rng = np.random.default_rng([a.bootstrap_seed, zlib.crc32(rec["set"].encode()), rec["map"]])
            bt = episode_bootstrap([values[n][g] for n in COVERAGE_VARIANTS], w, e, a.bootstrap, rng, a.weighting)
            rec["bootstrap"] = {n: boot_summary(bt[:, j]) for j, n in enumerate(COVERAGE_VARIANTS)}
        out_points.append(rec)
        csv_rows.append({**rec, **{f"{n}_sd": ((rec.get("bootstrap") or {}).get(n) or {}).get("sd")
                                   for n in COVERAGE_VARIANTS}})

    train_pts = [p for p in out_points if p["training"]]
    new_pts = [p for p in out_points if not p["training"]]
    checks = {
        "training_maps_at_floor": floor_check(train_pts, new_pts, "coverage"),
        "temporal_sensitivity": shuffle_check(out_points, "rise_shuffled_transitions",
                                              "shuffling innovations against contexts raises every map's coverage"),
        "control_sensitivity": shuffle_check(out_points, "rise_shuffled_controls",
                                             "shuffling controls against transitions raises every map's coverage"),
        "state_share": {"what": "state-only coverage over coverage: near 1 means the score is still appearance",
                        "median": _median([p["state_share"] for p in out_points]),
                        "unseen_median": _median([p["state_share"] for p in new_pts])},
    }
    if halves is not None and len(out_points) > 1:
        checks["memory_halves"] = {
            "what": "coverage against two disjoint halves of the memory episodes ranks the maps alike",
            "spearman": ds.spearman([p["half_a"] for p in out_points], [p["half_b"] for p in out_points]),
            "spearman_unseen": ds.spearman([p["half_a"] for p in new_pts], [p["half_b"] for p in new_pts])
            if len(new_pts) > 1 else None}
    if floor is not None:
        lo, hi = floor["episode_band"]
        checks["validation_inside_floor"] = {
            "what": "every training map's target coverage lies inside the band of per-episode train-versus-train "
                    "coverage",
            "band": [lo, hi], "maps": {p["key"]: p["coverage"] for p in train_pts},
            "pass": all(lo <= p["coverage"] <= hi for p in train_pts) if train_pts else None}
    tag = f"_{a.tag}" if a.tag else ""
    stem = os.path.join(a.out, f"coverage_{a.space}{tag}")
    _atomic_npz(stem + "_windows.npz", {
        "point": owner, "keys": np.array([p["key"] for p, _ in points]),
        "set": np.array([points[i][0]["set"] for i in owner]),
        "map": tw["map"], "episode": tw["episode"], "row": tw["row"], "tic": tw["tic"], "klass": tw["klass"],
        "decile": tw["decile"], "motion": tw["motion"], "weight": tw["weight"],
        **{n: values[n] for n in COVERAGE_VARIANTS},
        "neighbour_map": mem["map"][nb], "neighbour_episode": mem["episode"][nb], "neighbour_row": mem["row"][nb],
        "neighbour_key": mem["key"][nb]})
    cols = ["key", "set", "map", "cluster", "training", "n_episodes", "windows", *COVERAGE_VARIANTS,
            *[f"{n}_sd" for n in COVERAGE_VARIANTS], "coverage_eligible_weighted", "coverage_episode_weighted",
            "coverage_motion_weighted", "state_share", "rise_shuffled_transitions", "rise_shuffled_controls",
            "half_a", "half_b"]
    _atomic_csv(stem + ".csv", cols, csv_rows)
    result = {"kind": "transition_coverage", "space": a.space, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "code": code_identity(), "seconds": seconds,
              "config": {"memory": list(a.memory), "targets": targets, "k": a.k, "distinct_episodes": dist,
                         "controls": list(controls), "projection_dims": dims, "projection_seed": a.projection_seed,
                         "memory_size": a.memory_size, "memory_seed": a.memory_seed, "weighting": a.weighting,
                         "bootstrap": a.bootstrap, "bootstrap_seed": a.bootstrap_seed, "shuffle_seed": a.shuffle_seed,
                         "splits": [os.path.abspath(d) for d in a.splits or []], "split_seed": a.split_seed,
                         "target_episodes": a.target_episodes, "halves": halves is not None,
                         "floor_queries": None if a.no_floor else a.floor_queries, "query_block": a.query_block,
                         "window_settings": settings},
              "memory": {"sets": {n: loaded[n]["index"]["source"] for n in a.memory}, "maps": memory_maps,
                         "windows": int(len(mem["row"])), "windows_before_subsample": int(len(mem_all["row"])),
                         "episodes": int(len(np.unique(mem["key"]))),
                         "per_map_windows": {str(m): int(np.sum(mem["map"] == m)) for m in memory_maps},
                         "metric": space.describe()},
              "floor": floor, "checks": checks, "points": out_points,
              "outputs": {"csv": os.path.abspath(stem + ".csv"), "windows": os.path.abspath(stem + "_windows.npz")}}
    _atomic_json(stem + ".json", result)
    print(json.dumps({"wrote": stem + ".json", "points": len(out_points), "seconds": seconds,
                      "floor_check": checks["training_maps_at_floor"]["pass"]}), flush=True)
    return 0


def strata(values, labels, weights):
    """{label: {"n", "mean"}} of a map's per-window values within each motion decile or control class."""
    return {str(int(x)): {"n": int(np.sum(labels == x)),
                          "mean": weighted_mean(values[labels == x], weights[labels == x])} for x in np.unique(labels)}


def shuffle_check(points, field, what):
    """A shuffle must raise coverage: on every map (`pass`) and on the training maps' held-out episodes, whose
    pairing the memory knows (`pass_training`). A map whose own pairing runs against the memory's (the same control
    producing the opposite change) can fall under a shuffle, since a random pairing lies nearer the memory."""
    train = [p for p in points if p["training"]]
    return {"what": what, "rises": {p["key"]: p[field] for p in points},
            "positive": int(sum(p[field] > 0 for p in points)), "maps": len(points),
            "pass": all(p[field] > 0 for p in points),
            "pass_training": all(p[field] > 0 for p in train) if train else None}


def _median(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def floor_check(train_pts, new_pts, field):
    """Every training map's value sits below every unseen map's: the distance puts them at the floor."""
    tv = [p[field] for p in train_pts if p.get(field) is not None and np.isfinite(p[field])]
    nv = [p[field] for p in new_pts if p.get(field) is not None and np.isfinite(p[field])]
    ok = bool(max(tv) < min(nv)) if tv and nv else None
    return {"what": f"every training map's {field} lies below every unseen map's",
            "training": {p["key"]: p.get(field) for p in train_pts}, "training_max": max(tv) if tv else None,
            "unseen_min": min(nv) if nv else None, "training_maps": len(tv), "pass": ok}


# =============================================================================================
# transfer gap
# =============================================================================================

def cmd_transfer_gap(a):
    targets = a.targets or default_targets(a)
    if not a.splits:
        raise SystemExit("--splits: the transfer gap scores each map's held-out episodes against its adaptation "
                         "episodes, so it needs the adaptation split files (adapt_split.py)")
    loaded, mem_all, settings = load_memory(a, targets)
    if 1 not in settings["lags"]:
        raise SystemExit("the transfer gap searches on the previous one-tic change: the sets need lag 1")
    lag_index = settings["lags"].index(1)
    memory_maps = sorted(int(m) for m in np.unique(mem_all["map"]))
    out_points, csv_rows, windows, unsplit = [], [], [], []
    subsamples = {}
    t0 = time.time()
    for name in targets:
        s = loaded[name]
        w = s["windows"]
        found = split_files(a.splits, name, a.split_seed)
        if not found:
            unsplit.append(name)
        for m, (path, split) in sorted(found.items()):
            pool = [int(e) for e in (split.get("adapt_pool") or split["adapt"])]
            own_eps = pool[:a.own_episodes_k] if a.own_episodes_k else [int(e) for e in split["adapt"]]
            held = [int(e) for e in split["held_out"]]
            rec = {"key": f"{name}/{m}", "set": name, "map": m, "cluster": ds.cluster_of(m),
                   "training": m in memory_maps, "split": os.path.abspath(path), "split_sha256": sha256_file(path),
                   "held_out": held, "own_episodes": sorted(own_eps)}
            present = {int(e) for e in np.unique(w["episode"][w["map"] == m])}
            rec["missing_episodes"] = sorted((set(held) | set(own_eps)) - present)
            ti = np.flatnonzero((w["map"] == m) & np.isin(w["episode"], held))
            oi = np.flatnonzero((w["map"] == m) & np.isin(w["episode"], own_eps))
            if not len(ti) or len(oi) < a.k:
                rec["skipped"] = f"{len(ti)} held-out and {len(oi)} own-memory windows"
                out_points.append(rec)
                continue
            if a.gap_memory_size == 0:
                size = len(oi)
            else:
                size = len(mem_all["row"]) if a.gap_memory_size < 0 else a.gap_memory_size
            if size not in subsamples:       # one training subsample per size, shared by every map of that size
                subsamples[size] = subsample(len(mem_all["row"]), size, np.random.default_rng([a.memory_seed, size]))
            train = take(mem_all, subsamples[size])
            tgt, own = take(w, ti), take(w, oi)
            lr_t, lr_o, usable, info = transfer_gap(tgt, own, train, a.k, a.min_class, lag_index, a.query_block,
                                                    a.min_innovation)
            keep = usable
            wt, ep = tgt["weight"][keep], tgt["episode"][keep]
            rec.update(windows=int(len(ti)), usable_windows=int(keep.sum()), own_memory_windows=int(len(oi)),
                       train_memory_windows=int(len(subsamples[size])), n_held_out=len(held), **info)
            if keep.any():
                rec["L_train"] = aggregate(lr_t[keep], wt, ep, a.weighting)
                rec["L_own"] = aggregate(lr_o[keep], wt, ep, a.weighting)
                rec["G"] = rec["L_train"] - rec["L_own"]
                rec["G_eligible_weighted"] = weighted_mean(lr_t[keep] - lr_o[keep], wt)
                rec["G_episode_weighted"] = episode_mean(lr_t[keep] - lr_o[keep], ep)
                if a.bootstrap > 0:
                    rng = np.random.default_rng([a.bootstrap_seed, zlib.crc32(name.encode()), m])
                    bt = episode_bootstrap([lr_t[keep], lr_o[keep]], wt, ep, a.bootstrap, rng, a.weighting)
                    rec["bootstrap"] = {"L_train": boot_summary(bt[:, 0]), "L_own": boot_summary(bt[:, 1]),
                                        "G": boot_summary(bt[:, 0] - bt[:, 1])}
            windows.append({"key": rec["key"], "episode": tgt["episode"], "row": tgt["row"], "klass": tgt["klass"],
                            "weight": tgt["weight"], "usable": usable, "log_ratio_train": lr_t, "log_ratio_own": lr_o})
            out_points.append(rec)
            csv_rows.append({**rec, **{f"{n}_sd": ((rec.get("bootstrap") or {}).get(n) or {}).get("sd")
                                       for n in ("L_train", "L_own", "G")}})
    if not windows:
        raise SystemExit("no map had an adaptation split with held-out and own-memory windows")
    seconds = round(time.time() - t0, 1)
    scored = [p for p in out_points if "G" in p]
    checks = {"training_maps_at_floor": floor_check([p for p in scored if p["training"]],
                                                    [p for p in scored if not p["training"]], "G")}
    tag = f"_{a.tag}" if a.tag else ""
    stem = os.path.join(a.out, f"transfer_gap_{a.space}{tag}")
    _atomic_npz(stem + "_windows.npz", {
        "keys": np.array([x["key"] for x in windows]),
        "point": np.concatenate([np.full(len(x["episode"]), i) for i, x in enumerate(windows)]),
        **{k: np.concatenate([x[k] for x in windows]) for k in ("episode", "row", "klass", "weight", "usable",
                                                                "log_ratio_train", "log_ratio_own")}})
    cols = ["key", "set", "map", "cluster", "training", "n_held_out", "windows", "usable_windows", "excluded_static",
            "own_memory_windows", "train_memory_windows", "L_train", "L_own", "G", "L_train_sd", "L_own_sd", "G_sd",
            "G_eligible_weighted", "G_episode_weighted", "fallback_train", "fallback_own", "skipped"]
    _atomic_csv(stem + ".csv", cols, csv_rows)
    result = {"kind": "transfer_gap", "space": a.space, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "code": code_identity(), "seconds": seconds,
              "config": {"memory": list(a.memory), "targets": targets, "k": a.k, "min_class": a.min_class,
                         "own_episodes_k": a.own_episodes_k, "gap_memory_size": a.gap_memory_size,
                         "memory_seed": a.memory_seed, "min_innovation": a.min_innovation, "weighting": a.weighting,
                         "bootstrap": a.bootstrap, "bootstrap_seed": a.bootstrap_seed,
                         "splits": [os.path.abspath(d) for d in a.splits], "split_seed": a.split_seed,
                         "search_space": "state and previous one-tic change, z-scored on the training memory, "
                                         "each block at unit total variance", "query_block": a.query_block,
                         "window_settings": settings},
              "memory": {"sets": {n: loaded[n]["index"]["source"] for n in a.memory}, "maps": memory_maps,
                         "windows": int(len(mem_all["row"])), "episodes": int(len(np.unique(mem_all["key"])))},
              "checks": checks, "points": out_points, "sets_without_splits": unsplit,
              "outputs": {"csv": os.path.abspath(stem + ".csv"), "windows": os.path.abspath(stem + "_windows.npz")}}
    _atomic_json(stem + ".json", result)
    print(json.dumps({"wrote": stem + ".json", "points": len(scored), "seconds": seconds, "unsplit": unsplit,
                      "floor_check": checks["training_maps_at_floor"]["pass"]}), flush=True)
    return 0


# =============================================================================================
# compare: D, coverage and G side by side
# =============================================================================================

COMPARE_COLUMNS = ["key", "set", "map", "cluster", "training", "D", "D_sd", "D_sd_kind", "coverage", "coverage_sd",
                   "coverage_p2.5", "coverage_p97.5", "state_only", "without_state", "shuffled_transitions",
                   "shuffled_controls", "state_share", "G", "G_sd", "G_p2.5", "G_p97.5", "L_train", "L_own",
                   "coverage_windows", "gap_windows"]
PAIRS = (("D", "coverage"), ("D", "G"), ("coverage", "G"), ("D", "state_only"), ("coverage", "state_only"))


def frame_distances(path):
    """{map: (D, bootstrap SD, which bootstrap)} of the frozen frame distance, primary records only."""
    with open(path) as f:
        d = json.load(f)
    out = {}
    for rec in d["maps"]:
        if rec.get("role", "primary") != "primary":
            continue
        b = rec.get("bootstrap_full") or rec.get("bootstrap") or {}
        kind = "bootstrap_full" if rec.get("bootstrap_full") else ("bootstrap" if rec.get("bootstrap") else None)
        out[int(rec["map"])] = (float(rec["D"]), b.get("sd"), kind)
    return out, d.get("floor"), d.get("primary_arm")


def cmd_compare(a):
    cov_path = a.coverage or os.path.join(a.out, f"coverage_{a.space}.json")
    gap_path = a.transfer_gap or os.path.join(a.out, f"transfer_gap_{a.space}.json")
    with open(cov_path) as f:
        cov = json.load(f)
    gap = None
    if os.path.isfile(gap_path):
        with open(gap_path) as f:
            gap = json.load(f)
    elif a.transfer_gap:
        raise SystemExit(f"no transfer-gap file at {gap_path}")
    Dmap, d_floor, d_arm = frame_distances(a.frame_distances)
    gaps = {p["key"]: p for p in (gap or {}).get("points", [])}
    rows = []
    for p in cov["points"]:
        D = Dmap.get(p["map"])
        g = gaps.get(p["key"], {})
        cb = (p.get("bootstrap") or {}).get("coverage") or {}
        gb = (g.get("bootstrap") or {}).get("G") or {}
        rows.append({"key": p["key"], "set": p["set"], "map": p["map"], "cluster": p["cluster"],
                     "training": p["training"],
                     "D": D[0] if D else None, "D_sd": D[1] if D else None, "D_sd_kind": D[2] if D else None,
                     "coverage": p["coverage"], "coverage_sd": cb.get("sd"), "coverage_p2.5": cb.get("p2.5"),
                     "coverage_p97.5": cb.get("p97.5"), "state_only": p["state_only"],
                     "without_state": p["without_state"],
                     "shuffled_transitions": p["shuffled_transitions"], "shuffled_controls": p["shuffled_controls"],
                     "state_share": p.get("state_share"), "G": g.get("G"), "G_sd": gb.get("sd"),
                     "G_p2.5": gb.get("p2.5"), "G_p97.5": gb.get("p97.5"), "L_train": g.get("L_train"),
                     "L_own": g.get("L_own"), "coverage_windows": p["windows"], "gap_windows": g.get("usable_windows")})

    def rank_agreement(sel):
        out = {}
        for x, y in PAIRS:
            pts = [(r[x], r[y]) for r in sel if r[x] is not None and r[y] is not None]
            out[f"{x}~{y}"] = {"n": len(pts), "spearman": ds.spearman(*zip(*pts)) if len(pts) > 2 else None}
        return out
    train_rows = [r for r in rows if r["training"]]
    new_rows = [r for r in rows if not r["training"]]
    floor = {f: floor_check(train_rows, new_rows, f) for f in ("D", "coverage", "G")}
    if d_floor and d_arm in d_floor and train_rows:
        hi = d_floor[d_arm]["max"]
        floor["D"]["frozen_floor_max"] = hi
        floor["D"]["training_inside_frozen_floor"] = all(r["D"] is not None and r["D"] <= hi for r in train_rows)
    result = {"kind": "transition_distance_compare", "name": a.name, "space": a.space,
              "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code": code_identity(),
              "inputs": {"coverage": {"file": os.path.abspath(cov_path), "sha256": sha256_file(cov_path),
                                      "config": cov.get("config")},
                         "transfer_gap": {"file": os.path.abspath(gap_path), "sha256": sha256_file(gap_path),
                                          "config": gap.get("config")} if gap else None,
                         "frame_distances": {"file": os.path.abspath(a.frame_distances),
                                             "sha256": sha256_file(a.frame_distances), "primary_arm": d_arm}},
              "spearman": {"all": rank_agreement(rows), "unseen": rank_agreement(new_rows)},
              "floor": floor, "floor_all_pass": all(v["pass"] for v in floor.values() if v["pass"] is not None),
              "maps": rows}
    stem = os.path.join(a.out, f"compare_{a.name}_{a.space}")
    _atomic_csv(stem + ".csv", COMPARE_COLUMNS, rows)
    result["outputs"] = {"csv": os.path.abspath(stem + ".csv")}
    _atomic_json(stem + ".json", result)
    print(json.dumps({"wrote": stem + ".json", "maps": len(rows),
                      "floor": {k: v["pass"] for k, v in floor.items()},
                      "spearman_unseen": {k: v["spearman"] for k, v in result["spearman"]["unseen"].items()}}),
          flush=True)
    return 0


# =============================================================================================
# command line (registered as subcommands of distance_study.py)
# =============================================================================================

def _common(p, memory=True):
    p.add_argument("--space", default="sd1", choices=list(LATENT_SPACES))
    p.add_argument("--out", default="results/distance_study")
    if memory:
        p.add_argument("--memory", nargs="+", default=["train"],
                       help="transition sets that form the training memory (default: train)")
        p.add_argument("--targets", nargs="+", default=None,
                       help="transition sets to score (default: every drawn set that is not memory)")
        p.add_argument("--splits", nargs="+", default=None, metavar="DIR",
                       help="directories of adapt_split.py files (split_adapt_<set>_map<NN>_seed<S>.json)")
        p.add_argument("--split-seed", dest="split_seed", type=int, default=0)
        p.add_argument("--memory-seed", dest="memory_seed", type=int, default=0)
        p.add_argument("--weighting", choices=WEIGHTINGS, default="eligible",
                       help="primary map mean: windows weighted by eligible/drawn (default) or equal per episode")
        p.add_argument("--bootstrap", type=int, default=BOOTSTRAP, help="target-episode redraws per map")
        p.add_argument("--bootstrap-seed", dest="bootstrap_seed", type=int, default=0)
        p.add_argument("--query-block", dest="query_block", type=int, default=QUERY_BLOCK,
                       help="queries per block of the exact search (memory use, not results)")
        p.add_argument("--tag", default="", help="suffix for the output file names")


def add_parsers(sub):
    """Register `transitions`, `coverage`, `transfer-gap` and `compare` on distance_study's subparsers."""
    t = sub.add_parser("transitions", help="draw and cache per-episode transition windows (CPU)")
    _common(t, memory=False)
    t.add_argument("--set", required=True, help="the set's name (train, val, arenas13, ...)")
    t.add_argument("--latents", required=True, help="the per-tic latent directory")
    t.add_argument("--ids", default="", help="episode ids to draw from, A:B or a list")
    t.add_argument("--episodes-per-map", dest="episodes_per_map", type=int, default=0,
                   help="a seeded map-balanced draw of this many episodes per map (the memory; 50 is D's reference)")
    t.add_argument("--draw", type=int, default=1, help="with --episodes-per-map: 1, or 2 for the disjoint second draw")
    t.add_argument("--splits", nargs="+", default=None, metavar="DIR",
                   help="restrict to the episodes the set's adaptation split files list")
    t.add_argument("--split-seed", dest="split_seed", type=int, default=0)
    t.add_argument("--frames-per-episode", dest="frames_per_episode", type=int, default=ds.FRAMES_PER_EPISODE,
                   help="windows per episode (default 250)")
    t.add_argument("--bins", type=int, default=ds.MOTION_BINS, help="motion strata per episode (default 10)")
    t.add_argument("--context-frames", dest="context_frames", type=int, default=ds.CONTEXT_FRAMES)
    t.add_argument("--lags", default=",".join(map(str, LAGS)), help="context dynamics lags (default 1,4,16,31)")
    t.add_argument("--seed", type=int, default=0, help="the window draw, and the episode draw with --episodes-per-map")
    t.add_argument("--chunk", type=int, default=ds.MOTION_CHUNK, help="latent rows read at a time for motion")
    t.add_argument("--limit", type=int, default=0, help="smoke runs: at most this many episodes per map")
    t.add_argument("--workers", type=int, default=1, help="episodes extracted in parallel processes")
    t.add_argument("--force", action="store_true", help="redraw cached episodes whose settings differ")

    c = sub.add_parser("coverage", help="candidate 1: directed coverage of transition windows by the memory")
    _common(c)
    c.add_argument("--k", type=int, default=K_COVERAGE)
    c.add_argument("--no-distinct-episodes", dest="no_distinct_episodes", action="store_true",
                   help="diagnostic only: let several neighbours come from one memory episode")
    c.add_argument("--state-dim", dest="state_dim", type=int, default=PROJECTION_DIMS["state"])
    c.add_argument("--dynamics-dim", dest="dynamics_dim", type=int, default=PROJECTION_DIMS["dynamics"])
    c.add_argument("--innovation-dim", dest="innovation_dim", type=int, default=PROJECTION_DIMS["innovation"])
    c.add_argument("--projection-seed", dest="projection_seed", type=int, default=0)
    c.add_argument("--controls", default=",".join(CONTROL_FEATURES), help="control features (newest,freq,switch)")
    c.add_argument("--memory-size", dest="memory_size", type=int, default=0,
                   help="memory windows kept by a seeded subsample (0: all)")
    c.add_argument("--target-episodes", dest="target_episodes", choices=("all", "held_out"), default="all",
                   help="with --splits: score every episode of the map (default) or its held-out ones")
    c.add_argument("--shuffle-seed", dest="shuffle_seed", type=int, default=0)
    c.add_argument("--no-halves", dest="no_halves", action="store_true", help="skip the memory-halves stability check")
    c.add_argument("--no-floor", dest="no_floor", action="store_true", help="skip the train-versus-train floor")
    c.add_argument("--floor-queries", dest="floor_queries", type=int, default=20000,
                   help="memory windows scored for the floor (0: all)")

    g = sub.add_parser("transfer-gap", help="candidate 2: the kNN transfer gap G")
    _common(g)
    g.add_argument("--k", type=int, default=K_GAP)
    g.add_argument("--min-class", dest="min_class", type=int, default=MIN_CLASS)
    g.add_argument("--own-episodes-k", dest="own_episodes_k", type=int, default=0,
                   help="own-map memory: the first K of the split's adapt pool (0: the whole adapt list)")
    g.add_argument("--gap-memory-size", dest="gap_memory_size", type=int, default=0,
                   help="training memory windows: 0 equal to the own-map memory, -1 all, N a fixed size")
    g.add_argument("--min-innovation", dest="min_innovation", type=float, default=0.0,
                   help="windows whose true pooled change has squared norm <= this are excluded (default: exact zeros)")

    m = sub.add_parser("compare", help="D, coverage and G per map: one CSV and JSON")
    _common(m, memory=False)
    m.add_argument("--name", default="fresh", help="the corpus label in the output names")
    m.add_argument("--coverage", default="", help="coverage JSON (default <out>/coverage_<space>.json)")
    m.add_argument("--transfer-gap", dest="transfer_gap", default="",
                   help="transfer-gap JSON (default <out>/transfer_gap_<space>.json, skipped if absent)")
    m.add_argument("--frame-distances", dest="frame_distances", default="results/distance_study/distances_sd1.json",
                   help="the frozen frame distance D")


COMMANDS = {"transitions": cmd_transitions, "coverage": cmd_coverage, "transfer-gap": cmd_transfer_gap,
            "compare": cmd_compare}
