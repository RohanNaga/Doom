"""Distributional distance between evaluation footage and training footage.

The distance study asks whether a world model's quality on a map falls with how far that map's
footage sits from the training footage. Design, decisions and statistics:
`.claude/analyses/distance-study-design-2026-09-24.md`. This module is the metric core, numpy only:

  * a map (or an episode) is a CLOUD of frames, one point per sampled tic, in some feature space;
  * each frame is WEIGHTED by how much the latent moved into it, with a floor so the quietest half
    of the cloud still holds a quarter of the weight (Rohan's NVIDIA rule, `motion_weights`);
  * the distance between two weighted clouds is the SLICED WASSERSTEIN distance: project both clouds
    onto many random unit directions, solve optimal transport exactly in 1-D on each direction, and
    average (`sliced_wasserstein`);
  * the primary study distance is to the NEAREST TRAINING MAP, not to the pooled training corpus,
    because Wasserstein to a mixture charges a seen map for not also looking like the other maps
    (`nearest_reference_distance`; the pooled form is `pooled_reference`).

There is no time coordinate: an evaluation map and the training corpus share no clock to align, so
what the NVIDIA method's time axis did is carried here by the motion weights and, in the feature
space of the world model's own context, by the 32-tic context itself.

Every distance computed with the same `seed` and feature dimension uses the same projection
directions, so two maps' distances differ by their data, not by their Monte Carlo draw.

    from distance_study import motion_weights, nearest_reference_distance, pool_latents
    feats = pool_latents(latents)                     # (N, 192) from (N, 4, 32, 40) SD 1.x latents
    d = nearest_reference_distance(feats, refs, x_weights=motion_weights(motion))

The study pipeline runs on the CPU beside the latents, as three subcommands (memo section 6):

    python distance_study.py clouds --space sd1 --out results/distance_study \
        --reference $D/latents_arnold_dense_pertic/arenas --reference-ids 0:2000 --draw 1 \
        --eval val=$D/latents_arnold_dense_pertic_eval/val ...
    python distance_study.py splits --out results/distance_study --eval val=... --eval unseen2=...
    python distance_study.py distances --space sd1 --out results/distance_study --bootstrap 200

`clouds` cuts every episode into lives the way the scored windows are cut, draws 250 target-eligible
frames per episode (25 per motion decile) and writes `clouds/<space>/<set>.npz`; the reference is 50
seeded episodes per training map, and `--draw 2` is a disjoint second draw for the noise floor.
`--frames-from` carries one space's draw into another latent space or into raw pixels by tic.
`splits` writes one `make_dense_eval_splits.py`-format split file per map for `eval_tf.py`.
`distances` writes `distances_<space>.json`, `per_episode_<space>.csv` and, with `--bootstrap N`,
`bootstrap_<space>.npz`. Everything reads latents through numpy memory maps, one episode and one
chunk of rows at a time, so the roughly 10 GB of reference latents are never resident at once.
"""
import argparse
import csv
import glob
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import zlib
from itertools import combinations

import numpy as np

QUIET_SHARE = 0.25          # the quietest half of a cloud keeps this share of the weight
N_PROJECTIONS = 1000
VISIBLE_ROWS = 30           # 240 pixel rows / 8; the encoder pads the bottom to 256 (encode_parquet.py:161)
POOL_BLOCK = 5              # 30 x 40 latent cells -> 6 x 8 blocks
MAX_ELEMENTS = 4_000_000    # bound on (points x projections) held at once, about 32 MB per float64 array


def random_directions(dim, n_projections=N_PROJECTIONS, seed=0):
    """(dim, n_projections) unit vectors, uniform on the sphere, reproducible from `seed`."""
    if dim < 1 or n_projections < 1:
        raise ValueError(f"need dim >= 1 and n_projections >= 1, got {dim} and {n_projections}")
    d = np.random.default_rng(seed).standard_normal((int(dim), int(n_projections)))
    return d / np.linalg.norm(d, axis=0, keepdims=True)


def _weights(w, n, name):
    """Normalised non-negative weights for n points; uniform when w is None."""
    if w is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if w.shape[0] != n:
        raise ValueError(f"{name}: {w.shape[0]} weights for {n} points")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError(f"{name}: weights must be finite and non-negative")
    total = w.sum()
    if total <= 0:
        raise ValueError(f"{name}: the weights carry no mass")
    return w / total


def _transport_cost(xp, yp, wx, wy, p):
    """Per-column W_p^p between weighted 1-D samples: xp (n, L) and yp (m, L) projected values.

    The exact 1-D optimal transport matches quantiles: W_p^p = integral over u in (0, 1] of
    |F^-1(u) - G^-1(u)|^p. Both quantile functions are step functions whose steps sit at the
    cumulative weights, so the integral is a sum over the merged breakpoints of the two clouds. On
    each merged interval, the quantile index of a cloud is the number of that cloud's breakpoints
    strictly before it in the merged order; ties contribute zero-width intervals.
    """
    n, m = xp.shape[0], yp.shape[0]
    ix = np.argsort(xp, axis=0, kind="stable")
    iy = np.argsort(yp, axis=0, kind="stable")
    xs, ys = np.take_along_axis(xp, ix, 0), np.take_along_axis(yp, iy, 0)
    cx, cy = np.cumsum(wx[ix], axis=0), np.cumsum(wy[iy], axis=0)
    cx /= cx[-1:]
    cy /= cy[-1:]
    u = np.concatenate([cx, cy], axis=0)
    order = np.argsort(u, axis=0, kind="stable")        # x breakpoints first on a tie
    us = np.take_along_axis(u, order, 0)
    from_x = order < n
    before_x = np.cumsum(from_x, axis=0) - from_x
    before_y = np.cumsum(~from_x, axis=0) - ~from_x
    qx = np.take_along_axis(xs, np.minimum(before_x, n - 1), 0)
    qy = np.take_along_axis(ys, np.minimum(before_y, m - 1), 0)
    du = np.diff(us, axis=0, prepend=0.0)
    return np.sum(du * np.abs(qx - qy) ** p, axis=0)


def sliced_wasserstein(x, y, x_weights=None, y_weights=None, p=2, n_projections=N_PROJECTIONS, seed=0,
                       directions=None, max_elements=MAX_ELEMENTS):
    """Sliced p-Wasserstein distance between two weighted clouds of points.

    SW_p(mu, nu) = ( mean over directions theta of W_p^p(theta . mu, theta . nu) )^(1/p), with each
    1-D term solved exactly. x (n, d) and y (m, d) share the feature space; weights are relative
    masses per point (None is uniform). `directions` (d, L) overrides `n_projections` and `seed`;
    otherwise they are drawn by `random_directions(d, n_projections, seed)`, so equal seeds give
    common directions across calls. Projections run in chunks of at most `max_elements` values per
    array, which bounds memory without changing the result.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(f"clouds must be (n, d) and (m, d) in one feature space, got {x.shape} and {y.shape}")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("an empty cloud has no distribution")
    if p < 1:
        raise ValueError(f"p = {p}: the Wasserstein distance is a metric only for p >= 1")
    wx, wy = _weights(x_weights, len(x), "x_weights"), _weights(y_weights, len(y), "y_weights")
    dirs = random_directions(x.shape[1], n_projections, seed) if directions is None else np.asarray(directions, float)
    if dirs.ndim != 2 or dirs.shape[0] != x.shape[1]:
        raise ValueError(f"directions must be ({x.shape[1]}, L), got {dirs.shape}")
    chunk = max(1, int(max_elements) // (len(x) + len(y)))
    cost = np.concatenate([_transport_cost(x @ dirs[:, i:i + chunk], y @ dirs[:, i:i + chunk], wx, wy, p)
                           for i in range(0, dirs.shape[1], chunk)])
    return float(np.mean(cost) ** (1.0 / p))


def latent_motion(segment):
    """Per-frame motion ||z_t - z_(t-1)|| along one contiguous life of consecutive tics.

    `segment` is (T, ...) with T >= 2, one latent (or feature vector) per tic. The first frame has no
    predecessor in its life, so it takes the second frame's motion; a respawn or a tic gap must
    therefore start a new segment, or the jump across it would count as motion.
    """
    z = np.asarray(segment, dtype=np.float64)
    if z.shape[0] < 2:
        raise ValueError(f"motion needs at least two consecutive frames, got {z.shape[0]}")
    step = np.linalg.norm(np.diff(z.reshape(z.shape[0], -1), axis=0), axis=1)
    return np.concatenate([step[:1], step])


def motion_weights(motion, quiet_share=QUIET_SHARE):
    """Weights proportional to motion plus the smallest floor that gives the quietest half `quiet_share`.

    w_t is proportional to m_t + c. Adding c moves the quiet half's share monotonically towards
    one half, so c has a closed form: with S the total motion, S_q the quiet half's, h frames in the
    quiet half and n frames in all, (S_q + h c) / (S + n c) = q gives c = (q S - S_q) / (h - q n).
    A cloud whose quiet half already holds the share gets c = 0 (motion alone); q >= h / n is
    uniform weighting, which is the sensitivity arm of the study. Returns weights summing to one.
    """
    m = np.asarray(motion, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(m)) or np.any(m < 0):
        raise ValueError("motion must be finite and non-negative")
    if not 0.0 <= quiet_share <= 0.5:
        raise ValueError(f"quiet_share {quiet_share} is outside [0, 0.5]: the quietest half cannot hold "
                         "more than half of the weight while motion still ranks the frames")
    n = m.shape[0]
    total = m.sum()
    if total <= 0:
        raise ValueError("no frame in the cloud moved; identical frames usually mean a broken sidecar")
    h = n // 2
    quiet = np.sort(m)[:h].sum()
    if h == 0 or quiet_share * n >= h:
        return np.full(n, 1.0 / n)
    floor = max(0.0, (quiet_share * total - quiet) / (h - quiet_share * n))
    w = m + floor
    return w / w.sum()


def pool_latents(latents, rows=VISIBLE_ROWS, block=POOL_BLOCK):
    """Feature space (b): VAE latents with the padding rows removed, average-pooled in blocks.

    (N, C, H, W) with H >= `rows` -> (N, C * rows/block * W/block) float32. For the SD 1.x corpus
    (4, 32, 40) that is 4 x 6 x 8 = 192 dimensions; for SD 3.5 (16, 32, 40) it is 768. The bottom
    two latent rows are the encoder's zero padding and would otherwise be a constant shared by every
    map; the HUD (bottom 32 pixel rows, latent rows 26 to 29) is kept because the metric scores it.
    """
    z = np.asarray(latents)
    if z.ndim != 4:
        raise ValueError(f"latents must be (N, C, H, W), got {z.shape}")
    n, c, hgt, wid = z.shape
    if hgt < rows:
        raise ValueError(f"{hgt} latent rows, fewer than the {rows} visible ones")
    if rows % block or wid % block:
        raise ValueError(f"a {block} x {block} block does not tile {rows} x {wid}")
    z = z[:, :, :rows].astype(np.float32)
    return z.reshape(n, c, rows // block, block, wid // block, block).mean(axis=(3, 5)).reshape(n, -1)


def _cloud(entry):
    """A reference given as an array (uniform weights) or as (array, weights)."""
    if isinstance(entry, tuple):
        return entry[0], entry[1]
    return entry, None


def pooled_reference(references):
    """The whole training corpus as one cloud, each reference holding an equal share of the mass.

    This is the literal "distance to the training footage", kept as a secondary: it is the right
    object only when every evaluation map should resemble the whole corpus, which a seen map does not.
    """
    xs, ws = [], []
    for entry in references.values():
        x, w = _cloud(entry)
        xs.append(np.asarray(x, dtype=np.float64))
        ws.append(_weights(w, len(x), "reference weights") / len(references))
    return np.concatenate(xs), np.concatenate(ws)


def nearest_reference_distance(x, references, x_weights=None, **kwargs):
    """Distance from one cloud to its nearest reference cloud, with every per-reference distance.

    `references` maps a name (a training map id) to an array or to (array, weights). All distances
    use the same directions (same seed and dimension). Returns
    {"distance": min, "nearest": name of the minimiser, "per_reference": {name: distance}}.
    """
    if not references:
        raise ValueError("no reference clouds")
    per = {}
    for name, entry in references.items():
        ref, w = _cloud(entry)
        per[name] = sliced_wasserstein(x, ref, x_weights=x_weights, y_weights=w, **kwargs)
    nearest = min(per, key=per.get)
    return {"distance": per[nearest], "nearest": nearest, "per_reference": per}


# =============================================================================================
# The study pipeline (memo section 6): clouds, splits, distances. CPU only, numpy memory maps.
# =============================================================================================

# latent channels and pooled feature dimension per space; pixels are RGB block means at 20 x 15
SPACES = {"sd1": {"channels": 4, "dim": 192}, "sd35": {"channels": 16, "dim": 768},
          "pixels": {"channels": None, "dim": 900}}
CONTEXT_FRAMES = 32         # a target is eligible only with a full 32-tic context of its own life behind it
FRAMES_PER_EPISODE = 250    # frames drawn from every episode, reference or evaluation
MOTION_BINS = 10            # ... in equal numbers from each motion decile of the episode
EPISODES_PER_MAP = 50       # reference episodes per training map and draw
EPISODES_PER_CLOUD = 4      # the smallest per-map episode count (maps 1 and 9 to 15)
SUBSETS = 10                # 4-episode subsets averaged per map, drawn as five disjoint pairs
NUM_WINDOWS = 256           # scored windows per map (eval_tf.draw_windows), recorded in every split file
MOTION_CHUNK = 1024         # latent rows read at a time when streaming an episode's motion
DIRECTION_BLOCK = 50        # directions per block of the sorted-reference engine (bounds its memory)
MIXTURE_PROJECTIONS = 100   # directions the best-mixture search runs on; the reported value uses all
CAMPAIGN_FIRST_MAP = 18     # Freedoom Phase 2 campaign maps 18 to 32 (unseen2); arenas are maps 1 to 17
SAME_WAD_UNSEEN = (6, 7, 8)  # unseen arenas from the training WAD, check (iv)
PIXEL_GRID = (15, 20)       # 240 x 320 frames averaged in 16 x 16 blocks
NEFF_LIMIT = 0.4            # check (v): |Spearman(D, Kish n_eff)| must stay below this
SUBSET_TOLERANCE = 0.10     # check (ii): disjoint subsets, and a corpus's copy of a map, agree within this
DRAW2_SPEARMAN = 0.95       # check (iii)
ARMS = ("motion", "uniform")
FRAME_KEYS = ("feats", "motion", "episode", "map", "tic", "row", "life", "decile")
EPISODE_KEYS = ("id", "map", "rows", "lives", "eligible", "valid_fraction", "drawn")
NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


# ---------------------------------------------------------------------------------------------
# small statistics, numpy only
# ---------------------------------------------------------------------------------------------

def rank_average(a):
    """1-based ranks with ties sharing their average rank."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + 1 + ends) / 2.0)[inv.reshape(-1)]


def spearman(a, b):
    """Spearman's rho (Pearson on average ranks); NaN when either side is constant."""
    ra, rb = rank_average(a), rank_average(b)
    if len(ra) != len(rb):
        raise ValueError(f"{len(ra)} values against {len(rb)}")
    if len(ra) < 2 or ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def kendall_tau(a, b):
    """Kendall's tau-b between two rankings of the same items; NaN when either side is constant."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    i, j = np.triu_indices(len(a), 1)
    da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
    den = np.sqrt(float(np.sum(da != 0)) * float(np.sum(db != 0)))
    return float(np.sum(da * db) / den) if den > 0 else float("nan")


def kish_neff(w):
    """Kish's effective sample size (sum w)^2 / sum w^2 of a weighted cloud."""
    w = np.asarray(w, dtype=np.float64)
    return float(w.sum() ** 2 / np.sum(w * w))


# ---------------------------------------------------------------------------------------------
# reading a per-tic corpus: lives, eligible targets, motion, the stratified draw
# ---------------------------------------------------------------------------------------------

def eligible_rows(meta, context_frames=CONTEXT_FRAMES):
    """Target rows of the windows `eval_tf.py` scores at one tic: `tic_window_starts(meta, L, 1) + L`.

    So a frame is eligible exactly when a full L-tic context of its own life, with consecutive tics,
    sits behind it, which is the population the per-map scores are drawn from.
    """
    from doom_data import tic_window_starts
    return tic_window_starts(meta, int(context_frames), 1) + int(context_frames)


def life_index(deaths):
    """Per-row life number: `deaths` changes on the respawn row (`transitions.life_segments`)."""
    d = np.asarray(deaths).astype(np.int64).reshape(-1)
    if len(d) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([[0], np.cumsum(np.diff(d) != 0)]).astype(np.int64)


def episode_info(meta, context_frames=CONTEXT_FRAMES, rows=None):
    """Rows, lives, eligible targets and the valid-window fraction of one episode's sidecar.

    The valid-window fraction is the share of the T - L candidate windows of L + 1 rows that
    `tic_window_starts` keeps, the complement of `eval_tf.py`'s `window_validity.excluded_fraction`.
    """
    T = len(meta["tic"])
    rows = eligible_rows(meta, context_frames) if rows is None else rows
    cand = max(0, T - int(context_frames))
    return {"rows": T, "lives": int(life_index(meta["deaths"])[-1]) + 1 if T else 0,
            "eligible": int(len(rows)), "valid_fraction": len(rows) / cand if cand else 0.0,
            "map": int(np.asarray(meta["map_id"])[0]) if T else -1}


def episode_motion(lat, chunk=MOTION_CHUNK):
    """`latent_motion` over a whole episode, read `chunk` rows at a time from a memory map.

    Row t gets ||z_t - z_(t-1)||; row 0 copies row 1, as `latent_motion` does. Only rows whose
    predecessor is in the same life are ever used (the eligible ones), so the jump across a respawn
    that the whole-episode pass computes is never read.
    """
    T = lat.shape[0]
    if T < 2:
        raise ValueError(f"motion needs at least two frames, got {T}")
    out = np.empty(T)
    for a in range(0, T - 1, max(1, int(chunk))):
        b = min(T, a + max(1, int(chunk)) + 1)
        out[a + 1:b] = latent_motion(lat[a:b])[1:]
    out[0] = out[1]
    return out


def motion_at(frames, rows):
    """||z_t - z_(t-1)|| at scattered rows, reading only those rows and their predecessors."""
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) and rows.min() < 1:
        raise ValueError("row 0 has no predecessor")
    cur = np.asarray(frames[rows], dtype=np.float64).reshape(len(rows), -1)
    prev = np.asarray(frames[rows - 1], dtype=np.float64).reshape(len(rows), -1)
    return np.linalg.norm(cur - prev, axis=1)


def stratified_draw(motion, per_bin, bins=MOTION_BINS, rng=None):
    """`per_bin` frames from each of `bins` equal-count motion bins, without replacement.

    The bins are the motion ranks cut into `bins` near-equal groups (`np.array_split`), so the draw
    keeps the episode's motion distribution (proportional allocation adds no bias) while pinning the
    loud tail that carries the weight. Returns (indices into `motion`, ascending; bin of each) or
    None when some bin holds fewer than `per_bin` frames.
    """
    m = np.asarray(motion, dtype=np.float64).reshape(-1)
    if len(m) < per_bin * bins:
        return None
    rng = np.random.default_rng(0) if rng is None else rng
    groups = np.array_split(np.argsort(m, kind="stable"), bins)
    idx = np.concatenate([rng.choice(g, size=per_bin, replace=False) for g in groups])
    lab = np.repeat(np.arange(bins), per_bin)
    o = np.argsort(idx, kind="stable")
    return idx[o].astype(np.int64), lab[o].astype(np.int64)


def pixel_features(frames, grid=PIXEL_GRID):
    """Feature space (a): RGB frames (N, H, W, 3) uint8 averaged in blocks to (N, 3 * 15 * 20) in [0, 1]."""
    f = np.asarray(frames)
    n, h, w, c = f.shape
    gh, gw = grid
    if h % gh or w % gw:
        raise ValueError(f"a {h} x {w} frame does not tile into {gh} x {gw} blocks")
    f = f.astype(np.float64).reshape(n, gh, h // gh, gw, w // gw, c).mean(axis=(2, 4)) / 255.0
    return f.transpose(0, 3, 1, 2).reshape(n, -1).astype(np.float32)


class RawRecording:
    """One raw episode recording: the frames at exactly the requested tics, never the nearest ones."""

    def __init__(self, parquet_dir, episode):
        import pyarrow.parquet as pq
        path = os.path.join(parquet_dir, f"ep_{int(episode):05d}.parquet")
        if not os.path.isfile(path):
            raise SystemExit(f"no raw recording at {path}")
        t = pq.read_table(path, columns=["tic", "frame"])
        self.path, self.tics, self.col = path, np.asarray(t["tic"]).astype(np.int64), t["frame"]

    def frames(self, tics):
        from PIL import Image
        tics = np.asarray(tics, dtype=np.int64)
        i = np.searchsorted(self.tics, tics)
        hit = (i < len(self.tics)) & (self.tics[np.minimum(i, len(self.tics) - 1)] == tics)
        if not hit.all():
            raise SystemExit(f"{self.path} has no frame at tic(s) {tics[~hit][:5].tolist()}; the draw and the "
                             "recording do not agree")
        return np.stack([np.asarray(Image.open(io.BytesIO(self.col[int(k)].as_py())).convert("RGB"), np.uint8)
                         for k in i])


# ---------------------------------------------------------------------------------------------
# cloud files
# ---------------------------------------------------------------------------------------------

def save_cloud(path, frames, episodes, meta):
    """One set's cloud: per-frame arrays, a per-episode table (`ep_` prefix) and a JSON `meta`, atomically."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in frames.items()}
    arrays.update({"ep_" + k: np.asarray(v) for k, v in episodes.items()})
    arrays["meta"] = np.array(json.dumps(_jsonable(meta)))
    tmp = f"{path}.tmp.{os.getpid()}.npz"
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_cloud(path):
    """{"frames": {...}, "episodes": {...}, "meta": {...}} of a file `save_cloud` wrote."""
    with np.load(os.fspath(path), allow_pickle=False) as z:
        frames = {k: z[k] for k in z.files if not k.startswith("ep_") and k != "meta"}
        episodes = {k[3:]: z[k] for k in z.files if k.startswith("ep_")}
        meta = json.loads(str(z["meta"]))
    return {"frames": frames, "episodes": episodes, "meta": meta}


def _jsonable(v):
    """Plain Python for json: numpy scalars and arrays converted, NaN and infinities as null."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    return v


def code_identity():
    """The commit and the content hash of this file, written beside every distance it produces."""
    here = os.path.abspath(__file__)
    with open(here, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    def git(*cmd):
        try:
            r = subprocess.run(["git", "-C", os.path.dirname(here), *cmd], capture_output=True, text=True, timeout=20)
            return r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--", os.path.basename(here))
    return {"git_commit": commit or "unversioned", "distance_study_sha256": sha,
            "distance_study_modified": bool(status) if status is not None else None}


# ---------------------------------------------------------------------------------------------
# which episodes: an evaluation corpus, the seeded reference draw
# ---------------------------------------------------------------------------------------------

def _named(spec):
    """"NAME=DIR" -> (NAME, DIR)."""
    if "=" not in spec:
        raise SystemExit(f"expected NAME=DIR, got {spec!r}")
    name, d = spec.split("=", 1)
    if not NAME_RE.match(name) or name.startswith("train_draw"):
        raise SystemExit(f"set name {name!r}: letters, digits and underscores only, and train_draw* is reserved")
    return name, d


def listed_episodes(latents_dir):
    """[(ep, latents, sidecar)] of an evaluation corpus, and the published split that restricted it.

    When `make_dense_eval_splits.py` has published the corpus's split file (`<parent>/split_<corpus>.json`,
    the one `after_nexttic.sh` scores), only the episodes it lists are used, so a map's cloud, its split
    file and its score all come from the corpus as published.
    """
    from doom_data import list_latent_episodes
    from make_dense_eval_splits import split_path
    eps = list_latent_episodes(latents_dir)
    sp = split_path(latents_dir)
    if os.path.isfile(sp):
        with open(sp) as f:
            keep = {int(e) for e in json.load(f)["val"]}
        return [t for t in eps if t[0] in keep], sp
    return eps, None


def _sidecar(meta_path, columns=("tic", "deaths", "map_id")):
    with np.load(meta_path) as z:
        missing = [c for c in columns if c not in z.files]
        if missing:
            raise SystemExit(f"{meta_path}: no {missing} column(s)")
        return {c: np.asarray(z[c]) for c in columns}


def _map_of(meta_path):
    with np.load(meta_path) as z:
        return int(z["map_id"][0])


def reference_episodes(latents_dir, ids, per_map, draw, seed, usable):
    """Draw `draw` of the reference: `per_map` usable episodes per map, disjoint from every other draw.

    Per map, one permutation of the map's episode ids seeded by (seed, map); usable episodes are taken
    in that order, draw 1 the first `per_map`, draw 2 the next `per_map`. Returns the sorted ids.
    """
    from doom_data import list_latent_episodes
    listed = {ep: mp for ep, _, mp in list_latent_episodes(latents_dir)}
    want = {int(e) for e in ids} if ids else None
    by_map = {}
    for ep in sorted(listed):
        if want is None or ep in want:
            by_map.setdefault(_map_of(listed[ep]), []).append(ep)
    chosen = []
    for m, eps in sorted(by_map.items()):
        ok, need = [], per_map * draw
        for i in np.random.default_rng([int(seed), int(m)]).permutation(len(eps)):
            if usable(eps[i]):
                ok.append(eps[i])
                if len(ok) == need:
                    break
        if len(ok) < need:
            raise SystemExit(f"reference map {m}: {len(ok)} usable episode(s) of {len(eps)}, draw {draw} of "
                             f"{per_map} needs {need}")
        chosen += ok[(draw - 1) * per_map:need]
    return sorted(chosen)


# ---------------------------------------------------------------------------------------------
# clouds
# ---------------------------------------------------------------------------------------------

def _check_latents(lat, space, path):
    want = (SPACES[space]["channels"], 32, 40)
    if tuple(lat.shape[1:]) != want:
        raise SystemExit(f"{path}: latent shape {tuple(lat.shape[1:])}, the {space} space is {want}")


def _empty_frames(dim):
    return {"feats": np.zeros((0, dim), np.float32), "motion": np.zeros(0), "episode": np.zeros(0, np.int64),
            "map": np.zeros(0, np.int64), "tic": np.zeros(0, np.int64), "row": np.zeros(0, np.int64),
            "life": np.zeros(0, np.int64), "decile": np.zeros(0, np.int64)}


def draw_episode(ep, lat_path, meta_path, space, a):
    """(episode record, frame arrays or None): the stratified draw of one episode in a latent space."""
    meta = _sidecar(meta_path)
    lat = np.load(lat_path, mmap_mode="r")
    _check_latents(lat, space, lat_path)
    if lat.shape[0] != len(meta["tic"]):
        raise SystemExit(f"{meta_path}: {len(meta['tic'])} sidecar rows for {lat.shape[0]} latents")
    rows = eligible_rows(meta, a.context_frames)
    info = {"id": int(ep), **episode_info(meta, a.context_frames, rows), "drawn": 0}
    if len(rows) < a.frames_per_episode:
        return info, None
    motion = episode_motion(lat, a.chunk)[rows]
    idx, dec = stratified_draw(motion, a.frames_per_episode // a.bins, a.bins,
                               np.random.default_rng([int(a.seed), int(ep)]))
    r = rows[idx]
    info["drawn"] = int(len(r))
    return info, {"feats": pool_latents(lat[r]), "motion": motion[idx], "episode": np.full(len(r), ep, np.int64),
                  "map": np.full(len(r), info["map"], np.int64), "tic": meta["tic"][r].astype(np.int64),
                  "row": r.astype(np.int64), "life": life_index(meta["deaths"])[r], "decile": dec}


def carry_episode(ep, src, target, space, a):
    """The frames of `src` (one episode of another space's cloud) read in this space, matched by tic."""
    tics = src["tic"]
    if space == "pixels":
        rec = RawRecording(target, ep)
        feats, motion = [], []
        for s in range(0, len(tics), 32):                                  # bound the float64 frames held
            now = rec.frames(tics[s:s + 32])
            prev = rec.frames(tics[s:s + 32] - 1)
            feats.append(pixel_features(now))
            diff = (now.astype(np.float64) - prev.astype(np.float64)) / 255.0
            motion.append(np.linalg.norm(diff.reshape(len(now), -1), axis=1))
        return {**src, "feats": np.concatenate(feats), "motion": np.concatenate(motion)}
    lat_path, meta_path = target
    meta = _sidecar(meta_path)
    lat = np.load(lat_path, mmap_mode="r")
    _check_latents(lat, space, lat_path)
    t = meta["tic"].astype(np.int64)
    rows = np.searchsorted(t, tics)
    hit = (rows < len(t)) & (t[np.minimum(rows, len(t) - 1)] == tics)
    if not hit.all():
        raise SystemExit(f"{meta_path} holds no row at tic(s) {tics[~hit][:5].tolist()}; the two spaces' corpora "
                         "do not describe the same frames")
    life = life_index(meta["deaths"])
    if np.any(rows < 1) or np.any(t[rows - 1] != tics - 1) or np.any(life[rows - 1] != life[rows]):
        raise SystemExit(f"{meta_path}: a carried frame's predecessor is not the previous tic of its life")
    if np.any(meta["map_id"][rows] != src["map"]):
        raise SystemExit(f"{meta_path}: map ids differ from the source draw")
    return {**src, "feats": pool_latents(lat[rows]), "motion": motion_at(lat, rows),
            "row": rows.astype(np.int64), "life": life[rows]}


def build_set(name, source, kind, a):
    """(frames, episode table, extra meta) of one set: a fresh draw or a carried one."""
    space = a.space
    dim = SPACES[space]["dim"]
    table, parts, extra = [], [], {}
    if a.frames_from:
        src_path = os.path.join(a.frames_from, f"{name}.npz")
        if not os.path.isfile(src_path):
            raise SystemExit(f"--frames-from: no {src_path} to carry")
        src = load_cloud(src_path)
        extra["frames_from"] = os.path.abspath(src_path)
        extra["source_meta"] = src["meta"]
        f = src["frames"]
        listing = {}
        if space != "pixels":
            from doom_data import list_latent_episodes
            listing = {ep: (lp, mp) for ep, lp, mp in list_latent_episodes(source)}
        for i, ep in enumerate(src["episodes"]["id"].tolist()):
            row = {k: src["episodes"][k][i] for k in EPISODE_KEYS}
            if space != "pixels":
                if ep not in listing:
                    raise SystemExit(f"episode {ep} of {src_path} is not in {source}")
                row.update(episode_info(_sidecar(listing[ep][1]), a.context_frames))
                row["id"] = ep
            table.append(row)
            m = f["episode"] == ep
            if not m.any():
                continue
            parts.append(carry_episode(ep, {k: f[k][m] for k in FRAME_KEYS},
                                       source if space == "pixels" else listing[ep], space, a))
        if a.limit:
            keep = _first_per_map([(r["id"], r["map"]) for r in table], a.limit)
            table = [r for r in table if r["id"] in keep]
            parts = [p for p in parts if int(p["episode"][0]) in keep]
    else:
        if kind == "reference":
            from doom_data import list_latent_episodes, parse_episode_ids
            listing = {ep: (lp, mp) for ep, lp, mp in list_latent_episodes(source)}
            per_map = min(a.episodes_per_map, a.limit) if a.limit else a.episodes_per_map

            def usable(ep):
                return len(eligible_rows(_sidecar(listing[ep][1]), a.context_frames)) >= a.frames_per_episode
            ids = parse_episode_ids(a.reference_ids) if a.reference_ids else None
            chosen = reference_episodes(source, ids, per_map, a.draw, a.seed, usable)
            eps = [(ep, *listing[ep]) for ep in chosen]
            extra.update(reference_ids=a.reference_ids or None, episodes_per_map=per_map)
        else:
            eps, published = listed_episodes(source)
            extra["published_split"] = published
            if a.limit:
                keep = _first_per_map([(ep, _map_of(mp)) for ep, _, mp in eps], a.limit)
                eps = [t for t in eps if t[0] in keep]
        for ep, lp, mp in eps:
            info, fr = draw_episode(ep, lp, mp, space, a)
            table.append(info)
            if fr is not None:
                parts.append(fr)
    if not parts:
        raise SystemExit(f"{name}: no episode had {a.frames_per_episode} eligible frames to draw")
    frames = {k: np.concatenate([p[k] for p in parts]) for k in FRAME_KEYS}
    if frames["feats"].shape[1] != dim:
        raise SystemExit(f"{name}: {frames['feats'].shape[1]}-dimensional features, the {space} space has {dim}")
    episodes = {k: np.array([r[k] for r in table]) for k in EPISODE_KEYS}
    return frames, episodes, extra


def _first_per_map(pairs, limit):
    """The first `limit` episode ids of every map, by id: what `--limit` keeps."""
    kept, count = set(), {}
    for ep, m in sorted(pairs):
        if count.get(m, 0) < limit:
            kept.add(ep)
            count[m] = count.get(m, 0) + 1
    return kept


def cmd_clouds(a):
    if a.frames_per_episode % a.bins:
        raise SystemExit(f"--frames-per-episode {a.frames_per_episode} is not a multiple of {a.bins} motion bins")
    if a.space == "pixels" and not a.frames_from:
        raise SystemExit("the pixel space has no latents to rank motion in; pass --frames-from with a latent "
                         "space's cloud directory (results/distance_study/clouds/sd1) so pixels are read at its "
                         "frames, and name the raw recording directories as the sets")
    jobs = []
    if a.reference:
        jobs.append((f"train_draw{a.draw}", a.reference, "reference"))
    jobs += [(*_named(s), "eval") for s in a.eval or []]
    if not jobs:
        raise SystemExit("nothing to draw: pass --reference DIR and/or --eval NAME=DIR")
    out_dir = os.path.join(a.out, "clouds", a.space)
    code = code_identity()
    for name, source, kind in jobs:
        t0 = time.time()
        frames, episodes, extra = build_set(name, source, kind, a)
        meta = {"space": a.space, "set": name, "kind": kind, "draw": a.draw if kind == "reference" else None,
                "source": os.path.abspath(source), "seed": a.seed, "frames_per_episode": a.frames_per_episode,
                "motion_bins": a.bins, "context_frames": a.context_frames, "limit": a.limit or None,
                "frames": int(len(frames["episode"])), "episodes_drawn": int(np.sum(episodes["drawn"] > 0)),
                "episodes_listed": int(len(episodes["id"])), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "code": code, **extra}
        path = os.path.join(out_dir, f"{name}.npz")
        save_cloud(path, frames, episodes, meta)
        print(json.dumps({"wrote": path, "frames": meta["frames"], "episodes": meta["episodes_drawn"],
                          "listed": meta["episodes_listed"], "seconds": round(time.time() - t0, 1)}), flush=True)
    return 0


# ---------------------------------------------------------------------------------------------
# splits: one eval_tf.py split file per map
# ---------------------------------------------------------------------------------------------

def cluster_of(map_id):
    """The two map clusters of the study: the deathmatch arenas and the curated campaign maps."""
    return "campaign" if int(map_id) >= CAMPAIGN_FIRST_MAP else "arena"


def assign_roles(counts, order):
    """{(set, map): "primary" | "replication"}. A map held by several sets is one point of the study,
    taken from the set with the most episodes of it (ties: the earlier set in `order`); the other
    copies replicate it (memo: the seeded corpus's maps 2 to 8 are a cross-corpus replication)."""
    rank = {s: i for i, s in enumerate(order)}
    by_map = {}
    for (s, m), n in counts.items():
        by_map.setdefault(m, []).append((s, n))
    roles = {}
    for m, held in by_map.items():
        best = min(held, key=lambda t: (-t[1], rank.get(t[0], len(rank)), t[0]))[0]
        for s, _ in held:
            roles[(s, m)] = "primary" if s == best else "replication"
    return roles


def split_name(set_name, map_id):
    return f"split_{set_name}_map{int(map_id):02d}.json"


def cmd_splits(a):
    from doom_data import tic_window_starts
    from make_dense_eval_splits import SUBSET
    sets = [_named(s) for s in a.eval or []]
    if not sets:
        raise SystemExit("pass the evaluation corpora as --eval NAME=DIR")
    members, windows, published = {}, {}, {}
    for name, d in sets:
        eps, published[name] = listed_episodes(d)
        for ep, _, mp in eps:
            meta = _sidecar(mp)
            members.setdefault((name, int(meta["map_id"][0])), []).append(int(ep))
            # the window counts eval_tf.py draws from at horizons 1 and 4, per episode
            windows[(name, int(ep))] = (len(tic_window_starts(meta, a.context_frames, 1)),
                                        len(tic_window_starts(meta, a.context_frames, 4)))
    roles = assign_roles({k: len(v) for k, v in members.items()}, [n for n, _ in sets])
    out_dir = os.path.join(a.out, "splits")
    os.makedirs(out_dir, exist_ok=True)
    # the scorer derives its map list from these files, so a file from an earlier run must not linger
    for old in glob.glob(os.path.join(out_dir, "split_*_map*.json")):
        os.remove(old)
    order = {n: i for i, (n, _) in enumerate(sets)}
    index = []
    for key in sorted(members, key=lambda k: (order[k[0]], k[1])):
        name, m = key
        eps = sorted(members[key])
        if a.limit:
            eps = eps[:a.limit]
        w1, w4 = (sum(windows[(name, e)][h] for e in eps) for h in (0, 1))
        rec = {"set": name, "map": m, "role": roles[key], "cluster": cluster_of(m), "episodes": eps,
               "windows_h1": w1, "windows_h4": w4, "file": None}
        if roles[key] == "primary":
            rec["file"] = split_name(name, m)
            split = {SUBSET: eps, "meta": {
                "corpus": f"{name}_map{m:02d}", "set": name, "map": m, "cluster": cluster_of(m), "role": "primary",
                "latents_dir": os.path.abspath(dict(sets)[name]), "published_split": published[name],
                "num_episodes": len(eps), "subset_key": SUBSET, "num_windows": a.num_windows,
                "windows_h1": w1, "windows_h4": w4, "short": w1 < a.num_windows, "limit": a.limit or None,
                "note": "one map of one evaluation corpus for the distance study; eval_tf.py scores it with "
                        f"--subset {SUBSET} --num-windows {a.num_windows}"}}
            path = os.path.join(out_dir, rec["file"])
            tmp = f"{path}.tmp.{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(split, f, indent=1)
            os.replace(tmp, path)
        index.append(rec)
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump({"maps": index, "sets": [n for n, _ in sets], "num_windows": a.num_windows,
                   "context_frames": a.context_frames, "code": code_identity()}, f, indent=1)
    print(json.dumps({"wrote": out_dir, "split_files": sum(r["file"] is not None for r in index),
                      "replications": sum(r["role"] == "replication" for r in index)}))
    return 0


# ---------------------------------------------------------------------------------------------
# many distances against fixed references: each reference is sorted once per block of directions
# ---------------------------------------------------------------------------------------------

def _project_sort(x, directions):
    """(sorted projections (n, b), sort order (n, b)) of a cloud on a block of directions."""
    xp = np.asarray(x, dtype=np.float64) @ directions
    order = np.argsort(xp, axis=0, kind="stable")
    return np.take_along_axis(xp, order, 0), order


def _cumulative(w, order):
    """(weights (n, b), cumulative weights (n, b), last exactly 1) of a cloud in its sorted order."""
    ws = _weights(w, order.shape[0], "cloud weights")[order]
    cx = np.cumsum(ws, axis=0)
    return ws / cx[-1:], cx / cx[-1:]


def sorted_cloud(x, w, directions):
    """(sorted projections, their weights, cumulative weights) of one weighted cloud."""
    xs, order = _project_sort(x, directions)
    return (xs, *_cumulative(w, order))


def _w2_columns(xs, ws, cx, ref):
    """Per-direction W_2^2 between a sorted cloud and a prepared reference (see `SortedReference`)."""
    ysT, cyT, gT, flat, m2 = ref
    b, m = cyT.shape
    n = xs.shape[0]
    u = cx.T.ravel()                                                   # column-major, column j contiguous
    col = np.repeat(np.arange(b), n)
    pos = np.searchsorted(flat, u + 2.0 * col, side="left")
    k = col * m + np.clip(pos - col * m, 0, m - 1)
    G = (gT.ravel()[k] - (cyT.ravel()[k] - u) * ysT.ravel()[k]).reshape(b, n).T
    cross = np.sum(xs * np.diff(G, axis=0, prepend=0.0), axis=0)
    return np.maximum(np.sum(ws * xs * xs, axis=0) + m2 - 2.0 * cross, 0.0)


def _prepare(ysT, wsT):
    """(ysT, cumulative weights, G at the breakpoints, column-offset search array, second moment)."""
    cyT = np.cumsum(wsT, axis=1)
    tot = cyT[:, -1:].copy()
    cyT /= tot
    gT = np.cumsum(wsT * ysT, axis=1) / tot
    m2 = np.sum(wsT * ysT * ysT, axis=1) / tot[:, 0]
    flat = (cyT + 2.0 * np.arange(cyT.shape[0])[:, None]).ravel()
    return ysT, cyT, gT, flat, m2


class SortedReference:
    """A reference cloud projected on a block of directions and sorted once, for many W_2 evaluations.

    In 1-D, W_2^2 = integral over u of (F^-1(u) - Q(u))^2, with F^-1 the evaluated cloud's quantile
    function and Q the reference's. Let G(u) = integral_0^u Q, which is piecewise linear with its
    breakpoints at the reference's cumulative weights. Splitting the integral at the cloud's own
    cumulative weights c_j, where F^-1 = x_j is constant, gives exactly

        W_2^2 = sum_j a_j x_j^2 + E_nu[y^2] - 2 sum_j x_j (G(c_j) - G(c_(j-1)))

    for sorted values x_j with weights a_j. Everything about the reference is computed once per block;
    an evaluation then costs a sort of the n-point cloud and n binary searches per direction, instead
    of the merge of n + m breakpoints `_transport_cost` performs. The two agree to rounding
    (`test_the_sorted_reference_gives_the_committed_sliced_wasserstein`). Each column's cumulative
    weights are offset by twice the column index, so one `searchsorted` locates every query of every
    column; G is continuous, so a query that rounding places one breakpoint off still gets its value.
    """

    def __init__(self, y, directions, arms):
        yp = np.asarray(y, dtype=np.float64) @ directions
        order = np.argsort(yp, axis=0, kind="stable")
        ysT = np.ascontiguousarray(np.take_along_axis(yp, order, 0).T)
        self.arms = {a: _prepare(ysT, np.ascontiguousarray(_weights(w, yp.shape[0], "reference weights")[order].T))
                     for a, w in arms.items()}

    def cost(self, xs, ws, cx, arm):
        return _w2_columns(xs, ws, cx, self.arms[arm])


def distance_table(clouds, references, directions, arms=("motion",), block=DIRECTION_BLOCK):
    """(len(arms), len(clouds), len(references)) sliced W_2 distances on shared directions.

    `clouds` is a sequence of (x, {arm: weights}); x is an (n, d) array or a zero-argument callable
    returning one, so a bootstrap can pass thousands of clouds as index arrays. `references` maps a
    name to (y, {arm: weights}). An arm absent from a weights dict, or mapped to None, is uniform.
    Directions are processed `block` at a time, which bounds memory and does not change the result.
    """
    dirs = np.asarray(directions, dtype=np.float64)
    names = list(references)
    L = dirs.shape[1]
    total = np.zeros((len(arms), len(clouds), len(names)))
    for s in range(0, L, max(1, int(block))):
        db = dirs[:, s:s + max(1, int(block))]
        refs = []
        for k in names:
            y, w = references[k]
            refs.append(SortedReference(y, db, {a: (w or {}).get(a) for a in arms}))
        for c, (x, cw) in enumerate(clouds):
            xv = x() if callable(x) else x
            if np.ndim(xv) != 2 or np.shape(xv)[1] != dirs.shape[0]:
                raise ValueError(f"cloud {c} is {np.shape(xv)}, the directions are {dirs.shape[0]}-dimensional")
            xs, order = _project_sort(xv, db)
            for ai, a in enumerate(arms):
                ws, cx = _cumulative((cw or {}).get(a), order)
                for r, ref in enumerate(refs):
                    total[ai, c, r] += ref.cost(xs, ws, cx, a).sum()
    return np.sqrt(total / L)


class MixtureReference:
    """The training maps pooled and sorted once per block, weighted by mixture weights alpha at evaluation.

    The reference mixture sum_k alpha_k nu_k gives each point of map k the mass alpha_k times its
    within-map weight. Its sort order does not depend on alpha, so only the cumulative sums are redone.
    """

    def __init__(self, references, directions, arm):
        ys, comp, wb = [], [], []
        for k, (y, arms) in enumerate(references):
            y = np.asarray(y, dtype=np.float64)
            ys.append(y)
            comp.append(np.full(len(y), k))
            wb.append(_weights((arms or {}).get(arm), len(y), "reference weights"))
        yp = np.concatenate(ys) @ directions
        order = np.argsort(yp, axis=0, kind="stable")
        self.ysT = np.ascontiguousarray(np.take_along_axis(yp, order, 0).T)
        self.compT = np.ascontiguousarray(np.concatenate(comp)[order].T)
        self.wbT = np.ascontiguousarray(np.concatenate(wb)[order].T)
        self.k = len(references)

    def cost(self, xs, ws, cx, alpha):
        return _w2_columns(xs, ws, cx, _prepare(self.ysT, np.asarray(alpha, dtype=np.float64)[self.compT] * self.wbT))


def _simplex_search(f, k, steps=(0.5, 0.25, 0.125, 0.0625, 0.03125)):
    """Minimise f over the probability simplex by pattern search from the best vertex.

    SW_2^2 to a mixture is convex in the mixture weights (W_2^2 is jointly convex, and the slices
    average), so moving mass between two components in shrinking steps finds the minimum to the
    final step's resolution. Every accepted move strictly lowers f on a finite grid, so it stops.
    """
    cache = {}

    def F(v):
        key = tuple(np.round(v, 10))
        if key not in cache:
            cache[key] = f(v)
        return cache[key]
    eye = np.eye(k)
    a = min((eye[i] for i in range(k)), key=F)
    fa = F(a)
    for d in steps:
        while True:
            best, fb = None, fa
            for i in range(k):
                if a[i] <= 1e-12:
                    continue
                t = min(d, a[i])
                for j in range(k):
                    if j != i:
                        b = a.copy()
                        b[i] -= t
                        b[j] += t
                        v = F(b)
                        if v < fb * (1 - 1e-12):
                            best, fb = b, v
            if best is None:
                break
            a, fa = best, fb
    return a


def mixture_distances(clouds, references, directions, arm, search_projections=MIXTURE_PROJECTIONS,
                      block=DIRECTION_BLOCK):
    """(distances, weights): the best mixture of `references` (a list) for every cloud, one arm.

    The weights are searched on the first `search_projections` of the shared directions (common random
    numbers with every other distance); the reported distance is SW_2 to that mixture on all of them.
    """
    dirs = np.asarray(directions, dtype=np.float64)
    L = dirs.shape[1]
    ds_ = dirs[:, :min(int(search_projections), L)]
    search = MixtureReference(references, ds_, arm)
    alphas = []
    for x, cw in clouds:
        xs, ws, cx = sorted_cloud(x() if callable(x) else x, (cw or {}).get(arm), ds_)
        alphas.append(_simplex_search(lambda v, xs=xs, ws=ws, cx=cx: float(np.mean(search.cost(xs, ws, cx, v))),
                                      len(references)))
    alphas = np.array(alphas).reshape(len(clouds), len(references))
    total = np.zeros(len(clouds))
    for s in range(0, L, max(1, int(block))):
        db = dirs[:, s:s + max(1, int(block))]
        mix = MixtureReference(references, db, arm)
        for c, (x, cw) in enumerate(clouds):
            xs, ws, cx = sorted_cloud(x() if callable(x) else x, (cw or {}).get(arm), db)
            total[c] += mix.cost(xs, ws, cx, alphas[c]).sum()
    return np.sqrt(total / L), alphas


def episode_subsets(episodes, size=EPISODES_PER_CLOUD, n=SUBSETS, rng=None):
    """(subsets, disjoint pairs) of a map's episodes for its equal-size clouds.

    A map with at most `size` episodes is one cloud of all of them. With at least 2 x `size`, the `n`
    distinct random subsets are drawn as n // 2 disjoint PAIRS, so the same draws that average into
    the map's distance also give check (ii)'s disjoint-subset agreement. In between, every
    combination when there are at most `n`, else `n` distinct random ones.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    eps = sorted(int(e) for e in episodes)
    if len(eps) <= size:
        return [eps], []
    if len(eps) < 2 * size:
        combos = [list(c) for c in combinations(eps, size)]
        if len(combos) <= n:
            return combos, []
        return [combos[i] for i in sorted(rng.choice(len(combos), size=n, replace=False))], []
    subsets, pairs, seen, tries = [], [], set(), 0
    while len(pairs) < n // 2 and tries < 10_000:
        tries += 1
        p = rng.permutation(len(eps))[:2 * size]
        a, b = sorted(eps[i] for i in p[:size]), sorted(eps[i] for i in p[size:])
        if tuple(a) in seen or tuple(b) in seen:
            continue
        seen |= {tuple(a), tuple(b)}
        pairs.append((len(subsets), len(subsets) + 1))
        subsets += [a, b]
    while len(subsets) < n and tries < 20_000:
        tries += 1
        a = sorted(eps[i] for i in rng.permutation(len(eps))[:size])
        if tuple(a) not in seen:
            seen.add(tuple(a))
            subsets.append(a)
    return subsets, pairs


# ---------------------------------------------------------------------------------------------
# distances
# ---------------------------------------------------------------------------------------------

def _frame_index(frames):
    """{episode: indices of its frames} of one cloud file."""
    ep = frames["episode"]
    order = np.argsort(ep, kind="stable")
    uniq, start = np.unique(ep[order], return_index=True)
    ends = list(start[1:]) + [len(ep)]
    return {int(e): order[s:t] for e, s, t in zip(uniq, start, ends)}


def map_references(cloud, maps):
    """{map: (features, {"motion": motion weights of that map's cloud, "uniform": None})}."""
    f = cloud["frames"]
    out = {}
    for m in maps:
        sel = f["map"] == m
        if not sel.any():
            raise SystemExit(f"{cloud['meta'].get('set')}: no frames of training map {m}")
        out[m] = (f["feats"][sel].astype(np.float64), {"motion": motion_weights(f["motion"][sel]), "uniform": None})
    return out


def pooled_arms(refs):
    """The training corpus as one cloud per arm, every map holding an equal share of the mass."""
    ys, arms = [], {a: [] for a in ARMS}
    for y, w in refs.values():
        ys.append(y)
        for a in ARMS:
            arms[a].append(_weights(w.get(a), len(y), "reference weights") / len(refs))
    return np.concatenate(ys), {a: np.concatenate(v) for a, v in arms.items()}


def _summary(values):
    v = np.asarray(values, dtype=np.float64)
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "p2.5": float(np.percentile(v, 2.5)),
            "p97.5": float(np.percentile(v, 97.5))}


def cmd_distances(a):
    space_dir = os.path.join(a.out, "clouds", a.space)
    ref_path = os.path.join(space_dir, f"train_draw{a.reference_draw}.npz")
    if not os.path.isfile(ref_path):
        raise SystemExit(f"no reference cloud at {ref_path}; run `clouds --reference ... --draw {a.reference_draw}`")
    ref1 = load_cloud(ref_path)
    floor_path = os.path.join(space_dir, f"train_draw{a.floor_draw}.npz")
    ref2 = load_cloud(floor_path) if os.path.isfile(floor_path) else None
    if ref2 is not None and set(ref1["frames"]["episode"].tolist()) & set(ref2["frames"]["episode"].tolist()):
        raise SystemExit(f"{ref_path} and {floor_path} share episodes; the floor needs disjoint draws")
    names = a.sets or sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(space_dir, "*.npz"))
                             if not os.path.basename(p).startswith("train_draw"))
    if not names:
        raise SystemExit(f"no evaluation cloud in {space_dir}")
    sets = {n: load_cloud(os.path.join(space_dir, f"{n}.npz")) for n in names}
    train_maps = sorted(int(m) for m in np.unique(ref1["frames"]["map"]))
    dim = ref1["frames"]["feats"].shape[1]
    for n, c in sets.items():
        if c["frames"]["feats"].shape[1] != dim:
            raise SystemExit(f"{n}: {c['frames']['feats'].shape[1]}-dimensional, the reference is {dim}")
    dirs = random_directions(dim, a.projections, a.direction_seed)
    refs1 = map_references(ref1, train_maps)
    refs2 = map_references(ref2, train_maps) if ref2 is not None else {}
    table_refs = {("d1", k): v for k, v in refs1.items()}
    table_refs[("d1", "pooled")] = pooled_arms(refs1)
    table_refs.update({("d2", k): v for k, v in refs2.items()})
    rnames = list(table_refs)
    d1_cols = [rnames.index(("d1", k)) for k in train_maps]
    d2_cols = [rnames.index(("d2", k)) for k in train_maps] if refs2 else []
    pool_col = rnames.index(("d1", "pooled"))

    # the map points and their clouds
    counts = {}
    for n in names:
        ep = sets[n]["episodes"]
        for m in np.unique(ep["map"]):
            counts[(n, int(m))] = int(np.sum(ep["map"] == m))
    roles = assign_roles(counts, names)
    index = {n: _frame_index(sets[n]["frames"]) for n in names}
    points = []
    for n in names:
        ep = sets[n]["episodes"]
        for m in sorted(int(x) for x in np.unique(ep["map"])):
            eps = sorted(int(e) for e, mm, dr in zip(ep["id"], ep["map"], ep["drawn"]) if mm == m and dr > 0)
            if eps:
                points.append({"set": n, "map": m, "key": f"{n}/{m}", "role": roles[(n, m)],
                               "cluster": cluster_of(m), "episodes": eps})
    if a.limit:
        points = points[:a.limit]
    clouds, tags = [], []

    def add(kind, key, cloud, idx):
        w = motion_weights(cloud["frames"]["motion"][idx])
        clouds.append((lambda f=cloud["frames"]["feats"], i=idx: f[i], {"motion": w, "uniform": None}))
        tags.append((kind, key, kish_neff(w), len(idx)))
    for pi, p in enumerate(points):
        rng = np.random.default_rng([a.seed, zlib.crc32(p["set"].encode()), p["map"]])
        p["subset_eps"], p["pairs"] = episode_subsets(p["episodes"], a.episodes_per_cloud, a.subsets, rng)
        p["clouds"] = []
        for si, sub in enumerate(p["subset_eps"]):
            p["clouds"].append(len(clouds))
            add("map", (pi, si), sets[p["set"]], np.concatenate([index[p["set"]][e] for e in sub]))
    episode_rows = []
    for p in points:
        for e in p["episodes"]:
            episode_rows.append((p, e, len(clouds)))
            add("episode", (p["set"], e), sets[p["set"]], index[p["set"]][e])
    floor_clouds = []
    if ref2 is not None:
        idx2 = _frame_index(ref2["frames"])
        eps2 = ref2["episodes"]
        for k in train_maps:
            ek = sorted(int(e) for e, mm, dr in zip(eps2["id"], eps2["map"], eps2["drawn"]) if mm == k and dr > 0)
            subs, _ = episode_subsets(ek, a.episodes_per_cloud, a.floor_subsets,
                                      np.random.default_rng([a.seed, 7, k]))
            for si, sub in enumerate(subs):
                floor_clouds.append((k, len(clouds)))
                add("floor", (k, si), ref2, np.concatenate([idx2[e] for e in sub]))
    t0 = time.time()
    T = distance_table(clouds, table_refs, dirs, ARMS, a.block)
    print(json.dumps({"clouds": len(clouds), "references": len(rnames), "seconds": round(time.time() - t0, 1)}),
          flush=True)

    def nearest(ai, c, cols=d1_cols):
        v = T[ai, c, cols]
        return float(v.min()), train_maps[int(np.argmin(v))]

    # per point, both arms
    for p in points:
        cs = p["clouds"]
        p["arm"] = {}
        for ai, arm in enumerate(ARMS):
            dsub = [nearest(ai, c)[0] for c in cs]
            near = [nearest(ai, c)[1] for c in cs]
            p["arm"][arm] = {"values": dsub, "nearest": near,
                             "per_reference": {k: float(np.mean(T[ai, cs, col]))
                                               for k, col in zip(train_maps, d1_cols)},
                             "pooled": float(np.mean(T[ai, cs, pool_col])),
                             "draw2": float(np.mean([nearest(ai, c, d2_cols)[0] for c in cs])) if d2_cols else None}
        p["n_eff"] = float(np.mean([tags[c][2] for c in cs]))
        p["frames_per_cloud"] = float(np.mean([tags[c][3] for c in cs]))
    prim = [p for p in points if p["role"] == "primary"]

    # check (v) first: it decides which arm is primary
    rho_v = spearman([np.mean(p["arm"]["motion"]["values"]) for p in prim], [p["n_eff"] for p in prim]) \
        if len(prim) > 1 else float("nan")
    pass_v = bool(np.isnan(rho_v) or abs(rho_v) < NEFF_LIMIT)
    arm = "motion" if pass_v else "uniform"
    ai_p = ARMS.index(arm)
    for p in points:
        p["D"] = float(np.mean(p["arm"][arm]["values"]))

    floor = None
    if floor_clouds:
        floor = {}
        for ai, name in enumerate(ARMS):
            vals = [nearest(ai, c)[0] for _, c in floor_clouds]
            floor[name] = {**_summary(vals), "values": vals,
                           "per_map": {k: float(np.mean([nearest(ai, c)[0] for kk, c in floor_clouds if kk == k]))
                                       for k in train_maps}}

    checks = {}
    val = [p for p in prim if p["set"] == a.validation_set]
    if floor and val:
        lo, hi = floor[arm]["min"], floor[arm]["max"]
        checks["i"] = {"what": "every validation map sits inside the train-versus-train floor band [min, max]",
                       "band": [lo, hi], "maps": {str(p["map"]): p["D"] for p in val},
                       "pass": all(lo <= p["D"] <= hi for p in val)}
    else:
        checks["i"] = {"what": "validation maps inside the floor band", "pass": None,
                       "note": "needs the second reference draw and the validation set"}
    per_map, rel_ok = {}, []
    for p in points:
        if p["pairs"]:
            v = p["arm"][arm]["values"]
            rel = [abs(v[i] - v[j]) / ((v[i] + v[j]) / 2) for i, j in p["pairs"]]
            per_map[p["key"]] = {"pairs": len(rel), "median": float(np.median(rel)), "max": float(np.max(rel))}
            rel_ok.append(np.median(rel) <= SUBSET_TOLERANCE)
    rep = {}
    for p in points:
        if p["role"] == "replication":
            q = next((r for r in prim if r["map"] == p["map"]), None)
            if q is not None:
                rd = abs(p["D"] - q["D"]) / q["D"] if q["D"] > 0 else float("inf")
                rep[str(p["map"])] = {"replication": p["key"], "primary": q["key"], "D_replication": p["D"],
                                      "D_primary": q["D"], "relative_difference": rd, "pass": rd <= SUBSET_TOLERANCE}
    checks["ii"] = {"what": f"disjoint 4-episode subsets agree within {SUBSET_TOLERANCE:.0%} (median over pairs), and "
                            "a map's copy in another corpus lands where its primary point does",
                    "tolerance": SUBSET_TOLERANCE, "subsets": per_map, "replication": rep,
                    "pass": bool(all(rel_ok) and all(r["pass"] for r in rep.values())) if (per_map or rep) else None}
    if d2_cols and len(prim) > 1:
        d2 = [p["arm"][arm]["draw2"] for p in prim]
        rho3 = spearman([p["D"] for p in prim], d2)
        checks["iii"] = {"what": "the two reference draws rank the maps alike", "spearman": rho3,
                         "threshold": DRAW2_SPEARMAN, "pass": bool(rho3 >= DRAW2_SPEARMAN)}
    else:
        checks["iii"] = {"what": "the two reference draws rank the maps alike", "pass": None,
                         "note": "needs the second reference draw"}
    same = [p for p in prim if p["map"] in SAME_WAD_UNSEEN]
    camp = [p for p in prim if p["cluster"] == "campaign"]
    checks["iv"] = {"what": "arenas 6 to 8 (the training WAD) sit nearer than every campaign map",
                    "max_same_wad": max(p["D"] for p in same) if same else None,
                    "min_campaign": min(p["D"] for p in camp) if camp else None,
                    "mean_same_wad": float(np.mean([p["D"] for p in same])) if same else None,
                    "mean_campaign": float(np.mean([p["D"] for p in camp])) if camp else None,
                    "pass": bool(max(p["D"] for p in same) < min(p["D"] for p in camp)) if same and camp else None}
    checks["v"] = {"what": "|Spearman(D, Kish n_eff)| over the primary maps, motion arm", "spearman": rho_v,
                   "threshold": NEFF_LIMIT, "pass": pass_v,
                   "consequence": "the motion arm is primary" if pass_v else "the uniform arm becomes primary"}
    checks["all_pass"] = all(c["pass"] for c in checks.values() if isinstance(c, dict) and c["pass"] is not None)

    # the best mixture of training maps, in the primary arm
    if not a.no_mixture and points:
        cs = [c for p in points for c in p["clouds"]]
        dmix, alphas = mixture_distances([clouds[c] for c in cs], [refs1[k] for k in train_maps], dirs, arm,
                                         a.mixture_projections, a.block)
        at = {c: (d, al) for c, d, al in zip(cs, dmix, alphas)}
        for p in points:
            vals, ws = [], []
            for c in p["clouds"]:
                d, al = at[c]
                dn, kn = nearest(ai_p, c)
                if dn < d:                                     # a vertex is a mixture too
                    d, al = dn, np.eye(len(train_maps))[train_maps.index(kn)]
                vals.append(d)
                ws.append(al)
            p["mixture"] = (float(np.mean(vals)), np.mean(ws, axis=0))

    # the bootstrap of the distance: episodes redrawn with replacement within each map
    boot = None
    if a.bootstrap > 0 and points:
        bclouds = []
        for p in points:
            rng = np.random.default_rng([a.seed, 2, zlib.crc32(p["set"].encode()), p["map"]])
            size = min(a.episodes_per_cloud, len(p["episodes"]))
            for _ in range(a.bootstrap):
                pick = rng.choice(p["episodes"], size=size, replace=True)
                idx = np.concatenate([index[p["set"]][int(e)] for e in pick])
                c = sets[p["set"]]["frames"]
                w = motion_weights(c["motion"][idx]) if arm == "motion" else None
                bclouds.append((lambda f=c["feats"], i=idx: f[i], {arm: w}))
        bt = distance_table(bclouds, {k: refs1[k] for k in train_maps}, dirs, (arm,), a.block)[0]
        draws = bt.min(axis=1).reshape(len(points), a.bootstrap)
        for p, row in zip(points, draws):
            p["bootstrap"] = {k: v for k, v in _summary(row).items() if k in ("n", "mean", "sd", "p2.5", "p97.5")}
        sds = [p["bootstrap"]["sd"] for p in prim]
        spread = float(np.std([p["D"] for p in prim], ddof=1)) if len(prim) > 1 else float("nan")
        ratio = float(np.median(sds) / spread) if spread and np.isfinite(spread) else float("nan")
        bpath = os.path.join(a.out, f"bootstrap_{a.space}.npz")
        np.savez(bpath, keys=np.array([p["key"] for p in points]), draws=draws, arm=np.array(arm))
        boot = {"draws": a.bootstrap, "arm": arm, "file": bpath, "attenuation_ratio": ratio,
                "attenuation_below_1pct": bool(ratio < 0.1) if np.isfinite(ratio) else None,
                "what": "median bootstrap SD of D over the SD of D across primary maps; under 0.1 the "
                        "attenuation of a correlation with D is below 1 percent"}

    # outputs
    ep_tab = {n: {int(e): i for i, e in enumerate(sets[n]["episodes"]["id"])} for n in names}

    def ep_mean(p, key):
        tab = sets[p["set"]]["episodes"]
        return float(np.mean([tab[key][ep_tab[p["set"]][e]] for e in p["episodes"]]))
    maps_out = []
    for p in points:
        near = p["arm"][arm]["nearest"]
        rec = {"key": p["key"], "set": p["set"], "map": p["map"], "role": p["role"], "cluster": p["cluster"],
               "episodes": p["episodes"], "n_episodes": len(p["episodes"]), "subsets": len(p["clouds"]),
               "subset_episodes": p["subset_eps"], "disjoint_pairs": [list(x) for x in p["pairs"]],
               "frames_per_cloud": p["frames_per_cloud"], "arm": arm,
               "D": p["D"], "nearest": max(set(near), key=near.count),
               "nearest_counts": {str(k): near.count(k) for k in sorted(set(near))},
               "per_reference": {str(k): v for k, v in p["arm"][arm]["per_reference"].items()},
               "D_pooled": p["arm"][arm]["pooled"], "D_draw2": p["arm"][arm]["draw2"]}
        for name in ARMS:
            rec[f"D_{name}"] = float(np.mean(p["arm"][name]["values"]))
            rec[f"per_reference_{name}"] = {str(k): v for k, v in p["arm"][name]["per_reference"].items()}
            rec[f"D_pooled_{name}"] = p["arm"][name]["pooled"]
            rec[f"D_draw2_{name}"] = p["arm"][name]["draw2"]
        rec["per_subset"] = {name: {"mean": float(np.mean(p["arm"][name]["values"])),
                                    "sd": float(np.std(p["arm"][name]["values"], ddof=1)) if len(p["clouds"]) > 1
                                    else 0.0, "values": p["arm"][name]["values"]} for name in ARMS}
        rec["n_eff"] = p["n_eff"]
        rec["lives"] = ep_mean(p, "lives")
        rec["valid_fraction"] = ep_mean(p, "valid_fraction")
        f = sets[p["set"]]["frames"]
        rec["motion_mean"] = float(np.mean(f["motion"][np.isin(f["episode"], p["episodes"])]))
        if "mixture" in p:
            rec["D_mixture"] = p["mixture"][0]
            rec["mixture_weights"] = {str(k): float(w) for k, w in zip(train_maps, p["mixture"][1])}
        if "bootstrap" in p:
            rec["bootstrap"] = p["bootstrap"]
        maps_out.append(rec)
    csv_path = os.path.join(a.out, f"per_episode_{a.space}.csv")
    cols = ["set", "map", "episode", "role", "cluster", "frames", "lives", "valid_fraction", "motion_mean", "n_eff",
            "D", "nearest", "D_motion", "nearest_motion", "D_uniform", "nearest_uniform", "D_pooled",
            "D_pooled_motion", "D_pooled_uniform"] + [f"d_{k}" for k in train_maps] + \
        [f"d_{name}_{k}" for name in ARMS for k in train_maps]
    os.makedirs(a.out, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for p, e, c in episode_rows:
            tab = sets[p["set"]]["episodes"]
            i = ep_tab[p["set"]][e]
            row = {"set": p["set"], "map": p["map"], "episode": e, "role": p["role"], "cluster": p["cluster"],
                   "frames": tags[c][3], "lives": int(tab["lives"][i]),
                   "valid_fraction": float(tab["valid_fraction"][i]),
                   "motion_mean": float(np.mean(sets[p["set"]]["frames"]["motion"][index[p["set"]][e]])),
                   "n_eff": tags[c][2]}
            for ai, name in enumerate(ARMS):
                d, k = nearest(ai, c)
                row[f"D_{name}"], row[f"nearest_{name}"] = d, k
                row[f"D_pooled_{name}"] = float(T[ai, c, pool_col])
                for kk, col in zip(train_maps, d1_cols):
                    row[f"d_{name}_{kk}"] = float(T[ai, c, col])
            row["D"], row["nearest"], row["D_pooled"] = row[f"D_{arm}"], row[f"nearest_{arm}"], row[f"D_pooled_{arm}"]
            for kk in train_maps:
                row[f"d_{kk}"] = row[f"d_{arm}_{kk}"]
            wr.writerow(row)
    out = {"space": a.space, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code": code_identity(),
           "config": {"projections": a.projections, "direction_seed": a.direction_seed, "seed": a.seed,
                      "subsets": a.subsets, "episodes_per_cloud": a.episodes_per_cloud,
                      "floor_subsets": a.floor_subsets, "bootstrap": a.bootstrap, "limit": a.limit or None,
                      "block": a.block, "mixture": not a.no_mixture, "mixture_projections": a.mixture_projections,
                      "quiet_share": QUIET_SHARE, "p": 2, "reference_draw": a.reference_draw,
                      "floor_draw": a.floor_draw, "sets": names, "validation_set": a.validation_set},
           "references": {"file": os.path.abspath(ref_path),
                          "floor_file": os.path.abspath(floor_path) if ref2 else None,
                          "maps": train_maps,
                          "frames_per_map": {str(k): int(len(refs1[k][0])) for k in train_maps},
                          "episodes_per_map": {str(k): int(np.unique(ref1["frames"]["episode"][
                                                   ref1["frames"]["map"] == k]).size) for k in train_maps}},
           "primary_arm": arm, "floor": floor, "checks": checks, "bootstrap": boot, "maps": maps_out,
           "outputs": {"per_episode": os.path.abspath(csv_path)}}
    path = os.path.join(a.out, f"distances_{a.space}.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(_jsonable(out), fh, indent=1)
    os.replace(path + ".tmp", path)
    print(json.dumps({"wrote": path, "maps": len(maps_out), "episodes": len(episode_rows), "primary_arm": arm,
                      "checks": {k: v["pass"] for k, v in checks.items() if isinstance(v, dict)}}), flush=True)
    return 0


# ---------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="The map-distance study: clouds, splits, distances (CPU).")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("clouds", help="draw per-episode frame clouds and write clouds/<space>/<set>.npz")
    c.add_argument("--space", choices=list(SPACES), required=True)
    c.add_argument("--out", default="results/distance_study")
    c.add_argument("--reference", default="", help="the training latents; writes train_draw<draw>.npz")
    c.add_argument("--reference-ids", dest="reference_ids", default="",
                   help="training episode ids to draw from, A:B or a list (the run's training range, 0:2000)")
    c.add_argument("--draw", type=int, default=1, help="1 is the reference; 2 is the disjoint draw for the floor")
    c.add_argument("--episodes-per-map", dest="episodes_per_map", type=int, default=EPISODES_PER_MAP)
    c.add_argument("--eval", action="append", default=[], metavar="NAME=DIR",
                   help="an evaluation corpus (a raw recording directory for --space pixels); repeatable")
    c.add_argument("--frames-from", dest="frames_from", default="",
                   help="another space's cloud directory: read the same episodes at the same tics in this space")
    c.add_argument("--frames-per-episode", dest="frames_per_episode", type=int, default=FRAMES_PER_EPISODE)
    c.add_argument("--bins", type=int, default=MOTION_BINS)
    c.add_argument("--context-frames", dest="context_frames", type=int, default=CONTEXT_FRAMES)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--limit", type=int, default=0, help="smoke runs: at most this many episodes per map and set")
    c.add_argument("--chunk", type=int, default=MOTION_CHUNK, help="latent rows read at a time")

    s = sub.add_parser("splits", help="one make_dense_eval_splits.py-format split file per map, for eval_tf.py")
    s.add_argument("--out", default="results/distance_study")
    s.add_argument("--eval", action="append", default=[], metavar="NAME=DIR")
    s.add_argument("--num-windows", dest="num_windows", type=int, default=NUM_WINDOWS)
    s.add_argument("--context-frames", dest="context_frames", type=int, default=CONTEXT_FRAMES)
    s.add_argument("--limit", type=int, default=0, help="smoke runs: at most this many episodes per map")

    d = sub.add_parser("distances", help="distances_<space>.json, per_episode_<space>.csv, bootstrap_<space>.npz")
    d.add_argument("--space", required=True)
    d.add_argument("--out", default="results/distance_study")
    d.add_argument("--sets", nargs="*", default=None, help="evaluation clouds to score (default: every one)")
    d.add_argument("--reference-draw", dest="reference_draw", type=int, default=1)
    d.add_argument("--floor-draw", dest="floor_draw", type=int, default=2)
    d.add_argument("--validation-set", dest="validation_set", default="val")
    d.add_argument("--projections", type=int, default=N_PROJECTIONS)
    d.add_argument("--direction-seed", dest="direction_seed", type=int, default=0,
                   help="the directions every distance shares (common random numbers)")
    d.add_argument("--seed", type=int, default=0, help="the subset and bootstrap draws")
    d.add_argument("--subsets", type=int, default=SUBSETS)
    d.add_argument("--episodes-per-cloud", dest="episodes_per_cloud", type=int, default=EPISODES_PER_CLOUD)
    d.add_argument("--floor-subsets", dest="floor_subsets", type=int, default=SUBSETS)
    d.add_argument("--bootstrap", type=int, default=0, help="episode redraws per map for the distance error")
    d.add_argument("--no-mixture", dest="no_mixture", action="store_true", help="skip the best-mixture secondary")
    d.add_argument("--mixture-projections", dest="mixture_projections", type=int, default=MIXTURE_PROJECTIONS)
    d.add_argument("--block", type=int, default=DIRECTION_BLOCK)
    d.add_argument("--limit", type=int, default=0, help="smoke runs: only the first N map points")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    return {"clouds": cmd_clouds, "splits": cmd_splits, "distances": cmd_distances}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
