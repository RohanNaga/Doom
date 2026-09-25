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
"""
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
