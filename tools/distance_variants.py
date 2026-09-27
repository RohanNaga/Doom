"""Distance variants on the frozen SD 1 clouds: does the choice of distance matter?

Rohan asked (Sep 27 2026) whether a KL divergence, or another distance, would do better than the frame
distance D, and whether the motion weighting should go. This tool answers by computing. It rebuilds D's
clouds exactly as `distance_study.py distances` does (per target map, ten 4-episode clouds of 1,000
frames drawn as five disjoint pairs with the same seeds; the nearest of the four training maps'
12,500-frame references per cloud; the mean over clouds; the floor from 1,000-frame clouds of the
disjoint second reference draw against the same references) and evaluates ten distances on them:

    D               motion-weighted sliced W2, nearest map (the frozen D, reproduced as the check)
    sw_uniform      sliced W2 with every frame weighted equally, nearest map
    frechet         Frechet distance between Gaussian fits (the FID formula, a squared W2), nearest map
    gauss_kl_ts     Gaussian KL(target || source), nearest map
    gauss_kl_st     Gaussian KL(source || target), nearest map
    mmd2            unbiased MMD^2, Gaussian kernel at the median pairwise distance of the reference
    knn_kl_ts       kNN KL estimator (Wang, Kulkarni and Verdu 2009, k = 5), target || source
    knn_kl_st       the same estimator, source || target
    D_pooled        motion-weighted sliced W2 to the pooled corpus (each training map an equal share)
    frechet_pooled  Frechet distance to the Gaussian fit of the pooled corpus

The sliced variants run on `distance_study.distance_table`, the engine D itself runs on, with the same
1,000 directions (seed 0). The other variants use every frame with equal weight, their textbook forms;
the weighting question is answered by D against sw_uniform. For every variant the tool reports the value
per map, the floor range over the 40 training-versus-training clouds, whether every unseen arena lies
above the floor's maximum (the family test), and the Spearman correlations of the 13 arenas' values with
the adaptation outcomes (A0, A at the 4k budget, the gain, the half-gap budget with censored arenas tied
at the top) and with D. Nothing is bootstrapped: this is a rank-agreement table.

    python tools/distance_variants.py \
        --reference $C/train_draw1.npz --floor $C/train_draw2.npz \
        --target $C/arenas13.npz --target $C/val_v1_frozen.npz \
        --frozen $C/distances_sd1.json --frozen results/distance_study/distances_sd1.json \
        --adapt paper/tables/tuned/adapt_summary.json --out results/distance_variants

writes `variants_sd1.json`, `variants_sd1.csv` and `VARIANTS.md` under `--out`. numpy and scipy only,
CPU, about ten minutes on a laptop for the full table.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import zlib

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from distance_study import (DIRECTION_BLOCK, EPISODES_PER_CLOUD, N_PROJECTIONS, SUBSETS,  # noqa: E402
                            _frame_index, code_identity, distance_table, episode_subsets, load_cloud,
                            map_references, motion_weights, pooled_arms, random_directions)

K_NEIGHBOURS = 5
BANDWIDTH_POINTS = 5000     # reference frames the median heuristic is computed on (12.5 M pairs)
FROZEN_TOLERANCE = 5e-4     # D must reproduce the frozen table to three decimals
SIGNIFICANT = 0.05

# name -> (short label, one-line definition); the order is the table's
VARIANTS = {
    "D": ("D (motion-weighted SW2)", "sliced W2, motion-weighted, nearest training map; the frozen D"),
    "sw_uniform": ("SW2 (unweighted)", "sliced W2, every frame weighted equally, nearest training map"),
    "frechet": ("Frechet (FID formula)", "||m1 - m2||^2 + tr(S1 + S2 - 2 (S1 S2)^1/2) of Gaussian fits, nearest map"),
    "gauss_kl_ts": ("Gaussian KL(target||source)", "KL between Gaussian fits, target to source, nearest map"),
    "gauss_kl_st": ("Gaussian KL(source||target)", "KL between Gaussian fits, source to target, nearest map"),
    "mmd2": ("MMD^2 (Gaussian kernel)", "unbiased MMD^2, bandwidth the median pairwise distance of the reference"),
    "knn_kl_ts": ("kNN KL(target||source)", "Wang-Kulkarni-Verdu kNN KL estimator, k = 5, target to source"),
    "knn_kl_st": ("kNN KL(source||target)", "Wang-Kulkarni-Verdu kNN KL estimator, k = 5, source to target"),
    "D_pooled": ("SW2 to the pooled corpus", "motion-weighted sliced W2 to all four maps pooled, equal shares"),
    "frechet_pooled": ("Frechet to the pooled corpus", "Frechet distance to the Gaussian fit of all four maps"),
}
PER_MAP = ("D", "sw_uniform", "frechet", "gauss_kl_ts", "gauss_kl_st", "mmd2", "knn_kl_ts", "knn_kl_st")
OUTCOMES = {"A0": "A0", "A_budget": "A at 4k", "gain": "gain", "cost": "half-gap budget", "S0": "S0"}


# ---------------------------------------------------------------------------------------------
# the estimators, standalone (the tests pin these against closed forms)
# ---------------------------------------------------------------------------------------------

def gaussian_fit(x, weights=None):
    """(mean, covariance) of an (n, d) cloud; unbiased covariance, or reliability-weighted with `weights`."""
    x = np.asarray(x, dtype=np.float64)
    if weights is None:
        return x.mean(axis=0), np.atleast_2d(np.cov(x, rowvar=False))
    w = np.asarray(weights, dtype=np.float64)
    return (w @ x) / w.sum(), np.atleast_2d(np.cov(x, rowvar=False, aweights=w))


def ridge(cov, frac):
    """cov + frac x (mean variance) x I; `frac` 0 returns `cov` itself."""
    if not frac:
        return cov
    d = cov.shape[0]
    return cov + frac * (np.trace(cov) / d) * np.eye(d)


def _sqrt_psd(s):
    """The symmetric square root of a positive semi-definite matrix, by its eigendecomposition."""
    w, v = np.linalg.eigh(s)
    return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T


def _trace_sqrt_product(s0_half, s1):
    """tr((S0 S1)^1/2) = tr((S0^1/2 S1 S0^1/2)^1/2), a symmetric PSD form with real eigenvalues."""
    m = s0_half @ s1 @ s0_half
    return float(np.sum(np.sqrt(np.clip(np.linalg.eigvalsh((m + m.T) / 2), 0.0, None))))


def frechet_distance(m0, s0, m1, s1):
    """The Frechet distance between N(m0, s0) and N(m1, s1) as FID computes it: the SQUARED W2."""
    dm = np.asarray(m0, dtype=np.float64) - np.asarray(m1, dtype=np.float64)
    return float(dm @ dm + np.trace(s0) + np.trace(s1) - 2.0 * _trace_sqrt_product(_sqrt_psd(s0), s1))


def _logdet(s):
    sign, ld = np.linalg.slogdet(s)
    if sign <= 0:
        raise ValueError("a covariance is not positive definite; pass a ridge")
    return float(ld)


def gaussian_kl(m0, s0, m1, s1):
    """KL(N(m0, s0) || N(m1, s1)) = (tr(S1^-1 S0) + dm' S1^-1 dm - d + ln|S1| - ln|S0|) / 2, in nats."""
    s0, s1 = np.atleast_2d(s0), np.atleast_2d(s1)
    dm = np.asarray(m1, dtype=np.float64) - np.asarray(m0, dtype=np.float64)
    i1 = np.linalg.inv(s1)
    return float(0.5 * (np.trace(i1 @ s0) + dm @ i1 @ dm - s0.shape[0] + _logdet(s1) - _logdet(s0)))


def median_bandwidth(y, max_points=BANDWIDTH_POINTS, seed=0):
    """The median heuristic: the median Euclidean distance between pairs of (a seeded subsample of) y."""
    y = np.asarray(y, dtype=np.float64)
    if len(y) > max_points:
        y = y[np.sort(np.random.default_rng(seed).choice(len(y), size=max_points, replace=False))]
    return float(np.median(pdist(y)))


def _sq_dists(a, b, a2=None, b2=None):
    """(len(a), len(b)) squared Euclidean distances by the Gram expansion, clipped at zero."""
    a2 = np.sum(a * a, axis=1) if a2 is None else a2
    b2 = np.sum(b * b, axis=1) if b2 is None else b2
    return np.maximum(a2[:, None] + b2[None, :] - 2.0 * (a @ b.T), 0.0)


def mmd2_unbiased(x, y, bandwidth):
    """Unbiased MMD^2 (Gretton et al. 2012) with k(a, b) = exp(-|a - b|^2 / (2 bandwidth^2)).

    Within-cloud sums leave out the diagonal, so the estimate is unbiased and can be slightly negative.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    n, m = len(x), len(y)
    g = -0.5 / bandwidth ** 2
    kxx = np.exp(g * _sq_dists(x, x))
    kyy = np.exp(g * _sq_dists(y, y))
    kxy = np.exp(g * _sq_dists(x, y))
    return float((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                 - 2.0 * kxy.mean())


def _knn_kl_from(rho, nu, d, n, m):
    """(d / n) sum log(nu / rho) + log(m / (n - 1)): the estimate from the k-th neighbour distances."""
    if np.any(rho <= 0) or np.any(nu <= 0):
        raise ValueError("a k-th neighbour distance is zero (more than k duplicate frames); the kNN KL "
                         "estimator needs distinct points")
    return float(d * np.mean(np.log(nu / rho)) + np.log(m / (n - 1)))


def knn_kl(x, y, k=K_NEIGHBOURS):
    """The kNN estimate of KL(P || Q) from x ~ P (n, d) and y ~ Q (m, d), Wang, Kulkarni and Verdu 2009.

    rho_i is the distance from x_i to its k-th nearest neighbour among the other points of x (the point
    itself is left out, its duplicates are not), nu_i the distance to its k-th nearest neighbour in y.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    n, d = x.shape
    rho = cKDTree(x).query(x, k=k + 1)[0][:, k]
    nu = cKDTree(y).query(x, k=k)[0]
    nu = nu[:, k - 1] if nu.ndim == 2 else nu
    return _knn_kl_from(rho, nu, d, n, len(y))


# ---------------------------------------------------------------------------------------------
# the batched engine: each reference prepared once, each cloud once, one distance matrix per pair
# ---------------------------------------------------------------------------------------------

class _Stats:
    """Moments, the k-th neighbour distance within the cloud and the off-diagonal kernel sum."""

    def __init__(self, x, bandwidth, k, ridge_frac, block):
        self.x = np.asarray(x, dtype=np.float64)
        self.sq = np.sum(self.x * self.x, axis=1)
        self.n, self.d = self.x.shape
        self.k, self.gamma = int(k), -0.5 / bandwidth ** 2
        if self.n <= self.k:
            raise ValueError(f"{self.n} points, the kNN estimator needs more than k = {self.k}")
        self.mean, cov = gaussian_fit(self.x)
        self.cov = ridge(cov, ridge_frac)
        eig = np.linalg.eigvalsh(self.cov)
        self.condition = float(eig[-1] / eig[0]) if eig[0] > 0 else float("inf")
        self.inv = np.linalg.inv(self.cov)
        self.logdet = _logdet(self.cov)
        self.half = _sqrt_psd(self.cov)
        self.rho = np.empty(self.n)
        self.kself = 0.0
        for s in range(0, self.n, max(1, int(block))):
            e = min(self.n, s + max(1, int(block)))
            d2 = _sq_dists(self.x[s:e], self.x, self.sq[s:e], self.sq)
            d2[np.arange(e - s), np.arange(s, e)] = np.inf           # leave the point itself out
            self.rho[s:e] = np.sqrt(np.partition(d2, self.k - 1, axis=1)[:, self.k - 1])
            self.kself += float(np.exp(self.gamma * d2).sum())         # exp(-inf) = 0 on the diagonal


class Reference(_Stats):
    """A training map's reference cloud, prepared once for every cloud it is compared with."""

    def __init__(self, y, bandwidth, k=K_NEIGHBOURS, ridge_frac=0.0, block=1000):
        super().__init__(y, bandwidth, k, ridge_frac, block)


class CloudStats(_Stats):
    """A target (or floor) cloud, prepared once for the four references."""

    def __init__(self, x, bandwidth, k=K_NEIGHBOURS, ridge_frac=0.0, block=2000):
        super().__init__(x, bandwidth, k, ridge_frac, block)


def pair_variants(c, r):
    """{variant: value} of the six non-sliced variants between a cloud `c` and a reference `r`."""
    d2 = _sq_dists(c.x, r.x, c.sq, r.sq)                                # (n, m), shared by three estimators
    k = c.k
    nu_ts = np.sqrt(np.partition(d2, k - 1, axis=1)[:, k - 1])          # cloud points' k-th neighbour in r
    nu_st = np.sqrt(np.partition(d2, k - 1, axis=0)[k - 1])             # reference points' k-th neighbour in c
    kxy = float(np.exp(c.gamma * d2).sum())
    dm = r.mean - c.mean
    return {
        "frechet": float(dm @ dm + np.trace(c.cov) + np.trace(r.cov) - 2.0 * _trace_sqrt_product(c.half, r.cov)),
        "gauss_kl_ts": float(0.5 * (np.sum(r.inv * c.cov) + dm @ r.inv @ dm - c.d + r.logdet - c.logdet)),
        "gauss_kl_st": float(0.5 * (np.sum(c.inv * r.cov) + dm @ c.inv @ dm - c.d + c.logdet - r.logdet)),
        "mmd2": c.kself / (c.n * (c.n - 1)) + r.kself / (r.n * (r.n - 1)) - 2.0 * kxy / (c.n * r.n),
        "knn_kl_ts": _knn_kl_from(c.rho, nu_ts, c.d, c.n, r.n),
        "knn_kl_st": _knn_kl_from(r.rho, nu_st, c.d, r.n, c.n),
    }


# ---------------------------------------------------------------------------------------------
# D's clouds, rebuilt exactly
# ---------------------------------------------------------------------------------------------

def target_clouds(cloud, set_name, seed=0, size=EPISODES_PER_CLOUD, n_subsets=SUBSETS):
    """[(key, map, subset episodes, frame indices)] of every map of a target set, as `cmd_distances` draws them.

    The per-map generator is seeded with (seed, crc32(set name), map), so the set name must be the one
    the frozen table was computed under (the cloud file's `meta["set"]`).
    """
    index = _frame_index(cloud["frames"])
    ep = cloud["episodes"]
    out = []
    for m in sorted(int(v) for v in np.unique(ep["map"])):
        eps = sorted(int(e) for e, mm, dr in zip(ep["id"], ep["map"], ep["drawn"]) if mm == m and dr > 0)
        if not eps:
            continue
        rng = np.random.default_rng([seed, zlib.crc32(set_name.encode()), m])
        subsets, _ = episode_subsets(eps, size, n_subsets, rng)
        for sub in subsets:
            out.append((f"{set_name}/{m}", m, sub, np.concatenate([index[e] for e in sub])))
    return out


def floor_clouds(cloud, train_maps, seed=0, size=EPISODES_PER_CLOUD, n_subsets=SUBSETS):
    """[(key, map, subset episodes, frame indices)] of the floor: the second draw's clouds per training map."""
    index = _frame_index(cloud["frames"])
    ep = cloud["episodes"]
    out = []
    for k in train_maps:
        ek = sorted(int(e) for e, mm, dr in zip(ep["id"], ep["map"], ep["drawn"]) if mm == k and dr > 0)
        subsets, _ = episode_subsets(ek, size, n_subsets, np.random.default_rng([seed, 7, k]))
        out += [(f"floor/{k}", k, sub, np.concatenate([index[e] for e in sub])) for sub in subsets]
    return out


# ---------------------------------------------------------------------------------------------
# outcomes, correlations, reproduction
# ---------------------------------------------------------------------------------------------

def _read_json(path):
    with open(path) as fh:
        return json.load(fh)


def load_outcomes(path):
    """{arena: {A0, A_budget, gain, cost, censored, S0, D}} from adapt_summary.json, censored cost tied at the top."""
    s = _read_json(path)
    top = s.get("spearman_cost_censored_as", 8000)
    out = {}
    for r in s["per_arena"]:
        cens = bool(r.get("censored_half_gap")) or r.get("cost_half_gap") is None
        out[int(r["arena"])] = {"A0": r["A0"], "A_budget": r["A_budget"], "gain": r["gain"],
                                "cost": top if cens else r["cost_half_gap"], "censored": cens,
                                "S0": r.get("S0"), "D": r.get("D")}
    return out, top, s.get("spearman", {})


def _rho(a, b):
    rho, p = spearmanr(a, b)
    return {"rho": float(rho), "p": float(p), "n": len(a)}


def reproduction(results, frozen_paths, floor_frames, tol=FROZEN_TOLERANCE):
    """Differences from the frozen tables: D, D_uniform, D_pooled, the subsets and the floor."""
    checks, worst, subsets_ok, matched = [], 0.0, True, 0
    for path in frozen_paths:
        fz = _read_json(path)
        for rec in fz["maps"]:
            ours = results.get(rec["key"])
            if ours is None:
                continue
            matched += 1
            diff = {"D": abs(ours["D"] - rec["D"]), "D_uniform": abs(ours["sw_uniform"] - rec["D_uniform"]),
                    "D_pooled": abs(ours["D_pooled"] - rec["D_pooled"])}
            same = [list(map(int, s)) for s in rec["subset_episodes"]] == ours["subsets"]
            subsets_ok &= same
            worst = max(worst, *diff.values())
            checks.append({"file": path, "key": rec["key"], "frozen_D": rec["D"], "D": ours["D"],
                           "abs_diff": diff, "same_subsets": same})
        if fz.get("floor"):
            for arm, name in (("motion", "D"), ("uniform", "sw_uniform")):
                d = float(np.max(np.abs(np.asarray(fz["floor"][arm]["values"]) - np.asarray(floor_frames[name]))))
                worst = max(worst, d)
                checks.append({"file": path, "key": f"floor/{arm}", "max_abs_diff": d})
    return {"tolerance": tol, "matched_maps": matched, "max_abs_diff": worst, "same_subsets": subsets_ok,
            "pass": bool(matched > 0 and subsets_ok and worst <= tol), "checks": checks}


# ---------------------------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------------------------

def compute(a):
    """Every variant for every target cloud and floor cloud; returns the result dict the writers read."""
    t0 = time.time()
    ref1, ref2 = load_cloud(a.reference), load_cloud(a.floor)
    if set(ref1["frames"]["episode"].tolist()) & set(ref2["frames"]["episode"].tolist()):
        raise SystemExit(f"{a.reference} and {a.floor} share episodes; the floor needs disjoint draws")
    train_maps = sorted(int(m) for m in np.unique(ref1["frames"]["map"]))
    dim = ref1["frames"]["feats"].shape[1]
    refs = map_references(ref1, train_maps)
    pooled_y, pooled_w = pooled_arms(refs)

    clouds = []                          # (key, map, subset episodes, features, motion weights, kind)
    targets = {}
    for path in a.target:
        c = load_cloud(path)
        name = c["meta"].get("set") or os.path.splitext(os.path.basename(path))[0]
        if c["frames"]["feats"].shape[1] != dim:
            raise SystemExit(f"{path}: {c['frames']['feats'].shape[1]}-dimensional, the reference is {dim}")
        targets[name] = path
        for key, m, sub, idx in target_clouds(c, name, a.seed, a.episodes_per_cloud, a.subsets):
            clouds.append((key, m, sub, c["frames"]["feats"][idx].astype(np.float64),
                           motion_weights(c["frames"]["motion"][idx]), "target"))
    for key, m, sub, idx in floor_clouds(ref2, train_maps, a.seed, a.episodes_per_cloud, a.floor_subsets):
        clouds.append((key, m, sub, ref2["frames"]["feats"][idx].astype(np.float64),
                       motion_weights(ref2["frames"]["motion"][idx]), "floor"))
    print(json.dumps({"clouds": len(clouds), "targets": list(targets), "train_maps": train_maps}), flush=True)

    # the sliced variants, on D's own engine and directions
    dirs = random_directions(dim, a.projections, a.direction_seed)
    table_refs = {m: refs[m] for m in train_maps}
    table_refs["pooled"] = (pooled_y, pooled_w)
    T = distance_table([(x, {"motion": w, "uniform": None}) for _, _, _, x, w, _ in clouds], table_refs, dirs,
                       ("motion", "uniform"), a.block)
    print(json.dumps({"sliced": "done", "seconds": round(time.time() - t0, 1)}), flush=True)

    # the other variants: one bandwidth for everything, set on the whole reference draw
    bw = median_bandwidth(pooled_y, a.bandwidth_points, a.seed)
    prepared = {m: Reference(refs[m][0], bw, a.k, a.ridge) for m in train_maps}
    pm, ps = gaussian_fit(pooled_y, weights=pooled_w["uniform"])
    ps = ridge(ps, a.ridge)
    R = len(train_maps)
    per_ref = {v: np.empty((len(clouds), R)) for v in PER_MAP}
    per_ref["D"][:] = T[0, :, :R]
    per_ref["sw_uniform"][:] = T[1, :, :R]
    pooled = {"D_pooled": T[0, :, R], "frechet_pooled": np.empty(len(clouds))}
    cond = []
    for ci, (_, _, _, x, _, _) in enumerate(clouds):
        cs = CloudStats(x, bw, a.k, a.ridge)
        cond.append(cs.condition)
        for ri, m in enumerate(train_maps):
            for v, val in pair_variants(cs, prepared[m]).items():
                per_ref[v][ci, ri] = val
        dm = pm - cs.mean
        pooled["frechet_pooled"][ci] = float(dm @ dm + np.trace(cs.cov) + np.trace(ps)
                                             - 2.0 * _trace_sqrt_product(cs.half, ps))
        if (ci + 1) % 25 == 0:
            print(json.dumps({"clouds_done": ci + 1, "seconds": round(time.time() - t0, 1)}), flush=True)

    # per cloud: nearest map per variant; per map: the mean over its clouds
    nearest_val = {v: per_ref[v].min(axis=1) for v in PER_MAP}
    nearest_map = {v: [train_maps[i] for i in per_ref[v].argmin(axis=1)] for v in PER_MAP}
    nearest_val.update(pooled)
    by_key = {}
    for ci, (key, m, sub, _, _, kind) in enumerate(clouds):
        by_key.setdefault(key, {"map": m, "kind": kind, "clouds": [], "subsets": []})
        by_key[key]["clouds"].append(ci)
        by_key[key]["subsets"].append([int(e) for e in sub])
    maps = {}
    for key, g in by_key.items():
        if g["kind"] != "target":
            continue
        own = g["clouds"]
        rec = {"set": key.split("/")[0], "map": g["map"], "subsets": g["subsets"]}
        for v in VARIANTS:
            vals = nearest_val[v][own]
            rec[v] = float(np.mean(vals))
            rec[f"{v}_sd_over_clouds"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            if v in PER_MAP:
                near = [nearest_map[v][c] for c in own]
                rec[f"{v}_nearest"] = {str(k): near.count(k) for k in sorted(set(near))}
                rec[f"{v}_per_reference"] = {str(k): float(np.mean(per_ref[v][own, i]))
                                             for i, k in enumerate(train_maps)}
        maps[key] = rec
    floor_idx = [ci for ci, c in enumerate(clouds) if c[5] == "floor"]
    floor_vals = {v: [float(nearest_val[v][c]) for c in floor_idx] for v in VARIANTS}
    agree = {v: float(np.mean([nearest_map[v][c] == nearest_map["D"][c] for c, cl in enumerate(clouds)
                               if cl[5] == "target"])) for v in PER_MAP}
    return {"maps": maps, "floor": floor_vals, "train_maps": train_maps, "targets": targets, "bandwidth": bw,
            "condition": {"clouds_max": float(np.max(cond)), "clouds_median": float(np.median(cond)),
                          "references": {str(m): prepared[m].condition for m in train_maps}},
            "nearest_agreement_with_D": agree, "seconds": round(time.time() - t0, 1)}


def summarise(res, outcomes, arena_set, validation_set):
    """Per variant: the per-map values, the floor, the family test and the Spearman correlations."""
    arenas = sorted(m["map"] for m in res["maps"].values() if m["set"] == arena_set)
    missing = [m for m in arenas if m not in outcomes]
    if missing:
        raise SystemExit(f"no adaptation outcome for arenas {missing}")
    vals = sorted(m["map"] for m in res["maps"].values() if m["set"] == validation_set)
    D = [res["maps"][f"{arena_set}/{m}"]["D"] for m in arenas]
    out = {}
    for v, (label, what) in VARIANTS.items():
        av = [res["maps"][f"{arena_set}/{m}"][v] for m in arenas]
        vv = [res["maps"][f"{validation_set}/{m}"][v] for m in vals]
        fl = res["floor"][v]
        lo, hi = float(np.min(fl)), float(np.max(fl))
        fam = {"all_arenas_above_floor": bool(min(av) > hi), "arenas_above_floor": int(sum(x > hi for x in av)),
               "n_arenas": len(av), "margin": float(min(av) - hi),
               "ratio_min_arena_to_floor_max": float(min(av) / hi) if hi > 0 else None,
               "validation_inside_floor": bool(all(lo <= x <= hi for x in vv)) if vv else None,
               "validation_below_floor_max": bool(all(x <= hi for x in vv)) if vv else None}
        sp = {o: _rho(av, [outcomes[m][o] for m in arenas]) for o in OUTCOMES}
        sp["D"] = _rho(av, D)
        out[v] = {"label": label, "what": what,
                  "per_map": {k: {"value": m[v], "sd_over_clouds": m[f"{v}_sd_over_clouds"],
                                  **({"nearest": m[f"{v}_nearest"], "per_reference": m[f"{v}_per_reference"]}
                                     if v in PER_MAP else {})}
                              for k, m in res["maps"].items()},
                  "floor": {"min": lo, "max": hi, "mean": float(np.mean(fl)), "values": fl},
                  "arenas": {"min": float(min(av)), "max": float(max(av))},
                  "validation": {"min": float(min(vv)), "max": float(max(vv))} if vv else None,
                  "family": fam, "spearman": sp}
        if v in res["nearest_agreement_with_D"]:
            out[v]["nearest_agreement_with_D"] = res["nearest_agreement_with_D"][v]
    return out, arenas, vals


# ---------------------------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------------------------

def _f(x, nd=3):
    if x is None:
        return "n/a"
    if x != 0 and (abs(x) >= 1000 or abs(x) < 10 ** -nd):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def _rho_cell(x):
    """A Spearman table cell: signed rho, starred when p < 0.05."""
    return f"{x['rho']:+.2f}{'*' if x['p'] < SIGNIFICANT else ''}"


def reading(var, arenas, res, arena_set):
    """One factual paragraph from the numbers: the family test, ordering, agreement with D, the weighting."""
    names = list(var)
    fail = [v for v in names if not var[v]["family"]["all_arenas_above_floor"]]
    if not fail:
        fam = (f"All {len(names)} variants place every one of the {len(arenas)} unseen arenas above their own "
               "training floor, so the family separation does not depend on the choice of distance.")
    else:
        fam = (f"{len(names) - len(fail)} of {len(names)} variants place every unseen arena above their own training "
               "floor; " + "; ".join(f"{var[v]['label']} ({var[v]['family']['arenas_above_floor']} of {len(arenas)})"
                                     for v in fail) + " do not.")
    out_val = [v for v in names if var[v]["family"]["validation_below_floor_max"] is False]
    if out_val:
        fam += (" A training map's own validation clouds sit above the floor maximum under "
                + "; ".join(f"{var[v]['label']} ({_f(var[v]['validation']['max'])} against "
                            f"{_f(var[v]['floor']['max'])})" for v in out_val) + ".")
    tests = [(abs(var[v]["spearman"][o]["rho"]), v, o) for v in names for o in ("A0", "A_budget", "gain", "cost")]
    sig = [f"{var[v]['label']} / {OUTCOMES[o]}" for _, v, o in tests if var[v]["spearman"][o]["p"] < SIGNIFICANT]
    _, bv, bo = max(tests)
    b = var[bv]["spearman"][bo]
    order = (f"The largest rank correlation of any variant with any of the four outcomes is {b['rho']:+.2f} "
             f"({var[bv]['label']} against {OUTCOMES[bo]}, p = {b['p']:.2f}, n = {len(arenas)}); ")
    if sig:
        order += (f"{len(sig)} of {len(tests)} variant-outcome pairs reach p < {SIGNIFICANT} ({'; '.join(sig)}), "
                  f"where about {len(tests) * SIGNIFICANT:.0f} would by chance.")
    else:
        order += f"none of the {len(tests)} variant-outcome pairs reaches p < {SIGNIFICANT}."
    others = [v for v in names if v != "D"]
    rd = [var[v]["spearman"]["D"]["rho"] for v in others]
    agree = (f"Against D, the other variants rank the arenas with Spearman {min(rd):+.2f} "
             f"({var[others[int(np.argmin(rd))]]['label']}) to {max(rd):+.2f}.")
    diff = max(abs(m["D"] - m["sw_uniform"]) for m in res["maps"].values() if m["set"] == arena_set)
    weight = (f"Dropping the motion weights moves D by at most {diff:.3f} on the arenas (their spread is "
              f"{var['D']['arenas']['max'] - var['D']['arenas']['min']:.3f}) and ranks them with Spearman "
              f"{var['sw_uniform']['spearman']['D']['rho']:+.2f} against the weighted D, but moves the floor's "
              f"maximum from {var['D']['floor']['max']:.3f} to {var['sw_uniform']['floor']['max']:.3f}.")
    _, sv = max((abs(var[v]["spearman"]["S0"]["rho"]), v) for v in names)
    s0 = (f"The strongest link of any variant to the model's zero-shot skill S0 is "
          f"{var[sv]['spearman']['S0']['rho']:+.2f} ({var[sv]['label']}).")
    return " ".join([fam, order, agree, weight, s0])


def write_outputs(a, res, var, arenas, vals, outcomes, repro, adapt_spearman, top):
    """variants_<space>.json, variants_<space>.csv and VARIANTS.md under `a.out`."""
    os.makedirs(a.out, exist_ok=True)
    arena_set = a.arena_set
    text = reading(var, arenas, res, arena_set)

    def md5(p):
        with open(p, "rb") as fh:
            return hashlib.md5(fh.read()).hexdigest()
    with open(os.path.abspath(__file__), "rb") as fh:
        tool_sha = hashlib.sha256(fh.read()).hexdigest()
    try:
        commit = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    ins = [a.reference, a.floor, *a.target]
    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "code": {"git_commit": commit, "distance_variants_sha256": tool_sha, **code_identity()},
           "inputs": {os.path.abspath(p): md5(p) for p in ins},
           "adapt_summary": os.path.abspath(a.adapt), "frozen": [os.path.abspath(p) for p in a.frozen],
           "config": {"projections": a.projections, "direction_seed": a.direction_seed, "seed": a.seed,
                      "subsets": a.subsets, "episodes_per_cloud": a.episodes_per_cloud,
                      "floor_subsets": a.floor_subsets, "k": a.k, "bandwidth_points": a.bandwidth_points,
                      "ridge": a.ridge, "arena_set": arena_set, "validation_set": a.validation_set,
                      "censored_cost_as": top, "nearest": "per cloud, the minimum over the training maps; per map, "
                                                          "the mean over its clouds"},
           "bandwidth": res["bandwidth"], "covariance_condition": res["condition"],
           "reading": text, "reproduction": repro,
           "D_outcome_spearman_frozen": adapt_spearman,
           "arenas": arenas, "validation_maps": vals, "train_maps": res["train_maps"],
           "outcomes": {str(m): outcomes[m] for m in arenas},
           "variants": var, "seconds": res["seconds"]}
    jpath = os.path.join(a.out, f"variants_{a.space}.json")
    with open(jpath + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(jpath + ".tmp", jpath)

    cpath = os.path.join(a.out, f"variants_{a.space}.csv")
    with open(cpath, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["set", "map", *VARIANTS, "D_nearest", "A0", "A_budget", "gain", "cost", "censored", "S0"])
        for key in [f"{arena_set}/{m}" for m in arenas] + [f"{a.validation_set}/{m}" for m in vals]:
            m = res["maps"][key]
            o = outcomes.get(m["map"], {}) if m["set"] == arena_set else {}
            near = max(m["D_nearest"], key=m["D_nearest"].get)
            wr.writerow([m["set"], m["map"], *[f"{m[v]:.6g}" for v in VARIANTS], near,
                         *[o.get(x, "") for x in ("A0", "A_budget", "gain", "cost", "censored", "S0")]])
        for stat in ("min", "max"):
            wr.writerow(["floor", stat, *[f"{var[v]['floor'][stat]:.6g}" for v in VARIANTS], "", "", "", "", "",
                         "", ""])

    L = ["# Distance variants on the frozen SD 1 clouds", "",
         f"Generated by `tools/distance_variants.py` ({time.strftime('%Y-%m-%d')}); numbers in "
         f"`variants_{a.space}.json` and `variants_{a.space}.csv`.", "",
         "**Reading.** " + text, "",
         "## The variants against the floor and the outcomes", "",
         f"Spearman over the {len(arenas)} arenas (censored half-gap budgets tied at {top}); `*` marks p < 0.05. "
         "Family test: every arena above the floor's maximum over the 40 training-versus-training clouds.", "",
         "| variant | floor [min, max] | training validation | unseen arenas | arenas above floor | vs D | "
         "A0 | A at 4k | gain | budget | S0 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in var.values():
        fam = s["family"]
        val = f"{_f(s['validation']['min'])} to {_f(s['validation']['max'])}" if s["validation"] else "n/a"
        above = f"{'yes' if fam['all_arenas_above_floor'] else 'no'} ({fam['arenas_above_floor']}/{fam['n_arenas']})"
        rhos = " | ".join(_rho_cell(s["spearman"][o]) for o in ("D", "A0", "A_budget", "gain", "cost", "S0"))
        L.append(f"| {s['label']} | [{_f(s['floor']['min'])}, {_f(s['floor']['max'])}] | {val} | "
                 f"{_f(s['arenas']['min'])} to {_f(s['arenas']['max'])} | {above} | {rhos} |")
    L += ["", "## Values per map", "",
          "Mean over the map's ten clouds of the per-cloud nearest-map value (pooled variants: the pooled corpus).", "",
          "| map | " + " | ".join(s["label"] for s in var.values()) + " |",
          "|---|" + "---|" * len(var)]
    for key in [f"{arena_set}/{m}" for m in sorted(arenas, key=lambda m: res["maps"][f"{arena_set}/{m}"]["D"])] + \
               [f"{a.validation_set}/{m}" for m in vals]:
        m = res["maps"][key]
        label = f"arena {m['map']}" if m["set"] == arena_set else f"train map {m['map']} (validation)"
        L.append(f"| {label} | " + " | ".join(_f(m[v]) for v in var) + " |")
    L += ["| floor min | " + " | ".join(_f(s["floor"]["min"]) for s in var.values()) + " |",
          "| floor max | " + " | ".join(_f(s["floor"]["max"]) for s in var.values()) + " |", "",
          "## Definitions and configuration", ""]
    L += [f"- **{s['label']}**: {s['what']}." for s in var.values()]
    cond = res["condition"]
    ridge_note = ("no ridge was added." if not a.ridge
                  else f"a ridge of {a.ridge} x the mean variance is added to every covariance.")
    L += ["", f"- Clouds: {a.subsets} clouds of {a.episodes_per_cloud} episodes per target map (250 frames per "
              "episode), seeded as `distance_study.py distances` seeds them; references are the four training maps' "
              f"full draw-1 clouds; the floor is {a.floor_subsets} draw-2 clouds per training map against the same "
              "references, nearest map.",
          f"- Sliced variants: {a.projections} directions, seed {a.direction_seed}, `distance_study.distance_table`.",
          f"- MMD bandwidth: {res['bandwidth']:.4f}, the median pairwise distance of {a.bandwidth_points} seeded "
          "frames of the pooled draw-1 reference, one bandwidth for every comparison.",
          f"- Covariances (192-d): condition numbers up to {cond['clouds_max']:.3g} for the 1,000-frame clouds "
          f"(median {cond['clouds_median']:.3g}) and "
          f"{min(cond['references'].values()):.3g} to {max(cond['references'].values()):.3g} for the references; "
          + ridge_note,
          f"- Reproduction: D, the unweighted SW2 and the pooled SW2 match the frozen tables to "
          f"{repro['max_abs_diff']:.1e} over {repro['matched_maps']} maps and the 40 floor clouds, with identical "
          f"episode subsets ({'pass' if repro['pass'] else 'FAIL'}).",
          "- Nearest-map agreement with D (share of target clouds whose nearest training map is D's): "
          + ", ".join(f"{var[v]['label']} {var[v]['nearest_agreement_with_D']:.2f}" for v in PER_MAP if v != "D") + "."]
    with open(os.path.join(a.out, "VARIANTS.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(json.dumps({"wrote": [jpath, cpath, os.path.join(a.out, "VARIANTS.md")], "reproduced": repro["pass"]}),
          flush=True)


# ---------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--reference", required=True, help="train_draw1.npz, the four training maps' reference clouds")
    p.add_argument("--floor", required=True, help="train_draw2.npz, the disjoint second draw for the floor")
    p.add_argument("--target", action="append", required=True, help="a target cloud file (repeatable)")
    p.add_argument("--adapt", required=True, help="adapt_summary.json with per_arena outcomes")
    p.add_argument("--frozen", action="append", default=[], help="a frozen distances_<space>.json to reproduce")
    p.add_argument("--arena-set", default="arenas13")
    p.add_argument("--validation-set", default="val")
    p.add_argument("--out", required=True)
    p.add_argument("--space", default="sd1")
    p.add_argument("--projections", type=int, default=N_PROJECTIONS)
    p.add_argument("--direction-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subsets", type=int, default=SUBSETS)
    p.add_argument("--floor-subsets", type=int, default=SUBSETS)
    p.add_argument("--episodes-per-cloud", type=int, default=EPISODES_PER_CLOUD)
    p.add_argument("--block", type=int, default=DIRECTION_BLOCK)
    p.add_argument("--k", type=int, default=K_NEIGHBOURS)
    p.add_argument("--bandwidth-points", type=int, default=BANDWIDTH_POINTS)
    p.add_argument("--ridge", type=float, default=0.0,
                   help="add this fraction of the mean variance to every covariance (0: none)")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    outcomes, top, adapt_spearman = load_outcomes(a.adapt)
    res = compute(a)
    var, arenas, vals = summarise(res, outcomes, a.arena_set, a.validation_set)
    floor_frames = {v: res["floor"][v] for v in ("D", "sw_uniform")}
    repro = reproduction(res["maps"], a.frozen, floor_frames) if a.frozen else \
        {"pass": None, "matched_maps": 0, "max_abs_diff": float("nan"), "note": "no frozen table given"}
    write_outputs(a, res, var, arenas, vals, outcomes, repro, adapt_spearman, top)
    return 0 if repro["pass"] in (True, None) else 1


if __name__ == "__main__":
    sys.exit(main())
