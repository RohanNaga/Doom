"""Check that a re-encoded corpus mirrors the one it replaces, and report the numbers.

Re-encoding the corpus in a new latent space must change exactly one thing: the array of floats
per frame. The episodes kept, the frames kept inside each episode, the decision alignment, the
chain boundaries and every metadata column are functions of the recording and the canonical
control table, not of the autoencoder, so they have to come out identical. This compares the two
directories column by column and prints counts, so a mismatch names the episode and the column
instead of surfacing later as a silently shorter training set.

The frame/chain/transition identity is the same one `transition_stats.py` reports: a chain of n
verified transitions contributes n + 1 encoded frames (every source plus the chain's final
target), so `frames - chains == transitions` over the whole corpus.

    python verify_corpus.py --new $D/latents_arnold_sd35 --ref $D/latents_arnold_aligned \
        --latent-channels 16 --ref-latent-channels 4 \
        --transition-stats $D/raw_arnold/card/transition_stats.json --out /tmp/verify_sd35.json
"""
import argparse
import glob
import json
import os

import numpy as np

# Columns that describe the recording rather than the latent space; all of them must match.
META_COLS = ["action", "buttons", "health", "ammo", "kills", "deaths", "frags", "pos_x", "pos_y",
             "angle", "tic", "map_id", "episode_id", "chain_id"]

# Two encodings of one frame should be bit-identical, but the encoder runs under bf16 autocast and
# cuDNN may pick a different kernel for a different batch shape, so the same frame can land one
# float16 step apart. 2^-10 is that step at the magnitudes these latents take (abs mean about 0.7).
LATENT_TOL = 2.0 ** -10


def episode_files(latents_dir):
    """{episode_id: (latents_path, meta_path)} for every finished episode in a directory."""
    out = {}
    for lat in sorted(glob.glob(os.path.join(latents_dir, "ep_*_latents.npy"))):
        meta = lat.replace("_latents.npy", "_meta.npz")
        if os.path.isfile(meta):
            out[int(os.path.basename(lat).split("_")[1])] = (lat, meta)
    return out


def chain_count(chain_id):
    """Number of distinct chains inside one episode's stored frames.

    Chain ids are assigned per episode and stored in encounter order, so a change of value marks
    a boundary; counting transitions of the array rather than unique values keeps this correct
    even if a generator ever reuses an id.
    """
    if len(chain_id) == 0:
        return 0
    return int(1 + np.count_nonzero(chain_id[1:] != chain_id[:-1]))


def compare_episode(new_files, ref_files, latent_channels, ref_latent_channels, hw=(32, 40),
                    reduce_decisions=False):
    """Per-episode structural comparison. Returns (stats, [problem strings]).

    With `reduce_decisions` the new corpus is a per-tic one (`encode_parquet.py --every-tic`) and
    only its `is_decision` rows are compared, which is the claim that makes a per-tic corpus safe:
    selecting those rows must give back the stride-4 corpus it sits alongside. When both sides are
    in the same latent space the latent values are compared too, not merely the row structure.
    """
    lat = np.load(new_files[0], mmap_mode="r")
    meta = np.load(new_files[1])
    bad = []
    stored_frames = int(lat.shape[0])
    sel = None
    if reduce_decisions:
        if "is_decision" not in meta.files:
            bad.append("--reduce-decisions needs an is_decision column; encode with --every-tic")
        else:
            sel = np.asarray(meta["is_decision"]).astype(bool)
            lat = np.asarray(lat)[sel]
    if tuple(lat.shape[1:]) != (latent_channels,) + hw:
        bad.append(f"latent shape {tuple(lat.shape[1:])} != {(latent_channels,) + hw}")
    if lat.dtype != np.float16:
        bad.append(f"latent dtype {lat.dtype} != float16")
    diff = None
    ref_frames = 0
    if ref_files is not None:
        rlat = np.load(ref_files[0], mmap_mode="r")
        rmeta = np.load(ref_files[1])
        ref_frames = int(rlat.shape[0])
        if lat.shape[0] != rlat.shape[0]:
            bad.append(f"{lat.shape[0]} frames vs {rlat.shape[0]} in the reference")
        if tuple(rlat.shape[1:]) != (ref_latent_channels,) + hw:
            bad.append(f"reference latent shape {tuple(rlat.shape[1:])}")
        for col in META_COLS:
            if col in rmeta.files:
                if col not in meta.files:
                    bad.append(f"missing metadata column {col}")
                else:
                    got = np.asarray(meta[col])
                    got = got[sel] if sel is not None else got
                    if not np.array_equal(got, rmeta[col]):
                        bad.append(f"metadata column {col} differs")
        # same latent space and same row count: the floats themselves are comparable
        if latent_channels == ref_latent_channels and lat.shape == rlat.shape:
            d = np.abs(np.asarray(lat, dtype=np.float32) - np.asarray(rlat, dtype=np.float32))
            diff = (float(d.max()), float(d.mean()), int(d.size))
            if diff[0] > LATENT_TOL:
                bad.append(f"latent values differ by up to {diff[0]:.4g} (tolerance {LATENT_TOL})")
    cid = meta["chain_id"] if "chain_id" in meta.files else np.zeros(0)
    cid = np.asarray(cid)[sel] if (sel is not None and len(cid)) else cid
    return {"frames": int(lat.shape[0]), "stored_frames": stored_frames, "ref_frames": ref_frames,
            "chains": chain_count(cid), "latent_diff": diff,
            "map_id": int(meta["map_id"][0]) if "map_id" in meta.files else -1}, bad


def verify(new_dir, ref_dir, latent_channels, ref_latent_channels, transition_stats=None, max_report=20,
           reduce_decisions=False):
    new, ref = episode_files(new_dir), (episode_files(ref_dir) if ref_dir else {})
    report = {"new_dir": new_dir, "ref_dir": ref_dir, "episodes": len(new), "ref_episodes": len(ref),
              "frames": 0, "stored_frames": 0, "ref_frames": 0, "chains": 0, "maps": {},
              "reduce_decisions": bool(reduce_decisions), "problems": []}
    if ref:
        missing, extra = sorted(set(ref) - set(new)), sorted(set(new) - set(ref))
        if missing:
            report["problems"].append(f"{len(missing)} episodes missing, first {missing[:max_report]}")
        if extra:
            report["problems"].append(f"{len(extra)} episodes not in the reference, first {extra[:max_report]}")
    worst, total_abs, total_n = 0.0, 0.0, 0
    for ep in sorted(new):
        stats, bad = compare_episode(new[ep], ref.get(ep), latent_channels, ref_latent_channels,
                                     reduce_decisions=reduce_decisions)
        report["frames"] += stats["frames"]
        report["stored_frames"] += stats["stored_frames"]
        report["ref_frames"] += stats["ref_frames"]
        report["chains"] += stats["chains"]
        report["maps"][str(stats["map_id"])] = report["maps"].get(str(stats["map_id"]), 0) + 1
        report["problems"] += [f"ep_{ep:05d}: {b}" for b in bad[:max_report]]
        if stats["latent_diff"]:
            mx, mean, n = stats["latent_diff"]
            worst = max(worst, mx); total_abs += mean * n; total_n += n
    if total_n:
        report["latent_diff"] = {"max_abs": worst, "mean_abs": total_abs / total_n,
                                 "values": total_n, "tolerance": LATENT_TOL}
    report["transitions"] = report["frames"] - report["chains"]
    if transition_stats:
        with open(transition_stats) as f:
            ts = json.load(f)
        want = ts.get("transitions", ts.get("verified_transitions", ts.get("total_transitions")))
        want_chains = ts.get("chains", ts.get("num_chains"))
        report["transition_stats"] = {"transitions": want, "chains": want_chains}
        if want is not None and want != report["transitions"]:
            report["problems"].append(f"{report['transitions']} transitions vs {want} in transition_stats.json")
        if want_chains is not None and want_chains != report["chains"]:
            report["problems"].append(f"{report['chains']} chains vs {want_chains} in transition_stats.json")
    report["ok"] = not report["problems"]
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--new", required=True, help="the re-encoded corpus")
    p.add_argument("--ref", default="", help="the corpus it must mirror (omit to check --new alone)")
    p.add_argument("--latent-channels", type=int, default=16)
    p.add_argument("--ref-latent-channels", type=int, default=4)
    p.add_argument("--transition-stats", default="")
    p.add_argument("--reduce-decisions", action="store_true",
                   help="--new is a per-tic corpus (encode_parquet.py --every-tic): compare only its "
                        "is_decision rows against --ref, which must give back that stride-4 corpus")
    p.add_argument("--out", default="")
    a = p.parse_args()
    r = verify(a.new, a.ref or None, a.latent_channels, a.ref_latent_channels, a.transition_stats or None,
               reduce_decisions=a.reduce_decisions)
    print(json.dumps({k: v for k, v in r.items() if k != "problems"}, indent=1))
    for problem in r["problems"][:50]:
        print("PROBLEM:", problem)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1)
    raise SystemExit(0 if r["ok"] else 1)
