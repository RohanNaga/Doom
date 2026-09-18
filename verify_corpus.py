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


def compare_episode(new_files, ref_files, latent_channels, ref_latent_channels, hw=(32, 40)):
    """Per-episode structural comparison. Returns (stats, [problem strings])."""
    lat = np.load(new_files[0], mmap_mode="r")
    meta = np.load(new_files[1])
    bad = []
    if tuple(lat.shape[1:]) != (latent_channels,) + hw:
        bad.append(f"latent shape {tuple(lat.shape[1:])} != {(latent_channels,) + hw}")
    if lat.dtype != np.float16:
        bad.append(f"latent dtype {lat.dtype} != float16")
    if ref_files is not None:
        rlat = np.load(ref_files[0], mmap_mode="r")
        rmeta = np.load(ref_files[1])
        if lat.shape[0] != rlat.shape[0]:
            bad.append(f"{lat.shape[0]} frames vs {rlat.shape[0]} in the reference")
        if tuple(rlat.shape[1:]) != (ref_latent_channels,) + hw:
            bad.append(f"reference latent shape {tuple(rlat.shape[1:])}")
        for col in META_COLS:
            if col in rmeta.files:
                if col not in meta.files:
                    bad.append(f"missing metadata column {col}")
                elif not np.array_equal(meta[col], rmeta[col]):
                    bad.append(f"metadata column {col} differs")
    cid = meta["chain_id"] if "chain_id" in meta.files else np.zeros(0)
    return {"frames": int(lat.shape[0]), "chains": chain_count(cid),
            "map_id": int(meta["map_id"][0]) if "map_id" in meta.files else -1}, bad


def verify(new_dir, ref_dir, latent_channels, ref_latent_channels, transition_stats=None, max_report=20):
    new, ref = episode_files(new_dir), (episode_files(ref_dir) if ref_dir else {})
    report = {"new_dir": new_dir, "ref_dir": ref_dir, "episodes": len(new), "ref_episodes": len(ref),
              "frames": 0, "chains": 0, "maps": {}, "problems": []}
    if ref:
        missing, extra = sorted(set(ref) - set(new)), sorted(set(new) - set(ref))
        if missing:
            report["problems"].append(f"{len(missing)} episodes missing, first {missing[:max_report]}")
        if extra:
            report["problems"].append(f"{len(extra)} episodes not in the reference, first {extra[:max_report]}")
    for ep in sorted(new):
        stats, bad = compare_episode(new[ep], ref.get(ep), latent_channels, ref_latent_channels)
        report["frames"] += stats["frames"]
        report["chains"] += stats["chains"]
        report["maps"][str(stats["map_id"])] = report["maps"].get(str(stats["map_id"]), 0) + 1
        report["problems"] += [f"ep_{ep:05d}: {b}" for b in bad[:max_report]]
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
    p.add_argument("--out", default="")
    a = p.parse_args()
    r = verify(a.new, a.ref or None, a.latent_channels, a.ref_latent_channels, a.transition_stats or None)
    print(json.dumps({k: v for k, v in r.items() if k != "problems"}, indent=1))
    for problem in r["problems"][:50]:
        print("PROBLEM:", problem)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1)
    raise SystemExit(0 if r["ok"] else 1)
