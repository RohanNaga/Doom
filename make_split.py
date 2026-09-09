"""Write the episode-level, map-aware split for a latents directory produced by encode_parquet.py.

    python make_split.py --latents-dir data/latents_arnold --out data/split_arnold.json --holdout-maps 16,17
"""
import argparse
import json

from doom_data import make_split_by_map, save_split

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--latents-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--holdout-maps", default="16,17")
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    maps = tuple(int(x) for x in a.holdout_maps.split(",") if x)
    split = make_split_by_map(a.latents_dir, maps, a.holdout_frac, a.seed)
    save_split(split, a.out)
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in split.items()}, indent=1))
