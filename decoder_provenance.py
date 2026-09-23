"""Which frames a decoder was tuned on, and whether a score made with it may claim unseen maps.

The 2026-09-22 review (H4) found that the tuned decoders were trained on the 17-map corpus, whose
training maps include arenas 6 to 8, and that the planned dense tune would have streamed validation
and test episodes. An unseen-map claim is about the whole system: a decoder that saw the unseen maps
defeats it however the dynamics model was trained. So every score must name its decoder, and a
decoder may back an unseen-map claim only if its recorded training episodes are all dense training
ids of the seen arenas.

Provenance comes from, in order:
  1. `provenance.json` inside the decoder directory or its parent, which `finetune_decoder.py` now
     writes with the exact episode ids it trained on;
  2. `release/decoder_registry.json`, by the decoder's directory name or hub id, which records the
     decoders tuned before provenance existed as `provenance: unknown, corpus: all maps`;
  3. otherwise `unknown`.

    python decoder_provenance.py show  /data/doom/vae_decoder_arnold_lpips/vae
    python decoder_provenance.py check /data/doom/vae_decoder_sd1x_mse/vae --claim unseen-map
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(HERE, "release", "decoder_registry.json")


def load_registry(path=REGISTRY_PATH):
    with open(path) as f:
        return json.load(f)["decoders"]


def registry_key(path):
    """The name a decoder is registered under: its tune directory (the parent of `vae/`) or its hub id."""
    p = os.path.normpath(path) if path else "stock"
    if os.path.basename(p).startswith("vae"):
        base = os.path.basename(os.path.dirname(p))
        if base.startswith("vae_decoder"):
            return base
    return os.path.basename(p) if os.path.isdir(p) else (path or "stock")


def provenance_of(path, registry=None):
    """The provenance record of a decoder, with `source` saying where it came from."""
    for d in ([path, os.path.dirname(os.path.normpath(path))] if path and os.path.isdir(path) else []):
        f = os.path.join(d, "provenance.json")
        if os.path.isfile(f):
            with open(f) as fh:
                return {**json.load(fh), "source": f}
    reg = load_registry() if registry is None else registry
    key = registry_key(path)
    if key in reg:
        return {**reg[key], "source": f"registry:{key}"}
    return {"provenance": "unknown", "corpus": "unknown", "source": "none"}


def unseen_claim_problems(prov, split=None):
    """Why a decoder with this provenance cannot back an unseen-map claim; empty if it can.

    Refused: an unknown provenance; any training map among the unseen segment's maps; any training
    episode of the unseen segment; any training episode of a seen segment outside its train range.
    A public decoder that was never tuned on Doom frames passes.
    """
    from doom_data import dense_ids, load_dense_split
    split = load_dense_split() if split is None else split
    if prov.get("provenance") == "public":
        return []
    if prov.get("provenance") != "recorded":
        return [f"provenance {prov.get('provenance', 'unknown')} (corpus: {prov.get('corpus', 'unknown')})"]
    unseen_maps = set()
    unseen_segments = set()
    for name, seg in split["segments"].items():
        if "train" not in seg["ranges"]:
            unseen_segments.add(name)
            unseen_maps |= {int(m) for m in seg["maps"]}
    bad = []
    for t in prov.get("train_episodes") or []:
        seg, ids, maps = t.get("segment"), [int(e) for e in t.get("ids") or []], set(t.get("maps") or [])
        if maps & unseen_maps:
            bad.append(f"trained on unseen maps {sorted(maps & unseen_maps)} ({t.get('dir')})")
        if seg in unseen_segments:
            bad.append(f"trained on {len(ids)} episode(s) of the unseen segment {seg}")
        elif seg in split["segments"]:
            outside = sorted(set(ids) - set(dense_ids(split, seg, "train")))
            if outside:
                bad.append(f"trained on {len(outside)} {seg} episode(s) outside its train range: {outside[:8]}")
        elif not maps:
            bad.append(f"training episodes in {t.get('dir')} have no recorded maps")
    if not prov.get("train_episodes"):
        bad.append("no training episodes recorded")
    return bad


def describe(path, registry=None):
    """What an evaluator records beside every score: name, identity, provenance and the claim verdict."""
    from eval_identity import decoder_identity
    prov = provenance_of(path, registry)
    try:
        ident = decoder_identity(path)
    except SystemExit as e:
        ident = f"unreadable: {e}"
    problems = unseen_claim_problems(prov)
    return {"name": registry_key(path), "path": path or "stock", "identity": ident,
            "provenance": prov.get("provenance", "unknown"), "corpus": prov.get("corpus", "unknown"),
            "provenance_source": prov.get("source"), "unseen_map_claim": "ok" if not problems else "refused",
            "unseen_map_problems": problems}


def main(args):
    d = describe(args.decoder)
    if args.cmd == "show":
        print(f"{d['name']} identity={d['identity']} provenance={d['provenance']} "
              f"corpus={str(d['corpus']).replace(' ', '_')} unseen_map_claim={d['unseen_map_claim']}")
        return 0
    if args.claim == "unseen-map" and d["unseen_map_problems"]:
        print(f"{d['name']} cannot back an unseen-map claim: " + "; ".join(d["unseen_map_problems"]))
        return 1
    print(f"{d['name']} may back an {args.claim} claim")
    return 0


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["show", "check"])
    p.add_argument("decoder", help="a decoder directory, or a hub id, or 'stock'")
    p.add_argument("--claim", default="unseen-map", choices=["unseen-map"])
    return p


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
