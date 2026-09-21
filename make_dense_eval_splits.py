"""Write the split file an evaluation corpus needs, from the encoded episodes that are there.

`eval_tf.py` and `rollout_eval.py` both take `--split` and `--subset`, and both read
`split[subset]` as the list of episodes to score. A dense evaluation corpus is a whole held-out
range, so every episode in the directory is the evaluation set and there is nothing to draw: the
split file just names them.

**One subset key for every corpus.** The key is always `val`, whatever the corpus is called, so a
caller never has to remember that the test corpus's episodes live under `test` while the validation
corpus's live under `val`. `after_nexttic.sh` passes `--subset val` everywhere for that reason. The
corpus's real name and its id range are recorded alongside, so the file says what it is.

    python make_dense_eval_splits.py --latents-dir $D/latents_arnold_dense_pertic_eval/val --name val
    python make_dense_eval_splits.py --latents-dir $D/latents_arnold_dense_pertic_eval/test --name test
"""
import argparse
import json
import os

SUBSET = "val"          # the one key every evaluator is pointed at


def corpus_name(latents_dir):
    """The corpus's name IS its directory basename, so a caller cannot pass a different one.

    The encoder's internal tag for the unseen corpus is `unseen` while its directory is
    `arenas_678`; naming the file after the tag wrote `.../arenas_678/split_unseen.json` while
    `after_nexttic.sh` looked for `.../split_arenas_678.json`. Deriving both from the directory is
    what makes the writer and the reader agree by construction.
    """
    return os.path.basename(os.path.normpath(latents_dir))


def split_path(latents_dir):
    """THE canonical location: `split_<corpus>.json` in the PARENT of the corpus directory.

    The parent, because `after_nexttic.sh` holds one evaluation root (`$LE`) and reads
    `$LE/split_<corpus>.json` for every corpus; one directory of split files beside the corpora.
    """
    return os.path.join(os.path.dirname(os.path.normpath(latents_dir)),
                        f"split_{corpus_name(latents_dir)}.json")


def build(latents_dir, name=None, out=None):
    """Write the canonical split file for an encoded corpus and return (path, contents)."""
    from doom_data import list_latent_episodes
    name = name or corpus_name(latents_dir)
    eps = sorted(ep for ep, _, _ in list_latent_episodes(latents_dir))
    split = {SUBSET: eps, "meta": {"corpus": name, "latents_dir": latents_dir,
                                   "num_episodes": len(eps),
                                   "episode_range": f"{eps[0]}:{eps[-1] + 1}",
                                   "subset_key": SUBSET,
                                   "note": "every encoded episode of a held-out dense range; the key is "
                                           "'val' for every corpus so the evaluators take one --subset"}}
    path = out or split_path(latents_dir)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(split, f, indent=1)
    os.replace(tmp, path)
    return path, split


def main(args):
    path, split = build(args.latents_dir, args.name or None, args.out or None)
    print(json.dumps({"wrote": path, "num_episodes": split["meta"]["num_episodes"],
                      "episode_range": split["meta"]["episode_range"], "subset_key": SUBSET}))
    return 0


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--latents-dir", required=True, help="an encoded per-tic evaluation corpus")
    p.add_argument("--name", default="",
                   help="override the corpus name; by default it is the latents directory's basename, which is "
                        "what keeps the writer's filename and the evaluation script's identical")
    p.add_argument("--out", default="", help="write here instead of the canonical <parent>/split_<corpus>.json")
    return p


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
