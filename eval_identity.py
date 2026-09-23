"""The identity of a file a score depends on: its SHA-256, cached beside it.

`after_nexttic.sh` used to skip a scoring stage whenever its output file existed, so a result computed
for one checkpoint could be relabelled with a newer one's name by the provenance step that followed.
Every cached stage is now keyed by the checkpoint's path, stored step and content hash (and the
decoder's), and reruns when any of them differs.

Hashing a 37 GB recovery checkpoint takes about a minute, so the digest is cached in
`<file>.sha256` next to it together with the file's size and modification time; a changed file gets a
new size or mtime and is hashed again. An unwritable directory just means no cache.

    python eval_identity.py sha256 results/040-unet-nexttic/snap_0290000.pt
    python eval_identity.py decoder /data/doom/vae_decoder_arnold_lpips/vae
"""
import argparse
import hashlib
import json
import os

CHUNK = 1 << 24
DECODER_WEIGHTS = ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin")


def _stat_key(path):
    st = os.stat(path)
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def sha256_file(path, cache=True):
    """Hex SHA-256 of a file's bytes; the `<path>.sha256` cache is used only while size and mtime match."""
    key = _stat_key(path)
    side = path + ".sha256"
    if cache:
        try:
            with open(side) as f:
                c = json.load(f)
            if c.get("size") == key["size"] and c.get("mtime_ns") == key["mtime_ns"] and c.get("sha256"):
                return c["sha256"]
        except (OSError, ValueError):
            pass
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    digest = h.hexdigest()
    if cache:
        try:
            tmp = f"{side}.tmp.{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump({**key, "sha256": digest}, f)
            os.replace(tmp, side)
        except OSError:
            pass
    return digest


def decoder_identity(path):
    """`sha256:<digest>` of a local decoder's weight file, or `hub:<id>` for a named hub decoder.

    A local directory without weights is refused: `after_nexttic.sh` falls back to the stock decoder
    in that case, and the fallback has to be visible rather than hashed as if it were the tuned one.
    """
    if not path or not os.path.exists(path):
        return f"hub:{path or 'stock'}"
    if os.path.isfile(path):
        return "sha256:" + sha256_file(path)
    for name in DECODER_WEIGHTS:
        w = os.path.join(path, name)
        if os.path.isfile(w):
            return "sha256:" + sha256_file(w)
    raise SystemExit(f"{path} holds no decoder weights ({' or '.join(DECODER_WEIGHTS)})")


def main(args):
    if args.cmd == "sha256":
        for p in args.paths:
            print(sha256_file(p, cache=not args.no_cache))
    else:
        for p in args.paths:
            print(decoder_identity(p))
    return 0


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["sha256", "decoder"])
    p.add_argument("paths", nargs="+")
    p.add_argument("--no-cache", dest="no_cache", action="store_true", help="hash the bytes even if cached")
    return p


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
