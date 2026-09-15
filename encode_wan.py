"""
Encode per-tic Doom parquet recordings through the Wan 2.1 causal video VAE so that one
latent frame carries exactly one agent decision.

Why a video VAE at all
----------------------
`encode_parquet.py` encodes one *image* per decision with `sd-vae-ft-mse` (4 x 32 x 40 per
frame) and throws the three in-between tics away. The Wan 2.1 VAE
(`AutoencoderKLWan`, 16 channels, 8x spatial, 4x temporal) compresses four consecutive
frames into one latent frame, so the three discarded tics become part of the latent instead
of being dropped. One latent frame then carries the whole 4-tic execution of one decision,
which is exactly the unit our world model predicts.

Alignment schemes
-----------------
The Wan encoder is causal with a fixed temporal layout: the first frame is encoded alone,
then frames arrive in groups of four, so `1 + 4k` input frames give `1 + k` latent frames
(`diffusers/models/autoencoders/autoencoder_kl_wan.py:1143`, diffusers 0.40.0). We line our
decisions up with that layout.

Our corpus keeps every tic. `transitions.valid_transitions` recovers the rows where a
decision *began* and then ran unchanged for `repeat` (=4) tics inside one continuous life;
consecutive such rows spaced exactly 4 apart form a "chain", and a chain never crosses a
death or an interrupted decision. Write a chain's decision rows s_0 < s_1 < ... < s_{k-1},
with s_j = s_0 + 4j.

`wan-chain-v2` (default, `--layout v2`)
.......................................
Row semantics in `record_arnold.py` are that the action on row t is applied from t to t+1.
So a_j's effect spans tics s_j+1 .. s_j+4, and the frame sequence is

    [ s_0 ] + [s_0+1 .. s_0+4] + [s_1+1 .. s_1+4] + ... + [s_{k-2}+1 .. s_{k-2}+4]

The lone leading frame is s_0 itself: the observation before a_0 acts. Chunk j is exactly the
four tics a_j's effect spans, so **latent frame j+1 is fully determined by the history plus
a_j** — no pre-action frame leaks into the chunk the action is supposed to explain. Because
chain decisions are spaced exactly 4 apart, the last decoded frame of chunk j is
s_j+4 = s_{j+1}, the next decision tic, which is the frame the SD rows score.

That invariant is what forces the one loss in v2: for the final decision a_{k-1} the frame
s_{k-1}+4 is not a decision tic of the chain (the chain ended because the next 4-tic window
failed validation, or the life ran out), so **the last decision of every chain is dropped**.
A chain of k decisions yields k-1 chunks, `1 + 4(k-1)` frames and `k` latent frames; a chain
with k = 1 yields nothing. The count is recorded as `dropped_decisions` in the metadata.

`wan-chain-v1` (`--layout v1`)
..............................
The original layout, kept for comparison. The lone frame is the tic before the first decision
tic (s_0 - 1), duplicated from s_0 when the chain starts on the first row of a life, and
chunk j is [s_j .. s_j+3]. No decision is dropped, but each chunk mixes the *pre-action*
observation at s_j with three tics of a_j's effect, and its last frame is not a decision tic.

Common to both
..............
    latent frame 0      the lone leading frame; `action = -1`, `is_lone_first = 1`
    latent frame j >= 1 carries decision j-1's `action`, `tic_start`, `tic_end`,
                        `decision_tic` and `chain_id`

`decision_tic` differs by layout: in v2 it is the *next* decision tic, which equals `tic_end`
and is the frame the chunk's last decoded frame lands on; in v1 it is the decision tic the
chunk starts at, which equals `tic_start`.

The world-model contract is the same either way: given latent frames up to j and the action of
decision j, predict latent frame j+1.

Latents are stored **raw** — the posterior mean straight out of the encoder, no scaling. The
per-channel `latents_mean` / `latents_std` that the Wan pipelines apply
(`pipelines/wan/pipeline_wan_i2v.py:440-459`) are written into the metadata instead, so the
training-side normalisation is a data-loader decision and can be changed without re-encoding.
`wan_data.LatentWindowDataset` applies it by default.

Output per episode
------------------
    <episode_id>.npy        float16 [T_lat, 16, 32, 40], all chains of the episode concatenated
    <episode_id>.meta.json  per-latent-frame arrays (action, tic_start, chain_id,
                            is_lone_first, lone_duplicated) plus the VAE repo id,
                            latents_mean / latents_std, the padding, and the scheme name.

Frames are 320x240; the height is padded to 256 the same way `encode_parquet.py` does it —
zeros appended to the bottom rows of the *already normalised* tensor, i.e. mid-grey in pixel
terms, not black — so latents are 16 x 32 x 40 per frame.

Usage
-----
    python encode_wan.py --parquet-dir /sata2/data/rnagabhi/doom/raw_arnold \
        --out-dir /sata2/data/rnagabhi/doom/latents_arnold_wan \
        --hf-cache /sata2/data/rnagabhi/doom/hf/hub \
        --device cuda:0 --fp16 --batch-chains 4 --workers 8 --resume
"""
import argparse
import glob
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from transitions import canonical_table, life_segments, valid_transitions

LAYOUTS = ("v2", "v1")
SCHEMES = {"v1": "wan-chain-v1", "v2": "wan-chain-v2"}
VAE_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
VAE_SUBFOLDER = "vae"
PAD_TO = 256                 # 320x240 -> 320x256, as in encode_parquet.py
REPEAT = 4                   # tics per decision; also the VAE's temporal compression factor
LATENT_SHAPE = (16, 32, 40)  # z_dim x (256/8) x (320/8)


# --------------------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------------------
def decode_png(b):
    """PNG bytes from the `frame` column -> uint8 HWC RGB."""
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)


def to_model_input(frames_u8, device, dtype):
    """uint8 [T, H, W, 3] -> [1, 3, T, PAD_TO, W] in [-1, 1], height zero-padded at the bottom.

    The padding is applied *after* the [-1, 1] normalisation, which is what
    `encode_parquet.encode_batch` does; the pad value is therefore 0.0 (mid-grey), not black,
    despite that function's comment. Reproduced deliberately so the Wan and SD corpora see the
    same image.
    """
    x = torch.from_numpy(frames_u8).to(device).permute(3, 0, 1, 2).float() / 127.5 - 1.0
    if x.shape[2] < PAD_TO:
        x = torch.nn.functional.pad(x, (0, 0, 0, PAD_TO - x.shape[2]))
    return x.unsqueeze(0).to(dtype)


# --------------------------------------------------------------------------------------
# chain construction
# --------------------------------------------------------------------------------------
def chain_layout(sources, chains, deaths, repeat=REPEAT, layout="v2"):
    """Group accepted decisions into per-chain frame sequences.

    Returns (chains, dropped) where `chains` is a list of dicts, one per chain that yields at
    least one chunk, and `dropped` is the total number of decisions discarded (v2 only):

        chain_id        int, as produced by `transitions.valid_transitions`
        decisions       int64 [n], the decision rows this chain's latent frames 1..n carry
        dropped         int, decisions of this chain that produced no chunk
        lone_row        int, the row used as the lone leading frame
        lone_duplicated bool, True when the lone frame is a copy of s_0 (v1 only)
        chunks          list of int64 row arrays, one per causal encoder step
        rows            int64 [1 + 4n], the concatenated frame sequence in encode order
        tic_start       int64 [n], first row of each chunk
        tic_end         int64 [n], last row of each chunk
        decision_row    int64 [n], v2: the next decision row (== tic_end); v1: s_j (== tic_start)
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}, expected one of {LAYOUTS}")
    bounds = life_segments(deaths)

    def life_start_of(row):
        for a, b in bounds:
            if a <= row < b:
                return a
        raise ValueError(f"row {row} outside every life segment")

    out, dropped_total = [], 0
    for c in np.unique(chains):
        dec = sources[chains == c].astype(np.int64)
        if layout == "v2":
            # a_j's effect spans s_j+1 .. s_j+4, and that last frame must itself be the next
            # decision tic, so a decision is usable only while s_j + repeat is still a decision
            # of this chain. That always excludes the chain's final decision.
            present = set(dec.tolist())
            usable = np.array([s for s in dec.tolist() if s + repeat in present], dtype=np.int64)
            dropped = len(dec) - len(usable)
            dropped_total += dropped
            if len(usable) == 0:
                continue
            lone = int(dec[0])
            duplicated = False
            chunks = [np.array([lone], dtype=np.int64)] + \
                     [np.arange(int(s) + 1, int(s) + 1 + repeat, dtype=np.int64) for s in usable]
            decision_row = usable + repeat
        else:
            usable = dec
            dropped = 0
            s0 = int(dec[0])
            duplicated = s0 - 1 < life_start_of(s0)
            lone = s0 if duplicated else s0 - 1
            chunks = [np.array([lone], dtype=np.int64)] + \
                     [np.arange(int(s), int(s) + repeat, dtype=np.int64) for s in usable]
            decision_row = usable.copy()

        out.append(dict(chain_id=int(c), decisions=usable, dropped=int(dropped),
                        lone_row=int(lone), lone_duplicated=bool(duplicated), chunks=chunks,
                        rows=np.concatenate(chunks),
                        tic_start=np.array([ch[0] for ch in chunks[1:]], dtype=np.int64),
                        tic_end=np.array([ch[-1] for ch in chunks[1:]], dtype=np.int64),
                        decision_row=decision_row))
    return out, dropped_total


def chunk_rows(chain, repeat=REPEAT):
    """Frame rows per causal encoder step: step 0 is the lone frame, step j>=1 is decision j-1."""
    return chain["chunks"]


# --------------------------------------------------------------------------------------
# streaming causal encode
# --------------------------------------------------------------------------------------
class WanChainEncoder:
    """Drives `vae.encoder` chunk by chunk with a persistent feature cache.

    `AutoencoderKLWan.encode` calls `clear_cache()` at both ends of `_encode`
    (`autoencoder_kl_wan.py:1136` and `:1157`), so repeated calls cannot continue a sequence:
    a long chain would have to be materialised as one tensor. The encoder itself is fully
    streaming — `_encode` (`:1143-1154`) just loops over chunks passing `feat_cache` along —
    so we replicate that loop and keep the cache alive across chunks, holding only one chunk
    of decoded frames at a time.

    Several chains are stepped in lockstep on the batch axis. Chains are ordered longest
    first, so a chain that runs out is always a suffix of the batch and the cache is truncated
    with a plain `[:m]` slice.

    `quant_conv` is a `WanCausalConv3d` with kernel size 1 (`autoencoder_kl_wan.py:1049`), so
    its temporal padding is zero and applying it per chunk equals applying it to the
    concatenated sequence. `verify_wan.py` checks this against the stock `vae.encode()`.
    """

    def __init__(self, vae, device, dtype):
        if getattr(vae, "use_tiling", False):
            # width 320 exceeds the default 256 tile width, so tiling would silently blend tiles
            raise ValueError("tiling must stay disabled: 320px frames would be tiled and blended")
        self.vae = vae
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def encode_batch(self, chain_chunks, load_frames):
        """Encode a batch of chains.

        chain_chunks: list of per-chain chunk row lists (from `chunk_rows`), longest first.
        load_frames:  callable(rows) -> uint8 [len(rows), H, W, 3].
        Returns a list of float32 numpy arrays, one per chain, shaped [1 + k, 16, 32, 40].
        """
        lengths = [len(c) for c in chain_chunks]
        if lengths != sorted(lengths, reverse=True):
            raise ValueError("chains must be ordered longest first")
        n_steps = lengths[0]

        self.vae.clear_cache()
        feat = self.vae._enc_feat_map
        outs = [[] for _ in chain_chunks]
        prev_m = None
        for step in range(n_steps):
            m = sum(1 for L in lengths if L > step)
            if prev_m is not None and m < prev_m:
                for i, f in enumerate(feat):
                    if f is not None:
                        feat[i] = f[:m]
            prev_m = m
            per = len(chain_chunks[0][step])
            rows = np.concatenate([chain_chunks[i][step] for i in range(m)])
            frames = load_frames(rows)
            x = torch.cat([to_model_input(frames[i * per:(i + 1) * per], self.device, self.dtype)
                           for i in range(m)], dim=0)
            self.vae._enc_conv_idx = [0]
            h = self.vae.encoder(x, feat_cache=feat, feat_idx=self.vae._enc_conv_idx)
            h = self.vae.quant_conv(h)
            mean = h[:, : self.vae.config.z_dim].float().cpu().numpy()   # DiagonalGaussianDistribution.mode()
            for i in range(m):
                outs[i].append(mean[i])
        self.vae.clear_cache()
        return [np.concatenate(o, axis=1).transpose(1, 0, 2, 3) for o in outs]   # [C,T,H,W] -> [T,C,H,W]


# --------------------------------------------------------------------------------------
# per-episode driver
# --------------------------------------------------------------------------------------
def encode_episode(path, out_dir, encoder, canonical, pool, batch_chains, repeat=REPEAT, layout="v2"):
    """Encode one episode parquet. Returns a summary dict."""
    import pyarrow.parquet as pq

    ep = os.path.basename(path)[: -len(".parquet")]
    t = pq.read_table(path)
    names = set(t.schema.names)
    for c in ("action", "buttons", "deaths", "tic", "frame"):
        if c not in names:
            raise ValueError(f"{path}: missing column {c!r}; schema is {sorted(names)}")

    action = t["action"].to_numpy(zero_copy_only=False)
    buttons = np.array(t["buttons"].to_pylist())
    deaths = t["deaths"].to_numpy(zero_copy_only=False)
    tic = t["tic"].to_numpy(zero_copy_only=False)

    sources, chains = valid_transitions(action, buttons, deaths, repeat=repeat, canonical=canonical)
    if len(sources) == 0:
        return dict(episode=ep, chains=0, latent_frames=0, tic_frames=0, decisions=0,
                    dropped_decisions=0, seconds=0.0, frames_per_s=0.0)

    plan, dropped = chain_layout(sources, chains, deaths, repeat=repeat, layout=layout)
    if not plan:
        return dict(episode=ep, chains=0, latent_frames=0, tic_frames=0, decisions=0,
                    dropped_decisions=int(dropped), seconds=0.0, frames_per_s=0.0)
    plan.sort(key=lambda c: len(c["decisions"]), reverse=True)

    frames_col = t["frame"]

    def load_frames(rows):
        raw = [frames_col[int(r)].as_py() for r in rows]
        return np.stack(list(pool.map(decode_png, raw)))

    t0 = time.time()
    lat_parts, meta_parts, n_tic_frames = [], [], 0
    for i in range(0, len(plan), batch_chains):
        group = plan[i:i + batch_chains]
        chunks = [chunk_rows(c, repeat) for c in group]
        zs = encoder.encode_batch(chunks, load_frames)
        for c, z in zip(group, zs):
            k = len(c["decisions"])
            if z.shape != (1 + k,) + LATENT_SHAPE:
                raise ValueError(f"{ep} chain {c['chain_id']}: latent shape {z.shape} != {(1 + k,) + LATENT_SHAPE}")
            lat_parts.append(z.astype(np.float16))
            n_tic_frames += len(c["rows"])
            lone = tic[c["lone_row"]]
            meta_parts.append(dict(
                action=np.concatenate([[-1], action[c["decisions"]]]).astype(np.int64),
                tic_start=np.concatenate([[lone], tic[c["tic_start"]]]).astype(np.int64),
                tic_end=np.concatenate([[lone], tic[c["tic_end"]]]).astype(np.int64),
                decision_tic=np.concatenate([[lone], tic[c["decision_row"]]]).astype(np.int64),
                chain_id=np.full(1 + k, c["chain_id"], dtype=np.int64),
                is_lone_first=np.concatenate([[1], np.zeros(k, np.int64)]),
                lone_duplicated=np.concatenate([[int(c["lone_duplicated"])], np.zeros(k, np.int64)]),
            ))
    dt = time.time() - t0

    lat = np.concatenate(lat_parts)
    meta_arrays = {k: np.concatenate([m[k] for m in meta_parts]).tolist() for k in meta_parts[0]}
    cfg = encoder.vae.config
    meta = dict(
        scheme=SCHEMES[layout],
        layout=layout,
        episode=ep,
        episode_id=int(t["episode_id"][0].as_py()) if "episode_id" in names else -1,
        map_id=int(t["map_id"][0].as_py()) if "map_id" in names else -1,
        vae_repo=VAE_REPO,
        vae_subfolder=VAE_SUBFOLDER,
        latents_raw=True,
        latents_mean=list(cfg.latents_mean),
        latents_std=list(cfg.latents_std),
        z_dim=int(cfg.z_dim),
        scale_factor_spatial=int(cfg.scale_factor_spatial),
        scale_factor_temporal=int(cfg.scale_factor_temporal),
        frame_size=[320, 240],
        pad_to_height=PAD_TO,
        pad_value_normalized=0.0,
        pad_side="bottom",
        latent_shape=list(LATENT_SHAPE),
        repeat=repeat,
        num_latent_frames=int(lat.shape[0]),
        num_chains=len(plan),
        num_tic_frames=int(n_tic_frames),
        num_decisions=int(sum(len(c["decisions"]) for c in plan)),
        dropped_decisions=int(dropped),
        **meta_arrays,
    )

    npy = os.path.join(out_dir, f"{ep}.npy")
    tmp = npy + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, lat)
    os.replace(tmp, npy)
    with open(os.path.join(out_dir, f"{ep}.meta.json"), "w") as f:
        json.dump(meta, f)

    return dict(episode=ep, chains=len(plan), latent_frames=int(lat.shape[0]),
                tic_frames=int(n_tic_frames), decisions=int(meta["num_decisions"]),
                dropped_decisions=int(dropped), seconds=dt,
                frames_per_s=n_tic_frames / max(dt, 1e-9))


# --------------------------------------------------------------------------------------
def select_episodes(parquet_dir, spec=None, list_file=None):
    """Episode parquets to encode. `spec` is a comma list (or a file); `list_file` is a file with
    one episode id per line. Given both, the union is encoded. Blank lines and `#` comments are
    ignored. An id with no parquet is an error, not a silent skip."""
    paths = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    wanted = set()
    if spec:
        wanted |= ({l.split("#")[0].strip() for l in open(spec)} if os.path.isfile(spec)
                   else {s.strip() for s in spec.split(",")})
    if list_file:
        with open(list_file) as f:
            wanted |= {l.split("#")[0].strip() for l in f}
    wanted.discard("")
    if not wanted:
        return paths
    keep = [p for p in paths if os.path.basename(p)[: -len(".parquet")] in wanted]
    missing = wanted - {os.path.basename(p)[: -len(".parquet")] for p in keep}
    if missing:
        raise FileNotFoundError(f"episodes not found under {parquet_dir}: {sorted(missing)}")
    return keep


def build_canonical(paths, out_dir, canonical_path):
    """Modal control bits per action id. Must match the table `encode_parquet.py` uses, so that
    both encodings accept exactly the same transitions."""
    import pyarrow.parquet as pq
    if canonical_path:
        return {int(k): v for k, v in json.load(open(canonical_path)).items()}
    acts, btns = [], []
    for p in paths:
        t = pq.read_table(p, columns=["action", "buttons"])
        acts.append(t["action"].to_numpy(zero_copy_only=False))
        btns.append(np.array(t["buttons"].to_pylist()))
    table = canonical_table(np.concatenate(acts), np.concatenate(btns))
    with open(os.path.join(out_dir, "canonical_controls.json"), "w") as f:
        json.dump(table, f, indent=1)
    return table


def load_wan_vae(vae_path, hf_cache, device, dtype):
    try:
        from diffusers import AutoencoderKLWan
    except ImportError as e:                     # first shipped in diffusers 0.33.0
        import diffusers
        raise SystemExit(
            f"AutoencoderKLWan is not in diffusers {diffusers.__version__}; it was added in 0.33.0. "
            f"Install with:  pip install -U 'diffusers>=0.33.0'") from e
    kw = dict(torch_dtype=dtype)
    if vae_path:
        vae = AutoencoderKLWan.from_pretrained(vae_path, **kw)
    else:
        if hf_cache:
            kw["cache_dir"] = hf_cache
        vae = AutoencoderKLWan.from_pretrained(VAE_REPO, subfolder=VAE_SUBFOLDER, **kw)
    return vae.to(device).eval().requires_grad_(False)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float16 if args.fp16 else torch.float32
    if args.fp16:
        if device.type == "cpu":
            raise SystemExit("--fp16 needs a CUDA device; fp16 conv3d is not supported on CPU")
        # every diffusers Wan example loads this VAE at torch.float32 even when the transformer
        # runs bf16 (pipelines/wan/pipeline_wan.py:56 and the i2v/v2v/vace examples), so fp16 here
        # is off the supported path. Encode one episode both ways and compare before a full run.
        print("WARNING: diffusers pins the Wan VAE to fp32 in all of its own examples; "
              "validate --fp16 latents against an fp32 encode of the same episode first", flush=True)

    paths = select_episodes(args.parquet_dir, args.episodes, args.episode_list)
    canonical = build_canonical(paths, args.out_dir, args.canonical)
    if args.limit_episodes:
        paths = paths[: args.limit_episodes]
    if args.shard is not None:
        paths = paths[args.shard::args.num_shards]

    vae = load_wan_vae(args.vae_path, args.hf_cache, device, dtype)
    encoder = WanChainEncoder(vae, device, dtype)
    print(f"{len(paths)} episodes on {device}, dtype {dtype}, batch-chains {args.batch_chains}, "
          f"layout {args.layout} ({SCHEMES[args.layout]})", flush=True)

    summary_path = os.path.join(args.out_dir, f"encode_wan_{args.shard or 0:02d}.jsonl")
    t0, total_frames, total_dec, total_dropped = time.time(), 0, 0, 0
    with ThreadPoolExecutor(args.workers) as pool:
        for i, p in enumerate(paths):
            ep = os.path.basename(p)[: -len(".parquet")]
            if args.resume and os.path.exists(os.path.join(args.out_dir, f"{ep}.npy")) \
                    and os.path.exists(os.path.join(args.out_dir, f"{ep}.meta.json")):
                print(f"  [{i + 1}/{len(paths)}] {ep} skipped (resume)", flush=True)
                continue
            r = encode_episode(p, args.out_dir, encoder, canonical, pool, args.batch_chains,
                               layout=args.layout)
            total_frames += r["tic_frames"]
            total_dec += r["decisions"]
            total_dropped += r["dropped_decisions"]
            with open(summary_path, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"  [{i + 1}/{len(paths)}] {ep}: {r['chains']} chains, {r['latent_frames']} latent frames, "
                  f"{r['decisions']} decisions (+{r['dropped_decisions']} dropped), "
                  f"{r['tic_frames']} tic frames, {r['frames_per_s']:.1f} frames/s "
                  f"(run avg {total_frames / max(time.time() - t0, 1e-9):.1f})", flush=True)
    kept = total_dec + total_dropped
    print(f"DONE {total_frames} tic frames in {time.time() - t0:.1f}s; "
          f"{total_dec} decisions kept, {total_dropped} dropped "
          f"({100 * total_dropped / max(kept, 1):.2f}% of accepted transitions)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="encode Doom tic parquet through the Wan 2.1 causal video VAE")
    p.add_argument("--parquet-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--episodes", default=None, help="file with one episode name per line, or a comma-separated list")
    p.add_argument("--episode-list", default=None,
                   help="file with one episode id per line (# comments allowed) selecting a subset "
                        "to encode; union with --episodes when both are given")
    p.add_argument("--layout", choices=LAYOUTS, default="v2",
                   help="v2 (default): lone frame is s_0, chunk j is a_j's effect span "
                        "[s_j+1 .. s_j+4] and ends on the next decision tic, dropping each chain's "
                        "last decision. v1: lone frame is s_0-1, chunk j is [s_j .. s_j+3]")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-chains", type=int, default=4, help="chains stepped in lockstep through the causal encoder")
    p.add_argument("--fp16", action="store_true", help="run the VAE in fp16 (CUDA only); output is fp16 either way")
    p.add_argument("--vae-path", default=None, help="local directory holding the Wan vae/ folder")
    p.add_argument("--hf-cache", default=None, help="HF_HUB cache dir used when downloading the VAE")
    p.add_argument("--workers", type=int, default=8, help="threads for PNG decoding")
    p.add_argument("--resume", action="store_true", help="skip episodes whose .npy and .meta.json both exist")
    p.add_argument("--limit-episodes", type=int, default=0,
                   help="smoke test: encode only the first N episodes. The canonical table is still "
                        "built from every parquet in --parquet-dir unless --canonical is given, "
                        "because a table built from a subset would accept different transitions")
    p.add_argument("--canonical", default=None, help="canonical_controls.json from the SD encode, reused verbatim")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    main(p.parse_args())
