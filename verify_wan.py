"""
Local CPU verification for `encode_wan.py` / `wan_data.py`.

Four checks, all runnable without a GPU and without the real corpus:

1. `shapes`     synthetic 1 + 4*8 frame clip at 320x256 through the stock `AutoencoderKLWan`:
                latent shape, and round-trip PSNR of the decode.
2. `streaming`  our chunk-by-chunk `WanChainEncoder` against diffusers' own `vae.encode()` on
                the same clip, which is what proves the persistent feature cache, the
                lockstep batching over chains and the per-chunk `quant_conv` are all exact.
3. `endtoend`   a tiny fake parquet in the real corpus schema with two chains, run through
                `encode_episode`, checking that latent frame j maps to decision j-1 and that
                no encoder chunk straddles a chain.
4. `loader`     `wan_data.WanWindowDataset` shapes, chain confinement, the lone-frame rule and
                subset-directory skipping.

Checks 3 and 4 run for both layouts, `wan-chain-v2` (default) and `wan-chain-v1`.

Usage:
    python verify_wan.py --vae-path <dir with the Wan vae/ folder>   # or --hf-cache
"""
import argparse
import io
import json
import os
import resource
import sys
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image, ImageDraw

import encode_wan
from encode_wan import (LATENT_SHAPE, PAD_TO, WanChainEncoder, chain_layout, chunk_rows,
                        encode_episode, load_wan_vae)

H, W = 240, 320
NUM_DECISIONS = 8


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1024   # macOS bytes, Linux kibibytes


def synthetic_frames(n, h=PAD_TO, w=W, seed=0):
    """Moving shapes on a gradient: smooth, deterministic, and not degenerate for a VAE."""
    rng = np.random.RandomState(seed)
    bg = np.linspace(20, 200, w, dtype=np.float32)[None, :, None] * np.ones((h, 1, 3), np.float32)
    bg[..., 1] = np.linspace(200, 20, h, dtype=np.float32)[:, None]
    out = []
    cx, cy = 40.0, 40.0
    vx, vy = 7.0, 5.0
    for t in range(n):
        img = Image.fromarray(bg.astype(np.uint8))
        d = ImageDraw.Draw(img)
        d.ellipse([cx - 28, cy - 28, cx + 28, cy + 28], fill=(240, 60, 40))
        d.rectangle([w - cx - 40, h - cy - 40, w - cx + 40, h - cy + 40], fill=(30, 70, 230))
        d.line([0, int(cy), w, int(h - cy)], fill=(250, 250, 60), width=5)
        out.append(np.asarray(img, dtype=np.uint8))
        cx += vx
        cy += vy
        if not 40 < cx < w - 40:
            vx = -vx
        if not 40 < cy < h - 40:
            vy = -vy
    del rng
    return np.stack(out)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


# --------------------------------------------------------------------------------------
def check_shapes(vae, device):
    n = 1 + 4 * NUM_DECISIONS
    frames = synthetic_frames(n)
    x = encode_wan.to_model_input(frames, device, torch.float32)
    print(f"[shapes] input {tuple(x.shape)} range [{x.min():.2f}, {x.max():.2f}]")

    t0 = time.time()
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode()
    t_enc = time.time() - t0
    print(f"[shapes] latent {tuple(z.shape)}  ({n} frames -> {z.shape[2]} latent frames, "
          f"expected {1 + (n - 1) // 4})")
    assert z.shape == (1, 16, 1 + (n - 1) // 4, PAD_TO // 8, W // 8), z.shape

    rss_enc = peak_rss_mb()
    print(f"[shapes] peak RSS after encode only {rss_enc:.0f} MB "
          f"(the production job never decodes; the figure below includes the decode)")

    t0 = time.time()
    with torch.no_grad():
        rec = vae.decode(z).sample
    t_dec = time.time() - t0
    rec_u8 = ((rec[0].permute(1, 2, 3, 0).clamp(-1, 1).float().numpy() + 1) * 127.5).round().astype(np.uint8)
    p = psnr(frames, rec_u8)
    print(f"[shapes] round-trip PSNR {p:.2f} dB   encode {t_enc:.1f}s ({n / t_enc:.2f} frames/s), "
          f"decode {t_dec:.1f}s")
    assert p > 30.0, f"PSNR {p:.2f} dB <= 30 dB"
    return z, frames, n / t_enc


def check_streaming(vae, device, z_ref, frames):
    """Our streaming encoder must reproduce diffusers' `vae.encode()` bit-for-bit (up to fp32 noise)."""
    enc = WanChainEncoder(vae, device, torch.float32)
    k = (len(frames) - 1) // 4
    chunks = [[np.arange(1)] + [np.arange(1 + 4 * j, 1 + 4 * (j + 1)) for j in range(k)]]

    def load(rows):
        return frames[rows]

    z = enc.encode_batch(chunks, load)[0]
    ref = z_ref[0].permute(1, 0, 2, 3).numpy()
    d = np.abs(z - ref).max()
    print(f"[streaming] single chain vs vae.encode(): max abs diff {d:.3e}  shape {z.shape}")
    assert z.shape == ref.shape and d < 1e-4, (z.shape, ref.shape, d)

    # same thing with a shorter second chain sharing the batch, to exercise the cache truncation
    short = [[np.arange(1)] + [np.arange(1 + 4 * j, 1 + 4 * (j + 1)) for j in range(3)]]
    zs = enc.encode_batch(chunks + short, load)
    d0 = np.abs(zs[0] - ref).max()
    d1 = np.abs(zs[1] - enc.encode_batch(short, load)[0]).max()
    print(f"[streaming] batched (len {len(chunks[0])} + {len(short[0])}): long-chain diff {d0:.3e}, "
          f"short-chain diff vs solo {d1:.3e}")
    assert d0 < 1e-4 and d1 < 1e-4, (d0, d1)


# --------------------------------------------------------------------------------------
def fake_parquet(path, seed=0):
    """A two-chain episode in the real `record_arnold.py` schema.

    Rows 0..31 are life 0 and rows 32..35 life 1 (`deaths` increments at 32). Actions are held
    for exactly 4 tics on the life-anchored grid except for one deliberate mid-decision change
    at rows 12..13, which is what forces a second chain. Chain A starts on the first row of its
    life, so under v1 its lone leading frame has to be a duplicate; chain B starts mid-life, so
    it gets a real preceding tic. Life 1 is deliberately too short to contain a full decision,
    which checks that a truncated life contributes nothing.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    ctrl = {0: "100000000", 1: "010000000", 2: "001000000", 3: "000100000", 4: "000010000"}
    action, buttons, deaths = [], [], []

    def hold(a, n, death):
        for _ in range(n):
            action.append(a)
            buttons.append(ctrl[a] + "00")     # weapon-select bits past CONTROL_BITS, ignored
            deaths.append(death)

    # life 0: three clean decisions, then a decision cut short at 2 tics, then two clean ones
    hold(0, 4, 0); hold(1, 4, 0); hold(2, 4, 0)      # rows 0..11   -> chain A
    hold(3, 2, 0)                                     # rows 12..13  -> breaks the grid
    hold(4, 4, 0); hold(0, 4, 0); hold(2, 4, 0); hold(3, 4, 0)   # rows 14..29 -> chain B
    hold(1, 2, 0)                                     # rows 30..31  -> tail, not a full decision
    # life 1: too short to hold a whole decision, so it must contribute no chain
    hold(2, 4, 1)                                     # rows 32..35

    n = len(action)
    frames = synthetic_frames(n, h=H, w=W, seed=seed)
    png = []
    for f in frames:
        buf = io.BytesIO()
        Image.fromarray(f).save(buf, format="PNG", compress_level=1)
        png.append(buf.getvalue())

    table = pa.table({
        "episode_id": pa.array([7] * n, pa.int32()),
        "map_id": pa.array([3] * n, pa.int8()),
        "tic": pa.array(list(range(n)), pa.int32()),
        "action": pa.array(action, pa.int16()),
        "buttons": pa.array(buttons, pa.string()),
        "health": pa.array([100] * n, pa.int16()),
        "ammo": pa.array([50] * n, pa.int16()),
        "kills": pa.array([0] * n, pa.int16()),
        "deaths": pa.array(deaths, pa.int16()),
        "frags": pa.array([0] * n, pa.int16()),
        "pos_x": pa.array([0.0] * n, pa.float32()),
        "pos_y": pa.array([0.0] * n, pa.float32()),
        "angle": pa.array([0.0] * n, pa.float32()),
        "frame": pa.array(png, pa.binary()),
    })
    pq.write_table(table, path)
    fake_parquet.truth = dict(action=np.array(action), deaths=np.array(deaths), n=n)
    return fake_parquet.truth


def check_endtoend(vae, device, workdir, layout="v2"):
    import pyarrow.parquet as pq
    from transitions import canonical_table, valid_transitions

    pdir = os.path.join(workdir, "raw")
    odir = os.path.join(workdir, f"lat_{layout}")
    os.makedirs(pdir, exist_ok=True)
    os.makedirs(odir, exist_ok=True)
    ppath = os.path.join(pdir, "ep_00007.parquet")
    if not os.path.exists(ppath):
        fake_parquet(ppath)
    truth = fake_parquet.truth

    t = pq.read_table(ppath)
    btn = np.array(t["buttons"].to_pylist())
    canonical = canonical_table(truth["action"], btn)
    sources, chains = valid_transitions(truth["action"], btn, truth["deaths"], repeat=4, canonical=canonical)
    print(f"\n=== layout {layout} ===")
    print(f"[endtoend] {truth['n']} tics -> {len(sources)} decisions in {len(np.unique(chains))} chains")
    print(f"[endtoend] decision rows {sources.tolist()}")
    print(f"[endtoend] chain ids     {chains.tolist()}")
    assert len(np.unique(chains)) == 2, "fixture must produce exactly two chains"

    plan, dropped = chain_layout(sources, chains, truth["deaths"], layout=layout)
    n_chains_in = len(np.unique(chains))
    if layout == "v2":
        # every chain loses exactly its last decision, because s_{k-1}+4 is never a decision tic
        assert dropped == n_chains_in, (dropped, n_chains_in)
    else:
        assert dropped == 0, dropped
    print(f"[endtoend] dropped decisions: {dropped} of {len(sources)} "
          f"({100 * dropped / len(sources):.1f}%), {len(plan)} chains survive")

    decision_set = {int(c): set(sources[chains == c].tolist()) for c in np.unique(chains)}
    for c in plan:
        own = decision_set[c["chain_id"]]
        span = set(range(min(own), max(own) + 5))
        for j, rows in enumerate(chunk_rows(c)):
            if j == 0:
                assert len(rows) == 1 and rows[0] == c["lone_row"]
                continue
            s = int(c["decisions"][j - 1])
            want = list(range(s + 1, s + 5)) if layout == "v2" else list(range(s, s + 4))
            assert rows.tolist() == want, (c["chain_id"], j, rows.tolist(), want)
            assert set(rows.tolist()) <= span, "chunk leaves its chain"
            if layout == "v2":
                # the requested invariant: the chunk's last tic IS the next decision tic
                assert int(rows[-1]) == s + 4 and int(rows[-1]) in own, \
                    f"chunk {j} of chain {c['chain_id']} does not end on a decision tic"
                assert int(rows[-1]) == int(c["decision_row"][j - 1])
        if layout == "v2":
            nxt = [int(v) for v in c["decision_row"]]
            print(f"[endtoend] chain {c['chain_id']}: decisions {c['decisions'].tolist()} "
                  f"(dropped {c['dropped']}), lone row {c['lone_row']}, chunk ends {nxt} "
                  f"== next decision tics, {len(c['rows'])} frames = 1 + 4*{len(c['decisions'])}")
        else:
            print(f"[endtoend] chain {c['chain_id']}: decisions {c['decisions'].tolist()}, "
                  f"lone row {c['lone_row']} (duplicated={c['lone_duplicated']}), "
                  f"{len(c['rows'])} frames = 1 + 4*{len(c['decisions'])}")
        assert len(c["rows"]) == 1 + 4 * len(c["decisions"])

    enc = WanChainEncoder(vae, device, torch.float32)
    t0 = time.time()
    with ThreadPoolExecutor(4) as pool:
        r = encode_episode(ppath, odir, enc, canonical, pool, batch_chains=2, layout=layout)
    dt = time.time() - t0
    print(f"[endtoend] encode_episode: {r}")
    print(f"[endtoend] wall {dt:.1f}s for {r['tic_frames']} tic frames -> {r['tic_frames'] / dt:.2f} frames/s")

    lat = np.load(os.path.join(odir, "ep_00007.npy"))
    meta = json.load(open(os.path.join(odir, "ep_00007.meta.json")))
    assert lat.dtype == np.float16 and lat.shape[1:] == LATENT_SHAPE, (lat.dtype, lat.shape)
    assert meta["scheme"] == f"wan-chain-{layout}" and meta["z_dim"] == 16
    assert meta["dropped_decisions"] == dropped and meta["layout"] == layout
    assert len(meta["latents_mean"]) == 16 and len(meta["latents_std"]) == 16
    print(f"[endtoend] latents {lat.shape} {lat.dtype}; meta arrays "
          f"{sorted(k for k in meta if isinstance(meta[k], list) and len(meta[k]) == lat.shape[0])}")

    # latent frame j maps to decision j-1, per chain
    act = np.array(meta["action"]); t_start = np.array(meta["tic_start"])
    t_end = np.array(meta["tic_end"]); d_tic = np.array(meta["decision_tic"])
    cid = np.array(meta["chain_id"]); lone = np.array(meta["is_lone_first"])
    row = 0
    for c in sorted(plan, key=lambda c: len(c["decisions"]), reverse=True):
        k = len(c["decisions"])
        assert lone[row] == 1 and act[row] == -1 and cid[row] == c["chain_id"]
        assert t_start[row] == c["lone_row"]
        for j in range(1, k + 1):
            s = int(c["decisions"][j - 1])
            assert act[row + j] == truth["action"][s], (j, act[row + j], truth["action"][s])
            assert cid[row + j] == c["chain_id"] and lone[row + j] == 0
            if layout == "v2":
                assert t_start[row + j] == s + 1 and t_end[row + j] == s + 4
                assert d_tic[row + j] == s + 4 == t_end[row + j]
            else:
                assert t_start[row + j] == s and t_end[row + j] == s + 3 and d_tic[row + j] == s
        print(f"[endtoend] chain {c['chain_id']}: latent rows {row}..{row + k} verified "
              f"(frame j -> decision j-1, actions {act[row:row + k + 1].tolist()}, "
              f"tic spans {[f'{a}-{b}' for a, b in zip(t_start[row + 1:row + k + 1], t_end[row + 1:row + k + 1])]})")
        row += 1 + k
    assert row == lat.shape[0] == len(act)
    return odir


def check_loader(latents_dir):
    from wan_data import LatentWindowDataset, WanWindowDataset, chain_segments

    assert LatentWindowDataset is WanWindowDataset
    meta = json.load(open(os.path.join(latents_dir, "ep_00007.meta.json")))
    cid = np.array(meta["chain_id"])
    segs = chain_segments(cid)

    L = 2
    ds = WanWindowDataset(latents_dir, context_frames=L)
    expected = sum(max(0, b - (a + L + 1)) for a, b in segs)
    print(f"[loader] chains {segs}, L={L}: {len(ds)} windows (expected {expected})")
    assert len(ds) == expected

    ctx, tgt, act, ctx_act = ds[0]
    print(f"[loader] sample shapes ctx {tuple(ctx.shape)} tgt {tuple(tgt.shape)} "
          f"act {act.item()} ctx_act {ctx_act.tolist()}")
    assert ctx.shape == (L,) + LATENT_SHAPE and tgt.shape == LATENT_SHAPE
    assert ctx.dtype == torch.float32 and act.dtype == torch.long and ctx_act.shape == (L,)

    for i in range(len(ds)):
        ep, t = (int(v) for v in ds.index[i])
        seg = [(a, b) for a, b in segs if a <= t < b][0]
        assert t - L >= seg[0] + 1, "window reaches the lone frame or crosses a chain"
        _, _, a, ca = ds[i]
        assert a.item() != -1 and (ca != -1).all(), "a lone frame leaked into a sample"
    print(f"[loader] all {len(ds)} windows stay inside one chain and never touch the lone frame")

    ds_lone = WanWindowDataset(latents_dir, context_frames=L, allow_lone_context=True)
    print(f"[loader] allow_lone_context=True: {len(ds_lone)} windows "
          f"(+{len(ds_lone) - len(ds)}, one per chain)")
    assert len(ds_lone) == len(ds) + len(segs)

    raw = WanWindowDataset(latents_dir, context_frames=L, normalize=False, verbose=False)
    z_raw = raw[0][1].numpy()
    z_norm = ds[0][1].numpy()
    m = np.array(meta["latents_mean"], np.float32).reshape(16, 1, 1)
    s = np.array(meta["latents_std"], np.float32).reshape(16, 1, 1)
    assert np.allclose((z_raw - m) / s, z_norm, atol=1e-5)
    print(f"[loader] normalise check ok: raw mean {z_raw.mean():.3f} std {z_raw.std():.3f} -> "
          f"normalised mean {z_norm.mean():.3f} std {z_norm.std():.3f}")
    assert ds.scheme == meta["scheme"]

    # a split naming episodes that were not part of this pilot encode must skip, not crash
    subset = WanWindowDataset(latents_dir, episode_ids=["ep_00007", "ep_99998", "ep_99999"],
                              context_frames=L)
    assert len(subset) == len(ds) and subset.missing_episodes == ["ep_99998", "ep_99999"]
    print(f"[loader] subset directory: 3 ids requested, {len(subset.episodes)} encoded, "
          f"skipped {subset.missing_episodes}")


def main(args):
    device = torch.device("cpu")
    torch.set_grad_enabled(False)
    t0 = time.time()
    vae = load_wan_vae(args.vae_path, args.hf_cache, device, torch.float32)
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"[setup] AutoencoderKLWan loaded in {time.time() - t0:.1f}s, {n_params / 1e6:.1f}M params, "
          f"z_dim {vae.config.z_dim}, spatial {vae.config.scale_factor_spatial}, "
          f"temporal {vae.config.scale_factor_temporal}, tiling {vae.use_tiling}, slicing {vae.use_slicing}")

    z, frames, fps = check_shapes(vae, device)
    check_streaming(vae, device, z, frames)
    work = tempfile.mkdtemp(prefix="verify_wan_")
    try:
        for layout in ("v2", "v1"):
            latents_dir = check_endtoend(vae, device, work, layout=layout)
            check_loader(latents_dir)
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    print(f"\nALL CHECKS PASSED   peak RSS {peak_rss_mb():.0f} MB, "
          f"single-clip CPU encode {fps:.2f} frames/s, total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vae-path", default=None)
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--keep", action="store_true")
    main(p.parse_args())
