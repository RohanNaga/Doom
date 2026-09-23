"""Do the windows the loader emits from the REAL corpus obey the causal contract? A launch gate.

Review 2026-09-22, section 7.1 item 3. The contract is tested on synthetic episodes elsewhere; this
runs it on sampled windows of the corpus that will actually train, through the training loader
(`doom_data.TicWindowDataset` with the launch's action history), and compares each emitted sample
with an independent read of the files:

  * the context is the stored latents of rows s .. r-1 and the target the latent of row r = s + L;
  * the control history is `control_matrix(buttons)` of rows s .. r-1, so its NEWEST row is r-1,
    the control applied from the last context frame into the target;
  * changing the control on row r (chosen after the target is observed) leaves the emitted sample
    unchanged, and changing row r-1 changes it (the positive control, so the test can fail).

    python check_emitted_windows.py --latents-dir $D/latents_arnold_dense_pertic/arenas \\
        --episodes 0:2000 --windows 256 --out $D/logs/gate1e_windows_sd15.json
"""
import argparse
import json

import numpy as np
import torch


def check(latents_dir, episode_ids, context_frames=32, latent_channels=None, windows=256, seed=0):
    """The report; `ok` is False on any violation."""
    from doom_data import TicWindowDataset, control_matrix, list_latent_episodes
    files = {ep: (lp, mp) for ep, lp, mp in list_latent_episodes(latents_dir)}
    if latent_channels is None:
        first = next(iter(sorted(files)))
        latent_channels = int(np.load(files[first][0], mmap_mode="r").shape[1])
    L = int(context_frames)
    ds = TicWindowDataset(latents_dir, episode_ids, L, latent_channels=latent_channels, action_history=L)
    idx = np.random.RandomState(seed).choice(len(ds), size=min(int(windows), len(ds)), replace=False)
    problems, checked, cache = [], 0, {}
    for gi in sorted(int(i) for i in idx):
        slot, start = ds.locate(gi)
        ep = ds.episodes[slot][0]
        r = start + L
        if ep not in cache:
            lat = np.load(files[ep][0], mmap_mode="r")
            with np.load(files[ep][1]) as m:
                cache = {ep: (lat, control_matrix(m["buttons"]), np.asarray(m["tic"]).astype(np.int64))}
        lat, ctrl, tic = cache[ep]
        ctx, tgt, act = ds[gi]
        where = f"episode {ep} window {start}..{r}"
        want_ctx = torch.from_numpy(np.asarray(lat[start:r], dtype=np.float32)).reshape(ctx.shape)
        if not torch.equal(ctx, want_ctx):
            problems.append(f"{where}: the context is not the stored latents of rows {start}..{r - 1}")
        if not torch.equal(tgt, torch.from_numpy(np.asarray(lat[r], dtype=np.float32))):
            problems.append(f"{where}: the target is not the stored latent of row {r}")
        if not np.array_equal(act.numpy(), ctrl[start:r].astype(act.numpy().dtype)):
            problems.append(f"{where}: the control history is not buttons of rows {start}..{r - 1}")
        elif not np.array_equal(act.numpy()[-1], ctrl[r - 1].astype(act.numpy().dtype)):
            problems.append(f"{where}: the newest control is not row r-1 = {r - 1}")
        if not np.all(np.diff(tic[start:r + 1]) == 1):
            problems.append(f"{where}: the window's tics are not consecutive")
        table = ds.episodes[slot][7]
        if r < len(table):
            saved = table[r].copy()
            table[r] = 1 - table[r]
            again = ds[gi]
            table[r] = saved
            if not (torch.equal(again[0], ctx) and torch.equal(again[1], tgt) and torch.equal(again[2], act)):
                problems.append(f"{where}: changing buttons[{r}] (chosen after the target) changed the sample")
        saved = table[r - 1].copy()
        table[r - 1] = 1 - table[r - 1]
        moved = ds[gi]
        table[r - 1] = saved
        if torch.equal(moved[2], act):
            problems.append(f"{where}: changing buttons[{r - 1}] did not change the control history, so the "
                            "invariance above proves nothing")
        checked += 1
    return {"latents_dir": latents_dir, "windows_checked": checked, "context_frames": L,
            "latent_channels": latent_channels, "problems": problems[:32], "violations": len(problems),
            "ok": checked > 0 and not problems}


def main(args):
    from doom_data import parse_episode_ids
    rep = check(args.latents_dir, parse_episode_ids(args.episodes), args.context_frames,
                args.latent_channels or None, args.windows, args.seed)
    text = json.dumps(rep, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)
    print(("EMITTED_WINDOWS_OK " if rep["ok"] else "EMITTED_WINDOWS_FAILED ")
          + f"{rep['windows_checked']} windows, {rep['violations']} violations")
    return 0 if rep["ok"] else 2


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--latents-dir", dest="latents_dir", required=True)
    p.add_argument("--episodes", required=True, help="the training ids, A:B or a comma list")
    p.add_argument("--windows", type=int, default=256)
    p.add_argument("--context-frames", dest="context_frames", type=int, default=32)
    p.add_argument("--latent-channels", dest="latent_channels", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    return p


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
