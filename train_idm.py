"""
Inverse dynamics model (IDM) for the action-following metric (MineWorld-style).

A small conv net reads two consecutive decision-frame latents (8 channels at 32x40) and
predicts the action applied between them. Trained on real training-split windows, its
accuracy on real held-out windows is the ceiling of the metric; run on a world model's rollout
it measures whether the generated frames obey the conditioning action. Also reports a
movement-only collapse over the 29 Arnold actions using the button strings from
`buttons.json`, since attack-versus-no-attack is invisible in most frames.

    python train_idm.py --latents-dir data/latents_arnold --split data/split_arnold.json \
        --buttons data/raw_arnold/buttons.json --out results/idm --steps 6000
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from doom_data import list_latent_episodes, load_split


class PairDataset(Dataset):
    def __init__(self, latents_dir, episode_ids):
        keep = set(int(e) for e in episode_ids)
        self.eps, counts = [], []
        for ep, lat_path, meta_path in list_latent_episodes(latents_dir):
            if ep not in keep:
                continue
            lat = np.load(lat_path, mmap_mode="r")
            meta = np.load(meta_path); act = meta["action"].astype(np.int64)
            if lat.shape[0] < 2:
                continue
            if "chain_id" in meta.files:
                cid = meta["chain_id"]; starts = np.flatnonzero(cid[1:] == cid[:-1])
            else:
                starts = np.arange(lat.shape[0] - 1)
            if len(starts) == 0:
                continue
            self.eps.append((lat, act, starts)); counts.append(len(starts))
        self.offsets = np.concatenate([[0], np.cumsum(counts)])

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, i):
        slot = int(np.searchsorted(self.offsets, i, side="right") - 1)
        lat, act, starts = self.eps[slot]; s = int(starts[i - self.offsets[slot]])
        x = torch.from_numpy(np.asarray(lat[s:s + 2], dtype=np.float32)).reshape(8, 32, 40)
        return x, torch.tensor(int(act[s]), dtype=torch.long)


class IDM(nn.Module):
    def __init__(self, num_actions=29, width=128):
        super().__init__()
        c = width
        self.net = nn.Sequential(
            nn.Conv2d(8, c, 3, padding=1), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), nn.GroupNorm(8, 2 * c), nn.SiLU(),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.GroupNorm(8, 2 * c), nn.SiLU(),
            nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1), nn.GroupNorm(8, 4 * c), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * c, 4 * c), nn.SiLU(), nn.Linear(4 * c, num_actions))

    def forward(self, x):
        return self.net(x)


def movement_class(buttons_str, names):
    """Collapse a 0/1 button string to a coarse movement class id."""
    on = {n for n, b in zip(names, buttons_str) if b == "1"}
    fb = "F" if "MOVE_FORWARD" in on else ("B" if "MOVE_BACKWARD" in on else "")
    lr = "L" if "MOVE_LEFT" in on else ("R" if "MOVE_RIGHT" in on else "")
    turn = "l" if "TURN_LEFT" in on else ("r" if "TURN_RIGHT" in on else "")
    return fb + lr + turn or "still"


def build_action_to_movement(buttons_json, latents_dir, episode_ids, num_actions):
    """Map each action id to its movement class using observed (action, buttons) pairs."""
    names = json.load(open(buttons_json))["available_buttons"]
    from collections import Counter
    counts = {a: Counter() for a in range(num_actions)}
    keep = set(int(e) for e in episode_ids)
    for ep, _, meta_path in list_latent_episodes(latents_dir):
        if ep not in keep:
            continue
        m = np.load(meta_path)
        for a, b in zip(m["action"].tolist(), m["buttons"].tolist()):
            counts[int(a)][str(b)] += 1
    # the modal button string per action id is the nominal one; the anti-stuck override is a minority
    mapping = {a: movement_class(c.most_common(1)[0][0], names) for a, c in counts.items() if c}
    missing = [a for a in range(num_actions) if a not in mapping]
    assert not missing, f"actions never observed in the training split: {missing}"
    classes = sorted(set(mapping.values()))
    return {a: classes.index(c) for a, c in mapping.items()}, classes


@torch.no_grad()
def accuracy(model, loader, device, act2mov):
    model.eval(); top1 = n = mov = 0
    mov_t = torch.tensor([act2mov.get(a, -1) for a in range(model.net[-1].out_features)], device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        top1 += (p == y).sum().item(); n += y.numel()
        mov += (mov_t[p] == mov_t[y]).sum().item()
    model.train()
    return top1 / n, mov / n


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    split = load_split(args.split)
    train = PairDataset(args.latents_dir, split["train"]); val = PairDataset(args.latents_dir, split["val"])
    act2mov, classes = build_action_to_movement(args.buttons, args.latents_dir, split["train"], args.num_actions)
    print(f"train pairs {len(train):,}, val pairs {len(val):,}, movement classes {classes}")
    tl = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    vl = DataLoader(torch.utils.data.Subset(val, np.random.RandomState(0).choice(len(val), min(20000, len(val)), replace=False).tolist()),
                    batch_size=256, num_workers=2)
    model = IDM(args.num_actions, args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps)
    step, t0 = 0, time.time()
    while step < args.steps:
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
            if step % 200 == 0:
                print(f"step {step} loss {loss.item():.3f} ({step / (time.time() - t0):.1f} steps/s)", flush=True)
            if step % 1000 == 0 or step == args.steps:
                top1, mov = accuracy(model, vl, device, act2mov)
                print(f"  val top1 {top1:.4f} movement {mov:.4f}", flush=True)
            if step >= args.steps:
                break
    top1, mov = accuracy(model, vl, device, act2mov)
    torch.save({"model": model.state_dict(), "num_actions": args.num_actions, "width": args.width,
                "act2mov": act2mov, "classes": classes, "val_top1": top1, "val_movement": mov}, os.path.join(args.out, "idm.pt"))
    json.dump({"val_top1": top1, "val_movement": mov, "train_pairs": len(train), "val_pairs": len(val), "classes": classes},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
    print("DONE", json.dumps({"val_top1": top1, "val_movement": mov}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--latents-dir", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--buttons", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-actions", type=int, default=29)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=6000)
    main(p.parse_args())
