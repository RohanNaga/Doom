# Trained Model Weights

The trained DiT-XL-2 checkpoints (~5.2 GB total) live in a [GitHub Release](https://github.com/RohanNaga/Doom/releases/tag/002-DiT-XL-2-best-90k) on this repo, not in the git tree (GitHub blocks >100 MB files in regular git).

## Quick start

```bash
bash download_weights.sh
```

This downloads + reassembles:
- `results/002-DiT-XL-2/checkpoints/best.pt` (2.6 GB) — best validation-loss checkpoint
- `results/002-DiT-XL-2/checkpoints/0090000.pt` (2.6 GB) — checkpoint at training step 90,000

## md5 checksums
- `best.pt`: `ff54cab2093665f00d997f3e5fd1dd27`
- `0090000.pt`: `3f1be8922d4ffdfd8cd698fe9dede922`

## Loading in PyTorch

```python
import torch
state = torch.load('results/002-DiT-XL-2/checkpoints/best.pt', map_location='cpu', weights_only=True)
```

## What's already in the repo

- All 91 sample directories (`results/002-DiT-XL-2/samples/step_*/`) — generated frames at each saved checkpoint
- `results/002-DiT-XL-2/log.txt` — full training log

## Why are weights not in the repo tree?

GitHub blocks pushes of files larger than 100 MB via regular git. Each .pt file is 2.6 GB, and even split chunks would exceed the limit. Options to put them inline would be:
- Git LFS (paid for >1 GB total)
- Many tiny chunks (~28 per checkpoint, ugly)

GitHub Releases is the standard ML practice: weights as release assets, code in repo, one script bridges them.
