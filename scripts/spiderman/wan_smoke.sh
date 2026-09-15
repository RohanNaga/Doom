#!/bin/bash
# one-episode Wan-VAE encode smoke on a given gpu. Usage: wan_smoke.sh <gpu>
D=/sata2/data/rnagabhi/doom; mkdir -p $D/tmp/wan_smoke
cd $D/repo && CUDA_VISIBLE_DEVICES=$1 ~/wanenc/bin/python encode_wan.py --parquet-dir $D/raw_arnold --out-dir $D/tmp/wan_smoke --hf-cache $D/hf/hub --canonical $D/latents_arnold_aligned/canonical_controls.json --layout v2 --device cuda:0 --batch-chains 1 --workers 8 --limit-episodes 1 --resume
echo SMOKE_EXIT $?
