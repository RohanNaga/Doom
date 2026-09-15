#!/bin/bash
# Wan-VAE encodes in order: eval seen, eval unseen, pilot train+val subset. Usage: encode_chain.sh <gpu>
D=/sata2/data/rnagabhi/doom; cd $D/repo; export CUDA_VISIBLE_DEVICES=$1
PY=~/wanenc/bin/python; CAN=$D/latents_arnold_aligned/canonical_controls.json
$PY encode_wan.py --parquet-dir $D/raw_arnold_eval/seen --out-dir $D/latents_arnold_wan_eval/seen --hf-cache $D/hf/hub --canonical $CAN --layout v2 --device cuda:0 --batch-chains 1 --workers 8 --resume > $D/logs/encode_wan_eval_seen.log 2>&1; echo "EVAL_SEEN exit $?" >> $D/logs/encode_chain.log
$PY encode_wan.py --parquet-dir $D/raw_arnold_eval/unseen --out-dir $D/latents_arnold_wan_eval/unseen --hf-cache $D/hf/hub --canonical $CAN --layout v2 --device cuda:0 --batch-chains 1 --workers 8 --resume > $D/logs/encode_wan_eval_unseen.log 2>&1; echo "EVAL_UNSEEN exit $?" >> $D/logs/encode_chain.log
$PY encode_wan.py --parquet-dir $D/raw_arnold --out-dir $D/latents_arnold_wan --hf-cache $D/hf/hub --canonical $CAN --layout v2 --device cuda:0 --batch-chains 1 --workers 8 --episode-list $D/pilot_episodes.txt --resume > $D/logs/encode_wan_pilot.log 2>&1; echo "PILOT exit $?" >> $D/logs/encode_chain.log
echo ENCODE_CHAIN_DONE >> $D/logs/encode_chain.log
