#!/bin/bash
# wait for the pip install, then build the cached null-prompt tensor on cpu
D=/sata2/data/rnagabhi/doom; cd $D/repo
until grep -q PIP_DONE $D/logs/wanpip.log 2>/dev/null; do sleep 20; done
mkdir -p $D/weights
HF_HUB_OFFLINE=0 ~/wanenc/bin/python make_null_prompt.py --out $D/weights/skyreels_null_prompt.pt --device cpu --dtype bfloat16 --hf-cache $D/hf/hub > $D/logs/null_prompt.log 2>&1
echo "NULL_PROMPT exit $?" >> $D/logs/null_prompt.log
