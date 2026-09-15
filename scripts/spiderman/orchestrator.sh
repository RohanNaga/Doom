#!/bin/bash
# Unattended steward for Spiderman, Sep 15 to Sep 17 noon EST. Every 10 minutes:
#  1. self-heal the main runs (resume_run.sh), the chained runs (launch_pixart.sh / launch_seed1.sh resume) and the video row;
#  2. keep the post-run waiters alive (after_run2.sh) and start waiters for pixart and seed 1 once their logs exist;
#  3. keep the Wan-VAE encode chain alive; start a second encoder on GPU 2 for the reversed pilot list once the fit checks are done;
#  4. launch the SkyReels row when its gates pass: encode chain done, a fit check that finished with peak_mem_gb <= 40 (throughput is
#     logged, not gated, because the fit checks share GPU 2 with the DiT; Claude judges the 36-hour budget on Wednesday) (L16 preferred, else L8), the null prompt present, and a GPU (3 preferred, then 2) with at least 30 GB free.
# Everything it does is appended to logs/orchestrator.log. State lives in state/.
D=/sata2/data/rnagabhi/doom; LOG=$D/logs/orchestrator.log; mkdir -p $D/state
say() { echo "$(date -Iseconds) $*" >> $LOG; }
alive() { tmux has-session -t "$1" 2>/dev/null; }
ended() { [ -f $D/results_spiderman/$1/log.jsonl ] && grep -q "\"event\": \"end\"" $D/results_spiderman/$1/log.jsonl; }
started() { [ -f $D/results_spiderman/$1/log.jsonl ]; }
evaldone() { grep -q AFTER_RUN_DONE $D/logs/$1_rollout.log 2>/dev/null; }
free_mb() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $1 | awk -F, '{print $2-$1}'; }
fit_ok() { # prints "L OBJ" of the best passing fit check, or nothing
  for cfg in "16 flow" "8 flow"; do set -- $cfg
    f=$D/logs/fit_video_$1_$2.log; [ -f $f ] || continue
    line=$(grep '"event": "fit_check"' $f | tail -1); [ -n "$line" ] || continue
    ok=$(echo "$line" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(int(d['peak_mem_gb']<=40 and d['steps_per_s']>0))")
    [ "$ok" = "1" ] && { echo "$1 $2"; return; }
  done
}
say "orchestrator started"
while true; do
  # 1. self-heal
  out=$(bash $D/resume_run.sh 030-dit-l32-aligned dit 2 2>&1); [[ "$out" != *alive* && "$out" != *finished* ]] && say "resume 030: $out"
  out=$(bash $D/resume_run.sh 031-unet-l32-aligned unet 3 2>&1); [[ "$out" != *alive* && "$out" != *finished* ]] && say "resume 031: $out"
  if started 033-pixart-l32-aligned && ! ended 033-pixart-l32-aligned && ! alive train-pixart; then say "pixart dead: $(tail -3 $D/logs/train_pixart.log | tr '\n' ' ' | cut -c1-300)"; say "$(bash $D/launch_pixart.sh 2)"; fi
  if started 032-dit-l32-aligned-seed1 && ! ended 032-dit-l32-aligned-seed1 && ! alive train-dit-seed1; then say "seed1 dead: $(tail -3 $D/logs/train_dit_seed1.log | tr '\n' ' ' | cut -c1-300)"; say "$(bash $D/launch_seed1.sh $(cat $D/state/seed1_gpu 2>/dev/null || echo 3))"; fi
  if [ -f $D/state/video ] && ! alive train-video; then read VG VL VO < $D/state/video; R=050-skyreels-l${VL}-${VO}; if started $R && ! ended $R; then say "video dead: $(tail -3 $D/logs/train_video.log | tr '\n' ' ' | cut -c1-300)"; say "$(bash $D/launch_video.sh $VG $VL $VO)"; fi; fi
  # 2. waiters
  for spec in "030-dit-l32-aligned dit 2 after-dit" "031-unet-l32-aligned unet 3 after-unet" "033-pixart-l32-aligned pixart 2 after-pixart" "032-dit-l32-aligned-seed1 dit 3 after-seed1"; do set -- $spec
    if started $1 && ! evaldone $1 && ! alive $4; then tmux new-session -d -s $4 "bash $D/after_run2.sh $1 $2 $3"; say "started waiter $4"; fi
  done
  # 3. encoders
  if ! grep -q ENCODE_CHAIN_DONE $D/logs/encode_chain.log 2>/dev/null && ! alive wan-encode; then tmux new-session -d -s wan-encode "bash $D/encode_chain.sh 1"; say "restarted encode chain on gpu 1"; fi
  if grep -q FIT_VIDEO_DONE $D/logs/fit_video.log 2>/dev/null && ! grep -q PILOT $D/logs/encode_chain.log 2>/dev/null && ! alive wan-encode-b && [ ! -f $D/state/encoder_b_done ]; then
    if [ $(free_mb 2) -gt 12000 ]; then tmux new-session -d -s wan-encode-b "cd $D/repo && CUDA_VISIBLE_DEVICES=2 ~/wanenc/bin/python encode_wan.py --parquet-dir $D/raw_arnold --out-dir $D/latents_arnold_wan --hf-cache $D/hf/hub --canonical $D/latents_arnold_aligned/canonical_controls.json --layout v2 --device cuda:0 --batch-chains 1 --workers 8 --episode-list $D/pilot_episodes_rev.txt --resume > $D/logs/encode_wan_pilot_b.log 2>&1; touch $D/state/encoder_b_done"; say "started encoder b on gpu 2"; fi
  fi
  # 4. video row gate
  if [ ! -f $D/state/video ] && grep -q ENCODE_CHAIN_DONE $D/logs/encode_chain.log 2>/dev/null && grep -q FIT_VIDEO_DONE $D/logs/fit_video.log 2>/dev/null && [ -f $D/weights/skyreels_null_prompt.pt ]; then
    cfg=$(fit_ok)
    if [ -n "$cfg" ]; then set -- $cfg; VL=$1; VO=$2; VG=""
      for g in 3 2; do [ $(free_mb $g) -gt 30000 ] && { VG=$g; break; }; done
      if [ -n "$VG" ]; then echo "$VG $VL $VO" > $D/state/video; say "video gate passed: L=$VL $VO on gpu $VG (free $(free_mb $VG) MB)"; say "$(bash $D/launch_video.sh $VG $VL $VO)"; else say "video gate passed (L=$VL $VO) but no gpu with 30 GB free; waiting"; fi
    else [ -f $D/state/video_gate_failed ] || { say "video gate failed: no fit check passed; leaving for manual review"; touch $D/state/video_gate_failed; }
    fi
  fi
  sleep 600
done
