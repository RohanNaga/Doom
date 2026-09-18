#!/bin/bash
# Evaluation suite for one corrected run, fired when its log reports the end event, then the next run on that GPU.
# Chain: dit -> PixArt-alpha third row (launch_pixart.sh); unet -> DiT seed 1 (launch_seed1.sh). Usage: after_run2.sh <run> <backbone> <gpu>
RUN=$1; BB=$2; GPU=$3; D=/sata2/data/rnagabhi/doom; R=$D/results_spiderman/$RUN; PY=${PY:-~/miniconda3/envs/doom/bin/python}
export TMPDIR=/sata2/data/rnagabhi/doom/tmp/tmpdir; mkdir -p $TMPDIR
until grep -q "\"event\": \"end\"" $R/log.jsonl 2>/dev/null; do sleep 300; done
until [ -f $D/latents_arnold_eval/split_unseen.json ]; do sleep 60; done
cd $D/repo && git pull -q
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1
COMMON="--backbone $BB --ckpt $R/best.pt --vae-path $D/vae_decoder_arnold_lpips/vae --hf-cache $D/hf/hub --context-frames 32 --num-actions 29"
for S in seen unseen; do
  $PY eval_tf.py $COMMON --latents-dir $D/latents_arnold_eval/$S --parquet-dir $D/raw_arnold_eval/$S --split $D/latents_arnold_eval/split_$S.json --subset val --num-windows 2048 --batch-size 16 --steps 50 --out-dir $R/eval_tf_$S > $D/logs/${RUN}_eval_tf_$S.log 2>&1
done
# EMA weights live only in the recovery checkpoints; score the final one on the seen corpus next to the live best.pt
LAST=$(ls $R/[0-9]*.pt 2>/dev/null | sort | tail -1)
[ -n "$LAST" ] && $PY eval_tf.py --backbone $BB --ckpt $LAST --use-ema --vae-path $D/vae_decoder_arnold_lpips/vae --hf-cache $D/hf/hub --context-frames 32 --num-actions 29 --latents-dir $D/latents_arnold_eval/seen --parquet-dir $D/raw_arnold_eval/seen --split $D/latents_arnold_eval/split_seen.json --subset val --num-windows 2048 --batch-size 16 --steps 50 --out-dir $R/eval_tf_seen_ema > $D/logs/${RUN}_eval_tf_seen_ema.log 2>&1
$PY rollout_eval.py --rollout $COMMON --latents-dir $D/latents_arnold_eval/seen --split $D/latents_arnold_eval/split_seen.json --subset val --num-rollouts 256 --horizon 64 --batch-size 16 --steps 50 --out $R/rollouts_seen.npz > $D/logs/${RUN}_rollout.log 2>&1
$PY rollout_eval.py --score --rollouts $R/rollouts_seen.npz --idm $D/results_spiderman/idm_aligned/idm.pt --vae-path $D/vae_decoder_arnold_lpips/vae --out-dir $R/rollout_metrics_seen --save-clips 256 >> $D/logs/${RUN}_rollout.log 2>&1
for F in 16 32; do $PY fvd.py --clips $R/rollout_metrics_seen/clips_u8.npz --frames $F --i3d $D/weights/i3d_torchscript.pt --out $R/rollout_metrics_seen/fvd$F.json >> $D/logs/${RUN}_rollout.log 2>&1; done
echo AFTER_RUN_DONE $RUN >> $D/logs/${RUN}_rollout.log
[ "$BB" = dit ] && bash $D/launch_pixart.sh $GPU >> $D/logs/resumes.log 2>&1
[ "$BB" = unet ] && bash $D/launch_seed1.sh $GPU >> $D/logs/resumes.log 2>&1
