#!/bin/bash
# Evaluation suite for the SD 3.5 Medium row (035-sd35-l32-aligned), the same protocol after_run2.sh
# runs for the 4-channel rows, moved into the 16-channel latent space.
#
#   usage: [PY=..] [NOWAIT=1] [HF_HUB_OFFLINE=1] after_sd35.sh <gpu>
#
# It waits for the training log's end event (NOWAIT=1 skips the wait), then runs, on one card:
#
#   teacher-forced seen / unseen / unseen2, live weights and EMA          eval_tf.py
#   256 rollouts at horizon 64 on the seen corpus, then scoring           rollout_eval.py
#   FVD at 16 and 32 frames on the decoded clips                          fvd.py
#
# What differs from after_run2.sh, and why:
#
# * --latent-channels 16 and --sd35-path: the wrapper has to be rebuilt as it was trained before the
#   checkpoint loads.
# * --latent-scale 1.5305 --latent-shift 0.0609: SD 3.5's autoencoder stores latents as
#   (z - shift) * scale, not the SD 1.x bare 0.18215. build_vae checks these against the decoder's
#   own config, so a wrong pair fails loudly instead of decoding to noise.
# * the decoder is the LPIPS-tuned SD 3.5 decoder the VAE gate produced. If that directory has no
#   weights (a half-written save, which has happened once) the stock SD 3.5 decoder from the hub is
#   used instead and the choice is written to $R/decoder_used.txt, because the two give different
#   VAE ceilings and a table must not mix them.
# * --idm-reencode-vae: the IDM reads 4-channel SD latents (its encoder opens with a 4-channel
#   convolution), so the rollout is decoded with the SD 3.5 decoder and re-encoded with
#   sd-vae-ft-mse before it reaches the judge, the same round trip eval_video.py does for the Wan
#   row. CAVEAT, and it is a real one: both the rollout and its ground-truth counterpart make that
#   trip, so idm_real_* is this row's own reference and idm_top1 must be read against it. The IDM's
#   published validation accuracy was measured on native SD latents and is NOT comparable, and the
#   other rows' idm_top1 are not comparable to this one either until they are rescored the same way.
#
# HF_HUB_OFFLINE defaults to 0 because the stock-decoder fallback and the gated transformer repo may
# both still need the hub; set it to 1 once everything is resident in $D/hf/hub.
set -u
GPU=${1:?gpu}
D=/sata2/data/rnagabhi/doom
RUN=035-sd35-l32-aligned
R=$D/results_spiderman/$RUN
L=$D/latents_arnold_eval_sd35
PY=${PY:-$HOME/wanenc/bin/python}
SD35=${SD35:-stabilityai/stable-diffusion-3.5-medium}
IDM_ENC=${IDM_ENC:-stabilityai/sd-vae-ft-mse}
export TMPDIR=$D/tmp/tmpdir; mkdir -p $TMPDIR $D/logs
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

[ "${NOWAIT:-0}" = 1 ] || until grep -q "\"event\": \"end\"" $R/log.jsonl 2>/dev/null; do sleep 300; done
until [ -f $L/split_unseen2.json ]; do sleep 60; done
cd $D/repo && git pull -q

# the tuned decoder the VAE gate produced, or the stock one; whichever ran is written down next to the numbers
TUNED=$D/vae_decoder_sd35_lpips/vae
if ls $TUNED/diffusion_pytorch_model.safetensors $TUNED/diffusion_pytorch_model.bin >/dev/null 2>&1; then
  VAE="--vae-path $TUNED"; USED="tuned: $TUNED"
else
  VAE="--vae-path $SD35 --vae-subfolder vae"; USED="stock fallback: $SD35#vae (the tuned decoder at $TUNED has no weights)"
fi
mkdir -p $R && echo "$(date -Iseconds) decoder_used $USED" | tee -a $R/decoder_used.txt

SCALE="--latent-scale 1.5305 --latent-shift 0.0609"
COMMON="--backbone sd35 --latent-channels 16 --sd35-path $SD35 $VAE $SCALE --hf-cache $D/hf/hub --context-frames 32 --num-actions 29"

for S in seen unseen unseen2; do
  $PY eval_tf.py $COMMON --ckpt $R/best.pt --latents-dir $L/$S --parquet-dir $D/raw_arnold_eval/$S \
    --split $L/split_$S.json --subset val --num-windows 2048 --batch-size 16 --steps 50 \
    --out-dir $R/eval_tf_$S > $D/logs/${RUN}_eval_tf_$S.log 2>&1
  echo "$RUN eval_tf $S exit $?" >> $D/logs/${RUN}_eval.log
done

# EMA weights live only in the recovery checkpoints, never in best.pt; score the newest one on all three corpora
LAST=$(ls $R/[0-9]*.pt 2>/dev/null | sort | tail -1)
for S in seen unseen unseen2; do
  [ -n "$LAST" ] || break
  $PY eval_tf.py $COMMON --ckpt $LAST --use-ema --latents-dir $L/$S --parquet-dir $D/raw_arnold_eval/$S \
    --split $L/split_$S.json --subset val --num-windows 2048 --batch-size 16 --steps 50 \
    --out-dir $R/eval_tf_${S}_ema > $D/logs/${RUN}_eval_tf_${S}_ema.log 2>&1
  echo "$RUN eval_tf ${S}_ema exit $?" >> $D/logs/${RUN}_eval.log
done

$PY rollout_eval.py --rollout $COMMON --ckpt $R/best.pt --latents-dir $L/seen --split $L/split_seen.json \
  --subset val --num-rollouts 256 --horizon 64 --batch-size 16 --steps 50 \
  --out $R/rollouts_seen.npz > $D/logs/${RUN}_rollout.log 2>&1
$PY rollout_eval.py --score --rollouts $R/rollouts_seen.npz --idm $D/results_spiderman/idm_aligned/idm.pt \
  --idm-reencode-vae $IDM_ENC $VAE $SCALE --hf-cache $D/hf/hub \
  --out-dir $R/rollout_metrics_seen --save-clips 256 >> $D/logs/${RUN}_rollout.log 2>&1
for F in 16 32; do
  $PY fvd.py --clips $R/rollout_metrics_seen/clips_u8.npz --frames $F --i3d $D/weights/i3d_torchscript.pt \
    --out $R/rollout_metrics_seen/fvd$F.json >> $D/logs/${RUN}_rollout.log 2>&1
done
echo "AFTER_SD35_DONE $RUN ($USED)" >> $D/logs/${RUN}_rollout.log
