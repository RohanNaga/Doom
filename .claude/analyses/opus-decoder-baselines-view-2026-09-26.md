# Opus view: decoder in adaptation; public Doom baselines (2026-09-26)

**Q1.** (a) Freeze the decoder at the training-map-tuned weights for the primary arm. Run (c) only as a rendering check on two or three arenas, and only if tonight's tune opens a train-versus-unseen ceiling gap above about 0.3 dB. Never run (b).

**Q2.** Keep it to one related-work sentence for Sep 30. Two public Doom models (Stiegler, Taketani) have weights, run per tic at 320x240 and could be scored, but neither action set covers Arnold's controls. Score them in an October appendix, home map against our arenas.

## Q1

- `finetune_decoder.py` trains decode(encode(x)) against x with the encoder frozen. The decoder never sees predicted latents, so it cannot learn the backbone's errors.
- Headline A passes the prediction, copy-last and the true frame through one decoder, so the decoder is the ruler. The 3.60 dB line is frozen through the training-map-tuned decoder.

| | Cost (A) | Decomposition | "Cheap adaptation" |
|---|---|---|---|
| (a) frozen | Same ruler as the line at every step and map | Pure latent and control legs; rendering = the tuned ceiling per arena | Clean: LoRA updates are the whole cost |
| (b) joint | The ruler moves along the step axis and differs from the line's ruler | Legs confounded at every point | Weakened: about 1 h of uncounted decoder training on target pixels. With no shared loss, "joint" is two trainings reported as one number |
| (c) separate | Unchanged; the arm reports ΔA and ΔC by re-decoding the saved adapted latents | It is the rendering leg | Clean, if labelled as rendering |

**Cost of (c):** about 1 A6000-hour per arena, adapt episodes only. That is 13 h for all arenas, or 2 to 3 h for arena 7 plus one near arena. **What it shows:** whether rendering stays flat once the decoder has seen the map. The stock ceiling is flat (23.91 against 23.71 dB), so the expected ΔA is small. **The deciding unknown** is whether tonight's training-map-only tune raises the training maps' ceiling more than the arenas'. If it does, the gap comes from our own procedure, and (c) is its control.

**Precedents (verified).**
- GameNGen §3.2.2 (2408.14837) tunes only the decoder with MSE, "completely separately" from the U-Net, on its training data.
- iVideoGPT (2405.15223, Fig. 8) tunes only the tokenizer for unseen grippers and gets "similar perceptual quality" to a full fine-tune. Its tokenizer includes the encoder, so its latents change and ours would not.
- I found no few-shot decoder adaptation to a new map.

## Q2: verified (HF API file lists and READMEs read raw)

| Model | Weights | Map; frames | Actions | Scorable here? |
|---|---|---|---|---|
| Stiegler [repo](https://github.com/arnaudstiegler/gameNgen-repro), [100k](https://huggingface.co/arnaudstiegler/game-n-gen-sd-model-500-eps-100k) | yes, SD 1.4 plus tuned VAE | `deathmatch_simple`; per tic, 320x240, context 9 | 18 ids over 6 buttons; ATTACK exclusive; no BACKWARD, no no-op | Partly; opposite row convention. Training set (a16z Lrg or own 500 episodes) **VERIFY** |
| Taketani [repo](https://github.com/Masao-Taketani/GameNGen), [HF](https://huggingface.co/Masao-Taketani/vizdoom-diffusion-dynamic-model) | yes (Dec 2025), plus decoder | `deathmatch_simple`; per tic, padded 320x256, context 64 | 12 ids; ATTACK and TURN each exclusive | Partly; turn-while-moving unmappable |
| a16z [sd-model-gameNgen](https://huggingface.co/P-H-B-D-a16z/sd-model-gameNgen) | yes | ViZDoom deathmatch; **VERIFY** | "Action Dimension: 17" | Weak |
| GameNGen [page](https://gamengen.github.io/) | no | | | no |
| PlayGen [repo](https://github.com/GreatX3/Playable-Game-Generation) | Mario only; Doom is a TODO | Doom 128x128 | | no |
| MultiGen [2603.06679](https://arxiv.org/abs/2603.06679) | no link found, **VERIFY** | Doom plus minimap | | no |
| DIAMOND CS:GO [HF](https://huggingface.co/eloialonso/diamond) | yes | Dust II, 150x280 | 51 | no |
| Oasis [HF](https://huggingface.co/Etched/oasis-500m) | yes | Minecraft | VPT | no |
| [diamond-doom-hg](https://huggingface.co/Karajan42/diamond-doom-hg-v5.3), [lucrbrtv](https://huggingface.co/lucrbrtv/doom-world-model) | yes | Health Gathering plus minimap; id Doom footage | 5; inferred | no |

**What it would show.** It is an out-of-distribution test of one-map models, not a matched baseline.
- The honest read is each model's own home-versus-away drop in decoded advantage over copy-last, through its own decoder.
- Home is `raw_stiegler`: its card histogram has exactly 18 action ids, so Stiegler scores in-domain with native ids.
- Away is our 17 arenas, restricted to windows whose whole context history maps exactly.
- A drop toward copy-last would extend our finding beyond our recipe. That is worth a sentence, not the headline.

**Cost.**
- Adapter code (buffers, one-row shift, action mapping): 5 to 8 engineer-hours.
- Scoring: 1 to 2 A6000-hours per model, on cards the pilot and the tunes need Sunday and Monday.

**Recommendation.** For Sep 30, one sentence: open reproductions exist, each trains on one map, is evaluated on held-out trajectories of it, and reports no persistence reference. For October, an appendix with both models, home and away.
