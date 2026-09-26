# Astra's view: decoder adaptation and external baselines (2026-09-26)

Codex thread `01a0e011-92c2-7581-92ed-8ac183afa6be`. Model gpt-6-astra, reasoning effort high, sandbox workspace-write with network (used only for public weight checks), one round, no follow-up needed. Brief: `.claude/analyses/brief-decoder-and-baselines-2026-09-26.md`, passed verbatim.

## One-line answers

- **Q1:** (a) Freeze the training-map-tuned decoder throughout the primary LoRA adaptation arm.
- **Q2:** Keep external models in related work for Sep 30; defer a numerical, explicitly unmatched Doom reference comparison (Stiegler only) to October.

## Q1 reasoning

- The headline is a per-window ratio, 10 log10 [MSE(D(z_last), D(z_true)) / MSE(D(z_hat), D(z_true))]. Changing D changes this ratio even when every latent stays fixed. Passing both sides through the same decoder does not cancel D's nonlinear effects. A pixel-level endpoint therefore makes a fixed decoder more important.
- **(a) Frozen:** one common measurement across maps and updates. A threshold crossing measures better latent predictions rendered by one fixed decoder. Control response and latent error stay independently interpretable. "Cheap adaptation of the world model" describes the intervention accurately. The shared decoder tune is reported as an upfront cost.
- **(b) Joint:** a crossing then measures the predictor-plus-renderer package. Gains can come from either part, including a changed rendering of copy-last and of the target. Decoder training cannot improve the decoder-free control response or latent rollouts. The latent diffusion loss gives the decoder no gradient, so "joint" really means a second, separate reconstruction objective. "Cheap" would have to count decoder compute, parameters and storage.
- **(c) Separate rendering arm:** keeps attribution if the primary curve keeps its frozen decoder. Suggested design: a crossed 2x2 of {base, adapted predictor} x {shared, arena decoder} on the same saved latents, exposing rendering gains and their interaction. Train the arena decoder only on adaptation episodes; score held-out episodes, reconstruction fidelity and raw-frame LPIPS. Do not add the three decomposition legs numerically.
- **Challenge to the Sep 26 18:30 decision:** the 0.2 dB figure is not a gain bound. Similar stock ceilings show little map-specific degradation on average, not how much tuning could improve either group. The stock ceilings are full-frame and HUD-dominated, while the endpoint is now scene-only. Reconstruction of true latents does not bound decoder behaviour on erroneous predicted latents. Tonight's tuned scene-only ceilings are the missing evidence.
- **Cost and meaning:** about 4 A6000-hours for four arenas, 13 for thirteen, plus scoring and engineering. It would show whether target footage improves rendering with latent prediction unchanged, not that the model learned better transitions.
- **Precedents:** GameNGen (2408.14837, section 3.3) tunes the decoder only, on training footage. iVideoGPT (2405.15223, section 3.3, Figs. 8 and 9) adapts its tokenizer with a frozen transformer on BAIR and studies scarce-data transfer separately; its tokenizer includes the encoder, so it is not our decoder-only intervention. Astra found no verified few-shot decoder-only unseen-map precedent.

## Q2 reasoning

- **Stiegler is the only candidate plausibly scorable before Tuesday**, and only on a supported subset. The released 60k checkpoint has a U-Net, VAE and action embedding at 320x240 input. Its recorder captures every tic while repeating each action four times, and the dataset uses adjacent frames, so 4-tic action repeat does not force 4-tic evaluation.
- Its 18-action vocabulary lacks backward movement, use, weapon selection, no-op and attack-plus-movement combinations. Map executed controls exactly, drop windows whose history or target is unsupported, report coverage, and score our models on the same windows. Never silently substitute controls.
- Cost: 6 to 12 engineering hours plus 1 to 3 A6000-hours for a limited teacher-forced comparison with a native-domain sanity check (planning estimates, not measured).
- What it shows: transfer from Stiegler's distribution to Arnold on Freedoom arenas, confounded by maps, assets, behaviour, rendering and recipe. Claiming that it "loses its advantage" needs its own in-domain reference; failure on our arenas alone shows only OOD performance. October form: an appendix reference table with copy-last and reconstruction-ceiling rows.
- DIAMOND CS:GO (2405.12399): weights downloadable, but 280x150 output, different controls and game dynamics; Oasis is Minecraft and gated. Either costs 1 to 2 engineering days plus several GPU-hours for a cross-game stress test that cannot isolate unseen-map generalisation. PlayGen's Doom weights are unreleased; MultiGen has no verified download.

## Public models Astra checked

| Model | URL | Weights |
|---|---|---|
| Stiegler GameNGen reproduction | https://huggingface.co/arnaudstiegler/sd-model-gameNgen-60ksteps | yes |
| DIAMOND CS:GO | https://huggingface.co/eloialonso/diamond/tree/main/csgo/model | yes |
| Oasis 500M | https://huggingface.co/Etched/oasis-500m | yes, gated (401 unauthenticated) |
| PlayGen (Doom) | https://github.com/GreatX3/Playable-Game-Generation | no (Mario only) |
| MultiGen | https://ryanpo.com/multigen | VERIFY, no release linked |
| GameNGen (original) | https://gamengen.github.io/ | no public weights found |

Astra could not run inference, measure supported-window coverage, or open gated weights.

## Supervisor rechecks (Sep 26, via the Hugging Face API and GitHub raw)

- Stiegler: `api/models/arnaudstiegler/sd-model-gameNgen-60ksteps` reports `gated: False` and lists `unet/diffusion_pytorch_model.safetensors`, `vae/diffusion_pytorch_model.safetensors`, `action_embedding_model.safetensors`. `embedding_info.json` reads `{"num_embeddings": 18, "embedding_dim": 768}`, confirming the 18-action vocabulary. The U-Net config has `in_channels 40`, which suggests 4 latent channels x (9 context frames + 1 noisy target); this context length must be reproduced when scoring it (my inference, not checked against its code).
- DIAMOND: the tree API lists `csgo/model/csgo.pt`, 1,526,844,223 bytes. Confirmed.
- Oasis: API reports `gated: auto`. Confirmed gated.
- PlayGen: README checklist shows `[x] inference module and model weight of Mario` and `[ ] model weight of DOOM`. Confirmed no Doom weights.
- Not rechecked: the every-tic recorder claim for Stiegler, the MultiGen status, the iVideoGPT section and figure numbers, and the cost estimates.
- Codex wrote nothing to the repo (`git status --short` unchanged apart from this memo).
