# DoomShift weights release

Private Hugging Face model repo `RohanNaga/doomshift-weights` (https://huggingface.co/RohanNaga/doomshift-weights), uploaded from Spiderman on Sep 28, 2026 (22:28 to 22:35 EDT) with the stored login. Every file has a `<path>.sha256` sidecar in `sha256sum` format and the repo holds `MANIFEST.tsv` (repo path, bytes, sha256, source path, description) and a model card (`README.md`).

Not uploaded: the every-10k training snapshots and the coarse-grid SD 3.5 and PixArt-alpha adapters (their full-grid replacements follow once scored). The EMA milestones at 50k, 100k and 150k (`backbones/milestones/`) are being added by the same steward and will be appended here.

56 files, 44.59 GB.

| repo path | bytes | sha256 | source on Spiderman |
|---|---|---|---|
| `adapters/unet_r16/map01/adapter_0004000.pt` | 33,905,489 | `10f96caecdcef2ccf265a3e91ec1941b811fc9a58a0d695a901032dfb6b355e4` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map01_r16_k8_s0_g8k/adapter_0004000.pt` |
| `adapters/unet_r16/map01/adapter_0008000.pt` | 33,905,489 | `98764510b1ad8752942c81a9fa389a36cb8940654f487dc136e29ce808836c40` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map01_r16_k8_s0_g8k/adapter_0008000.pt` |
| `adapters/unet_r16/map01/config.json` | 5,492 | `705b49ff8af0ee24d96cd17649f70c92bed8d145759ba82421f1a68ca1a647a9` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map01_r16_k8_s0_g8k/config.json` |
| `adapters/unet_r16/map06/adapter_0004000.pt` | 33,905,489 | `e4090eb9b5c82a382490cddf4c3430e35314197328ecc14cd26717f83ecdf7cc` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map06_r16_k8_s0_g8k/adapter_0004000.pt` |
| `adapters/unet_r16/map06/adapter_0008000.pt` | 33,905,489 | `43949847ac10d279524750d86f026df5ba9174be5668e44d3680172fd3b40918` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map06_r16_k8_s0_g8k/adapter_0008000.pt` |
| `adapters/unet_r16/map06/config.json` | 5,492 | `9a61e2c7e0394a187a4715021de9fb389f044c683d55c20f8fdde89eecc7191c` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map06_r16_k8_s0_g8k/config.json` |
| `adapters/unet_r16/map07/adapter_0004000.pt` | 33,905,489 | `16ffcf29868a4fb9b9c029cac62501645735b467cf1a2675591728971efbd3ee` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map07_r16_k8_s0_g8k/adapter_0004000.pt` |
| `adapters/unet_r16/map07/adapter_0008000.pt` | 33,905,489 | `30820f432615f8b5b63b4183d135e6d8782d240aeb6f8aed79bf99517fb5c167` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map07_r16_k8_s0_g8k/adapter_0008000.pt` |
| `adapters/unet_r16/map07/config.json` | 5,492 | `269b5265815b52199e924b452591302520d5db4665a942548a971b7ef35a9a6e` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map07_r16_k8_s0_g8k/config.json` |
| `adapters/unet_r16/map08/adapter_0004000.pt` | 33,905,489 | `e2a530f51afe2b0daf3d1e5695f1d1382f53230446e06d83ace270851274b4ed` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map08_r16_k8_s0_g8k/adapter_0004000.pt` |
| `adapters/unet_r16/map08/adapter_0008000.pt` | 33,905,489 | `28a8fe2e4814d8618adedd6886df99b23e3d1c209c8689f63aac361449fbdfca` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map08_r16_k8_s0_g8k/adapter_0008000.pt` |
| `adapters/unet_r16/map08/config.json` | 5,491 | `808498ca3e07db2c43434dd9b4f191abbdfa1f69177f25955f8df4f27edf3f38` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map08_r16_k8_s0_g8k/config.json` |
| `adapters/unet_r16/map09/adapter_0004000.pt` | 33,905,489 | `c181a18e17fc886032ef6c1e8b97bb8967e676432d96baa74e54000a477d1636` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map09_r16_k8_s0_g8k/adapter_0004000.pt` |
| `adapters/unet_r16/map09/adapter_0008000.pt` | 33,905,489 | `8af0e070a3c70515786d7a17ba9414f8f7a4e395a4987d55be510afc9d3de0d4` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map09_r16_k8_s0_g8k/adapter_0008000.pt` |
| `adapters/unet_r16/map09/config.json` | 5,491 | `b952892991ef94323aad3a200d7e695c61d2011f468ccff486760ade7533c226` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map09_r16_k8_s0_g8k/config.json` |
| `adapters/unet_r16/map10/adapter_0004000.pt` | 33,905,489 | `f1bd732d72f195a3550aa87884df886c22f1c75edcffbd22db5fc50326fbb298` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map10_r16_k8_s0_g8k/adapter_0004000.pt` |
| `adapters/unet_r16/map10/adapter_0008000.pt` | 33,905,489 | `e2bc1f990f8014f82c427f049ced654b9b16e648ff25307546cc79d09cffc80c` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map10_r16_k8_s0_g8k/adapter_0008000.pt` |
| `adapters/unet_r16/map10/config.json` | 5,492 | `290aee8b8ae85473b96fecc3e533917a92cb0a702265affef806fd2ce47e8784` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map10_r16_k8_s0_g8k/config.json` |
| `adapters/unet_r16/map11/adapter_0004000.pt` | 33,905,489 | `b679203677ef4a3b4b2204e822572a60c39ff54c7c342b3d4c185fa7b9c7407e` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map11_r16_k8_s0_g8k/adapter_0004000.pt` |
| `adapters/unet_r16/map11/adapter_0008000.pt` | 33,905,489 | `35e413708003878179cf6d1c869616200439e0ba4d22cb92032165421211ab2a` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map11_r16_k8_s0_g8k/adapter_0008000.pt` |
| `adapters/unet_r16/map11/config.json` | 5,492 | `b3ea7b9ef29d5e5c4f4b3b2650adbecce1bdb5ea78b12f511e86c369635ab97a` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map11_r16_k8_s0_g8k/config.json` |
| `adapters/unet_r16/map12/adapter_0004000.pt` | 33,905,489 | `788485bd52204d2786eda001f3e72cd9e2b990617c6e21b329dda11f480126cf` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map12_r16_k8_s0_g8k/adapter_0004000.pt` |
| `adapters/unet_r16/map12/adapter_0008000.pt` | 33,905,489 | `8e1c1a591c647a89f278a148dd73202c36ebba024b93bcb1d42a92ae37159380` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map12_r16_k8_s0_g8k/adapter_0008000.pt` |
| `adapters/unet_r16/map12/config.json` | 5,492 | `304022a6bb9128472aa5fe2f5a9a990acacec95ac06adf415964b903d7615ae7` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map12_r16_k8_s0_g8k/config.json` |
| `adapters/unet_r16/map13/adapter_0004000.pt` | 33,905,489 | `d5100de66c7ad232b6c78a4692d7e37d4ff5d1be349847deb3337948d2f3eb83` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map13_r16_k8_s0_g8k/adapter_0004000.pt` |
| `adapters/unet_r16/map13/adapter_0008000.pt` | 33,905,489 | `e691c0a7e518bfeaeb62b9abb7f1592c2ae84f3aaa9291764532218aff2721e8` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map13_r16_k8_s0_g8k/adapter_0008000.pt` |
| `adapters/unet_r16/map13/config.json` | 5,492 | `d75ef86a036d4d2207953ff35a37d469387d37d4397e9009208c7706b65a11ba` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map13_r16_k8_s0_g8k/config.json` |
| `adapters/unet_r16/map14/adapter_0004000.pt` | 33,905,489 | `89068244f51f6fe832ab32a48d40975aa7d8dec223765a22b25aa21c8c75e5dc` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map14_r16_k8_s0_g8k/adapter_0004000.pt` |
| `adapters/unet_r16/map14/adapter_0008000.pt` | 33,905,489 | `81bdc5acf7454fffc721d59d9bad14f9594fc06dbaea2263325c98b7b2c7c6c5` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map14_r16_k8_s0_g8k/adapter_0008000.pt` |
| `adapters/unet_r16/map14/config.json` | 5,493 | `6dc5ab934b1e39e16421c88ed4ab1a24fbb06c45812cd58cc32994063671389f` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map14_r16_k8_s0_g8k/config.json` |
| `adapters/unet_r16/map15/adapter_0004000.pt` | 33,905,489 | `7cf302ed9acee198a1aff415f616236c3fcaab9e73a1d4ca19aff35912d7a30f` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map15_r16_k8_s0_g8k/adapter_0004000.pt` |
| `adapters/unet_r16/map15/adapter_0008000.pt` | 33,905,489 | `c81dd096c6ad417cd6b1aacc29e020ae1217ce63c51081269547444e23e82ebe` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map15_r16_k8_s0_g8k/adapter_0008000.pt` |
| `adapters/unet_r16/map15/config.json` | 5,493 | `31c28de9924ff2245bc8b181171eb415ea69a39f7e88ecd0a9da47a57b559403` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map15_r16_k8_s0_g8k/config.json` |
| `adapters/unet_r16/map16/adapter_0004000.pt` | 33,905,489 | `2c4bba31d77dc3cc483f57b9e35a527698c2b4e655ab2310b99954a190b0dbf7` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map16_r16_k8_s0_g8k/adapter_0004000.pt` |
| `adapters/unet_r16/map16/adapter_0008000.pt` | 33,905,489 | `6c3d98b4193c03bbea0eb00f253669f384d895bbcb74ec17e0346cdc1a4ae795` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map16_r16_k8_s0_g8k/adapter_0008000.pt` |
| `adapters/unet_r16/map16/config.json` | 5,493 | `c4c221f199f83b2377b2ab227564df42a2a51bf03901ee42399d4df889da099e` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map16_r16_k8_s0_g8k/config.json` |
| `adapters/unet_r16/map17/adapter_0004000.pt` | 33,905,489 | `0f3d0519cd102bdc5f63e3308e0da4723a42aed9064592739fa854d6999b1501` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map17_r16_k8_s0_g8k/adapter_0004000.pt` |
| `adapters/unet_r16/map17/adapter_0008000.pt` | 33,905,489 | `2fee82c90676878f5efc5f67ef587fed49d3d41789fa9397b11bd2685355b223` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map17_r16_k8_s0_g8k/adapter_0008000.pt` |
| `adapters/unet_r16/map17/config.json` | 5,493 | `0df7ad8da28381e0741318d35a2fec69377e6f8cd53a84128456c9336d3ea39d` | `$D/tmp/steward/adapters_g8k/unet200k_arenas13_map17_r16_k8_s0_g8k/config.json` |
| `backbones/pixart_alpha_200k.pt` | 2,514,339,349 | `46b1139b50ea5860fe03563fe9aad1b5b3fe802bfb518ffcb7c71f4b6d704b0a` | `$D/results_spiderman/041-pixart-nexttic/snap_0200000.pt` |
| `backbones/sd35_medium_200k.pt` | 9,540,599,156 | `b7b99ee601f3c2c7c90cf65345c416d69fc30dc6a979b34775cc53051774fcc1` | `$D/results_spiderman/042-sd35-nexttic/snap_0200000.pt` |
| `backbones/unet_sd14_200k.pt` | 3,442,605,807 | `0728fb73086c00da7bb2b40eb843fbde7836b0dfae8e38adbc9b7300ef8fd9cb` | `$D/results_spiderman/040-unet-nexttic/snap_0200000.pt` |
| `decoders/sd1_mse_lpips/config.json` | 812 | `b894a72b7a79dda82748929a9a447d3e1ddc6e17d209939c90718b48ea3cf1ca` | `$D/vae_decoder_sd1x_mse_lpips/vae/config.json` |
| `decoders/sd1_mse_lpips/diffusion_pytorch_model.safetensors` | 334,643,268 | `21b26edf65b301e0685b3b24ddbe29e379a6c2f54ac60b9c516e8f635b38f646` | `$D/vae_decoder_sd1x_mse_lpips/vae/diffusion_pytorch_model.safetensors` |
| `decoders/sd1_mse_lpips/provenance.json` | 85,385 | `50716cb52d9619028c776942970ce5d9059c3e3bb578f40431c34aeb45a4f9d7` | `$D/vae_decoder_sd1x_mse_lpips/vae/provenance.json` |
| `decoders/sd35_mse_lpips/config.json` | 831 | `7dac58f47f99a026b6d49db9b2310be548a90f2f16614c9019be1764346bc65b` | `$D/vae_decoder_sd35_mse_lpips/vae/config.json` |
| `decoders/sd35_mse_lpips/diffusion_pytorch_model.safetensors` | 335,306,212 | `a8d22f2af42b10dd322ce0fa823403cedbe350b6375150d2e9a22a600add1d05` | `$D/vae_decoder_sd35_mse_lpips/vae/diffusion_pytorch_model.safetensors` |
| `decoders/sd35_mse_lpips/provenance.json` | 85,399 | `2502deee88630032d6b0f46497674b8c71fa53102fbf3a0efccc2c76517e9f38` | `$D/vae_decoder_sd35_mse_lpips/vae/provenance.json` |
| `fullft/unet_map06_config.json` | 5,376 | `74f45095be7309a47a9b93a332c122466691641ebf7addb172280ca5a07b5bd6` | `$D/results_spiderman/adapt_fullft/map06/config.json` |
| `fullft/unet_map06_step4000.pt` | 6,884,734,951 | `8f4e7041f1a9e56fc156c4747cb3ba4faaff497d85d2f399b51cc0d0c75883be` | `$D/results_spiderman/adapt_fullft/map06/adapter_0004000.pt` |
| `fullft/unet_map07_config.json` | 5,376 | `e868bb20f4e37f1bd35ef0ddedd7eed2db690a8902c74eafe569dd2c80abddc0` | `$D/results_spiderman/adapt_fullft/map07/config.json` |
| `fullft/unet_map07_step4000.pt` | 6,884,734,951 | `821b696e0ce4bf7a31c27618ac61ef1f0b2fcb3b76f8183c918904f2fdfcfb09` | `$D/results_spiderman/adapt_fullft/map07/adapter_0004000.pt` |
| `fullft/unet_map08_config.json` | 5,375 | `5462386ad68e934afb442a2958926ae4d59a1b00c06fc5a9001aaecbc827f977` | `$D/results_spiderman/adapt_fullft/map08/config.json` |
| `fullft/unet_map08_step4000.pt` | 6,884,734,951 | `a8b18af66292c0dfdceff224b90b73f9004317f11fceb3dec305e28e86af9983` | `$D/results_spiderman/adapt_fullft/map08/adapter_0004000.pt` |
| `fullft/unet_map16_config.json` | 5,377 | `82a43d57c9ed367aa5bb95c61c22b18f9ab2b1fba6d5047d1ee58f7354fbc60d` | `$D/results_spiderman/adapt_fullft/map16/config.json` |
| `fullft/unet_map16_step4000.pt` | 6,884,734,951 | `06fa1aef24819fe1b9914d04d5184abba1a957677b983840838d15f733c3a8b7` | `$D/results_spiderman/adapt_fullft/map16/adapter_0004000.pt` |

## What the files are

- `backbones/*_200k.pt`: `torch.load` dicts with `model` (live weights, bf16), `ema` (EMA weights, bf16 at save), `step` 200000, `val_loss`, `args`, `episodes`, `corpus`; no optimizer state. `unet_sd14` is 040-unet-nexttic, `pixart_alpha` is 041-pixart-nexttic, `sd35_medium` is 042-sd35-nexttic.
- `decoders/sd1_mse_lpips`, `decoders/sd35_mse_lpips`: diffusers AutoencoderKL directories whose decoder was fine-tuned on Doom frames with MSE + 0.1 LPIPS (`vae_decoder_sd1x_mse_lpips/vae`, `vae_decoder_sd35_mse_lpips/vae`, the tuned decoders of the paper).
- `adapters/unet_r16/map<NN>/`: rank-16 LoRA adapters of the U-Net 200k EMA on 8 episodes of each unseen arena, seed 0, from the 8k-grid runs (`unet200k_arenas13_map<NN>_r16_k8_s0_g8k`), steps 4000 and 8000, with the run config.
- `fullft/unet_map<NN>_step4000.pt`: U-Net full fine-tunes (all 860.5M parameters, lr 2e-5) on arenas 6, 7, 8 and 16 at step 4000.
