Updated [forward sheet](maps345_forward.png), [attack sheet](maps345_attack.png), and [selection JSON](maps345_selection.json). No commits.

Guard: either truth endpoint C* > its window median + 6, or an 8-connected blob with HSV saturation >=0.8 and value >=0.8 covering >8% of scene pixels. Medians use every exported truth frame, scene rows 0–207. Ranking remains U-Net PSNR first, then context luma; delta is reported, not used to rerank.

All three maps remain eligible. Surviving forward/attack counts: map 3 **16/487**, map 4 **342/21**, map 5 **73/448**. Scores are continuous-rollout proxies; copy-last compares context t with truth t+16. Held lists buttons active throughout all 16 steps; other controls may vary. C* is context/target; luma is 0–255.

**Forward**

| Window | t | Held | U-Net dB | Copy-last dB | Delta dB | Luma | C* t / t+16 |
|---|---:|---|---:|---:|---:|---:|---:|
| train_map04_ep6082_s4168 | 111 | forward + speed | 20.337 | 11.463 | +8.874 | 63.06 | 23.71 / 11.81 |
| train_map04_ep6082_s4168 | 112 | forward + speed | 20.117 | 10.775 | +9.342 | 69.16 | 25.88 / 12.02 |
| train_map05_ep6059_s4104 | 239 | forward + speed | 19.996 | 16.109 | +3.887 | 31.78 | 13.11 / 11.63 |
| train_map04_ep6082_s4168 | 110 | forward + speed | 19.987 | 12.431 | +7.556 | 58.35 | 21.87 / 11.72 |
| train_map04_ep6074_s3912 | 481 | forward + speed + turn right | 19.954 | 12.327 | +7.627 | 59.73 | 21.45 / 11.47 |

**Attack**

| Window | t | Held | U-Net dB | Copy-last dB | Delta dB | Luma | C* t / t+16 |
|---|---:|---|---:|---:|---:|---:|---:|
| train_map03_ep6037_s2824 | 52 | attack + speed | 21.867 | 15.720 | +6.147 | 46.16 | 12.82 / 10.79 |
| train_map03_ep6037_s2824 | 55 | attack + speed | 21.613 | 15.675 | +5.938 | 43.77 | 13.23 / 12.78 |
| train_map03_ep6037_s2824 | 53 | attack + speed | 21.444 | 16.442 | +5.002 | 45.53 | 12.75 / 11.44 |
| train_map03_ep6037_s2824 | 56 | attack + speed | 21.443 | 15.651 | +5.792 | 42.78 | 14.29 / 12.70 |
| train_map03_ep6037_s2824 | 57 | attack + speed | 21.234 | 16.433 | +4.801 | 46.05 | 15.90 / 12.05 |

My forward pick is **train_map05_ep6059_s4104, t=239** (+3.89 dB): the approach toward the wall/water boundary visibly advances, whereas the higher-scoring map-4 predictions depict the wrong room. My attack pick is **train_map03_ep6037_s2824, t=55** (+5.94 dB): firing at the nearby opponent is clear, but the U-Net omits that opponent, so a convincing hit still needs verification after the steward’s restart.
