# Next-tic rows (SD 3.5, PixArt): closed-loop collapse events per read

Rollouts: 16 x 256 tics per set, val maps 2-5; seed 0 = the standing windows, seed 1 = a second draw (different windows).
(a) any frame < 10 dB raw PSNR; (b) PSNR at tic 256 < 12 dB; (c) channel-13 per-frame mean < -0.9 at tic 256 (data range -0.44..0.80).

| row | step | weights | seed | n | (a) frame < 10 dB | (b) end < 12 dB | (c) ch13 end < -0.9 |
|---|---|---|---|---|---|---|---|
| sd35 | 50k | live | 1 | 16 | 10 (r01/m2, r02/m3, r03/m2, r04/m2, r05/m4, r07/m4, r08/m5, r10/m4, r12/m4, r13/m4) | 9 (r01/m2, r02/m3, r03/m2, r04/m2, r07/m4, r08/m5, r10/m4, r12/m4, r13/m4) | 1 (r07/m4) |
| sd35 | 55k | ema | 0 | 16 | 1 (r01/m2) | 0 | 0 |
| sd35 | 55k | live | 0 | 16 | 2 (r01/m2, r05/m4) | 1 (r05/m4) | 1 (r05/m4) |
| sd35 | 60k | ema | 0 | 16 | 1 (r01/m2) | 1 (r01/m2) | 1 (r01/m2) |
| sd35 | 60k | live | 0 | 16 | 0 | 0 | 0 |
| sd35 | 65k | ema | 0 | 16 | 0 | 0 | 0 |
| sd35 | 65k | live | 0 | 16 | 1 (r08/m5) | 1 (r08/m5) | 0 |
| sd35 | 70k | ema | 0 | 16 | 1 (r08/m5) | 1 (r08/m5) | 0 |
| sd35 | 70k | live | 0 | 16 | 12 (r00/m2, r01/m2, r02/m5, r03/m5, r04/m2, r05/m4, r06/m2, r07/m3, r08/m5, r11/m3, r12/m3, r15/m5) | 11 (r00/m2, r01/m2, r02/m5, r03/m5, r04/m2, r05/m4, r06/m2, r07/m3, r08/m5, r12/m3, r15/m5) | 10 (r00/m2, r02/m5, r03/m5, r04/m2, r05/m4, r06/m2, r07/m3, r08/m5, r12/m3, r15/m5) |
| sd35 | 70k | live | 1 | 16 | 15 (r01/m2, r02/m3, r03/m2, r04/m2, r05/m4, r06/m2, r07/m4, r08/m5, r09/m2, r10/m4, r11/m5, r12/m4, r13/m4, r14/m3, r15/m5) | 11 (r02/m3, r03/m2, r04/m2, r06/m2, r08/m5, r09/m2, r11/m5, r12/m4, r13/m4, r14/m3, r15/m5) | 9 (r02/m3, r04/m2, r08/m5, r09/m2, r11/m5, r12/m4, r13/m4, r14/m3, r15/m5) |
| sd35 | 75k | ema | 0 | 16 | 1 (r08/m5) | 0 | 0 |
| sd35 | 75k | live | 0 | 16 | 0 | 0 | 0 |
| sd35 | 80k | ema | 0 | 16 | 0 | 0 | 0 |
| sd35 | 80k | live | 0 | 16 | 1 (r02/m5) | 0 | 0 |
| sd35 | 85k | ema | 0 | 16 | 0 | 0 | 1 (r05/m4) |
| sd35 | 85k | live | 0 | 16 | 1 (r06/m2) | 0 | 1 (r01/m2) |
| sd35 | 90k | ema | 0 | 16 | 0 | 0 | 0 |
| sd35 | 90k | ema | 2 | 16 | 2 (r04/m5, r09/m5) | 1 (r04/m5) | 0 |
| sd35 | 90k | live | 0 | 16 | 0 | 0 | 0 |
| sd35 | 95k | ema | 0 | 16 | 1 (r15/m5) | 0 | 0 |
| sd35 | 95k | live | 0 | 16 | 1 (r02/m5) | 0 | 0 |
| sd35 | 100k | ema | 0 | 16 | 0 | 0 | 0 |
| sd35 | 100k | live | 0 | 16 | 1 (r15/m5) | 1 (r15/m5) | 1 (r15/m5) |
| sd35 | 105k | ema | 0 | 16 | 1 (r02/m5) | 0 | 2 (r02/m5, r05/m4) |
| sd35 | 105k | ema | 1 | 16 | 3 (r07/m4, r08/m5, r11/m5) | 1 (r08/m5) | 0 |
| sd35 | 105k | live | 0 | 16 | 0 | 0 | 0 |
| sd35 | 110k | ema | 0 | 16 | 0 | 0 | 0 |
| sd35 | 110k | live | 0 | 16 | 1 (r08/m5) | 0 | 0 |
| sd35 | 115k | ema | 0 | 16 | 1 (r15/m5) | 1 (r15/m5) | 0 |
| sd35 | 115k | ema | 1 | 16 | 3 (r05/m4, r11/m5, r15/m5) | 1 (r12/m4) | 0 |
| sd35 | 115k | live | 0 | 16 | 0 | 0 | 0 |
| sd35 | 120k | ema | 0 | 16 | 0 | 0 | 0 |
| sd35 | 120k | ema | 1 | 16 | 2 (r07/m4, r15/m5) | 0 | 0 |
| sd35 | 120k | live | 0 | 16 | 2 (r08/m5, r15/m5) | 2 (r02/m5, r15/m5) | 1 (r02/m5) |
| sd35 | 125k | ema | 0 | 16 | 1 (r15/m5) | 0 | 0 |
| sd35 | 125k | ema | 1 | 16 | 3 (r05/m4, r11/m5, r15/m5) | 1 (r15/m5) | 1 (r15/m5) |
| sd35 | 125k | live | 0 | 16 | 1 (r01/m2) | 1 (r01/m2) | 0 |
| sd35 | 130k | ema | 0 | 16 | 1 (r15/m5) | 1 (r05/m4) | 0 |
| sd35 | 130k | ema | 1 | 16 | 3 (r04/m2, r08/m5, r15/m5) | 1 (r04/m2) | 1 (r04/m2) |
| sd35 | 130k | live | 0 | 16 | 0 | 0 | 0 |

## Totals

- sd35 live: 288 rollouts; (a) 48/288 [50ks1, 55k, 65k, 70k, 70ks1, 80k, 85k, 95k, 100k, 110k, 120k, 125k]; (b) 37/288 [50ks1, 55k, 65k, 70k, 70ks1, 100k, 120k, 125k]; (c) 24/288 [50ks1, 55k, 70k, 70ks1, 85k, 100k, 120k]
- sd35 ema: 352 rollouts; (a) 25/352 [55k, 60k, 70k, 75k, 90ks2, 95k, 105k, 105ks1, 115k, 115ks1, 120ks1, 125ks1, 125k, 130k, 130ks1]; (b) 9/352 [60k, 70k, 90ks2, 105ks1, 115ks1, 115k, 125ks1, 130k, 130ks1]; (c) 6/352 [60k, 85k, 105k, 125ks1, 130ks1]
