# HI17 `bd4c552` parity reduction

This is an offline reduction of the three existing Brutus tune artifacts. No
new hardware run was used.

Sources:

- `/tmp/hi17-parity-final-tune` (R1)
- `/tmp/hi17-parity-repeat-tune` (R2)
- `/tmp/hi17-parity-third-tune` (R3)

Each source contains the same four representatives on gfx1030, gfx1100, and
gfx1201. The reported native and forced-native rows each contain 100 final
samples, `status=ok`, `nmse=0`, and `max_abs=0`.

The declared tuner noise tolerance is 5% (`noise_canary_pct`). For each cell,
the parity gate is:

```text
abs(forced_median - native_median) / native_median
    <= max(5%, native_cross_run_median_spread)
```

Native cross-run spread is `(max(native medians) - min(native medians)) /
median(native medians)`. The table reports median/p95/MAD in microseconds.

| Cell | N1 median/p95/MAD | N2 median/p95/MAD | N3 median/p95/MAD | F1 median/p95/MAD | F2 median/p95/MAD | F3 median/p95/MAD | Native spread | F/N deltas | Envelope | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gfx1030-gemmex | 84.413/89.085/0.920 | 83.653/85.166/0.333 | 83.705/84.940/0.407 | 84.610/119.966/0.705 | 83.720/85.250/0.312 | 83.693/85.165/0.320 | 0.91% | 0.23%, 0.08%, 0.01% | 5.00% | PASS |
| gfx1030-pointer_batched | 138.996/152.300/7.086 | 138.900/152.831/5.615 | 140.300/152.040/4.981 | 138.086/152.030/5.786 | 138.376/150.481/5.205 | 137.370/152.831/6.410 | 1.01% | 0.65%, 0.38%, 2.09% | 5.00% | PASS |
| gfx1030-sgemm | 47.187/49.415/0.316 | 47.593/49.725/0.582 | 47.250/48.919/0.413 | 47.213/48.720/0.270 | 47.708/52.445/0.575 | 47.263/48.639/0.385 | 0.86% | 0.06%, 0.24%, 0.03% | 5.00% | PASS |
| gfx1030-strided | 68.990/70.635/1.038 | 68.948/70.710/1.150 | 68.898/70.755/1.160 | 69.091/70.646/1.118 | 68.940/70.316/1.120 | 68.783/70.425/0.835 | 0.13% | 0.15%, 0.01%, 0.17% | 5.00% | PASS |
| gfx1100-gemmex | 84.126/108.400/2.735 | 78.835/102.201/0.930 | 80.320/106.351/1.124 | 84.265/108.081/3.150 | 78.605/105.690/0.820 | 80.596/107.321/1.200 | 6.59% | 0.17%, 0.29%, 0.34% | 6.59% | PASS |
| gfx1100-pointer_batched | 122.761/128.621/4.660 | 122.956/128.591/3.985 | 122.341/129.481/4.145 | 121.906/128.401/3.875 | 122.341/130.781/3.810 | 121.560/129.141/3.890 | 0.50% | 0.70%, 0.50%, 0.64% | 5.00% | PASS |
| gfx1100-sgemm | 49.985/70.965/0.640 | 50.827/72.806/0.855 | 50.518/73.670/0.485 | 49.860/74.561/0.525 | 50.683/76.076/0.775 | 50.440/73.440/0.400 | 1.67% | 0.25%, 0.28%, 0.15% | 5.00% | PASS |
| gfx1100-strided | 62.760/116.721/8.320 | 61.001/98.841/5.359 | 60.560/96.561/5.000 | 64.541/117.321/9.420 | 58.601/95.881/4.640 | 59.161/94.720/4.360 | 3.61% | 2.84%, 3.93%, 2.31% | 5.00% | PASS |
| gfx1201-gemmex | 62.870/66.650/0.945 | 62.746/64.851/0.967 | 62.815/65.416/0.880 | 62.980/66.706/1.050 | 62.723/64.820/1.015 | 63.015/65.350/0.917 | 0.20% | 0.17%, 0.04%, 0.32% | 5.00% | PASS |
| gfx1201-pointer_batched | 12677.461/13230.403/122.416 | 12626.993/13197.427/146.135 | 12656.874/13052.440/135.120 | 12719.790/13230.164/164.426 | 12664.869/13126.127/154.886 | 12643.535/13085.532/126.351 | 0.40% | 0.33%, 0.30%, 0.11% | 5.00% | PASS |
| gfx1201-sgemm | 50.143/54.400/1.660 | 50.406/53.190/1.635 | 50.068/53.285/1.312 | 49.803/53.420/1.265 | 50.160/53.160/1.423 | 49.958/53.445/1.177 | 0.67% | 0.68%, 0.49%, 0.22% | 5.00% | PASS |
| gfx1201-strided | 60.981/72.050/0.813 | 61.020/66.290/0.797 | 61.276/66.465/0.553 | 61.335/66.895/0.565 | 61.408/66.770/0.757 | 61.085/66.275/0.890 | 0.48% | 0.58%, 0.64%, 0.31% | 5.00% | PASS |

All 12 cells pass. The earlier large-K non-parity artifact remains historical
diagnostic evidence; it is not the prescribed same-representative acceptance
matrix and does not block HI17 after this reduction.
