# Tuning detail: hot signatures and selected kernels

Generated from the 2026-08-05 production tune (RV19) joined against the
200k-context record (RV16). 121 signatures, 416940 matmul calls.

`gen/elig/meas` = candidates generated / eligible / actually measured.
A large gap between eligible and measured means screening discarded them early.

## Top 15 signatures by call count

### `4dc5d20446ffea2f` — 110,160 calls (26.4% of all)

- src0 `[5120, 512, 1, 1]` · src1 `[5120, 512, 1, 1]` · dst `[512, 512, 1, 1]`
- upstream picks: `mmq:native:v1`
- tuned winner: `mmq:q8_0:j48:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` (**+44.35%**), gen/elig/meas 270/11/9

| rank | candidate | median us | vs best |
| --- | --- | --- | --- |
| 1 | `mmq:q8_0:j48:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 71.28 | +0.0% |
| 2 | `mmq:q8_0:j64:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 90.24 | +26.6% |
| 3 | `mmq:q8_0:j32:fb0:t128:o2:i64:sram-q8_0:k256:sk0:v1` | 90.64 | +27.2% |
| 4 | `mmq:q8_0:j16:fb0:t128:o2:i64:sram-q8_0:k256:sk0:v1` | 92.24 | +29.4% |
| 5 | `mmq:q8_0:j80:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 99.12 | +39.1% |
| 6 | `mmq:q8_0:j96:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 109.36 | +53.4% |
| 7 | `mmq:q8_0:j112:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 123.32 | +73.0% |
| 8 | `mmq:native:v1` | 128.08 | +79.7% |

### `035282e415599068` — 61,800 calls (14.8% of all)

- src0 `[3072, 5120, 1, 1]` · src1 `[3072, 512, 1, 1]` · dst `[5120, 512, 1, 1]`
- upstream picks: `mmq:native:v1`
- tuned winner: `mmq:q8_0:j64:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` (**+2.68%**), gen/elig/meas 270/11/9

| rank | candidate | median us | vs best |
| --- | --- | --- | --- |
| 1 | `mmq:q8_0:j64:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 264.00 | +0.0% |
| 2 | `mmq:q8_0:j96:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 267.16 | +1.2% |
| 3 | `mmq:q8_0:j80:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 271.04 | +2.7% |
| 4 | `mmq:native:v1` | 271.28 | +2.8% |
| 5 | `mmq:q8_0:j128:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 271.76 | +2.9% |
| 6 | `mmq:q8_0:j112:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 291.68 | +10.5% |
| 7 | `mmq:q8_0:j48:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 292.88 | +10.9% |
| 8 | `mmq:q8_0:j32:fb0:t128:o2:i64:sram-q8_0:k256:sk0:v1` | 394.72 | +49.5% |

### `6608d41c90aa4da1` — 55,080 calls (13.2% of all)

- src0 `[5120, 6144, 1, 1]` · src1 `[5120, 512, 1, 1]` · dst `[6144, 512, 1, 1]`
- upstream picks: `mmq:native:v1`
- tuned winner: `mmq:native:v1` (**+0.00%**), gen/elig/meas 270/11/9

| rank | candidate | median us | vs best |
| --- | --- | --- | --- |
| 1 | `mmq:native:v1` | 465.80 | +0.0% |
| 2 | `mmq:q8_0:j64:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 501.28 | +7.6% |
| 3 | `mmq:q8_0:j112:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 511.56 | +9.8% |
| 4 | `mmq:q8_0:j128:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 531.40 | +14.1% |
| 5 | `mmq:q8_0:j96:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 552.24 | +18.6% |
| 6 | `mmq:q8_0:j48:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 617.61 | +32.6% |
| 7 | `mmq:q8_0:j80:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 691.21 | +48.4% |
| 8 | `mmq:q8_0:j32:fb0:t128:o2:i64:sram-q8_0:k256:sk0:v1` | 759.25 | +63.0% |

### `d4d829f52511ebc2` — 13,440 calls (3.2% of all)

- src0 `[5120, 24, 1, 1]` · src1 `[5120, 512, 1, 1]` · dst `[24, 512, 1, 1]`
- upstream picks: `mmq:native:v1`
- tuned winner: `mmq:q8_0:j16:fb1:t128:o2:i64:sram-q8_0:k256:sk0:v1` (**+69.76%**), gen/elig/meas 270/7/5

| rank | candidate | median us | vs best |
| --- | --- | --- | --- |
| 1 | `mmq:q8_0:j16:fb1:t128:o2:i64:sram-q8_0:k256:sk0:v1` | 38.80 | +0.0% |
| 2 | `mmq:q8_0:j64:fb1:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 81.72 | +110.6% |
| 3 | `mmq:q8_0:j32:fb1:t128:o2:i64:sram-q8_0:k256:sk0:v1` | 97.64 | +151.7% |
| 4 | `mmq:native:v1` | 128.32 | +230.7% |
| 5 | `mmq:q8_0:j128:fb1:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 129.32 | +233.3% |

### `20a1ffdbaedfc690` — 11,960 calls (2.9% of all)

- src0 `[5120, 8704, 1, 1]` · src1 `[5120, 512, 1, 1]` · dst `[8704, 512, 1, 1]`
- upstream picks: `mmq:native:v1`
- tuned winner: `mmq:native:v1` (**+0.00%**), gen/elig/meas 270/11/9

| rank | candidate | median us | vs best |
| --- | --- | --- | --- |
| 1 | `mmq:native:v1` | 685.85 | +0.0% |
| 2 | `mmq:q8_0:j128:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 685.97 | +0.0% |
| 3 | `mmq:q8_0:j64:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 745.13 | +8.6% |
| 4 | `mmq:q8_0:j80:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 749.97 | +9.3% |
| 5 | `mmq:q8_0:j112:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 796.45 | +16.1% |
| 6 | `mmq:q8_0:j96:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 858.01 | +25.1% |
| 7 | `mmq:q8_0:j48:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 878.49 | +28.1% |
| 8 | `mmq:q8_0:j32:fb0:t128:o2:i64:sram-q8_0:k256:sk0:v1` | 1072.09 | +56.3% |

### `021751e78d1d49e0` — 11,212 calls (2.7% of all)

- src0 `[5120, 8704, 1, 1]` · src1 `[5120, 5, 1, 1]` · dst `[8704, 5, 1, 1]`
- upstream picks: `mmvq:native:v1`
- tuned winner: `mmvq:q8_0:w5:nw4:rpb4:sk1:v1` (**+17.41%**), gen/elig/meas 270/26/24

| rank | candidate | median us | vs best |
| --- | --- | --- | --- |
| 1 | `mmvq:q8_0:w5:nw4:rpb4:sk1:v1` | 42.32 | +0.0% |
| 2 | `mmvq:q8_0:w5:nw1:rpb2:sk0:v1` | 42.80 | +1.1% |
| 3 | `mmvq:q8_0:w5:nw2:rpb2:sk0:v1` | 43.20 | +2.1% |
| 4 | `mmvq:q8_0:w5:nw2:rpb2:sk1:v1` | 43.40 | +2.6% |
| 5 | `mmq:q8_0:j16:fb0:t128:o2:i64:sram-q8_0:k256:sk0:v1` | 44.16 | +4.3% |
| 6 | `mmvq:q8_0:w5:nw4:rpb2:sk0:v1` | 44.80 | +5.9% |
| 7 | `mmvq:q8_0:w5:nw8:rpb2:sk0:v1` | 47.12 | +11.3% |
| 8 | `mmvq:q8_0:w5:nw6:rpb2:sk0:v1` | 47.32 | +11.8% |

### `7bedae49d322d5d7` — 8,064 calls (1.9% of all)

- src0 `[5120, 24, 1, 1]` · src1 `[5120, 5, 1, 1]` · dst `[24, 5, 1, 1]`
- upstream picks: `mmvq:native:v1`
- tuned winner: `mmvq:q8_0:w5:nw8:rpb1:sk0:v1` (**+20.30%**), gen/elig/meas 270/22/20

| rank | candidate | median us | vs best |
| --- | --- | --- | --- |
| 1 | `mmvq:q8_0:w5:nw8:rpb1:sk0:v1` | 12.56 | +0.0% |
| 2 | `mmvq:q8_0:w5:nw6:rpb1:sk0:v1` | 12.80 | +1.9% |
| 3 | `mmvq:q8_0:w5:nw4:rpb1:sk0:v1` | 13.00 | +3.5% |
| 4 | `mmvq:q8_0:w5:nw8:rpb2:sk0:v1` | 13.20 | +5.1% |
| 5 | `mmvq:q8_0:w5:nw6:rpb2:sk0:v1` | 13.52 | +7.7% |
| 6 | `mmvq:q8_0:w5:nw4:rpb2:sk0:v1` | 13.72 | +9.2% |
| 7 | `mmvq:q8_0:w5:nw2:rpb1:sk0:v1` | 13.96 | +11.1% |
| 8 | `mmvq:q8_0:w5:nw2:rpb2:sk0:v1` | 14.36 | +14.3% |

### `e7c8abf6f5d4c97d` — 7,570 calls (1.8% of all)

- src0 `[3072, 5120, 1, 1]` · src1 `[3072, 5, 1, 1]` · dst `[5120, 5, 1, 1]`
- upstream picks: `mmvq:native:v1`
- tuned winner: `mmvq:q8_0:w5:nw2:rpb2:sk0:v1` (**+9.04%**), gen/elig/meas 270/26/24

| rank | candidate | median us | vs best |
| --- | --- | --- | --- |
| 1 | `mmvq:q8_0:w5:nw2:rpb2:sk1:v1` | 25.36 | +0.0% |
| 2 | `mmvq:q8_0:w5:nw2:rpb2:sk0:v1` | 25.44 | +0.3% |
| 3 | `mmvq:q8_0:w5:nw1:rpb2:sk0:v1` | 25.68 | +1.3% |
| 4 | `mmvq:q8_0:w5:nw4:rpb2:sk0:v1` | 25.88 | +2.1% |
| 5 | `mmvq:q8_0:w5:nw4:rpb4:sk1:v1` | 26.16 | +3.2% |
| 6 | `mmvq:q8_0:w5:nw6:rpb2:sk0:v1` | 26.72 | +5.4% |
| 7 | `mmq:q8_0:j16:fb0:t128:o2:i64:sram-q8_0:k256:sk0:v1` | 26.80 | +5.7% |
| 8 | `mmq:native:v1` | 27.36 | +7.9% |

### `daa773495d465dfd` — 7,280 calls (1.7% of all)

- src0 `[5120, 8704, 1, 1]` · src1 `[5120, 4, 1, 1]` · dst `[8704, 4, 1, 1]`
- upstream picks: `mmvq:native:v1`
- tuned winner: `mmvq:q8_0:w4:nw4:rpb4:sk1:v1` (**+17.78%**), gen/elig/meas 270/26/24

| rank | candidate | median us | vs best |
| --- | --- | --- | --- |
| 1 | `mmvq:q8_0:w4:nw4:rpb4:sk1:v1` | 37.92 | +0.0% |
| 2 | `mmvq:q8_0:w4:nw1:rpb2:sk0:v1` | 38.48 | +1.5% |
| 3 | `mmvq:q8_0:w4:nw2:rpb2:sk0:v1` | 38.52 | +1.6% |
| 4 | `mmvq:q8_0:w4:nw2:rpb2:sk1:v1` | 38.52 | +1.6% |
| 5 | `mmvq:q8_0:w4:nw6:rpb6:sk1:v1` | 39.76 | +4.9% |
| 6 | `mmvq:q8_0:w4:nw4:rpb2:sk0:v1` | 39.88 | +5.2% |
| 7 | `mmvq:q8_0:w4:nw6:rpb2:sk0:v1` | 40.80 | +7.6% |
| 8 | `mmvq:q8_0:w4:nw8:rpb2:sk0:v1` | 42.00 | +10.8% |

### `858c79fbfc355cd6` — 7,076 calls (1.7% of all)

- src0 `[5120, 512, 1, 1]` · src1 `[5120, 5, 1, 1]` · dst `[512, 5, 1, 1]`
- upstream picks: `mmvq:native:v1`
- tuned winner: `mmvq:q8_0:w5:nw8:rpb2:sk0:v1` (**+11.77%**), gen/elig/meas 270/26/24

| rank | candidate | median us | vs best |
| --- | --- | --- | --- |
| 1 | `mmvq:q8_0:w5:nw8:rpb2:sk0:v1` | 14.40 | +0.0% |
| 2 | `mmvq:q8_0:w5:nw4:rpb2:sk0:v1` | 14.52 | +0.8% |
| 3 | `mmvq:q8_0:w5:nw6:rpb2:sk0:v1` | 14.76 | +2.5% |
| 4 | `mmvq:q8_0:w5:nw2:rpb2:sk1:v1` | 14.80 | +2.8% |
| 5 | `mmvq:q8_0:w5:nw4:rpb1:sk0:v1` | 14.96 | +3.9% |
| 6 | `mmvq:q8_0:w5:nw2:rpb1:sk0:v1` | 15.00 | +4.2% |
| 7 | `mmvq:q8_0:w5:nw2:rpb2:sk0:v1` | 15.08 | +4.7% |
| 8 | `mmvq:q8_0:w5:nw8:rpb1:sk0:v1` | 15.24 | +5.8% |

### `157e7b98d6b14460` — 6,720 calls (1.6% of all)

- src0 `[5120, 5120, 1, 1]` · src1 `[5120, 512, 1, 1]` · dst `[5120, 512, 1, 1]`
- upstream picks: `mmq:native:v1`
- tuned winner: `mmq:q8_0:j64:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` (**+2.93%**), gen/elig/meas 270/11/9

| rank | candidate | median us | vs best |
| --- | --- | --- | --- |
| 1 | `mmq:q8_0:j64:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 432.84 | +0.0% |
| 2 | `mmq:q8_0:j96:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 440.36 | +1.7% |
| 3 | `mmq:q8_0:j80:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 445.64 | +3.0% |
| 4 | `mmq:native:v1` | 445.92 | +3.0% |
| 5 | `mmq:q8_0:j128:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 446.08 | +3.1% |
| 6 | `mmq:q8_0:j112:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 493.52 | +14.0% |
| 7 | `mmq:q8_0:j48:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 534.16 | +23.4% |
| 8 | `mmq:q8_0:j32:fb0:t128:o2:i64:sram-q8_0:k256:sk0:v1` | 636.92 | +47.1% |

### `469c6f4be346520f` — 6,720 calls (1.6% of all)

- src0 `[5120, 3072, 1, 1]` · src1 `[5120, 512, 1, 1]` · dst `[3072, 512, 1, 1]`
- upstream picks: `mmq:native:v1`
- tuned winner: `mmq:native:v1` (**+0.00%**), gen/elig/meas 270/11/9

| rank | candidate | median us | vs best |
| --- | --- | --- | --- |
| 1 | `mmq:native:v1` | 238.20 | +0.0% |
| 2 | `mmq:q8_0:j128:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 240.00 | +0.8% |
| 3 | `mmq:q8_0:j64:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 257.00 | +7.9% |
| 4 | `mmq:q8_0:j80:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 282.64 | +18.7% |
| 5 | `mmq:q8_0:j96:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 285.24 | +19.7% |
| 6 | `mmq:q8_0:j112:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 306.32 | +28.6% |
| 7 | `mmq:q8_0:j48:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 310.72 | +30.4% |
| 8 | `mmq:q8_0:j16:fb0:t128:o2:i64:sram-q8_0:k256:sk0:v1` | 417.92 | +75.4% |

### `615f6214f9597bef` — 5,980 calls (1.4% of all)

- src0 `[8704, 5120, 1, 1]` · src1 `[8704, 512, 1, 1]` · dst `[5120, 512, 1, 1]`
- upstream picks: `mmq:native:v1`
- tuned winner: `mmq:q8_0:j64:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` (**+3.33%**), gen/elig/meas 270/11/9

| rank | candidate | median us | vs best |
| --- | --- | --- | --- |
| 1 | `mmq:q8_0:j64:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 766.85 | +0.0% |
| 2 | `mmq:q8_0:j96:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 785.25 | +2.4% |
| 3 | `mmq:q8_0:j80:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 787.93 | +2.7% |
| 4 | `mmq:q8_0:j128:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 789.57 | +3.0% |
| 5 | `mmq:native:v1` | 793.25 | +3.4% |
| 6 | `mmq:q8_0:j48:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 859.53 | +12.1% |
| 7 | `mmq:q8_0:j112:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 880.77 | +14.9% |
| 8 | `mmq:q8_0:j32:fb0:t128:o2:i64:sram-q8_0:k256:sk0:v1` | 1052.97 | +37.3% |

### `286e24a5f6d96b13` — 5,606 calls (1.3% of all)

- src0 `[8704, 5120, 1, 1]` · src1 `[8704, 5, 1, 1]` · dst `[5120, 5, 1, 1]`
- upstream picks: `mmvq:native:v1`
- tuned winner: `mmvq:q8_0:w5:nw4:rpb4:sk1:v1` (**+17.48%**), gen/elig/meas 270/26/24

| rank | candidate | median us | vs best |
| --- | --- | --- | --- |
| 1 | `mmvq:q8_0:w5:nw4:rpb4:sk1:v1` | 41.92 | +0.0% |
| 2 | `mmvq:q8_0:w5:nw2:rpb2:sk0:v1` | 43.16 | +3.0% |
| 3 | `mmvq:q8_0:w5:nw6:rpb6:sk1:v1` | 43.20 | +3.1% |
| 4 | `mmvq:q8_0:w5:nw2:rpb2:sk1:v1` | 43.24 | +3.2% |
| 5 | `mmvq:q8_0:w5:nw8:rpb2:sk0:v1` | 44.44 | +6.0% |
| 6 | `mmvq:q8_0:w5:nw6:rpb2:sk0:v1` | 44.44 | +6.0% |
| 7 | `mmvq:q8_0:w5:nw1:rpb2:sk0:v1` | 45.00 | +7.3% |
| 8 | `mmvq:q8_0:w5:nw4:rpb2:sk0:v1` | 46.44 | +10.8% |

### `f099775dd9793993` — 5,376 calls (1.3% of all)

- src0 `[5120, 24, 1, 1]` · src1 `[5120, 4, 1, 1]` · dst `[24, 4, 1, 1]`
- upstream picks: `mmvq:native:v1`
- tuned winner: `mmvq:q8_0:w4:nw8:rpb1:sk0:v1` (**+20.16%**), gen/elig/meas 270/22/20

| rank | candidate | median us | vs best |
| --- | --- | --- | --- |
| 1 | `mmvq:q8_0:w4:nw8:rpb1:sk0:v1` | 12.36 | +0.0% |
| 2 | `mmvq:q8_0:w4:nw6:rpb1:sk0:v1` | 12.56 | +1.6% |
| 3 | `mmvq:q8_0:w4:nw4:rpb1:sk0:v1` | 12.72 | +2.9% |
| 4 | `mmvq:q8_0:w4:nw8:rpb2:sk0:v1` | 12.92 | +4.5% |
| 5 | `mmvq:q8_0:w4:nw6:rpb2:sk0:v1` | 13.12 | +6.1% |
| 6 | `mmvq:q8_0:w4:nw4:rpb2:sk0:v1` | 13.28 | +7.5% |
| 7 | `mmvq:q8_0:w4:nw2:rpb1:sk0:v1` | 13.56 | +9.7% |
| 8 | `mmvq:q8_0:w4:nw2:rpb2:sk1:v1` | 14.12 | +14.2% |

## Winners weighted by calls

| candidate | calls | share |
| --- | --- | --- |
| `mmq:q8_0:j48:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 111,384 | 26.7% |
| `mmq:q8_0:j64:fb0:t256:o2:i128:sram-q8_0:k256:sk0:v1` | 84,664 | 20.3% |
| `mmq:native:v1` | 84,040 | 20.2% |
| `mmq:q8_0:j16:fb1:t128:o2:i64:sram-q8_0:k256:sk0:v1` | 17,856 | 4.3% |
| `mmvq:q8_0:w5:nw4:rpb4:sk1:v1` | 17,092 | 4.1% |
| `blas:native:v1` | 14,678 | 3.5% |
| `mmvq:q8_0:w5:nw2:rpb2:sk0:v1` | 11,108 | 2.7% |
| `mmvq:q8_0:w4:nw4:rpb4:sk1:v1` | 10,024 | 2.4% |
| `mmvq:q8_0:w5:nw8:rpb1:sk0:v1` | 8,064 | 1.9% |
| `mmvq:q8_0:w5:nw8:rpb2:sk0:v1` | 7,160 | 1.7% |
| `mmvq:q8_0:w4:nw8:rpb1:sk0:v1` | 5,376 | 1.3% |
| `mmvq:q8_0:w4:nw2:rpb2:sk0:v1` | 4,592 | 1.1% |
| `mmvq:q8_0:w1:nw1:rpb2:sk0:v1` | 4,394 | 1.1% |
| `mmvq:q8_0:w5:nw4:rpb2:sk0:v1` | 4,032 | 1.0% |
| `mmvq:q8_0:w5:nw2:rpb2:sk1:v1` | 4,032 | 1.0% |
| `mmvq:q8_0:w4:nw6:rpb6:sk1:v1` | 3,640 | 0.9% |
| `mmvq:q8_0:w2:nw1:rpb2:sk0:v1` | 3,160 | 0.8% |
| `mmvq:q8_0:w4:nw4:rpb2:sk0:v1` | 2,688 | 0.6% |
| `mmvq:q8_0:w4:nw4:rpb1:sk0:v1` | 1,904 | 0.5% |
| `mmvq:q8_0:w2:nw2:rpb2:sk1:v1` | 1,816 | 0.4% |

## Boundary analysis — is the search space wide enough?

A winner sitting at the edge of an enumerated range suggests the range
should be extended; a winner in the interior suggests it is adequate.

**MMQ J values chosen** (table offers 16,32,48,64,80,96,112,128):

- J=16   18,196 calls
- J=48   111,384 calls
- J=64   84,664 calls

**MMVQ widths chosen** (enumerated 1..8):

- w=1   7,594 calls
- w=2   7,050 calls
- w=3   2,020 calls
- w=4   28,236 calls
- w=5   51,488 calls
- w=6   4 calls
- w=8   56 calls
