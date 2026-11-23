# gfx1100 IQ8R group-256 scale-fold experiment

Date: 2026-11-23. Status: source-valid scratch experiment; **not promoted to the managed production rewrite**.

## Method

Compared the existing gfx1100 `gemm_iq8r_tiled` implementation with one scratch-only managed native rewrite candidate. The experiment keeps integer WMMA accumulation in native gfx11 register order through its group-256 scale fold, gathers corresponding even/odd per-output-row scales, and converts floating output to the existing gfx12 store order only once. It applies only to `I8 && RB == 0 && SB == 256`; other arithmetic stays unchanged.

Both arms were run separately on one gfx1100 XTX under GPU leases. Each used the same IQ8R model and `rad-kbench --op gemm_nt_q --max-tok 33833 --max-cases 1 --timing-only --bench-reps 6 --bench-warm-ms 200` with a 256 MiB weight rotation. Both returned 369 timed cases; **neither pass checked numerical outputs**. Source rewrite `--check` and the isolated gfx1100 HIP GEMM build passed.

## Shape results

| M × N × K | Stock µs | Candidate µs | Logical TOPS (stock → candidate) | Gain |
| --- | ---: | ---: | ---: | ---: |
| 2048 × 30720 × 3840 (video FFN up) | 5567.80 | 5561.17 | 86.78 → 86.89 | +0.12% |
| 2048 × 3840 × 15360 (video FFN down) | 2859.28 | 2850.04 | 84.49 → 84.77 | +0.32% |
| 2048 × 4096 × 3840 | 752.58 | 735.05 | 85.60 → 87.65 | +2.38% |
| 2048 × 6144 × 2048 | 597.00 | 548.12 | 86.33 → 94.03 | +8.92% |
| 4160 × 4096 × 4096 | 1628.50 | 1571.45 | 85.72 → 88.83 | +3.63% |
| 32640 × 4096 × 4096 | 12570.60 | 12577.98 | 87.12 → 87.07 | -0.06% |

**1.83% geometric-mean gain** across 40 comparable large shapes after excluding physically implausible timings (greater than 200 logical TOPS). Some rows in both arms reported thousands of TOPS and must not be treated as evidence. The `M=6144` gain above is an **output width**, not the requested exact input token-count sweep. The model-derived M sweep includes 2048, 4160, 8320, 16640 and 32640; exact M6144 and M13376 were not sampled.

## Comparator and decision

Installed rocBLAS gfx1100 Tensile int8 metadata includes 128×128×32 solutions with 16×16×16 matrix instructions, depth-U 32 and four-wide global reads. The precise solution chosen for the independent 2048×15360×3840 plain-int8 reference benchmark was not traced. The plain-int8 reference at 127.2 TOPS is **not equivalent work** to IQ8R scale folding with bf16 output.

The scratch candidate adds scale-repacking complexity but gains only 0.1–0.3% on the two dominant LTX FFN projections. It remains **unverified for bf16 byte parity**, and cannot be timed as a full issued block with a focused-unit plugin; no candidate AQL block-time result is claimed. Prior baseline evidence for a single gfx1100 XTX at 2048 tokens is 6.702 ms video FFN and 13.046 ms video attention-plus-FFN block; see `docs/evidence/2026-11-22-ltx03-issued-block-timing/`.

**Decision:** retain the existing managed kernel. Do not ship or merge the scratch fold rewrite on timing-only results. Revisit only after byte-parity, a complete issued-block AQL A/B, and additional tile/K/weight-staging comparisons on the actual LTX shapes. No production source was edited for this experiment.
