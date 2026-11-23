# VTP37: decode A/B of the K-parallel tile look-ahead removal, 2020.8.5 against 2020.8.4

Release 2020.8.5 adds one change over 2020.8.4: the K-parallel int8 tile no longer prefetches the weight and token fragments a scale group ahead in the register rings that cost occupancy on gfx1100
(#1462; bit-identical, the removal only changes register allocation and scheduling). The cold replay measured the K-parallel row 2-9% faster and a bound of about 0.8-0.9 ms a pass.
This note records the serving-level measurement.

## Setup

Same harness and layout as `docs/evidence/2026-11-24-vtp37-logits-acc-once-ab/`: `qwen27b-tp3`, IQ8R linears, DFlash2 drafter, speculation depth 3 in both arms, untraced; two ABBA cycles
(A B B A A B B A; A = 2020.8.4, B = 2020.8.5), one engine launch per arm, each arm under its own GPU lease, seeds 1-3, three repetitions per seed, streams 6 and 8, 192 new tokens, temperature 0,
512-token prompts. Three fixed prompts at temperature 0 per arm for a text check.

## Result (within this session only)

Each arm mean is the mean of the engine's decode ms/step over 3 seeds x 3 repetitions; difference of the four A arm means and the four B arm means, Welch standard error.

| streams | 2020.8.4 ms/step, mean (sd of arm means) | 2020.8.5 ms/step, mean (sd) | difference (s.e.) | draft acceptance | tokens/step |
|---|---|---|---|---|---|
| 6 | 30.30 (0.51) | 30.22 (0.66) | -0.08 ms, -0.3% (0.42) | 0.721 / 0.728 | 16.1 / 16.2 |
| 8 | 31.68 (0.06) | 31.76 (0.15) | +0.07 ms, +0.2% (0.08) | 0.721 / 0.721 | 18.1 / 18.3 |

Arm means (6 streams): A 29.73, 30.06, 30.88, 30.54; B 29.53, 31.10, 30.21, 30.03. (8 streams): A 31.71, 31.59, 31.70, 31.73; B 31.62, 31.77, 31.67, 31.96 ms/step.
The text of the three fixed prompts is byte-identical in all eight arms.

## Reading

- No decode ms/step difference is measurable: -0.08 ms (s.e. 0.42) at 6 streams and +0.07 ms (s.e. 0.08) at 8 streams. At 8 streams the spread is small, so the effect is bounded to
  below about 0.2 ms (two standard errors); at 6 streams the bound is about 0.8 ms, which does not exclude the replay's 0.8-0.9 ms.
- The replay's per-pass bound is therefore not visible in serving at 8 streams, where the isolated gain should have shown. Likely reasons, not separated here: the kernel is stream-bound
  on the XTX ranks and the verify step is paced by the collective, not by the XTX GEMM time alone; or the look-ahead had already cost little at the M of the serving mix.
- Draft acceptance, tokens per step and output text are unchanged, as expected for a bit-identical change. The change is kept on its replay and bit-identity evidence only; do not chain it
  with earlier A/B deltas (absolute levels drift between sessions).
