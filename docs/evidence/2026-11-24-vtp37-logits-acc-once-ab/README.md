# VTP37: decode A/B of the logits accumulate-once change, 2020.8.4 against 2020.8.3

Release 2020.8.4 adds one change over 2020.8.3: the int8 vocabulary head (`lm_wmma_kernel`) keeps its WMMA partial sums in the gfx11 layout per scale block and converts once
(bit-identical by construction; see the logits evidence note under `docs/evidence/`). The replay predicted about 0.2-0.3 ms of the step. This note records the serving-level measurement.

## Setup

Same harness and layout as `docs/evidence/2026-11-24-vtp37-kpar-acc-once-ab/`: `qwen27b-tp3`, IQ8R linears, DFlash2 drafter, speculation depth 3 in both arms, untraced.
Because the expected effect is near the noise floor the protocol has more power: two ABBA cycles (A B B A A B B A; A = 2020.8.3, B = 2020.8.4), one engine launch per arm,
each arm under its own GPU lease, seeds 1-3, three repetitions per seed, streams 6 and 8, 192 new tokens, temperature 0. Each cell below is a per-seed median, n = 12 per release and stream count.
Three fixed prompts at temperature 0 per arm for a text check.

## Result (within this session only)

| streams | 2020.8.3 ms/step, mean (sd, n=12) | 2020.8.4 ms/step, mean (sd, n=12) | difference of means (s.e.) | draft acceptance | aggregate tok/s |
|---|---|---|---|---|---|
| 6 | 30.58 (1.77) | 29.67 (1.63) | -0.91 ms (0.69) | 0.725 / 0.730 | 338.9 / 331.1 |
| 8 | 31.82 (0.52) | 31.79 (0.62) | -0.03 ms (0.23) | 0.725 / 0.719 | 358.6 / 357.2 |

Arm means over both stream counts: A1 31.28, B1 31.22, B2 31.28, A2 31.52, A3 31.47, B3 30.08, B4 30.35, A4 30.53 ms/step.
The text of the three fixed prompts is byte-identical in all eight arms.

## Reading

- The measured differences (-0.9 ms at 6 streams, -0.03 ms at 8) are inside the noise: neither is distinguishable from zero (t = -1.3 and -0.1), and the seed-to-seed spread
  (sd 0.5-1.8 ms) is larger than the predicted 0.2-0.3 ms effect. This measurement neither confirms nor refutes the replay prediction; it only bounds the effect to about +-1 ms.
- Draft acceptance and tokens per step are unchanged, and output text is identical, as expected for a bit-identical change.
- Absolute levels drift between sessions; compare within a session only. This note must not be chained with the earlier A/B deltas.
- The change is kept on the strength of its isolated replay and bit-identity evidence, not on this serving measurement.
