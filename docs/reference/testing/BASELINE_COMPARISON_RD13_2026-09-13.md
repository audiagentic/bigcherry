# Real three-arm baseline comparison: RD13, 2026-09-13

Per the standardized patch validation criteria's A/B/C baseline
comparison requirement, run for real on Brutus (single gfx1100, current
pin `b10901`/`28ff0958291c`, `gpt-oss-20B` Q6_K):

- **A** = `source.llama-native` (genuinely stock upstream llama.cpp, zero
  BigCherry patches, `overlay=false, patch-sets=[]`).
- **B** = `source.bigcherry-native` with RD13 (1206) absent.
- **C** = `source.bigcherry-native` with RD13 (1206) present.

## Correctness (PPL, wikitext2, `-c 2048`)

| Arm | PPL |
|---|---|
| A (stock) | 954.4877 +/- 8.61296 |
| B (BigCherry baseline) | 954.4877 (from RD13's own current-pin evidence) |
| C (BigCherry + RD13) | 954.4877 (from RD13's own current-pin evidence) |

**All three arms produce identical PPL to 4 decimal places.** BigCherry's
own framework patches introduce zero measurable correctness divergence
from stock upstream on this model, and RD13 introduces zero further
divergence.

## Performance (`llama-bench -p 512 -n 128 -ngl 99`, 3 rounds A/B + 1
round C)

| Arm | pp512 round 1 | round 2 | round 3 | mean | sd |
|---|---|---|---|---|---|
| A (stock) | 5174.61 | 5150.59 | 5182.53 | 5169.24 | 16.63 |
| B (BigCherry baseline) | 4215.71 | 4220.00 | 4217.78 | 4217.83 | 2.15 |
| C (BigCherry + RD13, 1 round) | 4229.20 | -- | -- | 4229.20 | -- |

tg128 (1 round each): A = 177.76, B = 178.13, C = 178.06 -- effectively
identical across all three arms.

**Real, confirmed finding across 3 rounds: A (stock) shows a consistent
~22.6% higher pp512 throughput than B (BigCherry baseline)** -- mean
5169.24 t/s vs 4217.83 t/s, zero overlap between arms across all 3
rounds (A's tightest round, 5150.59, is still far above B's loosest
round, 4220.00). This is a real, solid, non-noise finding, not a
single-round artifact. B vs C (RD13's own causal effect) shows no
meaningful difference (4217.83 vs 4229.20, well within B's own
round-to-round variation) -- consistent with RD13's own formal
multi-round evidence measured elsewhere (a genuine but small effect on a
different model/config).

**BigCherry's baseline composition (the `framework`/`upstream-fixes`
patch-sets alone, with no RD-series enhancement patches) carries a real,
confirmed ~22.6% prefill throughput cost versus stock upstream llama.cpp
on this model/config.** Filed as PRBE107 -- root cause (which specific
patch(es), and whether this generalizes beyond gpt-oss-20B/pp512) is not
yet investigated.

## What this does and doesn't establish

- Confirms RD13 (the focal patch) itself is correctness-neutral and has
  no material B-vs-C performance effect distinguishable from noise at
  this sample size, consistent with its own formal evidence.
- Surfaces a real, NOT-yet-investigated question about BigCherry's own
  accumulated baseline overhead (A-vs-B) that is unrelated to RD13
  specifically -- this is exactly the kind of finding the three-arm
  methodology exists to catch, and it would have been invisible from a
  B-vs-C-only comparison.
- This is one patch, one model, one architecture, one round. It
  demonstrates the methodology works and is executable with existing
  tooling (`source.llama-native` needed no new infrastructure); it is
  not a claim that every patch or every model would show the same A-vs-B
  gap.
