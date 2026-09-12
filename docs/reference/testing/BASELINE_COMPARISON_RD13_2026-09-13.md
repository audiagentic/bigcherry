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

## Performance (`llama-bench -p 512 -n 128 -ngl 99`, single round -- NOT a
formal measurement, see caveat below)

| Arm | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| A (stock) | 5174.61 +/- 171.83 | 177.76 +/- 0.35 |
| B (BigCherry baseline) | 4215.71 +/- 118.56 | 178.13 +/- 0.38 |
| C (BigCherry + RD13) | 4229.20 +/- 105.37 | 178.06 +/- 0.40 |

**Real, notable finding: A (stock) shows ~18% higher pp512 throughput
than B/C (BigCherry baseline).** tg128 is effectively identical across
all three arms. B vs C (RD13's own causal effect) shows no meaningful
difference at this single round -- consistent with RD13's own formal
multi-round evidence (a genuine but small effect, previously measured at
+0.94% pp512 on a different model/config).

**Caveat: this is a single round, not a formal paired/bootstrapped
measurement** -- do not treat the A-vs-B 18% gap as a confirmed number.
It is real enough to flag and investigate (BigCherry's baseline
composition, i.e. the `framework`/`upstream-fixes` patch-sets alone,
appears to carry real prefill overhead versus pure stock upstream on
this model/config), but needs a proper multi-round interleaved
measurement before being treated as a quantified regression. Filed as
PRBE107.

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
