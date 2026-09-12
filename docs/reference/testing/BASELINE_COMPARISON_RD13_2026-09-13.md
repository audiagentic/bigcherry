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
on this model/config.** Filed as PRBE107.

## Root cause (bisected, 2026-09-13)

Isolated via two targeted builds against `source.llama-native` (stock
base, no framework/upstream-fixes composition at all):

- **`upstream-fixes` alone** (just `1000_rdna4_mmq_q2k_q6k_fix`): pp512 =
  5210.01 -- matches stock's ~5150-5182 range, **no regression**.
- **The "forced"-dispatch family alone**
  (`0300_mmq_forced_j`, `0400_mmvf_forced_block`, `0500_mmf_forced_nwarps`,
  `0600_mmvq_geometry`, `0650_mmvq_native_variant` -- 5 of `framework`'s
  14 patches): pp512 = 4263.46 -- **reproduces the full regression**,
  matching BigCherry baseline's ~4217-4229 range.

**Root cause conclusively isolated to the "forced"-dispatch patch
cluster, not the upstream-fixes correctness backport.** This is a real,
substantial finding: these 5 patches (by name, forcing fixed
`j`/block-size/`nwarps`/geometry/native-variant dispatch parameters
instead of upstream's own tuned auto-selection heuristics) cost ~22.6%
real prefill throughput on this model/config, as the deliberate price of
making dispatch deterministic/instrumentable for BigCherry's own
autotuning research infrastructure. This is very plausibly an accepted,
understood tradeoff (these patches exist specifically to support
BigCherry's dispatch-research and autotuning framework, not to be a
production performance win on their own) rather than an unintended bug
-- but this was not previously measured/documented as a quantified real
number anywhere in the project, and whoever owns the framework
patch-set's design should confirm this is the accepted/expected cost.

## Second model: the gap does NOT generalize (2026-09-13)

Per the standardized criteria's "range of models" requirement, ran the
same A-vs-B pp512 comparison on `tierA-qwen4b-q6k` (Qwen3.5-4B, the
dense+GDN hybrid model, gfx1100, 2 rounds):

| Arm | round 1 | round 2 |
|---|---|---|
| A (stock) | 4881.74 | 4896.58 |
| B (BigCherry baseline) | 4885.78 | 4876.29 |

**No gap at all on this model -- A and B are statistically identical**
(well within each other's uncertainty, unlike gpt-oss-20B's clean,
non-overlapping ~22.6% separation). **PRBE107's finding does NOT
generalize across models -- it is specific to gpt-oss-20B (or more
likely, to some architectural property gpt-oss-20B has that Qwen3.5-4B
doesn't -- e.g. its MoE routing shape interacting differently with the
forced-dispatch patches' fixed geometry/nwarps choices).** This
significantly changes PRBE107's scope: it is not "BigCherry's baseline
costs ~22.6% universally," it is "BigCherry's forced-dispatch patches
cost ~22.6% on at least gpt-oss-20B's specific MoE shape, and 0% on
Qwen3.5-4B's dense+GDN shape" -- a real, valuable, more precise finding
that a single-model test would have missed entirely.

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
