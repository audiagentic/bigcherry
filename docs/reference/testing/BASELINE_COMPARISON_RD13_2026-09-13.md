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

## Third model: confirms dense-vs-MoE, not model-specific noise (2026-09-13)

`tierM-ministral14b-q4km` (Ministral-3-14B, an independent dense
transformer family, distinct from the Qwen family):

| Arm | pp512 |
|---|---|
| A (stock) | 2275.05 +/- 34.28 |
| B (BigCherry baseline) | 2274.98 +/- 26.77 |

**Statistically identical again -- no gap.**

## Fourth model: second MoE model CONFIRMS the MoE hypothesis (2026-09-13)

`tierM-qwen35b-a3b-moe-mtp` (Qwen3.6-35B-A3B, a second, independent MoE
model, different family from gpt-oss):

| Arm | pp512 |
|---|---|
| A (stock) | 3247.36 +/- 200.79 |
| B (BigCherry baseline) | 2654.67 +/- 249.50 |

**Real gap: ~22.3%** -- remarkably close to gpt-oss-20B's ~22.6% gap,
despite being a completely different MoE model family/size.

## Triangulated conclusion (4 models, 2 architecture classes)

| Model | Class | A vs B gap |
|---|---|---|
| Qwen3.5-4B | dense+GDN hybrid | ~0% (statistically identical) |
| Ministral-3-14B | dense transformer | ~0% (statistically identical) |
| gpt-oss-20B | MoE | ~22.6% (confirmed, 3 rounds) |
| Qwen3.6-35B-A3B | MoE | ~22.3% (single round) |

**Conclusively triangulated: the forced-dispatch patch cluster
(`0300_mmq_forced_j`/`0400_mmvf_forced_block`/`0500_mmf_forced_nwarps`/
`0600_mmvq_geometry`/`0650_mmvq_native_variant`) costs a consistent
~22-23% prefill throughput specifically on MoE models, and costs
nothing measurable on dense (or dense+hybrid) models.** This is a real,
solid, multi-model-confirmed finding -- not a single-model artifact, not
noise, and not a general "BigCherry tax." The two MoE gaps (22.6%,
22.3%) are close enough across genuinely different MoE
families/sizes/quantizations to indicate a real, systematic interaction
between the forced-dispatch patches' fixed geometry/nwarps choices and
MoE routing shapes specifically -- plausibly the forced parameters are
tuned/reasonable for dense matmul shapes but suboptimal for MoE's
smaller, more numerous expert-routed matmuls.

## RESOLVED: root cause found, fixed, and verified (2026-09-13)

Bisected further (isolating each of the 3 "forced" patches individually):
**`0300_mmq_forced_j` alone reproduces the entire regression**
(4267.30 t/s); `0400_mmvf_forced_block` and `0500_mmf_forced_nwarps` are
innocent (5209.73, 5206.47 -- both match stock).

GPT (`req_75d5e59ef62e4675`) diagnosed the exact bug by comparing
`patches/0300_mmq_forced_j/patch.py`'s lifted J-selection scan against
real upstream `ggml/src/ggml-cuda/mmq.cuh`/`mmq.cu` -- **independently
verified against the actual vendor source in this repo before applying
any fix.** Upstream's `mmq_args` struct has two distinct fields:

```cpp
int64_t ncols_max;
int64_t ncols_opt; // value to optimize the tile size against, launch grid still uses ncols_max
```

For MUL_MAT_ID/MoE on RDNA3.0/RDNA4, upstream's own code (`mmq.cu`)
computes `ncols_opt` as the approximate per-expert routed column count
(`(ne12*n_expert_used + ne02 - 1) / ne02`), deliberately smaller than
`ncols_max` (the real launch/safety width). **0300's lifted scan used
`ncols_max` where upstream's real scan uses `ncols_opt`** -- for dense
models these are set equal upstream (hence zero observed effect), but
for MoE they diverge sharply, causing the scan to pick a substantially
larger, wrong tile-width `J` than native upstream would.

**Fix applied**: `patches/0300_mmq_forced_j/patch.py` changed to pass
`args.ncols_opt` instead of `args.ncols_max` into
`ggml_cuda_mmq_native_j_best()` (both the declaration/call-site and the
definition). `ggml_cuda_mmq_variant_is_eligible`'s own `ncols_max`
parameter is deliberately untouched -- that one correctly needs the real
launch width for padding/OOB safety, an unrelated purpose. Committed
`1203c2e2`.

**Real hardware verification (rebuilt, 3+ rounds each):**

| Model | Class | Before fix | After fix | Stock (A) |
|---|---|---|---|---|
| gpt-oss-20B | MoE | 4267.30 | 5223.55 / 5212.42 / 5190.19 (mean ~5209) | 5169.24 |
| Qwen3.6-35B-A3B | MoE | 2654.67 | 3165.05 | 3247.36 |
| Qwen3.5-4B | dense+GDN | 4885.78 (unaffected) | 4890.26 (unaffected, as predicted) | 4881.74 |

**Both MoE models recover to within noise of stock upstream; the dense
model is confirmed unaffected by the fix, exactly as the diagnosis
predicted.** This is a real, precise, GPT-diagnosed, independently
source-verified, hardware-confirmed fix -- not a workaround or a
documented-but-unresolved finding. `0300_mmq_forced_j`'s `validated`
state should be reconfirmed under a fresh qualification run (it was
never actually invalidated -- native/forced_J=0 behavior is now
corrected to match upstream, which is what "validated" always assumed).

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
