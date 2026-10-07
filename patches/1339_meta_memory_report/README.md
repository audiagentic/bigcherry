# 1339_meta_memory_report

Promotion record (QFP18 lightweight evidence-reuse tier, pin b11402 / d89651a7). Mechanism is in SUMMARY.md. This is a
diagnostic, promoted as a neutral enabler: with `BIGCHERRY_META_MEM=1` the tensor split reports the size of every
buffer on every device and the fusion overlap counters; without the variable it prints nothing and changes nothing.
1340 (per-device compute arena) is written on top of its hooks, so the two are promoted together.

## Evidence

- No effect when off: every "production" arm of the MSM02 / MSM03 runs of 2026-10-07 (builds `b-metamem-msm3kq`,
  `b-metamem-msm2s2`, `b-metamem-msm2t2`: production set + 1339 + 1340, flags off) gave the production greedy texts
  and probe distributions (production against itself: 24 probes, top-1 24/24, TV 0.0000).
- Activation: the report lines themselves (`BIGCHERRY_META_MEM compute|arena|static ... dev=`), which
  `tools/lab/flash-next/queue-meta-mem.sh` sums per device; they are the source of every memory figure in 1340's and
  1341's promotion records.
- Offline: package tests (`tools/tests/patch/test_1340_meta_per_device_arena.py` and
  `test_1341_meta_subset_mirrored.py` compose it), patch-lint, production composition check.
