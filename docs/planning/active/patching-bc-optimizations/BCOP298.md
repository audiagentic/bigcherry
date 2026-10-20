---
id: BCOP298
order: 298
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-18T08:06:21Z'
created-by: agent
priority: P2
work: S
---

# PRBE117: gate new Q6_K grouped-integer-dot VDR2 after rejected RD08

**Discovery:** Merged llama.cpp [#30517](https://github.com/ggml-org/llama.cpp/pull/30517) (2026-10-17) changes Q6_K VDR2 arithmetic materially from rejected BigCherry 1204: two signed integer dp4a contributions and pairwise integer accumulation before one F32 scale, rather than per-chunk signed-byte subtraction and F32 accumulation. This is new upstream evidence, **not** a revival of 1204. Upstream R9700/gfx1201 Q6_K kernel N=1 48.0→42.1 us and N=2 63.0→43.2 us are external, not BigCherry gains.

**Ownership/consolidation:** PRBE117 owns the new Q6_K Q8_1 MMVQ arithmetic gate and exact implementation/qualification design. Rejected 1204/RD08 and deprecated PRBE08 remain terminal; PRBE47 owns post-dot DPP, PRBE110/PRBE04 Q6 MMQ prefill, 0600/0650 geometry, validated 1274 raw-F32 gfx1100 dense width1, PRBE116 Q3 and PKC05 Q2. No second kernel, allocator, dispatch table, telemetry framework, scheduler or hardware queue.

**12-hour/novelty:** Independent live branch `feat/rad06-balance-kld` at `44103dfc081c` (2026-10-18 08:03:55 UTC). No independent Q6_K VDR2 patch, plan, PR or recorded queue in inspected preceding-12-hour commits/PRs. Excluded live RAD06 FP16 WMMA, GEMV, AHALF, VECIO, long-row split, per-layer balance, KLD/replay/profiling and qualification. No protected files touched; off-repository queues unverified.

**Measured boundary:** Historical BigCherry 1204 gfx1201 decode +0.204% (CI95-low +0.094%, below +0.3% gate), prefill -0.089%; separately validated 1274 gfx1100 Q6 dense width1 +4.17..+4.28%, **bypassing Q8_1 VDR2**. No current-pin new GPU measurement. Host-only source-shaped fixture 8,192/8,192 signed dot comparisons exact; 904/2,048 synthetic F32 pair sums not bit-identical. No HIP build, repository tests or GPU run.

**Unresolved action:** Gate 0: identify real eligible Q6_K Q8_1 route, compiled geometry and >=3% optimistic exposed E2E ceiling, with physical-work/critical-path receipts. Reject `NO_Q6_Q81_ROUTE`, `NO_EXPOSED_COST` or `BELOW_CEILING` without patch/queue. Only if admitted: isolated new-identity Q6 grouped-int candidate, full logits/KLD/graph/multi-request and work accounting, then four-session/architecture ABBA >=3% CI95-low E2E, <=1% control regression. `NUMERIC_DRIFT`, `MISSING_WORK`, `NO_GAIN` or `REGRESSION` retain baseline; `QUALIFIED_Q6_GROUPED_INT` requires all gates. Technical detail is solely in PRBE117.

**Sources:** [upstream #30517](https://github.com/ggml-org/llama.cpp/pull/30517), [BigCherry 1204](https://github.com/audiagentic/bigcherry/tree/main/engines/llamacpp/patches/1204_rd08_q6k_mmvq_vdr2), [1274](https://github.com/audiagentic/bigcherry/tree/main/engines/llamacpp/patches/1274_mmvq_kquant_f32_decode).
