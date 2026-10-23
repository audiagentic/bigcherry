---
id: BCOP340
order: 340
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-20'
created-by: agent
priority: P2
work: S
---

# PRBE116: Q3_K HIP MMVQ VDR2 packed-scale admission

**Discovery:** Merged [llama.cpp #30517](https://github.com/ggml-org/llama.cpp/pull/30517) (17 October 2026) changes pinned b11552 Q3_K MMVQ VDR1 to VDR2, packed scale loads and signed DP4A regrouping. External gfx1151 Q3_K kernel gains 1.08–1.37× across widths 1/2/4/8; external gfx1200 HIP Q3_K n=1 remains slower than Vulkan. **No first-party BigCherry Q3_K executed-route or gain is established.** Host-only 200,000 scale and 400,000 integer-dot comparisons pass; 72,804/200,000 illustrative F32 regroupings differ. Not a HIP correctness test.

**Owner/consolidation:** New PRBE116 is the single Q3_K dot-body technical owner because existing 0600/0650 owns only MMVQ geometry/forced-instance plumbing, PRBE111/1273 IQ, QFP10 Q4_K MoE, PRBE47 Q6_K, and QFP44/0300 MMQ. Do not create another dispatch/allocator/cache/telemetry/config or patch until route/cost admission; technical detail and benchmarks belong in PRBE116.

**12-hour/novelty:** Selected `fix/patch-suite-green` from independent Audumla `dc0669a7c541` (20 Oct 13:57:27 UTC), 22 seconds newer than the Radiance release head. Inspected branch/PR/commit activity in 06:09–18:09 UTC. Excluded RAD14 FP8/W4/MTP, RAD06 Flash-Next/shape harness, RCS05/06, 0300 MMQ test repair, Radiance 1.9.0 bump/queued GPU validation. No independent Q3_K MMVQ dot work/queue found; off-repository queues unverified. Novel versus Q4_K/Q6_K/IQ MMVQ and MMQ compact-grid scheduled audits.

**Unresolved/terminal:** `NO_Q3K_ROUTE` or `BELOW_CEILING` without an executed Q3_K dense width route and ≥2.9126% optimistically removable exposed E2E wall. Otherwise source-compose only Q3_K VDR2 in existing MMVQ path; gate numeric/graph/K coverage, wave/ISA/VGPR/spill/physical work, then paired gfx1100/gfx1201 CI95-low ≥3% E2E and ≤1% unaffected controls. `WAIT_UPSTREAM` if next pin absorbs #30517; reject `NUMERIC_FAIL`, `SPILL_REGRESSION`, `NO_GAIN`. No GPU work queued or implementation changed.
