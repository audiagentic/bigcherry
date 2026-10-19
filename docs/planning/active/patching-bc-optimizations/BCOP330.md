---
id: BCOP330
order: 330
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-19'
created-by: agent
priority: P3
work: S
---

# PRBE18: HIP GDN Q/K paired L2 normalization launch gate

**Discovery:** Pinned b11552 already fuses each Q/K RMS_NORM→SCALE independently. [RDNA fork #179](https://github.com/stew675/llama-cpp-rdna-boosts/pull/179) combines the two 256-thread F32 norm launches without reducing their 2R CTAs or required logical bytes. Fork gfx1201 Qwen3.6-35B-A3B decode +1.2%, MTP +0.2–0.3%, hybrid Flash-Next +0.37% CI95 [+0.04,+0.70]; **not BigCherry measurements**, and below the ≥3% E2E bar. The fork matcher lacks an explicit full-interval two-output subgraph/memory-range proof before skipping intervening nodes; require this if ever ported.

**Owner:** Existing PRBE18; no second technical plan. PRBE21/BCOP328 conv-input CONCAT, PNRO04/05 recurrence, PRBE41 SSM, RNX08 norm/RoPE and PRVP05 Vulkan RMSNorm remain distinct. Historical PRBE18 16-node fusion stays rejected. No new dispatcher, allocator, telemetry, config, patch or hardware queue.

**Independent work/novelty:** Selected feat/rad13-fp8-worker at independent Audumla fa22e03 (2026-10-19 16:10:20 UTC), newer than main f792f51e. PRBE18 last independent edit 2026-10-11 03:05 UTC; no repository-visible matching capability commit, PR, plan edit or queue in preceding 12h. Protected RAD13/PR #330, RCS09/RCS05, RAD11/12/10, workspace and release work untouched; off-repository queues not verified. Unlike BCOP328, this is Q/K normalization launch fusion, not GDN CONCAT.

**Completed:** b11552 model/fusion/norm source and fork #179 patch/benchmarks inspected. Host-only 144 geometry cases, 288 same-ordered reference comparisons, 24 fallback cases, four output alias-span cases (two overlaps), Python compilation passed. No GPU build/test/benchmark.

**Unresolved action:** PRBE18 Gate 1 read-only first-party graph + per-rank exposed-wall accounting. **Terminal:** NO_PAIR_ROUTE or BELOW_CEILING (<2.9126% optimistically removable E2E wall) → reject without patch; PIN_ABSORBED → use approved upstream pin; if eligible, single existing-fusion-path prototype with two-output alias/lifetime guards, gfx1100/gfx1201 ISA/spill and logits/KLD/MTP/graph/physical-work qualification. Promote only CI95-low ≥3% E2E without material regression; otherwise NUMERIC_FAIL, ALIAS_UNSAFE, REGRESSION or NO_GAIN. Do not interfere with active protected work.
