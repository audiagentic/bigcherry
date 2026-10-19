---
id: BCOP322
order: 322
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-19'
created-by: agent
priority: P2
work: S
---

# PRVP05: Vulkan wave-local RMSNorm subgroup reduction admission

**Discovery:** Pinned b11552 Vulkan `rms_norm.comp` uses a fixed 512-thread shared-memory reduction with nine tree levels/barriers. Open upstream [llama.cpp #29882](https://github.com/ggml-org/llama.cpp/pull/29882/files) adds subgroupAdd, 128/256/512 variants, and an explicit feature gate. AMD 7900 XT external prefill gains range from ~0.8% (9B) to ~7.2% (0.6B), with a slight 9B decode regression. These are not BigCherry gains. No local implementation is justified during RRVP02's Vulkan pause.

**Owner/consolidation:** New PRVP05 owns this **Vulkan shader-only** capability; no existing PRVP item owns RMSNorm. RNX08's recently audited HIP F32 vector-load kernel, PRBE15's HIP IMRoPE scatter, PRVP02 CM1 and PRVP03 Vulkan AllReduce are independent and untouched. Reuse upstream pipeline/shader dispatch; no duplicate dispatcher, allocator or config.

**Branch/exclusion/novelty:** Independently active `feat/rcs09-04-expert-index` at Audumla `d54690b0` (2026-10-19 08:09:30 UTC), based on main `504282d0`. Repository-visible 12-hour independent-work exclusion PASSED for Vulkan RMSNorm; no matching recent commit, PR, patch, plan or recorded queue. Excluded protected RCS09/RCS05, RAD11, RAD12, RAD10, WSL and release work. Off-repository queues not verifiable. Distinct Vulkan wave64 subgroup reduction is not RNX08 HIP register reuse.

**Cheap discriminator:** Host-only wave64 integer address/reduction oracle 204/204 cases passed for 128/256/512-thread workgroups, 17 row widths and four patterns; no GPU build or benchmark. It cannot establish F32 parity, occupancy or performance.

**Unresolved action:** Keep `BLOCKED_VULKAN_PAUSE` until RRVP02 lifts the pause. Then check upstream merge/pin absorption, actual Vulkan RMSNorm route, wave size/feature gate and exposed >=2.9126% E2E removable ceiling. If positive, qualify exact F32/KLD/MTP/graph and paired CI95-low >=3% E2E. Terminal dispositions: `PIN_ABSORBED`, `NO_ROUTE`, `BELOW_CEILING`, `NO_GAIN`, `REGRESSION`, or `PROMOTED`. Full technical design and benchmarks belong solely to PRVP05.
