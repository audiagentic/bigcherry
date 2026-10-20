---
id: BCOP299
order: 299
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-18'
created-by: agent
priority: P2
work: S
---

# PRBE38: count only the residual literal GEMV-to-gated-pointwise launch

**Discovery:** Pinned b11552 and inspected upstream master already fuse eligible UNARY(SILU/SIGMOID/SOFTPLUS)+MUL into one pointwise kernel and canonical paired-GEMV GLU into the vector epilogue. They do not prove a literal GEMV→UNARY→MUL triple is executed, nor expose a generic external-operand GEMV epilogue. QFP13's ~26/token old neighbourhood count cannot be claimed as 52 removable launches. Conditional one-launch gap ceiling 0.091 ms/token (~0.714% at illustrative 78.5 t/s); not measured.

**Owner/consolidation:** PRBE38 is the existing backend/correctness owner (its completed planning state is not runtime qualification); QFP13 owns launch attribution and performance threshold. PRBE37 canonical GLU, PRBE39/1206 ADD, PRBE40/1205 dual projections and QFP35 HC remain separate. No new technical plan, patch, runtime knob, dispatcher, allocator, scheduler or telemetry.

**Branch/12-hour/novelty:** Selected independently updated main at 67ddabd8e47731e759ea6a0bb45e12a79a9510e8 (2026-10-18 08:17:32 UTC), newer than independent feat/rad06-balance-kld 44103dfc (08:03:55 UTC). Main, other active branches and recent PRs show no independent PRBE38/literal-epilogue implementation, planning, benchmark or queued lane within 12h; external/private queues are not verifiable. Protected/excluded RAD06 helper WMMA, AHALF, VECIO, GEMV, long-row splitting, balance, profiling/KLD and their hardware lanes. Unlike prior PRBE38 planning (Oct 4/8), this source-level audit establishes native pointwise-launch accounting and the missing external-operand epilogue semantics.

**Measured boundary / checks:** QFP13 historical ~26 neighbourhoods/token and ~3.5 us gap are not per-route critical-path measurements. Host-only 8192/8192 F32 bit differences for each SILU, SIGMOID, SOFTPLUS under invalid swapped-operand GLU substitution; 5/5 scoped pinned-source assertions. No repository tests, HIP build, GPU benchmark or new measured gain. External [P100 patch 20](https://github.com/shinbunbun/llama-cpp-p100-patches/blob/main/docs/patches.md) reports +0.72% for a different ADD→UNARY→MUL fusion but documents unsafe allocator aliasing; use only as safety precedent.

**Unresolved action:** PRBE38 owns exact graph/consumer/launch evidence first; QFP13 supplies existing rocprof/1319/1320/1325 receipts after RAD06 lanes end. Stop if no literal eligible chain, already fused, alias unsafe or exposed ceiling below QFP13 gate. Only on a real route test a default-off same-backend GEMV external-operand epilogue with source-shaped matcher/alias tests, full logits/KLD/MTP/graph/physical-work and gfx1100/gfx1201 resource/e2e ABBA. No duplicate queued hardware lane.

**Terminal:** NO_LITERAL_CHAIN / ALREADY_NATIVE_FUSED / ALIAS_UNSAFE / BELOW_EXPOSED_CEILING / NUMERIC_DRIFT / NO_GAIN, or QUALIFIED_LITERAL_EPILOGUE only after >=0.5% E2E with positive CI95-low or >=0.10 ms/token serialized gap, no E2E regression, <=1% control regression and zero correctness failures.
