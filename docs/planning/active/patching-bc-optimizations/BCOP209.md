---
id: BCOP209
order: 209
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-14T12:10:48Z'
created-by: agent
priority: P2
work: S
---

# QFP13: reconcile rollback copy classes and gate residual launches

**Discovery/change:** b11552 Qwen4Exp `build_conv_state_at` creates a strided convolution-tail VIEW and flattened cache VIEW. Validated 1308 eliminates the CONT intermediate but retains a mandatory cache CPY. `cpy.cu` admits neither contiguous memcpy nor its same-shape 2D path for multi-channel source/destination; F32 direct copies use a strided scalar kernel. Historical ~89 of ~108 `copyBuffer*` launches were rollback pairs; ~19 were other neighbourhoods. Counts cannot be subtracted to infer current `copyBuffer*` counts because kernel family changes.

**Owners / consolidation:** QFP13 ranks only the residual copy opportunity; 1308 is the validated rollback implementation. QFP16/1326/QFP27 owns scheduler input copy/lifetime; native GDN/SSM state-cache fusions are different producer shapes. No duplicate matcher, scheduler, cache, allocator or telemetry introduced. QFP13 and 1308 SUMMARY corrected.

**Already acting / exclusion:** 1308 was validated with historical +3% at ~10K and +3.6% at ~80K, greedy identity. Last QFP13 independent plan edit 2026-10-04; 1308 implementation last mechanically relocated 2026-10-09. No independent QFP13/1308 commits or PRs within the inspected prior 12h. Protected current RAD02/RAD03 remote-MoE/mailbox and RAD04 two-half prefill were not modified. Prior BCOP199-208 audit slices cover different mechanisms.

**Unresolved action:** One read-only b11552 trace correlating each residual copy to GGML graph node, shape/stride, owning GPU and HIP kernel/event. Require exposed serial saving >=0.10 ms/token or >=0.5% E2E before routing a bounded candidate to its existing owner. Illustrative 19-copy arithmetic using historical global medians is 0.0893-0.1045 ms/token, not measured. Upstream #25057 2D memcpy is already in b11552 but shape-ineligible; vLLM causal_conv1d_update is not a direct transport/rollback port.

**Terminal:** `ALREADY_COVERED_1308`, `REQUIRED_STATE_WRITE`, `UNVERIFIED_ROUTE` or `BELOW_EXPOSED_GATE`; otherwise owner-local default-off correction with multi-seq/ubatch, rollback, graph and exact-output tests.

**Validation:** 216 host shape/semantic cases, two same-shape guards and nine launch identities passed; Python compilation passed. No repository pytest, patch apply, HIP build, GPU execution or new benchmark.
