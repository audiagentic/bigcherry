---
id: BCOP217
order: 217
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-14T21:09:20Z'
created-by: agent
priority: P1
work: S
---

# FKE01: gate SOFT_MAX top-k fusion on ordered expert/weight identity

**Discovery/change:** Historical FKE01 first-op bisect established same greedy text for nine families, **not bit identity**. SOFT_MAX-off changed greedy text and cost ~4% decode in historical b11402 runs; no causal numerical producer was isolated. Pinned b11552 `tests/test-backend-ops.cpp::test_topk_moe::err` sorts expert IDs and weights independently before NMSE, so it cannot prove ordered expert→weight pairing. 1342 skip/count is family-level, not correlated fused/unfused execution. Correct FKE01 and 1342 prose; no production patch.

**Owners:** FKE01 equivalence evidence; existing upstream `test_topk_moe` test primitive; 1342 diagnostic; MSM02/1340 allocator; MUL-start reduction and 1207/RD17 separate. No edits to active Radiance RAD02/03/04/05/06 routing, helper, GEMV or prefill.

**Selection / exclusion:** `feat/rad03-gpu-lanes` selected from newest independent Audumla Radiance development (re-read branch head before publication). FKE01 last independently updated 2026-10-07 15:37Z; 1342 updated 2026-10-09 05:37Z. No same-capability independent work, PR, evidence or queued experiment in prior 12h. Excluded RAD02/03/05/06 helper/mailbox/routing/GEMV and Radiance evidence; RAD04 prefill untouched. Recent BCOP208–216 are different capabilities.

**Upstream/fork:** llama.cpp #27621's multi-token/barrier/alias mechanism already pinned; #30441 is nonsemantic. SGLang #36811 (2026-09-01 merged) tests exact ordered indices/weights for zero-bias elimination; native ggml already has compile-time `has_bias`. vLLM routing semantics are reference only. No external gains transferred.

**Unresolved action:** Reuse `test_topk_moe` and 1342 with saved deterministic logits, positive matched route evidence, ordered integer IDs and corresponding F32 weight bits/ULP, tie/NaN/normalization, nrows 8/9, alias, MTP, repeated requests and graph replay. Terminal `NO_CORRELATED_ROUTE`, `ROUTER_CORRECTNESS_FAIL`, `NUMERICALLY_DIFFERENT` or `EQUIVALENT`. Only then paired current-pin E2E gates (pure speed change CI95-low >=3%, unaffected-lane regression <=1%). No new scheduler, selector, allocator or hardware queue.

**Actual validation:** Host-only 1024-case 64-expert F32 reduction-order discriminator: 734 differed in weight bits with identical top-8 IDs; independently sorted IDs/weights can hide changed expert→weight mapping; same greedy argmax can hide different logits. Synthetic, not HIP or historical-cause proof. Python compilation passed. No repository pytest, patch application, HIP compilation, GPU execution or new throughput measurement.
