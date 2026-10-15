---
id: PHA04
order: 0
plan: patching-hip-autotune
state: pending
created-at: '2026-09-09T10:48:57.946592+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# RCCL admission gate: per-call candidate identity, not requested-plan identity

## Authoritative implementation audit — 2026-10-14, BigCherry b11552

**Disposition: PENDING / OBSERVATION_BLOCKED. No positive level-2 RCCL candidate allowlist, runtime selector, or default promotion.** This is a documentation-only audit; no RCCL binary, GPU, BigCherry patch composition or repository pytest was run. PHA03's topology/AtomicOps safety admission is prerequisite, and PHA05 owns promotion. The existing 1225 fail-closed AtomicOps guard and 0840 provider paths must not be replaced.

### Source-confirmed candidate-verification gap

1. `tools/bigcherry/profiling/rccl_qualify.py::RcclCase.case_id` includes an optional `channels` value, and `RcclCaseResult` stores both `requested_channels` and `observed_channels`. **But `build_command()` applies only `RCCL_OVERRIDE_ALGO` and `RCCL_OVERRIDE_PROTO`; it never applies `case.channels`.** `_classify()` compares only algorithm and protocol. Consequently a request for Ring/Simple/1 channel and an observation of Ring/Simple/4 channels can be labelled `PLAN_VERIFIED`. That status verifies **only algorithm/protocol**, not the complete PHA04 candidate. This is a source-confirmed evidence-integrity defect, not evidence of a bad GPU result. Preserve existing historical result semantics; do not promote those records to channel-qualified evidence.
2. `_parse_observed_plan()` uses the first matching `-M 1` table line. It is not a per-collective correlation protocol; its regex recognizes only standard ring/tree/etc, not RCCL's newer CE/DDA/symmetric/addon backend identities. An empty or substituted plan must remain unknown/unsupported, never silently pass. A multi-call production trace needs an exact communicator, call/epoch, graph-capture and selected-implementation correlation.
3. Patch 0830's `ggml_hip_reduce_telemetry_provider()` and `hip-autotune-reduce-telemetry.cpp::make_signature/write_event` record requested/effective **BigCherry provider**, tensor size/type/shape and a peer-access topology observation. They do **not** record RCCL algorithm/protocol/channels, and the telemetry topology key is **not** PHA03's canonical `RcclTopologyEvidence.qualification_id`. Do not equate `effective_provider=rccl` with a qualified RCCL candidate or join unrelated topology keys by string.
4. Patch 1225's `ggml_backend_cuda_comm_rccl_admission_ok` tests per-device host native atomics at communicator initialization; it is necessary but not a level-2 per-call algorithm gate. Patch 0840's adaptive RCCL branch and explicit `rccl` provider remain under existing ownership. A candidate rejection after `ncclAllReduce` has been enqueued is **too late** to fail safely or transparently switch providers.

### New upstream mechanism and limits (inspected source)

ROCm/rocm-systems `develop` at `05c6bb90e6bc12dc3c3f4ae1826403be8eedcddf` has `projects/rccl/src/rccl_wrap.cc::rcclGetCollImplInfo` and `src/include/rccl_decision.h::rcclCollDecision`. Unlike the older size-only `rcclGetAlgoInfo`, the query uses `rcclSelectAllReduce(..., query=true)` with the **actual send/recv pointers, dtype, reduction operation and an explicit graph-capture hint**. It reports RCCL-specific DDA, CE, symmetric and kernel backends, protocol and channel count. `test/CollImplInfoTests.cpp` compares query output with real dispatch logs. This is a promising *conditional observation reference*, not evidence that BigCherry's installed RCCL exposes it: `RCCL_STATIC_EXPOSE_CHECK` requires an `RCCL_EXPOSE_STATIC`-enabled build. Symbol presence alone is insufficient; a call can reject use. A query predicts selection; it is not proof that a later enqueue executed that candidate when buffers/capture/runtime changed. Revalidate ABI and semantics against the **exact installed librccl SHA/build ID**, never assume `develop` matches ROCm 7.2.x.

RCCL's tuner plugin `getCollInfo` is a candidate-selection *proposal*, not a fail-closed safety boundary: plugin errors fall back to RCCL tuning and partial outputs can leave channels to RCCL. Newer tuner ABI versions differ from older RCCL documentation; neither `NCCL_ALGO/NCCL_PROTO` nor `RCCL_OVERRIDE_ALGO/PROTO` attests to the executed plan. Do not install a plugin or modify RCCL merely to satisfy a proposed optimization.

References:
- https://github.com/ROCm/rocm-systems/blob/05c6bb90e6bc12dc3c3f4ae1826403be8eedcddf/projects/rccl/src/rccl_wrap.cc
- https://github.com/ROCm/rocm-systems/blob/05c6bb90e6bc12dc3c3f4ae1826403be8eedcddf/projects/rccl/src/include/rccl_common.h
- https://github.com/ROCm/rocm-systems/blob/05c6bb90e6bc12dc3c3f4ae1826403be8eedcddf/projects/rccl/test/CollImplInfoTests.cpp
- https://rocmdocs.amd.com/projects/rccl/en/latest/how-to/using-rccl-tuner-plugin-api.html

### Bounded implementation path (no duplicate control plane)

**Gate 0 — repair evidence semantics, no hardware:** In `tools/bigcherry/profiling/rccl_qualify.py`, keep legacy `plan_verification` explicitly algorithm/protocol-only. Introduce a separate, versioned **candidate-verification disposition** (not a second global registry): `UNOBSERVED`, `REQUEST_NOT_APPLIED`, `QUERY_ONLY`, `EXECUTED_MATCH`, `SUBSTITUTED`, `UNKNOWN_BACKEND`. If `RcclCase.channels` is set without a proven channel-control mechanism, refuse full candidate verification; do not pass by matching a coincidental observed value. An absent/zero/negative observed channel count is unknown. Require exact one-to-one match of the selected row to the executed call. Add host tests for 1-requested/4-observed, missing channels, multiple candidate lines, unknown addon backend, fallback, incompatible build and duplicate call IDs. Existing tests live in `tools/tests/profiling/test_rccl_qualify.py` and `test_rccl_qualify_campaign.py`. Do not rewrite historical result files.

**Gate 1 — determine feasible observation seam:** In a crash-isolated host probe against the exact installed librccl, record SHA/build ID, RCCL/ROCm/driver versions, `RCCL_EXPOSE_STATIC` capability, available query ABI, addon enum mapping and graph-capture behavior. Missing symbol, incompatible ABI, `ncclInvalidUsage` or unavailable actual-dispatch correlation => `OBSERVATION_UNAVAILABLE`, stop before a candidate allowlist. Prefer existing RCCL profiler/dispatch logs for executed-route evidence; use `rcclGetCollImplInfo` only as an independent predictor with the same buffers, operation and capture state. Do not parse a single global log line as a call receipt.

**Gate 2 — exact per-call identity:** The existing 0830 provider observer may consume an optional immutable RCCL receipt from the *same* communicator/epoch. Required fields: exact library build/ABI, canonical PHA03 topology qualification ID, ordered physical participant identities, communicator generation, operation (SUM), dtype, element count/bytes, tensor layout, send/recv ownership and registration, graph capture/replay identity, selected backend (including addon class), algorithm, protocol, actual positive channels, source of observation, call ID and post-dispatch outcome. No P2P inference from GPU architecture. No peer-only or NCCL default route on the gfx1030 chipset participant. Bind a candidate key to these fields using the existing `RcclCompatibilityRevision` and PHA03 evidence types; do not invent another topology hash or dispatcher.

**Gate 3 — safe admission:** Only after PHA03 passes and executed-route receipts match a complete qualified tuple may a default-off pre-enqueue gate be considered. Check `qualified && exact_build && exact_topology && exact_signature && exact_candidate && capture_identity && observed_execution_contract` **before any collective side effect**. Unknown, query-only, stale, substituted, performance-excluded and unqualified records fail closed with an explicit error; no implicit Meta or internal fallback when the user requested RCCL. Re-check the actual dispatch against the prediction; mismatch invalidates the qualification and blocks promotion. Never call this a safe production gate while the RCCL library can silently substitute a candidate after the check.

### Cheapest discriminator and hardware qualification

- Static source audit: `RcclCase.channels` appears in case ID/results but not command controls or `_classify`; first-match regex has no call identity. A source-shaped 1-channel requested/4-channel observed fixture currently yields algorithm/protocol `PLAN_VERIFIED`; the required candidate oracle rejects it. **Five source assertions and seven synthetic admission-oracle cases passed in this audit**; these are not repository tests or GPU measurements.
- Positive controls only after observation is proven: homogeneous dual gfx1100; separately CPU-direct gfx1100+gfx1201; short decode (10–40 KiB), large prefill pp2048, 10K/80K/160K where fit, graph capture on/off, same-process repeat and multiple ubatches. Negative controls: gfx1030 chipset-routed participant, missing/changed librccl, substituted algorithm, channel mismatch, addon backend, capture mismatch, wrong call/epoch and failed/partial initialization.
- Use existing RQ08/RQ10 crash-isolated `rccl_qualify_campaign.py` and Brutus runbook. Do not run tests concurrently with RAD02/RAD03/RAD04 hardware experiments. Compare exact-F32 CPU oracle, greedy/logits/KLD and MTP acceptance where relevant; preserve physical transfer and launch accounting. A channel-only improvement is not an end-to-end claim. Require >=4 independent sessions and paired ABBA; promotion only with CI95-low >=3% E2E on a relevant lane, <=1% unaffected-lane regression, zero correctness faults and a verified executed candidate. If exposed RCCL critical-path opportunity is <3%, close without tuning.

### Ownership, dependencies, terminal dispositions

- **PHA04:** per-call candidate identity, verification contract and future pre-enqueue candidate admission. **PHA03/1225:** physical topology, AtomicOps and communicator-init safety; no modifications in this audit. **PHA05:** default-build promotion; remains blocked. **0830:** provider-level telemetry; extend only through its existing optional observation seam. **RQ08/RQ10:** qualification campaign, not runtime selector. **PGC09/PGC14/QFP01:** provider/wire/performance work; no second RCCL selector, communicator, scheduler or telemetry registry.
- Terminal: `OBSERVATION_UNAVAILABLE` (no supported executed-route receipt), `NO_ELIGIBLE_CANDIDATE`, `CANDIDATE_MISMATCH`, `SAFETY_FAIL`, `BELOW_EXPOSED_GATE`, `QUALIFIED_DEFAULT_OFF`, or `PROMOTION_READY` only after PHA03/PHA05 and full hardware gates. Until then PHA04 stays pending and **does not change production behavior**.

## Historical assessment (2026-09-09)

The original assessment correctly identified that 0830 does not expose RCCL algorithm/protocol/channel identity. It did not distinguish the existing channel-request/verification gap, newer backend-aware RCCL query API, query versus actual execution, or RCCL's plugin fallback. Preserve its pending disposition; the former generic "bind immediately before dispatch" step is conditional on a proven pre-enqueue identity seam, not an implemented feature.

## Change Log

- 2026-09-09T10:48:57.946592+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:05:02.519090+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events



- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.848546+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T13:42:25.760165+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:acceptance_criteria
- chg_20260909_134238_pha04s-dependency-and-evidenc_1280
- 2026-09-09T13:42:38.013204+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.356558+00:00 (updated-by): Updated: section:ledger-events
