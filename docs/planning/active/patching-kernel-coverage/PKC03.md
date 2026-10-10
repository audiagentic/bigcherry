---
id: PKC03
order: 0
plan: patching-kernel-coverage
state: pending
created-at: '2026-09-09T10:51:51.425061+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: P3
---

# Decide whether collective identity needs a cross-backend extension

## Description

Decision-only gate: determine whether collective identity needs a cross-backend extension. Do not implement a collective registry.

## Steps

1. Inspect EC16/EC19, RRVP identity/persistence, and GP evidence.
2. Test whether collective provider, topology, root, threshold, protocol, fallback, and negative evidence are representable without ambiguity.
3. Assign any proven gap to the existing GP, provider, or contract owner.
4. Close this gate when existing machinery is sufficient.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

EC16/EC19 contract identity fields; RRVP identity/persistence references; GP03/GP07/GP08/GP10 evidence schemas; decision record and owner handoff.

## Validation

Representability matrix for provider, topology, root, threshold, protocol, fallback and negative evidence; ambiguity examples; CPU-only/provider misuse negative checks; owner assignment for any proven gap.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

A documented decision establishes whether existing identity/contracts are sufficient; any gap is assigned to an existing GP/provider/contract owner; no collective registry or backend implementation is introduced.

## Notes

Supersedes: KC03
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-kernel-coverage-kc03

Supersedes: KC03
Inherited constraints: RV107, RV112, RV123 — this is a cross-backend identity sufficiency decision only.
Migration: capability-rebaseline-v3-2026-09

Supersedes: KC03
Inherited constraints: RV107, RV112, RV123 — cross-backend identity sufficiency decision only.
semantic-carryforward: concrete files and validation restored 2026-09-10.

## 2026-10-10 implementation audit: distinguish contract intent from executed collective route

**Decision:** No cross-backend collective registry, scheduler, or new contract identity namespace. Existing EC16 `target.kind="tp_topology"`, EC05 `ContractEvidenceRef`, build/execution attestation, and PGC12's existing collective trace can express the decision **provided the executed route is attached as provenance-bound evidence**. PKC03 remains pending until the bounded PGC12 receipt/admission test below passes. This is an evidence-identity correctness gate, not a proposed throughput patch. No active QFP49, QFP50, QFP41, Radiance or 1330 implementation/experiment is touched.

### Pinned implementation and exact ownership

- `tools/bigcherry/experiment/contract.py::TARGET_KINDS` already includes `tp_topology`; `ExperimentContract.contract_hash` hashes contract semantics, not a runtime route. `ContractEvidenceRef` stores contract ID/hash, optimization ID, role, workload/model and optional boundary dimension/value; it does **not** store selected or successfully completed collective provider, root, wire, switch threshold, RCCL protocol or per-call fallback. Do not extend the runtime kernel-family enum or conflate contract hash with actual execution.
- `tools/bigcherry/experiment/attestation.py` already owns measured-process GPU/backend/device identity. Architecture alone cannot distinguish the two physical gfx1100 devices. This establishes **which devices ran**, not which collective provider completed.
- `tools/bigcherry/experiment/bundle.py::ALLOWED_ENV` captures HIP tuner/visibility variables, not the `--allreduce` CLI arguments. Bundle provenance cannot be treated as proof of selected provider, threshold, wire or root without an explicit launch-arguments/receipt binding. Do not add an unrelated global environment-based identity scheme.
- `engines/llamacpp/patches/0860_allreduce_provider_cli/patch.py::ggml_backend_comm_set_config` accepts requested provider/wire/switch bytes (96 KiB default). `0840_hybrid_allreduce_dispatch/patch.py::ggml_backend_cuda_comm_try_allreduce_hybrid` snapshots `adaptive_switch_bytes` in the communication context and sets `provider_name="internal"` or `"rccl"` **before** invoking that provider. A failed internal attempt can fall through to RCCL; a failed provider can fall back to Meta. Therefore a requested setting, predicted `prefer_internal`, or mutable `provider_name` is **not a successful-provider receipt**.
- `docs/planning/active/patching-gpu-collectives/PGC12.md` already identifies that `1277_ar_size_trace` reports predicted routing rather than completed routing and owns the missing phase/ubatch/provider-completion trace. PGC09 owns production crossover; PGC10 owns 3-GPU/root; PGC11 owns wire precision; PGC13 consumes traces for offline threshold decisions; RRVP01/02 own future HIP/Vulkan stack/capability identity (Vulkan implementation remains paused). No second owner is needed.

### Concrete ambiguity and cheapest discriminator

Two executions may have the **same** contract hash, EC05 sidecar, model, architecture and nominal 96 KiB threshold, yet take different paths: (A) internal succeeds on a 20 KiB decode collective; (B) internal rejects and RCCL or Meta completes. The existing sidecar does not distinguish these. Likewise a requested root=0 is not evidence that a root3 path completed, and a missing record is not proof of zero collective work. A synthetic host identity fixture confirmed that retaining only contract/role/model aliases these cases; including completed-provider, root, threshold, wire and terminal outcome separates them. This is an illustrative fixture, **not** a repository or GPU test.

### Representability and fail-closed admission matrix

| Fact | Authoritative owner and evidence | Rejection condition |
| --- | --- | --- |
| Contract hypothesis, scope, lane | EC16/EC05 contract hash and sidecar | Missing/mismatched contract ID/hash/role |
| Backend, physical device set, architecture, P2P capability | Measured-process attestation + PHA03 topology | Intent-only GPU ordinals, unknown or mismatched devices |
| Requested provider/wire/threshold/root | 0860 launch arguments and comm-context snapshot; root from existing root3 owner | CLI/config absent, malformed, or not bound to measured process |
| **Completed** provider, fallback and bytes | PGC12 per-call post-success trace, with attempted/rejected route and explicit Meta fallback | Predicted route, pre-call label, missing terminal result, or uncorrelated trace |
| RCCL protocol and transport | Effective RCCL environment/config + completed-provider trace | Requested protocol without effective confirmation |
| Negative result | Explicit zero eligible calls or explicit rejected attempts with complete request scope | Silence, process exit, timeout, or missing telemetry presented as a negative result |

No new persistent schema is authorized by PKC03. PGC12 should emit/attach an existing run-artifact receipt keyed to the same process/session/build, request, collective call, rank set, graph/ubatch and source evidence as the lane. Record monotonic per-call sequence, bytes, selected/attempted route, actual successful route, terminal fallback/error, root and wire **only where applicable**; unsupported fields are `unknown`, never inferred. Retain the original trace artifact digest under the existing evidence-reference machinery. Consumer-side validation must refuse a success claim when the receipt is missing, mismatched, partial, or reports no completed target-provider calls. Never count attempted routes as executed or treat provider rejection after output mutation as safe fallback.

### Bounded validation / terminal disposition

1. **Host only:** in existing PGC12/experiment tests, exercise same contract+model with (a) internal success, (b) internal rejection then RCCL success, (c) RCCL rejection then Meta fallback, (d) requested root different from executed root, (e) missing/duplicate/out-of-order/foreign-request receipt, (f) zero-call and failed-call negative evidence, and (g) two physical gfx1100 locators. Verify the EC05 sidecar remains stable while executed-route receipts differ, and invalid evidence cannot promote.
2. **No new hardware queue:** consume an already qualified dual-XTX PGC09 control and an existing mixed 3-GPU PGC10/PGC12 trace only when available. Compare observed provider and exact transferred/reduced bytes against graph work; enforce full-vocabulary/greedy/MTP and repeated-request correctness. Do not borrow QFP49's actively measured cross-card lane or schedule overlapping jobs.
3. **Close PKC03** when existing EC05 evidence references plus the PGC12 receipt can represent every row above and invalid/missing routes fail closed. If one row cannot be represented, assign its minimal missing field to PGC12 (executed collective), PHA03 (topology), RRVP (stack), or existing contract/evidence validation; do not create a collective registry or a new identity system.

### Measured context and external mechanisms

Historical BigCherry dual-XTX PGC09 evidence: exact-F32 adaptive decode ~+4.1% versus RCCL in a specific Q8_0/MTP configuration; that is **not** a PKC03 speedup and does not qualify 3-GPU/no-P2P behavior. Current llama.cpp `ggml/src/ggml-cuda/allreduce.cu` remains the implementation baseline. vLLM's custom AllReduce gates P2P and supports a distinct completed communication path; its October 2026 P2P gating RFC #51513 highlights conflicting per-consumer trust decisions. SGLang's `parallel_state.py::_resolve_outplace_all_reduce_method` orders eligible providers, illustrating why a configured preference cannot substitute for observed completion. These are identity/correctness mechanisms only; no AMD performance transfer is claimed.

References: [llama.cpp allreduce.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/allreduce.cu), [vLLM custom AllReduce](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/custom_all_reduce.py), [vLLM P2P gate RFC #51513](https://github.com/vllm-project/vllm/issues/51513), [SGLang provider selection](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/parallel_state.py).

**Audit execution:** 9/9 connector-source static assertions and 4/4 illustrative host identity cases passed. No repository pytest, compiler/build, GPU execution, or benchmark was run. The 12-hour capability exclusion and novelty checks passed: PKC03 last semantic change 2026-09-10 (2026-10-08 line-ending-only rewrite); no PKC03 PR or queued experiment found. Independent recent QFP17/1330, QFP41/1356, QFP48/49, QFP50/1359, 1357 and Radiance work is excluded.

## Change Log

- 2026-09-09T10:51:51.425061+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:18.957220+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.043490+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.677197+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:52:43.052122+00:00 (updated-by): Updated: section:description, section:steps, section:acceptance_criteria, section:notes
- chg_20260910_005948_legacy-planning-folders-now-co_1240
- 2026-09-10T00:59:49.018109+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:39:37.092116+00:00 (updated-by): Updated: section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_023948_hip-collective-and-kernel-cove_9788
- 2026-09-10T02:39:48.417756+00:00 (updated-by): Updated: section:ledger-events
