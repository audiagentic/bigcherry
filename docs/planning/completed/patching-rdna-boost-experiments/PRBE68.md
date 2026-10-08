---
id: PRBE68
order: 0
plan: patching-rdna-boost-experiments
state: deprecated
created-at: '2026-09-09T10:58:18.039731+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P3
---

# VK-SUB-001: Architecture-aware Vulkan submission cap

## Description

AUDIT 2026-10-07: reject the proposed architecture-aware max_nodes_per_submit table. Current llama.cpp already has two submission bounds: fixed node ceiling 100 with GGML_VK_MAX_NODES_PER_SUBMIT, and a FLOP budget in ggml_backend_vk_graph_compute(). Upstream 0bbc87b / PR #25240 already makes the FLOP budget AMD architecture/CU-aware. The old plan targets the wrong layer.

Issue #26679 measured gfx1201/RADV pp128 63.6 vs 1025.7 t/s and tg64 8.41 vs 66.05 t/s after reverting PR #26371's submission-threshold-placement change. This proves submission-boundary semantics can dominate performance; it does not establish a numeric gfx1201 node ceiling.

Sources:
- https://github.com/ggml-org/llama.cpp/commit/0bbc87b163ff7826656b1024dac5703e3f2bd6b6
- https://github.com/ggml-org/llama.cpp/pull/26371
- https://github.com/ggml-org/llama.cpp/issues/26679

## Ownership and implementation gate

PRBE68 owns qualification of existing Vulkan submission policy only. Device init owns the hard node ceiling; ggml_backend_vk_graph_compute() owns FLOP batching; PRBE62 owns queue-family selection. Do not add another scheduler, architecture table, queue selector, or batching state machine.

No patch first. Same binary, sweep GGML_VK_MAX_NODES_PER_SUBMIT=1,16,32,64,100,200 on gfx1201 RADV and gfx1100 RADV; gfx1030 is a safety control when available. Run pp512/2048, tg128, production MoE/Flash-Next, MTP depth 3/7 where applicable, shallow/~80K and one >=160K gfx1201 stability lane.

Record architecture/driver/CU count, E2E throughput, submission count, nodes/submission, estimated FLOPs/submission, and which existing predicate closed each submission: node cap, FLOP cap, last-node, or almost-ready. Record DeviceLost/timeout/corruption.

Only if existing logs cannot identify the closing reason, add a disposable diagnostic at the existing submit predicate. It may count those four reasons; it must not change thresholds, ordering, synchronization, queue choice, or graph construction.

## Decision rules

- If the node-cap sweep changes neither firing distribution nor E2E performance by >=1%, close PRBE68.
- If one value achieves CI95-low-positive >=3% in a production lane with <=1% regression in all controls and no stability loss, prefer the existing env override; do not add an architecture table.
- Reconsider a code default only when first-party evidence proves the node bound, rather than the FLOP bound, causal for a stable architecture+driver class and the rule survives long-context and repeated same-process runs.
- If the causal seam is FLOP-bound placement, close/transfer the remainder to the existing Vulkan batching seam rather than expanding PRBE68.

Correctness requires deterministic/greedy output parity, expected node/work counts, MTP acceptance/work accounting where applicable, repeated same-process requests, long-context stability, and zero DeviceLost/timeouts. Faster because work disappeared is failure.

## Acceptance criteria

Terminate as close/no-op, deployment configuration using the existing override, or a bounded upstream/default patch only after the causal gate. No architecture-aware node-cap table without first-party causal evidence.

## Notes

2026-10-07 audit: PR #25240 already owns architecture/CU-sensitive submission safety through the FLOP budget. #26679 is evidence about threshold-placement sensitivity, not a numeric node-cap requirement. No BigCherry hardware run, build, or prototype was performed by this audit.

## Change Log

- 2026-10-08 (triage): Terminal planning state `deprecated` records a **deferred-hardware** decision; no local qualifying hardware/driver measurement and no performance rejection. Reopen under a new evidenced platform scope.

- 2026-10-08 (triage): Parked as deferred-hardware; Vulkan submission-node-cap design rejected in PRBE68 audit: upstream 0bbc87b/PR #25240 already has architecture-aware FLOP bound, no measured numeric gfx1201 node ceiling. No Vulkan lab lane; park as hardware-inapplicable, not as a new HIP optimization. Reopen only with the named platform and its measured evidence.

- 2026-09-09T10:58:18.039731+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:29.747464+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.434952+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.268296+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:18:48.866865+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031907_repaired-the-vulkan-submission_2651
- 2026-09-10T03:19:07.580884+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:32:24.544952+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:48:25.374063+00:00 (updated-by): Updated: section:description, section:steps
- 2026-09-24T04:48:28.309628+00:00 (updated-by): Updated: section:notes
