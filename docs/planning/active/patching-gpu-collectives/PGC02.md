---
id: PGC02
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-09-09T10:47:53.449711+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Complete qualification and promotion decision for landed N-way internal AllReduce

## Description

Complete qualification and promotion decision for landed N-way internal AllReduce with explicit negative evidence retention.

## Steps

1. Reconcile patch 1244 metadata and SUMMARY with current evidence.
2. Run soak and root/topology/size matrix across supported N-way regimes.
3. Compare against the correct native baseline and record correctness before performance.
4. Retain patch 1245 as negative/dispositioned evidence; do not reopen it.
5. Record K-COLLECTIVE-03/04/05 as conditional/deferred unless evidence changes their scope.
6. Make promotion/disposition fail-closed and provenance-bound.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

patch 1244 source/metadata and SUMMARY; N-way AllReduce qualification harness; RCCL/native baseline; root/topology/size soak artifacts; patch 1245 retained negative evidence; K-COLLECTIVE-03/04/05 disposition records.

## Validation

Correctness before performance; N=3 boundary and supported N-way regimes; root/topology/size matrix; soak; native baseline; 1245 negative evidence; conditional K-item dispositions; exact patch/revision provenance.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Patch 1244 has correctness, soak, root/topology/size coverage and a provenance-bound promotion decision; patch 1245 remains retained negative evidence; conditional K items are explicitly dispositioned.

## Notes

Supersedes: GP11
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-gpu-collectives-gp11

Supersedes: GP11
Inherited constraints: RV103 and RV114 — retain N=3 qualification boundary, soak/root/topology/baseline gates, duplicate-claim reconciliation, 1245 negative evidence, and conditional K-item dispositions.
Migration: capability-rebaseline-v3-2026-09

### 2026-09-12: real-hardware N>=3 topology-coverage gate -- DEFERRED-HARDWARE (GPT-approved, req_fa5e1027390941fe)

Attempted the real N>=3 root/topology/size coverage matrix this item requires. Brutus (this project's only real GPU host) has exactly 4 GPUs across 3 different architectures: 2x RX 7900 XTX (gfx1100), 1x R9700-class (gfx1201), 1x RX 6900 XT (gfx1030). Only 2 GPUs share an architecture -- there is no real same-architecture N>=3 topology available to test on this hardware at all.

Built a real multi-arch (gfx1100;gfx1201;gfx1030) bigcherry-native composition (1001+0840+1244+1225) and ran llama-bench -sm tensor across HIP_VISIBLE_DEVICES=0,1,2 (2x XTX + the gfx1201 card). Segfaults immediately on first prompt processing, identically under GGML_CUDA_ALLREDUCE=hybrid and GGML_CUDA_ALLREDUCE=rccl (pure upstream RCCL, no BigCherry allreduce code in the path at all). gdb backtrace: crash is inside libamdhip64.so.7 (the HIP runtime itself), called from ggml_backend_cuda_cpy_tensor_async via the generic "meta" backend's cross-device tensor-copy path during graph_compute -- nothing in the stack touches allreduce.cu, 1244, or 0840.

**Mandatory discriminator (GPT's explicit condition before accepting this disposition): reproduced the identical crash on a completely stock, patch-free `bigcherry-native` build (zero extra patches) at the same pin, same 0,1,2 topology, same -sm tensor.** Same SIGSEGV signature. This proves the failure is a pre-existing HIP-runtime/upstream limitation with heterogeneous-architecture tensor-split (-sm tensor across gfx1100+gfx1201), not something introduced by 1244 or 0840, and not fixable within PGC02's own scope.

GPT's disposition (approved, with an important correction to my initial framing -- 1244 already has REAL positive N=3 evidence from its original validated topology, 2x XTX + this same R9700, so this is NOT "N=3 was never tested"):
- Same-architecture N>=3 topology coverage: **deferred-hardware** -- valid and evidenced; this host physically has only 2 same-arch GPUs, so the remaining topology/size/root qualification MATRIX (beyond the single already-proven N=3 decode point) cannot be completed here.
- Today's specific heterogeneous 3-GPU tensor-split attempt: environment/upstream-runtime blocker (HIP driver crash in generic cross-device copy), confirmed pre-existing via the stock-build discriminator above -- not a defect in 1244 or 0840. Do not reject or demote either patch on this evidence.
- GPT explicitly warned against substituting `-sm row`/`-sm layer` to manufacture "N=3 coverage" for this gate -- `-sm tensor` is the execution mode this gate is about, and a different split mode would not actually exercise 1244/0840's allreduce dispatch at all.

Remaining PGC02 gates (soak, provider/threshold telemetry for the pp1024-4096 softening, RCCL/baseline comparison) are unaffected by this and still open -- this closes specifically the "topology coverage beyond the already-proven N=3 point" sub-gate as blocked by real hardware unavailability, not skipped.

## Change Log

- 2026-09-09T10:47:53.449711+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:03:53.119748+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.771380+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.231843+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:53:05.364557+00:00 (updated-by): Updated: section:description, section:steps, section:acceptance_criteria, section:notes
- chg_20260910_005948_legacy-planning-folders-now-co_1240
- 2026-09-10T00:59:49.055704+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:38:05.053695+00:00 (updated-by): Updated: section:files, section:validation, section:acceptance_criteria
- chg_20260910_023824_the-gpu-collective-successors_5773
- 2026-09-10T02:38:24.269287+00:00 (updated-by): Updated: section:ledger-events
- chg_20260912_045133_documented-the-n-way-allreduce_7094
- 2026-09-12T04:51:33.420806+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T05:24:03.851834+00:00 (updated-by): Updated: section:notes
- chg_20260912_052450_investigated-why-real-3-gpu-n_2608
- 2026-09-12T05:24:50.656811+00:00 (updated-by): Updated: section:ledger-events
