---
id: PNRO05
order: 0
plan: patching-nasone-rdna-optimizations
state: in_progress
created-at: '2026-09-09T10:52:33.316789+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P0
---

# Chunk GDN MTP prefill prefix while preserving snapshot tail

## Description

TODO, NOT-READY (rescoped). Verified via patch.py: 1254 only defines the eligibility predicate `bigcherry_nro05_mtp_prefix_candidate()` -- prefix execution, temporary state ownership, the K-token sequential tail, snapshot publication, and fallback are all unwired (nothing actually runs the prefix/tail split). Qualify the MTP-specific GDN prefix/tail composition independently from PNRO04's raw kernel once wired.

## Steps

1. Require PNRO04 and enable only non-KDA, K>1, n_seqs==1, sufficiently long supported sequences (existing predicate).
2. Extend ggml_cuda_op_gated_delta_net_impl() (verify exact function name/location at implementation time) to actually run [0, n_tokens-K) through PNRO04's chunked kernel into pool-owned temporary state, then exactly K tokens through the existing stock sequential recurrence -- this wiring does not exist today.
3. Publish output/snapshot slots ONLY after both stages succeed; preserve original state/output until success so a rejection can safely rerun the full sequential path.
4. Add a prefix-tail activation marker (BIGCHERRY_PATCH_TRACE-gated) -- none exists today.
5. On synchronous rejection/failure, use original full-sequential path.
6. Validate every snapshot slot, final recurrent state, output/logits, threshold edges, K=2/3/5/8, sequence count, graph capture/replay, and pool lifetime.
7. Include FP32 chunked-prefix control where feasible to separate orchestration errors from BF16 kernel error.
8. Measure prefix savings versus transition/allocation overhead and MTP acceptance.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

patches/1254_nro05_gdn_mtp_prefix_tail; orchestration/snapshot path; snapshot fixtures; MTP evidence

## Validation

Patch mechanics: `PYTHONPATH=tools python -m bigcherry patch-lint patches/1254_nro05_gdn_mtp_prefix_tail`; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay 1254_nro05_gdn_mtp_prefix_tail --source bigcherry-tuning --requires 1253_nro04_gfx1100_bf16_chunked_gdn`; package pytest offline. Per-slot/final-state parity, nonqualifying-shape fallback, threshold+/-1, K=2/3/5/8, single/multi-sequence, launch rejection, pool reuse, graph replay fixtures; FP32 chunked-prefix control to separate orchestration vs kernel error. Hardware (Brutus, gfx1100): `python -m bigcherry.patch.validation_campaign --overlay 1254_nro05_gdn_mtp_prefix_tail --requires 1253_nro04_gfx1100_bf16_chunked_gdn --arch gfx1100` measuring prefix savings vs transition/allocation overhead and MTP acceptance.

## Effort & Risk



## Standards

Snapshot/state restoration correctness; preserve sequential fallback; preregister tolerances inherited from PNRO04.

## Acceptance Criteria

Every snapshot and final state matches registered tolerance; all nonqualifying shapes retain sequential path; MTP acceptance is not regressed; positive E2E effect is statistically supported.

## Notes

Supersedes: NRO05
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro05

Supersedes: NRO05
Inherited semantic scope: preserve prefix/tail boundary, snapshot ordering, failure fallback, state lifetime, and acceptance-aware qualification.
Migration: capability-rebaseline-v3-2026-09

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH. patches/1254_nro05_gdn_mtp_prefix_tail exists, state=untested, depends on PNRO04's patch 1253. No upstream equivalent. Disposition: validate/qualify existing patch; no GPT design needed.

2026-09-24 GPT review req_215c89d0b13a4bb7 applied: verified 1254 only defines the eligibility predicate with no prefix execution, temp state, tail, publication, or fallback wired -- added the required extension to ggml_cuda_op_gated_delta_net_impl(), the success-gated publication order, and a required activation marker (none existed).

2026-09-25 (d423644d): 1254 ports block 02's MTP prefix path on top of 1253: for K>1, single sequence, n_tokens > K+64, BF16 chunked GDN on the first n_tokens-K tokens then sequential on the last K (snapshot slots exact); RDNA3/RDNA4 S_v==128 only (fork's fp32 fallback not ported). Marker patch=1254_nro05 path=gdn_mtp_prefix_bf16. Contract NRO05-GDN-MTP-PREFIX added (mtp_verify on qwen35b MoE+MTP). Producer still to write (needs the MTP server lane).

## Change Log

- 2026-09-09T10:52:33.316789+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:52.793774+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.071633+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.713426+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:26:22.478825+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260910_022800_five-nasone-successor-plans-no_4030
- 2026-09-10T02:28:00.303796+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:26:15.579509+00:00 (updated-by): Updated: section:validation, section:notes
- 2026-09-24T04:49:16.618864+00:00 (updated-by): Updated: section:description, section:steps, section:notes
- 2026-09-25T04:23:42.505423+00:00 (updated-by): Updated: section:notes
- 2026-09-25T04:23:45.397057+00:00 (state-transition): State: pending → in_progress
