---
id: PRBE66
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:58:09.575084+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: null
---

# HIP-GRAPH-001: Selective HIP graph bypass for unstable FA shapes

## Description

TODO, fallback-only path per item's own preference (PRBE67's recapture fix is primary if its root-cause hypothesis is confirmed). Real gap confirmed: b11126's `ggml_cuda_graph_get_key` (ggml/src/ggml-cuda/ggml-cuda.cu) keys the graph cache on `cgraph->nodes[0]` only -- no shape/topology awareness -- so a narrow eligibility bypass (skip graph capture entirely for the failing kernel family/depth) is a viable, simpler fallback if PRBE67's recapture-cost turns out to be unbounded or the topology-change hypothesis is not confirmed.

## Steps

1. This item is gated on PRBE67 step 1's reproduction/instrumentation: only implement this bypass if PRBE67's recapture approach is rejected (unbounded recapture cost, or root cause turns out not to be topology staleness).
2. If proceeding: add a narrow eligibility predicate in the graph-capture decision path (near `ggml_cuda_graph_set_enabled` / the `USE_CUDA_GRAPH` block in `ggml_cuda_graph_optimize` and the main capture loop in ggml-cuda.cu ~line 4186-4400) that disables `use_cuda_graph` specifically when the current cgraph contains a FLASH_ATTN_EXT node with `ne[1]` (KV length) above a measured-unsafe threshold on gfx1201, gated so RDNA3/other architectures and short-context FA are entirely unaffected.
3. Verify no crash/output drift on the previously-failing 100K+ shape with capture now skipped (falls back to direct eval, `graph_evaluated_or_captured` path already exists for that in the same loop) and measure the resulting PP/TG cost (graph capture's speedup is lost only for the narrow bypassed shape).
4. Hand off root-cause instrumentation/evidence to PRBE67 regardless of which item ships, since PRBE67 remains preferred if provably robust.

## Detailed Solution & Technical Design

Fallback design: a pure eligibility gate, no new caching/fingerprint machinery. Simpler and lower-risk than PRBE67's recapture but permanently forfeits the graph speedup for the bypassed shape range instead of only paying a recapture cost at topology-change boundaries.

## Code Samples & Guidance

Real gating precedent (b11126, ggml-cuda.cu ~4594-4598): `ggml_cuda_graph_set_enabled(cuda_ctx, graph_key)` already exists as a per-key enable/disable decision point -- the bypass predicate for this item should extend that same decision, not create a parallel mechanism. Exact anchor text must be re-verified by reading the full body of `ggml_cuda_graph_set_enabled` (grep + read before writing patch.py) -- not reproduced here since only the call site, not the definition, was inspected this session.

## Files

ggml/src/ggml-cuda/ggml-cuda.cu; patches/12xx_hip_graph_fa_bypass/{patch.toml,patch.py,SUMMARY.md} (only authored if PRBE67 is rejected).

## Validation

No crash, output parity vs graphs-disabled reference on the previously-failing shape; PP/TG and graph-capture/bypass-event telemetry; hardware run on gfx1201/R9700 via python -m bigcherry.patch.validation_campaign (not run here).

## Effort & Risk

S-M; simpler than PRBE67 but permanently gives up graph speedup for the bypassed range -- acceptance criteria in the original item already state PRBE67 is preferred when its recapture is robust.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Use the narrowest bypass that eliminates reproducible failure without broad graph regression; prefer PRBE67 when robust recapture fixes the root cause.

## Notes

Supersedes: RD83
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd83

2026-09-24 relevance at b11126: TODO, conditional/fallback to PRBE67 -- same underlying gap (stale-key graph cache, no topology awareness) supports either fix; this item should only be implemented if PRBE67's approach is rejected by its own step 1 evidence. GPT design request: gateway rejected all submissions this session (VAL-AGW-025 / EXT-GPTAUTO-003); plan authored directly -- no GPT request id.

## Change Log

- 2026-09-09T10:58:09.575084+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:15.987575+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.426062+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.254495+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:17:45.266636+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031805_repaired-four-more-graph-and-v_2834
- 2026-09-10T03:18:05.350207+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:34:02.491568+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
