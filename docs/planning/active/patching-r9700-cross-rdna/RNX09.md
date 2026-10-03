---
id: RNX09
order: 9
plan: patching-r9700-cross-rdna
state: pending
created-at: '2026-10-03T01:33:32.238212+00:00'
breadth: ''
skill: advanced
created-by: codex
work: L
priority: P1
---

# R9X09 — Cross-RDNA integration, ablation, and promotion

## Description

Turn promising R9X experiments into a coherent production set for RDNA4, RDNA3, and RDNA2. Own combined-stack testing, conflict resolution, promotion/rejection, documentation, and removal of redundant experiments without hiding regressions in averages.

## Steps

- Re-resolve actual patch numbers from the branch rather than treating 1300–1312 as ownership locks.
- Test gfx1201, gfx1100, gfx115x separately, gfx103x when available, and mixed 2x gfx1100 + gfx1201/no-P2P systems.
- For each survivor measure baseline, alone, full stack, and pairwise interacting groups with median/geometric throughput, variance, ABBA where practical, and kernel census.
- Apply correctness gates for backend tests, token parity/quality, MTP, tensor split zero semantics, fallback, reset/join/leave, graph replay, and long-run state.
- Assign promote, keep experimental, subsumed, or reject with evidence; update README and superseded PNRO/RDNA docs, then produce the one-page production matrix.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

- docs/planning/active/patching-r9700-cross-rdna/README.md
- docs/planning/active/patching-r9700-cross-rdna/RNX*.md
- docs/planning/active/patching-nasone-rdna-optimizations
- docs/planning/active/rdna-boosts

## Validation

Independent combined-stack and regression evidence for every enabled candidate, exact fallback for unsupported cases, and production matrix mapping GPU/topology to enabled patches, thresholds, expected delta, and disable controls.

## Effort & Risk



## Standards

Promotion is architecture-specific; shared paths must show no material regression on selecting generations.

## Acceptance Criteria

- No candidate is promoted on microbenchmarks alone.
- No duplicate default implementation remains for one role.
- Campaign completion is reconstructable from committed evidence rather than chat history.

## Notes

Original source alias is R9X09. Depends on candidates RNX02–RNX08, RNX10, and RNX11 for custom collective candidates.

Verbatim legacy source retained during R9X→RNX migration:

# R9X09 — Cross-RDNA integration, ablation, and promotion

Status: planned
Patch: none by default
Depends on: completed candidates from R9X02–R9X08 plus R9X10; R9X11 resolved for custom collective candidates; R9X03 may yield both 1302 and 1312



GPT design pass (req_83e7cdc000be4b9e): sequence work as RNX01 mapping, RNX02 existing-Q8-vector dispatch experiment, RNX04 HC raw-gate microfusion after shape census, then RNX08 H2D attribution. Do not start replacement RNX03 code, RNX07 WHT, RNX06 expert-cache, or a custom collective implementation.

## Goal

Turn individually promising R9X experiments into a coherent production set for RDNA4, RDNA3 and RDNA2. This item owns combined-stack testing, dispatch conflict resolution, promotion/rejection, documentation, and removal of redundant experiments. It must not hide a regression by averaging unrelated wins.

## Inputs

Candidate patch slots are plan-time `1300`–`1312`; re-resolve actual numbers from the branch. Include existing production/evaluated dependencies that candidates extend, especially 1237/1241/1253/1254/1265/1270/1273/1274/1278/1291/1292 and relevant PNRO items.

## Required matrix

Hardware classes:
- RDNA4/gfx1201 (R9700-class source target).
- RDNA3/gfx1100 (7900 XTX-class production target); gfx115x separately when available because RDNA3.5 dispatch differs.
- RDNA2/gfx103x when available.
- Mixed tensor-split systems, especially 2x gfx1100 + gfx1201 and no-P2P/SHM collective cases.

Workloads:
- Qwen4Exp/Flash-Next production quant(s), single-token decode, MTP depths used in production, 47K-class and long-context decode, pp32/128/512/4096+, 1/2/3-rank tensor split.
- Operator microbenches only as explanatory evidence, never the sole promotion gate.

## Ablation protocol

For N surviving patches, measure baseline, each patch alone, and the full stack. For interacting groups add pairwise A/B: GDN decode+short-prefill, attention+QSA, router+HC+shared-expert, cache+MoE GEMM, collective+WHT/fixed-grid. Report median/geometric throughput and run-to-run variance; use interleaved ABBA where practical. Record kernel census/launch-count changes so a tok/s change can be attributed.

If two patches optimize the same dispatch, prefer the simpler patch or a unified selector. If one makes another redundant, mark the redundant patch rejected/subsumed rather than retaining dead branches. Re-tune thresholds after combination because crossover points can move.

## Correctness gates

- Existing backend/unit/contract tests pass.
- Greedy output/token parity for exact paths; documented numerical tolerance and quality suite for deliberately lossy formats/wires.
- MTP acceptance behavior remains statistically/effectively equivalent unless the numerical change is explicitly approved.
- Tensor split has correct rank ownership, zero contribution for uncomputed meta tensors, and no hidden synchronization/deadlock.
- Unsupported arch/shape/type uses the previous path.
- Repeated server load/unload, context reset, dynamic request join/leave, graph capture/replay and long-run generation show no lifetime/state corruption.

## Promotion policy

For each candidate set one disposition: `promote`, `keep experimental`, `subsumed`, or `reject`, with evidence links. Promotion requires a repeatable E2E benefit on its declared hardware scope and no material regression on shared-path generations. Architecture-specific wins may promote behind exact capability gates even if another generation rejects the same kernel.

Update the R9X README with final patch numbers/states and cross-links. Update existing PNRO/RDNA plan docs when R9X supersedes or consumes an item. Do not leave two default implementations for the same role.

## Final handoff

Produce a one-page production matrix: GPU generation/topology -> enabled R9X patches -> thresholds -> expected delta -> fallback. Include all env/config controls used for emergency disable. The campaign is complete only when an agent can reconstruct why each patch is enabled from committed evidence without relying on chat history.

## Change Log

- 2026-10-03T01:33:32.238212+00:00 (created-by): Created by codex

## Reviews

- RV4202
- 2026-10-03T01:39:17.288318+00:00 (updated-by): Updated: section:notes
- 2026-10-03T02:18:40.830999+00:00 (updated-by): Updated: section:notes
