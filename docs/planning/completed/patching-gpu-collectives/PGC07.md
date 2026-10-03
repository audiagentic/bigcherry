---
id: PGC07
order: 0
plan: patching-gpu-collectives
state: superseded
created-at: '2026-09-29T09:34:08.605811+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P3
work: M
---

# Additional wire formats (bf16/fp8) on --allreduce-wire

## Description

Evaluate bf16/fp8 wire formats only after native/q8 and the target dual-gfx1100 candidate stack have correctness/work-equivalence evidence. New code; depends on PGC04.

## Steps

1. Do not use the current 1241+1206+1245 combo result as a performance baseline: `combo-ab1` is VOID because greedy output and MTP acceptance differ.
2. Establish a correctness-clean, work-equivalent baseline composition first.
3. Require activation evidence for every behavior-affecting candidate; 1245 currently has no marker and must gain `BIGCHERRY_PATCH_HIT patch=1245_gp11 path=mmvq_fusion_q8_0_ncols6` before reuse.
4. Compare native, q8, then bf16/fp8 only if prior gates pass.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Full-vocabulary MTP logprobs <=5e-4 where required; greedy-token parity; drafted/accepted counts and acceptance parity; fixed-work `llama-bench`; order-balanced server A/B via `tools/lab/native-vs-patched/server-ab-*.json`. Preserve per-arm activation logs and raw pair records.

## Effort & Risk



## Standards



## Acceptance Criteria

No new wire format proceeds on a numerically divergent or non-work-equivalent baseline; no validation/sign-off claim from planning alone.

## Notes

2026-10-04 superseded by PGC11 (single owner item for all AllReduce wire formats: f16/bf16/fp8/q8/WHT). Constraints carried over: wire formats only after native/q8 correctness/work-equivalence evidence; depends on PGC04.

## Change Log

- 2026-09-29: added candidate-stack correctness/work-equivalence prerequisites.
- 2026-09-29T09:34:08.605811+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260929_135722_allreduce-methods-are-now-sele_7144
- 2026-09-29T13:57:35.114933+00:00 (updated-by): Updated: section:ledger-events
- chg_20260929_215656_dual-xtx-27b-q8_0-plain-decode_1707
- 2026-09-29T21:57:11.604858+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-03T15:22:04.205970+00:00 (updated-by): Updated: section:notes
- 2026-10-03T15:22:19.922468+00:00 (state-transition): State: pending → superseded
