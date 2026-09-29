---
id: PGC07
order: 0
plan: patching-gpu-collectives
state: pending
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

## Validation
Full-vocabulary MTP logprobs <=5e-4 where required; greedy-token parity; drafted/accepted counts and acceptance parity; fixed-work `llama-bench`; order-balanced server A/B via `tools/lab/native-vs-patched/server-ab-*.json`. Preserve per-arm activation logs and raw pair records.

## Acceptance Criteria
No new wire format proceeds on a numerically divergent or non-work-equivalent baseline; no validation/sign-off claim from planning alone.

## Change Log
- 2026-09-29: added candidate-stack correctness/work-equivalence prerequisites.
- 2026-09-29T09:34:08.605811+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260929_135722_allreduce-methods-are-now-sele_7144
- 2026-09-29T13:57:35.114933+00:00 (updated-by): Updated: section:ledger-events
