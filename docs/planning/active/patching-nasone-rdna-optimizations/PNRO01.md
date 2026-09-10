---
id: PNRO01
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:52:01.168387+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P0
---

# Q8_0 wire format for the internal HIP AllReduce

## Description

Evaluate and qualify the Q8_0 wire-format portion of nasone commit e06dcf630 on top of validated internal AllReduce. This item owns only FP32→Q8_0 quantization, wire-size accounting, Q8 dequant-add, threshold selection, and exact/BF16 fallback; residual fusion is PNRO02 and P2P transport is PNRO03.

## Steps

1. Freeze source e06dcf630 and BigCherry pin b10705; re-audit ancestry after pin bumps.
2. Implement block-Q8_0 quantization, logical/wire byte accounting, Q8 finish/dequant-add, opt-in threshold override, and activation counters without importing residual fusion/P2P/graph changes.
3. Preserve exact-FP32 behavior when disabled and keep copy-engine/chunked paths intact; document uncovered size ranges.
4. Validate zeros, alternating signs, dynamic range, small values, non-32 tails, and independent rank inputs against CPU FP32 and exact internal reference.
5. Run model quality/error and greedy-divergence gates before timing.
6. Sweep decode/medium/prefill sizes and compare RCCL, exact, BF16, and Q8 arms; promote only an explicit size/topology envelope.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

patches/1250_nro01_allreduce_q8_wire; HIP AllReduce provider; static/package tests; Q8 correctness/quality/performance evidence

## Validation

Patch apply/idempotence/current-pin/dependency/static tests; per-element synthetic correctness on both ranks including tails; actual activation and wire-byte evidence; model numerical quality; paired exact-vs-Q8 sessions across decode and hostile large-prefill controls; no production inclusion before qualification.

## Effort & Risk



## Standards

PATCH_SYSTEM/authoring/validation; immutable source SHA; fail-closed applicability; exact/BF16 fallback preservation; contract identity separate from runtime selector.

## Acceptance Criteria

Q8 is inert when disabled, tail-safe, numerically within preregistered tolerances, and activation/wire accounting is proven. A statistically supported winning size envelope exists versus exact internal with no unexplained control regression; no blanket default.

## Notes

Supersedes: NRO01
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro01

Supersedes: NRO01
Inherited semantic scope: preserve source identity, opt-in policy, numerical/tail correctness, four-arm matrix, and explicit-envelope promotion gates.
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-09T10:52:01.168387+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:30.603565+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.053474+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.690115+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:25:56.314497+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260910_022800_five-nasone-successor-plans-no_4030
- 2026-09-10T02:28:00.269059+00:00 (updated-by): Updated: section:ledger-events
