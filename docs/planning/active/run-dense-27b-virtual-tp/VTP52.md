---
id: VTP52
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-23T21:15:23.514546+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P2
---

# Decode launch-count audit and chained launches for the small ops between the GEMMs

## Description

Our isolated GEMM bounds (VTP37, VTP47) are inflated by launch cost: rank 0's pass runs 288 quantised GEMM launches (13.4 ms, 46 us on average) plus a long tail of small ops (norm, rope, quantise, conv, gate, store) each its own launch. The step's real launch overhead is not known: the scratch driver adds about 10 us a launch, a replayed graph far less.

**What we will try.** (1) Audit: from the tape list count launches a pass per op family and sum the gaps between consecutive segments on rank 0 (static read of the trace; segment start/end give the gaps). (2) Where the gaps are large and the neighbouring ops are tiny and data-dependent in a fixed order (norm -> quantise -> next GEMM prologue; rope -> store -> attention prologue), fuse them into one launch or into the neighbour's prologue. Each fusion keeps the arithmetic and its order, so it is byte-identical.

**Expected.** Unknown until step 1; if gaps average 3 us over about 700 launches the ceiling is about 2 ms a step.

## Steps

1. Launch and gap census from the existing traces (2020.8.3 qwen27b-tp3, the new flashnext-4gpu trace): launches per op family, total and mean gap, per rank.
2. Rank fusion candidates by gap x count; skip any that crosses a collective or changes arithmetic order.
3. One fusion as a slice with a bit-identity gate and a cold replay, then ABBA.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- Byte identity of outputs for every fused chain; temp-0 text and acceptance unchanged.
- ms/step ABBA over the seed spread.

## Effort & Risk

Medium; the audit can end it quickly if gaps are small in the replayed graph.

## Standards



## Acceptance Criteria

- The census table is recorded; a fusion ships only with a measured step gain above the seed spread; otherwise NO-GO with the gap table.

## Notes

**2026-11-24 static audit (qwen27b-tp3, 2020.8.3 step trace, rank 0, 214 verify-type passes of 17+ rows, traced so gaps are inflated by tracing).** Pass wall 20.6 ms; classified op segments 16.4 ms over 411 segments; unclassified 1.4 ms; 2.85 ms (14%) of the wall lies outside every segment, about 7 us per segment of launch gap and bookkeeping. Small ops (average under 12 us a call) are 113-135 calls a pass (rmsnorm / quantise pairs 57, gated quantise 28, gemm_nt 21 at 15 us, prefetch 56 at 21 us). The trace mixes pass types, so the absolute counts are not a full verify pass; a per-pass-type table from the untraced replay is the next step. Ceiling for step 2 (fusion) on these numbers is the 2.85 ms gap at best, about 1 ms realistic.



Ceiling basis (2026-11-24): rests on a tape sum (2.85 ms of the traced wall outside every segment) and so on a traced run whose gaps are inflated by tracing; downgraded to a pointer. The bound has to come from an untraced replayed-graph measurement of launch gaps, or from a fused-chain A/B of one candidate.

## Change Log

- 2026-11-23T21:15:23.514546+00:00 (created-by): Created by agent
- 2026-11-23T21:15:58.192931+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-23T21:45:22.185007+00:00 (updated-by): Updated: section:notes
- 2026-11-23T22:51:45.053129+00:00 (updated-by): Updated: section:notes
