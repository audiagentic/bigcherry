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



## Change Log

- 2026-11-23T21:15:23.514546+00:00 (created-by): Created by agent
- 2026-11-23T21:15:58.192931+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
