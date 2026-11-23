---
id: VTP53
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-23T21:15:28.492767+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P2
---

# Delta-net verify pass: update the recurrent state once per accepted prefix instead of per drafted token

## Description

The delta-net decode update is 2.29 ms of rank 0's traced verify pass (VALU recurrence, 184-252 VGPRs per instantiation, no matrix unit). In a speculative step the verify pass runs the recurrence over the drafted tokens for every sequence, and the state after the rejected tokens is discarded or restored. If the engine updates and snapshots the full per-sequence state for each drafted token, most of that work (state read, update, write) is wasted on the rejected suffix.

**What we will try.** Read how the engine commits delta-net state in a verify step (first: static, from our source and the trace: per-token state writes, snapshot copies, rollback), then, if per-token state traffic exists, keep the state in registers or LDS through the drafted tokens of one sequence and write it back once for the accepted prefix, with intermediate states only where acceptance can stop. Result must match the per-token recurrence bit for bit because the arithmetic order per token is unchanged.

**Expected.** Unknown until step 1; the bound is the state traffic share of the 2.29 ms (state is large: heads x key dim x value dim per layer).

## Steps

1. Static read: per verify step, bytes of state read and written per layer per sequence versus the minimum (one read, one write of the accepted state); share of the 2.29 ms.
2. If the gap is above 0.5 ms, design the change in our own kernel and engine plumbing; scratch kernel with byte identity against the current outputs for tokens 1..depth+1.
3. Cold replay, then ABBA at 6 and 8 streams with acceptance and temp-0 text unchanged.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- State and outputs byte-identical to the current path at every accepted length 0..depth+1, including rollback.
- ms/step ABBA.

## Effort & Risk

Large: touches engine state management and the kernel; correctness at every accepted length is the main risk.

## Standards



## Acceptance Criteria

- NO-GO at step 1 if the state traffic bound is below 0.5 ms; otherwise ships only with a measured step gain above the seed spread.

## Notes



## Change Log

- 2026-11-23T21:15:28.492767+00:00 (created-by): Created by agent
- 2026-11-23T21:16:03.089284+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
