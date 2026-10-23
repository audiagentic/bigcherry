---
id: BCOP170
order: 170
plan: patching-bc-optimizations
state: completed
created-at: '2026-10-12T20:05:37+00:00'
created-by: agent
priority: P2
work: S
---

# QFP35: reconcile 1355 HC_POST screening and gate recomputation

## Description

**Discovery:** PR #54 merged on 2026-10-09. Patch 1355 remains untested/experiment-only despite three historical ABBA rounds (~+1% pooled decode, identical greedy, no prefill separation). Its kernel expresses one embedding-invariant sigmoid per output thread; source-level factor n_embd, actual ISA/SFU cost unknown. 1344 is already validated/production; standalone 1311 is absorbed by 1307. Earlier rebase/1347 instructions are stale.

**Owner:** QFP35 / existing 1355. Do not duplicate 1313 scale-act fusion, 1344 grid indexing, 1307 Q8_1 or 0910 profile handling. vLLM PR #56633 is a distinct delayed-PRE boundary; no port.

**Bounded action:** Inspect actual gfx1100/gfx1201 SFU instructions and paired producer+HC_POST critical-path time. Only if repeated work survives and dominates, try a disposable wave-broadcast discriminator without new cache. Before production require backend-op/logits, fixed-draft MTP, multi-request/graph and mixed no-P2P correctness plus four independent paired ABBA sessions with positive decode CI95-low and no control regression; otherwise retire. No hardware job queued.

**Validation:** 36/36 host scalar reference cases; source/recipe inspected. No repository pytest, HIP build, GPU run or new measured gain.

## Steps


## Detailed Solution & Technical Design


## Code Samples & Guidance


## Files


## Validation



Source-static and host-only checks recorded in this item's disposition/evidence text; no hardware run was required for this closure.

## Effort & Risk


## Standards


## Acceptance Criteria



The disposition and ownership decisions described above are recorded; no further implementation is owed by this item.

## Notes


## Change Log

- 2026-10-23: state 'done' (invalid) corrected to completed
