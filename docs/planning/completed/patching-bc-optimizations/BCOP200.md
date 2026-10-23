---
id: BCOP200
order: 200
plan: patching-bc-optimizations
state: deprecated
created-at: '2026-10-14T03:13:06Z'
created-by: agent
priority: P2
work: S
---

# PRBE115: reject raw-F32 MTP-width widening; require exact Q8_1 producer

## Description

**Discovery:** 1301's widened raw-F32 MMVQ path was rejected after neutral width2/3 and -6% to -12% width5 throughput; its FMA arithmetic cannot preserve stock Q8_1 verify numerics. Stock b11552 quantizes Q8_1 once per call, then reuses it across weight rows. Validated 1241 width1 is distinct and requires 0600. No new GPU measurement.

**Owners:** PRBE113/BCOP180 owns current-pin width2–5 receipt-first cost attribution. PRBE115 owns only a conditional numerics-preserving candidate; 1241 owns width1, 0600 geometry, PRBE12/1206 reshape/add, QFP48 MMQ prefill. Do not introduce another cache, scheduler, dispatcher, profiler or allocator.

**Protection:** No independent PRBE115/1301 work or queued experiment identified within 12h; Radiance R9700 helper, mailbox, protected expert map and patch/release work excluded. Recent BCOP190–199 topics avoided.

**External:** llama.cpp issue #25618 documents quantized N=1/N>1 speculative divergence (different backend); PR #30021 tunes MMQ, not MMVQ. vLLM FP8 fusion and SGLang GGUF dispatch are not directly portable to AMD GGUF Q8_1.

**Disposition:** Gate on PRBE113 terminal per-width receipt. If optimistic exposed removable E2E latency <2.9126%, close BELOW_CEILING. Otherwise first prove exact stock Q8_1 bytes, lifetime and graph compatibility; if impossible close NO_EQUIVALENT_PRODUCER. Only one default-off existing-owner candidate with MTP acceptance/logit/greedy parity and paired gfx1100/gfx1201 CI95-low >=3% E2E, controls <=1% regression. Source assertions 12/12; illustrative host arithmetic 6/6; no repository pytest, patch composition, HIP build or GPU run.

## Steps


## Detailed Solution & Technical Design


## Code Samples & Guidance


## Files


## Validation


## Effort & Risk


## Standards


## Acceptance Criteria


## Notes


## Change Log

- 2026-10-23: state 'done' (invalid) corrected to deprecated
