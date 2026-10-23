---
id: BCOP197
order: 197
plan: patching-bc-optimizations
state: completed
created-at: '2026-10-14T00:04:01Z'
created-by: agent
priority: P2
work: S
---

# QFP05: gate static GGUF d2t MTP vocabulary before adaptive cache

## Description

**Discovery/change:** BigCherry 1297 currently copies the entire loaded full `model.output` to host and adds a separate trimmed draft head, retaining both GPU payloads (644.14 MiB + >=170 MiB for Q8_0 65,536 rows); it is a compute trim, not a model residency trim. Pinned b11552 Qwen4Exp lacks a `d2t` loader. Existing EAGLE3 and upstream research #25187, plus the Qwen4Exp fork `drluoto/llama.cpp@fb367b8`, demonstrate a narrow offline-GGUF `output.weight` + `d2t` mechanism. The external producer does not enforce nonnegative map IDs or special/EOG membership; do not import it unvalidated. Source-derived sizes and external gains are not BigCherry GPU measurements.

**Ownership and exclusions:** QFP05 owns the exact implementation/qualification gate; QFN01/1297 remains the validated runtime control. FMTP owns scheduling; QFP08 owns draft prefill; QFP13/QFP17 own generic TOP_K. No new allocator, scheduler, sampler or dynamic cache. Last independent QFP05 update 2026-10-05; 1297 patch content last substantively updated before the 12-hour window (2026-10-09 directory move only). No independent QFP05/1297 work, PR, hardware result or queued lane was found in the preceding 12 hours. Radiance mailbox/R9700 helper, engine patch-prefix/release and their hardware lanes remain protected and untouched. BCOP181-196 cover other capabilities; BCOP196 was previously prepared but unpublished.

**Unresolved action:** Host-validate one frozen 65,536-row offline GGUF `d2t` sidecar (byte-equal Q8_0 rows, I64 unique in-range IDs, required EOG/control tokens, exact tokenizer and map hashes); reject double-trim, malformed mapping and trunk activation before graph execution. Compare same-ID offline vs 1297 runtime, then frequency-ranked vs first-N, with full-head control, 10K/80K/160K gfx1030 MTP lanes. Record head timing, effective decode, acceptance, startup/host/device memory and greedy/KLD/repeated-request safety. Promote only if >=3% E2E gain over best 1297 static control at 10K and 80K with CI95-low >0 and <=2% long-context regression; or label a separately measured capacity-only enablement. If no qualified model/trace, `AWAIT_D2T_SIDECAR`. Reject dynamic vocabulary unless later exact-pin exposed head/coverage evidence clears >=3% E2E opportunity.

**Evidence and validation:** b11552 `src/models/qwen4exp.cpp`, `src/models/eagle3.cpp`; BigCherry 1297 `patch.py`/SUMMARY; llama.cpp issue #25187 and `avifenesh@047bfa508`; `drluoto@fb367b8` Qwen4Exp consumer and producer; SGLang `eagle_worker_v2.py` static token map. 13/13 actual-source checks and 12/12 host-only map/scatter tests passed; no GGUF conversion, HIP compile, repository pytest or GPU run. QFP05 is the technical design; this BCOP is disposition only.

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
