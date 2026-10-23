---
id: BCOP193
order: 193
plan: patching-bc-optimizations
state: completed
created-at: '2026-10-14T00:05:00Z'
created-by: agent
priority: P1
work: S
---

# PNRO10: block 1261 until ctx_other backend insertion leaves GPU loop

## Description

**Discovery/change:** Pinned llama.cpp b11552 `src/llama-context.cpp` has two `backends.emplace_back(backend);` anchors. Current 1261 `insert_after, occurrence=0` injects target-device registration *inside* the draft GPU loop. Empty draft device list adds nothing; overlap with a future ordinary draft device duplicates a backend. The 2026-09-24 PNRO10 claim of a corrected placement, README/TESTING's no-marker wording, and validation.toml's unconditional-activation comment contradict current code. The const-correctness fix **is** present. No new performance measurement.

**Owner / exclusion:** PNRO10/1261 is the sole patch/technical owner; QFP43/1286 owns borrowed Meta-tensor copies, not foreign-device scheduler registration. PNRO11 target-wide mirroring remains closed. Last independent PNRO10/1261 work 2026-09-30, with no subsequent independent implementation, submission, hardware lane or queued experiment identified in the preceding 12 hours. Active Radiance GPU-mailbox/R9700 expert-helper, patch-prefix/release work and their GPU lanes are protected; no changes under those owners. Recent BCOP181–192 address other capabilities.

**Action / terminal gate:** Correct 1261's single anchor to insert **before** ACCEL, outside the ordinary loop; require empty/overlap/subset/disjoint source fixtures, dedup/ordering/init-failure checks, patch composition and HIP build. Positive qualification requires a *real separate* draft on missing target devices, conditional trace, buffer-owner evidence and greedy/logits/acceptance/repeated-request parity. Same-device dual-XTX MTP is negative-only. If no eligible draft exists, `AWAIT_ELIGIBLE_DRAFT`; do not queue performance work or promote. If upstream adopts an equivalent safe fix, retire the local duplicate.

**Evidence:** Pinned b11552 source and 1261 `patch.py`; 8/8 source-shaped assertions; four host topology cases (two expected current-code failures, two controls). [Upstream #26636](https://github.com/ggml-org/llama.cpp/pull/26636) remains open and places registration after the loop. vLLM/SGLang speculative worker ownership is a design contrast, not a GGML patch. No patch application, pytest, HIP build, GPU execution or new speedup.

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
