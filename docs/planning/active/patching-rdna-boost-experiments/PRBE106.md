---
id: PRBE106
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-12T20:57:48.492290+00:00'
breadth: ''
skill: ''
created-by: agent
---

# RD12's activation marker uses GGML_LOG_INFO, should be GGML_LOG_WARN (RD08 precedent)

## Description

patches/1205_rd12_paired_mmvq_dual_output/patch.py line 155 emits its BIGCHERRY_PATCH_HIT marker via GGML_LOG_INFO. llama-bench gates INFO-level ggml logs behind its own --verbose CLI flag (off by default), so any validator running llama-bench without --verbose gets a false 0-hit negative -- exactly what happened in this session's first activation attempt (2026-09-13), corrected only after adding --verbose. This project already found and fixed the identical bug for RD08 (GGML_LOG_INFO -> GGML_LOG_WARN, commit 991e761, VA21) and added --verbose to the shared trace-probe primitive as defense in depth. RD12 was not updated to match.

## Steps

1. Change patch.py's GGML_LOG_INFO call to GGML_LOG_WARN for the marker at line 155, matching RD08/RD13's pattern.
2. Re-verify the patch's existing anchor/idempotence tests still pass after the change.
3. Re-run the activation check without --verbose to confirm the marker now fires without it (WARN is not gated behind --verbose).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Low-risk, mechanical, single-line fix. Filed rather than fixed inline since patch.py changes go through the bigcherry-patch-author workflow, not ad-hoc edits during a validation session.

## Change Log

- 2026-09-12T20:57:48.492290+00:00 (created-by): Created by agent
