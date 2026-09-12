---
id: PRBE106
order: 0
plan: patching-rdna-boost-experiments
state: completed
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

**Fix applied and verified (2026-09-13), but with a corrected expectation.** Changed GGML_LOG_INFO to GGML_LOG_WARN at line 155 (now ~160), matching RD08's precedent. Rebuilt and tested: marker fires WITH --verbose (1 hit), but still does NOT fire WITHOUT --verbose (0 hits) -- WARN is NOT actually exempt from llama-bench's --verbose gating on this build, contrary to the original assumption in this item's description (based on RD08's own docstring claim). This matches a real, separately-confirmed finding for RD08 itself this session (RD08's own WARN-level marker also required --verbose to appear). The fix is still correct and worth keeping (WARN is the more appropriate log level regardless, and matches project precedent), but the real, load-bearing fix for future validators is: always pass --verbose when probing for BIGCHERRY_PATCH_HIT markers, regardless of log level used. Recommend updating docs/reference or the shared trace-probe primitive's documentation to state this plainly, since the GGML_LOG_INFO->WARN change alone does not solve the underlying gating behavior on this llama-bench build.

## Change Log

- 2026-09-12T20:57:48.492290+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260912_205810_found-and-corrected-a-false-ne_3827
- 2026-09-12T20:58:10.868990+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T22:29:26.082214+00:00 (updated-by): Updated: section:notes
- 2026-09-12T22:29:31.067171+00:00 (state-transition): State: pending → completed
