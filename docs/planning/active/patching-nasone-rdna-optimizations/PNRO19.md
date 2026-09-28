---
id: PNRO19
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-27T12:55:18.984027+00:00'
breadth: ''
skill: ''
created-by: agent
priority: P2
work: S
---

# 1270 (PNRO14) patch mechanics test fails: anchor matched 3x / 0x against vendor source

## Description

tools/tests/patch/test_1270_pnro14_rdna35_fa_tile_d256.py::test_apply_and_idempotent fails offline (no hardware needed to reproduce). Two anchor failures in patches/1270_pnro14_rdna35_fa_tile_d256/patch.py against the current vendor/llama.cpp tree: edit 'pnro14-includes' matched 3 lines (expected <=2, gone greedy on the #include "common.cuh"/#include "fattn-common.cuh" anchor), and edit 'pnro14-device-select' matched 0 times (expected 1) against the ggml_cuda_fattn_tile_get_config device selector -- the anchored source shape is no longer present, likely drifted since the patch was authored (single commit b353bc17). Found 2026-09-27 while running the full offline patch suite for an unrelated change (PVPS13); confirmed pre-existing and unrelated to that session's work (no vendor tree or 1270 files were touched). Needs bigcherry-patch-author attention: re-anchor pnro14-includes narrower/more specific, and re-check whether ggml_cuda_fattn_tile_get_config's current source shape still matches what pnro14-device-select expects to replace.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

test_missing_host_selector_fails_closed (the other test in the same file) still passes -- only test_apply_and_idempotent is affected.

## Change Log

- 2026-09-27T12:55:18.984027+00:00 (created-by): Created by agent
