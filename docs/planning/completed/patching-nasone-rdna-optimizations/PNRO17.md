---
id: PNRO17
order: 0
plan: patching-nasone-rdna-optimizations
state: completed
created-at: '2026-09-11T10:33:08.032961+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P1
---

# Root-caused and fixed: HIP+ROCR_VISIBLE_DEVICES double-filter broke non-prefix device selection

## Description

CORRECTED 2026-09-11: this was initially filed as a gfx1201-specific hardware/driver finding. It is NOT architecture-specific and NOT a hardware issue -- it was a real bug in PVPS02's own require_device_visibility() contract, now root-caused and fixed (see PVPS02's notes for the fix commit).

Real root cause, confirmed directly on Brutus outside the harness: setting BOTH HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES to the SAME absolute device index double-filters. ROCR_VISIBLE_DEVICES filters and re-indexes the system device list FIRST (to 0..N-1); HIP_VISIBLE_DEVICES then indexes INTO that already-filtered pool, not the original absolute list. So `HIP=1 ROCR=1` asks for index 1 of a pool ROCR already narrowed to a single item (device 1, relabeled index 0) -- out of range, and llama-bench SILENTLY falls back to CPU while still printing a `backend: ROCm` result row (confirmed: ~8.4 t/s, matching CPU-only speed, vs ~75 t/s real GPU on the same binary/model with only HIP_VISIBLE_DEVICES set).

Reproduced identically on gfx1100 (single non-zero index), gfx1201, and gfx1030 -- confirming this is universal, not architecture-specific. Also confirmed: `HIP=1 ROCR=0,1` (a case where ROCR's filtered pool actually contains 2 items) correctly selects physical device 1 -- proving the re-indexing mechanism precisely, not just correlating it.

Separately, exposing all 4 heterogeneous real GPUs simultaneously (ambient env, no restriction) and using llama-bench's own `-dev ROCm2`/`-dev ROCm3` CLI flag to select gfx1201/gfx1030 SEGFAULTS during full 4-GPU enumeration, regardless of which device is finally selected -- this is a separate, still-unexplained crash in heterogeneous-topology enumeration, NOT fixed by this item, and NOT hit by the actual fix (which restricts visibility via HIP_VISIBLE_DEVICES alone before the process starts, so ggml only ever enumerates the single relevant device and never touches the crashing full-enumeration path).

The fix: require_device_visibility() (tools/bigcherry/experiment/execution.py) now validates and sets ONLY HIP_VISIBLE_DEVICES; ROCR_VISIBLE_DEVICES is no longer required, checked, or set by any PVPS02 call site (the generic matrix, RD58's preflight guard, RD73's selector_env). HIP_VISIBLE_DEVICES alone was directly confirmed correct for every architecture and topology tested on Brutus.

## Steps

DONE -- see description. No further action needed on the double-filter bug itself.

Remaining, if ever revisited (low priority, does not block anything -- the working fix avoids this path entirely): the separate `-dev`-with-all-devices-visible segfault during heterogeneous 4-GPU enumeration could be investigated (dmesg/kernel logs, stock llama.cpp repro, driver version) if a future use case specifically needs `-dev`-based selection with full ambient visibility rather than HIP_VISIBLE_DEVICES-based restriction.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Filed directly from PVPS02's real-hardware merge gate run (2026-09-11) -- see that item's notes for the full campaign context. This is a real, reproducible, hardware-level finding, not a hypothesis.

Filed from PVPS02's real-hardware merge-gate run (2026-09-11); root-caused and fixed same day after initially being mis-filed as gfx1201-specific -- corrected here rather than left wrong. See docs/planning/active/patching-validation-package-standard/PVPS02.md's notes for the fix commit and full context.

## Change Log

- 2026-09-11T10:33:08.032961+00:00 (created-by): Created by agent
- 2026-09-11T10:33:24.107336+00:00 (updated-by): Updated: section:description, section:steps, section:notes
- 2026-09-11T10:59:58.433341+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:notes
- 2026-09-11T11:00:03.922226+00:00 (state-transition): State: pending → completed
