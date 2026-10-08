---
id: BCOP70
order: 70
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-08T15:10:33+00:00'
created-by: agent
priority: P2
work: S
---

# PNRO13: native PLE prefetch baseline

## Discovery / disposition

Pinned b11474 already includes merged upstream #29599 (2026-09-30): lazy PLE rows are prefetched before GET_ROWS. Direct-read #28136 and #29030 closed unmerged, with #29030 superseded by #29599. No BigCherry pread-versus-prefetch result exists. POSIX_FADV_RANDOM does not bypass the page cache.

## Ownership

PNRO13 is authoritative for PLE file-row staging. QFP17 prefill/QSA and PNRO08 MoE expert cache are separate, recently active capabilities; neither was modified. No duplicate cache, scheduler, allocator or experiment was created. BCOP69 is reserved for an earlier unpublished audit.

## Bounded next action

Measure cold/warm real-text PLE critical-path cost on gfx1201 and gfx1100 against native prefetch, recording faults, cache residency, I/O and row spans. Close if residual PLE <5% E2E, ceiling <3%, or prefetch wins. Otherwise prototype only opt-in per-context positioned reads with checked offsets, dedup/scatter, existing quant conversion and mmap fallback. Require correctness plus CI95-low >=3% E2E gain and <=1% control regression over four sessions and ten paired ABBA rounds. Six host fixtures passed; no hardware run.

## References

PNRO13; QFP17; PNRO08; https://github.com/ggml-org/llama.cpp/pull/29599 ; https://github.com/ggml-org/llama.cpp/pull/29030 .
