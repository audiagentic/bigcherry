---
id: BCOP25
order: 25
plan: patching-bc-optimizations
state: superseded
created-at: '2026-10-05T07:24:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: S
---

# Execute bounded D=72 HIP Flash-Attention root-cause qualification

## Superseded disposition (2026-10-08)

BCOP25 recorded the historical b11402 D=72 FA tail-guard proposal; it did not establish a hardware fix, build or benchmark. PHA08 was rebaselined to b11474 and 18 host-only Q/K/V/output vector-width checks show no partial 16-byte vector in the visible contiguous D=72 paths. The old speculative tail-guard first action is withdrawn, not claimed to have failed on hardware. The aperture issue remains open and unresolved.

**Terminal:** completed as superseded by BCOP66. PHA08 retains the only technical design. No new kernel/selector, no duplicate queue or separate investigation. See PHA08 and upstream llama.cpp #28608/#28664.

## Change Log

- 2026-10-10T10:24:52.301374+00:00 (state-transition): State: completed → superseded
