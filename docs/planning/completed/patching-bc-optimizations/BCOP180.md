---
id: BCOP180
order: 180
plan: patching-bc-optimizations
state: superseded
created-at: '2026-10-13T07:11:43Z'
created-by: agent
priority: P2
work: S
---

# PRBE113: retire duplicate MMVQ width-6 profile

## Description



## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

2026-11-22 consolidation review RV4332: secondary decision note now superseded by PRBE113 (patching-rdna-boost-experiments). Existing PRBE113 authoritative BCOP180 disposition rejects historical width-6 Q8 MMVQ patch 1245 (-9.4% single run) and retains a genuinely distinct unqualified per-width trace/receipt decision. No duplicate width-6 profile. PRBE113 remains active; no new runtime implementation or acceptance asserted.

## Discovery / change

Historical 1245 Q8_0 gfx1100 width-6 fusion was already measured and rejected: +620.4 ms MMVQ busy, 22.4 ms quantize/GLU saved, throughput 51.54 to 46.70 t/s (-9.4%, single-run). n_max=5 width-6 evidence does not qualify n_max=4 widths 2-5. Existing rocprof name aggregation does not prove execution phase, physical rank or E2E critical path. RD13/1206 has separate shape-safety blocker BCOP147. No new speedup.

## Owners and action

PRBE113 owns stock per-width evidence admission; 1245 owns its negative result; PRBE115 owns any future Q8 kernel; PRBE111 owns IQ; PRBE12 owns 1206. Reuse existing rocprof parser and flash-next trace scripts, no new profiler or scheduler. Recent Radiance expert-helper work is excluded. First search complete, exact-pin, model, physical-UUID, n_max=4 steady-decode receipts. If none, permit one host-exclusive stock capture only after queued work is terminal. Close if optimistic E2E ceiling <3%; otherwise hand one qualified width signature to PRBE115, requiring full correctness and four independent paired sessions with CI95-low >=3%, <=1% control regression.

## Evidence

Pinned and current upstream MMVQ still fuse only ncols=1. llama.cpp PR #29948 is NVIDIA-only MMQ prefill, not Q8 MMVQ decode. Synthetic host discriminator 10/10 passed; no repository tests, builds or GPU runs.

## Reviews

- RV4332

## Change Log

- 2026-11-22T05:01:29.891558+00:00 (updated-by): Updated: section:notes
- 2026-11-22T05:01:40.064749+00:00 (state-transition): State: pending → superseded
