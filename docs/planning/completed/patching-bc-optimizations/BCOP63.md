---
id: BCOP63
order: 63
plan: patching-bc-optimizations
state: deprecated
created-at: '2026-10-08T19:09:00+11:00'
created-by: agent
priority: P2
---

# PNRO14/19: retire unqualified gfx1151 TILE row and stale anchor task

## Discovery / action

Patch 1270's host/device gfx1151 D256/ncols32 row is not qualified on BigCherry's gfx1100/gfx1201/gfx1030 fleet. Upstream b11474 selects MMA before TILE for eligible D256 GQA prefill, so the row may not execute on the target workload; its marker proves config lookup only. The PNRO19 anchor failures were fixed by commit f422854 (2026-09-27) and should not be reworked.

## Authoritative ownership / already-acted work

PNRO14 owns the gfx1151 tile-row candidate and terminal no-qualified-lane disposition; PNRO19 is completed as already fixed. Patch 1270 stays untested for provenance. Existing HIP FA dispatch/kernel remains upstream-owned. No competing selector, runtime flag, patch or benchmark lane is created. Last independent capability work 2026-09-27, outside the 12-hour exclusion. Recent PA47/PA45, MTP, QFP and MoE cache/patch work is protected and untouched.

## Terminal gate / evidence

No action without real gfx1151 hardware plus trace-proven D256/ncols32 TILE dispatch and >=5% target E2E wall share. If eligible, >=4 sessions, >=10 paired rounds/session, full correctness, CI95-low >=3% E2E and <=1% control regression; otherwise reject. External Strix Halo fork reports upstream MMA +11.2% pp at 16K over its different custom TILE patch (2026-09-25), not a 1270 A/B. Ten pinned-source assertions and two arithmetic checks passed; no build, patch apply or hardware run.

## References

- https://github.com/ggml-org/llama.cpp/blob/b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea/ggml/src/ggml-cuda/fattn.cu
- https://github.com/ggml-org/llama.cpp/pull/28102
- https://github.com/justinappler/llama.cpp-strix-halo/blob/master/strix-halo/fa-mma-d256-26419.md

## Change Log

- 2026-10-10T10:25:58.353034+00:00 (state-transition): State: completed → deprecated
