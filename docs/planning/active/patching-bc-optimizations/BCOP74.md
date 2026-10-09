---
id: BCOP74
order: 74
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-09T00:00:00+00:00'
created-by: agent
priority: P2
work: S
---

# PNRO14: gate RDNA3.5 D256 tile on actual route and ISA

## Discovery / disposition

Pinned b11474 and current upstream share the same tile-config source. Patch 1270 is already present; PNRO19's anchor failures were fixed independently in f4228540b7 (2026-09-27), so PNRO19 is terminal. Normal aligned masked GQA D256 prefill can select MMA before the candidate's tile32 path; route proof is required before performance work. Current patch's RDNA3.5 gate covers gfx1150/1151/1152/1153 while metadata qualifies gfx1151 only. AMD PR #41 reported +0.3% pp1024 at r=1 (noise); no BigCherry hardware result exists.

## Ownership / exclusions

PNRO14 owns tile-config qualification; existing patch 1270 owns the candidate; PNRO19 is closed as duplicate resolved mechanics. Do not create another selector, patch or telemetry system. Recent BPB01/1346 diagnostics, PRBE52/MTP, QFP35/37/41/42/43, PA44-D and 1328 expert work remain protected by the 12-hour active-work rule; none was modified. BCOP71-73 were referenced in earlier unpublished audit runs and are intentionally not reused.

## Bounded gate / terminal outcome

No gfx1151 in the BigCherry fleet: keep 1270 untested/deferred. On qualified gfx1151 hardware only, prove actual TILE+ncols32 execution and >=5% E2E prefill attribution, then reconcile the gfx115x-wide selector with validation scope before any promotion. Compare stock/candidate on D256 MHA positive and masked GQA negative controls with exact work, logits and memory parity; 4 sessions x 10 paired ABBA, CI95-low >=3% E2E prefill, <=1% decode/control regression. Otherwise reject/close without implementation. 11 source assertions and 132 host selector cases passed; no compiled test or hardware run.

## References

PNRO14; PNRO19; 1270; f4228540b7; https://github.com/AMD-Ecosystem/llama.cpp/pull/41 ; https://github.com/ggml-org/llama.cpp/pull/26046 ; https://github.com/vllm-project/vllm/issues/54438 .
