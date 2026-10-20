---
id: BCOP234
order: 234
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-15T14:04:49Z'
created-by: agent
priority: P1
work: S
---

# PRBE41: fence channels-major SSM_CONV activation evidence

**Discovery:** Patch 1263 emits `ssm_conv_channels_major_declined_split` for tensor-split fallback, but its producer and TOML accept the unbounded `ssm_conv_channels_major` prefix, allowing a false `executed` promotion trigger. Positive logging is process-once and precedes HIP completion; server and benchmark processes have separate route evidence. Source/host finding, not reproduced GPU fault.

**Eligibility:** `main` selected from independent `08441aa4a69e` (2026-10-15 05:21 UTC, MEN10/RAD09). PRBE41 last independent change `63cc952f5df6` (2026-10-15 00:20 UTC) was outside 12h; PR #152 last updated 2026-10-11; no newer patch, PR, queued experiment or hardware work found. Recent BCOP219–233 cover other slices. Excluded MEN10, RHP01–06, Radiance/RAD09.

**Owner/dependencies:** PRBE41/1263 owns single-device layout and activation. PRBE114 Meta tensor-split rewrite is closed; PRBE18/21 own distinct CONCAT fusion. No duplicate scheduler, kernel, cache, allocator, dispatcher or telemetry.

**Action:** Bound both regexes, enforce exclusive positive/declined classification, add host regression matrix and require a benchmark-side HIP kernel/graph witness before interpreting prefill gains. PRBE41 owns exact files, algorithm, matrix, controls and thresholds. Upstream llama.cpp #30591 is Vulkan-only, not a HIP replacement; AMD fork #52 gains are external.

**Terminal:** `FALSE_ACTIVATION` until fixed, then `AMBIGUOUS_ROUTE`, `UNPROVEN_EXECUTION`, `NO_EXPOSED_HEADROOM`, `NUMERICAL_OR_STATE_FAILURE`, `REJECT_1263`, or qualified default-off. Promote only if CI95-low >=3% E2E prefill, <=1% controls and full logits/MTP/replay parity. No hardware queue.

**Validation:** 8/8 source assertions, 12/12 host-only marker tests and Python compilation passed. No repository pytest, HIP build or GPU benchmark ran.
