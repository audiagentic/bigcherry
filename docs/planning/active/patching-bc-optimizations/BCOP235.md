---
id: BCOP235
order: 235
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-15T15:09:29Z'
created-by: agent
priority: P2
work: S
---

# QFP35: fence upstream CUDA HC_POST convergence at the next pin

**Discovery:** Upstream llama.cpp [#30602](https://github.com/ggml-org/llama.cpp/pull/30602), opened 2026-10-15 13:33 UTC (open/unmerged at audit), ports merged Vulkan [#29520](https://github.com/ggml-org/llama.cpp/pull/29520). It duplicates BigCherry validated 1355's four-op HC_POST gate fusion but uses flat 1-D indexing and rejects nonzero biases. Its changed launch/declaration collide with exact 1344/1355 anchors. No new BigCherry performance gain.

**Owners/consolidation:** Completed QFP35 owns 1344 dimensional PRE/POST and 1355 gate fusion; 1313/QFP13 owns standalone SCALE/activation; QFP48 owns other trace-led fusions. Keep QFP35 completed and both validated patches on b11552. No duplicate kernel/matcher/allocator/scheduler/telemetry or hardware job.

**Eligibility:** selected `main` at independent `08441aa4a69e34d6905e6449e19467900ae19c95` (MEN10/RAD09, 2026-10-15 05:21 UTC). QFP35 last independent plan change 2026-10-10, 1355 source 2026-10-12, 1344 source 2026-10-09; no same-path BigCherry PR, patch, queued experiment or evidence in the preceding 12 hours. MEN10/RAD09 protected and untouched. This is new upstream evidence, not a repeat of BCOP223–234.

**Action/terminal:** only if #30602 reaches the actual pin, use QFP35's exact composition/route/numerical/graph gate to choose **one** implementation and preserve 1344 geometry/bias fallback. Historical 1355 +0.5–0.7% decode and 1344 +1.3–1.6% prefill are pin-bound; upstream RTX Pro 6000 ~3% is not AMD evidence. Terminal `UPSTREAM_NOT_PINNED`, `COMPOSITION_CONFLICT`, `ROUTE_MISMATCH`, `NUMERIC_OR_LIFETIME_FAIL`, `NO_NET_GAIN`, or `UPSTREAM_REPLACES_1355` only after actual AMD parity and <=1% regression. Any new gain claim needs CI95-low >=3% E2E. Do not delete validated 1355 on source similarity alone.

**Audit validation:** 12/12 source/diff assertions; 4/4 host-only unittest methods (including 10,000 F32 zero-bias cases); Python compile and four draft-structure checks passed. No repository suite, HIP build, patch application, GPU run or new benchmark. Technical design remains in QFP35.
