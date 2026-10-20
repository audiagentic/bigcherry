---
id: BCOP236
order: 236
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-15T16:11:54Z'
created-by: agent
priority: P2
work: S
---

# QFP27/1326: bound pageable staging retention without duplicate async-input ownership

**Discovery:** 1326's function-local thread-local map retains <=4 MiB per distinct input-copy tensor pointer for the server-thread lifetime, independent of scheduler reset/free. This is a source-confirmed *capacity-retention risk*, not measured RSS growth. Pinned b11552 scheduler is calloc/free allocated; directly embedding C++ containers is not valid.

**Owner:** QFP16/1326 owns bounded scheduler-owned POD pageable scratch and lifecycle tests; QFP27 owns upstream applicability and Meta all-child admission; QFP24 owns only conditional large pinned prefill. Reuse 1325 trace, native scheduler, Meta fanout and synchronous fallback. No new ring, cache, scheduler, event API, allocator or telemetry.

**Eligibility:** Selected independent main HEAD 08441aa4a69e (MEN10/RAD09, 2026-10-15 05:21 UTC). Latest related independent plan work 63cc952f5d (2026-10-15 00:20 UTC), outside 12h; no newer QFP16/24/27/1326 commit, PR or queued work found. MEN10/RAD09 and Radiance implementation paths untouched. Recent BCOP226-235 concern other slices.

**Action:** Host-prove graph/scheduler lifetime retention and no stale data; replace only <=4 MiB pageable staging map with one scheduler-owned bounded POD scratch. Preserve sync fallback if host consumption before return is not guaranteed for every child. No pinned allocation in this change. QFP24 pinned prefill remains behind >=1 ms AND >=3% exposed input-cost gate. Upstream [llama.cpp #29963](https://github.com/ggml-org/llama.cpp/pull/29963) (open, head 6325cf421) is layer-pipeline-only, not Meta completion.

**Terminal:** NO_RETAINED_GROWTH (close after host/RSS proof), SAFE_BOUNDED_PAGEABLE (lifecycle + HIP correctness, <=1% decode/control regression), BACKEND_LIFETIME_UNPROVEN (retain safe existing path), or REGRESSION_REJECTED. No performance claim without paired current-pin measurements.

**Validation:** 10/10 source assertions, 7/7 host-model tests, Python compilation passed. No repo pytest, HIP build, GPU test, RSS measurement or new benchmark.
