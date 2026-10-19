---
id: BCOP327
order: 327
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-19'
created-by: agent
priority: P2
work: S
---

# RNX08: Q8_0 SET_ROWS warp-group cache-write viability

**Discovery:** Pinned b11552 `set-rows.cu::k_set_rows_quant` quantizes each 32-value Q8_0 block serially in one thread. [RDNA fork #171](https://github.com/stew675/llama-cpp-rdna-boosts/pull/171) uses one 32-lane group per block, trading five shuffles and a 32x larger CTA grid for coalesced accesses. External gfx1100 5.1 -> 3.7 us/call and gfx1201 1.51 -> 1.33 us/call are **not BigCherry gains**. Combined fork E2E +1.7% (noisy) / +0.6% is below the >=3% promotion gate.

**Authoritative owner:** RNX08 Q8_0 cache-write microkernel disposition. Reuse native HIP SET_ROWS slot/stream dispatch; PRBE15 owns ROPE/IMROPE scatter fusion, RNX02 attention reads, 1307 Q8_1 activation cache. Fork #171's separate RMS_NORM_Q8_1 change is out of scope. No new kernel, dispatcher, allocator, scheduler, telemetry or config.

**Eligibility:** Selected independently active `feat/rad13-fp8-worker` at Audumla `deb232fe8ed` (2026-10-19 12:52 UTC), newer than main `f792f51e` (12:23 UTC). Repository-visible 12-hour Q8 SET_ROWS exclusion and novelty passed. Protected RAD13 PR #330, RCS09, RAD11/12/10, WSL and release work excluded; external queues unverified. Distinct from BCOP322–326.

**Cheapest discriminator:** Host-only 10,005 finite Q8_0 blocks byte-equal; 13,260 logical addresses complete/unique; Python compilation passed. Old `ceil(B/256)` versus new `ceil(B/8)` CTAs; D256/four KV heads/tokens 1/2/4/8: 1 old versus 4/8/16/32 new. No GPU build/test/benchmark.

**Unresolved action:** Do not port or queue hardware on external microseconds. First prove BigCherry Q8_0 SET_ROWS physical route and >=2.9126% exposed removable E2E ceiling. Otherwise close `NO_Q8_WRITE_ROUTE` / `BELOW_CEILING`; if upstream absorbed close `PIN_ABSORBED`. Only if admitted, one default-off existing-dispatch group kernel, byte/canary/logit/KLD/MTP/graph correctness, ISA/VGPR/spills and paired CI95-low >=3% E2E/no >1% control regression. Otherwise `NUMERIC_FAIL` / `REGRESSION` / `NO_GAIN`. Technical design belongs solely to RNX08.

**No implementation or hardware work queued.** Leave active independent work and the recurring optimisation task unchanged.
