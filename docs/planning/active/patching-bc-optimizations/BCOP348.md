---
id: BCOP348
order: 348
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-20T20:05:30Z'
created-by: agent
priority: P2
work: S
---

# QFP54 — Gate recurrent checkpoint D2H batching

**Discovery:** Pinned b11552 llama_io_write_host serializes deferred synchronous state reads; tensor-split Meta may touch two physical GPUs per range. Open [llama.cpp #30694](https://github.com/ggml-org/llama.cpp/pull/30694) merges **same-tensor** KV reads (external CUDA 194–258 -> 50–71 ms interleaved saves). [RDNA fork #189](https://github.com/stew675/llama-cpp-rdna-boosts/pull/189) batches **distinct recurrent tensors** with pinned staging (external 2x gfx1201 60 MiB checkpoint 11.76 -> 5.5 ms; follow-up TTFT 606 -> 566 ms). Direct fork port invalid at pinned Meta nonzero-offset async assertions. No BigCherry gain measured.

**Owner/consolidation:** New QFP54 is the sole state-save D2H technical owner; QFP48 owns existing checkpoint-arm trace, RD58/1234 H2D restore, 1005 checkpoint selection, QFP16/24/27 scheduler H2D input, MSM02/03 Meta memory. Do not edit protected QFP48 or add a duplicate profiler, cache, scheduler, allocator or permanent config. All technical design is in QFP54.

**Eligibility:** fix/patch-suite-green selected from independent Audumla dc0669a7c541 (20 Oct 13:57:27Z), ignoring later BCOP-only d5a2959a95d2. Inspected selected/main/release commits and PRs for 08:05–20:05Z. Protected RAD14 FP8/W4/MTP, RAD06/shape harness, RAD15, RCS06, MEN24/release, 0300 patch-suite. QFP48's 13:02 router-fusion edit is unrelated and untouched. Checkpoint-arm script last changed Oct 11; 1005/1234 implementation Oct 9. No independent D2H save work or queued lane found in the visible 12h; off-repository queues unverified. Novel versus recent kernel, sampler and attention audits.

**Completed:** inspected pinned server/state serializer, synchronous and Meta async split paths, fork/upstream diffs and vLLM pinned async KV boundary. Host-only 1,500 byte-equivalence cases/23,483 ranges, modeled 75-tensor 2-rank sync counts 150/150/2 and cap fallback; py_compile passed. No C++/HIP build, GPU benchmark or first-party performance result.

**Action/terminal:** HOLD_ROUTE_AND_EXPOSED_WALL until existing QFP48 checkpoint trace proves executed recurrent D2H and >=2.9126% optimistically removable E2E wall. Else NO_CHECKPOINT_ROUTE, NO_DISTINCT_TENSOR_READS or BELOW_EXPOSED_CEILING. Surviving candidate compares sync / #30694 merge / bounded #189 async, proves split/offset, physical work, pinned lifetime, RD58 restore, state/logits/MTP parity and multi-request safety; require paired gfx1100/gfx1201 CI95-low >=3% E2E and <=1% unaffected regression. WAIT_UPSTREAM, META_ASYNC_UNSUPPORTED, STATE_FAIL or NO_GAIN terminate as appropriate. No implementation or hardware job queued.
