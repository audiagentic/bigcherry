---
id: MET12
order: 12
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-08T22:20:00+11:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Conditional expert-weight prefetch: overlap measured host-resident transfers with independent GPU work

## 2026-10-15 BCOP216 audit: retire duplicate MET13; distinguish exact, stream-all, and predicted transfers

**Disposition and ownership:** MET13's proposal for a dedicated copy stream, a multi-slot ring and layer-ahead host-expert staging duplicates this MET12 capability and is retired as `SUPERSEDED_DUPLICATE` (archived under `docs/planning/completed/patching-moe-expert-tiering/MET13.md`). MET12 alone owns the conditional *pre-router* weight-prefetch decision. MET06 owns the native host-weight scheduler/pipeline admission; MET07 owns persistent cache/profile policy; MET10 owns host-tail graph and expert-ID mapping; MET05 owns transport. Do not create a second scheduler, copy callback, residency policy, cache, or ring while an existing owner can implement the measured remainder. RAD02–RAD05 helper activation staging/GEMV and RAD04 prefill are independently active and are not modified.

**Pinned b11552 source, not a performance measurement:** `ggml/src/ggml-backend.cpp::ggml_backend_sched_copy_input` waits for prior destination-buffer use before copying; `ggml_backend_sched_compute_splits` copies non-host inputs first, then host weights, before executing that split. `src/llama-context.cpp::llama_context::sched_copy_experts` (2641+) uses only a split whose first node is `MUL_MAT_ID`, reads already-computed `ids` with `ggml_backend_tensor_get_async` followed by `ggml_backend_synchronize`, and uploads exact selected expert ranges with `ggml_backend_tensor_set_async`. This cannot supply **same-layer exact selected IDs before that layer's router**. Existing native `sched_copy_experts`/copy callback is the current implementation surface; historical patch `1336_sched_copy_callback` is `superseded`, so do not revive or extend it as a production patch. The `moe_cache` branch is separately owned by MET07.

**Three distinct candidate modes, with an explicit admission rule:**
1. **Exact post-router selected uploads (baseline):** existing native callback. It may enqueue asynchronous copies, but this alone does not prove a shorter exposed prefill critical path.
2. **Deterministic stream-all ahead (Strata mechanism):** before routing, stream every nonresident expert of a future layer, with a bounded ring and event-protected reuse. It does not require predicting IDs, but pays for unused bytes and may saturate PCIe. Strata `src/prefill/prefill.cpp` at `Niko1221/Strata@296b59d2` uses a configurable `stream_all_min()` (default 1024 tokens), an all-expert layer/ID sequence, `issued/consumed` ring indices and `used/copied` events; below its threshold it stages routed experts after routing. Its stream-all behaviour is **not** evidence that BigCherry's scheduler or model has the same cost balance. Source: https://github.com/Niko1221/Strata/blob/296b59d2f73c8f04a805df29c2fa97655adace78/src/prefill/prefill.cpp .
3. **Profile-hot predicted ahead (MET12's remaining question):** pre-stage only immutable predicted slabs before routing; after routing, enqueue and await **every** exact miss. A wrong prediction may waste bandwidth but must never omit an expert or change output. No new selector: reuse MET07/1338-compatible validated profile identity if it exists; otherwise use the exact baseline.

**Transfer-ceiling discriminator before any GPU reservation:** MET10's historical Flash-Next batches touched ~61% of a layer's experts at ub512 and ~79% at ub2048. For equal-sized slabs, copying all experts instead of only those unique IDs requires a **theoretical** 1/0.61 - 1 = 63.9% and 1/0.79 - 1 = 26.6% more expert payload, respectively, before cache effects, padding and repeat transfers. These are not measured H2D deltas. With MET01's historical gfx1201 effective host upload ~6.7 GB/s, the extra serial transfer floor is `(N_all-N_unique)*bytes_per_expert / 6.7e9` seconds; a stream-all design is inadmissible unless a measured independent overlap window covers that cost and the ring fits the available VRAM. On the R9700 PCIe 4.0 x4 and 6900 XT chipset path do not infer dual-XTX bandwidth or peer copies. Fully resident production is a zero-host-byte negative control.

**Upstream/fork reconciliation (14 October):** llama.cpp [#29963](https://github.com/ggml-org/llama.cpp/pull/29963) remains open, head `273cd26addace9264b44f79508416600d3109daf`, updated 2026-10-13; its exact host-weight uploads/input-event pipeline are **layer-split**, not BigCherry's production Meta tensor split or pre-router prediction. [#30112](https://github.com/ggml-org/llama.cpp/pull/30112) merged 2026-10-08 and supplies the native multi-GPU MoE cache: residency/cache ownership remains MET07. [SGLang OffloaderV2](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/offloader.py) is an event/lifetime reference, not an AMD performance control. Do not port a second copy stream/ring based only on external speed claims.

**Bounded next gate (no queued hardware job):** (0) inspect completed `moe-copy-ab.sh` receipts and actual current-pin host-weight route; never treat a historical 1336 PATCH_HIT as a b11552 event. (1) On one genuinely over-VRAM configuration only, obtain a matched, terminal trace with physical GPU UUID, owner, context, ubatch, exact selected/unique slabs, H2D bytes and device-copy intervals, router-ready timestamp, copy wait on the consuming critical path, spare PCIe bandwidth, ring/VRAM headroom and graph/cancellation identity. Reject `NO_HOST_RESIDENCY`, `NO_EXPOSED_WAIT` or `BELOW_3PCT_CEILING` if absent or if *all* removable exposed H2D wait is <3% of prefill wall time. (2) Only if that gate passes, compare default-off exact callback with one bounded ahead strategy (profile-hot first; stream-all only if its byte/overlap inequality passes). Pseudocode: `prefetch(generation,layer,expert,immutable_blob)->slot`; `wait(copy_done)` for predicted hits; `enqueue_exact_miss + wait(copy_done)` for every routed miss; `record(last_consumer_done)` before reusing a slot. Gate by request/generation/device/quant/model identity; fail closed to the exact baseline on ring exhaustion, event failure, graph replay mismatch, invalid profile, cancellation or insufficient pinned memory. Require unchanged exact expert work, full-vocab logits/KLD/greedy/MTP, multi-ubatch and repeated-request correctness, peak VRAM and PCIe accounting, plus CI95-low >=3% end-to-end prefill with <=1% decode/control regression. Otherwise terminate without a patch.

**Cheap discriminator actually run:** disposable host-only `met12_prefetch_host_gate.py`, 10/10 unittest cases (pre-router ID prohibition, miss completion, all-hit/all-miss, out-of-range, copy/consumer and generation fences, coverage/VRAM/Amdahl bounds). This is not repository pytest, C++/HIP compilation, a GPU run, or a new performance measurement. Historical MET01/MET10 numbers remain controls only.



## 2026-10-11 MET06 ownership fence

Upstream [llama.cpp #29963](https://github.com/ggml-org/llama.cpp/pull/29963) (open, ready for review) already proposes per-layer, split-owned **exact host-weight uploads** with scheduler input events and buffer reuse. It is layer-split only; it is not a prediction/prefetch cache and cannot be counted as a tensor-split production gain. MET06 owns its HIP source/correctness/performance admission and equal-VRAM comparison. **MET12 must not reimplement #29963's input-copy deduplication, host-weight copy callback, fit balancing, scheduler or transient allocator.** Keep MET12 gated on independently measured *exposed* host-expert copy stalls after MET06's controls, and on safe prediction before the router (never authoritative same-layer expert selection). Do not queue concurrent QFP49/1360/1361 or b11552 scheduler experiments; these are protected active work. No MET12 implementation or benchmark was performed in this documentation audit.

## Description

Strata-style CPU expert staging is NOT assumed to help the resident three-card Flash-Next production layout. MET07 owns expert cache/profile qualification (1337/1338; prior hardware regressions must be respected); MET10 owns range-aware resident-plus-host-tail graph correctness; MET05 owns auxiliary ROCm transport; MET11 owns load-time placement; MET01/RPL01 own residency decisions. This new item owns only **asynchronous predicted hot-weight transfer and event-protected staging** if and when selected expert H2D copies are a demonstrated prefill critical-path cost on an over-VRAM lane. No duplicate cache policy, 1281 range logic, or Meta scheduler.

Fundamental limit: an exact same-layer expert list is unavailable until that layer's attention/router executes. The only traffic that can be sent before that dependency is speculative, immutable prefetched weight data (e.g. profile-hot slabs), not authoritative routed outputs. Incorrect predictions must cost only time/bandwidth, never model accuracy.

## Steps

1. Gate 0: reuse historical 1336 receipts where applicable, then verify the current-pin native callback route with rocprofv3 to measure per-chunk exact selected experts, unique slab bytes, pinned CPU copy, H2D enqueue/device copy, wait before expert MMQ, PCIe bandwidth, cache hits, and overlappable attention/other device interval. Separate prefill ub512/1024/2048 (large batches select many experts), short/deep prompts, and gfx1100/gfx1201. Mark production all-resident as a negative control; do not implement if uncovered H2D time is negligible.
2. Build a deterministic read-only prefetch chooser over an existing MET01/1338 profile; cap hot set per layer and total staging VRAM. Track predicted, delivered, exact selected, required misses, eviction/churn and all bytes including useless prefetch. Keep profile inferences optional, bounded and per-model; no token-content approximation.
3. Only after the measured gate passes, extend the existing native host-weight scheduler/callback owner with a bounded pinned host/device staging ring (not a second scheduler). Enqueue only immutable slabs before their consuming MoE op. Each slot has request/chunk/layer/generation, rank/device owner, stream-ready and copy-done events. Reuse only after every consumer completes; fall back to current synchronous path on no event query, ring full, absent pins or graph-capture mismatch.
4. When authoritative router IDs become available, ensure every selected weight is resident or its exact miss has completed. No skipped experts, altered top-k, approximate weights or reordered F32 accumulation. Do not pre-stage host tail on all Meta ranks: only its designated owner gets it.
5. Test predicted-hit, all-miss, all-hot, empty routing, skew/Zipf, large ubatch high-coverage, rank cancellation, ring overflow, graph replay, context shift, two sessions, and restored decode placement. Record activation/fallback counters.
6. Hardware ABBA: default-off synchronous 1336/MET10 versus default-off-capable prefetch on, same full model, quant, placement, host residency, ubatch, context and profile. Measure prefill, exposed upload stalls, GPU copy/compute concurrency, peak VRAM, decode, acceptance, greedy IDs and CPU-f32 probes. Repeat on one genuine over-VRAM lane only; do not equate with Strata's published model/hardware.

## Detailed Solution & Technical Design

- Pre-stage can overlap *preceding independent compute or attention*, not a same-layer MoE computation whose router output is a prerequisite.
- Prefer a bounded transfer ahead of time for profile-hot candidates; after routing, issue exact-demand transfers for misses and gate the selected expert compute behind their completion event.
- Use MET10's id_base/global-id-to-local-id semantics for host tails and MET05 transport routing; no parallel reimplementation of either.
- Reserve GPU staging bytes explicitly; prefill scratch leasing in QFP47 must not evict a live slot. The 6900 XT PCH attachment is a separate negative/low-bandwidth lane, not an implicit fast prefetch worker.
- Disable prediction for requests without a matching validated profile, anomalous placement mapping, unknown quant block alignment, or adverse copy saturation.

## Files

Design owner: MET12. Future implementation, only if admitted, belongs to the current native b11552 host-weight callback/scheduler and narrow Meta/backend helpers owned by MET06/MET10/MET05. Patch 1336 is superseded; do not re-enable it. An opt-in patch package is permitted only for a genuinely independent, measured pre-router feature. Synthetic miss/consumer oracle: tools/tests/prefill/test_prefill_contract_model.py. Hardware tooling: tools/lab/flash-next/moe-copy-ab.sh / queue-moe-copy.sh.

## Validation

Gate 0 must show measurable exposed expert-upload stalls, not merely many host-resident bytes. Synthetic tests prove only that misses cannot be used early; production tests must prove exact slab contents, event ordering, bounded memory, and unchanged model outputs. A/B and hardware timeline must show both useful copy overlap and lower prompt critical-path wall; otherwise reject without a second cache implementation.

## Acceptance Criteria

- The default-off path is unchanged and production fully resident workloads do not pay overhead.
- Every required expert waits on an actual completed staged copy, with correct owner/range, despite arbitrary predicted misses and cancellation.
- No extra collective or replicated upload across Meta ranks.
- Exposed H2D wait falls and end-to-end prefill improves repeatably in a declared over-VRAM lane, without fidelity/TTFT/decode regression. Otherwise close as negative evidence.

## Change Log

- 2026-10-08: Created as conditional transfer-overlap gap; separated from MET07 cache, MET10 graph and MET05 transport.
