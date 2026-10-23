---
id: QFP54
order: 54
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-20T20:05:30Z'
created-by: agent
priority: P2
work: M
---

# QFP54 — Recurrent-state checkpoint D2H batching on HIP/Meta

## Decision and independent-work gate (BCOP348; 20 October 2026 UTC)

**HOLD: source-qualified, not admitted to implementation or a GPU queue.** This owns the narrow sequence-state **save** path, not state restore, KV-cache placement, kernel fusion, scheduler input copies or checkpoint-selection policy. Pinned llama.cpp b11552 serially executes deferred synchronous tensor reads. Upstream [#30694](https://github.com/ggml-org/llama.cpp/pull/30694) coalesces reads **within one tensor**; independently measured [RDNA fork #189](https://github.com/stew675/llama-cpp-rdna-boosts/pull/189) batches reads **across distinct recurrent tensors** into a pinned buffer. They are complementary, not interchangeable. Neither has a first-party BigCherry gain; the fork's newer-revision patch cannot be applied verbatim to b11552.

Branch selected: fix/patch-suite-green, from independent Audumla dc0669a7c541 (2026-10-20 13:57:27Z); the newer d5a2959a95d2 head is scheduled BCOP340 documentation and does not confer independent branch freshness. Checked selected/main/release commits and PRs in 08:05–20:05Z. Excluded RAD14 FP8/W4/MTP, RAD06 Flash-Next A/B and shape-harness, RAD15, RCS06, MEN24/release, 0300 offline patch-suite repair and Radiance 1.9.0. QFP48's 13:02 router-fusion disposition is a different capability and **QFP48 is not edited**. Checkpoint-arm script last changed Oct 11; RD58/1005 implementation Oct 9. No independently advanced checkpoint-save D2H implementation, patch, PR, benchmark or queued lane was found in repository-visible 12-hour evidence; off-repository queues are unverified. Distinct from recent Q3_K, M-RoPE, sampler and attention audits.

## Pinned production trace and ownership

1. At b11552, tools/server/server-context.cpp::server_slot::prompt_save (~line 308) sizes and allocates target/draft prompt-cache buffers and calls llama_state_seq_get_data_ext on ctx_tgt and then ctx_dft. src/llama-context.cpp::llama_state_seq_get_data_ext (~line 4445) calls ctx->synchronize() **before** serialization. Record that synchronization separately; batching later reads cannot eliminate it.
2. src/llama-context.cpp::llama_context::state_seq_get_data (~line 3294) constructs llama_io_write_host for host state; state_seq_write_data delegates to memory->state_write. llama_io_write_host::write_tensor (~line 2800) records tensor, output pointer, size and offset; its destructor (~line 2786) calls ggml_backend_tensor_get for every deferred range. ggml/src/ggml-backend.cpp::ggml_backend_tensor_get uses the buffer's synchronous get_tensor, not backend get_tensor_async. The output is complete only when the writer is destroyed. ON_DEVICE and file writers are separate paths.
3. Under tensor split, ggml/src/ggml-backend-meta.cpp::ggml_backend_meta_buffer_get_tensor (~line 1574) reconstructs logical split axes 0/1/2, multi-segment and MIRRORED ranges via simple tensors and synchronous per-device get/get_2d. Pinned ggml_backend_meta_get_tensor_async (~line 1943) asserts offset==0, n_segments==1 and nr[0]==1 for non-mirrored tensors: arbitrary recurrent row-offset reads cannot use it as-is. ggml_backend_meta_synchronize already joins simple backends. ggml_backend_tensor_get_2d_async may fall back to per-row operations and backend get_tensor_async may synchronize if unsupported; actual physical API counts are mandatory.
4. No GPU kernel math, KV layout, MTP accept decision, P2P or RCCL changes are needed. The potential saving is serialized D2H call/synchronization time on the host critical path. Measure actual per-rank bytes and pinned memcpy cost. Existing 1234_rd58_pin_state_buffer_multigpu_restore owns **H2D restore** registration and SDMA correctness, not D2H saving; 1005_prompt_cache_checkpoint_selection owns reuse selection. Neither is modified.

## Upstream and other-engine evidence

- Open [llama.cpp #30694](https://github.com/ggml-org/llama.cpp/pull/30694), updated 20 Oct 20:05Z, changes only src/llama-context.cpp::llama_io_write_host destructor. Small-range threshold 64 KiB, same-tensor increasing offsets, span bounded by 8x payload + 1 MiB; host scatters the merged read. External CUDA RTX 3090 interleaved save 194–258 -> 50–71 ms; contiguous 8,611-token save median 441 ms unchanged. Treat as an upstream baseline/control, **not** a cross-tensor recurrent-state solution or AMD measurement.
- [RDNA fork #189](https://github.com/stew675/llama-cpp-rdna-boosts/pull/189), updated 20 Oct 16:57Z, is based on r91/r95 after other state-read changes. Its source adds context-owned pinned state_stage (16 MiB allocation granularity), state_backend lookup, <=256 MiB admission, enqueue get_tensor_async for each write_info, deduplicated backend synchronize, and ordered host scatter. Meta refactors its synchronous split arithmetic into a common sync/async helper so nonzero row offsets work. Restore is unchanged. Fork-only LLAMA_STATE_ASYNC and LLAMA_STATE_ASYNC_MAX_MIB are not BigCherry flags.
- **External 2x gfx1201**, Flash-Next tensor split: 60.1 MiB checkpoint 11.76 -> 5.5 ms; short follow-up TTFT 606 -> 566 ms (13/14 faster); four concurrent streams +3.5% (3 pairs, CI +1.0 to +6.1). One-device 66.7 MiB checkpoint 6.64 -> 6.91 ms; layer split neutral/negative. r95 greedy/restore checks passed in the fork. These results are **not** BigCherry dual gfx1100 + gfx1201 + gfx1030 qualification.
- [vLLM CPU KV offload](https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_offload/cpu/gpu_worker.py)::SingleDirectionOffloadingHandler/submit_store uses pinned CPU blocks and explicit async transfer completion. Only bounded pinned lifetime and completion-before-consumption are reusable concepts; its KV-block cache and stream/job scheduler do not port to llama.cpp checkpoint serialization. No NVIDIA/SGLang performance extrapolation.

## One bounded discriminator and conditional implementation

**Gate 0 — first-party executed route and removable wall, no patch:** Reuse QFP48's tools/lab/flash-next/queue-checkpoint-arms.sh and existing warm per-batch HIP API/device trace, not a second profiler or queued experiment. Instrument server_slot::prompt_save, pre-save ctx->synchronize, state_seq_write_data, writer destructor, backend get/get_2d and the first subsequent kernel. Per physical rank/context record: state bytes, deferred range count, distinct tensor count, same-tensor mergeable count, split axis/segments, D2H API calls, sync calls, physical bytes, host memcpy, checkpoint frequency, draft versus target, pinned capacity and exposed TTFT/prefill/serving wall. Use matched warm multi-turn cache workloads, separate from cold prefill and MTP verification. Controls: single gfx1100, dual gfx1100 tensor split, supported mixed gfx1100/gfx1201, layer split and dense nonrecurrent. QFP48's historical ~8.7% batch-boundary idle is **not** checkpoint attribution. Disabling checkpoints changes reuse semantics: its ablation is diagnostic only, not final A/B.

**Reject at Gate 0:** NO_CHECKPOINT_ROUTE, NO_DISTINCT_TENSOR_READS, or BELOW_EXPOSED_CEILING if there is no executed recurrent checkpoint-save cost or perfect removal of **exposed** D2H save wall cannot reach 2.9126% of matched E2E wall (equivalent to a 3% throughput improvement). Do not claim the fork's 40 ms TTFT improvement is BigCherry's ceiling.

**Gate 1 — host/source comparison:** A = pinned per-range synchronous baseline; B = upstream #30694 same-tensor coalescing only; C = B plus optional bounded #189-style cross-tensor async batching. If the future pin includes B, do not re-port it. Preserve byte-exact output order, every range and source tensor, MIRRORED/axis0/1/2, multi-segment, arbitrary valid row offsets and original caller buffer. Enforce <=256 MiB pinned staging and bounded extra merge traffic; prove ownership of every tensor and backend and physical access to the pinned destination **before** enqueue. Do not add a second allocator, backend registry, scheduler, cache or permanent config. An experiment-only same-binary toggle must live in the existing patch framework and be removed if rejected.

    gather existing write_info[] and original output offsets
    if oversized, unsupported shape/backend or no safe pinned host type: original synchronous flush
    else obtain context-owned bounded pinned stage (no reuse while DMA outstanding)
         enqueue get_tensor_async(owner_backend, tensor, stage+prefix, offset, size) for each range
         synchronize every UNIQUE underlying physical backend once, including Meta children
         copy each staged slice to original output offset; publish completed save
    on partial async failure: join/cancel all outstanding work and FAIL the save
    never fall back to success with missing state after partial enqueue

The staging buffer must outlive all DMA, and state saves must not overlap on the same context without serialization. Keep RD58 H2D restore untouched. Account for fallback per-row sync behavior; a wrapper named async does not prove concurrency.

**Gate 2 — correctness/physical work:** Host split/offset byte fixture first, then pinned-source patch composition and test-llama-context frag/cp_h_s/cp_d_s on HIP/Meta with CPU/RPC negative controls. Exact serialized bytes vs stock, 48-token restored-vs-cold greedy, logits/KLD and MTP acceptance, >=10 same-process requests, multiple slots/interleaved KV, ubatches 1/4/16/512/2048, 8K/32K/98K/long context, tensor/layer split, cancellation and graph replay. Preserve 1005 selection and RD58 restore activation; preexisting 1005 warm/cold mismatch must not be attributed to this patch. Compare expected vs observed per-rank ranges/bytes/syncs, DMA bandwidth plausibility, peak pinned memory and host memcpy time. Faster because state disappeared is failure.

**Gate 3 — paired promotion:** Same-binary ABBA >=4 independent sessions per topology, warm multi-turn TTFT and 4-stream throughput, prefill/decode negative controls. CI95-low >=3% E2E throughput (or preregistered equivalent TTFT gate), <=1% unaffected-path regression, exact state and numerical parity, bounded host memory. Otherwise NO_GAIN, SINGLE_GPU_REGRESSION, META_ASYNC_UNSUPPORTED, STATE_FAIL or WAIT_UPSTREAM. No implementation or GPU queue until Gate 0.

## Consolidation and validation actually run

QFP54 solely owns state-save D2H batching. QFP48 owns checkpoint ablation and its separate batch-boundary/fusion evidence; QFP16/24/27 own scheduler **H2D inputs**; RD58/1234 owns **H2D restore**; 1005 owns checkpoint selection; MSM02/03 owns Meta allocation/placement. No neighbouring plan, patch or queue was changed.

Disposable host-only checkpoint_read_host_oracle.py: 1,500 randomized split-layout byte-equivalence cases (23,483 ranges), modeled 75-distinct-tensor/two-rank sync counts 150/150/2, byte accounting and cap fallback passed; py_compile passed. It is **not** a C++ Meta backend test or real synchronization measurement. No HIP build, GPU test, benchmark or first-party BigCherry speedup.
