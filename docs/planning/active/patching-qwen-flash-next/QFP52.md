---
id: QFP52
order: 52
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-10T21:23:48.693932+00:00'
breadth: ''
skill: advanced
created-by: claude
priority: P3
work: M
---

# Where the MTP hidden-state (NextN) output lives: owner device, copies and a placement ablation

## 2026-10-14 implementation audit — one-copy admission before placement (BCOP205)

**Disposition:** QFP52 owns only the residual **NextN host-output duplication/ownership proof**. Keep validated 1359 default-on and its fenced two-slot ordering. Do not implement the previously proposed `BIGCHERRY_MTP_TOPOLOGY=off|auto|device:N` placement selector: no source-correlated owner/copy census or exposed critical-path gain currently supports it. Treat the original Steps below as historical hypotheses; the gates here are authoritative. No new patch, benchmark or hardware queue was created by this audit.

### Current-pin source and data-flow contract (llama.cpp b11552, 23b0202a189c44a54625aadcb37a946dd1d6278)

1. `src/llama-context.cpp::llama_context::decode` extracts `t_h_nextn` with `ggml_backend_sched_get_tensor_backend` and `ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float))`. Unmasked rows use `offset=n_tokens_prev`; masked rows use `offset=n_outputs_prev`. `output_reserve` sizes `embd_nextn` to `n_batch*n_embd_out` for unmasked output and owns it within the normal output allocation. `get_embeddings_nextn`, `get_embeddings_nextn_ith` and `output_reorder` retain public/read-side semantics. **The original copy is not proven dead** just because the drafter normally consumes the fenced copy.
2. `engines/llamacpp/patches/1359_prefill_pipeline/patch.py::_N_C_EXTRACT` preserves that first D2H and, only for `bc_1359_on && !masked && offset==0`, enqueues a second `get_async` into `bc_nextn_buf[bc_slot]`, then records backend events. Two slots alternate; `bc_nextn_fence_wait(slot)` precedes reuse. `bc_nextn_slot_valid` is reset on noneligible batches. `bc_nextn_fenced` waits those events and exposes the slot. `common/speculative.cpp::bc_collect_outstanding` performs a **third transfer, CPU `memcpy`** into `bc_chunks[bc_write].h_nextn`, then `bc_process_snapshot` feeds the draft via `batch.set_embd_state`. Do not confuse D2H, host memcpy and drafter H2D.
3. `ggml/src/ggml-backend-meta.cpp::ggml_backend_meta_get_tensor_async` chooses simple backend **0 for MIRRORED** output; split-axis 0/1/2 uses per-rank `get_2d_async` splicing. The actual `t_h_nextn` split state, physical rank, device-to-host DMA bytes and stream must be observed. 1359 currently records events on **every** Meta child backend after the copy; this is a conservative all-rank fence, not proof all ranks transferred NextN. Do not remove fences until all producing/consumer dependencies are established.
4. `1348_mtp_deferred_catchup` owns deferred draft snapshot/poison/reset semantics; 1359 owns its fenced prefill pipeline. `tools/tests/patch/test_1359_prefill_pipeline.py` explicitly asserts first-copy-before-second-copy ordering, event-after-copy, two-slot ownership, flush and reset. Any one-copy prototype must replace those assertions with stronger public-output and lifetime tests, not simply delete the original copy.

**Byte accounting, not a performance measurement:** at Flash-Next `n_embd_out=10240`, 512 unmasked F32 rows occupy `512*10240*4=20,971,520 B = 20 MiB`. Eligible 1359 batches enqueue **two 20 MiB D2H requests** (40 MiB total), followed by one 20 MiB CPU snapshot. These are logical requested bytes, **not measured PCIe traffic or exposed wall time**; Meta MIRRORED may source rank0 only. At the historical ~1,300 tok/s two-XTX prefill control, 512 tokens take ~393.85 ms; achieving +3% throughput requires removing at least `393.85*(1-1/1.03)=11.47 ms` of **serialized** time per batch. This is a conditional arithmetic gate, not an observed copy cost or guaranteed upper bound.

### Ownership/consolidation and external comparison

- **Implementation:** extend only existing 1359/1348 and llama output APIs if a one-copy route survives; QFP52 is the technical qualification owner, not another pipeline. QFP16/1326 and QFP27/upstream [#29963](https://github.com/ggml-org/llama.cpp/pull/29963) concern **scheduler input** copies, not this output. QFP50's pipeline is validated; do not alter its ordering or add a second fence scheduler.
- **Placement/device handoff:** RD71 fork [41a8ca78](https://github.com/MrLordCat/llama.cpp-rdna-lab/commit/41a8ca7858bc9540415f37877dde5cc70e435f63) binds backend-resident target rows into a drafter input and calls `src->synchronize()` for separate contexts. Its reported Vulkan decode benefit and prompt result do **not** demonstrate prefill overlap or gfx1100/gfx1201 transfer feasibility. RD72 fork [1fcc05da](https://github.com/MrLordCat/llama.cpp-rdna-lab/commit/1fcc05daf22eec3203d17867c75b08a1d4385853) owns Vulkan NextN placement; do not recreate that selector here. vLLM's `vllm/v1/spec_decode/llm_base_proposer.py` forwards `target_hidden_states` as tensors to its drafter, but its device/runtime ownership differs from BigCherry's two-XTX target + gfx1030 draft behind the chipset. With no normal P2P, do not assume a GPU-resident cross-context handoff removes the host staging path.
- Upstream [#30020](https://github.com/ggml-org/llama.cpp/pull/30020) **merged 2026-10-06** and sets `sched_need_reserve` when NextN extraction mode changes; already part of b11552, so superseded 1343 must not be revived. [#30160](https://github.com/ggml-org/llama.cpp/pull/30160), merged 2026-10-09, reorders input `GET_ROWS` to avoid extra graph splits; it is not an output-copy solution. No external throughput claim is transferable to this AMD topology.

### Bounded decision sequence — no new standing infrastructure

**Gate 0, read-only discriminator:** In an existing Flash-Next two-XTX + 6900-XT-drafter production composition, correlate per ubatch `{graph_uid,request_id,ubatch,masked,offset,n_rows,n_embd,physical rank UUIDs,split axis,source backend,copy destination,bytes,copy enqueue us,event-record us,event-wait us,bc_collect_outstanding memcpy us,draft upload bytes}` with rocprof copy events and the existing 1346 prompt-timing receipt. Use existing trace hooks, not a new telemetry service. Compare `BIGCHERRY_PREFILL_PIPELINE=1` vs `=0` and no-drafter control at 8K/24K/98K, ub512 (ub1024 only if memory-safe). Separate host enqueue, actual GPU copy, CPU memcpy, all-rank fence and exposed idle on the **end-to-end** critical path. Require same GGUF/build/ROCm, rank UUID order, context and MTP depth; missing identity => `INELIGIBLE`. If an optimistic removal of only the second D2H (and any proven associated wait) cannot yield >=3% E2E, close `BELOW_CEILING` without code or hardware queue. Do not mistake the 1359 pipeline's measured +6..10% gains for a duplicate-copy saving.

**Gate 1, host/graph contract before any HIP change:** Prove the output accessor and allocation contract for both `get_embeddings_nextn()` and `get_embeddings_nextn_ith()`, `output_reorder`, masked/unmasked, `offset>0`, multi-ubatch, slot-wrap, consumer holding previous slot, context shift/cancel/reset, graph rebind and buffer reallocation. A naive `embd_nextn.data = bc_nextn_buf[slot]` alias is **not safe** without proving output-reserve ownership and all API readers. Reject `NO_SAFE_ALIAS` if this needs another global output allocator or scheduler. Retain original copy and current two-slot events as fallback on any noneligible case. Event-count reduction requires independent split-state and stream-dependency proof.

**Gate 2, conditional minimal implementation:** Only if Gate 0 clears >=3% and Gate 1 passes, prototype a **default-off, lab-only** one-D2H path inside 1359, reusing the existing two fenced buffers for eligible `!masked && offset==0` rows. Preserve public NextN output via a proven lazy accessor/owned view or retain the original copy if accessed; no new permanent `BIGCHERRY_MTP_TOPOLOGY` selector, placement solver, allocator, cache or backend transport. Document exact pointer generation and when the original output is materialized. Fail closed to current 1359 on unsupported shape, device, events, allocation or mixed request.

**Gate 3, promotion/rejection:** First source-shaped host fixture, then pinned patch-composition/mechanics tests and C++/HIP build. Paired one-binary lab-toggle ABBA, >=4 independent server sessions and >=10 paired rounds each at 8K/24K/98K; same-process >=3 requests, alternating ubatches, cancellation/shift, masked/decode controls, MTP4 fixed draft and greedy determinism. Require byte-identical NextN rows and accepted counts, full-vocab logits/KLD/greedy parity, no lost output/copy work, no use-after-free/slot overwrite, bounded pinned bytes and physically plausible PCIe traffic. Promote only CI95-low >=3% prefill/TTFT E2E with <=1% decode/control regression; else `NO_EXPOSED_GAIN`/`NUMERIC_DRIFT`/`LIFETIME_FAIL`. Do not queue experiments competing with active RAD03/RAD04 hardware work.

**Audit-only validation:** A local source-shaped Python discriminator checked 10 assertion sites, including 20 two-slot iterations, 20 MiB logical row sizing, negative admission cases and the 11.47 ms conditional gate; Python compilation passed. It does **not** execute repository tests, compose patches, build HIP or exercise GPUs. Existing measured 1359 prefill +6.0% (8K), +7.4..10.7% (24K), +10.0% (98K) is pipeline evidence only, not one-copy evidence. No new performance result.


## Description

Split out of QFP42 (closed 2026-10-11 as done by 1359), where it had been folded in as the PRBE57 follow-up. Not done, and not part of the barrier removal.

The target hands the drafter one row of hidden state a token (`t_h_nextn`, n_embd_out = 10240 wide on Flash-Next: about 21 MB a 512-token batch). With the tensor split it is not established which device owns `NEXTN_PROJ_PRE/POST` and `t_h_nextn`, how many device-to-host and host-to-device copies a batch costs, or whether a different owner would be cheaper. Since 1359 the rows are copied out twice a batch (llama's own output buffer and the fenced pinned buffer).

What is known: the same prefill with no drafter at all is +14.6% (run nomtp1) and 1359 recovers +7 to +13% of that by reordering; part of the rest is the hidden-state output itself. The external reference (fork commit 1fcc05da) changes Vulkan NextN placement and is not a HIP implementation; 41a8ca78 is a separate backend-resident hand-off (Vulkan, parked as PRBE58).

## Steps

1. Measure in the production HIP / Meta layout: the owner device of the NextN tensors, the real D2H copies and their sizes and stream, and the drafter-side upload.
2. Remove the duplicate copy if 1359's fenced buffer can serve llama's own readers (decode-phase callers read the masked output through the original buffer today).
3. Only if step 1 shows a copy or ownership bottleneck: a default-off placement ablation `BIGCHERRY_MTP_TOPOLOGY=off|auto|device:N`, costed as copy-in + copy-out + synchronise + host staging.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Byte-identical NextN rows, greedy identity, accepted counts, copies a token and no extra VRAM pressure; ABBA on one binary. Placement is not to be conflated with the asynchronous hand-off (1359).

## Effort & Risk



## Standards



## Acceptance Criteria

- Owner device, copy count and bytes a batch are recorded for the production layout.
- The duplicate copy is removed or shown to be needed.
- The placement ablation is run only on the evidence of step 1, and promoted or rejected with its measurement.

## Notes

Origin: PRBE57, folded into QFP42 on 2026-10-08, split out again 2026-10-11 on the owner's instruction. The third thing QFP42 carried (PRBE07, the BridgeSpec review) was closed as inspiration only; its one live idea, kernels for verify widths 2..8, belongs to QFP38 and is noted there.

## Change Log

- 2026-10-10T21:23:48.693932+00:00 (created-by): Created by claude
