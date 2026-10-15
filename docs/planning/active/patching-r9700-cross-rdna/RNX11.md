---
id: RNX11
order: 11
plan: patching-r9700-cross-rdna
state: pending
created-at: '2026-10-03T01:33:52.132134+00:00'
breadth: ''
skill: advanced
created-by: codex
work: L
priority: P1
---

# R9X11 — Custom AllReduce fixed-grid, sequence-counter, and graph-replay hardening

## Description

Audit and harden custom collectives whose scratch/epoch state can be reused across calls of different sizes or graph replays. The invariant is fixed maximum grid plus sequence advancement by every block on every call, with inactive blocks doing no payload work.

## Steps

- Map r9k fixed-grid protocol and race reproducer to BigCherry 1252/1291 provider sections.
- Record generation source, graph-capture behavior, scratch parity, size-dependent grid, inactive block progression, wraparound comparison, and stream/system ordering for each provider.
- Add a lab reproducer under tools/lab/rccl alternating 10 KB/3.3 MB/40 KB/6.5 MB and dynamic serving-like joins; run RCCL, cpu-root, adaptive/internal, and P2P with graph replay where supported.
- Fix only affected providers with fixed-grid advancement or an equally proven invariant; add debug trace/assertions without production host sync.
- Preserve 1291 device-derived small-path generation and audit the large path separately.

## Detailed Solution & Technical Design

Before any implementation, require a graph-safe collective protocol: a device-derived monotonically advancing sequence on every replay, no host-generated progress in captured kernel arguments, fixed rank participation, exact zeros for uncomputed PARTIAL outputs matching the 1291 lesson, fixed-grid behavior that does not depend on host progress, and specified teardown/re-entry/wraparound semantics. Acceptance must be an in-model prefill using the 96-AR pattern; standalone latency remains diagnostic only.

## Code Samples & Guidance



## Files

- kernels/r9k_ar.hip
- r9700_vllm/comm/r9k_ar.py
- tests/test_ar_race.py
- ggml/src/ggml-cuda/ggml-cuda.cu
- tools/lab/rccl/

## Validation

Thousands of alternating-size calls, wraparound-near counters, 2/3/4 ranks, dynamic joins/leaves, graph capture/replay, restart, mixed gfx11+gfx12, exact CPU/RCCL comparison after every call, and sanitizer/debug checks.

## Effort & Risk



## Standards

Do not impose fixed-grid changes on RCCL; preserve provider-specific ownership and fail closed on ambiguous evidence.

## Acceptance Criteria

- Complete only when an affected provider fails pre-fix and passes post-fix, or every custom provider is proven not vulnerable and the regression test is retained.
- Promotion of RNX07 is blocked until this item is resolved for the relevant protocol.


## 2026-10-15 protocol/lifetime audit — 1291 large-path replay (BCOP221)

**Disposition and novelty.** RNX11 is a bounded correctness gate for existing collective protocols, **not** authority to create patch 1311, a new fixed-grid kernel, scheduler, cache, allocator or hardware campaign. Last independent RNX11 planning was 2026-10-03; no same-path implementation, PR, hardware lane or queued run was identified in the preceding 12 hours on `feat/men10-engine-cli`, `feat/rad03-gpu-lanes` or main. Excluded contemporaneous MEN10 engine CLI/source isolation, RAD02–RAD09 helper/mailbox/placement, RWQ transport codec and PGC15 tiling. None of those paths is changed.

### Source-level finding: generation ownership differs between small and large

Current `engines/llamacpp/patches/1291_ar_cpu_root/patch.py::_IMPL` (composed `ggml/src/ggml-cuda/bc-ar-cpu-root.cuh`) has two **different** protocols:

- **Small F32 (<=64 KiB default):** `bc_cpu_root_produce<<<1,1024>>>` increments each rank's `ctr->v` **on the device** at execution time. `bc_cpu_root_worker` waits for every rank's `arrive[r].v`, sums in rank order, publishes `done->v`, and `bc_cpu_root_consume` reads the device counter to choose one of two result slots. There is no variable per-block grid to repair in this path. The device-derived generation is a valid *design for* graph replay; real replay correctness remains a hardware gate.
- **Large opt-in copy-engine path:** `ggml_backend_cuda_comm_try_allreduce_cpu_root` computes `const uint32_t gen = ++cr->l_gen` on the **host**, writes `l_ring[gen % 1024]`, publishes `l_posted.store(gen)`, then passes that host value into `hipStreamWriteValue32(l_d2h,...,gen)` and `hipStreamWaitValue32(l_h2d,...,gen,hipStreamWaitValueGte)`. `bc_cpu_root_large_worker` advances `want` only when the host publishes a new descriptor. **Conditional correctness risk:** if this stream sequence can be captured and replayed, replay reuses the captured `gen` but does not rerun the host enqueue or descriptor publication; a prior `l_reduced[c] >= gen` may release the H2D wait with a stale result. If HIP rejects capture, the issue is safely blocked; neither behavior has been exercised on hardware here. No capture-state admission check appears in the patch.
- The large path is **default off** (`BIGCHERRY_AR_CPU_ROOT_LARGE_MAX_BYTES` absent => 0); `ggml_backend_cuda_comm_init_cpu_root` retains RCCL for non-small calls. Historical 32 MiB large-host experiment regressed prefill ~1450 -> 1060 t/s (about -26.9%); no new measurement or production activation is inferred.
- **Do not assert an ordinary-stream staging race without evidence.** Each rank records `l_done` on its H2D stream and makes the same backend compute stream wait before its next `l_ready`; that dependency can prevent next-call D2H overwriting an unfinished result. `l_finished` tracks CPU reduction, **not** GPU H2D completion. A multi-stream/rebind/capture path must be proven before claiming missing cross-call lifetime. Existing 1024-descriptor/512-in-flight bound does not substitute for an H2D consumption fence if such a path exists.

`tools/tests/patch/test_1291_ar_cpu_root.py` checks patch application/idempotence/guards, not generation progress under graph replay. Existing `tools/lab/rccl/cpu-root-ar.hip`, `large-host-ar.hip`, `tools/bigcherry/tuning/reduction.py` and the normal patch tests are the only permitted experiment/test owners. `1252_nro03_allreduce_p2p_provider` has separate bidirectional content-probed two-rank P2P ownership; `1275` changes the stock two-rank event-slot wait and `1276/1244` owns unqualified root3. Do not apply RNX11's historical fixed-grid sketch indiscriminately to RCCL or these unrelated protocols.

### Cheapest discriminator and implementation-ready gate

1. **No-GPU source/recipe check:** pin branch SHA, llama.cpp source pin, `1291` patch digest and exact `--allreduce`/environment. Verify large default-off, device vs host generation sources, stream event chain, actual per-rank device mapping, and absence/presence of a capture gate. Record `CAPTURE_NOT_REACHED` when no valid large-path route is present. Do not enable large path in production.
2. **Disposable capture gate (only if a safe GPU slot becomes available):** run a minimal existing-lab control with RCCL and small cpu-root, then opt-in large on 2x gfx1100 before mixed 2x gfx1100+gfx1201. Explicitly record whether HIP stream capture **rejects** the memop/event sequence or succeeds. If rejected, `CAPTURE_REJECTED_SAFE`: no implementation. If captured, replay the same graph at least twice with changed input patterns while checking `l_gen`, `l_posted`, worker `want`, per-chunk arrival/reduced generations, all-rank output and timeouts; a stale/reused result is `REPLAY_STALE_RESULT`. No new production tracing subsystem: temporary bounded counters/logs under the existing lab only.
3. **Stream/lifetime negative controls:** alternate small 10/40 KiB and large 3.3/6.5 MiB sizes (within configured cap), different compute masks including zero-contributing ranks, reversed physical ordinals, repeated same-process requests, cancellation/restart and graph reallocation. Compare same-stream control with any **observed** multi-stream/rebind path. Prove `D2H(g+1)` cannot overwrite staging before CPU reads `g` and `CPU_publish(g+1)` cannot overwrite `l_result` before H2D consumes `g`; never infer these edges from `l_finished` alone. Use CPU-double F32 oracle, exact rank results, finite checks and no missing work.
4. **Minimal remedy only on positive evidence:** if capture succeeds and fails, prefer a fail-closed **large-path capture admission/fallback to RCCL** using existing provider selection, unless a bounded device-owned replay generation and consumption-lifetime proof is cheaper. No new fixed-grid implementation. Re-run the failing control plus graph replay, multi-ubatch, long-context, MTP acceptance/logits, allocator/rebind and 2/3-rank no-P2P. Do not route gfx1030 into unsupported RCCL; use only separately qualified provider controls.
5. **Performance gate after correctness:** compare RCCL and any safe opt-in large path in >=4 independent sessions of >=10 paired rounds, with explicit executed-provider/bytes/critical-wall receipts. Promote only if CI95-low >=3% E2E prefill/TTFT, <=1% decode/control regression, bitwise F32 collective parity and full inference correctness. If exposed collective wall is <3% or the existing negative large-path result persists, `REJECT_LARGE` and retain RCCL.

**Terminal outcomes:** `NO_LARGE_ROUTE`, `CAPTURE_REJECTED_SAFE`, `REPLAY_STALE_RESULT`, `STREAM_LIFETIME_UNPROVEN`, `NUMERIC_FAIL`, `REJECT_LARGE`, or `QUALIFIED_OPT_IN`. Do not queue a new hardware job while existing owners have an experiment in flight. RNX09 consumes only a terminal, evidence-backed outcome; RNX07 is absent from the current active RNX folder, so the historical RNX07 promotion blocker must not be treated as a live patch owner.

**Upstream/forks (source inspection, not transferred gains):** current [llama.cpp allreduce.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/allreduce.cu) has its own two-rank host-mapped per-slot event ring, explicit `cudaEventSynchronize` before slot reuse and host-supplied token; it is not a replacement for 1291's CPU-worker generation contract. [vLLM custom_all_reduce.cuh](https://github.com/vllm-project/vllm/blob/main/csrc/custom_all_reduce.cuh) registers graph buffers and owns rank signals; [SGLang custom_all_reduce.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/device_communicators/custom_all_reduce.py) gates supported topology and registers HIP graph buffers. Neither supplies a tested no-P2P large-host worker fix. The upstream [llama.cpp #26827](https://github.com/ggml-org/llama.cpp/pull/26827) MTP multi-ubatch serialization discussion is a *separate* scheduler/lifetime control, not proof of this failure.

**Validation performed in this audit:** 12/12 live `1291` source-marker checks and 9/9 deterministic **host-only** generation/ordering fixtures passed, including a synthetic stale-result replay and a same-stream-fence negative control. No HIP build, capture, GPU test, repository test suite, new throughput result or measured regression. The fixture establishes a conditional graph-generation hazard, **not** a reproduced production failure.

## Notes

Original source alias is R9X11. Proposed slot 1311. Current Flash-Next evidence identifies 1291 large path as not graph-safe and default-off; this is an open audit input.

Verbatim legacy source retained during R9X→RNX migration:

# R9X11 — Custom AllReduce fixed-grid, sequence-counter, and graph-replay hardening

Status: planned / correctness-first
Proposed patch: `1311_r9x_collective_fixed_grid_sequence`
Depends on: R9X01; must be reviewed before promoting R9X07 or any custom provider with per-block epochs/double buffering
External source: `kernels/r9k_ar.hip`, `r9700_vllm/comm/r9k_ar.py`, `tests/test_ar_race.py`
Existing BigCherry owners: `1252_nro03_allreduce_p2p_provider`, `1275_ar_small_latency`, `1276_ar_adaptive_nway`, `1277_ar_size_trace`, `1291_ar_cpu_root`; related graph correctness: 1233/1216



GPT design pass (req_83e7cdc000be4b9e): no 1311 package now. The 1291 large CPU-root path uses a host-side generation counter and is not graph-capture safe as-is; audit provider-by-provider and resolve RNX07 transport ownership before proposing a successor.

## Goal

Audit and harden every BigCherry custom collective whose scratch buffer or epoch/sequence state can be reused across calls of different sizes or graph replays. The motivating r9700 bug is protocol-level and portable across RDNA generations: changing active block count between calls can make different blocks advance different sequence parity and select different halves of double-buffered scratch.

## Source behavior to reproduce as a regression test

`r9700_vllm/comm/r9k_ar.py` documents the fix: every collective launches a fixed maximum grid and **all blocks advance their sequence counter on every call**, while blocks outside the payload do no data work. `tests/test_ar_race.py` reproduces serving corruption when a prompt joins running decode sequences. `kernels/r9k_ar.hip` is the device-side protocol reference.

## Exact BigCherry mapping

Audit `ggml/src/ggml-cuda/ggml-cuda.cu` sections inserted/modified by 1252 and 1291. For each custom provider record:
- where generation/epoch is produced (host or device);
- whether graph capture freezes a host-side value;
- scratch buffer parity/slot selection;
- launch grid as a function of message size;
- whether inactive ranks/blocks still advance protocol state;
- wraparound comparison semantics;
- stream/system-scope ordering.

1291 already deliberately derives its small-path generation on device for graph replay; preserve that design and audit its large-path generation/ring separately. Do not impose a fixed-grid change on RCCL, which owns its own protocol.

## Implementation

1. Add a lab reproducer under the existing `tools/lab/rccl/` area: alternate 10 KB/3.3 MB/40 KB/6.5 MB calls, then dynamic serving-like sequences where a large prefill joins repeated decode reductions.
2. Run it against RCCL, cpu-root, adaptive/internal and P2P providers, with graph capture/replay where supported.
3. If any BigCherry provider has per-block sequence state and size-dependent grid, convert to fixed-grid counter advancement or an equivalently proven invariant. Payload work remains bounded to active data blocks.
4. Verify uncomputed meta ranks still contribute zero and participate in protocol progression exactly as required.
5. Add trace counters/assertions in debug/evaluation mode for epoch divergence; no per-call host synchronization in production.

## RDNA adaptation

This is `common` protocol correctness. gfx12/gfx11/gfx103x must obey the same sequence invariant; only system-scope atomic primitives/launch tuning may differ. Mixed-generation ranks are mandatory because protocol agreement matters more than local ISA.

## Validation/acceptance

Stress thousands of alternating-size calls, wraparound-near counters where practical, 2/3/4 ranks, dynamic request joins/leaves, graph capture/replay, context/server restart, and mixed gfx11+gfx12. Compare outputs to RCCL/exact CPU reduction after every call, not only final generation. Run under sanitizer/debug checks where supported.

1311 completes only when either (a) an affected provider is fixed and the reproducer fails pre-fix/passes post-fix, or (b) every custom provider is proven not vulnerable and the regression test is retained as evidence. Promotion of R9X07 requires this item resolved.

## Change Log

- 2026-10-03T01:33:52.132134+00:00 (created-by): Created by codex
- 2026-10-03T01:39:34.483874+00:00 (updated-by): Updated: section:notes
- 2026-10-03T02:18:47.155191+00:00 (updated-by): Updated: section:detailed_solution, section:notes

## Reviews

- RV4210
