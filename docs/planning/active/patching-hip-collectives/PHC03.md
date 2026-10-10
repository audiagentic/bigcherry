---
id: PHC03
order: 0
plan: patching-hip-collectives
state: pending
created-at: '2026-09-10T02:11:29.105053+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# Fix META segfault for D=3+ heterogeneous no-P2P groups (ggml_backend_cuda_cpy_tensor_async)

## Description

RU01's real-hardware D=3 heterogeneous correctness qualification found META (GGML_HIP_REDUCE_PLAN=meta) segfaults (SIGSEGV) rather than falling back gracefully, when a SPLIT_REDUCE graph spans 3+ devices with no PCIe P2P access. HI84 already proved META works correctly for D=2 no-P2P pairs on this exact hardware; the fold/copy-back path apparently has an assumption that breaks specifically at N>2.

## Steps

1. Reproduce: build the current test-hip-reduce probe, run `GGML_HIP_REDUCE_PLAN=meta test-hip-reduce --case <any D=3 case> --plan meta --devices 0,1,3 --out result.json` on a no-P2P 3+ GPU box -- crashes with SIGSEGV.
2. Get a symbol-resolved backtrace (gdb -batch -ex run -ex bt) into ggml_backend_cuda_cpy_tensor_async / ggml_backend_meta_graph_compute's fold/copy-back lambda -- the observed crash is inside opaque HIP driver code (libamdhip64.so, no symbols) called from that copy path, so isolate exactly which copy (host-staged fallback vs peer-direct) is being attempted for device index >1 in a no-P2P group.
3. Determine whether the fold/copy-back logic assumes a fixed D=2 topology (e.g. a single pairwise host-staging buffer reused incorrectly for a 3rd participant) or whether it's a genuine missing-fallback case for N>2 no-P2P.
4. Fix the root cause -- likely in ggml_backend_meta_graph_compute's copy-back lambda or the cross-device cpy_tensor_async call it drives.
5. Re-run RU01's D=3 correctness qualification (same driver/devices/cases) to confirm the fix -- both {0,1,3} and {0,1,2} must pass cleanly before this closes.
6. Only then re-attempt D=4 qualification (RU01's own next stage, blocked until this fix lands and D=3 passes).

## Detailed Solution & Technical Design

Real crash evidence (2026-09-10, Brutus, build_plan_id 0bfc43c12b47fd79b26a80c14b656b72, AMDGPU_TARGETS gfx1100;gfx1201;gfx1030):

Thread 1 "test-hip-reduce" received signal SIGSEGV.
#0 in ?? () from /opt/rocm-7.2.4/lib/libamdhip64.so.7
#1-#4 more opaque libamdhip64 frames
#5 ggml_backend_cuda_cpy_tensor_async(ggml_backend*, ggml_backend*, ggml_tensor const*, ggml_tensor*)
#6 ggml_backend_tensor_copy_async
#7 ggml_backend_meta_graph_compute(...)::$_4::operator()(...)::{lambda}::operator()
#8 ggml_backend_meta_graph_compute(ggml_backend*, ggml_cgraph*)
#9-#11 scheduler/main

Reproduced twice on two different D=3 device sets ({0,1,3} and {0,1,2}), both segfault identically -- not a one-off/flaky hardware issue. AUTO's RCCL arm for the same D=3 groups hits the ALREADY-KNOWN separate RCCL SIGABRT (device 3/RX 6900 XT incompatible with RCCL/NCCL) -- that is not this item's concern, already tracked as a permanent RCCL exclusion.

## Code Samples & Guidance



## Files

src/ggml/src/ggml-cuda/ggml-cuda.cu (ggml_backend_cuda_cpy_tensor_async, referenced in the crash's own error path at line ~1088 for the separate RCCL issue -- confirm exact cpy_tensor_async location); wherever ggml_backend_meta_graph_compute's fold/copy-back lambda lives (likely ggml-backend-meta.cpp per HI84's own file references).

## Validation

Fixed build must pass RU01's D=3 META qualification cleanly (no crash, numerically correct output within tolerance) on both {0,1,3} and {0,1,2} device sets, using the same tools/bigcherry/tuning/reduction.py-based driver RU01 used. Do not claim the fix without re-running real hardware evidence -- this is a segfault fix, not a design change; a null/inconclusive result is not acceptable here, only clean pass or a documented different real failure.

## Effort & Risk



## Standards



## Acceptance Criteria

META SPLIT_REDUCE completes without crash for D=3 heterogeneous no-P2P groups, numerically verified correct against the CPU-double oracle, on real hardware. D=4 qualification (RU01's blocked next stage) may only proceed after this closes.

## Notes

Filed from RU01's real-hardware finding (2026-09-10) rather than attempted as part of RU01 itself -- RU01 is explicitly run-only/correctness-evidence scope ('not a new collective implementation'), this is real source-level patch work belonging to patching-hip-collectives per this project's Run-vs-Patch scope separation (matches RU01's own notes: 'Any implementation or patch change belongs under patching-hip-collectives, not this Run item').

## 2026-10-10 implementation audit (BCOP114)

**Evidence boundary.** RU01's Brutus ROCm 7.2.4 Meta D=3 SIGSEGV on both {0,1,3} and {0,1,2} is a recorded hardware failure; the D=2 control passed. The precise cause is **not established**. Neither the lack of peer access nor the Meta butterfly alone proves the crash mechanism. No new HIP execution was performed in this audit.

**Pinned b11474 source path.** `ggml/src/ggml-backend-meta.cpp::ggml_backend_meta_graph_compute()` calls `allreduce_fallback()`; its `push_data()` copies partials with `ggml_backend_tensor_copy_async()` into destination-owned `set_tmp_data(j_dst, i_buf)` buffers, then queues the ADD. The final copy-back uses the same copy API. `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_backend_cuda_cpy_tensor_async()` and `ggml_backend_cuda_buffer_cpy_tensor()` use `cudaMemcpyPeerAsync` for cross-physical-device transfers unless the build defines `GGML_CUDA_NO_PEER_COPY`. The copy dispatch does not check `cudaDeviceCanAccessPeer`; that query is used separately by optional `GGML_CUDA_P2P` peer-access enabling in `ggml_cuda_init()`. A peer-copy driver/runtime failure is a **testable hypothesis**, not a demonstrated cause. HIP peer copies may sometimes be staged despite no direct P2P.

**Existing safe fallback.** `ggml/src/ggml-backend.cpp::ggml_backend_tensor_copy_async()` synchronizes both backends when the backend-specific async copy returns false, then calls `ggml_backend_tensor_copy()`; after an unsupported buffer copy, that routine stages via host `tensor_get`/`tensor_set`. A potential fix must gate **both** CUDA copy paths; otherwise the generic fallback can re-enter the blocking peer-copy path. Preserve existing temporary-buffer ownership and lifetime; do not introduce another transport, allocator, scheduler or event ring.

**Host-model transfer accounting (not measured):** D=2 has butterfly 0→1,1→0 (2 cross-device copies); D=3 has fold 2→0, butterfly 0↔1, copy-back 0→2 (4 copies); D=4 has two butterfly stages (8 copies). If host staged, these imply 4/8/16 logical GPU↔host legs of `S=ggml_nbytes(node)` each, excluding ADD and scratch traffic. No bandwidth or speedup is inferred; the gfx1030 PCIe x4 lane is a separate potential bottleneck.

**Cheapest next discriminator (before patches or model benchmarks):**

1. Freeze exact RU01 input/driver/ROCm, physical ordinal mapping, pinned source and patch manifest. Make isolated, otherwise identical A=default and B=`-DGGML_CUDA_NO_PEER_COPY=ON` builds; record `CMakeCache.txt`, binary SHA and build flags. This is a **compile-time** switch, not an environment toggle. Use fresh processes and bounded timeouts.
2. Reuse the existing `engines/llamacpp/overlay/tests/test-hip-reduce.cpp` (its current source **already accepts D=2..4**) and `tools/bigcherry/tuning/reduction.py` CPU-double oracle. Run D=2 {0,1} control and original D=3 {0,1,3}/{0,1,2} cases, then reversed orders. Record per-arm exit/signal/timeout, stderr, backtrace, all-rank result and explicit selected copy path. Do not treat no result as pass. RCCL/AUTO gfx1030 is not a valid D=3 control because its incompatibility is separately recorded.
3. A crashes/B passes: preserve B as safe correctness fallback, instrument the exact failing copy edge using existing patch 1242. Only then consider a default-off, minimal physical-device-pair safety gate in **both** copy functions, returning false into existing host fallback. `cudaDeviceCanAccessPeer==0` is not sufficient to infer HIP peer-copy failure. A and B crash: reject peer-copy hypothesis and trace Meta fold/butterfly/copy-back, event ordering, scratch lifetime and driver with 1242/gdb. Both pass: rerun original frozen environment before declaring resolution.
4. Gate D=3 on all-rank CPU-double F32 correctness (zero/alternating/cancellation patterns, finite checks), repeated same-process requests, multi-ubatch, reordered physical ordinals and graph replay. Only then unblock RU01/HI84 D=4. If used by a model, add full-vocabulary logits/greedy/MTP acceptance and long-context controls.
5. After correctness, attribute host H2D/D2H traffic, synchronization, wall time and physical PCIe bandwidth; compare to the known-safe control. Promote any optional performance change only after four independent sessions, ten paired ABBA rounds each, CI95-low >=3% E2E and no >1% control regression. Close a speculative optimisation when its critical-path ceiling is <3%.

**Consolidation/ownership.** PHC03 owns the copy-safety decision. RU01/HI84 own D=3/D=4 hardware correctness; patch 1224 owns the native probe; patch 1242 owns stage tracing; PGC12 owns provider evidence; PGC14 owns unrelated RCCL transport. PHC01/Flash-Next is protected active work and remains untouched. Do not duplicate test harnesses, scheduler, buffer pool or telemetry.

**External mechanisms.** [llama.cpp #21648](https://github.com/ggml-org/llama.cpp/issues/21648) describes HIP non-P2P peer-copy corruption and the `GGML_CUDA_NO_PEER_COPY` workaround; it does not establish this D=3 SIGSEGV. Current upstream retains the copy/compile-guard pattern. [vLLM custom AllReduce](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/custom_all_reduce.py) gates >2 PCIe-only ranks and checks P2P outside ROCm, so its policy cannot simply be ported. [SGLang PCIe-IPC](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/device_communicators/pcie_ipc_ar.py) uses an independent workspace and is not qualified for this topology.

**Validation actually performed:** 12/12 pinned-source static checks; disposable deterministic D=2/3/4 route model passed 3 expected-edge fixtures, 2 extra-participant assertions and 3 non-self-copy checks. No C++/HIP build, GPU run, repository pytest, new benchmark or new measured speedup.

## Change Log

- 2026-10-10 (BCOP114): pinned copy-path and Meta route audit; build-time no-peer discriminator; bounded D=3 correctness and D=4 dependency gates. No code/hardware work.

- 2026-09-10T02:11:29.105053+00:00 (created-by): Created by agent
- 2026-09-10T02:11:53.757768+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260910_021202_ran-a-real-multi-gpu-correctne_4533
- 2026-09-10T02:12:02.089512+00:00 (updated-by): Updated: section:ledger-events
