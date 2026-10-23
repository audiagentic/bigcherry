---
id: PNRO03
order: 0
plan: patching-nasone-rdna-optimizations
state: done
created-at: '2026-09-09T10:52:12.743676+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P0
---

# HIP internal-AllReduce P2P provider — rejected on dev

## Authoritative disposition (2026-10-18; BCOP292)

**Done / REJECTED for the current dual-gfx1100 dev topology.** Patch `1252_nro03_allreduce_p2p_provider` is `state = "rejected"`, its SUMMARY records a 2026-10-07 hardware fault and non-composition at b11474, and its dedicated experiment/queue script were removed. Do not revive, harden, benchmark or schedule this transport on the current machine without materially new driver/kernel/topology evidence. This is a **terminal consolidation**, not a measured performance improvement. Prior design and build history remain in Git history. PNRO18's separate probe-hardening request is superseded here.

### Verified implementation and why probe hardening alone cannot promote it

- `engines/llamacpp/patches/1252_nro03_allreduce_p2p_provider/patch.py` ports nasone `7c5bb5cb` to `ggml/src/ggml-cuda/allreduce.cu`. `ggml_cuda_ar_pipeline_init` checks `GGML_CUDA_AR_P2P` (default 0), both `cudaDeviceCanAccessPeer` directions, both `cudaDeviceEnablePeerAccess` calls and `ggml_cuda_ar_p2p_probe`. Only then does `p->p2p_enabled` select `ggml_cuda_ar_allreduce_p2p_impl`; otherwise the native host-staged `ggml_cuda_ar_allreduce_copy_impl` remains.
- The P2P implementation records producer events, issues each `cudaMemcpyPeerAsync` from a **source-device-owned** `p2p_stream[direction]` into destination `p->dev_tmp[destination]`, records `p2p_done`, then waits before the local reduction. `dev_tmp_kernel_done` fences reuse. The two directional streams/events are destroyed by the existing pipeline teardown. No new transport scheduler, allocator or dispatch table is justified.
- The startup probe allocates **separate temporary** `dev_src/dev_dst` arrays, tests 4,096 / 65,792 / 1,048,576 / 4,194,304 bytes, both directions, two passes, with nonzero asymmetric byte patterns and a stream synchronize/readback. It does **not** exercise the actual `dev_tmp` lifetime, concurrent production copy/add event ordering, outer chunk boundaries, or the `copy_bytes` dispatch threshold. It does not cover `threshold-1/threshold/threshold+1` for 64 KiB or 1 MiB. This is a **coverage gap in a rejected candidate**, not a demonstrated cause of the observed fault.
- `tools/tests/fixtures/hardware/p2p_copy_correctness_test.cpp` and `tools/tests/hardware/test_p2p_copy_correctness.py` already own permanent gfx1100 byte-pattern regression coverage, but test `hipMemcpy(..., hipMemcpyDefault)` source-current versus destination-current. They **do not qualify** 1252's async peer-copy/stream/event implementation. Do not create a duplicate hardware suite.
- `engines/llamacpp/patches/1252_nro03_allreduce_p2p_provider/evidence/validation.json` has four `failed-activation` records: the P2P trace marker was not observed in subject or control. `docs/evidence/lab-run-summaries/ab-27b-p2p.json` reports six-run host/P2P-requested throughput that is essentially equal: pp4096 1138.38 versus 1132.42 t/s; tg512 95.74 versus 95.89 t/s. **Without an activation receipt these are not measurements of actual P2P speed.** The separate 2026-10-07 fault/rejection remains authoritative; no gain is claimed.
- AMD HIP's current API documents peer-capability/enable and async peer-copy calls, and recommends the source-attached stream for peer copies; this matches the adaptation but does not guarantee working dev PCIe peer routing. Upstream llama.cpp `allreduce.cu` still implements host-staged copy-engine transport. External dual-R9700 P2P reports and gfx1100 RCCL/IPC stalls are topology/runtime-specific, not a reason to reactivate 1252.

### Ownership and non-duplication

**PNRO03/1252** owns the rejected P2P provider and any future requalification. **PNRO18** is closed as a duplicate hardening subtask. **PGC17** owns independent host/kernel/ACS/P2P transport feasibility; **PGC14** owns RCCL SHM copy-engine behaviour; **PGC09/12/13** own provider and threshold evidence; existing hardware fixture owns `hipMemcpyDefault` correctness. No edits to RAD06 helper/replay, MEN release/qualification, RCCL, GPU kernels or experiment queues.

### Conditional reopening gate — only after new external evidence

1. Require a *changed* physical peer topology, installed kernel/ROCm driver or peer-mapping mechanism, plus an exact build/device/PCIe receipt. A mere `hipDeviceCanAccessPeer=1`, successful API return, synthetic startup probe, or external R9700 speedup is **not** admission. Keep 1252 rejected/default off until a new qualification is completed.
2. First run the **existing** hardware correctness fixture and a disposable, source-shaped `hipMemcpyPeerAsync` probe on both gfx1100 directions. Reuse the exact production source-owned streams, destination `dev_tmp`, event waits and slot reuse; test 0/1/2-compute ranks, repeated epochs, 2+ in-flight calls, 4 KiB through `copy_bytes`, and each observed dispatch/chunk boundary `t-1,t,t+1`. Poison destination and compare every byte, with errors checked after enqueue/sync. Do not silently retry a failed in-flight collective.
3. Only if the **production-path** probe is correct, recompose the current pinned source and capture a `BIGCHERRY_PATCH_HIT patch=1252_nro03` subject-only receipt plus actual copy-engine/device trace and all-rank work counts. Then paired same-binary host-stage versus P2P controls at tg128/tg512/pp1024/pp4096, >=4 independent server starts, multi-request, long-context, graph replay, rank skew and all-zero/zero-shard cases. Exact F32 reduction/logits/KLD/greedy/MTP acceptance and no lost bytes are mandatory.
4. Promote only with reproducible, CI95-low >=3% end-to-end throughput gain in an eligible envelope, <=1% regression elsewhere, zero wrong bytes/faults, and physically plausible link traffic. Otherwise record `NO_P2P_ROUTE`, `COPY_CORRUPTION`, `FAULT`, `NO_ACTIVATION`, or `NO_EXPOSED_GAIN` and remain rejected. Do not queue this gate while unrelated dev hardware work is active.

### Evidence and validation boundaries

This audit inspected the committed 1252 source, patch metadata, historical validation and lab summary; ran a **host-only seven-test gate model** plus Python compilation. The model confirms the four probe sizes miss the named threshold-edge triples and verifies fail-closed admission cases. It does **not** reproduce the GPU fault, compile HIP, run repository pytest, measure bandwidth, or exercise dev. No new hardware result.

Sources: [AMD HIP peer API](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___peer_to_peer.html); [AMD HIP copy stream guidance](https://rocmdocs.amd.com/projects/HIP/en/develop/doxygen/html/group___memory.html); [llama.cpp internal AllReduce](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/allreduce.cu); [external gfx1201 P2P discussion](https://github.com/ggml-org/llama.cpp/discussions/25197); [gfx1100 RCCL peer-stall report](https://github.com/ROCm/legacy-rocm-build/issues/6565).

## Change log

- 2026-10-18: BCOP292 consolidated PNRO18 and closed the rejected 1252 P2P capability. No patch/hardware change.
- 2026-10-08: 1252 rejected after 2026-10-07 lab fault and non-composition.
- 2026-09-25: source-current peer push, bidirectional content probe and default-off selector implemented.
