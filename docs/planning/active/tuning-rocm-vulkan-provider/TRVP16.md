---
id: TRVP16
order: 0
plan: tuning-rocm-vulkan-provider
state: pending
created-at: '2026-10-02T12:35:11.094367+00:00'
breadth: ''
skill: ''
created-by: agent
priority: P3
work: L
---

# Vulkan collective and mixed-architecture qualification

## 2026-10-11 implementation audit: inactive-shard safety before heterogeneous TP

**Disposition: no new transport or hardware campaign yet.** TRVP16 owns qualification of the existing PRVP03 communication SPI across all-Vulkan multi-RDNA configurations, not another AllReduce implementation. PRVP03/1290 is an evaluated, default-off, synchronous F32 host reference. An upstream Vulkan provider, [llama.cpp #25051](https://github.com/ggml-org/llama.cpp/pull/25051), is OPEN as inspected 2026-10-10 (head `99becee6fc0312964b8cde77e83baf46b5f133d6`); inspect/rebase its actual code before considering any independent phase-1 transport. No BigCherry mixed-RDNA correctness or throughput has been measured.

### Source-proven correctness boundary (pinned llama.cpp b11474)

- `ggml/src/ggml-backend-meta.cpp::ggml_backend_meta_context` (~1842-1866) initializes a backend-specific `comm_init` from rank zero's registry, retains its context, then frees it **before** the child backends. The provider's borrowed `ggml_backend_t` pointers are valid within this lifetime; a new allocator/lifetime registry is unnecessary.
- `ggml_backend_meta_graph_compute` (~2333-2360) implements the generic AllReduce fallback: ranks whose final node lacks `GGML_TENSOR_FLAG_COMPUTE` are **filled with zero** before butterfly reduction. This is required for empty/disabled split slices; merged upstream [#29793](https://github.com/ggml-org/llama.cpp/pull/29793) replaced SCALE(0) with FILL(0) because uninitialized Inf/NaN survived multiplication by zero.
- At ~2461-2479 the meta backend calls the provider first and immediately runs the generic fallback when it returns `false`. Therefore `false` is safe **only before any provider writes or irreversible submissions**. A provider that writes rank 0 and then returns false risks double reduction/stale partials; post-submit failure must be terminal or handled by an explicitly proven recovery protocol, not ordinary fallback.
- `engines/llamacpp/patches/1290_vulkan_allreduce_host_f32/patch.py::_PROVIDER_CODE::ggml_backend_vk_comm_allreduce_tensor` checks F32, contiguous, size and backend registry, then synchronizes and reads **every rank's tensor unconditionally**, without checking `GGML_TENSOR_FLAG_COMPUTE`. Its F32 sum can include stale/NaN contents from an inactive rank. This is a **source-level contract mismatch**; no production failure has been reproduced. The dual-XTX screening did not qualify empty-slice or heterogeneous layouts.
- The minimal safe 1290 action belongs to **PRVP03**, not TRVP16: before the first host read/write, if any final tensor lacks COMPUTE, return false so the existing meta FILL fallback owns that call. Alternatively implement explicit zero-source and zero-destination semantics, but only with a separate exact reference test. Preserve stock behavior when `BIGCHERRY_VK_ALLREDUCE` is unset.

### Upstream and other engines: adopt/wait/reject

- [llama.cpp #25051](https://github.com/ggml-org/llama.cpp/pull/25051) (open, last updated 2026-10-08) is **already a Vulkan provider implementation**, not merely a proposal: `ggml-vulkan.cpp` introduces `VK_EXT_external_memory_host` imports, `VK_KHR_external_semaphore_fd` opaque-FD timeline sharing, driverUUID/capability checks, per-device progress counters, imported host buffers and a ring path. It zeroes inactive **host staging** for its ordinary path; the ring is restricted to **all-COMPUTE ranks** and `nbytes >= 2 MiB`. Verify the destination rank is also initialized when inactive, the exact reduction order and all post-submit failure paths before adoption. An imported buffer does not imply hardware P2P.
- At BigCherry's proposed 32-256 KiB F32 collective sizes, the upstream 2 MiB ring threshold excludes the ring; benchmark the actual smaller ordinary mapped-host path rather than attributing external ring speedups. External PR benchmarks are primarily NVIDIA or mixed-vendor and do not qualify RDNA3/4.
- [vLLM custom AllReduce](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/custom_all_reduce.py) checks peer access before enabling its fast path. [SGLang PCIe-IPC AllReduce](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/device_communicators/pcie_ipc_ar.py) uses FlashInfer/CUDA IPC, BF16 and 2/4/8 ranks; neither is a Vulkan/no-P2P AMD drop-in. Reuse capability, synchronization and negative-control ideas only.
- **Decision:** wait for #25051 review or independently prove its pinned diff as an optional source-composed control. Do not build PRVP03's planned second Vulkan transport in parallel; if #25051 merges, compare stock pinned, upstream and 1290 using the same SPI, retaining 1290 only as a correctness reference.

### Implementation-ready qualification and admission

1. **Gate 0 (no GPU):** compare 1290 and upstream provider against the meta fallback contract. Host fixture: ranks D=2/3/4; all COMPUTE, one inactive with stale finite bytes, one inactive with NaN/Inf, all inactive, zero-sized slice, rank/shape/type mismatch, and return-false before/after a simulated rank write. Require exact zero contribution for inactive ranks, no double application and no false success. A source-only host model run in this audit passed six unittest methods covering these categories (initial test-script assertion typo corrected before the passing rerun); **no GPU assertion**.
2. **Gate 1 (device/driver admission):** obtain actual physical UUID/PCI ID, architecture gfx1100/gfx1201/gfx1030, RADV/AMDVLK ICD, driverUUID, Vulkan extension/feature import/export, host-import alignment, coherent visibility, timeline handle type, compute/transfer queue family and free VRAM per rank. Reject duplicate physical devices, mixed HIP/Vulkan registry, mismatched handle capability, unsafe queue ownership, unsupported F32 path or unknown rank mapping. Vulkan ordinals are not identity. If only the generic fallback is available, record `NO_PROVIDER_LANE` and stop.
3. **Gate 2 (reference semantics):** on dual gfx1100 RADV first, compare stock meta fallback, 1290 enabled and a separately built #25051 control only if it composes. Test D=2, D=3 (XTX+XTX+R9700), D=4 (+6900 XT) in that order, and XTX+6900 as a separate D=2 negative/slow link. Include uneven/zero slices, tensors with views, F32 exact/reproducible rank order, F16/BF16 negative controls, finite extremes/NaN, repeated same-process requests, 8K/80K/245K context, ubatch 16/512/2048, graph replay and MTP accept/reject. Prove source and destination byte counts, all-rank results and fallback identity; **do not queue new runs while any QFP49/50 or link-probe lane is active**.
4. **Gate 3 (bounded performance):** measure per-collective payload/count, host H2D/D2H bytes, GPU/CPU copy time, staging allocation/reallocation, timeline waits, actual exposed critical path, rank skew, peak VRAM and end-to-end PP/TG/MTP. The 1290 path necessarily transfers approximately `2 * D * nbytes` host-link bytes per call (one read and one write per rank), before CPU accumulation; this is **source-derived traffic**, not measured bandwidth. At 32-256 KiB and D=2/3/4, the bound is 128 KiB-2 MiB/call. Do not add overlapped durations as if serial.
5. **Promotion/rejection:** require four independent sessions per architecture/topology, >=10 paired ABBA rounds/session, exact/declared numerical parity, no missing work or partial-write fallback, CI95-low >=3% E2E throughput gain and <=1% unrelated-control regression. No positive exposed critical-path headroom, no supported import/semaphore pair, wrong results, or unbounded staging lifetime => `REJECT/DEFER`; retain stock meta fallback. Driver/ICD, graph topology or build identity change invalidates prior evidence. No automatic policy or new config surface.

### Existing BigCherry measured evidence (not a new benchmark)

RRVP05 stock RADV dual-XTX 27B Q8_0, 2026-10-02, one screening request: tensor 494 pp / 24.2 tg, layer 882 pp / 20.8 tg; three-card tensor 286 pp / 14.8 tg. PRVP03 same-binary three-request dual-XTX: stock tensor 496 pp / 24.6 tg versus 1290 host-F32 259 pp / 20.1 tg; MTP5 50.4 versus 40.7. The 1290 reference **regressed**, and these data cannot establish a three-/four-card gain or an upstream provider win. External CUDA/NVIDIA results are not transferable.

### Ownership, novelty, and exclusion

PRVP03 owns provider SPI, 1290 safety and any upstream-source adoption; TRVP16 owns **only** mixed-RDNA qualification/terminal gates; RRVP05 owns matched hardware campaign; RRVP02 retains its Vulkan implementation-pause boundary; TRVP12 owns Vulkan runtime/capability identity; PRVP02/TRVP14/15 own unrelated CM1. No second transport, scheduler, allocator, telemetry system or patch ID.

Branch `feat/qfp41-dispatch-lock-stats` was selected from independent commit `254110e95c8d39c4a4c5fbdff53061a91f17fc05` (2026-10-10 16:53:19 UTC), newer than `main`'s independent `c59c0a3fbc1e` (16:52:47). TRVP16 last independently changed 2026-10-02; PRVP03/1290 had no changes in the preceding 12 hours and no related active PR was found. Excluded active QFP41/1356 HIP graphs, QFP48/49/50 prefill and cross-card link probes, QFP36/1357 router, QFP17/1330, MTP/1348 and Radiance. These are read-only dependencies; no overlapping hardware job is queued. Previous BCOP77 addressed CM1, BCOP118 placement scoring, BCOP119 was unpublished PRBE28; this mixed-RDNA inactive-shard SPI gate is new.

### Validation actually executed

Pinned-source structural checks **10/10 passed** (1290 unconditional rank read, absent COMPUTE check, meta FILL fallback and provider-first semantics, provider lifetime, upstream #25051 inactive staging and 2 MiB ring gate). Disposable Python host-only model: **6/6 unittest methods passed** after fixing one test assertion typo; no BigCherry pytest, C++/Vulkan/SPIR-V build, device import, GPU execution, PCIe measurement or hardware benchmark.

## Change Log

- 2026-10-02T12:35:11.094367+00:00 (created-by): Created by agent.
- 2026-10-11: Source-level inactive-shard/fallback contract and upstream #25051 ownership rebaseline; no implementation/hardware action.
