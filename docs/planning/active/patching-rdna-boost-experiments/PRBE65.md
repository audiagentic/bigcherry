---
id: PRBE65
order: 0
plan: patching-rdna-boost-experiments
state: completed
created-at: '2026-09-09T10:58:03.403872+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: null
---

# VK-FA-001: 32 KiB Vulkan scalar-FA occupancy heuristic — no generic scale patch

## Disposition (2026-10-07 UTC)

**Close RD82/PRBE65's proposed automatic 64→32 KiB proportional scaling.** This is not a demonstrated BigCherry optimisation. Current b11474 and upstream master still use the exact 65536-byte AMD gate, but the first relevant 32 KiB proprietary-driver R9700 comparison reports worse prefill when the limiter is forced. Preserve upstream behaviour. No BigCherry patch, architecture table, runtime flag, or hardware campaign is authorised by this item. A *new* narrowly owned experiment may reopen only after the first-party gates below pass.

## Verified source and causal scope

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`: `get_fa_tuning_params_scalar(device,hsk,hsv,n_rows,n_kv,k_type,v_type,f32acc)` sets `result.limit_occupancy_shmem` only when `vendor_id == VK_VENDOR_ID_AMD && maxComputeSharedMemorySize == 65536`. The non-GCN/RDNA arm additionally requires `n_rows >= 64 && hsk <= 128`; the GCN arm requires `n_rows <= 8 && hsk >= 256`. The occupancy limiter is **scalar FA only**: `get_fa_tuning_params_coopmat1/2` do not set it. RDNA single-token decode and normal MTP verification widths 2–8 do not reach the RDNA arm.
- `get_fa_pipeline_state` transfers the scalar tuning value into `vk_fa_pipeline_state`; `get_fa_spec_constants` supplies specialization constant 11. `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl` defines it; `flash_attn.comp` declares `shared vec4 occupancy_limiter[LIMIT_OCCUPANCY_SHMEM > 0 ? LIMIT_OCCUPANCY_SHMEM : 1]` and deliberately **writes, barriers, then reads** it to prevent optimisation away. A new limiter adds a synchronization path as well as LDS occupancy pressure; it is not a free scheduling hint.
- Existing FA shared-memory support/legality checks remain authoritative. The old proposed `min(dummy_size, maxComputeSharedMemorySize)` is **not** a sufficient legality proof: the shader also allocates other shared arrays. Verify *total compiled workgroup shared bytes*, pipeline creation and dispatch, not only the dummy array size.
- The old proposed `>=16384` condition contradicts its own 16 KiB “no limiter” fixture: it would activate a GCN 16 KiB case. No scaled implementation is approved. Do not change the existing exact-equality gate or assume a device's reported 32 KiB equals its physical LDS.

Source: https://github.com/ggml-org/llama.cpp/blob/b11474/ggml/src/ggml-vulkan/ggml-vulkan.cpp and https://github.com/ggml-org/llama.cpp/blob/b11474/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp . The exact gate and shader behavior were checked against the pinned source; the gate also remains in upstream master as inspected 2026-10-07.

## Evidence and cheapest discriminator

- **External, not BigCherry:** llama.cpp discussion #21043, 2026-09-03, R9700 gfx1201 / Windows AMD proprietary driver reporting **32768**: pp512 stock **719**, forced 32 KiB-scaled limiter **704** (−2.1%), limiter off **744** (+3.5% vs stock); tg128 **30.33 / 30.39 / 30.44** respectively. This is a limited external comparison, not a statistical promotion result; it directly argues against assuming scaling helps this architecture/driver.
- **External, not causal proof:** llama.cpp issue #26163 (2026-07-27; closed stale 2026-10-02) associates a Vega gfx90c driver 65536→32768 report change with diffusion throughput loss. The reporter did not build an altered FA gate; most of the workload uses other kernels. A comment proposes a *GCN+proprietary-specific* 32 KiB arm, not a general RDNA rule. Do not transfer the reported ~17% diffusion difference to BigCherry FA.
- **Static/mock actually executed this audit:** seven pinned-source assertions verified the scalar-only predicate, two architecture/shape arms, specialization handoff, and shader dummy-array/barrier. Five arithmetic fixtures passed: RDNA 64 KiB prefill 1664 vec4; RDNA 32 KiB prefill 0 stock / 832 proposed; RDNA 32 KiB MTP width4 0/0; GCN 32 KiB 0/448; GCN 16 KiB 0/224 (exposes the old floor contradiction). This proves routing/arithmetic only, **not** shader compilation, occupancy, or performance.
- No BigCherry device-trait dump, scalar-FA attribution, build, hardware test or benchmark was obtained in this audit.

References: https://github.com/ggml-org/llama.cpp/discussions/21043 ; https://github.com/ggml-org/llama.cpp/issues/26163 . The inspected AMD-LLAMA-CPP fork retains the same 65536 gate; no fork code establishes a general safe 32 KiB policy. AITER/vLLM attention tuning is architecture/operator-specific and does not establish a transferable Vulkan dummy-LDS formula.

## Conditional reopen / bounded experiment (new owner only)

1. **Gate 0 / stop:** record actual `ggml_vulkan` device line (gfx1100/gfx1201/gfx1030, RADV/AMDVLK/proprietary, driver version, reported LDS), and production scalar-FA pipeline signatures with `path=FA_SCALAR,n_rows,hsk,hsv,k_type,v_type`. If no AMD **32 KiB** device runs `n_rows >=64,hsk<=128` scalar FA, or those calls account for **<5% E2E wall time**, terminate without patch/hardware queue. Do not test an unavailable gfx1151/GCN lane as though it were fleet evidence.
2. **Offline gate:** on one confirmed hot signature, use disposable *build-time* baseline/no-limiter/32 KiB-scaled variants of the existing scalar-FA tuning function. Confirm actual compiled SPIR-V specialization constant 11, shader workgroup shared-memory usage (including all arrays), pipeline creation, and one-to-one FA dispatch/work counts. Reject compile failure, shared-memory over-limit, changed topology/work, or output mismatch. No persistent config surface.
3. **Hardware gate only if offline passes:** ≥4 interleaved process-level pairs on that **exact architecture+driver** with matched pp512/pp2048, tg128, long-context and MTP-depth 3/7 controls where supported; direct FA timing, barrier/sync, E2E throughput, greedy/logits/KLD, MTP acceptance, multi-request/multi-ubatch, memory safety and DeviceLost/timeout checks. Preserve original upstream branch for all other devices.
4. **Promote only** with positive causal scalar-FA timing and CI95-low **≥3% E2E** gain on the targeted production prefill lane, **≤1% regression** on decode/MTP/other-driver controls, unchanged work and correctness. Otherwise record a terminal rejection and remove disposable variants. Any driver-specific result belongs to existing Vulkan FA tuning/upstream ownership, not a second BigCherry selector.

## Ownership and consolidation

PRBE65 is the terminal RD82 research/disposition record. Upstream Vulkan scalar FA owns the heuristic and compiled shader. QFP07 owns Flash-Next attention placement; PRBE62 queue-family selection and PRBE68 submission batching are independent and must not be modified or used to explain a limiter A/B. No new scheduler, allocator, shader registry or device table. BCOP54 is the thin audit ledger.

## Historical provenance

Original 2026-09-09 plan and 2026-09-24 implementation-readiness review proposed scaling 26/30/14 KiB occupancy padding by the reported 32/64 KiB limit, with mocked 16 KiB and non-AMD controls. Those implementation instructions are superseded by the source-level and negative external evidence above; they were **not** implemented or benchmarked in BigCherry. RD82 lineage retained; no patch package existed to retire.
