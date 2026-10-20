---
id: TRVP12
order: 0
plan: tuning-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T11:01:00.469466+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Vulkan stack and runtime integration

## Description

Integrate Vulkan runtime and candidate lifecycle using shared stack/build/attestation/persistence/replay architecture, with complete capability context, negative matrix, route correctness, and fail-closed identity.

## Steps

1. Consume RRVP/BRVP/TRVP canonical stack, BuildPlan, runtime-attestation, manifest/DB, and replay contracts; do not create a second identity architecture. 2. Implement Vulkan loader/device capability and negative-matrix integration, including no ICD, multiple ICDs, software ICD, explicit ICD selector, device-order changes, and missing required extension/feature. 3. Integrate real MUL_MAT interception and complete pipeline recipes including preparation, shader/pipeline, split/reduction, and conversion; include all preparation costs in correctness/timing evidence. 4. Integrate record/tune/force/correctness/replay routes for reachable Vulkan operations with provider identity and deterministic fail-closed fallback. 5. Persist stable device identity, selected ICD/driver, API/features, and shader compiler/SPIR-V target in the appropriate shared identity. 6. Keep attention, transport, and collective owners separate; hand generic closure to TRVP13 and CM1-specific work to PRVP/TRVP successors.

## Detailed Solution & Technical Design

TRVP12 owns Vulkan-specific adapter behavior and capability semantics while common probing, BuildPlan/cache identity, runtime attestation, manifest/DB, and replay policy remain shared services. Driver-generated executable data is diagnostics unless selectable through a supported API. Candidate eligibility receives the full Vulkan capability context. Native fallback and provider mismatch rejection are explicit; no parallel stack digest or duplicate Vulkan MUL_MAT plan is created.

## Code Samples & Guidance



## Files

Vulkan backend probe and capability adapter; shared BuildPlan/manifest/DB integration; Vulkan runtime/candidate registry; record/tune/force/replay routes; negative-matrix, correctness, timing, fallback, and identity tests.

## Validation

Run the full negative matrix and verify stable selected ICD/device/driver/API/features/extensions and shader compiler/SPIR-V identity are persisted. Prove actual Vulkan implementation on strict route; validate correctness before timing, including shader/pipeline, split/reduction, conversion, and preparation cost. Confirm build/runtime fingerprint mismatch and provider mismatch reject evidence, unsupported capabilities fall back safely, and record-to-promotion replay lineage is preserved. Verify attention/transport/collective boundaries and no duplicate stack architecture.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Vulkan tuning starts only with shared stack/device/build identity and full capability context. Negative matrix, strict-route correctness, complete pipeline-cost timing, fallback, replay, and mismatch rejection pass. No parallel identity architecture or duplicate MUL_MAT plan exists; attention/transport/collective scope remains with its owners.

## Notes

Supersedes: RO17
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-rocm-vulkan-provider-ro17

Supersedes: RO17
Inherited constraints: RV104 and RV118 — no duplicate Vulkan MUL_MAT plans or stack architecture; attention/transport/collective scope remains with specific owners.
Migration: capability-rebaseline-v3-2026-09

Supersedes RO17. Preserve RV104/RV118: do not serialize behind TRVP11, do not duplicate Vulkan plans/stack architecture, and keep CM1-specific qualification in its successors. Preserve patch 1225 and governance.

## Change Log

- 2026-09-09T11:01:00.469466+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:17:54.644516+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.583617+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:47.441300+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:52:16.738565+00:00 (updated-by): Updated: section:description, section:steps, section:acceptance_criteria, section:notes
- chg_20260910_005948_legacy-planning-folders-now-co_1240
- 2026-09-10T00:59:48.982745+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T01:03:08.549152+00:00 (updated-by): Updated: section:validation
- chg_20260910_010342_successor-plans-now-have-expli_8662
- 2026-09-10T01:03:42.583184+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:45:26.409662+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_034539_repaired-trvp10-12-with-the-co_1657
- 2026-09-10T03:45:39.468223+00:00 (updated-by): Updated: section:ledger-events

## 2026-10-20 GPU-kernel audit: RDNA3/4 Vulkan large-tile split-K (BCOP337)

**Disposition: HOLD / PROFILE_FIRST; documentation only.** Native llama.cpp b11552 (23b0202a) already has a Vulkan split-K matmul and F32 reduction shader. Open [upstream #30280](https://github.com/ggml-org/llama.cpp/pull/30280), updated 2026-10-19, changes the existing Vulkan MUL_MAT kernel's split count for 128x128 output tiles, not its arithmetic, scheduler, cache, or MoE MUL_MAT_ID. The PR remains open and had merge-conflict/review discussion on 2026-10-19. RRVP02's Vulkan implementation pause remains binding. No implementation or hardware queue is authorized.

### Source, dispatch and physical work

- Pinned ggml/src/ggml-vulkan/ggml-vulkan.cpp::ggml_vk_mul_mat_q_f16 selects the actual pipeline via ggml_vk_get_mul_mat_mat_pipeline_map and ggml_vk_guess_matmul_pipeline_map; it may quantize F32 src1 to Q8_1. ggml_vk_guess_split_k(ctx,ne01,ne11,ne10,disable_split_k,pipeline) selects K partitions. It then reserves prealloc_split_k = split_k * d_sz (d_sz = 4 * output_elements), before ggml_vk_matmul dispatches the same matmul pipeline per K segment and pipeline_matmul_split_k_reduce merges partial F32 outputs. ggml_vk_matmul_id is separate and unaffected.
- The pinned heuristic splits only when n >= pipeline->wg_denoms[1], k >= 2048 and output tile count is at most 2/3 of shader_core_count. It caps at eight splits and decreases split count until 256-aligned K chunks cannot leave an empty final split. For 2/3*CU < tiles <= 2*CU, it often leaves a partially occupied last CU wave.
- #30280 adds ggml_vk_guess_split_k_large_tile when native split=1, RDNA3/RDNA4, n>=512, k>=2048, tile area>=128x128 and total output workgroups<=2*CU. Workgroups = ceil(m/tile_m)*ceil(n/tile_n)*batch. Compare split 1/2/3 by modeled waves * 256-aligned K steps plus a hand-chosen (not measured) 64 K-step charge per extra split. A k>=8192 shortcut chooses split=3 when workgroups*6<=CU*5 and pipeline LDS statistics suggest two resident workgroups.
- The PR obtains per-pipeline LDS through getPipelineExecutableStatisticsKHR (RADV: LDS size; AMDVLK: ldsUsageSizeInBytes), compiling/loading the selected pipeline before split choice if necessary. Vulkan maxComputeSharedMemorySize is a resource proxy, not measured per-CU occupancy; require actual wave32/64, LDS, VGPR, spills and executable statistics on gfx1100/gfx1201.
- **New code-level discriminator:** #30280's fills_waves tie-break can choose split=3 over split=2 at equal modeled compute despite higher modeled reduction charge. For CU=96, tile128x128, m4096,n512,k4096,batch1, pinned split=1; candidate split=3, while estimated split2 cost=6272 versus split3=6400 K-step units. This is not an observed slowdown. Fixed split 1/2/3 must be tested.
- That shape has 8 MiB F32 output, 24 MiB split3 scratch, and up to 48 MiB additional logical partial write/read traffic versus unsplit output (3 partial writes + 3 reads). No physical DRAM bandwidth measurement exists. Three 1536-element aligned K chunks represent 4608 slots for 4096 valid K indices; tail masking must preserve each valid contribution exactly once.

### Evidence, upstream conflict and ownership

- External #30280 ABBA pp512: gfx1100 RX 7900 XTX Gemma 3 12B Q8_0 +9.8% FP16 / +5.7% INT8; gfx1201 R9700 Gemma4 E4B Q4_K_M +7.1% FP16, Gemma4 E4B Q4_0 +6.1% FP16 / +4.3% INT8. Q6_K INT8 operator controls were neutral/slightly negative. None is a BigCherry measurement; no first-party incremental split-K gain exists.
- Separate open [upstream #30146](https://github.com/ggml-org/llama.cpp/pull/30146) changes the same function for small-N wave-aware split-K and adds subgroup/workgroup pipeline metadata; its gfx1100 W7800 Qwen3.8 27B Q8_0 ub13/24/32 prefill improvements were +19.8/+18.5/+12.4%, while ub64/512 and ordinary TG128 were neutral. Do not stack patches or duplicate selectors; reconcile disjoint N gates and upstream merge state first.
- Existing BigCherry PKC04 Vulkan-versus-HIP dual-XTX 27B Q8_0 measurements (Vulkan layer 882 pp/20.8 tg vs HIP 1086/23.8; tensor 494/24.2 vs HIP 1479/38.5) compare entire backends/collectives, not split-K. vLLM ROCm's need_split_k / wvsplitkrc_dispatch and SGLang AMD GEMM/MoE tuning provide shape/occupancy controls, not portable Vulkan shaders or gains.
- **TRVP12 is the sole native Vulkan MUL_MAT split-K/merge recipe owner.** TRVP15/PRVP02 own CM1-specific shader safety and qualification, TRVP14 route identity, RRVP02 the implementation pause. HIP QFP44 MMQ/J, PRVP06 FWHT, and proposed PRVP07 FA token folding are distinct. Do not add another technical plan, scheduler, dispatch table, allocator, telemetry or config.

### Bounded next discriminator and terminal outcomes

1. After RRVP02 releases Vulkan implementation work, read-only profile an actual bound 128x128 FP16 or INT8 Vulkan MUL_MAT (not MUL_MAT_ID) on single gfx1100 and gfx1201: device/driver/ICD/shader digest, quant type, m/n/k/batch, tile and wave geometry, current split, aligned K ranges, Q8_1 conversion, reduction/sync, scratch bytes, exposed wall time and graph replay. BigCherry pp512/pp1024 Qwen3.8 27B Q8_0 is a candidate only if it executes that tile; n64/128/256, TG and MTP verify are negative controls.
2. Reject cheaply if no route (NO_LARGE_TILE_ROUTE), less than 2.9126% optimistically removable E2E latency (BELOW_CEILING), upstream absorbed it (PIN_ABSORBED), or upstream source/merge is unresolved (WAIT_UPSTREAM). No speculative patch or queue.
3. If eligible, compare pinned split1, forced split2, forced split3 and upstream heuristic, with compilation/warmup excluded. Matrix m={1536,2560,3840,4096}, n={128,256,512,1024}, k={2048,4096,8960,10240,15360}, batch={1,2}, F16/Q4_0/Q4_K/Q8_0 and Q6_K INT8 control; 64-CU gfx1201 versus 96-CU gfx1100, RADV/AMDVLK separately when available. Capture workgroups/waves, LDS/VGPR/spills, scratch, physical HBM/L2 traffic and reduction/preparation costs. No cross-GPU P2P assumption.
4. Require backend-op correctness, full K/output coverage, K tails, noncontiguous/batched/quantized scales, CPU F32/logits/KLD, greedy/MTP acceptance, multi-ubatch, two requests in one process, long-context, graph allocation/reallocation and unchanged fallback. Unsupported shader/resource query retains native split1.
5. Promote only with paired ABBA CI95-low >=3% end-to-end prefill versus current native pin, <=1% unaffected TG/MTP regression, no missing work, and resource/traffic plausibility; otherwise NUMERIC_FAIL, RESOURCE_REGRESSION, NO_E2E_GAIN or REJECT.

**Completed:** Pinned source/dispatch and upstream PR/diff/review inspection; host-only Python oracle 3,360 geometry/resource/tail cases passed (168 modeled split decisions changed), Python compilation passed. No Vulkan/SPIR-V build, BigCherry repository test, GPU profiling or hardware benchmark.
## 2026-10-18 Vulkan Q4 KV shader ownership note
PRVP04 (BCOP295) owns only the prospective upstream #30738 Q4_0 KV dequant+transpose shader and its prefill route/physical-work gate. TRVP12 remains the shared Vulkan provider/capability adapter, **not** the attention shader owner. RRVP02's implementation pause is unchanged; no candidate, queue, provider, cache or allocator is added here. The n>=64 Vulkan FA scratch admission must not be misreported as a single-query decode speedup.
