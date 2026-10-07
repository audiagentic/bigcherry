---
id: QFP34
order: 34
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:39:33.618549+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Thin-F32 prefill kernel: <=16-row F32 matmuls off generic SGEMM

## Description

External report: a dedicated F32 matmul path for <=16 activation columns reduced one reported kernel/call class from 401 to 21 us and improved pp4096 from 1602 to 1836 t/s. Those numbers are from different hardware and are hypotheses only.

At b11402 the relevant gap is narrower. The CUDA/HIP backend already has an F32 MMVF kernel and on AMD selects it for `ne11 <= 8`. RDNA3/RDNA4 do not have the F32 MMF/MMA path: `fp32_mma_hardware_available()` and F32 `ggml_cuda_should_use_mmf()` are CDNA/MFMA-oriented. Consequently an otherwise eligible F32 `GGML_OP_MUL_MAT` with 9-16 activation columns reaches `ggml_cuda_mul_mat_cublas()` and the simple non-batched case uses rocBLAS `cublasSgemm`.

Gate 0 is therefore a post-fusion census of **F32 weight x F32 activation** MUL_MATs at ub512, especially widths 9-16. If that class is not material on our Flash-Next graph, close the item. Do not infer the external shape from its timing.

## What we already have

### b11402

- `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_mul_mat`
  - dispatches eligible F32/F16/BF16 matrix-vector work first through `ggml_cuda_should_use_mmvf()`;
  - then checks `ggml_cuda_should_use_mmf()`;
  - finally falls back to `ggml_cuda_mul_mat_cublas()`.
- `ggml/src/ggml-cuda/mmvf.cu::ggml_cuda_should_use_mmvf`
  - requires aligned/compatible strides;
  - for `GGML_TYPE_F32` on AMD returns true through `MMVF_MAX_BATCH_SIZE` when no F32 MMA hardware is available.
- `ggml/src/ggml-cuda/mmvf.cuh`
  - defines `MMVF_MAX_BATCH_SIZE = 8`.
- `ggml/src/ggml-cuda/mmvf.cu::mul_mat_vec_f_cuda_switch_ncols_dst`
  - already instantiates F32-capable kernels for widths 1..8.
- `ggml/src/ggml-cuda/mmf.cu::ggml_cuda_should_use_mmf`
  - F32 support requires NVIDIA Ampere MMA or AMD MFMA;
  - RDNA3/gfx1100 and RDNA4/gfx1201 use AMD WMMA, so this F32 route does not cover them.
- `ggml/src/ggml-cuda/common.cuh::fp32_mma_hardware_available`
  - returns true for CDNA, not RDNA.
- `ggml/src/ggml-cuda/common.cuh::amd_wmma_available`
  - identifies RDNA3/RDNA4, but F32 MMF does not select WMMA.
- `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>`
  - reaches `cublasSgemm` for the ordinary 2D F32 case.
- `src/models/qwen4exp.cpp`
  - builds several small F32 graph tensors in QSA/hyper-connection/MTP paths, but code inspection alone is insufficient to claim which surviving MUL_MAT shapes dominate after fusion.

### BigCherry overlap

- `patches/1311_hc_pre_q81` and `1313_scale_act_fuse` remove/fuse parts of the hyper-connection chain; census must be taken with the production patch set so QFP34 does not optimize nodes that no longer execute.
- `patches/1342_fusion_bisect` can help attribute whether a candidate node is consumed by an existing fusion, but QFP34 does not own fusion.
- Meta tensor splitting changes physical matrix shapes on each rank, so census must record the **post-Meta per-device** shape, not only the logical graph shape.

The mechanism is **partly covered upstream** for widths <=8. The plausible missing fast path is widths 9-16 on RDNA. Do not replace the working <=8 path.

## Steps

1. Gate 0: profile production ub512 prefill and enumerate F32/F32 MUL_MATs by graph name, per-device `(ne00, ne01, ne11)`, call count and total GPU time after current fusions.
2. Confirm whether widths 9-16 fall to rocBLAS on gfx1100/gfx1201 and whether their aggregate time is material.
3. First experiment: extend the existing MMVF implementation only for non-ID F32 widths 9-16, without changing the shared `MMVF_MAX_BATCH_SIZE`.
4. If that existing kernel structure loses to rocBLAS for the observed matrices, tune a dedicated F32 thin variant in the same package; do not add a second package.
5. Validate op equivalence, activation, micro timing and fully separated prefill ABBA.

## Detailed Solution & Technical Design

### Dispatch

New qualification package if Gate 0 is positive:

`patches/1346_rdna_f32_thin_mmvf`

Flag:

`BIGCHERRY_F32_THIN_MMVF=0|1`, default `0`.

The opt-in route belongs in `ggml_cuda_mul_mat()`, immediately adjacent to the existing MMVF selection, and is restricted to:

- `src0->type == GGML_TYPE_F32`;
- `src1->type == GGML_TYPE_F32`;
- `dst->type == GGML_TYPE_F32`;
- ordinary `GGML_OP_MUL_MAT`, not `MUL_MAT_ID`;
- AMD RDNA device;
- `9 <= ne11 <= 16`;
- all existing MMVF alignment/stride requirements;
- no `bad_padding_clear`;
- no layout that the current MMVF helper rejects.

Do not globally change `MMVF_MAX_BATCH_SIZE` from 8 to 16. That constant is shared with MUL_MAT_ID assumptions and has a compile-time equality relationship with `MMVQ_MAX_BATCH_SIZE` in `ggml_cuda_mul_mat_id`. Widening it globally would broaden unrelated MoE routing/ID paths.

Instead add an F32-thin-specific eligibility helper/limit and add 9..16 template cases to the existing `mul_mat_vec_f_cuda_switch_ncols_dst`. The ordinary non-ID path can safely call the same `ggml_cuda_mul_mat_vec_f`; its assertion only caps `ids != nullptr`, so non-ID widths are not semantically constrained by the ID limit.

### Kernel shape

V1 reuses the established MMVF decomposition and accumulation order:

- one template specialization per activation-column count;
- F32 source/activation/output;
- same row/channel/sample stride semantics as existing MMVF;
- no temporary quantization and no datatype conversion;
- no change to tensor layout.

Do not implement split-K in this item. The objective is to remove rocBLAS launch/setup/poor thin-GEMM selection for a measured class, not redesign a general GEMM.

If 9-16 specializations create excessive register pressure or occupancy loss on a measured shape, tune only the F32 thin branch (for example grouping columns or reducing per-thread accumulators) after the basic reuse experiment. Preserve the normal <=8 MMVF route.

### Architectures

- gfx1100 / RDNA3: primary qualification target.
- gfx1201 / RDNA4: primary qualification target, independently timed; do not assume the same crossover.
- gfx1030 / RDNA2: no WMMA and the existing F32 MMVF also tops out at 8. It may use the new 9-16 route only if an op-level sweep proves it faster. Production target prefill does not normally execute on the sidecar, so do not expand gfx1030 merely for symmetry.

Use runtime architecture checks, not device ordinals.

### Meta split

Meta dispatches a logical op onto physical devices after splitting. Kernel eligibility must use the actual tensor descriptors received by the CUDA/HIP backend. This naturally supports arbitrary card counts and attention splits.

A logical matrix can have different physical row counts per rank. Qualification therefore records activation width plus local output/input dimensions on each device. The slowest participating target device determines end-to-end gain.

### Fusion

Do not intercept fused nodes. Existing graph fusions that call MMVF directly for one-column cases remain unchanged. QFP34 only replaces the terminal rocBLAS fallback for a surviving ordinary MUL_MAT. This avoids duplicating QFP35/1313 work and preserves fusion ownership.

## Code Samples & Guidance

Likely production anchors:

- `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_mul_mat` around:
  `if (ggml_cuda_should_use_mmvf(...)) { ... }`
- `ggml/src/ggml-cuda/mmvf.cu::mul_mat_vec_f_cuda_switch_ncols_dst` after `case 8`.
- `ggml/src/ggml-cuda/mmvf.cu::ggml_cuda_should_use_mmvf` only if a shared eligibility predicate is factored without changing default behavior.
- `ggml/src/ggml-cuda/mmvf.cuh` for a separate F32-thin maximum/helper declaration if needed; leave `MMVF_MAX_BATCH_SIZE 8` intact.

Activation evidence should be once per device/representative shape, e.g.:

`BIGCHERRY_PATCH_HIT 1346 f32_thin_mmvf cc=<cc> m=<ne01> k=<ne00> n=<ne11>`

Do not log every matmul.

## Files

If Gate 0 is positive, create:

- `patches/1346_rdna_f32_thin_mmvf/patch.py`;
- `patches/1346_rdna_f32_thin_mmvf/README.md`;
- `patches/1346_rdna_f32_thin_mmvf/SUMMARY.md`;
- `tools/tests/patch/test_1346_rdna_f32_thin_mmvf.py`.

Production files patched by that package:

- `ggml/src/ggml-cuda/ggml-cuda.cu`;
- `ggml/src/ggml-cuda/mmvf.cu`;
- `ggml/src/ggml-cuda/mmvf.cuh` only if needed for the dedicated limit/helper.

If Gate 0 finds no material 9-16 F32 class, create no package and close QFP34.

## Validation

### Offline

- Patch-lint and exact/idempotent patch mechanics.
- Flag off preserves existing dispatch.
- Backend-op sweep for F32 MUL_MAT widths 1, 3, 8, 9, 12, 16, 17 and representative dimensions/strides.
- Widths <=8 remain on existing logic; width 17 remains rocBLAS/fallback.
- Noncontiguous/unaligned inputs fail closed.
- MUL_MAT_ID dispatch is unchanged.

### Equivalence

A custom reduction can differ from rocBLAS in F32 accumulation order. Require `test-backend-ops` equivalence within the existing F32 MUL_MAT tolerance, plus greedy target-output identity in the end-to-end ABBA. If bit identity happens to hold, record it, but do not make it a false design premise.

### Activation and timing

- One `BIGCHERRY_PATCH_HIT 1346 ...` proves the route.
- rocprof/kernel census shows the selected 9-16 shapes no longer invoke rocBLAS SGEMM.
- Per-shape timing must beat rocBLAS on both gfx1100 and gfx1201 for the shapes actually present in production. gfx1030 is optional and must be independently proven before enabling there.

### Hardware ABBA

Use `tools/lab/flash-next/queue-env-ab.sh` with complete process separation.

A: `BIGCHERRY_F32_THIN_MMVF=0`
B: `BIGCHERRY_F32_THIN_MMVF=1`

Keep ub512 and all production Meta/fusion flags identical. Run a short pp4096 lane to expose launch-class cost, then a representative 24K or longer prefill lane. Record prefill t/s, candidate-kernel total time, per-device critical path and greedy output identity.

No multi-session contract campaign.

## Effort & Risk

Effort: M.

Risk: medium. Template expansion is straightforward, but the performance crossover is shape- and architecture-specific; widths 9-16 increase accumulator/register pressure and can lose to rocBLAS. The principal correctness risk is accidentally broadening MUL_MAT_ID/shared MMVF contracts by changing the global limit.

Expected gain on our three-card topology: medium only if Gate 0 shows a repeated 9-16 F32 SGEMM class on the critical gfx1100/gfx1201 ranks; otherwise zero. This is lower-confidence than the cheap QFP31/QFP32 checks but a reasonable first kernel item.

## Standards

- No change to the global MMVF/MMVQ ID limit without a separate proof.
- No fixed topology/device ordinal.
- No q4 KV.
- No legacy/back-compat path.
- Default off during qualification.
- Preserve existing <=8 dispatch and all fusion ownership.
- External 4-GPU timings are hypotheses.

## Acceptance Criteria

- A post-fusion, post-Meta census proves a material F32 9-16 MUL_MAT class, or QFP34 closes.
- Flag off is source/dispatch equivalent.
- Flag on changes only qualified ordinary F32 MUL_MATs.
- MUL_MAT_ID, widths <=8 and widths >16 are unchanged.
- Candidate shapes beat rocBLAS on the production gfx1100/gfx1201 devices.
- Backend-op tolerance passes and greedy target output is identical.
- Fully separated ABBA shows a repeatable prefill win with activation evidence and no decode regression.

## Notes

Execution order: fifth, after QFP32/QFP31/QFP33/QFP40. Final priority depends on Gate 0; do not rank the external pp4096 result above measured production bottlenecks.

## Change Log

- 2026-10-07T00:39:33.618549+00:00 (created-by): Created by agent
- 2026-10-07: grounded at b11402; narrowed the RDNA gap to F32 widths 9-16, preserved shared MMVF/MMVQ limits, and added post-fusion/post-Meta census and qualification design.


## Code-level review (2026-10-07)

### 1. Verified facts and corrections

Checked against llama.cpp `d89651a7b205` and the current validated-enhancements composition.

- Native dispatch is `static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst)` in `ggml/src/ggml-cuda/ggml-cuda.cu`. The relevant sequence is exactly `if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) { ... ggml_cuda_mul_mat_vec_f(...); return; }`, then `ggml_cuda_should_use_mmf(..., /*mul_mat_id =*/ false)`, then MMVQ/MMQ, then `ggml_cuda_mul_mat_cublas(...)`.
- `bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11)` in `ggml/src/ggml-cuda/mmvf.cu` returns `ne11 <= 8` for AMD F32 when `fp32_mma_hardware_available(cc)` is false. It also requires even `src0_ne[0]`, `src0_nb[0] == ggml_type_size(type)`, and all higher strides divisible by `2*ts`.
- `#define MMVF_MAX_BATCH_SIZE 8` is in `ggml/src/ggml-cuda/mmvf.cuh`. Do not widen it: `ggml_cuda_mul_mat_id` and ID-path assertions use the shared limit.
- `static void mul_mat_vec_f_cuda_switch_ncols_dst(...)` in `mmvf.cu` has explicit non-ID cases 1 through 8 and `default: GGML_ABORT("fatal error")`. Therefore merely changing the predicate to 16 would crash; cases 9..16 must be added before broadening dispatch.
- `void ggml_cuda_mul_mat_vec_f(...)` asserts `!ids || ne12 <= MMVF_MAX_BATCH_SIZE`; this is an ID-only cap. A non-ID 9..16 call is legal once the switch has matching specializations.
- `bool ggml_cuda_should_use_mmf(...)` in `ggml/src/ggml-cuda/mmf.cu` returns F32 true only for `ampere_mma_available(cc) || amd_mfma_available(cc)`. `amd_mfma_available` is CDNA-only at this pin; `amd_wmma_available` is used for F16/BF16, not F32. The design's RDNA3/RDNA4 F32 gap is therefore correct.
- `static bool fp32_mma_hardware_available(const int cc) { return GGML_CUDA_CC_IS_CDNA(cc); }` and `amd_wmma_available(...) { return GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3(cc); }` are in `ggml/src/ggml-cuda/common.cuh`.
- `ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>` in `ggml-cuda.cu` uses `cublasSgemm(... ne01, ne11, ne10 ...)` for the ordinary 2-D F32 case `ne12 == 1 && ne13 == 1`.
- Correction to the existing plan: the qualification must not say only "ordinary GGML_OP_MUL_MAT" at the CUDA entry point; `ggml_cuda_mul_mat` is already the ordinary-op handler, so the robust exclusion is simply to add the branch there and never touch `ggml_cuda_mul_mat_id`. No op-code re-test is needed inside the helper.
- gfx1030 is not a production target for this target-model prefill because the 6900 XT is the drafter device. Keep gfx1030 inert in V1 rather than adding a speculative third architecture gate.

### 2. Composition and anchors

Validated production patches relevant to these files:

- `1281_moe_mul_mat_id_range` edits `ggml/src/ggml-cuda/mmvf.cu` with edit ids `mmid-range-mmvf-kernel` and `mmid-range-mmvf-host`, and `ggml-cuda.cu` with `mmid-range-cuda-policy`, `mmid-range-cuda-fusion-rule`, and `mmid-range-cuda-fuse-plain`. These edits are for `MUL_MAT_ID` range translation/fusion, not the non-ID 1..8 switch.
- `1307_q81_activation_cache_mmvq` edits `ggml-cuda.cu` for graph-generation/capture state (`q81-generation` and related cache plumbing) and MMVQ consumer code, not the ordinary MMVF dispatch.
- `1313_scale_act_fuse` edits `ggml-cuda.cu::ggml_cuda_try_fuse` via `scale-act-match`; it does not edit `ggml_cuda_mul_mat`.
- `1241_rd33_mmvq_q8_0_f32_decode`, `1274_mmvq_kquant_f32_decode`, `1237_rd30_moe_mmq_compact_grid`, and `1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2` alter MMVQ/MMQ paths, not this F32 MMVF switch.
- `1344_dsv4_hc_grid_index` edits only `dsv4-hc.cu`; use its packaging/marker/off-switch pattern, not its anchors.

No validated enhancement checked above replaces the exact native ordinary-dispatch anchor
`if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {`.
The new dispatch edit should therefore anchor on that composed text and insert a preceding opt-in branch; do not replace the native branch.

In `mmvf.cu`, anchor after the composed 1281 tree on the complete native `case 8: ... break;\n        default:\n            GGML_ABORT("fatal error");` block. 1281's MMVF edits are elsewhere in the same file, so there is no anchor collision if this narrow switch anchor is used.

### 3. Gaps and risks

- Dispatch breadth: `src0->type == src1->type == dst->type == F32`, AMD RDNA, and `9 <= ne11 <= 16` are still too broad without the native MMVF stride predicate. The new predicate must call/reuse the same even-K and stride/alignment checks; otherwise it can catch transposed/views that currently fall safely to rocBLAS.
- Do not use `GGML_CUDA_CC_IS_RDNA3/4` alone if it admits unmeasured sub-architectures. V1 should gate exact production families/capabilities proved by Gate 0: gfx1100 and gfx1201, expressed by cc predicates, never device ordinals.
- Meta split: eligibility sees physical per-rank tensors after Meta splitting. Census and marker must report `ne00/ne01/ne11/ne02/ne03` from those local descriptors. A logical graph node may qualify on one rank and not another; that is acceptable only if Meta semantics do not require identical kernel choice, but end-to-end gain is bounded by the slowest rank.
- 1281/1283 expert-parallel range ops are `MUL_MAT_ID`; this item must never route them through the new branch. Do not change `MMVF_MAX_BATCH_SIZE`, `ggml_cuda_mul_mat_id_needs_sync`, or any range helper.
- 1235/1307/1309-1312 Q8_1 activation-cache paths concern quantized MMVQ consumers. This F32/F32 branch must allocate no Q8_1 slab, publish no cache entry, and not change generation/capture state.
- CUDA graph capture: reuse `ggml_cuda_mul_mat_vec_f` and the existing stream only. The flag should be read through a process-stable cached helper; the marker must be one-shot host logging and must not synchronize the stream. No pool allocation or host/device copy should be introduced.
- FKE01 is mandatory: fused and unfused kernels are not numerically equivalent and fusion admission depends on allocation-address overlap. Even though V1 adds no allocation, identity claims must compare A/B with `GGML_CUDA_DISABLE_FUSION=1` on both arms. Fusion-on runs are probes, not the identity proof.
- Template cases 9..16 increase per-thread accumulator/register use. Gate separately on gfx1100 and gfx1201; a win on one does not justify enabling the other.
- No host thread is useful here; this is a synchronous dispatch/kernel specialization problem.

### 4. Concrete implementation outline

Package: `patches/1346_rdna_f32_thin_mmvf`.

Flag: `BIGCHERRY_F32_THIN_MMVF=1` enables; default `0` during qualification. No compatibility shim.

Add in `mmvf.cu`:

`static bool bc_f32_thin_mmvf_enabled();`

`bool bc_cuda_should_use_f32_thin_mmvf(int cc, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11);`

The helper must return true only for exact qualified RDNA cc, `9 <= ne11 <= 16`, and the same F32 MMVF layout checks as `ggml_cuda_should_use_mmvf`. Prefer factoring a small internal layout predicate shared by native and BigCherry eligibility without changing native results; if that makes the anchor larger, duplicate the three checks locally rather than rewriting native dispatch.

Extend `mul_mat_vec_f_cuda_switch_ncols_dst<T, type_acc>(...)` with cases 9..16, each calling
`launch_mul_mat_vec_f_cuda<T, type_acc, N>(...)`. Do not alter the ID branches at the top of the switch.

In `ggml_cuda_mul_mat`, immediately before the native MMVF `if`, add:
`if (BIGCHERRY flag && src0->type == GGML_TYPE_F32 && bc_cuda_should_use_f32_thin_mmvf(cc, src0->ne, src0->nb, ne11)) { ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst); return; }`.
The enclosing native function has already rejected non-F32 `src1/dst` and `bad_padding_clear`.

Activation marker, once per device/shape class when `BIGCHERRY_PATCH_TRACE` is set:
`BIGCHERRY_PATCH_HIT patch=1346_rdna_f32_thin_mmvf path=f32_thin_mmvf cc=<cc> ne00=<...> ne01=<...> ne11=<...>`.

Required patch edits:
- `f32-thin-include-flag`: `mmvf.cu`, insert-after the local includes; add only `<cstdlib>` if not already available. Mode `insert_after`.
- `f32-thin-eligibility`: `mmvf.cu`, insert-before exact signature `bool ggml_cuda_should_use_mmvf(...)`. Mode `insert_before`.
- `f32-thin-switch-9-16`: `mmvf.cu`, replace the exact composed `case 8 ... default GGML_ABORT` tail with case 8 unchanged plus cases 9..16. Mode `replace`, `expect_matches=1`.
- `f32-thin-dispatch`: `ggml-cuda.cu`, insert-before exact composed ordinary MMVF `if (ggml_cuda_should_use_mmvf(...))`. Mode `insert_before`, `expect_matches=1`.
- Avoid `mmvf.cuh` entirely unless a declaration is genuinely needed across translation units; do not add a second maximum constant there.

Offline test should follow 1344's mechanics: assert exact new kernel/template text, dispatch switch, flag/off path, marker text, and that native `MMVF_MAX_BATCH_SIZE 8` plus the native 1..8 route remain present.

### 5. Gate 0 and lightweight validation

Gate 0 before package coding:

1. Run the production target model with the current validated composition and f16 KV using `tools/lab/flash-next/long-ctx-profile.sh ... prefillprof` at `DEPTH=8192` and `DEPTH=24576`, `UB=512 B=512 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16`, current `BIGCHERRY_ATTN_TS=1,1,0`, and the normal three target GPUs.
2. Pair the rocprof kernel trace with a host-side dispatch census at `ggml_cuda_mul_mat` (graph name if available, device cc, local `ne00/ne01/ne11/ne02/ne03`, src types, chosen path). Count only F32/F32/F32 ordinary matmuls with `9 <= ne11 <= 16` that choose rocBLAS.
3. Gate is positive only if, on at least one of 8K or 24K, those calls consume **>=0.75% of prefill wall time on the critical participating rank** and at least **1.5 ms per 512-token ubatch** in aggregate, with the class present on gfx1100 or gfx1201. If the trace cannot attribute >=0.75%, close QFP34; do not code from the external report.

After implementation:
- patch mechanics/offline test + patch-lint;
- marker must fire only on 9..16 F32 ordinary calls and never on `MUL_MAT_ID`;
- per-shape backend-op sweep at widths 8,9,12,16,17 and observed physical dimensions on gfx1100/gfx1201;
- ABBA through `tools/lab/flash-next/queue-env-ab.sh`, complete process separation, same production flags, first pp4096 then 24K;
- identity ABBA with `GGML_CUDA_DISABLE_FUSION=1` on **both** A and B; compare greedy output and backend-op tolerances;
- separate fusion-on probes to confirm no changed fusion selection/regression under FKE01;
- require each enabled architecture's observed candidate shapes to beat rocBLAS and end-to-end B to be non-negative at both depths.

### 6. Verdict

**GO AFTER GATE 0.** The code gap is real and the implementation is contained, but the unmeasured 9..16 F32 class may be too small after 1311/1313/1344.

Expected gain on the production three-card topology if Gate 0 passes: **+0.2% to +0.8% prefill**. If Gate 0 is below the threshold, expected gain is effectively zero and the item should close.
