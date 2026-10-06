---
id: QFP30
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-06T15:41:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Range-MMQ broadcast activation dedup for whole-expert MoE prefill

## Description

Current-head review: `patch-refactor` at `e4798a1083a9636decc91ad7238e67a3b4ab5ba8`, llama.cpp pin b11402
(`d89651a7b205c03c4a0b13cd0646d400dc929f79`).

This is a narrow prefill optimisation for the existing whole-expert path, not a second grouped-MoE design.

QFP26 already owns persistent/static-strided MoE MMQ scheduling and gate/up fusion. Patch 1283 already owns whole-expert placement
and the one-AllReduce block contract. Patch 1281 owns range `MUL_MAT_ID`. The missing first-order optimisation found in the
current code is inside 1281's large-batch MMQ preparation:

- upstream b11402 detects broadcast gate/up input with
  `dedup_bcast = ne11 == 1 && n_expert_used > 1`;
- it quantizes each physical token to Q8_1 **once** with
  `quantize_scatter_mmq_q8_1_cuda()`, then scatters that quantized row to the compact routed rows;
- 1281 changes that predicate to
  `ne11 == 1 && n_expert_used > 1 && !bc_range`;
- therefore every range gate/up projection falls back to `quantize_mmq_q8_1_cuda()` over
  `ne12 * n_expert_used` flattened rows even though the range MMQ only consumes rows belonging to local experts.

For Qwen4Exp top-k=10 this makes range-mode gate/up activation quantization perform roughly 10 source-row quantizations per token
instead of one. The exact end-to-end impact is unmeasured; MMQ and collectives can still dominate. This item restores the upstream
dedup mechanism for range MMQ without changing expert placement, MMQ tile math, compact scheduling, reduction order, graph shape,
or collective semantics.

Implement this directly inside `patches/1281_moe_mul_mat_id_range`, behind
`BIGCHERRY_MOE_RANGE_DEDUP=0/1` (default 0) during qualification. This restores an upstream optimisation disabled by 1281,
so 1281 remains the sole mechanism owner. If qualification is bit-identical with no regression, remove the permanent flag and make
eligible Q8_1 broadcast range MMQ use dedup unconditionally. Keep the old path only for shapes/types to which dedup does not apply
(per-route down projection and, until separately proven, native FP4).

Priority is P1: range nodes require `BIGCHERRY_MOE_EP=1`, which is not deployed; production row split already uses upstream
dedup. The reason to do this now is [MET08](../patching-moe-expert-tiering/MET08.md): expert split currently matches row-split
prefill while doing about 10x the gate/up activation quantization work.

## Current code review

### b11402 ordinary MMQ path

`ggml/src/ggml-cuda/mmq.cu` allocates:

```cpp
const int64_t n_expert_used = ids->ne[0];
const int64_t ne_get_rows   = ne12 * n_expert_used;

ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

ggml_cuda_launch_mm_ids_helper(
    (const int32_t *) ids->data,
    ids_src1.get(), ids_dst.get(), expert_bounds.get(),
    ne02, ne12, n_expert_used, ne11, si1, sis1,
    /*write_inverse=*/dedup_bcast, stream);
```

When `dedup_bcast` is true the Q8_1 quantizer launches over `n_tokens`, not `n_tokens * top_k`:

```cpp
quantize_scatter_mmq_q8_1_cuda(
    src1_d, ids_src1.get(), src1_q8_1.get(), src0->type,
    ne10, /*stride_token=*/s12, ne10_padded,
    ne12, ne11_flat, n_expert_used, stream);
```

The helper's inverse map is exactly what is needed: `ids_src1[token * top_k + slot] = compact_row`.

### 1281 range path

1281 first translates global ids into local ids:

```cpp
const int64_t local = (int64_t) ids[i] - (int64_t) id_base;
ids_local[i] = local >= 0 && local < n_local ? (int32_t) local : INT_MAX;
```

Then it deliberately disables broadcast dedup:

```cpp
if (bc_range) {
    ...
    CUDA_CHECK(cudaMemsetAsync(ids_src1.get(), 0,
                               ne_get_rows*sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(dst_d, 0, ggml_nbytes(dst), stream));
}

const bool dedup_bcast =
    ne11 == 1 && n_expert_used > 1 && !bc_range;
```

That was a safe first implementation because inactive route slots have no inverse-map entry. The consequence is now measurable:
the ordinary quantizer processes the full flattened `token * top_k` row count. The MMQ itself does not need those inactive
rows: `mm_ids_helper` builds `expert_bounds[]` only from local experts, and 1237/QFP26's compact task map also consumes those
same bounds.

### Existing scatter kernel seam

`ggml/src/ggml-cuda/quantize.cu` already quantizes once per token, then writes to every inverse-map slot:

```cpp
const int nwrite = scatter ? n_expert_used : 1;
for (int slot = 0; slot < nwrite; ++slot) {
    if constexpr (scatter) {
        const int64_t i =
            ids[(int64_t) blockIdx.x * n_expert_used + slot];
        ib = k_block * ne1 + i;
    }
    // store q8_1 block
}
```

The minimal range extension is therefore a sentinel for inactive slots; no new histogram, route sort, task map or matrix kernel
is needed.

## Steps

1. Modify `patches/1281_moe_mul_mat_id_range` directly. Add `BIGCHERRY_MOE_RANGE_DEDUP` default 0 as a qualification switch.
2. For range MMQ broadcast gate/up, prefill `ids_src1[]` with `-1`, request `mm_ids_helper`'s inverse map, and use a range-only Q8_1 scatter specialization that skips only `-1`.
3. Leave ordinary non-range scatter source/codegen unchanged. Leave per-route down projection, MMVQ/MMVF decode, and native-FP4 scatter unchanged in v1.
4. Keep `src1_q8_1` allocation unchanged; add no host readback/synchronization.
5. Extend `tests/test-mul-mat-id-range.cpp` with the real broadcast shape (`ne11 == 1`) and MMQ-forced cases. Cover flag off and flag on; require bit-identical F32 output.
6. Include nonzero `id_base`, inactive local slots, hostile routing, partial-J final-local-expert cases, and dirty/reused pool memory. Range dedup must populate every compact row in `[0, expert_bounds[n_local])`.
7. Prove ordinary non-range dedup unchanged and native FP4 remains on the old path.
8. Profile one production-shaped ~31-32K lane first; mandatory before broad A/B. Require profiler evidence that range-only scatter executes, quantizer work falls from ~top-k copies/token to one/token, and no new synchronization/kernel/task-count change appears.
9. Run same-binary ABBA with `tools/lab/flash-next/queue-env-ab.sh`: both arms export `BIGCHERRY_MOE_EP=1` plus identical expert shares; set `AB_ENV=BIGCHERRY_MOE_RANGE_DEDUP=1`; use greedy md5 and `FIDELITY=1`.
10. Test ub512 first for isolation; also test validated QSA ub1024 + `BIGCHERRY_QSA_CHUNK=256` as a separate lane.
11. Promotion gate: bit-identical backend/greedy output, activation evidence, and no repeatable prefill/decode/MTP/memory regression. Quantizer-time reduction is activation evidence, not an end-to-end materiality bar.
12. If promoted, remove the qualification flag and keep dedup unconditional for eligible Q8_1 broadcast range MMQ. Only then consider the separate range-id helper-fold phase if profiling shows it material.

## Detailed Solution & Technical Design

### 1. Range inverse-map representation

Use `-1` as the only inactive inverse-map sentinel.

For an active global route `g` on device d:

```text
local = g - id_base[d]
0 <= local < n_local[d]
```

`mm_ids_helper` sees that local expert and writes the same compact row number it already computes for ordinary MMQ:

```text
ids_src1[token * top_k + slot] = compact_row
```

For a route not held on d, 1281 translates it to `INT_MAX`; no local expert matches it, so the prefilled `-1` survives.

Invariant:

```text
every compact row r in [expert_bounds[e], expert_bounds[e+1])
has exactly one active inverse-map entry equal to r;
inactive route slots remain -1 and are never read by MMQ.
```

The candidate therefore changes only how the source activation is quantized/copied into the existing compact-row scratch.
The MMQ receives the same `ids_dst`, `expert_bounds`, weights, tile selection and destination layout as before.

### 2. `mmq.cu` host-side change

Source-shaped implementation:

```cpp
const bool bc_range = ggml_mul_mat_id_is_range(dst);

const bool bc_range_dedup =
    bc_range &&
    bc_moe_range_dedup_enabled() &&
    !use_native_fp4 &&
    ne11 == 1 &&
    n_expert_used > 1;

ggml_cuda_pool_alloc<int32_t> bc_ids_local(ctx.pool());
const int32_t * bc_ids = (const int32_t *) ids->data;

if (bc_range) {
    const int64_t bc_n_ids =
        (int64_t) (ids->nb[1] / ggml_element_size(ids)) * ne12;

    bc_ids_local.alloc(bc_n_ids);
    ggml_cuda_mm_ids_range_translate(
        bc_ids, bc_ids_local.get(), bc_n_ids,
        ggml_mul_mat_id_range_base(dst),
        (int32_t) ne02, stream);
    bc_ids = bc_ids_local.get();

    // -1 is an inactive inverse-map entry only when scatter dedup is active.
    CUDA_CHECK(cudaMemsetAsync(
        ids_src1.get(),
        bc_range_dedup ? 0xff : 0x00,
        ne_get_rows * sizeof(int32_t),
        stream));

    // Range contract remains exact +0 for routes not held on this device.
    CUDA_CHECK(cudaMemsetAsync(
        dst_d, 0, ggml_nbytes(dst), stream));
}

const bool dedup_bcast =
    ne11 == 1 &&
    n_expert_used > 1 &&
    (!bc_range || bc_range_dedup);
```

No second helper is needed. Keep the existing call:

```cpp
ggml_cuda_launch_mm_ids_helper(
    bc_ids,
    ids_src1.get(), ids_dst.get(), expert_bounds.get(),
    ne02, ne12, n_expert_used, ne11, si1, sis1,
    /*write_inverse=*/dedup_bcast, stream);
```

This means the ordinary path is byte-for-byte source-equivalent when the experiment flag is off.

### 3. Q8_1 scatter-kernel change

In `quantize_mmq_q8_1<ds_layout, true>`:

```cpp
for (int slot = 0; slot < nwrite; ++slot) {
    int64_t ib;

    if constexpr (scatter) {
        const int64_t i =
            ids[(int64_t) blockIdx.x * n_expert_used + slot];

        // BigCherry 1281/QFP30: range-MMQ inverse maps leave non-local route
        // slots at -1. They have no compact row and need no quantized copy.
        if (i == -1) {
            continue;
        }

        GGML_ASSERT(i >= 0 && i < ne1);
        ib = k_block * ne1 + i;
    } else {
        ...
    }

    // existing q8_1 stores unchanged
}
```

If device-side `GGML_ASSERT` is not supported/appropriate in this compilation path, keep the runtime branch exactly
`i == -1` and prove the positive bound in the backend test. Do not silently treat arbitrary negative/oversized indices as
inactive; that would hide route-map corruption.

### 4. Why this is preferable to another route compactor

Do not compact active tokens a second time.

At top-k 10 and an approximately balanced three-way expert split, almost every token has at least one local route on every large
share device. A second per-token activity scan would add another kernel/map and complicate graph capture for little expected gain.
The existing inverse map already contains the information required to avoid the 10x source quantization work.

Do not shrink `src1_q8_1` in v1 either. The exact active compact-row total is device-computed in
`expert_bounds[ne02]`; reading it to the CPU would add the synchronization this path is designed to avoid. The existing allocation
is an upper bound and keeps scheduler/pool behaviour identical for a clean performance attribution.

### 5. Interaction with 1237/1265 and QFP26

1237's compact-grid path consumes `expert_bounds[]`; QFP30 does not change them. Therefore:

```text
1281 range-id translation
        |
        v
mm_ids_helper
  |             |
  |             +--> ids_src1 inverse map -> range-only Q8_1 scatter
  |
  +--> ids_dst + expert_bounds -> existing MMQ / 1237 compact map
```

QFP26 can later schedule/fuse those same compact MMQ tasks. QFP30 must not add a persistent queue, change J, merge gate/up,
or retain the quantized activation between the two projections. Those remain QFP26 territory.

### 6. Optional phase 2: remove the translation launch only after measurement

1281 currently launches `mm_ids_range_translate` into `bc_ids_local`, then `mm_ids_helper`. If rocprof shows translation
plus its scratch handling at >=0.5% of prefill wall or >=3% of routed-MoE wall after v1, a second experiment may extend
`mm_ids_helper` with `id_base/n_local` and translate the loaded global id in registers.

Phase 2 must preserve the exact helper output and be benchmarked separately. It is rejected by default; eliminating one small
kernel is not justification for mixing it into the high-value v1 result.

## Patch integration

Owner: `patches/1281_moe_mul_mat_id_range`.

During qualification, add `BIGCHERRY_MOE_RANGE_DEDUP` to 1281's env docs with default `0`. Update
`tools/tests/patch/test_1281_moe_mul_mat_id_range.py` for patch mechanics/idempotence and exact guards. Do not create a second
range-MMQ package. After successful qualification, remove the experiment switch and promote the eligible Q8_1 broadcast range
path inside 1281.

## Files

Implementation files:

- `patches/1281_moe_mul_mat_id_range/patch.py` - qualification switch and range-MMQ/scatter edits.
- `ggml/src/ggml-cuda/mmq.cu` - range inverse-map initialization and range-enabled `dedup_bcast`.
- `ggml/src/ggml-cuda/quantize.cu` - `-1` skip in Q8_1 scatter mode.
- `tools/tests/patch/test_1281_moe_mul_mat_id_range.py` - patch composition/idempotence and exact guard assertions.
- existing `tests/test-mul-mat-id-range.cpp` or an adjacent backend test - MMQ-forced range correctness fixture.
- `tools/lab/flash-next/queue-env-ab.sh` - end-to-end same-binary A/B.
- existing QFP17/QFP22 rocprof lane - kernel attribution.

Do not modify:

- `src/llama-graph.cpp` - graph topology is already correct.
- `src/llama-model.cpp` - expert placement remains 1283.
- `ggml/src/ggml-backend-meta.cpp` - collective and split semantics remain 1283/MSM.
- `mmq.cuh` task scheduling - QFP26/1237 owner.
- QSA/chunking patches - independent experiment.

## Validation

### Offline mechanics

Update 1281's patch test to prove idempotence/fail-closed anchors; flag-off preserves current 1281; enabled range mode uses a
`-1` inverse-map sentinel and a range-only Q8_1 scatter specialization; ordinary non-range scatter remains source/codegen
unchanged; production patch composition remains valid.

### Backend correctness

Force MMQ and test both `ne11 == 1` broadcast and existing per-route shapes. Include zero/nonzero bases, beginning/middle/end
local ranges, top-k 2/4/10, zero-local-route tokens, skewed routing, and partial-J tails with deliberately dirty/reused pool memory.
For eligible Q8_1 broadcast cases require bitwise F32 equality flag-off vs flag-on. Both forms use the same
`quantize_mmq_q8_1` arithmetic; scale/sum state is derived only from the source row, so quantize-once+scatter should equal
repeated quantization.

If equality holds, close disabled dedup as a cause of MET09's expert-split vs row-split numeric difference; investigate the
remaining arithmetic/summation-order causes there.

### Hardware / promotion

Profile one ~31-32K lane first and require visible range-scatter activation plus the expected quantizer-work drop, with no extra
sync/kernel/task-count change. Then use `tools/lab/flash-next/queue-env-ab.sh` for same-binary ABBA with identical
`BIGCHERRY_MOE_EP=1`/expert shares, `AB_ENV=BIGCHERRY_MOE_RANGE_DEDUP=1`, greedy md5 and `FIDELITY=1`. Run ub512 first;
QSA ub1024 + chunk 256 is validated and may be a separate lane.

Promote on bit-identical output + activation evidence + no repeatable regression. There is no minimum end-to-end percentage bar;
small wins count when nothing regresses.

## Expected performance model

For gate/up broadcast input:

```text
old range source quantizations ~= n_tokens * top_k
new range source quantizations ~= n_tokens
```

For top-k 10, source loads/max-reductions/quantization arithmetic are therefore reduced by about 10x for this preprocessing
kernel. Destination scatter writes remain proportional to local active route slots. The full `src1_q8_1` scratch remains
allocated in v1, so this is primarily a bandwidth/compute/launch-duration optimisation, not a memory-capacity optimisation.

This does **not** imply 10x MoE or prefill speed. Weight MMQ, attention and one AllReduce per block remain. The required end-to-end
gate above is deliberately much smaller.

## Effort & Risk

Medium/low blast radius relative to QFP26.

Main risks:

- a `-1` inverse entry reaches an unmodified scatter kernel and becomes an out-of-bounds write;
- an active compact row is not populated because inverse-map generation and `expert_bounds` disagree;
- a future native-FP4 activation path accidentally inherits Q8-only assumptions;
- an apparent kernel win is hidden by or confused with QFP22/QSA changes;
- patch composition with 1237/1265 changes the same MMQ neighborhood.

Mitigations are strict Q8-only gating, exact backend equality, hostile routing, same-binary ABBA and explicit composition tests.

## Standards

- One owner per mechanism: QFP30 = range broadcast-activation dedup only.
- QFP26 = persistent compact scheduling + gate/up fusion.
- 1281 = range `MUL_MAT_ID` semantics.
- 1283 = expert placement + delayed one-AllReduce block semantics.
- No model-specific numeric tuning in code; QFP23 owns profile values.
- Default off only during qualification; remove the switch after successful promotion.
- No host readback/synchronization for scratch sizing.
- No new graph op and no graph-topology change.
- Record the measured result; retain any bit-identical no-regression win regardless of end-to-end size.

## Acceptance Criteria

- QFP30 is implemented inside 1281; no new patch id/package.
- Qualification flag-off is identical to current 1281; eligible Q8_1 broadcast flag-on is bit-identical and uses the range-only scatter path.
- Broadcast `ne11 == 1` MMQ range coverage exists for both flag states, including nonzero bases, inactive slots and dirty partial-J tails.
- Ordinary non-range dedup is unchanged; per-route down projection and native FP4 remain on their existing paths.
- Profiler proves the expected activation-quantizer work reduction without new synchronization/task-count changes.
- Same-binary ABBA uses queue-env-ab, greedy md5 and `FIDELITY=1`; no repeatable regression is introduced.
- On promotion, remove the qualification flag and make eligible Q8_1 broadcast range dedup unconditional.
- QFP26 remains free to build scheduler/fusion work on top without duplicate route maps or quantizers.

## Notes

Review result: the large Strata-style grouped/fused prefill project already exists as QFP26. QFP30 is P1 and linked to MET08. It is intentionally smaller:
restore an optimisation already present in b11402 ordinary routed MMQ but disabled by the first 1281 range implementation. It is
a good pre-QFP26 benchmark because it is low-risk, requires no new matrix kernel, and gives a clean answer about how much of the
current whole-expert prefill gap is activation preparation rather than MMQ/collectives.

## Change Log

- 2026-10-06T15:41:00+00:00: created from current-head review of 1281/1283, b11402 MMQ/mmid/quantize paths and QFP26 ownership.
- 2026-10-07: converged RV4221: P1/MET08; implementation stays in 1281; qualification-only flag; broadcast op test both states; QSA status corrected; queue-env ABBA specified; percentage promotion bars removed; MET09 dedup hypothesis closes on bit-identity; partial-J rationale corrected to the upstream padded-tail invariant.


## Self-review correction — 2026-10-07

Two corrections override the earlier implementation sketch before QFP30 is coded into 1281.

1. Do not add the missing-slot check to the existing generic Q8_1 scatter instantiation. That would change ordinary non-range MoE codegen even with BIGCHERRY_MOE_RANGE_DEDUP=0. Add a range-only compile-time specialization/wrapper (for example a third template boolean with an if-constexpr missing-slot check), and call it only from the enabled range arm. The existing ordinary scatter wrapper/instantiation must stay source/codegen-equivalent.
2. MMQ reads full J-wide Y tiles and masks the tail at writeback. Upstream ordinary dedup already leaves only the padded tail beyond the final compact row unwritten; range dedup must not create holes inside `[0, expert_bounds[n_local])`. Keep the partial-J final-local-expert dirty-pool test as regression proof of that invariant. If padded-tail data affects active outputs, fix the shared MMQ invariant rather than treating it as range-specific.

Also strengthen activation evidence: the range-only wrapper/kernel symbol must be visible in profiling, while the ordinary non-range scatter symbol/path remains unchanged.

Verdict after these corrections: implementable and worth benchmarking; promotion requires bit identity, activation evidence and no regression, with no minimum end-to-end percentage bar.

## Reviews

- RV4221
