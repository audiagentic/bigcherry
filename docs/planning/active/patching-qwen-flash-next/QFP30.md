---
id: QFP30
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-06T15:41:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
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

The benchmarkable first step is a new default-off experiment patch:

`patches/1342_moe_range_mmq_dedup/`

with `BIGCHERRY_MOE_RANGE_DEDUP=1`.

If it passes the gates below, fold the mechanism into 1281 (range semantics owner) and remove/retire 1342 rather than maintaining
two permanent range-MMQ implementations.

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

1. Add experiment package `1342_moe_range_mmq_dedup`, default off, anchored to 1281's inserted range-MMQ code so it fails
   closed if 1281 is absent or changes.
2. For range MMQ gate/up only, initialize `ids_src1[]` to `-1` instead of zero and request the existing inverse map from
   `mm_ids_helper`.
3. Teach only the existing Q8_1 scatter quantizer to skip inverse entries equal to `-1`.
4. Enable upstream `dedup_bcast` for range MMQ when `ne11 == 1`, `n_expert_used > 1`, Q8_1 activation quantization is
   selected, and `BIGCHERRY_MOE_RANGE_DEDUP=1`.
5. Keep `src1_q8_1` allocation size unchanged in v1. Do not copy `expert_bounds[n_local]` to the host and do not introduce
   a synchronization just to shrink scratch.
6. Leave down-projection MMQ unchanged: its input is per-route, not a `ne11 == 1` broadcast.
7. Leave MMVQ/MMVF decode paths unchanged. This item is large-batch MMQ prefill only.
8. Leave MXFP4/NVFP4 native-FP4 activation scatter unchanged in v1; on the AMD target `use_native_fp4` is false. Extend FP4
   only after separate correctness evidence if it ever becomes relevant.
9. Add an MMQ-forced range backend test with nonzero `id_base`, top-k routing and inactive local slots. Compare candidate to
   the old range-MMQ path.
10. Profile one 31-32K prefill lane before broad A/B. Continue end-to-end testing only if the Q8_1 quantizer time materially
    drops and no extra synchronization/kernel appears.
11. Run production ABBA at short and long context with the same binary, toggling only
    `BIGCHERRY_MOE_RANGE_DEDUP=0/1`.
12. Only if the separate range-id translation kernel remains material after this win, test a phase-2 helper fold. Do not combine
    that change with v1.

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

        // BigCherry 1342: range-MMQ inverse maps leave non-local route
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

1237's compact-grid path consumes `expert_bounds[]`; 1342 does not change them. Therefore:

```text
1281 range-id translation
        |
        v
mm_ids_helper
  |             |
  |             +--> ids_src1 inverse map -> 1342 Q8_1 scatter
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

## Patch package

Proposed package:

```text
patches/1342_moe_range_mmq_dedup/
  patch.py
  README.md                 # only after hardware evidence/promotion
tools/tests/patch/
  test_1342_moe_range_mmq_dedup.py
```

`patch.py`:

```python
GROUP = "core"
STATE = "untested"

ENV_DOCS = (
    EnvDoc(
        "BIGCHERRY_MOE_RANGE_DEDUP", "0|1", "0",
        "range MUL_MAT_ID MMQ: quantize a broadcast gate/up activation "
        "once per token and scatter only to locally active compact rows; "
        "Q8_1 activation path only"
    ),
)
```

Anchors should deliberately target 1281's BigCherry strings in `mmq.cu`, not pristine upstream text. That makes the dependency
mechanical: applying 1342 without 1281 fails instead of accidentally modifying ordinary MMQ.

If patch composition tooling supports an explicit dependency declaration, add 1281. 1283 is required for the production
whole-expert benchmark, but 1342 correctness can also be tested against a standalone range op without 1283.

## Files

Implementation files:

- `patches/1342_moe_range_mmq_dedup/patch.py` - experiment switch and anchored edits.
- `ggml/src/ggml-cuda/mmq.cu` - range inverse-map initialization and range-enabled `dedup_bcast`.
- `ggml/src/ggml-cuda/quantize.cu` - `-1` skip in Q8_1 scatter mode.
- `tools/tests/patch/test_1342_moe_range_mmq_dedup.py` - patch composition/idempotence and exact guard assertions.
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

`test_1342_moe_range_mmq_dedup.py` must:

1. apply 1281 then 1342 to pristine b11402 source;
2. prove idempotence;
3. prove 1342 fails closed if 1281's range-MMQ anchor is absent;
4. assert flag-off leaves 1281's old `!bc_range` behaviour reachable;
5. assert candidate range mode fills inverse map with `0xff`;
6. assert only Q8_1 scatter mode accepts/skips `-1`;
7. assert ordinary non-range dedup remains unchanged;
8. compose with 1237/1265/1283 in the production patch order.

### Backend correctness fixture

Force a token count above the MMVQ/MMVF threshold so `MUL_MAT_ID` dispatches MMQ. Cases:

- `id_base = 0` and nonzero `id_base`;
- local expert ranges at beginning/middle/end of global expert space;
- top-k `{2,4,10}`;
- uniform routing;
- one hot expert;
- highly skewed/Zipf-like routing;
- tokens with zero local routes;
- local range containing only one of a token's routes;
- tail token counts not aligned to MMQ J;
- production quant types at least `IQ4_XS`, `IQ3_S`, `IQ4_NL`, `Q8_0` where backend support exists.

Compare:

```text
A = 1281 range MMQ, dedup flag off
B = 1281 + 1342, dedup flag on
```

Require exact/bitwise F32 destination equality for the same MMQ path. This is a data-movement change before unchanged Q8_1/MMQ
math; numerical tolerance is not the intended contract.

Also verify the ordinary non-range op against pristine b11402 and ensure native-FP4 configurations take the old path.

### Hardware performance discriminator

First profile one production-shaped run:

```text
model: Flash-Next UD-IQ4_XS
target: 2x RX 7900 XTX (gfx1100) + R9700 (gfx1201)
mode: tensor split + BIGCHERRY_MOE_EP=1
KV: f16
ubatch: 512
prompt/depth: ~31.8K
A: BIGCHERRY_MOE_RANGE_DEDUP=0
B: BIGCHERRY_MOE_RANGE_DEDUP=1
all other environment/profile values identical
```

Use the existing QFP17/QFP22 rocprof workflow. Required profiler evidence per device:

- range gate/up calls switch from ordinary `quantize_mmq_q8_1<..., false>` to scatter
  `quantize_mmq_q8_1<..., true>`;
- quantizer grid token dimension changes from approximately `n_tokens * top_k` work to `n_tokens` work;
- no extra host synchronization;
- no change in MMQ task count, J selection, compact map, AllReduce count, or graph topology;
- summed gate/up Q8_1 quantizer time per layer/device;
- total routed-MMQ time and total prefill wall.

If quantizer time does not drop >=30%, stop and diagnose activation/dispatch before broad A/B; the intended path is probably not
executing.

Then run same-binary ABBA, at least four samples per arm where practical:

```text
~8K prefill
~32K prefill
~80K or ~98K prefill
~200K prefill if the current production context fits
```

Use ub512 first to isolate this mechanism from unresolved QSA chunk identity. Add ub1024 only after QFP22/1332 has a green
correctness gate; do not use a correctness-blocked QSA path to claim the MoE result.

Capture:

- prefill t/s and TTFT;
- per-rank quantizer, MMQ and RCCL/AllReduce kernel sums;
- critical-rank wall;
- peak compute/pool bytes;
- greedy output hash;
- decode/MTP throughput and acceptance as a regression lane.

### Promotion gate

Promote/fold into 1281 only if all hold:

- MMQ-forced range backend output is bit-identical A/B across hostile routing cases;
- production greedy output hash is identical A/B;
- activation evidence proves range gate/up used scatter dedup;
- range gate/up Q8_1 quantizer summed time drops >=50% on the critical rank in the profiler lane;
- end-to-end prefill improves >=3% median on at least one representative 32K-or-longer lane with no >1% repeated regression
  on another qualified lane;
- no >1% decode/MTP regression;
- no unexpected graph/scheduler reallocation;
- peak compute memory does not increase >1%;
- no new synchronization or D2H copy.

If quantizer time drops strongly but E2E gain is <1.5%, park the patch and let QFP26's gate/up fusion subsume the opportunity.
Do not carry permanent complexity for an invisible end-to-end gain.

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
- Default off until hardware evidence.
- No host readback/synchronization for scratch sizing.
- No new graph op and no graph-topology change.
- Record a negative result and stop if the Amdahl gate is not met.

## Acceptance Criteria

- QFP30 plan has an isolated 1342 implementation path that composes with the current production patch set.
- Flag-off is identical to current 1281.
- Flag-on reuses upstream inverse-map/scatter machinery and skips only `-1` non-local routes.
- MMQ range correctness is bit-identical to flag-off.
- gfx1100 and gfx1201 both pass backend tests.
- Profiler proves the expected gate/up quantizer work reduction.
- Production ABBA meets the promotion gate or the experiment is explicitly parked.
- A winner is folded into 1281; 1342 does not become a second permanent owner of range semantics.
- QFP26 remains free to build its scheduler/fusion work on top without duplicate route maps or quantizers.

## Notes

Review result: the large Strata-style grouped/fused prefill project already exists as QFP26. This item is intentionally smaller:
restore an optimisation already present in b11402 ordinary routed MMQ but disabled by the first 1281 range implementation. It is
a good pre-QFP26 benchmark because it is low-risk, requires no new matrix kernel, and gives a clean answer about how much of the
current whole-expert prefill gap is activation preparation rather than MMQ/collectives.

## Change Log

- 2026-10-06T15:41:00+00:00: created from current-head review of 1281/1283, b11402 MMQ/mmid/quantize paths and QFP26 ownership.


## Self-review correction — 2026-10-07

Two corrections override the earlier implementation sketch before 1342 is coded.

1. Do not add the missing-slot check to the existing generic Q8_1 scatter instantiation. That would change ordinary non-range MoE codegen even with BIGCHERRY_MOE_RANGE_DEDUP=0. Add a range-only compile-time specialization/wrapper (for example a third template boolean with an if-constexpr missing-slot check), and call it only from the enabled range arm. The existing ordinary scatter wrapper/instantiation must stay source/codegen-equivalent.
2. Range scatter fills only active compact rows. MMQ reads full J-wide Y tiles and masks the tail at writeback. Add a partial-J final-local-expert test after deliberately dirtying/reusing the GPU pool, and require bit-identical active outputs. If stale padded rows affect active results, add the smallest required guard/zero initialization and benchmark that cost separately; do not assume the sparse write is safe.

Also strengthen activation evidence: the range-only wrapper/kernel symbol must be visible in profiling, while the ordinary non-range scatter symbol/path remains unchanged.

Verdict after these corrections: implementable and worth benchmarking; end-to-end value remains gated by the existing >=3% qualified prefill threshold.

## Reviews

- RV4221
