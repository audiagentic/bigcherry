---
id: QFP38
order: 38
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:39:48.982783+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Small-batch vector kernels for MTP verify: row packing, wide-K, MoE vector path to 8 tokens

## Description

External decode work reported gains from small-batch MMVQ/MMVF launch geometry: small-K row packing, wider K participation, keeping MoE verify on a vector path, packing more expert rows per block, and larger blocks for tiny long-K F32 matrices. Those results are from different hardware and are hypotheses only.

At b11402 several parts already exist, and one tempting Q8_0 variant has already been rejected on the correct BigCherry model. QFP38 must start from the actual MTP verify kernel census, not re-run the external patch set wholesale.

## What we already have

### b11402 MMVQ

- `ggml/src/ggml-cuda/mmvq.cuh` sets `MMVQ_MAX_BATCH_SIZE=8`.
- `ggml/src/ggml-cuda/mmvq.cu::get_mmvq_mmid_max_batch_rdna3`
  - IQ4_XS: 6 tokens;
  - other unlisted types, including Q8_0, use the global max 8.
- `get_mmvq_mmid_max_batch_rdna4`
  - IQ4_XS: 5;
  - Q8_0: 7.
- Therefore a 4-token verify batch, and the external 5-token IQ4_XS example, already stay on MMVQ on gfx1100/gfx1201. **Do not add a threshold patch for our current verify width.**
- `mul_mat_vec_q_switch_ncols_dst` compiles widths 1..8.
- For `ids && ncols_dst > 1`, it uses the dedicated `mul_mat_vec_q_moe_launch`.
- `mul_mat_vec_q_moe_launch` currently fixes `rows_per_block=2` and launches `block_dims=(warp_size,ncols_dst)`.
- For ordinary multi-token MMVQ, RDNA3/RDNA4 `calc_nwarps()` returns 1 when `ncols_dst>1`; the existing 8-warp RDNA tuning is single-token only.
- The upstream `should_use_small_k()`/row-packing mechanism is explicitly disabled for every RDNA architecture, and `calc_rows_per_block()` does not assign an RDNA small-K packing geometry.

### b11402 MMVF

- `ggml/src/ggml-cuda/mmvf.cu::launch_mul_mat_vec_f_cuda` chooses a block size in warp increments by reducing K-loop iterations.
- The current search caps RDNA at 256 threads and only instantiates block sizes through 256.
- Thus the external 512/1024-thread F32 experiment is not already present. It is only relevant if the MTP verify census shows tiny-row, long-K F32 MMVF nodes on the critical path.

### BigCherry 1301: read before changing Q8 verify widths

`patches/1301_prbe115_q8_f32_mtp_widths` is **rejected**.

It widened 1241's raw-F32-activation Q8_0 MMVQ path so widths 2..5 bypassed Q8_1 activation quantization. On Qwen3.8-27B Q8_0 dual-XTX with built-in MTP4:

- width cap 2: neutral;
- width cap 3: neutral;
- width cap 5: about -12% at 10K and -6% at 32K;
- activation was proven for widths 1..5.

Conclusion: raw-F32 Q8_0 MMVQ is not the answer for MTP verify on our model. QFP38 must leave the normal quantize-to-Q8_1 + MMVQ data path intact.

### BigCherry 1273

`patches/1273_iq_mmvq_rdna_tuning` is **evaluated, not validated/promoted**.

It provides independently gated lower-VDR and nwarps variants for **single-token** IQ4_XS/IQ3_XXS on gfx1100/gfx1201. Its summary intentionally makes no hardware performance/correctness claim. It also adds compile-time launch-geometry override plumbing that can be reused conceptually.

Do not infer multi-token benefit from 1273. Any new single-token IQ4_XS VDR/nwarps work belongs inside 1273; QFP38 should own only distinct multi-token/MoE/F32 verify geometry.

## Steps

1. Gate 0: capture a kernel/graph census for one MTP verify round on:
   - Flash-Next with RX6900XT sidecar;
   - 27B Q8_0 built-in MTP on two XTXs.
2. Record, per verify width 2..5 as observed: op name, type, ids/non-ids, physical M/N/K, chosen MMVQ/MMQ/MMVF path, launch geometry, kernel time and fraction of round.
3. Close the already-covered MMVQ threshold sub-item unless a real production verify width exceeds the current architecture/type threshold.
4. Qualify only the highest-time remaining class, one launch change at a time.
5. Do not combine with 1301 raw-F32 activation.
6. ABBA each candidate with MTP acceptance and greedy target identity.

## Detailed Solution & Technical Design

### A. Ordinary multi-token wide-K MMVQ

If Gate 0 shows ordinary quantized widths 2..5 with large K and one-warp under-utilization, create:

`patches/1352_mtp_mmvq_widek`

Flag:

`BIGCHERRY_MTP_MMVQ_WIDEK=0|1`, default `0`.

Keep the existing Q8_1 activation path and vec-dot implementation. Compile an alternate multi-token specialization with an explicit nwarps value; start with 8 only because that is the external hypothesis.

Dispatch only when:

- `ids == nullptr`;
- `2 <= ncols_dst <= 5` or the exact observed verify width;
- type/architecture/physical K match a measured winning class;
- the wider block actually reduces K-loop trips enough to justify its reduction/register cost;
- no fusion contract is lost.

Use the same per-output arithmetic and warp reduction as MMVQ. This is launch geometry, not a new quant kernel.

Qualify gfx1100 and gfx1201 independently. gfx1030 is only relevant if the sidecar census shows the candidate; it must not inherit an XTX threshold.

### B. RDNA small-K row packing

The existing generic `small_k` concept packs more output rows into a block when K is too short to keep a multi-warp block busy, but upstream disables it on RDNA.

If the census finds a matching ordinary MMVQ node, keep this in the same 1352 launch-geometry package behind a separate gate:

`BIGCHERRY_MTP_MMVQ_SMALLK_ROWS=<N>`, default 0.

Compile an explicit rows-per-block specialization. Do not simply remove `GGML_CUDA_CC_IS_RDNA(cc)` from the upstream exclusion: RDNA's current parameter table does not define the intended packed layout, and a blanket enable would retune all quant types.

Eligibility must name the measured type, width and K range and use physical post-Meta dimensions. Sweep N=2/4 only before considering larger packing.

### C. Multi-token MoE expert rows

Current `mul_mat_vec_q_moe_launch` already owns the width 2..8 vector path and uses `rows_per_block=2`.

If profiler evidence shows short physical expert output slices are launch/occupancy limited, create a separate mechanism:

`patches/1353_moe_mmvq_rowpack`

Flag:

`BIGCHERRY_MOE_MMVQ_ROWS=<2|4|8>`, default 2/current behavior.

Compile `mul_mat_vec_q_moe<type,rows_per_block,...>` variants and select only for:

- `ids != nullptr && ncols_dst > 1`;
- current call already passed `get_mmvq_mmid_max_batch(type,cc)`;
- physical `nrows_x` lies in the measured short-row range;
- type/architecture was benchmarked.

This composes with 1281 range ids because the kernel continues to consume the same ids/fusion fields. Do not change id translation, expert selection, range zero semantics or QFP30 activation scatter.

For our current IQ4_XS verify width, do **not** raise `get_mmvq_mmid_max_batch`: width 4/5 is already covered on gfx1100/gfx1201. If a future adaptive depth produces width >5 on gfx1201, treat threshold extension as a separate measured crossover experiment rather than part of row packing.

### D. Tiny long-K F32 MMVF blocks

Only if Gate 0 attributes material verify time to F32 MMVF with very few output rows and long K, create:

`patches/1354_mtp_mmvf_wideblock`

Flag:

`BIGCHERRY_MTP_MMVF_WIDEBLOCK=0|512|1024`, default 0.

Extend `launch_mul_mat_vec_f_cuda` with separately compiled 512/1024 block-size cases, guarded by runtime `maxThreadsPerBlock`/architecture capability and a narrow physical shape predicate. Do not globally raise `max_block_size`.

The kernel's F32 dot reduction order changes with block size, so this is an explicit equivalence experiment. Also watch occupancy/VGPR/LDS: a 1024-thread block can reduce residency and lose despite fewer K iterations.

QFP34 is distinct: it targets F32 MUL_MAT widths 9-16 that fall to rocBLAS. QFP38-D only tunes already-selected MMVF at small verify widths <=8.

### Architectures, Meta and fusion

- gfx1100/gfx1201: built-in target/MTP verify qualification.
- gfx1030: sidecar MTP qualification only where its actual census hits the candidate.
- Use architecture family + physical tensor shape, never device ordinal.

Meta may split matrices before backend dispatch. All thresholds use the local physical `nrows_x/ncols_x/ncols_dst`, not logical model dimensions.

Existing MMVQ/MMVF fusion eligibility must be preserved. Width>1 MMVQ currently does not use the single-token fusion branch; do not claim a fusion win. The MoE path retains current fusion arguments exactly.

## Code Samples & Guidance

Primary b11402 anchors:

- `ggml/src/ggml-cuda/mmvq.cu::get_mmvq_mmid_max_batch_rdna3/rdna4` — audit only; no current width-4/5 change required.
- `calc_nwarps()` — RDNA width>1 currently returns 1.
- `should_use_small_k()` — explicit RDNA disable.
- `calc_rows_per_block()`.
- `mul_mat_vec_q_moe_launch()` — `constexpr int rows_per_block = 2`.
- `mul_mat_vec_q_switch_ncols_dst()` — real width dispatch.
- `ggml/src/ggml-cuda/mmvf.cu::launch_mul_mat_vec_f_cuda` — current 256-thread cap and block-size switch.

Do not anchor new work in 1301's raw-F32 branch.

Suggested markers:

`BIGCHERRY_PATCH_HIT 1352 mtp_mmvq width=<N> k=<K> nwarps=<W> rows=<R>`

`BIGCHERRY_PATCH_HIT 1353 moe_mmvq_rowpack type=<...> width=<N> rows=<R> physical_rows=<M>`

`BIGCHERRY_PATCH_HIT 1354 mtp_mmvf_wideblock width=<N> k=<K> block=<B>`

Once per representative shape/process.

## Files

Only create a package after Gate 0 identifies its class:

- `patches/1352_mtp_mmvq_widek/` + `tools/tests/patch/test_1352_mtp_mmvq_widek.py`;
- `patches/1353_moe_mmvq_rowpack/` + matching test;
- `patches/1354_mtp_mmvf_wideblock/` + matching test.

Likely production file for 1352/1353: `ggml/src/ggml-cuda/mmvq.cu`.
Likely production file for 1354: `ggml/src/ggml-cuda/mmvf.cu`.

If the only remaining idea is single-token IQ VDR/nwarps, update **existing patch 1273** instead of creating 1352.

Do not modify 1301; it remains rejected evidence.

## Validation

### Offline mechanics

For each qualified sub-change:

- exact/idempotent patch mechanics;
- patch-lint;
- flag off/current value is source/dispatch equivalent;
- backend-op sweep at widths 1..8 around the targeted physical M/K sizes;
- near-threshold cases prove fallback to current dispatch.

For 1353 include ordinary and 1281 range MUL_MAT_ID cases and fused/unfused MoE paths.

### Activation/equivalence

- Census proves the intended kernel replaces only the measured class.
- Quantized MMVQ geometry changes should preserve quant math; compare against current backend output at normal op-test tolerances and require greedy target identity.
- MMVF wider blocks use explicit F32 tolerance because reduction grouping changes.
- MTP acceptance/depth is recorded for every ABBA; a kernel win that lowers acceptance enough to reduce end-to-end decode is rejected.

### Hardware ABBA

Use `tools/lab/flash-next/queue-env-ab.sh` with complete process separation.

Run candidates first on the model/device where Gate 0 found them:
- 27B Q8_0 built-in MTP for Q8 multi-token work;
- Flash-Next sidecar for gfx1030-specific MTP shapes;
- Flash-Next target ranks for IQ4_XS MoE verify if applicable.

Use 8K and 24K decode points; ~98K only if the verify kernel remains a material fraction there. Record ms/speculative round, verify-kernel time, accepted tokens/round, total t/s and greedy target identity.

No multi-session contract campaign.

## Effort & Risk

Effort: M per actual launch-geometry candidate; L only if all external sub-ideas are attempted, which is explicitly not recommended.

Risk: medium. These are numerically simple launch changes, but register pressure/occupancy crossovers are architecture/type/width specific and the wrong variant can regress badly, as 1301 demonstrated for a different data path.

Expected gain on our topology: low-to-medium until census proves otherwise. The current pin already has a dedicated multi-token MoE kernel and already keeps our 4/5-token IQ4_XS verify on MMVQ. Therefore QFP38 ranks below QFP35/QFP36 unless the verify census identifies a dominant ordinary wide-K kernel.

## Standards

- Do not repeat 1301's raw-F32 Q8 activation experiment.
- Do not raise the current MMVQ MoE threshold for width 4/5; it is already sufficient.
- 1273 remains the owner of single-token IQ launch/VDR tuning.
- One sub-change/flag at a time.
- No fixed GPU ordinal/topology or attention split.
- No q4 KV.
- No legacy/back-compat shim.
- Default off during qualification.
- External numbers are hypotheses.

## Acceptance Criteria

- A verify census identifies the exact remaining kernel class and physical shape before code.
- The current 4/5-token IQ4_XS MMVQ threshold is documented as already covered and receives no duplicate patch.
- No 1301 raw-F32 route is reintroduced.
- Each implemented candidate has independent activation evidence and a separated ABBA.
- Backend equivalence passes; target greedy output is identical.
- MTP acceptance is neutral or the end-to-end gain still survives.
- A candidate is promoted only on architectures/shapes with a repeatable win; otherwise close that sub-item.

## Notes

Execution order: ninth. First action is a census, not implementation. Expected priority is below QFP35/QFP36 and probably QFP37 unless verify profiling shows a large ordinary MMVQ/MMVF hotspot.

## Change Log

- 2026-10-07T00:39:48.982783+00:00 (created-by): Created by agent
- 2026-10-07: grounded at b11402, rejected 1301 and evaluated 1273; closed the duplicate MMVQ-threshold idea for current verify widths and split remaining launch-geometry candidates by exact kernel owner.
