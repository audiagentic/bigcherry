---
id: QFP37
order: 37
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:39:44.284926+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# MMQ work distribution: stream-k for few-tile matmuls and expert tile shape for our quant

## Description

External report: (a) Stream-K for a few-tile Q8_0 projection spread K work over the device instead of leaving most CUs idle; (b) a smaller output-row tile improved a small Q4_K expert slice. Those measurements and exact shapes came from different hardware and are hypotheses only.

b11402 already contains a full MMQ Stream-K kernel/fixup path, but **all Q8_0 and IQ4_XS entries in the RDNA2, RDNA3 and RDNA4 config tables set `stream_k=false`**. BigCherry 1237/1265 already solve a different MoE under-utilization problem by compacting the non-Stream-K expert grid. Therefore QFP37 should not invent another MMQ algorithm: first expose the existing Stream-K decomposition for measured ordinary few-tile shapes, and separately qualify a smaller IQ4_XS output-row tile for MoE.

Keep the two mechanisms independently switchable and independently packaged.

## What we already have

### b11402 MMQ

- `ggml/src/ggml-cuda/mmq.cuh::ggml_cuda_mmq_config` contains:
  - `I`: output-row tile in `src0->ne[1]/dst->ne[0]`;
  - `J`: activation/output-column tile;
  - `K_vram`;
  - `stream_k`;
  - occupancy/thread/layout data.
- `launch_mul_mat_q<type,J,fallback>()`
  - ordinary mode launches the rectangular `nty * ntx * ntzw` tiling;
  - Stream-K mode already computes total destination tiles, normally launches `nsm` blocks on AMD, allocates fixup workspace when tile/K partitions do not line up, and launches `mul_mat_q_stream_k_fixup`.
- `mul_mat_q` has a compile-time Stream-K branch through `ggml_cuda_mmq_get_stream_k(...)`.
- `mul_mat_q_switch_J` selects J by minimizing the number of J tiles among supported configs.
- `mmq-config-rdna2.cuh`:
  - Q8_0 and IQ4_XS use I=128;
  - every such entry has `stream_k=false`.
- `mmq-config-rdna3.cuh`:
  - Q8_0 uses I=64 for common J and I=128 for wide J;
  - IQ4_XS likewise uses I=64/128;
  - every entry has `stream_k=false`.
- `mmq-config-rdna4.cuh`:
  - Q8_0 and IQ4_XS use I=64 for J<=64 and I=128 for wider J;
  - every entry has `stream_k=false`.
- `ggml/src/ggml-cuda/mmq.cu::ggml_cuda_should_use_mmq` already admits Q8_0 and IQ4_XS on our AMD architectures.

Thus the Stream-K arithmetic/temporary/fixup mechanism is upstream code that is currently unreachable for these RDNA configs. The output-row geometry is compile-time configuration, so a runtime experiment needs separately compiled I variants rather than mutating a constexpr table from an environment variable.

### BigCherry overlap

- `patches/1237_rd30_moe_mmq_compact_grid` is validated on gfx1100.
  - It compacts **non-Stream-K** MoE MMQ from a worst-case rectangular expert grid to the actual expert/J-tile list.
  - Its implementation explicitly leaves Stream-K untouched.
  - Hardware evidence measured about +7.3-7.4% MoE prefill with byte-identical output.
- `patches/1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2` extends that compact path to gfx1201 and gfx1030.
- `patches/1281_moe_mul_mat_id_range` and `1283_qwen4exp_expert_parallel` alter local expert ownership/grouping but still feed ordinary MMQ tensor descriptors/maps.
- QFP30 inside 1281 changes Q8_1 activation dedup/scatter for range MMQ; QFP37 must not alter its ids/inverse-map contracts.

Finding: MoE launch compaction is already covered and should remain the production baseline. Stream-K for ordinary few-tile Q8_0/IQ4_XS and the IQ4_XS I-tile sweep are **not** covered.

## Steps

1. Gate 0: census all MMQ calls on Flash-Next and the 27B Q8_0 model by physical device, type, `nrows_x/ncols_x/ncols_max`, chosen I/J, destination tile count, CU count, MoE/non-MoE, call count and aggregate GPU time.
2. Identify ordinary non-MoE calls whose destination tile count is materially below device CU count and whose K dimension is large enough to split.
3. Qualify existing Stream-K only for that measured class; leave MoE on 1237/1265 compact tiling in v1.
4. Separately sweep an I=32 IQ4_XS MoE specialization against the existing I=64/128 choices using actual physical expert-row sizes on gfx1100/gfx1201.
5. Only test gfx1030 if the MTP census shows the same candidate class.
6. ABBA each mechanism alone before combination.

## Detailed Solution & Technical Design

### A. Existing Stream-K for ordinary few-tile MMQ

New mechanism/package:

`patches/1350_mmq_few_tile_streamk`

Flag:

`BIGCHERRY_MMQ_FEW_TILE_STREAMK=0|1`, default `0`.

Do not edit the RDNA config tables to make Stream-K globally true. They are compile-time selectors and a global change would affect every Q8_0/IQ4_XS shape.

Instead compile a second specialization of the existing MMQ path with a template-level `force_stream_k` boolean. The specialization must use the same upstream config for I/J/thread count/layout/K_vram and only replace the compile-time Stream-K decision. Both the device branch in `mul_mat_q` and the host launch/fixup decision must see the same compile-time value.

Initial runtime eligibility should be deliberately narrow:

- AMD RDNA;
- ordinary MMQ only: `ids_dst == nullptr`;
- type Q8_0 first; add IQ4_XS only if Gate 0 shows an ordinary IQ4_XS candidate;
- selected upstream I/J config is valid;
- destination tile count `ntiles_dst < nsm`;
- K has enough MMQ work units to distribute over more blocks;
- fixup workspace size is bounded and allocation succeeds;
- no native path/fusion contract changes.

The external example's exact matrix dimensions must not appear as a dispatch rule.

On AMD the existing Stream-K host path already chooses `nsm` workgroups, partitions the flattened tile/K work and invokes the existing deterministic fixup when needed. Reuse that mechanism; do not add atomics or another reduction kernel.

For very small K or when normal tiling already fills the device, use the current path.

#### Why v1 excludes MoE

1237/1265 already remove empty expert/J tiles and have production evidence. Stream-K's current `ntiles_dst` is based on rectangular logical tiling, while real MoE occupancy is encoded in `expert_bounds`. Enabling Stream-K for MoE without a measured real-tile heuristic could undo the compact-grid gain or over-distribute sparse work.

If a later census proves a remaining MoE under-occupancy case after 1237, extend the plan with a real expert-bound tile count; do not silently broaden v1.

### B. IQ4_XS expert output-row tile sweep

New mechanism/package:

`patches/1351_iq4xs_expert_i32`

Flag:

`BIGCHERRY_IQ4XS_EXPERT_I32=0|1`, default `0`.

This experiment is for `MUL_MAT_ID`/MoE only:

- `type == GGML_TYPE_IQ4_XS`;
- `ids_dst != nullptr`;
- AMD RDNA;
- non-Stream-K path;
- measured physical `nrows_x` range where current I=64/128 leaves too few output-row blocks or excessive tail waste.

Do **not** hard-code “160 rows”; use the physical tensor descriptor after Meta splitting/range placement. The production row-split and whole-expert modes can present different local shapes.

Because I is compile-time throughout MMQ loading/writeback, add an I=32 specialization rather than rewriting `ggml_cuda_mmq_get_config()` at runtime. The specialization retains the selected J, SRAM layout, K_vram, quantized loads, vector dot and output arithmetic; only output-row workgroup geometry changes.

The host chooses current-config I or I=32 after seeing the actual local `nrows_x`. Gate 0 should determine the useful threshold. Candidate sweep should include at least current I and 32; add another I only if profiling justifies it.

### 1237/1265 interaction

I=32 remains a non-Stream-K MoE launch, so the validated compact expert/J block map remains active. 1237 compacts the J/expert enumeration; I controls the independent output-row dimension. Workspace sizing in 1237 must be checked against the selected specialization, but its map format and expert ordering must not change.

The ordinary Stream-K package never enters the compact MoE path in v1, so the two ownership domains are disjoint.

### 1281/1283 interaction

The I=32 path consumes the same `ids_dst`, `expert_bounds` and Q8_1 activation buffer as current IQ4_XS MMQ. It must therefore work unchanged with:

- ordinary row-split MUL_MAT_ID;
- 1281 range-local ids and QFP30 `-1` inverse sentinel;
- 1283 local expert counts.

No id translation, expert selection or zero-lane semantics move into QFP37.

### Architectures

- gfx1100: primary target for both Flash-Next IQ4_XS and 27B Q8_0.
- gfx1201: primary target for Flash-Next; qualify independently because CU count and current MMQ config differ.
- gfx1030: only enable after an MTP-side census and independent timing. 1265's presence is not evidence that I=32 or Stream-K is beneficial there.

Use compute capability/family and physical dimensions, never ROCm ordinal.

### Numerical behavior

I-tile change should preserve each output element's K accumulation order; require byte identity in the op test where the specialization truly changes only workgroup geometry.

Stream-K partitions K and fixup changes F32 accumulation order. Treat it as equivalence, not identity: backend-op tolerance plus greedy end-to-end identity. No floating atomics.

## Code Samples & Guidance

Primary anchors:

- `ggml/src/ggml-cuda/mmq.cuh::mul_mat_q` around the compile-time `ggml_cuda_mmq_get_stream_k(...)` branch.
- `launch_mul_mat_q` around:
  - `const ggml_cuda_mmq_config config = ...`;
  - `if (!config.stream_k)`;
  - existing `ntiles_dst`, `block_nums_stream_k`, `tmp_fixup`, and `mul_mat_q_stream_k_fixup`.
- `mul_mat_q_switch_J` for selecting the normal J first, then choosing a qualified specialization.
- `mmq-config-rdna{2,3,4}.cuh` are reference geometry only; avoid globally changing their production entries during qualification.

Suggested activation evidence:

`BIGCHERRY_PATCH_HIT 1350 mmq_streamk type=<...> I=<...> J=<...> tiles=<...> cu=<...> k=<...>`

`BIGCHERRY_PATCH_HIT 1351 iq4xs_i32 rows=<...> J=<...> experts=<...>`

Emit once per representative device/shape, not per layer.

## Files

### Stream-K

Create:

- `patches/1350_mmq_few_tile_streamk/{patch.py,README.md,SUMMARY.md}`;
- `tools/tests/patch/test_1350_mmq_few_tile_streamk.py`.

Primary production target: `ggml/src/ggml-cuda/mmq.cuh`; touch `mmq.cu` only if template instantiation/dispatch requires it.

### IQ4_XS I=32

Create:

- `patches/1351_iq4xs_expert_i32/{patch.py,README.md,SUMMARY.md}`;
- `tools/tests/patch/test_1351_iq4xs_expert_i32.py`.

Primary targets: `ggml/src/ggml-cuda/mmq.cuh` and, only as required for explicit compiled specialization selection, the relevant MMQ config/instantiation files.

Do not modify 1237/1265 unless their workspace calculation is proven to require awareness of the extra specialization; if so, that compatibility adjustment belongs in 1237 because it owns compact-map workspace.

## Validation

### Offline mechanics

For both packages:

- exact/idempotent patch mechanics;
- patch-lint;
- flag off preserves current config/dispatch.

Stream-K op sweep:

- Q8_0 representative K/M/N around measured production shapes;
- tile counts below/equal/above CU count;
- fixup-needed and no-fixup cases;
- unsupported/large-tile shapes remain normal tiling.

I=32 sweep:

- IQ4_XS MUL_MAT_ID with physical row counts around 32/64 boundaries and measured production sizes;
- routed expert occupancy empty/sparse/dense;
- ordinary and 1281 range ids where available;
- 1237 compact grid on.

### Activation/performance evidence

- Stream-K marker plus profiler grid proves qualified few-tile calls launch enough blocks to occupy the device and use the existing fixup only when required.
- I=32 marker plus profiler proves the extra output-row parallelism; compact-map activation remains present for MoE.
- Record temporary/fixup memory and occupancy/register/shared-memory metrics.

### Equivalence

- I=32: require byte-identical backend-op results if K arithmetic order is unchanged; otherwise stop and explain the unexpected order change before hardware promotion.
- Stream-K: existing F32 MMQ tolerance, no NaN/Inf drift, plus greedy target identity.
- 1281 range exact-zero semantics and QFP30 tests remain unchanged.

### Hardware ABBA

Use `tools/lab/flash-next/queue-env-ab.sh` with full process separation.

Stream-K:
- 27B Q8_0 on two XTX first;
- Flash-Next only if census identifies relevant ordinary Q8_0/IQ4_XS calls.

I=32:
- Flash-Next UD-IQ4_XS on gfx1100/gfx1201 target split;
- row-split baseline first; repeat with 1283 expert parallel if that is the candidate production mode.

Run pp4096 and a representative 24K prefill. Record per-shape MMQ time, total prefill t/s, peak workspace/VRAM and greedy identity. Decode must not regress if the dispatch can appear at small batch.

No multi-session contract campaign.

## Effort & Risk

Stream-K effort: M; risk: medium. Arithmetic implementation already exists upstream, but exposing a second compile-time specialization cleanly and choosing a real occupancy threshold requires care.

IQ4_XS I=32 effort: M; risk: medium. Geometry-only change is numerically safer, but a smaller I can increase block count, shared-memory pressure and redundant weight/activation traffic.

Expected gain on our topology:
- Q8_0 Stream-K: medium potential on the 27B two-XTX model if Gate 0 finds few-tile projections; lower relevance to the UD-IQ4_XS target unless its mixed Q8 tensors hit the same class.
- IQ4_XS I=32: low-to-medium potential for Flash-Next; 1237/1265 already captured the largest MoE grid waste, so only measured residual under-occupancy justifies it.

## Standards

- Reuse upstream Stream-K/fixup; no duplicate reduction algorithm.
- Keep 1237/1265 compact MoE grid as production baseline.
- Do not hard-code external dimensions, expert count, card count or attention split.
- Use physical post-Meta tensor dimensions for dispatch.
- No q4 KV.
- No legacy/back-compat shim.
- Default off during qualification.
- External timings are hypotheses.

## Acceptance Criteria

- Gate 0 identifies a material few-tile or IQ4_XS expert geometry class on our hardware, or the corresponding sub-item closes.
- Stream-K affects only qualified ordinary MMQ in v1 and uses the upstream fixup path.
- I=32 affects only qualified IQ4_XS MoE MMQ and preserves 1237/1265 compact-grid behavior.
- 1281 range/id-map semantics remain valid.
- Stream-K passes explicit F32 equivalence and greedy identity.
- I=32 is byte-identical if accumulation order is unchanged.
- Fully separated ABBA shows a repeatable prefill gain on the model for which the candidate exists with no decode/VRAM regression.

## Notes

Execution order: eighth. The existing 1237/1265 result lowers the expected residual gain for expert tiling. Prioritize the Stream-K half only if the Q8_0 27B census shows severe CU underfill; otherwise QFP35/QFP36 are stronger Flash-Next candidates.

## Change Log

- 2026-10-07T00:39:44.284926+00:00 (created-by): Created by agent
- 2026-10-07: grounded at b11402 and validated 1237/1265; identified existing-but-disabled RDNA Stream-K, separated it from compact MoE tiling, and defined an IQ4_XS I=32 physical-shape qualification.
