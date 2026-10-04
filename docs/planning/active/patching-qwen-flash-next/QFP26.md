---
id: QFP26
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-05T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Persistent grouped MoE prefill MMQ and routed gate/up fusion

## Description

QFP17 already owns `MOE-MMQ-PP2` after the validated 1237/1265 compact-grid work. Those patches remove rectangular overlaunch by mapping only real `(expert, expert-local-tile)` work, but they retain the existing per-operation MMQ execution model and separate gate/up/GLU stages. QFP26 is the concrete implementation owner for the next prefill MoE step: build compact expert/token work once, feed persistent/grouped MMQ workgroups, then fuse the routed gate+up projection/GLU where the measured graph and quant types permit it.

The target is prefill, where hundreds/thousands of tokens produce enough routed work to amortize a histogram/scan/task-list build. Decode/MTP small-N stays on the existing MMVQ/MMQ paths unless independently proven. QFP10 owns small-N expert activation/VDR work; QFP13 owns decode launch reduction; QFP26 must not broaden into those paths.

Upstream dense FFN MMQ+GLU fusion (#29948 cited by QFP09/QFP11) is mechanism evidence only: Flash-Next uses routed `MUL_MAT_ID` with expert IDs and highly uneven token occupancy, so the implementation needs grouped expert tasks rather than a dense FFN matcher copied verbatim.

## Steps

1. Profile routed expert prefill after current validated patches at 10K/80K/200K and ub512/1024 where available. For each layer/GPU collect expert occupancy histogram, live expert count, routed rows, MMQ tile count, launch count, gate/up/down kernel wall, routing/scatter wall, Q8_1 activation conversion, LDS/VGPR/spill data and critical-rank excess. Continue only if routed MoE is >=8% of prefill critical wall or an Amdahl bound >=3% exists.
2. Reuse 1237's compact block-map preparation as the first task-list source. Extend it to emit a stable flattened task list with `{expert, token_begin/local_tile, rows, output_tile}` rather than rebuilding equivalent occupancy metadata independently.
3. Implement a grouped/persistent MMQ launch for one projection first. Launch a bounded number of workgroups (e.g. a multiple of CUs), each obtains task IDs from a static prefix assignment or global atomic counter and processes only live expert tiles. Compare persistent queue vs existing compact static grid; keep static mapping if atomics/queue contention lose.
4. Tune multi-row expert tiles for prefill occupancy buckets. Candidate routed-row tile sizes `{8,16,32,64}` and MMQ J/tile geometry must use the same allocation-width invariant as upstream #29953/#29941; scratch allocation and launch selector must agree on the exact chosen width.
5. Add routed gate+up fusion only after the grouped projection is correct. Both matrices consume the same routed activation and expert/token map. One workgroup should load/quantized-activation data once, accumulate gate and up outputs, apply the existing SWIGLU/GLU epilogue, and write the fused intermediate expected by down projection. Preserve exact model gate function/order from the current graph.
6. Avoid materializing separate gate/up outputs where possible. The fused kernel may keep both accumulators in registers and emit only `silu(gate) * up` (or the exact Qwen4Exp GLU variant). If D/register pressure spills, test split accumulation or smaller row tiles rather than forcing fusion.
7. Keep down projection separate initially. A later `gate+up+down` mega-kernel is out of scope unless profiling shows the fused intermediate is itself the dominant traffic and resource pressure is acceptable.
8. Ensure tensor-split rank semantics remain unchanged. Each rank computes only its local expert-weight shard/rows as today; output partials enter the existing local routed/shared join and AllReduce. QFP26 must not add expert-parallel communication or change expert placement.
9. Add architecture/type dispatch for the actual production routed expert quant types (currently expected IQ3_S/IQ4_NL mix, verify from GGUF). Implement only confirmed types first. gfx1100/gfx1201 can select different tile/workgroup geometry through existing architecture tuning tables; no model-specific constants in code (QFP23).
10. Validate combined with 1237/1265; if QFP26 subsumes their block-map kernel, consolidate rather than run two prep passes. Run QFP17 E2E acceptance after each stage: persistent scheduler alone, gate/up fusion alone if separable, then combined.

## Detailed Solution & Technical Design

### Work construction

The logical routing input is token x top-k expert IDs. Build expert occupancy and prefix offsets on GPU:

```text
ids[token,k]
   |
histogram[expert]
   |
exclusive_scan -> expert_offsets
   |
scatter token ids/route slots grouped by expert
   |
build MMQ tasks for only nonempty expert row/tile ranges
```

Do not automatically add all four passes if the current `MUL_MAT_ID`/1237 machinery already exposes grouped indices and counts. Inspect and reuse `ids`, `ids_src`, row maps and 1237's `mmq_build_moe_block_map`. The best implementation may only need to enrich the existing block map with row counts and a queue order.

A task descriptor can be compact:

```cpp
struct bc_moe_task {
    uint16_t expert;
    uint16_t row0;
    uint16_t n_rows;
    uint16_t tile_j;
};
```

Use 32-bit fields if dimensions exceed bounds; do not truncate for convenience. Sort tasks by expert then contiguous output tile to maximize weight locality. A second experiment may interleave large experts to balance workgroups, but deterministic task order is preferable until imbalance is demonstrated.

### Persistent scheduler

Static compact grid (1237) already removes empty blocks. Persistent scheduling is only useful if expert occupancy leaves significant per-block duration variance/critical-rank tail. Candidate kernel structure:

```cpp
__global__ void mul_mat_q_moe_persistent(..., const bc_moe_task * tasks, int n_tasks, uint32_t * next) {
    for (;;) {
        uint32_t t = atomicAdd(next, 1);
        if (t >= n_tasks) return;
        run_one_expert_tile(tasks[t], ...);
    }
}
```

Benchmark against static assignment `t = blockIdx.x; t += gridDim.x`. A global atomic per task may be too expensive for short tasks; chunk claims (`atomicAdd(next, 4/8)`) or host/static task partitions can reduce contention. The objective is lower total kernel tail and fewer launches, not ideological persistence.

### Gate/up fused accumulation

The routed gate/up projections share:
- the same selected expert;
- the same activation row(s);
- the same quantized activation/Q8_1 source where the existing MMQ path uses one;
- compatible output row/tile geometry.

Design the fused body around one activation load and two weight streams:

```cpp
acc_gate = dot_q(w_gate[expert,...], x);
acc_up   = dot_q(w_up  [expert,...], x);
float y = silu(acc_gate) * acc_up; // replace with exact existing GLU math
store(fused_intermediate, y);
```

For IQ weights, reuse their existing vec-dot/dequant traits; do not convert weights to f16/f32 temporaries. If gate/up quant types differ in a layer, either instantiate a mixed-type template deliberately or fall back; do not reinterpret blocks.

### Activation conversion

QFP10/1307-1312 show Q8_1 activation materialization can be a meaningful small-N tax. For prefill MMQ the conversion may already be amortized. Measure first. If the grouped gate/up pair currently quantizes the same source twice or creates separate Q8_1 temporaries, share one conversion/task input; otherwise leave it alone. Direct-F32 IQ MMVDQ belongs to QFP10 unless prefill profiling proves it is also the right MMQ path.

### Task reuse across gate/up/down

Routing IDs are identical across routed gate/up/down in a layer. Preserve one grouped token/expert order across the three projections so routing/scatter metadata can be reused. Down consumes the fused intermediate and same expert identity; a later optimization may reuse the task list with different row geometry without rebuilding histogram/scan.

### Determinism

Persistent atomic task acquisition changes block execution order but each output tile must have a single writer or deterministic local reduction. Never use atomics to accumulate multiple expert contributions into the same final token output if the current path uses a fixed-order sum; keep route-slot outputs separate and retain the existing reduction/scatter stage unless a deterministic fused reduction is separately proven.

## Code Samples & Guidance

Likely code ownership from 1237/1265:

- `ggml/src/ggml-cuda/mmq.cu` / `mmq.cuh`: MMQ tile body, launch geometry and MoE block map.
- `ggml/src/ggml-cuda/mmq.cuh`/quant traits for IQ/K quant dot products at the current pin; inspect exact source before adding type templates.
- `ggml/src/ggml-cuda/ggml-cuda.cu` or current MUL_MAT_ID dispatcher: gate grouped/persistent prefill path by rows/tokens/type/arch.
- Qwen4Exp graph only if gate/up topology needs one explicit fused op; prefer existing `GGML_OP_GLU`/fusion plumbing where it can represent routed IDs.
- 1237/1265 patch packages: extend/consolidate their block-map mechanism rather than duplicate it.

Required trace per kernel/layer:

```text
BIGCHERRY_MOE_PP layer=... tokens=1024 live_experts=... tasks=... max_rows=... p50_rows=... sched=static|persistent gateup=fused|split wall_us=...
```

Record task count and occupancy distribution so a gain can be explained by reduced tail/traffic instead of unrelated drift.

## Files

1237/1265 MoE MMQ patch sources; CUDA/HIP MMQ and MUL_MAT_ID dispatcher; quant vec-dot traits for confirmed routed types; QFP17 profiling scripts; direct backend/MoE tests; runtime-profile selector values under QFP23 after tuning.

## Validation

Direct op fixtures: random routing with expert occupancy patterns `{uniform, one-hot/skewed, many empty, long-tail}`, top-k matching model, token counts `{64,256,512,1024,2048}`, tail tiles and multiple quant types. Compare static current path vs grouped/persistent outputs and route-slot ordering.

Hardware: gfx1100/gfx1201 standalone routed MMQ and model ABBA at 10K/80K/200K. Record per-layer MoE wall, critical-rank excess, launch count, HBM bytes if available, VGPR/LDS/spills, prefill t/s and peak workspace. Test outer ubatch 512 and 1024 when QSA memory permits.

Correctness: CPU/reference or current GPU path logits through representative layers; model greedy/KLD contract. Decode/MTP controls must remain on existing path and change <=1%.

## Effort & Risk

L/high. Persistent scheduling alone is moderate; routed gate/up fusion across quant types and expert ID mappings is substantial. Main risks are register pressure/spills, task-queue overhead, mismatched allocation/launch tile widths, route-slot ordering, and optimizing summed kernel time rather than the critical rank.

## Standards

QFP17 acceptance umbrella; QFP26 implementation owner; reuse 1237 block-map/routing metadata; one allocation/launch tile-width decision; no expert-parallel semantics change; confirmed quant types only; deterministic output ownership; runtime thresholds via QFP23 profiles.

## Acceptance Criteria

- Profiling establishes >=3% E2E Amdahl bound before fusion work proceeds.
- Grouped/persistent stage reduces routed MoE critical-path wall >=10% or kernel-tail p90 >=20% versus 1237/1265 static compact grid on a representative prefill lane.
- Gate/up fusion, if promoted, removes separate gate/up materialization/launches and improves routed MoE wall a further >=10% without spills that erase E2E gain.
- Combined path improves Flash-Next prefill >=3% at a representative lane with no decode/MTP regression >1% and accepted output contract.
- If persistent scheduling is neutral but multi-row/static tuning wins, keep the simpler static mechanism and record persistent scheduling as rejected rather than forcing it.

## Notes

QFP17 step 6 already names tokens-per-expert tile sizing/multi-row expert tiles; QFP26 deepens that work and adds persistent scheduling/fused routed gate+up as the concrete implementation owner. Upstream dense #29948 is useful for epilogue/fusion structure but not a drop-in routed-MoE patch.

## Change Log

- 2026-10-05T00:00:00+00:00 (created-by): Created by agent as implementation owner for QFP17 MOE-MMQ-PP2.
