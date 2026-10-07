---
id: QFP36
order: 36
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:39:40.654749+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# MoE router: split-K router GEMM and multi-warp routing helper

## Description

External report: (a) a small F32 MoE router GEMM split across eight K partitions plus a deterministic reduction; (b) a 16-warp routing helper replacing the single-warp token scan. The reported timings came from a different 4-GPU build and are hypotheses only.

Both target mechanisms are identifiable at b11402:

- Qwen4Exp's `ffn_gate_inp` is a 2D `[n_embd, n_expert]` router weight and `build_moe_ffn()` computes router logits with an ordinary MUL_MAT before top-k. The production model has 512 experts.
- `ggml_cuda_launch_mm_ids_helper()` launches **one physical warp per expert**, scans all tokens/routes for that expert, stores matches in shared memory sized by `n_tokens`, then writes compact row maps and expert bounds.
- BigCherry 1281 translates range-MUL_MAT_ID global ids to local ids (or `INT_MAX` for a nonlocal route) before calling that same helper. A replacement helper must preserve this exact contract.

Gate each optimization separately. The router GEMM changes F32 accumulation order and is higher numerical risk; the helper can and should be output-identical.

## What we already have

### Router at b11402

- `src/models/qwen4exp.cpp::llama_model_qwen4exp::load_arch_tensors`
  - creates `layer.ffn_gate_inp` as `{n_embd, n_expert}`.
- `src/models/qwen4exp.cpp::graph::build_layer_ffn`
  - passes that tensor to generic `build_moe_ffn(..., n_expert, n_expert_used, ...)`.
- `src/llama-graph.cpp::llm_graph_context::build_moe_ffn`
  - owns the generic router-logit MUL_MAT, top-k ids/weights and routed expert graph.
- `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_mul_mat`
  - routes an ordinary F32 router matmul through MMVF/MMF/rocBLAS according to width/shape; there is no router-specific split-K path.
- The logical router output is `n_expert x n_tokens`; with Meta tensor split, the CUDA backend must use the **physical local tensor dimensions** and leave Meta's existing split/reduction semantics unchanged.

The external “512x512” label must not be hard-coded: 512 experts is true for our production model, but `n_embd`, local split dimensions and token width must come from tensors.

### Routing helper at b11402

- `ggml/src/ggml-cuda/mmid.cu::mm_ids_helper<n_expert_used_template>`
  - one block per expert;
  - `__launch_bounds__(physical_warp_size, 1)`;
  - one physical warp per block;
  - specialized cases for top-k 2/4/6/8/10/16/32;
  - top-10 pads route slots to 16 and packs multiple token slots into a warp;
  - scans all `n_tokens` for every expert;
  - tracks `nex_prev` = number of route assignments to lower-numbered experts;
  - stores matched `(token, route_slot)` in shared memory in token order;
  - writes `ids_dst`, forward/inverse `ids_src1`, and `expert_bounds`.
- `launch_mm_ids_helper`
  - allocates `n_tokens*sizeof(mm_ids_helper_store)` dynamic shared memory per expert block;
  - asserts that this fits the device shared-memory-per-block limit.
- `ggml_cuda_launch_mm_ids_helper`
  - is the public entry point used by MMQ.

### BigCherry overlap

- `patches/1281_moe_mul_mat_id_range`
  - adds `ggml_cuda_mm_ids_range_translate()`;
  - range ids become local expert indices or `INT_MAX`;
  - then the unchanged public helper groups only valid local experts;
  - QFP30 additionally uses the helper's inverse map for broadcast activation dedup with `-1` preinitialization for nonlocal routes.
- `patches/1283_qwen4exp_expert_parallel` can reduce the local expert count per device, so helper qualification must include both upstream row-split and whole-expert range modes.
- QFP34 targets thin ordinary F32 MUL_MAT generally. Router split-K must be separately tagged/qualified so these two experiments do not accidentally intercept one another.

Neither split-K routing nor a multi-warp ids helper is currently implemented.

## Steps

1. Gate 0: profile ub512 prefill and record router MUL_MAT + `mm_ids_helper` call count/time per layer and per physical device in row-split and, if enabled, expert-parallel mode.
2. Qualify the helper first because it is exact and has a clean API boundary.
3. Qualify router split-K separately using an explicit router marker; do not dispatch by device ordinal or by “512” alone.
4. Run helper adversarial identity tests; run router score/top-k probes.
5. ABBA each change independently before a combined run.

## Detailed Solution & Technical Design

### A. Multi-warp ids helper

New mechanism:

`patches/1348_moe_ids_multiwarp`

Flag:

`BIGCHERRY_MOE_IDS_MULTIWARP=0|1`, default `0`.

Keep the signature of `ggml_cuda_launch_mm_ids_helper()` unchanged so ordinary MMQ and 1281 range MMQ share the implementation.

For the production top-10 case, add a specialized block with e.g. 16 physical warps per expert. Do not assume 16 is universally optimal; make the compiled candidate explicit and benchmark gfx1100/gfx1201/gfx1030 separately.

#### Two-pass stable algorithm

The current helper's observable ordering is expert-major, then ascending token order within each expert, with the original route-slot index attached. Preserve that ordering exactly.

Pass 1, inside one expert block:

1. Divide `[0,n_tokens)` into contiguous token ranges, one per warp. Contiguous ranges are important for stable global order.
2. Within each warp use the existing top-k padded-slot technique (top-10 -> 16 slots) to test whether each token uses this expert.
3. Accumulate per-warp:
   - number of tokens matching this expert;
   - number of route assignments whose expert id is lower than this expert (`nex_prev` contribution).
4. Store only O(number_of_warps) counts in shared memory.
5. Warp 0 computes exclusive prefixes of match counts and the block-wide `nex_prev`.

Pass 2:

1. Rescan the same contiguous token range.
2. Compute each matching token's stable rank inside its warp.
3. Write at `nex_prev + warp_prefix + local_rank`:
   - `ids_dst`;
   - forward `ids_src1`, or inverse `ids_src1` when `write_inverse=true`.
4. One thread writes `expert_bounds[expert]`; the final expert also writes `expert_bounds[n_experts]`.

The second read pass is intentional: it removes the `n_tokens`-sized shared store and exposes enough parallelism without introducing a global temporary or atomics.

#### Range compatibility

1281's `INT_MAX` sentinel must retain current behavior:

- never equals a local expert;
- contributes false to `expert_used < expert`;
- generates no compact row.
- when inverse mode is requested, nonlocal slots remain at the caller's preinitialized `-1`.

Do not move range translation into the helper. Keeping translation separate preserves 1281 ownership and lets the same helper serve ordinary/range ids.

Fallback to the current one-warp helper for unsupported top-k, very small token counts where extra warps lose, or resource-limit failures.

### B. Split-K router GEMM

New mechanism, same QFP but separate qualification package to keep bisectability:

`patches/1349_moe_router_splitk`

Flag:

`BIGCHERRY_MOE_ROUTER_SPLITK=0|1`, default `0`.

#### Explicit router marking

Do not identify the router by tensor name or exact dimensions. In `src/llama-graph.cpp::build_moe_ffn`, mark only the ordinary MUL_MAT that produces router logits using an unused BigCherry op-param marker. The marker is copied with the graph node through Meta transformation.

The CUDA dispatch checks:

- marker present;
- `GGML_OP_MUL_MAT`;
- src0/src1/dst F32;
- compatible contiguous/aligned layout;
- AMD HIP backend;
- dimensions inside the qualified range;
- no conflicting fusion;
- flag on.

All other F32 MUL_MATs retain normal dispatch/QFP34 behavior.

#### Split-K layout

For a physical local router matrix:

- K = local/full inner dimension `src0->ne[0]`;
- M = local output expert rows `src0->ne[1]`;
- N = activation token columns.

Start with eight K partitions only because that is the external hypothesis. Require K divisibility/alignment needed by vector loads; otherwise fall back.

Use a temporary F32 workspace from the existing CUDA/HIP pool:

`[split_k, M, N]`.

Kernel 1 computes one partial dot per `(partition, output_expert, token)`. Kernel 2 reduces partitions in a fixed ascending partition order into `dst`. Avoid floating atomic accumulation: it would make run-to-run order nondeterministic and would weaken the routing test.

If profiling shows workspace or second-launch overhead dominates at smaller N, gate by measured token width rather than forcing split-K.

### Architectures

- gfx1100/gfx1201: primary router/helper prefill targets.
- gfx1030: helper/router may execute in the MTP sidecar; qualify rather than assume the same thresholds.
- Runtime CC and physical shape decide; no ROCm device ordinal is embedded.

### Meta split

Router split-K operates on the physical matrix passed to the backend and does not alter split state. If Meta has split the router output rows, only local M changes; existing Meta reduction/gather semantics remain authoritative.

For expert parallelism, the routing decision still refers to global 512 expert ids. 1283/1281 then translate/group local expert work. The helper optimization must accept the local `n_experts` it is passed and must not assume 512 experts per device.

No collective behavior changes.

### Fusion

The router marker applies only to the unfused router MUL_MAT; if a future fusion consumes it, split-K must fail closed until that fusion explicitly carries the contract. The helper sits inside MMQ and composes with current gate/up GLU fusion and QFP30 range dedup because its output buffers are unchanged.

## Code Samples & Guidance

Helper anchors:

- `ggml/src/ggml-cuda/mmid.cu::mm_ids_helper`;
- `launch_mm_ids_helper`;
- `ggml_cuda_launch_mm_ids_helper`;
- preserve `ggml/src/ggml-cuda/mmid.cuh` public signature if possible.

Router marker anchor:

- `src/llama-graph.cpp::llm_graph_context::build_moe_ffn` at router logits construction from `gate_inp`.

Router dispatch anchors:

- `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_mul_mat`;
- new implementation in a dedicated CUDA/HIP compilation unit/cuh if the patch framework can add it cleanly; do not bury split-K inside generic rocBLAS code.

Suggested activation markers:

- `BIGCHERRY_PATCH_HIT 1348 moe_ids_multiwarp experts=<...> tokens=<...> used=<...> inverse=<0|1>`;
- `BIGCHERRY_PATCH_HIT 1349 moe_router_splitk m=<...> n=<...> k=<...> split=8`.

Emit once per representative configuration, not per layer.

## Files

### Helper

Create:

- `patches/1348_moe_ids_multiwarp/{patch.py,README.md,SUMMARY.md}`;
- `tools/tests/patch/test_1348_moe_ids_multiwarp.py`.

Production targets:

- `ggml/src/ggml-cuda/mmid.cu`;
- `mmid.cuh` only if a private helper declaration is required.

Do not modify 1281 if the public helper contract remains unchanged.

### Router

Create:

- `patches/1349_moe_router_splitk/{patch.py,README.md,SUMMARY.md}`;
- `tools/tests/patch/test_1349_moe_router_splitk.py`.

Production targets:

- `src/llama-graph.cpp` for the router marker;
- `ggml/src/ggml-cuda/ggml-cuda.cu`;
- dedicated CUDA/HIP source/header and build-list edit if required.

## Validation

### Helper mechanics/identity

- Patch-lint and exact/idempotent mechanics.
- Standalone old-vs-new helper test with random/adversarial ids:
  - token counts 1, 2, 31, 32, 33, 128, 512;
  - top-k 2/4/6/8/10/16/32 plus generic;
  - duplicate/unsorted expert selections where the API permits them;
  - expert ids at 0 and `n_experts-1`;
  - `INT_MAX` sentinel patterns matching 1281 range translation;
  - forward and inverse modes.
- Require byte-identical `ids_src1`, `ids_dst`, `expert_bounds`.
- Range QFP30 test must still pass with identical active/inactive mapping.

### Router mechanics/equivalence

- Compare split-K scores against current F32 router over random/production-shaped inputs.
- Because K reduction order changes, use explicit F32 tolerance rather than bit identity for scores.
- Separately require **top-k ids unchanged** on captured production probes and adversarial near-tie probes. If near ties can change selected experts under normal inputs, reject rather than calling that harmless floating-point noise.
- End-to-end greedy target output must be identical for promotion.

### Activation/timing

- Kernel census shows one-warp helper replaced on qualified shapes and lower total helper GPU time.
- Router marker proves only router GEMMs use split-K; rocBLAS/generic F32 census changes only for those nodes.
- Record workspace peak.

### Hardware ABBA

Use `tools/lab/flash-next/queue-env-ab.sh`, full process separation, ub512.

1. helper A/B with router split-K off;
2. router split-K A/B with helper fixed;
3. combined B only after isolated wins.

Run pp4096 and a 24K production prefill; add ~98K only if the per-layer overhead remains material. Record prefill t/s, helper/router GPU time, VRAM peak, router top-k probes and greedy identity.

No multi-session contract campaign.

## Effort & Risk

Effort: M for helper, M/H for split-K depending on workspace/kernel integration.

Risk:
- helper: medium implementation risk, low numerical risk;
- router split-K: medium/high numerical/performance risk because expert selection is discontinuous near score ties.

Expected gain on our topology: helper medium-confidence for ub512 because the current one-warp/all-token implementation is visibly serial; router split-K is lower-confidence until its per-layer aggregate is measured. The slow R9700 PCIe link is not directly relevant because neither mechanism adds host/P2P traffic.

## Standards

- Exact helper ordering and maps.
- No atomics in router score reduction.
- No hard-coded 512-expert/device assumption.
- No device-ordinal/topology assumption.
- No q4 KV.
- No legacy shim.
- Default off during qualification.
- Preserve 1281 range sentinel/inverse-map semantics.
- External timings are hypotheses.

## Acceptance Criteria

- Gate 0 attributes material time to the router/helper on our ub512 workload.
- Multi-warp helper is byte-identical to the existing helper, including 1281 range/inverse cases.
- Router split-K affects only explicitly tagged router MUL_MAT nodes.
- Router score error stays within stated tolerance and production top-k ids are unchanged.
- Greedy target output is identical.
- Fully separated ABBA shows a repeatable prefill benefit without VRAM or decode regression.
- If either submechanism is not material, close it independently.

## Notes

Execution order: seventh. Within QFP36, qualify the exact multi-warp helper before the numerically riskier router split-K. In final gain/effort ranking, the helper can move ahead of QFP34 if Gate 0 confirms ~layer-per-layer serialization at ub512.

## Change Log

- 2026-10-07T00:39:40.654749+00:00 (created-by): Created by agent
- 2026-10-07: grounded at b11402 plus 1281/QFP30; documented stable two-pass multi-warp grouping and explicit-marker deterministic split-K router design.
