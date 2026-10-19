---
id: QFP44
order: 43
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-08T00:00:00+11:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Follow-up MMQ tile geometry and compact MoE block-map qualification

## Description

Consolidates PRBE22 (RDNA MMQ tile width/LDS), PRBE23 (MoE-aware tile selection) and PRBE24 (GPU compact block-map) beside QFP37. This is **residual work only**, not a new implementation of already validated functionality. Source target: composed llama.cpp b11474, gfx1100/gfx1201 HIP, physical post-Meta tensor shapes.

## Already covered (do not duplicate)

- `1237_rd30_moe_mmq_compact_grid` is `validated` for gfx1100; `1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2` is `validated` for gfx1201/gfx1030. Both are members of `[patch-set.validated-enhancements]`. The `mmq_build_moe_block_map` kernel builds `block_start[]/block_expert[]` from `expert_bounds` and compacts non-Stream-K expert/J work; there is no need to reimplement the map.
- `1345_moe_ids_multiwarp` is `validated`, in production, and its measured Flash-Next prefill improvement was +1.6-2.1%; this optimizes expert-ID handling, **not** the output-row I geometry.
- QFP37's `1350_mmq_few_tile_streamk` has owner-reported +1.2-1.9% prefill on 8K/24K/98K (PR #22 hardware comment); it is not present on `main` at this triage, so do not call it validated here. Its initial eligibility is ordinary non-MoE few-tile Q8_0, not a replacement for compact MoE grids.

## Remaining steps

1. Re-census `ggml/src/ggml-cuda/mmq.cuh::launch_mul_mat_q`/ `mul_mat_q_switch_J` on composed b11474 for `(gfx, quant type, physical M/N/K, nrows_x, J, I, experts, tiles, CU count)` and kernel time, with `1237/1265/1345` on and `1350` controlled separately. Include Flash-Next UD-IQ4_XS and 27B Q8_0.
2. **Tile width/LDS (PRBE22):** parse current `mmq-config-rdna{3,4}.cuh` CASE rows via `tools/bigcherry/tuning/catalog.py::enumerate_mmq/read_mmq_config_table`. CASE keys are `(type,J,fallback)`; replace the selected candidate row, never append a duplicate unreachable key. Sweep J and nthreads/occupancy/LDS for the measured shapes. Record rocprofv3 VGPR, LDS, waves/CU and kernel us; no global table swap.
3. **MoE output-row I (PRBE23):** only if compact-map profile still shows under-utilization, qualify QFP37's I=32 IQ4_XS MoE variant against native I=64/128 for physical rows, keeping the validated 1237/1265 map and 1281/1283 expert-ID semantics unchanged. Do not combine Stream-K for MoE until real expert-bound tile-count accounting exists.
4. **Block-map regression (PRBE24):** instrument the existing `mmq_build_moe_block_map` launch only in a validation build; dump `block_start[0..n_experts]` and `block_expert[0..actual_blocks)` after an opt-in sync. Compare to CPU enumeration for uniform/single-hot/Zipf/skew, empty/tiny/256-expert routings, token counts 1..4096. Independently record map kernel us, syncs (production zero additional), scratch bytes and overflow/fallback marker.
5. Preserve F32 equivalence for Stream-K (changed reduction order), byte-identical I-geometry outputs when K order is unchanged, independent test-backend-ops parity, greedy identity, and native fallback for ineligible shapes. Measure end-to-end via `tools/lab/flash-next/queue-env-ab.sh` in fully separated ABBA: 8K/24K/98K prefill, kernel time, peak VRAM, decode control and exact patch-hit shape. Promote narrowly only on reproducible per-shape gain with <=1% unaffected regressions.

## Scope and handoff

QFP37 owns Stream-K (1350) and I=32 (1351) implementation/qualification. QFP44 owns residual tile/LDS catalog sweeps and formal compact-map regression evidence; no duplicate patch packages. If 1237/1265 + 1350/1351 leave no material under-occupancy, close this item as a measured null.

## 2026-10-20: BCOP332 — gfx1201 MMA k01 spill gate

**Hold; no patch or GPU queue.** [Upstream #29536](https://github.com/ggml-org/llama.cpp/pull/29536) (open, updated 19 Oct) reports stock LLVM 23 RDNA4 MMQ spills: 5/64 IQ4_XS instantiations spilled before its change and none afterward. Its external R9700 Qwen3.8-27B Q4_K_M pp result was ~298→~1250 t/s, but the PR also fixes BF16 builtin types. Neither the gain nor the spill condition is established for BigCherry's compiler. Do not port the whole PR or infer an isolated IQ4_XS benefit.

**One codegen slice:** pinned b11552 \`ggml/src/ggml-cuda/mmq-vec-dot.cuh::ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma\`, k01 loop near line 165. Candidate: gfx1201-only \`#pragma unroll 1\` on that loop, preserving \`MMQ_TILE_NE_K=32\`, \`QI8_0=8\`, four K fragments, all MMA work, LDS loads, reductions, output writes and allocation. This shared template serves IQ4_XS, IQ4_NL, IQ3_XXS and Q8_0: all need no-regression controls. Do not alter the other two loops from #29536. Register live-range/spill reduction is a **hypothesis**; increased loop instructions or vendor-compiler regression is possible.

**Actual dispatch:** \`mmq.cu::ggml_cuda_mul_mat_q\` chooses \`J_best\` from config and shared-memory admission before Q8_1 activation padding and MoE ID scratch allocation. \`mmq-config-rdna4.cuh\` permits IQ4_XS nonfallback J16/32/48/64/80/96/112/128 and fallback J16/32/64/128. \`0300_mmq_forced_j\` owns forced-J safety; preserve it, existing compact-map and graph/stream lifetimes. No new J table, selector, allocator or telemetry.

**Cheap discriminator completed:** host-only modeled 8,192 J-selection cases (widths 1..4096, both fallback sets) matched before/after; 32/32 K elements covered once; Python compilation passed. Model assumes all listed J configs fit shared memory. No HIP compile, code-object inspection, backend-op run or GPU benchmark.

**Gates:** (0) Check last-12h independent work/queues and exact compiler/build/pin, real IQ4_XS MMQ rank, \`ncols_opt\`, J/fallback, physical M/N/K, launch duration and exposed E2E critical path; reject \`NO_ROUTE\`. (1) Compile baseline vs *one-loop* candidate with identical HIP flags; compare ISA, VGPR/SGPR, scratch bytes, LDS, wave32, occupancy, code size and K/MMA counts for J80/96/112/128 and shared-template controls; reject \`NO_SPILL\` or \`NO_CODEGEN_WIN\`. (2) Require optimistically removable exposed wall >=2.9126% of E2E, accounting for per-rank overlap and no P2P; otherwise \`BELOW_CEILING\`. (3) Only if admitted, test-backend-ops MUL_MAT/MUL_MAT_ID, tail and expert counts, graph/reallocation, multi-ubatch/same-process, logits/KLD/greedy/MTP, unchanged physical work and scratch traffic. (4) Paired 4-session >=10-round ABBA gfx1201 R9700 pp512/2048 and routed 8K/24K/98K, tg128 and gfx1100 controls; promote only CI95-low >=3% E2E with <=1% unaffected regression and no correctness failure. Otherwise close \`NUMERIC_FAIL\`, \`REGRESSION\` or \`NO_GAIN\`.

**Ownership:** QFP44 owns this compiler-specific MMQ codegen decision, separate from BCOP300 compact-grid work. \`0300\` owns J selection, \`1237/1265\` block maps, QFP37 Stream-K, PRBE110 Q6_K, PRBE111 IQ MMVQ. The [FireBurn AMD stack](https://github.com/FireBurn/llama.cpp/tree/amd-perf-stack) is not an AMD BigCherry benchmark baseline. Source: [pinned loop](https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/ggml/src/ggml-cuda/mmq-vec-dot.cuh#L143-L196), [PR diff](https://github.com/ggml-org/llama.cpp/pull/29536/files).

## Change Log

- 2026-10-08 (triage): Created by folding PRBE22/PRBE23/PRBE24; referenced validated 1237/1265/1345 and open PR #22/1350 separately.
