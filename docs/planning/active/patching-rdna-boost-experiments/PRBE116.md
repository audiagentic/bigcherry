---
id: PRBE116
order: 116
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-10-20'
created-by: agent
priority: P2
work: M
---

# HIP Q3_K MMVQ VDR2 packed-scale integer dot — source/route admission

## Decision (2026-10-20 UTC; BCOP340)

**PENDING / NO PATCH OR GPU QUEUE.** A distinct, dormant Q3_K dense MMVQ kernel capability has new upstream implementation and measured external AMD-adjacent evidence, but **no first-party BigCherry executed Q3_K critical-path receipt**. First establish the exact route and exposed cost; do not port on the strength of NVIDIA/gfx1151 numbers. This item alone owns Q3_K VDR2/packed-scale *dot-body* qualification. Existing 0600/0650 owns MMVQ geometry and its only forced-instance bridge. QFP10 owns MoE Q4_K, PRBE111/1273 owns IQ, PRBE47 Q6_K reductions, QFP44/0300 owns MMQ; none is a second Q3_K owner.

**Eligibility/novelty:** On inspected BigCherry branches, 2026-10-20 06:09–18:09 UTC, independent RAD14 FP8/W4 GEMV, RAD06 Flash-Next A/B/shape harness, RCS05/06 worker changes, 0300 MMQ patch-test repair, and Radiance 1.9.0 bump/queued GPU qualification are protected. No Q3_K MMVQ dot-body commit, PR, plan change, patch, benchmark or queued lane was found. Upstream #30517 merged 2026-10-17 13:35 UTC, outside 12 hours. Distinct from recent Q4_K VDR2/VDR4, Q6_K, IQ MMVQ and MMQ compact-grid BCOP audits. Off-repository hardware queues cannot be certified.

## Source/dispatch trace (not measured BigCherry performance)

- Pinned llama.cpp **b11552** (`23b0202a189c44a54625aadcb37a946dd1d6278d`), `ggml/src/ggml-cuda/vecdotq.cuh`: `VDR_Q3_K_Q8_1_MMVQ=1`; `vec_dot_q3_K_q8_1` loads Q3 `qs`/`hmask`, four Q8_1 words and per-block `d8`; `vec_dot_q3_K_q8_1_impl_mmvq` extracts 6-bit scales bytewise, forms signed 3-bit values with `__vsub4`, performs one `ggml_cuda_dp4a` per 4-byte group, accumulates F32, multiplies by Q3 block scale. `ggml-common.h`: QK_K=256, QR3_K=4, QI3_K=16.
- `ggml/src/ggml-cuda/mmvq.cu`: `get_vec_dot_q_cuda(GGML_TYPE_Q3_K)` selects that helper; `get_vdr_mmvq` supplies the compile-time vector ratio; `mul_mat_vec_q` computes `blocks_per_iter=vdr*nwarps*physical_warp_size/qi`, K traversal, per-output reduction and optional fusion; `calc_nwarps`/`calc_rows_per_block` and `calc_launch_params` must agree. The host `ggml_cuda_mul_mat_vec_q` path quantizes source activations to Q8_1 before typed MMVQ launch. Do not assume that all `MUL_MAT_ID` widths take this dense path: a dedicated MoE multi-token kernel and MMQ crossover exist.
- BigCherry `0600_mmvq_geometry/patch.py` adds defaulted geometry template arguments; `0650_mmvq_native_variant/patch.py` routes measured compiled variants through the *existing* launch call chain. Any upstream port must compose with both and retain native/default fallback; no new dispatcher, selector table, config or allocator.
- Merged upstream [llama.cpp #30517](https://github.com/ggml-org/llama.cpp/pull/30517), commit [2744e48d](https://github.com/ggml-org/llama.cpp/commit/2744e48d1e88d7416224e92bf58d4ffedda4a205), changes Q3_K VDR1→2, packs 12 scale bytes into three 32-bit registers, replaces bytewise signed `__vsub4` with a positive packed-dot plus signed -4 `dp4a`, and accumulates two adjacent 4-byte groups before each F32 scale. It also changes low-bit rows/CTA in the **GENERIC** parameter table, not automatically the gfx1100/gfx1201 RDNA tables. Do not transplant its Q2_K/Q6_K/GCN/GB10 geometry changes into this slice.
- This is **integer-algebra equivalent** per group, not necessarily F32 bit-identical after regrouping. VDR2 halves K-loop iterations at unchanged logical K and required weight/activation bytes, while increasing live packed registers and dp4a issue count. Any claim of reduced physical traffic, higher occupancy or speedup requires ISA/rocprof evidence. No P2P benefit is implied on the 2×gfx1100 + gfx1201 target; gfx1030 is a drafter/control, not a qualification rank.

## Evidence classes and cheapest discriminator

**External measured (not BigCherry):** #30517 reports gfx1151 Q3_K `test-backend-ops` kernel ratios 1.08×/1.23×/1.29×/1.37× for ne11=1/2/4/8, and Qwen3.5-4B Q3_K_S tg128 69.70→72.58 t/s (+4.1%). The 2026-09-09 gfx1200 follow-up in [llama.cpp discussion #22663](https://github.com/ggml-org/llama.cpp/discussions/22663) reports Q3_K HIP 122.21 µs vs Vulkan 88.14 µs for 4096×14336, n=1, and 58.03 vs 75.08 tg128. Those are different hardware/pins and do not prove gfx1100/gfx1201 gain or that unpack is the only bottleneck. A gfx1201 multi-node bandwidth-efficiency comparison in the same discussion is confounded by model and RPC differences.

**Host-only run (2026-10-20):** `q3k_vdr2_host_oracle.py` seeded 30517, 50,000 random packed 12-byte scale blocks and iqs∈{0,2,...,14}: 200,000/200,000 old/new signed 6-bit scales match; 400,000/400,000 signed integer dp4a groups match. Illustrative F32 regrouping differs in 72,804/200,000 cases. These are **not GPU F32 parity results**; random independent 4-byte groups deliberately probe arithmetic sensitivity. Python compilation passed. No HIP build, patch apply, GPU benchmark or first-party Q3_K receipt was run.

**Other engines:** vLLM GGUF exposes Q3_K among MMVQ/MMQ types; SGLang's GGUF kernel headers share the packed format. Neither is evidence that a CUDA/Triton/NVIDIA-specific layout outperforms the native HIP kernel on BigCherry. Prefer the already-merged ggml implementation as a code and correctness reference; wait for a pinned upgrade if it absorbs #30517 cleanly.

## Bounded next implementation/benchmark gate

1. **Route and ceiling, before writing kernel code.** On a reproducible BigCherry Q3_K_S GGUF (if available), capture `GGML_TYPE_Q3_K`, dense `MUL_MAT` vs `MUL_MAT_ID`, ncols 1/2/4/8, exact native MMVQ/MMQ dispatch, Q8_1 producer launches, per-rank kernel/stream/event intervals, model and pin hash, graph reuse, bytes and output rows. Use existing rocprof and BigCherry trace/contract tools. Include pp512/2048 and tg128, ub16/128/512, 8K/80K, no-MTP and MTP controls. No available executed route → `NO_Q3K_ROUTE`; exposed removable E2E fraction <2.9126% → `BELOW_CEILING`. No broad benchmark campaign if either fails.
2. **Only if admitted**, compose a *Q3_K-only* experimental dot-body against b11552 in the existing `vecdotq.cuh`/MMVQ launch path. Compare native, VDR2+packed-scale, and upstream-full (geometry changes isolated). Keep existing 0600/0650 forced-geometry bridge, existing Q8_1 activation pool and graph lifetime. No extra staging, duplicated dequant, persistent buffers or independent selector. If the next pin already contains #30517, benchmark stock instead of duplicating the patch.
3. **Cheap static/device proof first:** compile gfx1100 and gfx1201 plus gfx1030 fallback; inspect emitted ISA for `v_dot4_i32_i8`/dp4a, bitfield/perm instructions, VGPR/SGPR, LDS, spills/scratch, occupancy, wave32/64 launch bounds and exact K/CTA coverage. Compare all 16 Q3 scale slots, high-bit polarity, signed extremes, odd row tails, widths and fused/nonfused arms; run native-vs-subject `test-backend-ops` full output, randomized packed fixtures, repeated same-process requests, multi-ubatch/graph realloc and deterministic greedy/full-vocab logits/KLD/MTP acceptance. F32 changes require numerical-contract review; no missing work or transfer bytes may count as a gain.
4. **Performance only after correctness:** paired same-pin, same-model, same-process ABBA across 4 independent sessions/architecture, ≥10 rounds each, first-party gfx1100 and gfx1201; measure per-kernel and exposed per-rank E2E, pp/decode, MTP on/off, physical bytes, rank skew, clock/thermal variance. Promotion requires CI95-low ≥3% E2E incremental throughput, controls ≤1% regression, full correctness and physical-work parity. Otherwise `NUMERIC_FAIL`, `SPILL_REGRESSION`, `NO_GAIN` or `WAIT_UPSTREAM`. Preserve native fallback/default-off until qualified.

## Dependencies / reconciliation

- `0600/0650`: existing compiled geometry and native launch ownership; do not change during this planning audit.
- `PRBE111/1273`: IQ MMVQ only; do not extend IQ-specific VDR or nwarps to Q3_K.
- `QFP10/BCOP339`: Q4_K MoE VDR4; separate type, route and numeric contract.
- `QFP44/0300`: MMQ occupancy/forced-J; do not touch actively repaired 0300 path or run competing MMQ lanes.
- `RAD14/RAD06/RCS05/06`, Radiance 1.9.0 bump: protected active independent work, no shared edits or hardware queues.

Source references: [pinned vecdotq.cuh](https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278/ggml/src/ggml-cuda/vecdotq.cuh), [pinned mmvq.cu](https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278/ggml/src/ggml-cuda/mmvq.cu), [#30517 diff](https://github.com/ggml-org/llama.cpp/pull/30517/files), [gfx1200/1201 Q3 discussion](https://github.com/ggml-org/llama.cpp/discussions/22663).
