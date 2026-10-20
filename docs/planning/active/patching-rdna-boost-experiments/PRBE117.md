---
id: PRBE117
order: 117
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-10-18T08:06:21Z'
created-by: agent
priority: P2
work: M
---

# Q6_K HIP MMVQ: gate upstream grouped-integer-dot VDR2 after rejected RD08

## Decision and novelty — 2026-10-18 (BCOP298)

**Pending, receipt-first; no implementation, patch resurrection or hardware job queued.** Merged llama.cpp [#30517](https://github.com/ggml-org/llama.cpp/pull/30517) (2026-10-17, merge `2744e48d1e88d7416224e92bf58d4ffedda4a205`) supplies a materially **different Q6_K VDR2 inner dot** from rejected BigCherry `1204_rd08_q6k_mmvq_vdr2`: it replaces bytewise signed subtraction and separately scaled float dots with two packed integer dot products (raw Q6 and four signed -32 bytes), accumulates two adjacent chunks in **integer** registers, and applies one F32 scale per pair. It is not a requalification of the failed 1204 identity. The old candidate remains `rejected` and PRBE08 remains `deprecated`; a new, isolated patch identity would be required **only after** a real route and exposed-cost gate passes.

The current pinned BigCherry llama.cpp b11552 is `23b0202a189c44a54625aadcb37a946dd1d6278d`, where `VDR_Q6_K_Q8_1_MMVQ=1`. Upstream's GCN nwarps edits and generic two-row-per-block additions are **not** RDNA3_0/RDNA4 geometry changes. Do not port them or other Q2_K/Q3_K edits as part of this Q6_K experiment.

### Branch and 12-hour exclusion

Selected live independent branch `feat/rad06-balance-kld` at `44103dfc081c4e12b783200cba7f2cb640d7c194` (2026-10-18 08:03:55 UTC, Audumla), newer than `feat/rad06-balance-profile` (07:34:51 UTC) and `main` (07:19:03 UTC); scheduled BCOP/PRBE-only commits do not count as independent activity. Inspected selected/main/profile/profile2 commits since 2026-10-17 20:06 UTC, branch heads and open PRs: no independently advanced Q6_K VDR2, RD08, PRBE117 or Q6-specific queued lane in the visible repository. Off-repository hardware queues are not attested. **Excluded:** active RAD06 helper GEMV/long-row splitting, FP16 WMMA, AHALF, VECIO, per-layer balance, KLD/replay/profiling, associated experiments and qualification. Do not edit or queue their paths. The Q6_K grouped-integer-dot mechanism has been dormant in BigCherry since 1204's 2026-09-23 rejection; #30517's different arithmetic and new R9700 measurements justify a fresh, bounded decision.

## Production source and dispatch trace

1. `ggml/src/ggml-cuda/mmvq.cu::ggml_cuda_mul_mat_vec_q` owns the `src1_q8_1` pool allocation, `quantize_row_q8_1_cuda` producer on `ctx.stream()`, and `mul_mat_vec_q_switch_type` / `mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>` selection. The Q6_K kernel uses `get_vdr_mmvq`, `calc_nwarps`, `calc_rows_per_block` and `mul_mat_vec_q`. Do not duplicate the quantization, scratch lifetime, stream ordering, graph capture or native dispatch table.
2. `ggml/src/ggml-cuda/vecdotq.cuh::vec_dot_q6_K_q8_1` loads packed `block_q6_K::{ql,qh,scales,d}` and `block_q8_1::{qs,ds}`. `vec_dot_q6_K_q8_1_impl_mmvq` currently uses `__vsub4((vl|vh),0x20202020)`, one `ggml_cuda_dp4a` per chunk, then per-chunk F32 multiply/add. #30517 uses `VDR=2`, arrays of two low/high words, `ggml_cuda_dp4a(vl|vh,u,sumi)` plus `ggml_cuda_dp4a(0xE0E0E0E0,u,sumi)`, then scales the integer **pair sum** once. `0xE0` is signed -32; do not use unsigned dot or change Q6 scale offsets.
3. `0600_mmvq_geometry` and `0650_mmvq_native_variant` already own explicit compiled geometry and forced variants. **Validated 1274** intercepts dense, width-1 Q6_K on gfx1100 when no forced candidate is requested and uses the original F32 activation directly, bypassing Q8_1 and the Q6_K VDR dot entirely. Therefore a stock Q6_K VDR2 comparison on that production route is a **non-activation control**, not a performance experiment. gfx1201 and widths 2–8 may still use Q8_1 MMVQ. `has_ids && ncols_dst>1` uses `mul_mat_vec_q_moe_launch`, with a separate two-row geometry; exclude MoE from first admission and retain its correctness control.
4. No new buffer, synchronization barrier, LDS tile, host/device copy, graph node or kernel launch is proposed. Q6_K weight traffic and Q8_1 activation quantization remain. Only instruction selection, register pressure, per-lane work and F32 grouping may change. Verify actual compiler ISA (wave32, VGPR/SGPR, scratch/spills, occupancy, integer dot count, memory instructions, LDS/barriers), not inferred occupancy.

### Physical-work model and architecture gate

At K=4096, Q6_K uses 16 blocks of 256 elements, **210 bytes/block** (128 ql + 64 qh + 16 scales + 2 F16 d). For 14,336 output rows, logical packed weight bytes = **45.9375 MiB** per full dense matmul, unchanged by VDR2; Q8_1 activation storage is 36 bytes/32 values = **4,608 bytes per column**. These are logical byte bounds, not measured DRAM traffic (cache, reuse and extra loads matter). All 14,336 rows and every quant block must still be consumed.

For a native dense K=4096 Q6_K route, `blocks_per_iter=VDR*nwarps*wave32/QI6_K` with `QI6_K=16`, `rows_per_block=1` on RDNA:
- gfx1100 width1: native nwarps=2, 4→8 blocks/iteration and 4→2 K-loop trips **if 1274 is not selected**; in normal validated dense width1, 1274 bypasses the path.
- gfx1201 width1: native nwarps=8, 16→32 blocks/iteration, **1→1** K-loop trips; VDR2 can leave more lanes idle, so any gain is codegen, not loop-trip halving.
- gfx1100/gfx1201 widths2–5: native nwarps=1, 2→4 blocks/iteration and 8→4 K-loop trips. Check actual widths and forced variants before using these formulas.
- gfx1030/GCN/CDNA, MMQ prefill, raw-F32 1274, unrelated quant types and Vulkan are non-promotion controls. Do not infer wave64 or GCN behavior from RDNA wave32.

### Measured evidence and source comparisons

**BigCherry historical (not new):** 1204/RD08's four-session gfx1201 PA40 campaign at b10901 passed 15 backend-reference rows (worst 2.598e-05 vs 5e-4) but decode effect **+0.204%**, CI95 [+0.094,+0.330], below its preregistered +0.3% CI95-low threshold, with prefill **-0.089%**, CI95 [-0.162,-0.007]; `rejected` is terminal. Separately, 1274's gfx1100 Q6_K F32 decode passed four sessions on a 9B Q6_K model with +4.17..+4.28% tg128 versus validated base. Neither result measures #30517's new integer-dot variant.

**Upstream external (not BigCherry):** [#30517](https://github.com/ggml-org/llama.cpp/pull/30517) R9700/gfx1201 Windows Q6_K kernel N=1 **48.0→42.1 us** (~1.14x), N=2 **63.0→43.2 us** (~1.46x), N=4 **78.5→57.3 us** (~1.37x); gfx1151 Q6_K N=1 ~1.02x, N=8 ~1.32x. Different model, pin, OS and launch/caching conditions. Amdahl: for a +3% E2E target, an optimistic 1.14x kernel gain requires **>=23.72% exposed baseline wall** in eligible kernels; 1.46x requires **>=9.24%**. Summed concurrent kernel durations are not exposed critical-path fractions.

**This audit's disposable host-only source-shaped fixture:** 256 seeded random Q6 packed blocks × 8 adjacent lane pairs × 2 dot groups × 2 chunks = **8,192/8,192 exact signed integer-dot comparisons**, zero mismatches between source VDR1 signed-byte subtract and #30517's two dp4a terms. Synthetic F32 pair-reassociation gave **904/2,048 non-bit-identical** comparisons. This is neither HIP execution nor logits correctness; no repository tests, build or GPU benchmark ran. Do not require bit identity, but do require predeclared numerical tolerances.

**Other engines/forks:** The current [vLLM GGUF plugin](https://github.com/vllm-project/vllm-gguf-plugin) tests Q6_K model loading but does not supply a demonstrated interchangeable `block_q6_K × block_q8_1` RDNA dot kernel. [SGLang GGUF I-quant MMQ gap](https://github.com/sgl-project/sglang/issues/35019) concerns different formats; it is not an adoption mechanism. The older [stew675 fork VDR2](https://github.com/stew675/llama.cpp) is the already rejected 1204 control. Adopt #30517 only as a **new arithmetic candidate after receipt gates**, not as an unqualified full PR or architecture-table port.

## Bounded implementation and terminal gates

**Gate 0 — no new patch or hardware queue:** From existing model inventory and archived receipts, identify an actual Q6_K Q8_1 MMVQ route on a physical gfx1201 (or gfx1100 width2–5) with model GGUF digest, per-layer op type, `ncols_dst`, `has_ids`, forced-geometry state, split mode, context/ubatch/MTP depth, stream/graph replay and exact device UUID. Prefer a 9B Q6_K single-GPU negative control and a genuinely eligible Q6_K model; never confuse gfx1100 1274's default-on raw-F32 path with Q8_1. Reuse existing rocprofv3/per-device kernel receipts and patch activation trace; establish actual Q6_K kernel count/time, Q8_1 quantization launches/bytes, exposed serialized critical-path fraction, row/block work and cache/DRAM bytes. **Terminate `NO_Q6_Q81_ROUTE`, `NO_EXPOSED_COST`, `BELOW_CEILING`** if absent, ambiguous, or best-case <3% E2E. No duplicate queued RAD06 experiment.

**Gate 1 — isolated source candidate only after Gate 0:** Allocate a fresh free patch ID (not 1204), default-off and with an exact Q6_K-only selector/compile guard. Port only #30517 Q6_K `VDR=2`, `vec_dot_q6_K_q8_1_impl_mmvq`, `vec_dot_q6_K_q8_1` and any required Q6-specific template/selector signature in `vecdotq.cuh`. Leave `mmvq.cu` geometry, Q2/Q3/IQ, Q6 MMQ and `1274` route unchanged. If compilation requires modifying unrelated quant call sites, fail closed. Compose after 0600/0650/1241/1274; patch-anchor cardinality, idempotence, opt-in OFF baseline and forced-geometry/graph controls must pass. Preserve the Q8_1 pool/stream and native fallback. No second dispatch table, allocator, scheduler, telemetry system or patch resurrection.

**Gate 2 — correctness and physical accounting before timing:** Include packed Q6 low/high planes, all signed scales, positive/negative/subnormal F16 `d`, Q8_1 -128/127/zero activations, K=256/512/4096 and row/tile tails. Compare integer dots exactly; compare GPU F32 to baseline and CPU with explicit tolerances, recording first divergent op and full-vocabulary logits/KLD/greedy/MTP acceptance. Test N=1/2/3/4/5/8, MoE ids negative control, Q2/Q3/Q4/Q8, F32 1274, gfx1030, multi-request same-process, multiple ubatches, 8K/80K contexts, graph capture/replay/rebind and cancellation. Count expected vs observed rows, quant blocks, launches, scratch allocations and physical bytes. Reject `NUMERIC_DRIFT`, `MISSING_WORK`, `WRONG_ROUTE`, `GRAPH_LIFETIME_FAIL`.

**Gate 3 — isolated RDNA qualification:** Baseline = same pinned BigCherry production recipe with candidate OFF, not old rejected 1204. Candidate ON only for admitted Q6 Q8_1 route. Measure gfx1100/gfx1201 separately, same model/checkpoint and clocks; width1 tg128/tg512 and widths2/4/5 verify, pp512 unchanged control. Capture ISA VGPR/SGPR/spills, LDS, occupancy, wave32, integer dot/VALU, DRAM/PCIe, launch count, quantizer, graph nodes, kernel and exposed wall time. Avoid multi-rank PCIe/no-P2P/RCCL confounds until single-GPU evidence exists. >=4 independent sessions per architecture, >=10 paired balanced rounds/session, **CI95-low >=3% eligible E2E**, <=1% unaffected control regression, no correctness/work loss. Otherwise `NO_GAIN` or `REGRESSION`; keep baseline. Kernel-only speedup never promotes production.

## Ownership and consolidation

PRBE117 is the **new arithmetic candidate** owner because the old PRBE08 numerical contract and 1204/RD08 implementation are terminally rejected/deprecated; neither may be silently reopened. PRBE47 owns **post-dot DPP wave32 reduction**, PRBE110/PRBE04 own Q6_K **MMQ prefill scale-fold**, validated 1274/RD33 owns **raw-F32 dense gfx1100 width1**, 0600/0650 own geometry, PRBE116 owns Q3_K VDR2, PKC05 Q2_K VDR2, and RAD06 independently owns live helper kernels. No duplicate kernel, selector, allocation system or benchmark queue is authorized by this plan. BCOP298 records the thin disposition only.

Sources: [#30517 source diff and R9700 tests](https://github.com/ggml-org/llama.cpp/pull/30517), [pinned vecdotq.cuh](https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/ggml/src/ggml-cuda/vecdotq.cuh), [pinned mmvq.cu](https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/ggml/src/ggml-cuda/mmvq.cu), [rejected 1204](https://github.com/audiagentic/bigcherry/tree/main/engines/llamacpp/patches/1204_rd08_q6k_mmvq_vdr2), [validated 1274](https://github.com/audiagentic/bigcherry/tree/main/engines/llamacpp/patches/1274_mmvq_kquant_f32_decode).
