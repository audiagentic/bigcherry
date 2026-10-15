---
id: PRBE70
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:58:27.196577+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: P1
---

# CK-001 / PRBE70: bounded offline Composable Kernel GEMM oracle

## 2026-10-15 b11552 CK oracle: correct the physical output layout and dtype contract (BCOP227)

**Disposition:** PRBE70 remains an *offline*, default-off typed GEMM oracle; no runtime patch, CK dependency or new benchmark queue. This is a new source-level correction to the 2026-10-09 oracle plan, not a measured GPU regression. The old example `A[M,K]=src0; B[N,K]=src1; C[M,N]` is mathematically equivalent to ggml MUL_MAT but **not byte-layout equivalent**: CK `gemm` always emits row-major `C[M,N]`, whereas ggml's `dst[ne01,ne11]` has `ne01` contiguous. Comparing flat outputs or declaring native parity with the old CK operand order is invalid for nonsquare, multi-token shapes.

### Source-verified seam and canonical zero-copy mapping

- Pinned llama.cpp b11552 `ggml/src/ggml.c::ggml_mul_mat` creates F32 `dst` with `ne={src0->ne[1],src1->ne[1],...}`; `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_mul_mat_cublas_impl` uses `CUBLAS_OP_T,CUBLAS_OP_N` with `ne01,ne11,ne10`, source leading strides `s01,s11`, destination `ldc=nb1/sizeof(float)`. It may convert src0/src1 and use temporary output before final F32 storage. This is a *physical* layout and conversion contract, not merely the matrix equation.
- Current ROCm/rocm-libraries `projects/composablekernel/profiler/src/profile_gemm.cpp::profile_gemm` (source inspected 2026-10-15) supports layout 1 `A[m,k] * B[n,k] = C[m,n]`, with row-major C and typed `F32/F16/BF16` output. BF16 requires `CK_ENABLE_BF16`; source support does not establish an installed gfx1100/gfx1201 kernel instance. Its README's older GEMM datatype list omits BF16: **installed `ckProfiler gemm` help and a verified invocation are authoritative**. Its `argc` is 14 or 16 (with warmup and iterations); do not copy `gemm_universal`'s split-K syntax.
- For a **contiguous, non-batched** ggml `src0[K,O]` weight and `src1[K,T]` activation, use CK **layout 1 with swapped operands**: `A=src1[T,K]`, `B=src0[O,K]`, `M=T`, `N=O`, `K=ne00`, `StrideA=K`, `StrideB=K`, `StrideC=O`. The CK row-major `C[T,O]` then has the *same flattened index* `t*O+o` as ggml `dst[O,T]`. This is a transpose-equivalent matrix product, **not** a claim of identical kernel tile choice or arithmetic.
- Reject direct zero-copy admission when `ne12/ne02` broadcast, nontrivial batch, `src0/src1` views/strides, output `nb1` padding, conversion, or a different physical route is not explicitly modeled. Strided variants must be derived from actual element strides and checked for valid address bounds; never substitute GGUF storage strides for converted BLAS pointers.

### Output-precision gate before any oracle parity claim

`ggml_mul_mat`'s observable output is F32. CK `gemm` with datatype 1 or 2 produces F16 or BF16 **output** (despite F32 accumulation). Consequently `ckProfiler gemm 1/2 ...` cannot be advertised as a like-for-like F32-output native comparator. For F16/BF16 inputs, record `OUTPUT_TYPE_MISMATCH`: CK is a *kernel-only screening lower bound* with an explicitly costed cast/materialization back to F32, not a direct logits/parity proof. Either obtain a verified CK variant supporting the actual F32 output without an extra path, or retain native BLAS as the correctness authority and stop at a screening decision. F32 `gemm 0` has matching storage type, but still requires numerical-tolerance and source-route checks.

**Bounded host discriminator (run 2026-10-15):** five source-shaped Python tests passed: non-square `O=3,T=2,K=4` old-layout flat-order mismatch, corrected A/B swap equality, physical shape, F16-output rounding mismatch, and corrected mapping across five shape families. These tests do **not** compile CK or BigCherry, exercise HIP, establish kernel instance support, or measure performance.

### Revised implementation/terminal gate

1. From existing `tools/bigcherry/profiling/rocprof.py::parse_kernel_trace` and `tools/lab/flash-next/prefill-kernel-table.py`, select one physically executed **BLAS** `MUL_MAT` signature, not MMQ/MMVQ or `MUL_MAT_ID`; join to a typed ggml tensor/dispatch receipt, including actual `s01/s11/ldc`, output dtype, conversions, rank and context/ubatch. Kernel-family summed time is **not** critical-path time.
2. Host fixture first: build canonical CK `A=src1,B=src0,M=T,N=O`; compare flat `C` to a ggml row oracle, check non-square and multi-token cases, and reject unknown/batched/padded layouts. Record source SHA, `CK_ENABLE_*` and actual `ckProfiler` help/instance availability. Do not queue an unverified CK command.
3. If F16/BF16 CK output remains narrow, account for F32 materialization and return `OUTPUT_TYPE_MISMATCH`/screen-only; no native-parity or promotion claim. If dtype/layout/route match, require native typed reference, verified CK profiler output, exposed serialized GEMM share `f` and `1/(1-f)-1 >= 3%` theoretical E2E ceiling **before** any owner handoff. Account for conversion, scratch, GPU synchronization, and graph lifetime.
4. Owner handoff only to existing PRBE28 (original float layout) or PRBE29/30/31 (shadow/BLAS crossover) after their independent gates; no new CK runtime dispatch. gfx1100 and gfx1201 require separate instance, numerics and VRAM qualification; gfx1030 is a negative control. Stop `UNKNOWN_SIGNATURE`, `CK_UNAVAILABLE`, `OUTPUT_TYPE_MISMATCH`, `NO_EXPOSED_HEADROOM`, `NO_PORT` or `OWNER_HANDOFF` with source/receipt IDs.

**External comparison:** ROCm CK profiler's layout/type/compile-time admission is a concrete offline mechanism. llama.cpp #30168 (merged 2026-10-09) makes host MMQ precision explicit but does not replace a float BLAS oracle; vLLM/AITER CK MXFP4 grouped MoE is a different quantized, architecture-gated route. No NVIDIA/CDNA/gfx1151 result transfers to gfx1100/gfx1201 without physical qualification. No new BigCherry performance measurement.

## Authoritative audit and disposition (2026-10-09; b11474)

**Remain pending; no runtime patch or CK dependency.** PRBE70 owns an *offline, typed, architecture-local* comparison of eligible float GEMM signatures. Do not build a new GEMM dispatcher, allocator, profiler, CK runtime integration or per-model selector. Do not run or modify queued QFP/MMQ/MTP/Meta/Radiance experiments. Existing BigCherry profile receipts and per-device rocprof traces are the first source of cost attribution; an oracle result is **not** a production performance claim.

### Source-verified blockers to the former plan

1. `GGML_CUDA_OP_TIMING` was introduced by **rejected** patch `1203_rd050607_rdna4_wmma_fa_q6k_mmq`, not by the current production recipe. Its `ggml_cuda_graph_evaluate_and_capture` diagnostic prints only `MUL_MAT` `[KxMxN]`, tensor name, aggregate milliseconds and call count. It does **not** encode weight/activation type, strides, chosen MMQ/BLAS route, physical GPU, or `MUL_MAT_ID` dimensions. Setting that variable on a normal b11474 production build is not a valid shape census. The patch's event/synchronization path also changes graph timing. **Retire** the old `parse_op_timing.py`-first plan and the unverified `BIGCHERRY_CK_ORACLE=1` runtime switch.
2. `tools/bigcherry/profiling/rocprof.py::parse_kernel_trace` already records kernel count, total/mean/p95 microseconds, VGPR, scratch and GPU agent IDs. `tools/lab/flash-next/prefill-kernel-table.py` already groups per-device kernel families; `queue-prefill-profile.sh` and `long-ctx-profile.sh` already produce the traces. **Reuse them** for ranking, without changing active QFP scripts. Neither source reconstructs exact tensor shapes, dtypes or dispatch route from kernel names. Sums of concurrent GPU kernel durations (especially across ranks) are not end-to-end critical-path fractions.
3. Pinned `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_mul_mat_cublas_impl` takes source strides and may allocate/convert non-native source types; `ggml_cuda_mul_mat_cublas` can select F32/F16/BF16 based on architecture, precision and environment. Quantized GGUF weights generally use MMQ/MMVQ before BLAS; an F16 CK GEMM with pre-expanded weights is **not** a like-for-like Q8_0 MMQ control. Such a shadow requires PRBE29 ownership, PRBE30 padding and PRBE31 crossover accounting.
4. The maintained CK source is now `ROCm/rocm-libraries/projects/composablekernel` (the former `ROCm/composable_kernel` is a read-only mirror). Its `profiler/src/profile_gemm.cpp` accepts typed data/layout, verification, initialization, timing, M/N/K and three strides; `profile_grouped_gemm.cpp` expects **comma-separated vectors** of shapes and strides, not one scalar GEMM signature. Its own verification is a self-check on profiler-generated inputs, not full GGUF logits or KV correctness.

### Exact eligible workload and ownership

- **First candidate only:** a *single physical gfx1100 or gfx1201* F16/BF16/F32 `GGML_OP_MUL_MAT` already routed through native BLAS with known contiguous/strided source tensors and stable prefill phase. Match the current actual compute dtype, not just the GGUF storage dtype. Use isolated device runs to avoid attributing RCCL, P2P, PCIe or split-rank stalls to GEMM.
- **Explicit negative controls:** Q8_0/Q4_K/Q6_K MMQ/MMVQ, `MUL_MAT_ID`/grouped experts without real token-to-expert counts, F32-converted F16 with unaccounted conversion, gfx1030, CPU/offloaded weights, unknown device, missing stride, unproven route, graph/ubatch mismatch. Record `INELIGIBLE` or `UNKNOWN`, never silently synthesize a CK shape.
- PRBE70 owns the offline decision. PRBE28 owns float original-weight padding; PRBE29/30 own quantized F16 shadows and their memory/lifetime; PRBE31 owns BLAS crossover; PKC05 and HIP tuning own quantized MMQ/MMVQ; QFP36/41 own active router/Meta paths. Do not edit those active capabilities or queue competing runs.

### Bounded implementation-ready pipeline

1. **Preflight, no GPU:** inventory the installed CK binary, source SHA, compiler/ROCm version, supported `gfx1100`/`gfx1201` instances, `ckProfiler gemm` help and an actual successful verification invocation. The source-level CLI below is verified against current `profile_gemm.cpp`, but installed-binary syntax and instance support remain **unverified**. If absent or unsupported, record `CK_UNAVAILABLE` and terminate this candidate; do not fabricate performance.
2. **Cost census:** use existing `rocprofv3` traces and `prefill-kernel-table.py` to identify the top BLAS-like kernel family and physical GPU. Bind exactly **one** shape to a pinned `test-backend-ops` or graph fixture that supplies the tensor metadata and confirms the actual BLAS dispatch. Do not infer M/N/K, dtype or device from a mangled kernel name or the rejected 1203 log. Preserve raw traces under ignored `artifacts/lab/ck-oracle/`, with a compact evidence receipt under `docs/evidence/` only after a real run.
3. **Typed manifest** (`tools/lab/ck-oracle/manifests/<id>.json`, if the preflight passes): `source_sha`, `build_plan_id`, `patch_set`, `model`, `phase`, `context`, `ubatch`, `op`, `route`, `gpu_id`, `gfx`, `weight_storage_type`, `compute_type`, `activation_type`, `output_type`, `M,N,K,batch`, `layout`, `strideA/B/C`, `calls`, `kernel_us`, `wall_us`, `critical_path_share_proof`, `trace_id`. Reject unknowns for admission; do not treat total kernel-us divided by multi-GPU wall time as a critical-path share.
4. **CK invocation** (verified *source* form for a contiguous ggml weight `src0[K,M]` and activation `src1[K,N]`, i.e. CK layout 1, `A[M,K] * B[N,K]`):
   ```text
   ckProfiler gemm <dtype:0=f32|1=f16|2=bf16> 1 1 1 0 1 M N K K K N
   ```
   Here flags are `layout verify init print time`; adapt the three strides from the **observed** manifest, not the contiguous example. Require exit=0, explicit verification success, a winning instance ID, supported architecture and timing units. Use fixed warmups/iterations and record tool SHA/flags. No `grouped_gemm` arm until actual per-expert token counts, vectorized strides, routing semantics and an eligible non-protected owner exist.
5. **Native control:** run the same typed `test-backend-ops` shape through existing BLAS route, with identical input/output dtype, stride, device, warmup, working-set size and output tolerance. Compare kernel-only results as a *screen*; record conversion/materialization, scratch, VGPR/spills and any unsupported instances. CK's own verify flag checks its synthetic input only; require native reference and, for any later patch, full-vocabulary/greedy, multi-ubatch, graph replay and same-process multi-request correctness.
6. **Causal E2E ceiling:** let `f` be the *proven serialized critical-path wall-time fraction* of eligible GEMM. Even eliminating that GEMM gives at most `1/(1-f)-1` throughput gain; if this ceiling is below 3%, **close without a patch or hardware A/B**. If CK wins a microbenchmark but its tile cannot be reproduced in existing ggml-CUDA/BLAS without a new runtime dependency, record `NO_PORT`. Otherwise hand the one narrow candidate to the existing owner (PRBE28/31 or HIP tuning), not a new plan or dispatcher.
7. **Promotion is downstream only:** default-off owner patch, exact source/activation proof, no extra VRAM beyond its own contract, four independent sessions/architecture with >=10 paired ABBA rounds/session, CI95-low >=3% E2E prefill throughput, <=1% decode/control regression, logits/greedy parity and no graph/reallocation defects. CK-only speedups never satisfy this gate.

### Cheapest discriminator and validation matrix

Before hardware, test a typed manifest validator and argv builder with fixtures for float BLAS, Q8_0 MMQ, `MUL_MAT_ID`, missing GPU/stride, invalid strides, BF16, gfx1030, grouped variable-M, correct CK argc/layout and Amdahl early closure. The 2026-10-09 disposable host model passed **13/13** such checks; it did not import BigCherry modules, compile CK, execute a profiler or benchmark any GPU. Only implement a minimal lab script after installed CK availability is established. No new `bigcherry` production import, config surface, queue, scheduler or patch package.

### Upstream/fork decision references

- ROCm source/CLI: https://github.com/ROCm/rocm-libraries/blob/develop/projects/composablekernel/profiler/src/profile_gemm.cpp ; https://github.com/ROCm/rocm-libraries/blob/develop/projects/composablekernel/profiler/src/profile_grouped_gemm.cpp ; https://github.com/ROCm/rocm-libraries/blob/develop/projects/composablekernel/profiler/README.md
- llama.cpp #30168 (open 2026-10-09) changes MMQ host/device precision propagation; relevant as a **precision/dispatch identity** reminder, not a CK implementation to transplant: https://github.com/ggml-org/llama.cpp/pull/30168
- vLLM ROCm/AITER CK MXFP4 MoE imposes shape/architecture constraints (e.g. 256 intermediate alignment and gfx950 cases). Its CDNA results do not qualify RDNA3/4: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/mxfp4_utils.py
- CK Tile grouped GEMM supports variable group shapes and preshuffle, but only with matching expert routing/layout; do not assume GGUF `MUL_MAT_ID` equivalence: https://github.com/ROCm/composable_kernel/blob/develop/example/ck_tile/17_grouped_gemm/README.md

**Measured BigCherry CK gains: none.** No installed CK binary, GPU profiler execution or GEMM A/B was available in this audit. Existing unrelated gfx1151 float-padding or CDNA CK claims are not evidence for this item.

## Files and terminal disposition

Potential later *lab-only* files: `tools/lab/ck-oracle/README.md`, `manifests/*.json`, `run_ck.py`, `run_native.py`, `DECISIONS.md`; raw outputs go to ignored `artifacts/lab/ck-oracle/`. Do not create them before the preflight. Terminal states: `CK_UNAVAILABLE`, `UNKNOWN_SIGNATURE`, `NO_ELIGIBLE_HOT_GEMM`, `NO_PORT`, `NO_GAIN`, or `OWNER_HANDOFF` with immutable source/tool/shape evidence.

## Historical provenance

Supersedes RD88 (capability-rebaseline-v3). The pre-2026-10-09 duplicate Steps and illustrative `ckProfiler` skeleton assumed rejected 1203 timing was available and did not specify typed/strided identity; those instructions are superseded by the gates above. The 2026-10-08 broad 8K/24K/98K QFP capture proposal remains historical context only, **not a new queue authorization**. No protected QFP experiment definition or plan was changed.

## Change Log

- 2026-10-08 (triage): Experiment kept pending, priority P1; ranked and scoped b11474 mechanism, VRAM, env switch and queue-env-ab.sh evidence gates; no patch implemented.

- 2026-09-09T10:58:27.196577+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:38.618139+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.444745+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.280599+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:20:07.380935+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032047_repaired-five-more-active-succ_6361
- 2026-09-10T03:20:47.919311+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:34:41.598269+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:48:37.038568+00:00 (updated-by): Updated: section:steps
- 2026-09-24T04:48:42.852623+00:00 (updated-by): Updated: section:notes
