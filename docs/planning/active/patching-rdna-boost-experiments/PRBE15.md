---
id: PRBE15
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:30.186606+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P3
---

# IMRoPE / BF16 ROPE→VIEW→SET_ROWS fusion: prove route before extension

## 2026-10-20: M-RoPE Q/K RMSNorm–RoPE–FWHT kernel admission (BCOP344)

**HOLD_ROUTE_AND_CEILING; no port or GPU queue.** New [RDNA fork #175](https://github.com/stew675/llama-cpp-rdna-boosts/pull/175) (2026-10-18) extends [#172](https://github.com/stew675/llama-cpp-rdna-boosts/pull/172)'s in-kernel Hadamard transform to MROPE/IMROPE. This is distinct from the older IMRoPE `ROPE→VIEW→SET_ROWS` scatter/BF16 axis below. External gfx1201 Qwen3.6-35B-A3B: decode 123.8→125.3 t/s (+1.2%), 31.5K two-GPU tensor prefill 5,641/5,635→6,277/6,324 (+11.7%), **but single-GPU/layer prefill flat**. Fork trace removes 32,120 `rope_multi` and 32,120 `fwht_cuda` launches; chain GPU time ~162→101 ms. No BigCherry gain or per-rank attribution was measured. Do not credit #172's Q8_1 fusion to #175.

**Pinned b11552 trace:** `src/models/qwen35{,moe}.cpp` Q/K `build_norm` emits `RMS_NORM→MUL` then `ggml_rope_multi`; `src/llama-graph.cpp::build_attn` conditionally inserts Q/K `llama_mul_mat_hadamard` when `self_k_rot` exists. `ggml/src/ggml-cuda/fwht.cu::ggml_cuda_op_mul_mat_use_fwht` dispatches a separate F32 `fwht_cuda<64/128/256>`. `ggml-cuda.cu::ggml_cuda_should_fuse_rms_norm_mul_rope` accepts only NORMAL/NEOX, F32 contiguous rows, even dimensions, `n_offs==0`; `rope.cu::rms_norm_mul_rope_f32` has no multi-axis or FWHT stage. Fork `ggml-cuda-try-fuse.cu` is absent on this pin: use the existing `ggml-cuda.cu` matcher. **Pinned `ggml_cuda_should_fuse_rope_set_rows` already excludes BF16**; fork #175's BF16 five-op abort guard is not a b11552 port requirement.

**Kernel data flow:** Only after a proven graph chain `RMS_NORM→MUL→ROPE(MROPE/IMROPE)→RESHAPE→MUL_MAT(Hadamard)`, with unique intermediate consumers, F32 contiguous Q/K, valid 4-plane sections, `n_offs==0`, non-VISION, true Hadamard hint and n=64/128/256, consider a **default-off** specialization in existing `ggml-cuda.cu`/`rope.cu`/`fwht.cuh`. Read complete x row; RMS reduction; multiply weights; reproduce `rope_multi`'s NEOX pairing, `sector=(i0/2)%sum(sections)`, IMROPE `%3` interleave, `pos[channel+nchannels*plane]` and `powf` order; stage rotated row in LDS; barrier; run the same ordered warp FWHT; write complete F32 output. Preserve every butterfly, KV write and MTP token. Allow only **exact** contiguous same-pointer x/dst alias after full read-before-write; reject other overlaps, graph outputs, extra consumers, unsupported stride/dtype/offset and changed graph pointer lifetimes. No new allocator, cache, scheduler, telemetry or transfer. gfx1100/gfx1201 separately qualified; gfx1030 native. Inspect compiled wave, LDS, VGPR, spills and launch geometry rather than extrapolate.

**Gate A, before implementation:** capture actual b11552 graph adjacency, `self_k_rot`, op dispatch, consumer/alias spans, per-rank launch counts, physical work/bytes, graph replay and exposed wall for q8_0/f16 KV, 512/2048/31.5K prefill, 24K/99K context where supported, verify widths 1/2/4/8, single/two-XTX/R9700/mixed splits. Explain tensor-only prefill anomaly with paired rank traces and equal token/ubatch/work accounting. If no route close `NO_MROPE_FWHT_ROUTE`; if optimistic removable exposed E2E <2.9126%, close `BELOW_CEILING`. Logical intermediate upper bound is at most 24n bytes/Q-or-K row (6 KiB at n=256), **not measured HBM**; no required arithmetic or PCIe bytes disappear.

**Correctness and promotion:** Host selector/section/partial-rotation/offset/alias tests; GPU complete output and untouched-cache sentinels, repeated same-process multi-ubatch, graph on/off/reallocation, logits/KLD/greedy/MTP acceptance, full expected-versus-observed FWHT work. Existing native chain is fallback. At least four independent sessions and ten balanced paired rounds/session; CI95-low >=3% E2E and <=1% control regression with full numeric/physical-work parity, else `NUMERIC_FAIL`/`ALIAS_OR_GRAPH_UNSAFE`/`NO_GAIN`. Host-only 77,568 plane, 600 bitwise butterfly, 300 exact-alias comparisons, 10 pinned-source assertions and Python compilation passed; **no HIP build or GPU run**.

**Ownership:** PRBE15 sole multi-axis RoPE fusion owner. PRBE115/1307 Q8_1 cache, RNX08 other norm/RoPE, QFP17 prefill, PNRO04/05 GDN and PRBE41 SSM remain separate. Do not edit active shape-harness attribution, RAD14 FP8/MTP or RCS05 worker work. vLLM ROCm/AITER and SGLang Q/K RoPE fusions are semantic comparators, not drop-in GGUF FWHT kernels.



## Decision and scope (2026-10-09, llama.cpp b11474)

**Pending / P3 / no implementation authorised.** This is an opt-in extension of native CUDA/HIP ROPE→VIEW→SET_ROWS fusion, not a port of rejected patch 1004 (already ancestral). Two independent eligibility axes were previously bundled: (A) IMRoPE position semantics and indexed cache scatter; (B) BF16 destination conversion. Qualify A with F32/F16 first; B only if a BF16 destination actually occurs in the same graph and the native control works. Do not create separate dispatch, allocator, or cache mechanisms.

**Source facts at pinned b11474** (line numbers intentionally not frozen; symbols are anchors):
- `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_should_fuse_rope_set_rows` accepts only NORMAL/NEOX and F32/F16 destination; `ggml_can_fuse_subgraph` and `ggml_cuda_check_fusion_memory_ranges` already own edge/output/alias admission. Preserve both.
- `ggml/src/ggml-cuda/rope.cu::ggml_cuda_op_rope_impl<true>` computes `row_indices`, `set_rows_stride`, and `dst_type`, but the `is_mrope && !is_vision` branch calls `rope_multi_cuda<forward>` with `T *` source **and** destination, ignoring all three. `rope_multi_cuda` and `rope_multi` have no scatter index or stride. **Widening the mode gate alone can write the wrong KV rows.** A differing destination type can also be interpreted at the source element width.
- IMRoPE is `GGML_ROPE_TYPE_IMROPE=40` (MROPE bit set), uses four position planes and `mrope_sections`; its per-axis positions cannot be replaced by one scalar token position.
- `ggml/src/ggml-cuda/set-rows.cu::set_rows_cuda` **already supports BF16 destination** for F32 source. The missing BF16 support is in the fused ROPE output dispatch, not in the standalone SET_ROWS operation. The native `ggml_cuda_op_set_rows` source guard admits F32, or F16 only when destination is F16.
- `src/models/qwen3vl.cpp` builds `Kcur = ggml_rope_multi(...)`; `src/llama-graph.cpp::llm_graph_context::build_attn` passes Kcur to `llama_kv_cache::cpy_k`; `src/llama-kv-cache.cpp::cpy_k` makes a `ggml_view_2d` then `ggml_set_rows`. This is a **source-level candidate** for the three-node pattern, not proof that the composed HIP scheduler emits an activating, eligible contiguous graph or that its cache dtype is BF16.

**Cheap discriminator run:** 22 host/static assertions against pinned b11474 and a two-token/two-head/four-element scatter model passed. With cache indices [3,1], sequential output writes disagree with the reference scatter and overwrite an unrelated row. This proves a proposed gate-only port is invalid, not that native production misbehaves. No GPU run or speedup was measured.

## Implementation-level gate and bounded work

1. **Activation census, before patch authoring.** On an existing Qwen3-VL IMRoPE model, inspect composed `ggml_cgraph` around `Kcur`: op sequence ROPE→VIEW→SET_ROWS, edges, `ggml_can_fuse_subgraph`/range checks, destination dtype, source dtype, n_dims/n_offs, 4-plane position layout, sections, strides, slot-index dtype, alias flags, graph capture and backend placement. Record count and GPU time for the unfused ROPE and SET_ROWS (VIEW is metadata-only). Include text-only and mixed T/H/W positions, nonsequential KV slots, multi-ubatch and repeated same-process requests. If no eligible graph, close PRBE15 without kernel work. Do not infer coverage from the model name.
2. **A: IMRoPE scatter, F32/F16 only.** Extend existing `rope_multi` and `rope_multi_cuda` with `row_indices` and `set_rows_stride` (in destination elements); add a guarded indexed output-address path consistent with the native NORMAL/NEOX fused scatter and preserve the existing non-fused path. Keep four-plane `pos[i2 + ne02*axis]`, sections/interleaving, `n_offs`, unrotated elements and `inplace` semantics intact. Require source/destination type identity initially; do not reinterpret a F32 pointer as F16. For nonidentity output type, separately instantiate `<T,D>` and use explicit `ggml_cuda_cast<D>` only after parity tests. Reject unsupported stride/shape/index/offset/type combinations to native unfused execution.
3. **B: BF16 destination, separately gated.** Only after a real BF16 cache graph is observed and the standalone F32→BF16 SET_ROWS control passes, extend the fused NORMAL/NEOX `rope_norm_cuda/rope_neox_cuda` `D` instantiations and `ggml_cuda_op_rope_impl` dispatch to `nv_bfloat16`. For IMRoPE+BF16, require A's indexed multi-rope path and `<T,D>` conversion first. Do **not** widen `ggml_cuda_should_fuse_rope_set_rows` before every matching kernel variant exists. Never change the native standalone BF16 SET_ROWS dispatch.
4. **Ownership and lifetimes.** `ggml_cuda_try_fuse` retains graph-edge/range/consumer ownership; `ggml_cuda_op_rope_impl` retains launch and stream ownership; `set_rows->data` is the existing KV cache allocation, and `row_indices` is an existing scheduler-provided device tensor. No new buffer, persistent slot map, scheduler or synchronization path. Check output byte bounds using destination type size, slot capacity and stride; ensure no writes to untouched rows. Respect graph-capture replay and updated slot indices.
5. **Instrumentation.** Add a `BIGCHERRY_PATCH_TRACE`-gated WARN `PATCH_HIT` only for the newly admitted fused route, with mode, source/destination dtype, head/sequence shape, slot-index type, and capture status; do not emit per-token synchronous D2H telemetry. Capture a native-fallback reason in host-side selector fixtures, not another production registry.
6. **Cheapest correctness tests before hardware.** A host address oracle over permuted/duplicate/edge slots, multiple heads, varying strides and destination widths; a selector table that rejects unsupported dtype/offset/stride; a source fixture asserting four-plane position and section mapping; a composed patch/graph fixture proving exact edges and single-output memory-range checks. Do not queue GPU experiments until these pass.
7. **Hardware qualification (if activation and theoretical ceiling justify).** On gfx1100 then gfx1201 (gfx1030 only if actual eligible route), compare unmodified native B with B+extension, same binary where possible. Test Qwen3-VL text and mixed T/H/W, F32/F16 cache, BF16 only when valid, nonmonotonic and repeated slot writes, context 8K/24K/98K where model supports, single/multi-ubatch, graph on/off and repeated same-process requests. Require exact slot-by-slot K-cache parity for matching arithmetic; where conversion order differs, report max ULP/KLD and full-vocabulary logits, greedy text and MTP acceptance against a frozen unfused reference. Use canaries/sentinels for untouched KV rows and detect out-of-bounds writes. Faster because a write disappeared is failure.
8. **Promotion/rejection.** First prove actual fused route and account for ROPE/SET_ROWS launches and GPU critical-path share. Theoretical E2E ceiling cannot exceed the measured unfused share; close if <3% even with perfect fusion or if route does not activate. Otherwise require ≥4 independent sessions, ≥10 balanced paired rounds/session, CI95-low ≥3% E2E improvement, ≤1% control regression, unchanged work/bytes and all correctness gates. On failure, keep native unfused fallback and retire the extension.

## Dependencies, consolidation and external mechanisms

- Rejected `engines/llamacpp/patches/1004_rms_norm_mul_rope_fusion` remains provenance only. Native NORMAL/NEOX fusion owns the implementation surface. Do not duplicate `ggml_can_fuse_subgraph`, memory-range admission, or `set_rows_cuda`.
- PRBE54's Q5 KV conversion, QFP17's QSA mask and active QFP/MTP/Radiance/engine-move work own different mechanisms; **no edits to their plans, patches, tests or queued lanes**. This plan is the sole owner of this specific IMRoPE/BF16 fusion question.
- Upstream llama.cpp `master` inspected 2026-10-09: the same NORMAL/NEOX/F32-F16 gate and unindexed `rope_multi_cuda` remain. No upstream replacement was identified. Source: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/rope.cu and https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu .
- SGLang's ROCm/AITER `forward_prepare_aiter_fused_mrope` explicitly passes positions, sections, interleaving and KV slot mapping to a decode-only fused kernel. It is a **semantic/eligibility reference**, not a directly portable llama.cpp or RDNA performance result: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3.py . SGLang issue #35345 documents the risk of treating multi-axis positions as one-dimensional.
- Fork evidence on ROPE→SET_ROWS contraction-sensitive numerics is a reason to compare cache bytes and greedy outputs, not proof of a BigCherry regression: https://github.com/stew675/llama-cpp-rdna-boosts/ .

## Files and commands

Existing source anchors: `ggml/src/ggml-cuda/{ggml-cuda.cu,rope.cu,set-rows.cu}`, `src/models/qwen3vl.cpp`, `src/llama-graph.cpp`, `src/llama-kv-cache.cpp`. Future opt-in patch, only after activation: `engines/llamacpp/patches/<next>_prbe15_imrope_set_rows/` with `patch.toml`, `patch.py`, `SUMMARY.md` and focused `tools/tests/patch/` source/selector fixtures. Run repository `patch-lint`, `patch-rebase-check` and composed `test-backend-ops` before GPU A/B; derive exact CLI from current engine registry rather than stale root `patches/` paths. No patch or experiment was created in this audit.

## Historical provenance

The September b11126 design treated this as a two-guard change and incorrectly described BF16 standalone SET_ROWS as needing a write-path implementation. Both assumptions are superseded by the b11474 source trace above. The original 1004 fusion was already upstream-absorbed. PRBE15 remains pending rather than creating a speculative patch. No measured BigCherry speedup exists for this slice.


## Change Log

- 2026-10-08 (triage): Kept pending at P3. No new validated patch. 1004 is rejected as upstream-absorbed (its SUMMARY); b11474 needs an activating IMRoPE/BF16 SET_ROWS graph before widening ggml_cuda_should_fuse_rope_set_rows. No branch or measured benefit. Keep pending, not in_progress.

- 2026-09-09T10:54:30.186606+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:39.895272+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.196499+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.902384+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:49:05.374902+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024925_rdna-successors-prbe1416-now_9529
- 2026-09-10T02:49:25.519390+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:28:47.844482+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:39:34.219151+00:00 (updated-by): Updated: section:steps, section:notes
