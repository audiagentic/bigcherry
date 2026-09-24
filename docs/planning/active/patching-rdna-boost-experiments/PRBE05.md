---
id: PRBE05
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:53:47.797556+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Implement per-graph Q8_1 activation cache foundation

## Description

TODO, real design against verified b11126 anchors. Wire a context-owned Q8_1 activation cache into the MMVQ call path and CUDA/HIP graph lifecycle. Historical predecessor RD09 is closed/provenance-only (source identities ff6fde5046ffb86672e05da640d2bfb20d4bfdfc, rebased 299f6e985...); PRBE05 is a fresh foundation-plus-wiring item, not a reapplication.

## Steps

1. Add a small hip-q81-cache API: a key struct (generation, view-root pointer, exact view data address/offset, dimensions/strides, stream) plus find/reserve/publish functions, hard byte/entry caps, and a native-fallback path when the cache misses or is exhausted.
2. Seam (corrected): ggml_cuda_mul_mat_vec_q (mmvq.cu:1421) does NOT call ggml_cuda_op_mul_mat_vec_q (mmvq.cu:1538, a separate function used elsewhere) -- it locally allocates src1_q8_1 via ggml_cuda_pool_alloc<char>, calls quantize_row_q8_1_cuda(...) directly at mmvq.cu:1506, then feeds the freshly quantized pointer straight into mul_mat_vec_q_switch_type(...) at mmvq.cu:1531. Replace that exact block (src1_q8_1 alloc through the quantize_row_q8_1_cuda call) with a cache lookup/reserve; on hit, feed the cached pointer to mul_mat_vec_q_switch_type unchanged; on miss, quantize as before and publish into the cache.
3. Add context-owned cache storage: extend struct ggml_backend_cuda_context in ggml/src/ggml-cuda/common.cuh (existing file -- no new .h/.cu files, since the patcher is anchor-only on existing files) with the cache's ownership/state.
4. Seam B (graph lifecycle): begin a new cache generation inside ggml_backend_cuda_graph_compute (ggml-cuda.cu:4420), and gate cache slab growth/synchronous verification behind a capture_active flag so the cache never grows/reallocates while `use_cuda_graph && cuda_graph_update_required` capture is in progress (ggml-cuda.cu:4218/4368/4391 mark the capture window). Pin cached storage for the lifetime of each live ggml_cuda_graph/graph_key, or disable caching entirely under capture/replay -- never recycle a captured pointer.
5. Add an env-var gate (default OFF, e.g. BIGCHERRY_Q81_CACHE=1) so default behavior is byte-for-byte native; add a separate verify mode that independently re-quantizes and byte-compares against the cached value.
6. Add named hit/miss/wait/timeout/contention counters via BIGCHERRY_PATCH_TRACE-gated GGML_LOG_WARN, following patch 1205/RD12's precedent.
7. Add a correctness test targeting GGML_OP_MUL_MAT with quantized src0 + F32 src1 shapes that select the MMVQ path (test-backend-ops has no op literally named MUL_MAT_VEC_Q; the op under test is MUL_MAT with shapes that dispatch to mmvq).
8. Run adversarial correctness BEFORE any performance claim: same-tensor hit; offset-collision miss; shape/stride/stream miss; generation/pointer-reuse miss; capacity fallback; dual-GPU isolation; direct-producer equivalence (independent re-quantize + byte-compare). Only then validate capture/replay and off/on causal launch/memory effects. PRBE06 and PRBE10 remain separate dependent items.

## Detailed Solution & Technical Design

Context-owned bounded cache with generation invalidation and stable (non-relocating slab) addresses, independent of GGML_HIP_DISPATCH_MODE. Cache identity must include the exact view data address+offset (not just the root tensor). Default-off preserves byte-for-byte native behavior; on/verify are explicit experiment arms. Corrected call path: ggml_cuda_mul_mat_vec_q (mmvq.cu:1421) is self-contained -- it quantizes src1 itself (quantize_row_q8_1_cuda, mmvq.cu:1506) and dispatches directly via mul_mat_vec_q_switch_type (mmvq.cu:1531); ggml_cuda_op_mul_mat_vec_q (mmvq.cu:1538) is a distinct function, not a callee of the former, and is NOT the integration seam. The cache must live in an existing struct (ggml_backend_cuda_context in common.cuh) since BigCherry's patcher only edits existing files via anchored Edit()s -- no new .h/.cu files. No synchronous verification or slab growth is permitted during active graph capture; cached storage must be pinned per live graph/graph_key or disabled under capture/replay to avoid recycling captured pointers.

## Code Samples & Guidance

Real anchors verified in b11126 (`git -C work/upstream/llama.cpp.git grep` against b11126):
- ggml/src/ggml-cuda/mmvq.cu:1421 `void ggml_cuda_mul_mat_vec_q(...)` -- self-contained caller; quantizes and dispatches directly.
- ggml/src/ggml-cuda/mmvq.cu:1506 `quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);` -- exact anchor for the cache lookup/publish replacement.
- ggml/src/ggml-cuda/mmvq.cu:1531 `mul_mat_vec_q_switch_type(...)` -- consumes the (possibly cached) src1_q8_1 pointer.
- ggml/src/ggml-cuda/ggml-cuda.cu:4420 `static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph)` -- generation-begin/capture_active gating point.
- ggml/src/ggml-cuda/common.cuh: add cache ownership fields to the existing `struct ggml_backend_cuda_context` (exact member list TBD at implementation time; must be an Edit to this existing file, not a new file).
patch.toml sketch: schema=1, id="12xx_rd09_q81_activation_cache_foundation", state="untested", kind="new-capability", backend="hip", experiment-contract="RD09-Q81-CACHE-FOUNDATION", requires=[], validation-architectures=["gfx1100","gfx1201","gfx1030"].

## Files

New ggml/src/ggml-cuda/hip-q81-cache.{h,cu} (or equivalent); ggml/src/ggml-cuda/mmvq.cu (seam A); ggml/src/ggml-cuda/ggml-cuda.cu (seam B, graph lifecycle); patches/12xx_rd09_q81_activation_cache_foundation/{patch.toml,patch.py}; adversarial key/capture test fixtures; independent re-quantize/byte-compare verifier; campaign evidence.

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint`, `patch-rebase-check --focal-overlay 12xx_rd09_q81_activation_cache_foundation --source bigcherry-tuning`; unit tests for the key-matrix (hit/offset-collision-miss/shape-stride-stream-miss/generation-miss/capacity-fallback). Hardware (Brutus, not run here): dual-GPU isolation test; graph warm-up/capture/replay stability; off-vs-on causal launch/memory/perf evidence on gfx1100/gfx1201/gfx1030; test-backend-ops MUL_MAT_VEC_Q cases with cache on/off/verify.

## Effort & Risk

L effort, advanced skill -- new cache subsystem touching graph-capture-sensitive code; highest risk is a stale-pointer or graph-capture-time allocation bug, mitigated by the adversarial-correctness-first ordering mandated in acceptance criteria.

## Standards

Bounded cache; generation safety; no stale pointers; exact quantizer reference; no graph-time allocation; independent of dispatch mode.

## Acceptance Criteria

Cache integration passes all key and capture gates with zero Q8 block mismatches; off mode is unchanged; exhaustion and nonqualifying paths fall back natively; positive performance/launch evidence is required before promotion and dependent children remain separate.

## Notes

Supersedes: RD09
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd09

Supersedes: RD09 (closed historical predecessor). PRBE05 is the actionable owner. PRBE06 and PRBE18 consume PRBE05; PRBE11 consumes it only if a source audit proves a real dependency. Preserve source identities ff6fde5046ffb86672e05da640d2bfb20d4bfdfc and rebased 299f6e985... in provenance; do not make RD09 a live dependency target.

2026-09-24 relevance at b11126: TODO. Real anchors verified via `git -C work/upstream/llama.cpp.git.git grep` against b11126 (mmvq.cu:1421/1538, ggml-cuda.cu:4185/4420, norm.cu:478/502, binbcast.cu:445). GPT design request submitted (req_5e0e57d9e29f44b0, batched with PRBE06/PRBE10); gateway was heavily congested (10 active gpt-auto sessions, composer-operation-timeout failures observed via agent_task_gateway_overview) -- this plan was authored directly against the verified anchors in case that request never completes.

2026-09-24 GPT review req_7f4dea253b7247f0 applied: corrected the MMVQ call-path assumption (ggml_cuda_mul_mat_vec_q at mmvq.cu:1421 is self-contained -- it does NOT call ggml_cuda_op_mul_mat_vec_q at mmvq.cu:1538; verified via grep). Moved the cache seam to the exact quantize_row_q8_1_cuda call at mmvq.cu:1506. Moved cache ownership into the existing ggml_backend_cuda_context (common.cuh) instead of proposed new .h/.cu files, since the patcher is anchor-only on existing files. Added explicit graph-capture pointer-lifetime/ownership requirement and corrected the test target to GGML_OP_MUL_MAT (no literal MUL_MAT_VEC_Q op exists).

## Change Log

- 2026-09-09T10:53:47.797556+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:10:28.989477+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.147212+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.822900+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:31:39.715349+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023232_the-next-three-rdna-successors_5807
- 2026-09-10T02:32:32.971906+00:00 (updated-by): Updated: section:ledger-events
- chg_20260911_221341_documented-the-real-status-of_8365
- 2026-09-11T22:13:41.602225+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T09:52:10.845595+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:notes
- chg_20260912_095506_cleaned-the-active-rdna-boost_4906
- 2026-09-12T09:55:06.200665+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:32:42.590942+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:36:28.306424+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:notes
