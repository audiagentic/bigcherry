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
2. Seam A (MMVQ caller): ggml_cuda_mul_mat_vec_q at ggml/src/ggml-cuda/mmvq.cu:1421 already threads a fusion args struct (ggml_cuda_mm_fusion_args_host/_device with x_bias/gate/gate_bias/x_scale/gate_scale) through to the kernel dispatch -- add a cache lookup/reserve at this seam for src1's Q8_1 quantization before it reaches ggml_cuda_op_mul_mat_vec_q (mmvq.cu:1538), which calls mul_mat_vec_q_switch_type with src1_ddq_i as the already-quantized pointer; a cache hit substitutes a cached pointer here instead of a freshly quantized one.
3. Seam B (graph lifecycle): begin a new cache generation inside ggml_cuda_graph_evaluate_and_capture (ggml/src/ggml-cuda/ggml-cuda.cu:4185) or ggml_backend_cuda_graph_compute (ggml-cuda.cu:4420), and gate cache slab growth behind a capture_active flag so the cache never grows/reallocates while `use_cuda_graph && cuda_graph_update_required` capture is in progress (ggml-cuda.cu:4218/4368/4391 mark the capture window).
4. Add an env-var gate (default OFF, e.g. BIGCHERRY_Q81_CACHE=1) so default behavior is byte-for-byte native; add a separate verify mode that independently re-quantizes and byte-compares against the cached value.
5. Add named hit/miss/wait/timeout/contention counters exposed via the same tracing convention as other rdna-boost patches (BIGCHERRY_PATCH_TRACE-gated GGML_LOG_WARN, following patch 1205/RD12's precedent).
6. Run adversarial correctness BEFORE any performance claim: same-tensor hit; offset-collision miss (same root, different offset); shape/stride/stream miss; generation/pointer-reuse miss; capacity fallback; dual-GPU isolation; direct-producer equivalence (independent re-quantize + byte-compare). Only then validate capture/replay and off/on causal launch/memory effects. PRBE06 and PRBE10 remain separate dependent items -- do not fold their producer wiring into this foundation item.

## Detailed Solution & Technical Design

Context-owned bounded cache with generation invalidation and stable (non-relocating slab) addresses, independent of GGML_HIP_DISPATCH_MODE. Cache identity must include the exact view data address+offset (not just the root tensor) to correctly miss on offset-collisions within the same underlying allocation. Default-off preserves byte-for-byte native behavior; on/verify are explicit experiment arms. No synchronous verification or slab growth is permitted during active graph capture, since that would corrupt or crash CUDA/HIP graph capture semantics (illegal ops during capture, same class of constraint already seen in patch 1204's GGML_CUDA_OP_TIMING/graph-capture fix).

## Code Samples & Guidance

Real anchors verified in b11126 (paraphrased context, re-copy exact literal text from source at implementation time before authoring Edit() anchors):
- ggml/src/ggml-cuda/mmvq.cu:1421 `void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion)` -- cache lookup/reserve insertion point.
- ggml/src/ggml-cuda/mmvq.cu:1538 `void ggml_cuda_op_mul_mat_vec_q(... const char * src1_ddq_i ...)` calling `mul_mat_vec_q_switch_type(src0_dd_i, src0->type, src1_ddq_i, ...)` -- cached-pointer substitution point.
- ggml/src/ggml-cuda/ggml-cuda.cu:4420 `static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph)` and ggml-cuda.cu:4185 `ggml_cuda_graph_evaluate_and_capture(...)` -- generation-begin and capture_active gating point.
patch.toml sketch: schema=1, id="12xx_rd09_q81_activation_cache_foundation", state="untested", kind="new-capability", backend="hip", experiment-contract="RD09-Q81-CACHE-FOUNDATION", requires=[], conflicts=[] (TBD after auditing PRBE11's shared-quantized-X path for overlap), validation-architectures=["gfx1100","gfx1201","gfx1030"]. patch.py: `from bigcherry.patcher import Edit, FilePatch`; one Edit adding the new hip-q81-cache.{h,cu} files (insert_after an existing include block), one Edit at mmvq.cu:1421's function body (insert_before the fusion_local declaration) to add the lookup, one Edit at ggml-cuda.cu's graph-compute entry (insert_after cuda_graph_update_required computation) to bump the generation.

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
