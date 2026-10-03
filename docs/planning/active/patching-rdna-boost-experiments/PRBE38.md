---
id: PRBE38
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:05.203236+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-FUS-002: Fuse literal GEMV -> UNARY -> MUL into the vector epilogue

## Description

PRBE38 is the canonical backend/correctness owner for **literal** vector `MUL_MAT[/ID] -> UNARY(SILU|SIGMOID|SOFTPLUS) -> MUL` topologies when production tracing proves that the gated pointwise work remains a separate launch after the GEMV/MMVQ/MMVF producer(s).

This is deliberately distinct from PRBE37 and from b11126's already-native canonical GLU fusion:
- supported canonical `GGML_OP_GLU` / SWIGLU-family graphs are recognized by `ggml_cuda_should_fuse_mul_mat(...)` and feed `ggml_cuda_mm_fusion_args_host/_device` into `mul_mat_vec_q/f` epilogues;
- literal `UNARY(SILU|SIGMOID|SOFTPLUS) -> MUL` is handled by `ggml_cuda_op_unary_mul`, a separate pointwise fusion path;
- therefore `ggml_cuda_op_unary_mul` can save a unary+MUL elementwise pass without necessarily removing the preceding GEMV output store/reload or the post-GEMV pointwise launch.

PRBE38 exists only for that remaining literal topology. It must not reimplement or broaden the already-fused canonical `GGML_OP_GLU` path. QFP13 supplies launch-gap prioritisation and the common performance gate; implementation semantics live here.

## Ownership Boundary

PRBE38 owns:
- mapping a trace-proven literal gated topology to current CUDA/HIP graph nodes;
- eligibility/matcher logic for folding that topology into the existing vector fusion descriptor/epilogue where legal;
- alias, broadcast, stride, dtype, consumer-count, and graph-elision correctness;
- MMVQ/MMVF epilogue wiring and fail-closed fallback;
- backend-op/production correctness tests and launch-count verification.

PRBE38 does **not** own:
- canonical `GGML_OP_GLU` capability or its separate SIGMOID/GEMM gaps (PRBE37);
- generic launch prioritisation (QFP13);
- residual VIEW+ADD (PRBE39);
- paired K/V projection fusion (PRBE40);
- Q8_1 activation caching (PRBE05).

## Stage 0 — Prove A Real Separate Launch

Before implementation:
1. Use the QFP13 production n-gram/kernel census on gfx1100 and gfx1201 to identify the exact `MMVQ/MMVF -> unary_gated` (or equivalent) chain and frequency/token.
2. Map the traced pointwise kernel to `ggml_cuda_op_unary_mul` and map its input producer(s) to the corresponding `GGML_OP_MUL_MAT[/ID]` vector dispatch.
3. Prove the pointwise operation is a separate launch/global-memory round trip after the vector GEMV path. Source structure alone is insufficient.
4. Prove the intermediate(s) may be elided: no extra consumer that requires the materialized pre-gated value, and compatible broadcast/layout/alias semantics.

Stop/park PRBE38 if current production graphs are already canonical `GGML_OP_GLU`, if `ggml_cuda_op_unary_mul` is not a separate launch for the traced topology, or if the measured occurrence rate cannot plausibly meet QFP13's acceptance gate.

## Implementation Plan

1. Reuse the existing `ggml_cuda_mm_fusion_args_host/_device` path and `ggml_cuda_should_fuse_mul_mat(...)` infrastructure where its representation is semantically sufficient. Do not introduce a Flash-Next-specific fusion descriptor or new standalone kernel family.
2. Extend graph matching only for the trace-proven literal topology. Resolve which vector producer supplies the value and which operand supplies the post-activation multiply using the real graph; do not assume operand order from kernel names.
3. Require exact supported activation semantics. Start with the production-proven ops only; SILU/SIGMOID/SOFTPLUS are candidates because `ggml_cuda_op_unary_mul` supports them, not blanket permission to fuse every unary op.
4. Fail closed for unsupported broadcast/stride/dtype, aliases, multiple consumers, non-vector/GEMM shapes, or graph/scheduler cases where the intermediate cannot legally be elided.
5. Dispatch through the existing `mul_mat_vec_q/f(..., fusion_data)` vector path so activation/multiply happens before the final global-memory store. Keep `ggml_cuda_op_unary_mul` as the native fallback for non-GEMV and unsupported cases.
6. Remove only special-case plumbing made unreachable for the accepted fused arm. Do not delete the generic pointwise path.

## Technical Constraints

The optimized arm must preserve:
- the same accumulation precision as the native MMVQ/MMVF producer;
- activation semantics and operation order;
- output dtype conversion/rounding point unless equivalence is explicitly proven;
- broadcast indexing and strides;
- graph-capture legality and pointer lifetime;
- CUDA/HIP behavior through the existing shared backend path.

The desired dataflow is conceptually:

`vector accumulation -> required activation -> required elementwise multiply -> final output store`

instead of:

`vector accumulation -> GEMV output store -> pointwise kernel reload -> activation/multiply -> second store`.

Exact fusion-data fields must follow the current upstream structures. If the existing descriptor cannot express the literal topology without semantic ambiguity, stop and document that gap before adding fields or kernel variants.

## Upstream Reference / Source Anchors

Verified against llama.cpp b11126:
- `ggml/src/ggml-cuda/ggml-cuda.cu`: `ggml_cuda_should_fuse_mul_mat(...)`, `ggml_cuda_mm_fusion_args_host/_device`, canonical GLU matching, and `ggml_cuda_op_unary_mul` dispatch/matcher sites;
- `ggml/src/ggml-cuda/mmvq.cu`: quantized vector GEMV fusion-data consumer/epilogue;
- `ggml/src/ggml-cuda/mmvf.cu`: floating vector GEMV fusion-data consumer/epilogue.

The key source reconciliation is: native canonical GLU fusion and native literal `UNARY->MUL` pointwise fusion are two different mechanisms. PRBE38 targets only the launch/memory boundary that remains between a literal graph's vector producer and the separate pointwise mechanism.

## Validation

Offline/source:
- verify all current b11126 matcher/dispatch conditions before editing;
- add focused backend-op positive cases for every accepted literal activation/topology;
- add negative controls for alternate broadcast, non-contiguous layouts, aliases, multiple consumers, unsupported activation/dtype, and non-vector/GEMM paths;
- patch lint/rebase checks for the resulting BigCherry patch package.

Hardware/profile-v2 ABBA on the existing XTX+XTX+R9700 tensor split:
- shallow (~8-10K) and deep (~65-80K) context;
- exact target adjacency and standalone pointwise launch count before/after;
- total kernels/token and launch-gap ms/token;
- MMVQ/MMVF duration/register/occupancy regression check;
- target verify ms/step and effective TG;
- deterministic/greedy output equivalence against unfused control on gfx1100 and gfx1201.

## Acceptance Criteria

- Production trace proves a literal vector-producer -> gated pointwise topology and its frequency/token.
- The accepted topology executes without the previously separate post-GEMV pointwise launch and without a compensating new launch.
- No required intermediate value is incorrectly elided; aliases/broadcast/consumer-count guards are explicit and tested.
- Unsupported cases continue through the native unfused/`ggml_cuda_op_unary_mul` path.
- Canonical `GGML_OP_GLU` handling remains owned by the existing upstream mechanism/PRBE37 and is not duplicated.
- No new standalone fusion kernel family is introduced for this target unless reuse of the current descriptor is proven impossible and separately justified.
- QFP13's performance gate is met: >=0.5% end-to-end TG/ms-step improvement or >=0.10 ms/token measured serial launch-gap reduction without end-to-end regression. Otherwise park the item.

## Effort & Risk

M. Arithmetic is not novel; risk is proving graph identity, operand semantics, aliases/broadcast, and legal intermediate elision. A failed proof must fall back rather than widen the matcher.

## Standards

- Evidence before implementation.
- Reuse upstream/native fusion infrastructure.
- One canonical owner per topology.
- Fail closed on semantic uncertainty.
- Preserve numerical behavior and graph-capture safety.
- Re-profile after the single local optimization before adding another fusion.

## Notes

Supersedes: RD46
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd46

PRBE37 and PRBE38 are adjacent but not duplicates: PRBE37 covers canonical GLU epilogue capability/gaps; PRBE38 covers a trace-proven **literal** `UNARY->MUL` topology that remains a separate post-vector launch. QFP13 is the measurement/ranking umbrella and must not carry another copy of this implementation design.

Historical 2026-09-24 disposition claiming PRBE38 was fully upstream-absorbed was corrected after reading b11126: the canonical GLU matcher does not by itself prove literal `UNARY->MUL` is folded into the GEMV epilogue. The separate `ggml_cuda_op_unary_mul` path is the relevant fallback/native pointwise mechanism.

## Change Log

- 2026-09-09T10:56:05.203236+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-24: Reopened after source reconciliation showed canonical GLU epilogue fusion does not establish literal `UNARY->MUL` GEMV-epilogue fusion.
- 2026-10-04 (agent): Removed contradictory upstream-absorbed/reopened instructions; made PRBE38 the single backend owner for trace-proven literal vector `MUL_MAT[/ID] -> UNARY -> MUL`; moved duplicate implementation ownership out of QFP13.

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.301853+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- chg_20260910_030318_repaired-three-patching-succes_9681
- chg_20260910_030619_removed-migration-placeholder_7703
- chg_20260924_023553_re-scoped-11-rdna-boost-planni_1625
- 2026-09-24T02:32:19.601706+00:00 (state-transition): State: pending -> superseded
- 2026-09-24T04:35:22.931353+00:00 (state-transition): State: superseded -> pending
