---
id: PRBE38
order: 0
plan: patching-rdna-boost-experiments
state: completed
created-at: '2026-09-09T10:56:05.203236+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P2
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

- 2026-10-08 (triage): Completed as planning roll-up into QFP35 (decode-fusion follow-up, still pending). Native ggml_cuda_op_unary_mul handles literal UNARY->MUL but does not prove preceding GEMV epilogue fusion. 1312_mul_q81 is validated and in production for Q8_1 emission. Residual literal epilogue owner QFP35.

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

## 2026-10-18 b11552 kernel-source audit — literal GEMV to gated pointwise (BCOP299)

**Disposition: RECEIPT_FIRST / no implementation admitted.** This is a *new* audit of the literal GEMV-to-gated-pointwise boundary, not QFP35 HC gate fusion, PRBE37 canonical GLU, PRBE39 residual ADD, PRBE40 paired projections, or active RAD06 helper GEMM. The PRBE38 document is in the completed planning archive; that does **not** mean this unimplemented runtime candidate is validated. No new patch/dispatch/telemetry owner is created.

### Exact source and physical execution path

- Pinned llama.cpp b11552 (`23b0202a189c44a54625aadcb37a946dd1d6278d`), `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_try_fuse`: canonical `{MUL_MAT,MUL_MAT,GLU}` (and ID/bias variants) already dispatches `ggml_cuda_mul_mat_vec_{q,f}` with `ggml_cuda_mm_fusion_args_host`; do not re-port. Separately `{UNARY(SILU|SIGMOID|SOFTPLUS),MUL}` dispatches `ggml_cuda_op_unary_mul` and skips one node. There is no native literal `{MUL_MAT,UNARY,MUL}` matcher in this pinned source or the inspected current upstream master. The latter is **not** proof a live production graph matches that triple.
- `ggml/src/ggml-cuda/unary.cu::ggml_cuda_op_unary_mul_impl` reads both F32/F16/BF16 contiguous same-shape inputs and writes the MUL output in **one pointwise kernel**. Therefore a trace of `MMVF -> UNARY -> MUL` is **not** two removable pointwise launches if native unary+MUL already fused.
- `ggml/src/ggml-cuda/mmvf.cu::mul_mat_vec_f` reduces a float/BF16/F16 weight GEMV with wave reduction and optional shared-memory cross-wave reduction (`__syncthreads`), then applies bias and a **second weight GEMV** gate for SWIGLU/GEGLU/SWIGLU_OAI/SWIGLU_CLAMP before the F32 store. `fusion.gate` is a weight pointer, not an arbitrary already-materialized `other_src` tensor. The current descriptor cannot express generic `unary(GEMV_output) * external_vector` or arbitrary SIGMOID/SOFTPLUS by reinterpreting GLU. `mmvq.cu::ggml_cuda_mul_mat_vec_q` has the analogous F32 output/fusion descriptor and Q8_1 quantization/stream-pool lifetime; do not silently change quantized arithmetic.
- `ggml_cuda_check_fusion_memory_ranges` checks input/output overlap over a **contiguous** node span; `ggml_can_fuse_subgraph` also proves legal elision. A nonadjacent graph match needs a separately proven full skipped-node set, not an unguarded partial-range call. Preserve existing graph UID, pool allocation, stream ordering and no-P2P Meta rank boundaries.

### Evidence and cheapest discriminator

QFP13's older 10K profile describes **about 26** `MMVF -> unary_op -> k_bin_bcast` neighbourhoods/token and a roughly 3.5 us inter-kernel gap; these are *adjacency/aggregate estimates*, **not** 26 proven literal GEMV-to-UNARY/MUL candidates or 26 removable launches. If all 26 were eligible and exactly one exposed launch-gap per hit disappeared, the optimistic launch-only ceiling is `26 * 3.5 us = 0.091 ms/token`, ~0.714% of a hypothetical 78.5 tok/s (12.74 ms/token) control; this excludes fusion overhead and is **not a measured gain**. For illustrative F32 output width 10240, avoiding one materialized store+read saves at most `8*10240 = 81920` logical bytes/hit (~2.03 MiB for 26); actual DRAM bytes, occupancy, spills, LDS, wave32, and critical-path saving are unknown. Do not double-count native unary+MUL fusion or sum concurrent GPU intervals as serialized time.

Disposable host-only check: 8192 seeded F32 pairs for each of SILU, SIGMOID and SOFTPLUS gave **8192/8192 bit differences per activation** between `activation(gemv)*external` and the tempting but invalid `gemv*activation(external)` GLU-style substitution. This demonstrates a semantic trap, not HIP output error. Five scoped pinned-source assertions passed after correcting two overly literal symbol-name checks. No repository pytest, patch application, HIP compilation, GPU profiling, hardware run or E2E benchmark.

### Bounded admission, design and terminal gates

0. **Read-only exact-graph witness, after active RAD06 lanes finish:** reuse QFP13's existing graph/rocprofv3 and 1319/1320/1325 attribution. For 8K and 80K Flash-Next, decode width 1 and MTP verify widths 2–5, capture physical gfx1100/gfx1201 rank, exact ordered GGML ops, tensor pointer/shape/strides/dtype, producer and `other_src` identity, consumer count, actual `unary_gated` versus separate UNARY/MUL launch count, active fusion branch, elapsed exposed launch gap and output bytes. Disqualify canonical GLU, HC gate, unrelated unary/binary adjacency, multi-consumer intermediates, broadcast/strided inputs, graph-split/alias ambiguity and unsupported types. No new telemetry subsystem or competing GPU queue.
1. **Stop before patch** with `NO_LITERAL_CHAIN`, `ALREADY_NATIVE_FUSED`, `ALIAS_UNSAFE`, `NO_VECTOR_ROUTE` or `BELOW_EXPOSED_CEILING` if no exact legal chain or the optimistic serialized saving cannot reach QFP13's >=0.5% E2E or >=0.10 ms/token serial-gap gate. At a 78.5 tok/s baseline, >=0.5% would require ~0.064 ms/token, i.e. ~19 fully exposed 3.5 us launches/token before costs; the 26-neighbourhood count is not a qualifying numerator.
2. **Only on a proven route:** extend `ggml_cuda_try_fuse` and the existing MMVQ/MMVF fusion descriptor/epilogue **in the existing backend**, not a second kernel family. Require a single GEMV producer, same-device/same-shape contiguous F32 external operand, one downstream `UNARY(SILU|SIGMOID|SOFTPLUS)->MUL` consumer, exact operand order, no alias or other live consumer, and a contiguous safely skipped span. Pseudocode: `if (exact_triple && one_consumer && same_device && alias_safe && vector_route && exact_semantics) vector_gemv_with_external_operand_epilogue(); else native_gemv_then_unary_mul();`. If adding a pointer/op field is unavoidable, gate its load only in the admitted specialization and measure VGPR/LDS/spill/occupancy and wave32 launch geometry; retain native fallback, including gfx1030. Never reuse `fusion.gate` as an external activation pointer.
3. **Cheapest implementation discriminator first:** a source-shaped host matcher with positive SILU/SIGMOID/SOFTPLUS and negatives for reversed operands, F16/BF16, broadcasts, noncontiguous/view offsets, aliases, multiple consumers, unrelated/interleaved nodes, `MUL_MAT_ID` and graph rebind. Compare F32 reference outputs and source-order rounding, including near-ties/NaNs. If any alias or numerical gate fails, remove the disposable variant.
4. **Only then AMD qualification:** isolated gfx1100/gfx1201 (gfx1030 control), same-pin default-off A/B, 8K/80K, no-MTP and fixed-draft MTP1/4/5, multi-request same-process, alternating ubatches, graph capture/rebind, full logits/KLD/greedy/MTP acceptance, exact GEMV rows/bytes/work and no disappearing kernels other than the intended pointwise launch. Measure critical-rank E2E and compiled VGPR/SGPR/LDS/spills, wave32/64 applicability and physical bandwidth. Four independent sessions/architecture, >=10 paired ABBA rounds; promote only >=0.5% E2E with positive CI95-low or >=0.10 ms/token measured serial gap with no E2E regression, <=1% unaffected-control regression and zero correctness/lifetime failures. Otherwise `NO_GAIN`/ `NUMERIC_DRIFT` and retain stock.

**Upstream/fork decision:** pinned and inspected upstream master already provide canonical GLU and unary+MUL fusions, but not the literal producer-to-pointwise epilogue. [llama.cpp #27194](https://github.com/ggml-org/llama.cpp/issues/27194) concerns Vulkan unary+MUL parity, **not** a HIP GEMV epilogue. [P100 fork patch 20](https://github.com/shinbunbun/llama-cpp-p100-patches/blob/main/docs/patches.md) reports an external +0.72% for a **different** ADD→UNARY→MUL fusion but also documents allocator-alias nondeterminism and leaves it default-off; use its failure mode as a negative alias test, not as AMD performance evidence or code to port. vLLM activation-quant fusion is not equivalent to this GGUF GEMV epilogue. Existing PRBE38/QFP13 ownership remains authoritative; PRBE37 canonical GLU, PRBE39/1206 ADD, PRBE40/1205 dual projections and active RAD06 are untouched.
