---
id: PNRO02
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:52:07.096423+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P0
---

# Fuse residual ADD into internal AllReduce completion

## 2026-10-14 source-verified disposition (BCOP212)
**1250 cannot implement current exact-F32 decode fusion.** This section supersedes historical PNRO02 prose asserting that it can. In `engines/llamacpp/patches/1250_nro01_allreduce_q8_wire/patch.py::ggml_cuda_ar_allreduce_fused_add`, the only successful path requires `p->n_devices == 2`, `wire_override == q8_0` and `q8_nbytes >= p->copy_threshold`. Meta's optional `comm_allreduce_fused_add` callback suppresses the following ADD only after success; no exact-F32 implementation exists. 1250 depends on rejected 1252 P2P and untested 1272 Q8 wire. Its host-staging Q8 fallback exists but is not production-qualified. Deployed 1291 CPU-root is a distinct three-rank provider with no 1250 fused hook.

**Concrete admission:** pinned b11552 `ggml/src/ggml-cuda/allreduce.cu` sets `GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT=1 MiB` (https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278/ggml/src/ggml-cuda/allreduce.cu). Q8_0 stores 34 bytes/32 elements. At Qwen4Exp width 5120, one-token AR = 5,440 Q8 bytes; six-token MTP verify = 32,640; 192 tokens = 1,044,480; **193 tokens = 1,049,920**, first reaching the threshold. Current plain decode/MTP therefore cannot activate the fused hook at default settings even with hypothetical dual-GPU Q8. gfx1030 drafter is not an AR rank. Do not assume P2P for the gfx1100/gfx1201 topology. Lowering the threshold creates a separate lossy-wire experiment, not exact-F32 fusion.

**Ownership:** PNRO02 owns only conditional future exact-F32 residual epilogue; PGC11/1272 wire; QFP01/1291 CPU-root provider, epochs and buffers; QFP19 historical rebase; QFP11/PGC12 event attribution. No new dispatcher, communicator, cache, allocator, scheduler, codec or telemetry. Preserve 1250 evaluated/not adopted, 1252 rejected, 1272 untested. The measured ~3.9% dual-XTX exact-F32 adaptive gain belongs to 0840, not fused ADD. No new GPU performance result.

**Bounded next gate:** inspect actual composed Meta graphs at 10K/80K/160K for executed `AllReduce -> optional RESHAPE -> single-use mirrored F32 ADD`, both operand orders, correct use counts and no aliasing. Correlate per-rank same-epoch ADD kernels, serial critical-path time and physical traffic with existing QFP11/PGC12 evidence; counting theoretical ADD launches is insufficient. `NO_MATCHED_ADD` if none; `BELOW_EXPOSED_GATE` if optimistic end-to-end saving <3%. No hardware implementation is queued.

**Conditional design only after gate:** extend existing 1291 `bc_cpu_root_consume` small-message epilogue, not 1250 Q8. `if (all-rank CPU-root generation complete && F32 equal-shape contiguous mirrored single-use ADD && residual/output lifetimes and nonaliasing proven) write reduce+residual to output and suppress exactly the matched ADD; else use unchanged CPU-root/RCCL then ADD`. No partial-success skip; preserve stream/event dependencies, inactive ranks, multi-request/ubatch, MTP, long-context and graph replay. Test bit-identical F32 output or explicitly approved numeric tolerance (FMA/reassociation), logits/KLD/greedy/MTP acceptance and full work accounting. Promotion: CI95-low >=3% E2E decode, <=1% unrelated regression; otherwise `NUMERIC_OR_LIFETIME_FAIL` or `NO_VALIDATED_GAIN`.

**External reference:** vLLM's small-token AllReduce+residual+RMSNorm fusion has strict hardware/fallback gates (https://github.com/vllm-project/vllm/blob/main/docs/design/fusions.md and https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/allreduce_rms_fusion.py); NVIDIA FlashInfer SM90/SM100 performance is not transferable to AMD RDNA3/4. Source predicates 12/12 and host-only admission tests 10/10 passed; no HIP compilation, GPU test or new benchmark.


## Description

TODO, NOT-READY (rescoped). Verified via patch.py: 1251 only adds residual-capable kernel variants -- there is no reduction->RESHAPE->ADD graph matcher, no residual-carrying comm API, and no ADD elision anywhere in the package. The item's prior claim that it depends on PNRO01 'finishing the abstraction' does not match 1250's actual scope (1250 implements neither a finish abstraction nor anything PNRO02 currently calls) -- treat PNRO01 as required only for Q8-mode qualification, not for the exact-FP32 fusion path.

## Steps

1. Add the matcher at the Meta AllReduce boundary in ggml/src/ggml-backend-meta.cpp::ggml_backend_meta_graph_compute (verify this exact function name/location at implementation time) that detects reduction -> optional reshape-only chain -> single mirrored ADD.
2. Extend the AllReduce comm entry point (`ggml_backend_comm_allreduce_tensor_t` or equivalent -- verify exact type name) and the CUDA comm implementation to accept and carry a residual tensor plus destination, so the fused kernel variants 1251 already added can actually be invoked.
3. Elide the following ADD only after the fused AllReduce call returns success; on any failure, execute AllReduce then ADD unchanged (no partial state).
4. Make exact-FP32 fusion INDEPENDENT of Q8 -- require patch 1250 (PNRO01) only when qualifying the Q8 wire-mode combination, not for the base exact-FP32 fusion path.
5. Require same shape/type, mirrored residual/output, one consumer, and provider fused-entry support.
6. Cover both operand positions, use-count/aliasing, extra consumers, and provider fallback.
7. Test exact-FP32 first; test BF16/Q8 only under their own representation policies.
8. Measure kernel count, eliminated ADD launch, finish time, graph submission, and decode/prefill.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

patches/1251_nro02_allreduce_fused_residual; Meta matcher; AllReduce finish API; static graph/GPU tests; evidence

## Validation

Patch mechanics: `PYTHONPATH=tools python -m bigcherry patch-lint patches/1251_nro02_allreduce_fused_residual`; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay 1251_nro02_allreduce_fused_residual --source bigcherry-tuning`; package pytest offline. Positive direct/reshape-ADD fixtures; negative wrong-wiring/extra-consumer/non-mirrored/shape-mismatch/provider-failure/second-apply fixtures. Fused/unfused output equivalence per wire mode. Hardware (dev only): `python -m bigcherry.patch.validation_campaign --overlay 1251_nro02_allreduce_fused_residual --requires 1250_nro01_allreduce_q8_wire --arch gfx1100` measuring kernel count/eliminated ADD launch/finish time/graph submission/decode/prefill, with non-target regression gate.

## Effort & Risk



## Standards

Exact graph matching; one-consumer proof; fallback preservation; dependency-aware A/B; no epilogue generalization before this pattern is proven.

## Acceptance Criteria

Fusion activates only for the exact safe graph pattern, all negative fixtures retain ordinary ADD, correctness passes per representation, and performance evidence is causal. PNRO02 remains independently disableable from PNRO01 Q8 selection.

## Notes

Supersedes: NRO02
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro02

Supersedes: NRO02
Inherited semantic scope: preserve graph ownership/use-count, mirrored split, failure clearing, and independent-wire-mode qualification.
Migration: capability-rebaseline-v3-2026-09

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH. patches/1251_nro02_allreduce_fused_residual exists, state=untested. Depends on PNRO01's finish abstraction (patch 1250). No upstream equivalent (internal AllReduce fusion is BigCherry-internal). Disposition: validate/qualify existing patch; no GPT design needed.

2026-09-24 GPT review req_215c89d0b13a4bb7 applied: verified 1251 only adds residual-capable kernel variants with no matcher/comm-API/elision wiring at all. Added the required matcher location (ggml-backend-meta.cpp::ggml_backend_meta_graph_compute) and comm-API extension steps; decoupled exact-FP32 fusion from PNRO01 (only needed for Q8-mode qualification, not the base path).

2026-09-25: IMPLEMENTED inside patches/1250 (e06dcf63 is atomic: meta-backend reshape->ADD matcher, fused_add comm API, fused finish kernels). Opt-in GGML_CUDA_AR_FUSED_RESIDUAL; marker patch=1250_nro02 logged only when the fused path succeeds. 1251 scaffold removed. Must be exact vs unfused; measured as its own arm, separate from Q8.

2026-10-04 consolidation: owner item for residual-ADD fusion into AllReduce. PGC06 (--allreduce-fuse residual CLI over 1250/1251) is merged here and superseded: the CLI exposure is a sub-step once a real reduction->RESHAPE->ADD matcher and residual-carrying comm API exist. Flash-Next angle (RV4214/RV4215): with CPU-root (1291, QFP01) the residual add can be folded into the result write; AITER fused_allreduce_rmsnorm is a reference for AR+residual+RMSNorm(+quant) epilogues. GPT estimate +1-2.5% decode.

## Change Log

- 2026-09-09T10:52:07.096423+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:36.096098+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.057911+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.695018+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:26:03.477332+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260910_022800_five-nasone-successor-plans-no_4030
- 2026-09-10T02:28:00.275437+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:26:00.678612+00:00 (updated-by): Updated: section:validation, section:notes
- 2026-09-24T04:48:02.098895+00:00 (updated-by): Updated: section:description, section:steps, section:notes
- chg_20260925_111345_real-ports-of-the-nasone-allre_4524
- 2026-09-25T11:13:51.471143+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-25T11:14:06.579185+00:00 (updated-by): Updated: section:notes
- 2026-10-03T15:21:45.220306+00:00 (updated-by): Updated: section:notes
