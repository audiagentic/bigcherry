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

PRBE05 is the canonical owner for Q8_1 activation reuse in the MMVQ path. Historical RD09 is closed/provenance-only; patch 1235 is an inconclusive/timed-out child experiment; patch 1307 is the current A/B child experiment. Neither child is a separate planning owner.

The optimization is valid only if the same logical F32 activation is quantized to Q8_1 for more than one MMVQ consumer while cache identity and lifetime remain valid. A high `quantize_q8_1 -> MMVQ` adjacency count by itself is not reuse evidence.

Verified b11126 integration seam: `ggml_cuda_mul_mat_vec_q` locally allocates `src1_q8_1`, calls `quantize_row_q8_1_cuda(...)`, and passes that freshly quantized pointer to `mul_mat_vec_q_switch_type(...)`. PRBE05 may cache/reuse that quantized activation only after Stage 0 proves real repeated-consumer reuse.

## Stage 0 — Prove Cacheable Reuse Before Optimizing

QFP13 measured roughly 154 `quantize_q8_1 -> mul_mat_vec_q` adjacencies/token in one production census. That is an opportunity count, **not** a predicted hit count: every MMVQ may still consume a distinct activation.

Before accepting any cache performance result, instrument and report for each graph generation/device/stream:
1. total Q8_1 quantization requests entering the exact MMVQ seam;
2. cache lookups, hits, misses, waits/contention, and fallbacks;
3. a stable activation identity for every lookup: generation + view root + exact data address/offset + shape/strides + stream (and any additional identity required by the real source path);
4. number of distinct activation identities/token;
5. consumers per activation identity and the distribution of identities with 1, 2, 3+ MMVQ consumers;
6. quantization launches actually removed/token on the cache-on arm;
7. cache bytes/entries and lifetime/capture mode for every hit.

Required proof: at least one material recurring class must show the **same activation identity** consumed by >1 MMVQ within the valid generation/lifetime, and the cache-on arm must remove corresponding quantization launches. If same-identity reuse is zero or too rare to meet the QFP13 performance gate, stop and park/supersede the cache optimization rather than engineering a larger cache.

The 1307 A/B result is not promotable without this Stage-0 hit/miss/reuse-ID report, even if wall time moves positively. Conversely, do not supersede PRBE05 while 1307 is still live/incomplete.

## Steps

1. Add Stage-0 instrumentation first. Establish actual activation reuse and removable quantization launches on the production Flash-Next decode path before treating the cache as a performance optimization.
2. Add a small context-owned Q8_1 cache API/state: key (generation, view-root pointer, exact view data address/offset, dimensions/strides, stream, plus any required identity fields), lookup/reserve/publish, hard byte/entry caps, and native fallback on miss/exhaustion.
3. Seam A: in `ggml_cuda_mul_mat_vec_q` (`mmvq.cu`, b11126 around 1421), replace only the local `src1_q8_1` allocation + `quantize_row_q8_1_cuda(...)` block (around 1506) with cache lookup/reserve. On hit, feed the cached pointer to `mul_mat_vec_q_switch_type(...)` unchanged; on miss, quantize exactly as native and publish only after completion.
4. Add cache ownership/state to the existing `ggml_backend_cuda_context` in `ggml/src/ggml-cuda/common.cuh`. BigCherry's patcher should edit existing files rather than invent a separate cache compilation unit for this experiment.
5. Seam B: begin a new cache generation in `ggml_backend_cuda_graph_compute` (`ggml-cuda.cu`, b11126 around 4420). Never grow/reallocate cache storage or perform synchronous verification while active graph capture is recording. Cached addresses referenced by captured graphs must remain stable for the graph/graph-key lifetime, or caching must be disabled for that capture/replay path.
6. Keep an env gate default-OFF (for example `BIGCHERRY_Q81_CACHE=1`) so native behavior remains the control. Provide a separate verify mode that independently re-quantizes and byte-compares against the cached Q8_1 value outside unsafe capture windows.
7. Expose hit/miss/reuse/wait/timeout/contention/capacity counters through `BIGCHERRY_PATCH_TRACE`-gated logging, including the Stage-0 consumers-per-identity summary.
8. Add correctness coverage using `GGML_OP_MUL_MAT` with quantized src0 + F32 src1 shapes that dispatch to MMVQ. There is no literal `MUL_MAT_VEC_Q` graph op to target.
9. Run adversarial correctness before performance claims: same-tensor hit; offset-collision miss; shape/stride/stream miss; generation/pointer-reuse miss; capacity fallback; dual-GPU isolation; independent re-quantize/byte-compare; graph capture/replay pointer stability.
10. Only after Stage 0 + correctness pass, run cache-off/cache-on/verify ABBA and report causal launch reduction, memory use, ms/step, and TG.

## Detailed Solution & Technical Design

Context-owned, bounded cache with generation invalidation and stable non-relocating addresses. Cache identity must include the exact view data address/offset and graph generation, not just a root tensor pointer. Default-off preserves native behavior; on/verify are explicit experiment arms.

Corrected b11126 call path: `ggml_cuda_mul_mat_vec_q` is self-contained. It does **not** call `ggml_cuda_op_mul_mat_vec_q`; it quantizes src1 itself with `quantize_row_q8_1_cuda(...)` and dispatches directly via `mul_mat_vec_q_switch_type(...)`. Therefore the exact local quantization block is the reuse seam.

Concurrency/lifetime requirements:
- publish only after the quantization result is ready for consumers on the relevant stream;
- do not return a pointer whose storage can be recycled while a live captured graph references it;
- generation rollover must make stale identities unreachable;
- capacity exhaustion/contention/timeouts fall back to native quantization rather than blocking correctness;
- device/context ownership is isolated; no cross-device pointer reuse.

A cache hit is counted only when native quantization would otherwise have executed for the same valid activation identity. Merely finding an entry with a similar shape or adjacent producer is not a hit.

## Code Samples & Guidance

Verified planning anchors in llama.cpp b11126:
- `ggml/src/ggml-cuda/mmvq.cu`: `ggml_cuda_mul_mat_vec_q(...)` around 1421;
- `ggml/src/ggml-cuda/mmvq.cu`: local `quantize_row_q8_1_cuda(...)` around 1506 — exact cache lookup/publish seam;
- `ggml/src/ggml-cuda/mmvq.cu`: `mul_mat_vec_q_switch_type(...)` around 1531 — consumes the native or cached Q8_1 pointer;
- `ggml/src/ggml-cuda/ggml-cuda.cu`: `ggml_backend_cuda_graph_compute(...)` around 4420 — generation/capture-lifecycle seam;
- `ggml/src/ggml-cuda/common.cuh`: existing `ggml_backend_cuda_context` — cache state owner.

Patch implementation should stay anchor-based in existing source files. Do not create `hip-q81-cache.{h,cu}` solely for this experiment unless the patching architecture changes and a separate design review justifies it.

## Files

Expected production edit anchors:
- `ggml/src/ggml-cuda/mmvq.cu` — Stage-0 instrumentation, lookup/miss/native-quantize/publish path;
- `ggml/src/ggml-cuda/common.cuh` — bounded context-owned cache state;
- `ggml/src/ggml-cuda/ggml-cuda.cu` — graph generation/capture lifecycle and pointer-lifetime gating;
- existing BigCherry patch package for the current child experiment (1307 while active), plus focused tests/campaign evidence.

No new CUDA/HIP source/header file is required for the current design.

## Validation

Offline:
- `PYTHONPATH=tools python -m bigcherry patch-lint`;
- `patch-rebase-check` for the active patch overlay/source;
- unit/targeted tests for hit identity, offset collision, shape/stride/stream separation, generation rollover, capacity fallback, and publish/readiness behavior.

Hardware / production:
- Stage-0 hit/miss/reuse identity report on gfx1100 and gfx1201;
- dual-GPU/context isolation;
- graph warm-up/capture/replay stability;
- cache off/on/verify ABBA on shallow and deep production contexts;
- total `quantize_q8_1` launches/token and MMVQ launches/token;
- cache entries/bytes and hit-rate by consumers-per-identity class;
- ms/step, effective TG, and deterministic output parity.

A wall-time improvement without corresponding proven same-identity hits and removed quantization launches is not causal acceptance evidence.

## Effort & Risk

L/advanced. The largest correctness risks are false identity matches, stale pointers, publish-before-ready races, and capture-time storage relocation. The largest performance risk is that the production graph has little/no same-activation reuse, making the cache pure overhead. Stage 0 exists to reject that case early.

## Standards

- Evidence-first reuse proof.
- Bounded cache and stable ownership.
- Exact activation identity; no shape-only matching.
- Generation safety and no stale/cross-device pointers.
- No graph-time unsafe allocation/reallocation.
- Native fallback on every uncertain/error/capacity path.
- Default-off experiment until causal launch + performance evidence passes.

## Acceptance Criteria

- Stage 0 proves material repeated MMVQ consumers of the same valid activation identity.
- Cache-on removes the predicted corresponding `quantize_q8_1` launches; hit count and launch reduction reconcile.
- Independent re-quantize/byte-compare reports zero Q8 block mismatches.
- Offset/shape/stride/stream/generation/adversarial cases never false-hit.
- Capture/replay has stable pointer lifetime and no cache growth/reallocation hazard.
- Off mode remains native; miss/exhaustion/contention paths fall back safely.
- The QFP13 end-to-end gate is met with causal evidence. If reuse is absent or performance is negative/noisy after a valid A/B, park/supersede PRBE05 rather than broadening the cache.

## Notes

Supersedes: RD09
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd09

RD09 remains closed historical provenance. PRBE05 is the only actionable Q8_1 activation-cache owner. Patch 1235 is historical/inconclusive; patch 1307 is the active child A/B on `patch-refactor` and must satisfy Stage 0 before promotion. PRBE06/PRBE18 may consume proven PRBE05 capability but do not duplicate its cache identity/lifetime work.

2026-09-24 review corrected the integration seam: `ggml_cuda_mul_mat_vec_q` locally quantizes src1 and does not call `ggml_cuda_op_mul_mat_vec_q`. It also moved ownership into existing `ggml_backend_cuda_context` because the current BigCherry patcher is anchor-based on existing files.

2026-10-04 QFP13 census correction: ~154 `quantize_q8_1 -> MMVQ` adjacencies/token is not a cache-hit forecast. Actual same-identity consumers, hits/misses, and removed launches are now mandatory Stage-0 evidence.

## Change Log

- 2026-09-09T10:53:47.797556+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-24: Corrected MMVQ call-path seam and graph-capture/cache ownership design.
- 2026-10-04 (agent): Consolidated 1235/1307 under PRBE05; added Stage-0 activation-identity reuse proof; corrected stale new-file guidance; required causal hit-to-launch-reduction reconciliation.

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.147212+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- chg_20260910_023232_the-next-three_rdna-successors_5807
- chg_20260911_221341_documented-the-real-status-of_8365
- chg_20260912_095506_cleaned-the-active-rdna-boost_4906
