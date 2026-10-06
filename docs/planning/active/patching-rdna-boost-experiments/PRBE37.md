---
id: PRBE37
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:01.682197+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-FUS-001: Canonical GLU GEMV epilogue coverage gaps

## Description

PRBE37 is the canonical owner for **canonical `GGML_OP_GLU` epilogue capability/gaps**. It is not the owner for literal `UNARY -> MUL` graphs; that distinct topology belongs to PRBE38.

Verified against llama.cpp b11126: supported canonical GLU vector graphs are already fused natively. `ggml_cuda_should_fuse_mul_mat(...)` recognizes the eligible gate/up/`GGML_OP_GLU` shape, populates `ggml_cuda_mm_fusion_args_host/_device`, and the `mul_mat_vec_q/f` vector kernels consume that fusion data so the supported GLU operation executes in the GEMV epilogue.

The remaining PRBE37 question is narrower: whether production target models use a canonical GLU operation or non-vector dispatch path that current upstream does **not** cover. At b11126, the validated vector allow-list is SWIGLU/GEGLU-family (`SWIGLU`, `GEGLU`, `SWIGLU_OAI`, `SWIGLU_CLAMP`); the prior plan's assumption that SIGMOID and general GEMM/prefill were already covered was false. That does not by itself justify a patch: actual target-model graph usage must be proven first.

## Ownership Boundary

PRBE37 owns:
- current-upstream coverage of canonical `GGML_OP_GLU` graph forms;
- proving whether an unsupported canonical GLU op is actually emitted by target models;
- proving whether a target-model canonical GLU path selects GEMM/prefill rather than the already-fused vector GEMV path;
- a narrowly scoped extension only after that production gap is demonstrated.

PRBE37 does **not** own:
- literal `MUL_MAT[/ID] -> UNARY(SILU|SIGMOID|SOFTPLUS) -> MUL` graphs (PRBE38);
- launch-frequency ranking (QFP13);
- generic new GEMM epilogue architecture without a target-model trace proving need.

## Stage 0 — Prove A Target-Model Coverage Gap

1. Re-read the current pinned `ggml_cuda_should_fuse_mul_mat(...)` and every accepted `GGML_GLU_OP_*` value; record the exact canonical GLU allow-list and vector dispatch constraints.
2. Trace/inspect the production target model graph and identify every canonical `GGML_OP_GLU` op actually emitted during the relevant decode/prefill paths.
3. Classify each real occurrence as:
   - already fused vector GLU: no PRBE37 work;
   - unsupported canonical GLU op on vector path: candidate narrow allow-list/semantic extension;
   - canonical GLU on GEMM/non-vector path: separate larger candidate requiring a design pass;
   - literal UNARY+MUL rather than `GGML_OP_GLU`: hand to PRBE38.
4. Record frequency/token or frequency/request and predicted removable launch/memory cost for each uncovered real occurrence.

Stop/close PRBE37 as no-op-needed if target models do not exercise a material unsupported canonical GLU path. Do not patch an unused theoretical gap.

## Implementation Rules

For an unsupported canonical GLU op on the existing vector path:
- extend the current native matcher/allow-list only after proving its numerical semantics match the existing fusion-data bias/scale/gate handling;
- reuse `ggml_cuda_mm_fusion_args_host/_device` and current `mul_mat_vec_q/f` epilogue machinery;
- retain fail-closed fallback for unsupported shapes/dtypes/ops.

For a real GEMM/prefill coverage gap:
- do not pretend the vector fusion descriptor/kernel can be wired in with a one-line allow-list change;
- first document the selected non-vector kernel family, epilogue/store point, graph topology, expected launch reduction, and numerical/output-layout constraints;
- only then decide whether that larger work remains PRBE37 or should be split into a new implementation item. No split is warranted until a real production gap is observed.

## 2026-10-07 audit: MMQ prefill fusion

Upstream draft #29948 provides the concrete non-vector design previously missing here: pair/interleave dense FFN up and gate rows and apply canonical GLU during MMQ write-back while retaining the existing fusion descriptor and graph allocation dependencies. Its current matcher is explicitly NVIDIA-only because the write-back depends on NVIDIA MMA accumulator layout. Published Qwen3.8-27B pp16384 gains are not RDNA evidence.

AMD gate: first attribute the exact dense up+gate+GLU MMQ topology at ubatch 512/1024/2048 on gfx1100/gfx1201. Continue only if removable GLU/materialization cost is at least 3% of representative prefill wall time or 5% of MMQ+GLU critical-path time. On failure, park and wait for upstream AMD support. On pass, reuse #29948's matcher/fusion descriptor/allocation-dependency design and port only the interleaved-loader/write-back mechanism. Prove the AMD MFMA/WMMA accumulator lane mapping with a focused backend-op fixture rather than copying NVIDIA indexing. Require supported GLU/quant boundary correctness, fusion-off comparison, Qwen3.8-27B output agreement, at least 3% repeated production-lane prefill gain, no primary control regression above 1%, and no material VGPR/spill/occupancy regression. Otherwise remove/park the prototype.

Ownership remains PRBE37. PRBE38 owns literal UNARY-to-MUL graphs; QFP13 owns decode launch ranking. Do not add another matcher, fusion descriptor, scheduler or dispatch table.

## Source Anchors

Verified b11126 anchors:
- `ggml/src/ggml-cuda/ggml-cuda.cu`: `ggml_cuda_should_fuse_mul_mat(...)`, `valid_glu_ops`, fusion-data population and vector dispatch sites;
- `ggml/src/ggml-cuda/mmvq.cu`: fused quantized vector epilogue;
- `ggml/src/ggml-cuda/mmvf.cu`: fused floating vector epilogue.

The already-native supported vector GLU path is evidence, not work to reimplement.

## Validation

Stage-0/source:
- exact upstream allow-list and dispatch coverage;
- target-model graph evidence for any claimed uncovered canonical GLU op/path;
- QFP13 frequency/launch evidence when the candidate affects decode launch count.

If a vector coverage extension is implemented:
- focused backend-op positive case for the newly accepted canonical GLU op;
- negative/fallback controls;
- deterministic output parity;
- launch-count and paired performance evidence on gfx1100/gfx1201.

If GEMM/prefill work is proven necessary, define its own correctness/perf matrix before implementation; do not reuse vector acceptance evidence.

## Acceptance Criteria

- No work is performed for canonical GLU vector cases already fused by upstream b11126/current pin.
- Any implemented extension corresponds to a production target-model graph that upstream currently leaves unfused.
- Numerical semantics for the newly accepted op/path are explicitly proven; unsupported cases fall back unchanged.
- PRBE38 remains the sole owner for literal UNARY->MUL graphs.
- A launch/memory/performance benefit is measured for the real target path; otherwise close/park the uncovered theoretical gap.

## Effort & Risk

S for Stage-0 coverage proof. M for a narrow vector-op extension. A real GEMM/prefill epilogue design may be L and must be re-estimated after the source/trace proof.

## Standards

- Do not reimplement upstream-supported canonical GLU fusion.
- Production usage before theoretical capability work.
- One owner per graph representation.
- Preserve fallback and numerical semantics.

## Notes

Supersedes: RD45
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd45

Historical 2026-09-24 closure as fully upstream-absorbed was too broad: it assumed SIGMOID and general GEMM coverage without proving either. The subsequent reopening correctly identified those possible gaps, but mixed them with already-covered behavior. This revision retains only the unresolved **canonical GLU** coverage question and explicitly routes literal `UNARY->MUL` to PRBE38.

Related: PRBE38 (literal unary-gated topology), QFP13 (launch ranking).

## Change Log

- 2026-09-09T10:56:01.682197+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-24: Reopened after source review disproved blanket SIGMOID/GEMM coverage.
- 2026-10-04 (agent): Removed contradictory upstream-absorbed/reopened instructions; retained PRBE37 only as the canonical GLU coverage-gap owner and separated literal UNARY->MUL ownership to PRBE38.

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.296126+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- chg_20260910_030005_amd-streamfus-successors-prbe_8761
- chg_20260924_023553_re-scoped-11-rdna-boost-planni_1625
- 2026-09-24T02:31:20.271685+00:00 (state-transition): State: pending -> superseded
- 2026-09-24T04:35:14.220144+00:00 (state-transition): State: superseded -> pending
