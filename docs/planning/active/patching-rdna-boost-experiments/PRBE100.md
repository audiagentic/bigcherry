---
id: PRBE100
order: 100
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-11T23:50:02.222039+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Block 08 SSM fused-kernel audit, decomposition, and target-state design

## Description

Audit the full stew675-rdna-boosts Block 08 commit and produce an implementation-ready decomposition for its three previously missing SSM kernels and supporting graph/cache changes. This is design and provenance work first; no port is authorized until the audit establishes exact behavior, dependencies, and patch boundaries.

## Steps

1. Read the complete 5efcd85f diff, including mmvq.cu, norm.cu/.cuh, unary.cu/.cuh, and all graph wiring. 2. Reconstruct each fused kernel's source graph, tensor layouts, launch geometry, architecture predicates, numerical contract, and fallback path. 3. Reconcile shared files against current BigCherry and tracked RD12/RD15/RD17/RD24/RD25/RD26/RD09 work. 4. Decompose Block 08 into package-sized patch candidates with explicit dependency/order and conflict maps. 5. Define code target state, test fixtures, correctness/performance evidence, provenance registration, and a go/no-go decision for porting. 6. Do not edit production source or create patch modules in this item.

## Detailed Solution & Technical Design

Source is the external stew675/llama.cpp rdna-boosts commit 5efcd85fb4cd8845c6c7dd47c50e2666264aa4eb, titled 'rdna-boosts: block 08: fused-core prefill kernels and GPU bit-identical', fetched into tools/lab/rd25-block08-review/block08.diff. The audit must treat the three missing kernels as a coordinated SSM fusion family: ssm_gate_beta_fused_q8_0 (two Q8 projections plus softplus/sigmoid gating), ssm_conv_l2_gatebeta_fused (conv+SiLU+Q/K normalization+V+gate/beta pre-scan), and shexp_down_gated_q8_0 (gated quantized down projection plus residual). Verify whether the shared Q8_1 cache, small-batch flash-attention selection, graph capture behavior, and auxiliary tensor fields are prerequisites or independent changes. Reuse the existing package-only patch model, graph recipe lifecycle, experiment contracts, correctness evidence, and BuildPlan identity; no speculative second executor or cache identity system.

## Code Samples & Guidance



## Files

tools/lab/rd25-block08-review/block08.diff; config/external-sources.toml; docs/planning/active/patching-rdna-boost-experiments/PRBE11.md; docs/planning/completed/patching-rdna-boost-experiments/PRBE99.md; current vendor ggml-cuda sources; patches/1205_rd12*; patches/1207_rd17*; patches/1235_rd09*; patches/12xx RD15/RD24/RD26; tools/bigcherry/patch/**; tools/bigcherry/campaign/**; tools/bigcherry/tuning/**; tools/tests/**

## Validation

Full diff read with line/file counts; source-to-current symbol inventory; graph/operator and tensor-layout reconstruction; exact overlap/conflict table for tracked patches; current-pin ancestry/source registration check; static anchor feasibility; proposed synthetic per-op and end-to-end correctness matrix; deterministic output/bit-identity policy; launch/fallback/architecture coverage; no production edits during audit.

## Effort & Risk



## Standards



## Acceptance Criteria

The complete Block 08 diff is reviewed and its three kernels plus supporting changes are decomposed into explicit patch-sized follow-ups. Each follow-up has exact source provenance, current-tree anchors or a fail-closed non-port decision, dependencies/conflicts, target APIs/data flow, correctness/performance tests, and evidence identity requirements. RD25 remains blocked or is explicitly re-scoped based on this result; no guessed anchors or unreviewed production port is introduced.

## Notes

Provenance: stew675/llama.cpp branch rdna-boosts, commit 5efcd85fb4cd8845c6c7dd47c50e2666264aa4eb; title and diff header are preserved in tools/lab/rd25-block08-review/block08.diff. Discovery was made during PRBE11's attempt to port RD25, and summarized in PRBE99. Next agent must obtain/read the complete upstream diff and compare it to current main; the existing lab diff is an audit input, not an implementation patch.

## Change Log

- 2026-09-11T23:50:02.222039+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260911_235011_added-a-provenance-backed-bloc_4570
- 2026-09-11T23:50:11.493148+00:00 (updated-by): Updated: section:ledger-events
