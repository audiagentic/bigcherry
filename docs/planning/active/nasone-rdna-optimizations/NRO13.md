---
id: NRO13
order: 13
plan: nasone-rdna-optimizations
state: pending
created-at: '2026-09-08T09:50:40+10:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Qwen4exp gather-based sparse QSA decode

## Description

Track nasone commit `d2d89512d05a8142e7574f3c18262e50e65eea5f`, sourced from ggml-org/llama.cpp PR #28213. The change converts Qwen4exp QSA decode from masking/scanning the full KV cache to gathering only selected cells and attending densely over the gathered subset once context is sufficiently larger than the padded QSA width.

This is upstream-derived. Before any BigCherry patch is authored, resolve whether PR #28213 or an equivalent commit is ancestral to the active pin. If yes, treat it as baseline coverage/evidence, not a local backport.

## Steps

1. Resolve upstream PR/merge/head and `ancestral_to_pin` against the current and candidate pin.
2. If absent, port as an upstream-backport with retirement condition tied to #28213/equivalent.
3. Preserve the source activation predicate: QSA active, decode-like one-token-per-stream, `n_kv >= 4*width`, runtime escape hatch.
4. Gather K, V, and per-cell bias using identical selected indices; preserve rotation/cache semantics.
5. Validate padded top-k width and -inf masking so surplus gathered cells cannot affect attention.
6. Sweep context across below/at/above gather threshold and multiple compression ratios.
7. Compare gathered versus masked full-cache attention output and model behavior.
8. Measure KV bytes read, gather cost, FA cost, and total decode time.

## Detailed Solution & Technical Design

The asymptotic target is important: ordinary masked QSA still makes attention touch `n_kv` cells, while gathered QSA bounds attention by approximately `indexer_top_k + ratio` padded to FA requirements. The gather adds explicit GET_ROWS/dequant/cast work, so a crossover threshold is required.

Bias values carry visibility for selected cells. The gathered mask must clamp them to valid attention-mask range and preserve stream/position semantics. Single-token-per-stream is a hard initial restriction.

## Code Samples & Guidance

Upstream ancestry check precedes patch authoring. Keep `QWEN4EXP_QSA_GATHER=0` or equivalent rollback while experimental.

## Files

Planning-only until ancestry check. Likely source surfaces: Qwen4exp graph construction/model declarations and graph-input mask setup.

## Validation

Exact/near-exact attention output comparison, selected-index equality, context threshold boundary, multi-stream decode, quantized KV variants, long-context performance and memory traffic.

## Effort & Risk

High graph/attention correctness risk, potentially high long-context reward.

## Standards

Upstream-before-backport check; attention correctness first; explicit crossover selector; no duplicate patch if baseline already contains equivalent.

## Acceptance Criteria

- Correctness matches masked QSA over threshold matrix.
- Nonqualifying contexts execute original path.
- Long-context decode establishes positive effect after gather overhead.
- Patch is retired immediately when equivalent becomes baseline.

## Notes

This is model-specific Qwen4exp work and should not be generalized to ordinary attention without a new experiment.

## Change Log

- 2026-09-08T09:50:40+10:00 (created-by): Created from nasone/upstream #28213; P1.

## Ledger-events

- Pending: ag-ledger MCP unavailable in authoring session.
