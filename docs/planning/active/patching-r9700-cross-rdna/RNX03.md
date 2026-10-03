---
id: RNX03
order: 3
plan: patching-r9700-cross-rdna
state: pending
created-at: '2026-10-03T01:32:55.746545+00:00'
breadth: ''
skill: advanced
created-by: codex
work: L
priority: P0
---

# R9X03 — GDN MTP-decode and short-prefill fusion

## Description

Port only novel GDN decode/MTP-tail and short-prefill fusion ideas; preserve the existing long-prefill chunked recurrence and state semantics.

## Steps

- Map r9k_gdn_decode_mtp q/k/v/g/beta/state/output semantics to gated_delta_net.cu and the Qwen4Exp graph.
- Census plain decode and MTP launches, including rollback and state snapshots.
- Implement 1302 for decode/MTP-tail recurrence and 1312 only for measured short-prefill ranges; keep them independently selectable.
- Benchmark sequential versus chunked baselines across pp1/2/4/8/16/32/64/128/256 and MTP-tail shapes, selecting thresholds per architecture.
- Preserve fp32 fallback and add independent disable and trace markers.

## Detailed Solution & Technical Design

Treat this as extension and verification of the existing 1253_nro04_gfx1100_bf16_chunked_gdn and 1254_nro05_gdn_mtp_prefix_tail owners. Do not create a replacement generic MTP/GDN package. First validate coverage and identify only residual decode-tail or short-prefill cases not owned by those packages. Keep the gfx103x overlap with the existing FP32 chunked owner explicit.

## Code Samples & Guidance



## Files

- kernels/r9k_gdn.hip
- r9700_vllm/models/gdn.py
- tests/test_gdn_decode_r9k.py
- tests/test_gdn_prefill_r9k.py
- ggml/src/ggml-cuda/gated_delta_net.cu
- src/models/qwen4exp.cpp

## Validation

Backend parity across reset/continuation, multiple sequences, MTP accept/reject, K boundaries, and threshold-adjacent lengths; validate recurrent state and outputs, tensor split, mixed gfx11+gfx12, launch count, E2E decode/MTP/TTFT.

## Effort & Risk



## Standards

Do not replace long-prefix chunking; separate gfx11/gfx12 kernels where fragment layout differs; do not software-emulate gfx12 WMMA on gfx103x.

## Acceptance Criteria

- 1302 and 1312 can be promoted independently.
- If 1253/1254 already subsume a regime, record already-owned rather than duplicating it.
- Unsupported shapes and architectures take the prior path unchanged.

## Notes

Original source alias is R9X03. Existing owners: 1221, 1253, 1254, 1255. Proposed slots 1302 and 1312.

Verbatim legacy source retained during R9X→RNX migration:

# R9X03 — GDN MTP-decode and short-prefill fusion

Status: planned
Proposed patches: `1302_r9x_gdn_mtp_decode_fusion`, `1312_r9x_gdn_short_prefill_fusion`
Depends on: R9X01
External source: `kernels/r9k_gdn.hip`, `r9700_vllm/models/gdn.py`; tests `tests/test_gdn_decode_r9k.py`, `tests/test_gdn_prefill_r9k.py`
Existing BigCherry owners: `1221_rd50_gdn_chunked_recurrence`, `1253_nro04_gfx1100_bf16_chunked_gdn`, `1254_nro05_gdn_mtp_prefix_tail`, `1255_nro06_adaptive_mtp_depth`
Primary llama.cpp target: `ggml/src/ggml-cuda/gated_delta_net.cu`; graph/shape checks in `src/models/qwen4exp.cpp` only if existing `GGML_OP_GATED_DELTA_NET` dispatch cannot express the specialization.



GPT design pass (req_83e7cdc000be4b9e): 1253 is validated and 1254 already implements the K>1 single-sequence MTP BF16 prefix plus exact sequential tail route for RDNA3/RDNA4. Next action is coverage validation, not new code.

## Goal

Port the two r9700 GDN ideas that are not already represented by BigCherry's long-prefill chunked recurrence: (1) one-launch speculative/MTP decode recurrence + state update + gated normalization, and (2) a one-launch sequential recurrence for short prefills where launch overhead dominates chunked math. Keep these as separate acceptance units.

## Exact source-to-target mapping

Source symbol of first interest: `r9k_gdn_decode_mtp` in `kernels/r9k_gdn.hip`; the vLLM graph replacement/shape contract is in `r9700_vllm/models/gdn.py`. Map its q/k/v/g/beta/state/output contract against the existing launcher and kernels in `ggml/src/ggml-cuda/gated_delta_net.cu`.

`1221` already owns a chunked recurrence in that file; `1253` creates the gfx11/gfx12 BF16 chunked files and selector; `1254` owns MTP long-prefix + exact sequential tail; `1255` owns adaptive draft depth. Therefore 1302 must target the **decode/MTP tail recurrence**, not replace long-prefix chunking, and 1312 must target only the measured short-prefill range.

Implementation package may create `ggml/src/ggml-cuda/gated_delta_net_decode_fused.cu/.cuh` and `gated_delta_net_short_fused.cu/.cuh` if keeping the code in `gated_delta_net.cu` would make architecture separation unsafe; dispatch remains anchored in `gated_delta_net.cu`. Follow 1253's precedent for separate gfx11/gfx12 kernels rather than sharing fragment-layout code that differs by generation.

## 1302 — MTP/decode fusion

1. Census GDN launches for plain decode and MTP depth 1..N. Record conv/gate/recurrence/state/norm boundaries and bytes.
2. Match r9k's fused recurrence to llama.cpp state-slot semantics, especially multiple MTP snapshots and rejected-candidate rollback.
3. Fuse only operations whose input/output lifetime is internal to the existing GDN op. Do not change model graph semantics merely to mirror vLLM.
4. Gate exact S_v/head/K/sequence shapes and add disable + `BIGCHERRY_PATCH_HIT patch=1302_r9x_gdn_mtp_decode_fusion`.
5. Keep fp32 reference/fallback reachable for every unsupported shape/arch.

## 1312 — short-prefill fusion

1. Benchmark baseline sequential vs 1253/1254 chunked vs candidate at pp1/2/4/8/16/32/64/128/256 and MTP-tail shapes.
2. Select threshold independently per architecture if crossover differs materially; do not copy r9700-stack's threshold.
3. Never divert long prefill away from 1253/1254 unless direct A/B proves the new path wins.
4. Add its own disable/trace marker so 1302 and 1312 can be bisected independently.

## RDNA adaptation

- RDNA4/gfx12: r9k scheduling is the reference; use native wave32/gfx12 primitives when they materially help.
- RDNA3/gfx11: P0 target because production includes gfx1100. Reuse recurrence/state locality, retune registers/LDS and preserve 1253's separate gfx11 BF16 path. gfx115x is a distinct validation row.
- RDNA2/gfx103x: concept-port launch fusion/state locality using supported vector/BF16/FP16 operations. Do not software-emulate gfx12 WMMA. If register pressure destroys occupancy, remain on baseline.

## Validation/acceptance

Backend-op parity across reset/continuation, multiple sequences, MTP accept/reject, K boundaries, and adversarial lengths around every selector threshold. Validate recurrent state and outputs, not only final tokens. Run tensor-split and mixed gfx11+gfx12. Profile launch count and E2E decode/MTP/TTFT.

Promote 1302 and 1312 independently. If the source behavior is already subsumed by 1253/1254 for a regime, record `already-owned` instead of duplicating it.

## Change Log

- 2026-10-03T01:32:55.746545+00:00 (created-by): Created by codex
- 2026-10-03T01:38:35.029997+00:00 (updated-by): Updated: section:notes
- 2026-10-03T02:18:10.838676+00:00 (updated-by): Updated: section:detailed_solution, section:notes

## Reviews

- RV4208
