---
id: XDP02
order: 0
plan: run-xtx27-prefill
state: active
created-at: '2026-11-09T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Maximum resident KV: dual-XTX plus R9700 dense-stage feasibility

## Description

Owner update (2026-11-09): *maximise usable KV context* while improving
dual-XTX prefill speed. A future dense 27B may support 1M tokens; do not
mistake that expectation for a released or qualified Qwen4 checkpoint.
Use the local Qwen3.8-27B 64-layer geometry as a proxy, with native context
262,144 only. A proposed 1,048,576 context requires future model support
and separate end-to-end numerical/long-context quality validation.

Capacity and speed are **joint goals**, not serial gates: run independent
card-free three-card placement design now; qualify two-XTX kernel/prefill
while it proceeds. Implement a full-layer R9700 stage only after the
software routing/ownership and transfer microbenchmarks succeed.

## Quantified facts and candidate layouts

The local config has 64 transformer layers, 16 `full_attention` (at 3,
7, ..., 63), 48 `linear_attention` and 4 KV heads of width 256. Classical
K+V per token at fp8 is 2 * 16 * 4 * 256 = 32,768 bytes for the *whole*
model, or 32 GiB for 1,048,576 tokens. Per-layer GDN recurrent state
does not scale linearly per token the same way, but its per-session state,
convolution buffers and checkpoints **still count** toward a real plan.

| Model partition | Full-attention layers on XTX TP2 | FP8 attention KV / XTX at 1M | Full-attention layers on R9700 | FP8 attention KV / R9700 at 1M |
|---|---:|---:|---:|---:|
| 64 XTX, none R9700 | 16 | 16 GiB | 0 | 0 |
| 48 XTX + 16 R9700 | 12 | 12 GiB | 4 | 8 GiB |
| 40 XTX + 24 R9700 | 10 | 10 GiB | 6 | 12 GiB |
| 32 XTX + 32 R9700 | 8 | 8 GiB | 8 | 16 GiB |

The first table covers only *logical FP8 full-attention K+V* at one
session and one 1M context: TP2 rank ownership halves its own stage's
KV, while the single-card R9700 stage has full heads. The three-card
total remains 32 GiB; splitting does not make KV disappear. It spreads
that memory across 80 GiB nominal VRAM while also moving some resident
weights off the XTXs.

The measured all-64 TP2 static weight footprint was ~7.51 GiB per XTX
under the existing MXFP4 build. 16 GiB attention KV + those weights
already occupies ~23.51/24 GiB per XTX before buffers. A conservative
illustrative 2 GiB/card reserve makes it **not fit** on TP2. Stage-specific
weight residency, replicated norms/output head, tokenizer embedding,
activation buffers and architecture effects have *not* been measured.

`tools/lab/radiance/xtx27_kv_budget.py` implements exact layer-based
classical K+V arithmetic, optional stage cuts and explicitly **illustrative**
weights/reserve calculations. Its defaults of 7.51 GiB full TP2 rank
weights and 2 GiB reservation are **projections, not planner admission**.
It reports the declared trained context and flags requests beyond it.

## Steps

1. **Capacity envelope first:** calculate staged logical KV at 262K and
   hypothetical 1M for fp8 and bf16; when supported, measure compressed
   fp8_v4 and k6_v4 *actual* runtime allocations and quality. Capture
   single-session versus multi-session and resident versus host-tier
   semantics separately. Never count host/RAM spill as VRAM-resident KV.
2. **Place embedding according to context:** preserve RAM embedding as
   long-context default. VRAM embedding can be an *opt-in short-context
   speed experiment* only, and must surrender VRAM to KV under pressure.
   Do not assume the unused VRAM reported at 8K stays unused at 1M.
3. **Stage ownership proof:** prototype XTX prefix rank0/rank1 with
   head-sharded 2-rank KV/GDN state, plus R9700 full-width suffix with
   *its own* KV/GDN/state per session. Each token traverses stages in
   order; stage boundary activation is sent once at prefill and at
   decode, with explicit sequence, positions, stream fences and abort
   handling. A dense R9700 stage cannot use the existing remote-MoE
   expert helper protocol as-is.
4. **Minimum hardware proofs:** check stage-specific weight residency
   and allocator fit at each chosen cut (32/40/48), payload transfer
   size/latency over actual upstream PCIe links, host-copied vs mapped
   GPU memory, CPU coordination overhead, prefill chunk overlap, KV
   correctness and incremental prefix continuation.
5. **Capacity gates:** measure 32K, 64K and 128K, then 256K native
   Qwen3.8 with per-stage KV and GDN memory; future model only for 1M
   if its config/checkpoint genuinely supports it. Reserve model runtime
   and VRAM headroom, reject planner drift and OOM. Record the longest
   **resident** context with successful cold/warm prefill and decode.
6. **Performance gate:** compare same model, quantisation, TP wire,
   number of sessions and context against dual-XTX before/after,
   including TTFT, throughput at long context, decode ms/token,
   bandwidth, quality and reproducibility. A bigger cache is not
   automatically faster, and stage transfers can hurt low-latency
   decode. Balance placement cut from measurements, not just capacity
   math.

## Files

- `tools/lab/radiance/xtx27_kv_budget.py`
- `tools/tests/lab/test_xtx27_kv_budget.py`
- `tools/lab/radiance/xtx27_prefill.py` (separate measured speed lane)
- `docs/planning/active/run-xtx27-prefill/XDP01.md`

## Validation

Card-free 1M arithmetic, exact attention-layer split, invalid shapes,
head sharding, 262K-native warning; **no GPU stage exists yet**.
A runtime proof must show actual stage weight/VRAM budgets, KV ownership,
context fidelity and request admission.

## Acceptance Criteria

A qualified, safely reserved **maximum resident context** by layout,
quality, prefill and decode latency, including 1M only after model
availability; no silent host spill or missing recurrent state accounting.
The R9700 stage is reviewed for correctness and scheduling before any
production integration. No `vendor/radiance` direct edits, no push or PR.

## Notes

Existing Radiance heterogeneity is primarily remote-MoE expert
offload; this is a separate full-transformer-stage interface/design.
Link to HD2701 (heterogeneous dense workstream) in review; that plan is
in a different experimental worktree, so do not edit or merge it here.

## Change Log

- 2026-11-09: Added capacity-first owner constraint and card-free
  full-attention KV geometry and placement feasibility model.
