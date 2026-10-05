---
id: BCOP35
order: 35
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T21:49:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P3
work: L
---

# Evaluate context-parallel and distributed long-context attention

## Description

Investigate whether long-context inference can use multiple AMD GPUs by partitioning attention/KV work rather than only model layers. The production topology has no usable GPU P2P, so communication volume and synchronization are first-class gates. This item is experimental until a lower-bound model demonstrates feasibility.

## Steps

1. Define candidate decompositions: sequence/context shards, KV-cache shards, head shards and hybrid layer/context placement. Prefer upstream llama.cpp primitives if/when available.
2. Derive required communication for Q/K/V, attention statistics/output reductions and KV updates for decode and prefill separately.
3. Measure host-mediated transfer and collective controls on gfx1100/gfx1201; do not assume RCCL admission or benefit.
4. Build an offline trace/cost simulator using actual model dimensions, context lengths and batch sizes. Identify crossover regions where reduced per-device attention/KV work exceeds communication cost.
5. Prototype only a correctness path for the strongest candidate. Preserve exact attention semantics before considering reduced-communication approximations.
6. Test interaction with Flash Attention, sparse attention, quantized KV cache, MTP/speculation and RPL01 placement. Existing owners retain their local policy.
7. Remove the prototype if the no-P2P production topology cannot meet the end-to-end gate.

## Validation

- Short-context negative controls plus 32k/64k/128k/256k-class contexts where supported.
- Prefill and decode measured separately.
- Report KV bytes/device, bytes transferred/token, sync count, attention kernel time, idle time, VRAM and end-to-end PP/TG.
- Greedy/logit correctness contract before performance claims.

## Acceptance Criteria

- Communication lower bound and measured transport demonstrate a plausible >=5% end-to-end opportunity before permanent implementation.
- No assumption of NVLink/P2P is embedded in the design.
- Existing Flash Attention/KV/cache implementations are reused rather than forked without evidence.
- Failed experiments are removed or retained only as non-production research tooling.
