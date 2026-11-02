---
id: RHP14
order: 0
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-11-02T01:49:18.074244+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P3
---

# Parked: overlap helper round trips across MoE layers (cut fixed per-layer latency on short steps)

## Description

Each prefill or decode step crosses 48 MoE layers, each with a publish -> helper compute -> result round trip that the engine waits on before the next layer; on small steps (decode, short prompts) the fixed per-layer latency is a large share (6-stream decode: ~0.3-0.6 ms exposed helper wait per layer). Overlapping the next layer's non-MoE work, or pipelining two micro-batches so one's helper wait hides under the other's compute, would cut it. Parked until stage 1-3 placement shows what remains; recorded so the idea is not lost.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Interacts with the v23 placement epochs (one epoch per step) and the CPU combine; any design must keep per-step epoch consistency (docs/design/moe-remote-placement-epochs-v23.md). Evidence: 2026-11-01 decode-step breakdown in docs/reference/engines/radiance/models/qwen3.8-flash-next.md.

## Change Log

- 2026-11-02T01:49:18.074244+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261102_014941_documented-the-r9700-idle-powe_8134
- 2026-11-02T01:49:53.151690+00:00 (updated-by): Updated: section:ledger-events
