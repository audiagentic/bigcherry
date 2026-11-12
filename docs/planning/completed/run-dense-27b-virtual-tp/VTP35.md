---
id: VTP35
order: 0
plan: run-dense-27b-virtual-tp
state: superseded
created-at: '2026-11-12T21:59:13.226732+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P2
---

# TP3 MXFP4 prefill parity with TP2 (host-staged all-reduce cost)

## Description

Follow-up from VTP14. Native TP3 (XTX, XTX, R9700) beats dual-XTX TP2 in decode throughput and capacity, but MXFP4 one-stream prefill still trails: TTFT +24% at 2048 tokens and +5% at 8192 (N7 refresh, build 8a6f2776); burst TTFT at 4-8 streams is 5-16% higher. Every prefill all-reduce is host-staged because the cards have no peer access on this host (VTP14 'Peer access: not available'), and the R9700 sits on a Gen4 x4 link. int8 TP3 prefill already beats int8 TP2 (-7% to -13%).

## Steps

1. Re-check peer access after host/BIOS changes (Resizable BAR, IOMMU mode); if available, measure the exact one-shot over P2P.
2. Without P2P: profile prefill all-reduce time per chunk at 2048/8192 and test overlap levers (chunk size, pipelined-prefill rows, all-reduce grid) on the current build.
3. ABBA the best option against TP2 at pp512/2048/8192.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- One-stream TTFT at 512/2048/8192 tokens, TP3 vs TP2, matched ABBA.
- Decode throughput at 4/6/8 streams unchanged.

## Effort & Risk



## Standards



## Acceptance Criteria

- MXFP4 TP3 one-stream TTFT within 5% of TP2 at 2048 tokens, or a recorded NO-GO stating the transport limit.

## Notes

Superseded 2026-11-13 by VTP36 (TP3 performance programme), which carries this item's acceptance (MXFP4 TP3 one-stream TTFT within 5% of TP2 at 2048 tokens, or a NO-GO) and its steps (peer-access re-check, prefill all-reduce profile and levers).

## Change Log

- 2026-11-12T21:59:13.226732+00:00 (created-by): Created by agent
- 2026-11-12T22:44:59.937252+00:00 (updated-by): Updated: section:notes
- 2026-11-12T22:45:16.471480+00:00 (state-transition): State: pending → superseded
