---
id: RAD43
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-11-01T11:26:24.954166+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P2
---

# Kernel work: helper reduce/write stages, the tail's remaining latency, XTX gate/up/down, engine add

## Description

After #486 the per-request stage times on a 2048-row prefill step are, XTX head: routes ~0.42 ms, gate/up ~1.5, down ~?, reduce/write ~?, total 3.4-3.6 ms for ~9,200 routes; 6900 XT tail: routes 0.54, gate/up 1.53, down 1.25, reduce 0.84 for ~900 routes (1-row experts still latency-bound at ~165 GB/s of weight reads). Engine (R9700, --profile-ops, combine off): moe_mailbox_add 55.6% (wait + host read), dense gemm_nt_q 10.5%, local moe_gemm_q 7.9%, publish 5.3%. Each stage gets a model-free harness (replay-kernel-time.sh on captures, gfx1030/rowsk.hip) before any change.

## Steps

1. Stage table for the XTX helper from replay captures (rocprof): which of route compaction, gate/up, quant, down, reduce dominates and how far each is from its bandwidth or compute bound.
2. The reduce/write stage on every helper: it writes rows over the link; check it achieves the link's write rate and overlaps with the down GEMM tail.
3. The tail's 1-2-row experts: GEMV path or merged small experts in one block, measured in rowsk.
4. Engine add kernel (moe_mailbox_add_rows): achieved host-read GB/s against host-link-bw's pull rate.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Per-stage harness numbers before and after each change; live 42K probe and 8-session run for every kept change; bit-exact or KLD inside the noise band.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Measurement first; only changes with a harness-measured gain go on.

## Change Log

- 2026-11-01T11:26:24.954166+00:00 (created-by): Created by agent
