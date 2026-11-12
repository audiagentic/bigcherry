---
id: VTP33
order: 33
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-12T07:38:58.014935+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: S
priority: P2
---

# TP3 capacity: resident KV bounded by the R9700; capacity-oriented splits

## Description

TP3 serves the full trained context (262,144 tokens), but its resident KV is only 6-9% above TP2's. One block table serves every rank, so the carve is the smallest rank's share. The R9700 holds 2 of the 4 KV heads (twice the KV per token) and ~45% of the weights; the XTX ranks could page about 1.96M tokens each.

## Results so far

- MXFP4 at 262k context: TP3 3.7 against TP2 3.4 full-length sessions (4 sequences); 3.5 against 3.3 (8 sequences). TTFT for a 26,624-token prompt is 20.0 s against 12.7 s; decode after it is 93.4 against 68.9 tok/s.
- w8a8 at 262k, 8 sequences: kv 1,1,2 gives 12.15 GiB and 2.8 sessions; kv 2,1,1 gives 11.72 GiB and 2.7 sessions. The extra head stays on the R9700.

## Steps

1. Sweep capacity-oriented splits: move delta-net and feed-forward shares off the R9700 to free its VRAM for KV. Record resident KV, full-length sessions and tok/s at 4/6/8 streams.
2. Report the throughput/capacity frontier per format.
3. Ship a capacity layout (settings only) beside the throughput layout.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- Engine KV report at 262k (pool, sessions).
- stream-bench at 4/6/8 streams.
- A long-context correctness check.

## Effort & Risk



## Standards



## Acceptance Criteria

- A layout with measurably more resident KV than TP2 (target +25% or more), with its throughput cost stated.

## Notes

Data: VTP14.md (N5 first pass, KV head placement). Script pattern: ${BC_TASK_TMP}/vtp14/kvsplit.sh.

## Change Log

- 2026-11-12T07:38:58.014935+00:00 (created-by): Created by agent
- 2026-11-12T07:40:17.937281+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
