---
id: RAD10
order: 10
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-18T13:15:19.382833+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Multi-stream serving: measure and make the helper path correct and optimal under concurrent sessions

## Description

All helper work so far is single-stream. Radiance has vLLM-style concurrency (--max-num-seqs, paged attention, --max-num-batched-tokens) but we have not measured it, on 1.7 or 1.8. Measure, then fix what concurrency breaks: correctness first (rows from different sessions in one helper request, per-session ordering, no cross-talk, cancellation and session end mid-request), then throughput (aggregate tok/s and per-stream latency at 1, 2, 4, 6 streams).

## Steps

1. Add concurrent clients to the bench path (server-bench / ab-benchmark config: N parallel streams, fixed per-stream prompts, per-stream and aggregate tok/s, time to first token).
2. Baseline on radiance 1.8.0, helpers off, N = 1, 2, 4, 6: does the engine batch well on its own.
3. Same with helpers on (FP16, VECIO, FUSEROUTE, FUSESTAGE as promoted): helper wait share from the trace, small-lane rows histogram, lane switch when rows exceed 8.
4. Correctness: per-stream greedy identity (each stream's text equal to its single-stream text for a fixed seed/prompt) with helpers on vs off; stress with sessions starting and ending mid-run.
5. Replay harness: build multi-session captures (rows from different sequences) and gate changes on bitwise identity.
6. Optimise from the measurements: rows-per-request grouping across sessions, small-lane row limit and large-lane threshold, heat/ownership computed over the pooled routes (flatter hot set), per-session fairness. Each idea gets a harness first.
7. Record results on the Flash-Next model page and in evidence records; promote through the normal lifecycle.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Per-stream identity with helpers on vs off at N=1..6; aggregate throughput and p95 per-stream latency at each N; trace shows exposed wait at each N; no hang across 100 session start/stop cycles.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner request 2026-10-19: handle multiple streams correctly and optimally (radiance is meant to be good at multi-session like vLLM, where llama.cpp is strong single-stream). Runs after the 1.8.0 bump on the managed workflow (needs the concurrency bench from MEN20/MEN21-style commands, not lab scripts).

GPT review 2026-10-19 (Paiton, vllm-radiance, prior art), reconciled:

Order: (1) multi-session workload/replay spec and concurrent route captures at 1/2/4/8 streams (decode-only, prefill-only, mixed); reproduce single-stream helper timing within +-5% before trusting replay. (2) Baseline concurrent serving numbers (aggregate and per-stream tok/s, TTFT and inter-token p50/p95/p99, helper wait, queue wait, KV use). (3) Cross-session same-layer expert aggregation harness: policies = independent dispatch, same-layer aggregate immediate, bounded wait (10/25/50/100 us), plus short/long grouped lanes; one-stream performance is a regression gate; p95 decode regression budget 5%. (4) Scheduler/admission mock (continuous admission, bounded chunked prefill, decode priority, KV limits, helper backpressure, per-session cancel) before real kernels. (5) Cross-session heat and placement scorer on captured traces; compare alternating hot-first, per-layer balanced, queue-time-aware static, static + selective replication, periodic migration; score critical-path latency, not helper load.

Caveats from the review: decode rows per expert stay tiny (uniform model: 8 tokens x top-10 over 512 experts = 0.16 rows/expert), so decode gains come from fewer dispatches/syncs and dense projections, while prefill aggregation is the large win; an exposed 23% wait bounds the speedup at about 1.30x, realistically far less; do not put layers from different sessions in one expert GEMM without block-map support for multiple weight bases.

Rejected for now: replacing radiance with Paiton, dynamic expert migration before concurrent traces, porting Paiton's TP2 comms to no-P2P topology, moving all experts to GTT, assuming 3-bit is better, using the 6900 XT as an extra helper before an isolated feasibility test.

Separate items to raise (IDs to be reconciled with the planning index): W3A8 kernel/quality feasibility, GTT execution microbenchmark, Paiton reference-only engine (phase 0: pin revision/container digest, record external benchmarks, no patches; licence review first: plugin Apache-2.0, compiler proprietary, weights/drafter licences unknown), Paiton concurrency/quality comparison, and 2xP2P R9700 hardware evidence (external hardware).

## Change Log

- 2026-10-18T13:15:19.382833+00:00 (created-by): Created by agent
- 2026-10-18T14:07:22.367325+00:00 (updated-by): Updated: section:notes
