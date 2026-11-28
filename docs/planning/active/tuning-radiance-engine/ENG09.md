---
id: ENG09
order: 0
plan: tuning-radiance-engine
state: pending
created-at: '2026-11-28T12:30:54.401072+00:00'
breadth: ''
skill: ''
created-by: agent
work: M
priority: P1
---

# Performance baseline on the 2.3 build: Flash-Next 4-GPU and TP3 27B at 4, 6 and 8 bits

## Description

Owner direction 2026-11-28: the v2.3.0 bump changed the baseline, so earlier serving numbers and plan premises predate it. Record one performance baseline of the installed 2.3 build on the primary topologies, to tune against and to re-evaluate open plans: Flash-Next on four cards (flashnext-4gpu) and Qwen3.8-27B TP3 on XTX + XTX + R9700 at 4 bits (upstream IQ4R-L V4, replacing the October MXFP4), 6 bits (our IQR-V4, not yet optimised) and 8 bits (IQ8R). Per topology and model: maximum context (KV pool tokens and the 262,144-token session limit), single-stream decode and prefill, and concurrent throughput, with decode and prefill reported separately.

## Steps

1. Configs: qwen27b-tp3-long-iq4r (upstream IQ4R-L V4, variant of qwen27b-tp3-long-iq6r differing only in the model); the existing qwen27b-tp3-long (IQ8R) and qwen27b-tp3-long-iq6r.
2. Flash-Next: tools/lab/radiance/flashnext-parity.sh measure on the installed release (cold prefill, decode step at 1 and 6 streams, 8 sessions, 256K context), two launches; store as the 2.3 parity baseline.
3. TP3 27B, each model: start from the installed release; capacity from the start log (pool tokens per rank); TTFT at 512, 2048, 8192 and 32768 prompt tokens; decode tokens/s and ms/step at 1, 2, 4 and 8 streams with stream-bench (several seeded prompts, since speculative acceptance varies by prompt); one long-context session near 262,144 tokens.
4. One short GPU lease per model run, released between runs.
5. Evidence in docs/evidence (no host paths), and the numbers each open tuning/capacity plan rests on (IQ6R optimisation, RSC16-18, VTP36, ENG04) re-evaluated against it.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Every cell measured on the installed 2.3 release with recorded build identity; decode and prefill separate; per-prompt acceptance recorded; repeat launch within the noise the parity tool reports.

## Effort & Risk



## Standards



## Acceptance Criteria

- [ ] Flash-Next 4-GPU parity baseline on 2.3 stored.
- [ ] TP3 27B at 4, 6 and 8 bits: capacity, TTFT by prompt length, decode by stream count, one long session.
- [ ] Evidence committed; open tuning and capacity plans annotated with what the baseline changes.

## Notes



## Change Log

- 2026-11-28T12:30:54.401072+00:00 (created-by): Created by agent
- 2026-11-28T12:31:37.075551+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria
