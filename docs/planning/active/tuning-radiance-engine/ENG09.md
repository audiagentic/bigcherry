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

2026-11-29: measured on the installed bc-radiance-2030.5.0, stock power (plan profile stock, mode observe). Evidence: docs/evidence/2026-11-29-eng09-2.3-baseline (#1776).
- Flash-Next 4 cards: two launches. Every gated metric is within 1.5% of the 2020.11.2 baseline: decode 24.5 / 47.4 / 28.6 ms/step at 1 / 6 / 8 streams, cold TTFT 8.33 s, 256K prefill about 3,000 t/s. New baseline docs/reference/engines/radiance/baselines/flashnext-4gpu-parity-2.3.json. The parity tool needed two fixes for 2.3 (#1773): the 8-session covariate counts mixed steps, and each launch gets its own run id when the stored plan is reused.
- TP3 27B (XTX, XTX, R9700), 262,144-token sessions:
  - IQ8R: 1,442,463 pool tokens a rank. TTFT 0.79 / 3.18 / 13.7 s at 2K / 8K / 32K, 175 s at 256K. Decode 123 / 216 / 317 / 408 t/s at 1 / 2 / 4 / 8 streams (20.4 to 36.5 ms/step).
  - IQ6R (ours): 1,520,927 pool tokens. Prefill level or slightly faster (32K 13.4 s, 256K 172 s). Decode 107 / 178 / 282 / 377 t/s (23.4 to 37.8 ms/step).
- Tuning opportunity: the IQ6R decode step is 15% slower than IQ8R at one stream despite fewer weight bytes; that is the target of the IQ6R optimisation.
- Open: upstream q4 and q6. Their vocabulary heads are not cut for TP3, so the TP3 start refuses them; the owner is deciding. Their rows follow separately.

## Change Log

- 2026-11-28T12:30:54.401072+00:00 (created-by): Created by agent
- 2026-11-28T12:31:37.075551+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria
