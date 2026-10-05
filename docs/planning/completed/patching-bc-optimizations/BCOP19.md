---
id: BCOP19
order: 19
plan: patching-bc-optimizations
state: superseded
created-at: '2026-10-05T04:48:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Gate sequence-sharded long-context attention by actual topology cost

## Description

Backfill of the earlier QFP07 audit. Qwen Flash-Next's small KV-head count limits head-based distribution. Sequence-sharded decode attention is only worth pursuing if online-softmax partial-state communication over BigCherry's no-P2P topology can beat measured XTX attention skew.

## Steps

1. Finish static/context-aware attention placement and diagnose the long-context prefill/decode threshold behavior first.
2. Benchmark transfer latency for the exact O(head_dim) online-softmax partial state through available direct/host-staged paths.
3. Abandon sequence sharding before implementation if communication lower bound cannot beat measured XTX-only attention skew.
4. If viable, prototype token-range sharding with numerically stable `(max,sum_exp,weighted_value)` combine and adversarial correctness tests.
5. Reuse existing collective/transport ownership; QFP07 owns only attention/KV partitioning.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

- Communication lower bound is measured on the actual topology before implementation.
- Promote only for >=5% long-context decode improvement on two lanes with communication <=50% of attention time saved.
- Raw-logit/KLD and tail-shard correctness pass.
- No second communicator/scheduler is created.

## Notes

2026-10-05: superseded by QFP07 - findings, steps and gates folded into its notes. Not implemented; work stays open there.

## Related

QFP07, QFP09, QFP01/collective plans.

## Change Log

- 2026-10-05T04:55:23.293322+00:00 (updated-by): Updated: section:notes
- 2026-10-05T04:55:52.496813+00:00 (state-transition): State: pending → superseded
