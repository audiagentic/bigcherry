---
id: BCOP16
order: 16
plan: patching-bc-optimizations
state: superseded
created-at: '2026-10-05T04:45:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Gate fused CPU cold-tail MoE execution from measured overhead

## Description

Backfill of the earlier MET03 audit. Before adding CPU-tail fusion, attribute gate/up/down, activation/intermediate traffic, combine, scheduler and CPU service time. Fusion is justified only when non-matmul overhead is material.

## Steps

1. Profile MET03 CPU-tail service on representative tiered MoE lanes.
2. Attempt fused gate/up/activation/down execution only if non-matmul overhead is >=15% of CPU expert service time.
3. Reuse MET02 range semantics and MET01 placement; no new router/cache/expert store.
4. Keep decode and prefill gates separate; external OpenVINO/fork fusion gains are mechanism evidence only.
5. Avoid NUMA expert sharding unless actual multi-NUMA hardware measurements justify it.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

- CPU-tail service breakdown is measured.
- Fusion is either rejected early or benchmarked with correctness and PP/TG controls.
- No duplicate placement/router/cache mechanism is introduced.

## Notes

2026-10-05: superseded by MET03 - findings, steps and gates folded into its notes. Not implemented; work stays open there.

## Related

MET01, MET02, MET03.

## Change Log

- 2026-10-05T04:55:13.356221+00:00 (updated-by): Updated: section:notes
- 2026-10-05T04:55:42.483049+00:00 (state-transition): State: pending → superseded
