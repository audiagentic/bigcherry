---
id: PGC08
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-09-29T09:34:11.636242+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# 1254 GDN MTP prefix tail: implement routing, then validate 1253 vs 1253+1254 (and 1261)

## Description

1254_nro05 is currently only a fail-closed predicate; its README says runtime routing remains unchanged, so it is a no-op today and validating it now would measure nothing. GPT authors the routing (chunked prefix of n_tokens-K, sequential last K tokens), then paired control(1253) vs subject(1253+1254) on the production shape with marker patch=1254_nro05 path=gdn_mtp_prefix_bf16, at least 4 sessions, acceptance parity. Also 1261: add trace marker and validate control vs +1261.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

ab-benchmark dual gfx1100, -sm tensor, MTP n_max=4; token parity; marker per arm.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-09-29T09:34:11.636242+00:00 (created-by): Created by agent
