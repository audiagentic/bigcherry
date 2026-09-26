---
id: RCD06
order: 6
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:52:08.757524+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Campaign seams for the job service: evidence sink, frozen validated composition, producer preflights, progress events

## Description

validation_campaign changes the job service needs: --evidence-output (external evidence sink, so runner worktrees can be shared read-only), --frozen-validated-composition (exact promoted IDs + digests; fail before build on drift), a typed producer preflight API (run_preflight(ctx, phase=prebuild|premeasure) -> ok|retryable|invalid; MTP row completeness, -sm tensor topology/ctx cap), and structured progress events (phase, lane, round).

## Steps

1. write_validation_record(record, output=...).
2. validated_enhancement_patches(..., frozen=...).
3. ProducerPreflightResult + adoption in 1254/1241/1216 producers.
4. Progress event emitter consumed by the monitor.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline tests: frozen drift fails before build; preflight classification; evidence written only to the sink.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-09-26T00:52:08.757524+00:00 (created-by): Created by agent
