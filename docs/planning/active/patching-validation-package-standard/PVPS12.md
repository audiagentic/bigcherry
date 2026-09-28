---
id: PVPS12
order: 0
plan: patching-validation-package-standard
state: pending
created-at: '2026-09-27T02:53:34.798768+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# Persist an immutable run/session id in every validation evidence record

## Description

GPT review req_7aa35bcc2e364094 (2026-09-27): evidence withdrawals (evidence/withdrawn.json) currently link a record to its campaign run by exact equality of the full pair-ratio vector set (tools/lab/plan-qualification/withdraw_discarded.py). All 14 withdrawals so far were audited exact and unique, but the link should be an explicit identity: record the campaign run id (workdir name + start timestamp + host) and a session uuid in each record at write time, and require that id for withdrawal.

## Steps

1. Add run_id/session_uuid to the record schema (new schema version; migrate readers).
2. Write them in the producer when persisting the record.
3. withdraw_discarded.py links by run_id; the ratio-set match becomes a consistency check.
4. Tests: withdrawal by id; mismatched id refused.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Unit tests plus a campaign session whose record carries the id; withdrawal of a synthetic record by id.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Framework work (no promotion gate). Existing records keep ratio-set linkage (audited).

## Change Log

- 2026-09-27T02:53:34.798768+00:00 (created-by): Created by agent
