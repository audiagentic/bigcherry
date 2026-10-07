---
id: QFP40
order: 40
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:39:56.573121+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# Audits from the external list: lazy-mode embedding table, MTP decode fused copies, gathered QSA decode

## Description

Three cheap checks. (a) Per-layer embedding table: the report keeps the 26.8 GB table resident (--lazy-mode off), lookup 1.0-1.6 -> 0.1-0.3 ms, ~3-4% decode; we place it on the host - confirm it is resident and not read from disk per pass. (b) MTP decode: fused same-shape copies and leaner conv-state rollback (88.8 -> 92.7 t/s); we have 1308 - check the copies. (c) Gathered QSA at decode (67.9 -> 80.6 t/s at 48K): 1295_qsa_gather_decode is only 'evaluated' - re-read why it was not promoted.

## Steps

One short finding per check, then a follow-up item only where something is missing.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Each finding backed by a measurement or a code reference.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-07T00:39:56.573121+00:00 (created-by): Created by agent
