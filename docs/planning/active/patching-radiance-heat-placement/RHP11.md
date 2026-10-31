---
id: RHP11
order: 11
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-10-31T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Ownership epoch protocol

## Description

Separate the mailbox request sequence from a placement generation and define complete-map publication, acknowledgement and retirement. A mailbox sequence does not say which ownership map a request used.

## Steps

1. Write the protocol, experiment or collector against the code as it is (radiance 1.14.0, remote-MoE protocol v19).
2. Land card-free tests first where the item has any logic; measure only with the gpu lease.

## Detailed Solution & Technical Design

## Code Samples & Guidance

## Files

## Validation

Card-free protocol tests for epoch overlap, stale completion and partial publication before any live GPU mutation; no partial map is ever published.

## Effort & Risk

Depends on: RHP06. Added by the 2026-10-31 GPT review of this plan.
