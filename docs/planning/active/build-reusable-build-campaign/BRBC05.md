---
id: BRBC05
order: 0
plan: build-reusable-build-campaign
state: pending
created-at: '2026-11-18T03:35:11.986295+00:00'
breadth: ''
skill: ''
created-by: agent
priority: P1
work: M
---

# Bind Radiance benchmark evidence to installed runtime members

## Description

Record the engine, compiled plugins and helper payloads from the selected runtime tree. Use the same build-relative member identity at publication, reuse and benchmark capture.

## Steps

Follow the artifact-identity packet in docs/reviews/astra-performance-2026-11-18/IMPLEMENTATION-HANDOFF.md. Deliver before making new Radiance performance claims.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

tools/bigcherry/build/builds.py; tools/bigcherry/campaign/workers.py; tools/bigcherry/campaign/benchmark.py; tools/bigcherry/release/runtime_assets.py

## Validation

Existing card-free build/capture fixtures. Change only a runtime plugin, remove a required helper and exercise duplicate basenames. See the handoff for commands.

## Effort & Risk



## Standards



## Acceptance Criteria

New receipts attest the actual required runtime members. Clean reuse succeeds, drift or an incomplete legacy identity refuses or gets independently verified, and old evidence stays intact.

## Notes

## Change Log

- 2026-11-18T03:35:11.986295+00:00 (created-by): Created by agent
