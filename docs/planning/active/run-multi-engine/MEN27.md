---
id: MEN27
order: 0
plan: run-multi-engine
state: pending
created-at: '2026-11-04T03:44:40.168870+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: S
priority: P2
---

# Attach the lane build runtime bundle to each engine release as a release asset

## Description

Releases today are tags, notes and a release record (releases/<pin>.json); the GitHub release has no assets (bc-radiance-1140.0.0 and 1160.0.0 carry none). The CI workflow comment in radiance-build.yml names packaging into a release artifact as a later step. Attach the lane build's runtime bundle (the radiance binary, radiance_home with its kernel plugins, the helper binaries r3_moe_worker and libr3moe, bc-serve) to the release, with its build plan id and SHA-256, so a release is a downloadable, versioned binary set. Split out of MEN13 (superseded: release lines delivered) by the 2026-11-04 backlog triage. Owner order: after the core (hot placement RHP, RAD46); not before.

## Steps

1. Decide the bundle layout and what the lane build already publishes. 2. Add a release step that attaches it and the checksum file. 3. Record the asset digests in releases/<pin>.json.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

A release created by pin-release carries the bundle and its checksum; a clean host can verify the digest against the release record.

## Effort & Risk



## Standards



## Acceptance Criteria

Each engine release has a runtime bundle asset with a recorded SHA-256 that matches the lane build named in the release record.

## Notes

## Change Log

- 2026-11-04T03:44:40.168870+00:00 (created-by): Created by agent
