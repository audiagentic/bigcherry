---
id: MEN27
order: 0
plan: run-multi-engine
state: completed
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

Each engine release has a runtime bundle asset with a recorded SHA-256 that matches the lane build named in the release record. Met for bc-radiance-1170.0.0.

## Notes

DELIVERED 2026-11-05 by pin-release (tools/bigcherry/release/runtime_assets.py): the record phase packages the verified lane build and the release phase uploads and verifies it. bc-radiance-1170.0.0 (Radiance v1.17.0) carries radiance-e5c7cb3dcc920259656edb17ec57ffc0.tar.gz (226 MB: bin/radiance and tools, radiance_home/bin/bc-serve, helpers/r3_moe_worker and the n-gram helper, kernels libr4d/libr3/libr3moe/libr2/libbcgfx12/libref/libavx) and its .sha256; releases/v1.17.0.json runtime_assets records plan_id e5c7cb3dcc920259656edb17ec57ffc0, source_slice_id e01f7f1f30550c58f33bd0e398360f74, binary and archive SHA-256, manifest SHA-256, ROCm 7.2.4. The record also carries the MEN28 evidence_gate result.

## Change Log

- 2026-11-04T03:44:40.168870+00:00 (created-by): Created by agent
- 2026-11-05T22:17:28.602482+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
- 2026-11-05T22:18:02.642388+00:00 (state-transition): State: pending → completed
