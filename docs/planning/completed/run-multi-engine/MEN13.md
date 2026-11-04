---
id: MEN13
order: 3
plan: run-multi-engine
state: superseded
created-at: '2026-10-14T22:28:01.533375+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Engine release lines: formal release stages, tags and releases for every engine

## Description

Releases are llama.cpp-only: release-please-config.json has packages engines/llamacpp (component bc-llamacpp) and the platform (bc-platform); pin_release.py hardcodes RELEASE_PACKAGE = "engines/llamacpp"; tags are bc-llamacpp-<ver>, bc-llamacpp-b<pin>-r<n>, bc-platform-<ver>. Radiance has no release line, no changelog, no pin-release tag. Every engine must release through the same formal stages (ledger -> release notes -> release-please PR -> tag -> pin-release record), never a hand tag.

## Steps

1. Add an engines/radiance release-please package (component bc-radiance, CHANGELOG.md) and bootstrap it at the current pin.
2. Make pin_release.py and release-notes take --engine and resolve the package from engines/<name>/engine.toml; migrate the llama.cpp caller (no default shim).
3. Pin-release tag form per engine: bc-<engine>-<upstream-rev>-r<n>.
4. Ledger events carry the engine so each release's notes contain only that engine's changes (platform changes go to bc-platform).
5. Document the stages once, engine-generic, in docs/reference/build (RELEASES.md), and update PIN_BUMP.md and the bump skill to point at it.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline tests for engine resolution in pin-release and release-notes; a dry-run release-please for bc-radiance produces the expected PR; first real bc-radiance release cut after RAD09 merges.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner, 2026-10-15: 'We should be able to bump and commit any engine. We need to ensure tags and releases are built on formal release stages.' Pairs with MEN10 (bump) and MEN11 (validate/promote).



BACKLOG TRIAGE 2026-11-04 (SUPERSEDED; revalidated against main 6eb31a708, verdict CONFIRM): engines/radiance release-please package, pin_release --engine and radiance release stages are delivered (bc-radiance-1140.0.0, 1160.0.0). NOTE: attaching a runtime bundle to a release is not in this item and not delivered; tracked in the new item MEN27. Evidence: engines/radiance release-please package, pin_release --engine and radiance release stages are delivered (bc-radiance-1140.0.0, 1160.0.0). NOTE: attaching a runtime bundle to a release is not in this item and not delivered; tracked in the new item MEN27. Still referenced by: MEN14. Original content kept below for history.

## Change Log

- 2026-10-14T22:28:01.533375+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261017_012856_radiance-has-its-own-release-l_4308
- 2026-10-17T01:29:00.283464+00:00 (updated-by): Updated: section:ledger-events
- chg_20261018_201215_radiance-180-is-released-as_1561
- 2026-10-18T20:12:23.948481+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-04T03:46:56.235671+00:00 (updated-by): Updated: section:notes
- 2026-11-04T03:47:00.234603+00:00 (state-transition): State: pending → superseded
