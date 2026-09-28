---
id: PVPS13
order: 0
plan: patching-validation-package-standard
state: completed
created-at: '2026-09-27T09:51:20.294329+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# Push the MTP-lane locator fix into mtp_server_lane() itself, not per-producer copies

## Description

The ARCH_MISMATCH locator fix (server attestation names a device only by PCI locator; device.execution_identity omits it, so it must be merged in via dataclasses.replace(device.execution_identity, locators=(device.locator,))) has now been independently copy-pasted into FIVE separate producers (1254, 1256, 1257, 1268, 1269) as each one hit the bug live in the queue. mtp_server_lane() in tools/bigcherry/patch/producer_support.py still takes a bare `expected: ExecutionIdentity` and has no way to derive the locator itself.

## Steps

1. Change mtp_server_lane()'s signature to accept `device: vp.ProducerDeviceContext` instead of `expected: ExecutionIdentity`, deriving the locator-aware identity internally the same way _server_session_factory() already does.
2. Migrate all 7 call sites (1241, 1252, 1254, 1256, 1257, 1268, 1269) to pass device= instead of expected=device.execution_identity/dataclasses.replace(...).
3. Delete the now-redundant per-producer dataclasses.replace blocks.
4. Add a test asserting mtp_server_lane() itself builds the locator-aware identity from a device fixture with a locator set.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

tools/tests/patch suite; a live queue session for one migrated patch confirming attestation still passes.

## Effort & Risk



## Standards



## Acceptance Criteria

mtp_server_lane() accepts device= and derives the locator-aware ExecutionIdentity internally (exactly one of device=/expected= required, fails closed otherwise). The 5 real single-device call sites (1254, 1256, 1257, 1268, 1269) pass device= and no longer duplicate the dataclasses.replace(...) locator merge. 1241/1252 (dual-GPU, no single locator) remain on expected= unchanged. tools/tests/patch full suite green aside from one confirmed pre-existing, unrelated failure (patch 1270 anchor mismatch).

## Notes

Owner priority: low, framework cleanup only (no behavior change once all call sites are migrated -- the fix is already correct everywhere it has been applied). Per CLAUDE.md 'no legacy, always migrate up': do the full signature migration in one change, not an optional parameter alongside the old one.

## Change Log

- 2026-09-27T09:51:20.294329+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260927_125342_cleaned-up-5-duplicate-copies_7807
- 2026-09-27T12:53:47.554185+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-27T12:54:17.331333+00:00 (updated-by): Updated: section:acceptance_criteria
- 2026-09-27T12:54:23.584568+00:00 (state-transition): State: pending → completed
