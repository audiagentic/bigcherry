---
id: RSC09
order: 0
plan: run-serving-configuration
state: pending
created-at: '2026-11-04T07:54:17.228388+00:00'
breadth: ''
skill: intermediate
created-by: bigcherry-2
work: M
priority: P2
---

# Derive or calibrate the hand-picked launcher literals in the v1 manifest handoff section

## Description

bc-serve --write-manifest (RSC08 S2a) writes the five flashnext launcher literals into spec.handoff with provenance origin profile, class derived, confidence estimate, because they differ from the component defaults and were only ever hidden under the root claim: handoff.transport max_rows 4096 (setting default 64), small_rows 8 (64), gemv_large_rows 32 (0), skip_owned true (none), handoff.kernels moe_down_tile 1 (none). They are hand-picked tunables. Derive each from the layout, model or measured plan, or calibrate it (calibration entry with evidence), then drop the literal. Also the interim plan-cache key does not cover the owner map, helper settings or the row limit (S3).

## Steps

1. For each literal, find the measurement or rule that justifies it (RAD18/RAD23 A/Bs) and decide derive vs calibrate.
2. Add calibration entries with evidence, or derive in the resolver.
3. Remove the profile record and the literal from tools/lab/radiance/launch-flashnext-best.sh at S6.
4. Update the example manifests and the parity test.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Parity test passes without launcher_literal rows for these five; hardware A/B shows no regression with the derived or calibrated values.

## Effort & Risk



## Standards



## Acceptance Criteria

No handoff value in a manifest has origin profile with confidence estimate; each has a derivation or a calibration entry with evidence.

## Notes



## Change Log

- 2026-11-04T07:54:17.228388+00:00 (created-by): Created by bigcherry-2
- 2026-11-04T07:54:31.476342+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria

## Ledger-events

- chg_20261104_075438_the-serving-configuration-tool_6545
- 2026-11-04T07:54:46.777908+00:00 (updated-by): Updated: section:ledger-events
