---
id: RCS17
order: 0
plan: build-radiance-components
state: pending
created-at: '2026-11-18T08:41:50.662737+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Qualify RDNA3 transport and mixed-device integration with libr3d

## Description



## Steps

After the native baseline, track transport/collectives and remote-helper operations separately from math. Bind rows to actual rank ISA and preserve exact TP slices, cross-DSO group ownership, stream/fence semantics, tickets and shutdown. Coordinate RUM05/RUM06 and current collective owners.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Prove gfx1100 single-card, homogeneous TP, then each supported mixed rank order and actual helper topology. Check selected row/device masks, exact collective payload, replay, drain/failure and ordered devices. Validate negotiated wire/format support independently.

## Effort & Risk



## Standards



## Acceptance Criteria

Each declared topology is explicitly qualified or remains on the previous release. No generic gfx11 claim based only on an XTX run.

## Notes

Execution detail and source matrix: docs/reviews/radiance-native-libr3d/ASSESSMENT.md. Do not infer support from patch applicability or an upstream row name.

The helper loader dlopens one DSO and requires BC configurator and down-tile pin exports before row selection/relayout. Extract or replace that contract in the existing helper owner before changing archs.toml. Native libr3d has no equivalent collective rows; validate the shared group and per-rank loader eligibility separately.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. New upstream row storage/layout and same ABI integer with changed fields make mixed rank tests essential. Validate actual per-rank provider/encoding/layout, native rad_plugin_abi_version AND rad_plugin_abi_layout, ordered BDF/local ordinal and exact collective/chaining semantics, helper disjointness; reversed XTX/R9700 and gfx1030 helper where used. Reuse native loader and RUM06/13 transport, no new protocol. No old plugin mixed with candidate ABI.

## Change Log

- 2026-11-18T08:41:50.662737+00:00 (created-by): Created by agent
- 2026-11-18T08:49:16.767766+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4629
- 2026-11-25T02:38:51.092289+00:00 (updated-by): Updated: section:notes
