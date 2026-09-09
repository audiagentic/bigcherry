---
id: RHA09
order: 0
plan: run-hip-autotune
state: pending
created-at: '2026-09-09T19:16:56.626655+00:00'
breadth: ''
skill: intermediate
created-by: RV161
work: S
priority: P2
---

# Retain durable raw graph-lifecycle qualification artifacts

## Description

Preserve a repository-backed raw log bundle and machine-readable validator result for the already-proven dual-XTX graph lifecycle. This follow-up is evidence hygiene only and does not reopen RHA04 parity or require mixed-topology/long-context rows outside the current release target.

## Steps

1. Reproduce or retrieve the provenance-locked dual-XTX graph-on lifecycle run using the maintained environment settings. 2. Retain raw server log, lifecycle extraction, validator output, build/model/topology identity, and clean teardown evidence under docs/evidence. 3. Link the artifact to HI14/HI90 historical evidence and record limitations. 4. Close only when the raw bundle is reproducible and independently validated.

## Detailed Solution & Technical Design

Use graph_lifecycle_evidence.capture_lifecycle_from_log() and multi_gpu_validate.py as the acceptance authority. The durable bundle must include capture_begin, capture_end, instantiate, replay, real graphs-reused evidence, per-device identity, and provenance. Do not infer missing raw data from prose notes.

## Code Samples & Guidance



## Files

docs/evidence/<run-id>-graph-lifecycle/README.md
docs/evidence/<run-id>-graph-lifecycle/server.log
docs/evidence/<run-id>-graph-lifecycle/lifecycle.json
docs/evidence/<run-id>-graph-lifecycle/validator.json

## Validation

Run the strict graph lifecycle and multi-GPU validators against the retained machine-readable artifacts; confirm all four lifecycle stages, device identity, clean teardown, and provenance hashes. No mixed gfx1100/gfx1201 or long-context claim is required unless the release scope changes.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; docs/reference/testing/MULTI_GPU_LARGE_MODEL_VALIDATION.md

## Acceptance Criteria

A durable raw artifact bundle exists, validates offline, and documents exact build/model/topology and limitations; historical prose-only evidence is no longer the sole retained source.

## Notes

Successor follow-up created by review RV161. Keep separate from RHA04 performance admission.

## Change Log

- 2026-09-09T19:16:56.626655+00:00 (created-by): Created by RV161

## Ledger-events

- chg_20260909_191740_rha02s-core-graph-lifecycle-q_5460
- 2026-09-09T19:17:40.613889+00:00 (updated-by): Updated: section:ledger-events
