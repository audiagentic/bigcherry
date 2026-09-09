---
id: RHA09
order: 0
plan: run-hip-autotune
state: completed
created-at: '2026-09-09T19:16:56.626655+00:00'
breadth: ''
skill: intermediate
created-by: RV161
work: S
priority: P2
---

# Retain durable raw graph-lifecycle qualification artifacts

## Description

Retain durable raw graph-lifecycle artifacts from the historical HI14 dual-XTX run and make the evidence independently re-checkable offline.

## Steps

1. Reproduce or retrieve the provenance-locked dual-XTX graph-on lifecycle run using the maintained environment settings. 2. Retain raw server log, lifecycle extraction, validator output, build/model/topology identity, and clean teardown evidence under docs/evidence. 3. Link the artifact to HI14/HI90 historical evidence and record limitations. 4. Close only when the raw bundle is reproducible and independently validated.

## Detailed Solution & Technical Design

Use graph_lifecycle_evidence.capture_lifecycle_from_log() and multi_gpu_validate.py as the acceptance authority. The durable bundle must include capture_begin, capture_end, instantiate, replay, real graphs-reused evidence, per-device identity, and provenance. Do not infer missing raw data from prose notes.

## Code Samples & Guidance



## Files

- docs/evidence/2026-09-10-rha09-graph-lifecycle/server.log
- docs/evidence/2026-09-10-rha09-graph-lifecycle/dispatch-record.jsonl
- docs/evidence/2026-09-10-rha09-graph-lifecycle/lifecycle.json
- docs/evidence/2026-09-10-rha09-graph-lifecycle/validator.json
- docs/evidence/2026-09-10-rha09-graph-lifecycle/README.md

## Validation

- Run capture_lifecycle_from_log against server.log and require all four stages.
- Run validate_multi_gpu_evidence against validator.json.
- Verify SHA-256 hashes documented in README.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; docs/reference/testing/MULTI_GPU_LARGE_MODEL_VALIDATION.md

## Acceptance Criteria

A durable raw artifact bundle exists, validates offline, and documents exact build/model/topology and limitations; historical prose-only evidence is no longer the sole retained source.

## Notes

Successor follow-up created by review RV161. Keep separate from RHA04 performance admission.

Historical artifact provenance is explicit. The bundle supports graph lifecycle and dual-device dispatch coverage only; it is not a new performance admission and does not claim mixed topology or long-context coverage.

## Change Log

- 2026-09-09T19:16:56.626655+00:00 (created-by): Created by RV161

## Ledger-events


- chg_20260909_191740_rha02s-core-graph-lifecycle-q_5460
- 2026-09-09T19:17:40.613889+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T19:39:06.850044+00:00 (updated-by): Updated: section:description, section:files, section:validation, section:notes
- chg_20260909_193911_preserved-and-offline-validate_5866
- 2026-09-09T19:39:11.730959+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T19:39:20.895278+00:00 (state-transition): State: pending → completed
