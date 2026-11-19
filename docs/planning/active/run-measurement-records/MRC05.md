---
id: MRC05
order: 5
plan: run-measurement-records
state: pending
created-at: '2026-11-19T00:50:41.670049+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: M
---

# Platform baseline, compare and history over run records (one-sided gates, measured tolerances, drift flags); engine parity harnesses become workload profiles

## Description

Baseline, compare and history as platform tools over run records: one-sided gates with tolerances measured from spread (a single launch is refused), same-session A/B guidance, and `results history --layout X --metric Y` with drift flags. Engine harnesses become workload profiles feeding records.

Today: tools/lab/radiance/flashnext_parity.py (with flashnext-parity.sh) is a radiance lab tool: it runs, builds a baseline (`baseline OUT --name NAME`, tolerances derived from the spread across launches and repeats) and compares. Its baselines are JSON files in docs/reference/engines/radiance/baselines/. The v1.17.0 baseline stores only a build path under ${BC_WORK_ROOT}, a revision and a source_run path; it has no manifest and no heat file, so it cannot be reproduced (see the 2026-11-19 evidence). workloads.py has compare() over its own index.

## Steps

- 1. Lift the baseline format into the platform: a baseline is a set of records (MRC01) plus gate definitions; it embeds the provenance bundle, so it is reproducible from itself.
- 2. Compare: one-sided gates per metric direction (lower or higher is better), tolerance from measured spread across launches; refuse to create a baseline or a verdict from a single launch; print the host-fact and feature-state differences first (MRC02, MRC03) so a confounded A/B is visible.
- 3. History: `bigcherry results history --layout X --metric Y` over records, drift flags against the baseline and the trailing window.
- 4. Same-session A/B guidance: a compare of two records from different sessions warns and states which host facts differ.
- 5. Convert flashnext_parity.py into a workload profile that writes records; keep its CLI as a thin wrapper until MRC06 migrates its producers.
- 6. Re-baseline v1.17.0 -> new baseline with the provenance bundle, and document the old file as unreproducible.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

- tools/lab/radiance/flashnext_parity.py (run, baseline, compare; radiance lab)
- tools/lab/radiance/flashnext-parity.sh
- docs/reference/engines/radiance/baselines/flashnext-4gpu-parity.json (v1.17.0 baseline: build path, revision, source_run path, no manifest or heat file)
- docs/reference/engines/radiance/baselines/flashnext-4gpu-parity-main-20261119.json
- tools/bigcherry/bench/workloads.py (compare(), difference())
- docs/evidence/2026-11-19-flashnext-4gpu-regression/README.md

## Validation

- Tests: spread-derived tolerance, single-launch refusal, one-sided gate in each direction, history drift flag on a synthetic series.
- Run the new compare on the four arms of docs/evidence/2026-11-19-flashnext-4gpu-regression and confirm it reports the policy state difference.

## Effort & Risk

Medium. Risk: tolerances too tight on noisy metrics; derive them from measured spread and keep gates one-sided.

## Standards



## Acceptance Criteria

- A baseline contains everything needed to reproduce it, by content.
- A compare never gives a verdict from one launch and always lists host and feature-state differences.
- `results history` answers a layout/metric question without opening run folders.

## Notes



## Change Log

- 2026-11-19T00:50:41.670049+00:00 (created-by): Created by agent
- 2026-11-19T00:54:22.108305+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:effort_risk, section:acceptance_criteria
