---
id: MRC05
order: 5
plan: run-measurement-records
state: completed
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

Merged: #1198 (bigcherry results baseline|compare|history over run records: tolerances from measured spread, single launch refused, one-sided gates by recorded direction, decode vs prefill/TTFT verdicts separate, provenance diff, drift flags), #1214 (the Flash-Next parity harness consumes portable run records), #1223 (parity.json, the harness's own baseline/compare and the record toggle removed; flashnext-parity.sh baseline|compare call bigcherry results with the launch SHAs). Tests: tools/tests/core/test_results.py, tools/tests/lab/test_flashnext_parity.py (MRC05 verdict reproduces the former harness verdict on fixtures).

## Effort & Risk

Medium. Risk: tolerances too tight on noisy metrics; derive them from measured spread and keep gates one-sided.

## Standards



## Acceptance Criteria

Baselines, comparisons and history are platform operations over run-record v1 for any engine; the Radiance parity harness has no separate baseline/compare implementation or summary file; a comparison lists the provenance differences between the runs it compares.

## Notes

On main 2026-11-19: `baselines/flashnext-4gpu-parity-main-20261119.json` (a current-main parity file) and the parity harness warmup override for policy-on arms (#1178); the dense TP3 no-regression run the same day compares against baselines recorded only in evidence prose (RAD52, VTP37), which is the gap this item closes: those numbers have no machine-readable baseline, host facts or provenance bundle.

## Change Log

- 2026-11-19T00:50:41.670049+00:00 (created-by): Created by agent
- 2026-11-19T00:54:22.108305+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-19T07:39:07.531890+00:00 (updated-by): Updated: section:notes
- 2026-11-20T01:25:28.932213+00:00 (updated-by): Updated: section:validation, section:acceptance_criteria
- 2026-11-20T01:25:33.475307+00:00 (state-transition): State: pending → completed
