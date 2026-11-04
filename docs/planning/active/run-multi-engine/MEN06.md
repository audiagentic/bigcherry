---
id: MEN06
order: 6
plan: run-multi-engine
state: pending
created-at: '2026-10-08T20:36:35.018575+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: M
---

# Engine-neutral A/B and activation evidence

## Description

The lightweight promotion tier rests on four pieces of evidence: an ABBA with complete separation, activation proof, greedy identity and a no-regression check. Today each is produced by llama.cpp-specific means (BIGCHERRY_PATCH_HIT lines in the server log, llama-server timings, queue rows that pass environment variables to one binary).

## Steps

1. ABBA runner over the MEN03 adapter: arms are (engine, build, settings) triples; position recorded per run; rounds a multiple of the arm count.
2. Activation evidence as a per-engine rule: a log pattern, or a metrics counter that must be non-zero in the subject arm and zero in the control.
3. Greedy identity and acceptance equality checks from the normalised result schema.
4. bigcherry patch-promote accepts evidence that names its engine.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

An existing llama.cpp promotion (1350) can be re-expressed in the neutral form with the same verdict; a settings A/B on the vLLM container produces the same record shape.

## Effort & Risk



## Standards



## Acceptance Criteria

One evidence record format serves both engines and patch-promote reads it.

## Notes

After MEN03. Do not generalise scripts that are spent.



BACKLOG TRIAGE 2026-11-04 (SUPERSEDED; revalidated against main 6eb31a708, verdict CONFIRM): `ab-benchmark` and `stream-bench` are engine-neutral and used for every radiance A/B (e.g. runs/rhp11-e-ab*, rad116-ab1). Evidence: `ab-benchmark` and `stream-bench` are engine-neutral and used for every radiance A/B (e.g. runs/rhp11-e-ab*, rad116-ab1). Still referenced by: MEN07, MEN09. Original content kept below for history.



REOPENED 2026-11-04: an independent revalidation (against main 6eb31a708) rejected the closure recorded above, which had been based on a lighter check. It stays open. Why: campaign/benchmark.py:529-659 supports engine-specific sidecars and normalized A/B cells, and release/patch_promote.py:350-364 is engine-neutral. MEN06.md:39-53 additionally calls for re-expressing llama.cpp patch 1350 with the same verdict and a container settings A/B in the common record; neither acceptance fixture/evidence was located, while MEN07 and MEN09 still reference this prerequisite.

## Change Log

- 2026-10-08T20:36:35.018575+00:00 (created-by): Created by agent
- 2026-11-04T03:46:07.449803+00:00 (updated-by): Updated: section:notes
- 2026-11-04T03:46:11.544377+00:00 (state-transition): State: pending → superseded
- 2026-11-04T04:44:07.152237+00:00 (updated-by): Updated: section:notes
- 2026-11-04T04:44:15.360225+00:00 (state-transition): State: superseded → pending
