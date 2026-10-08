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

## Change Log

- 2026-10-08T20:36:35.018575+00:00 (created-by): Created by agent
