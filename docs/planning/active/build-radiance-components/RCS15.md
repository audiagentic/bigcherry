---
id: RCS15
order: 0
plan: build-radiance-components
state: pending
created-at: '2026-11-18T08:41:29.761201+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Baseline upstream libr3d against the BigCherry RDNA3 library

## Description



## Steps

Refresh upstream and freeze source/ABI/runtime identities with RUM03 and BRBC05. Build both providers as isolated comparison arms from the same target revision if possible. Use KAB01 for comparisons and explicit kernel orders. Begin with one 7900 XTX and one approved dense fixture. Record selected rows, refusals and no-device-row cases before timing.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Native selftest against matched libref, local rewrite check for the comparison arm, KAB01 correctness and A/A noise, then matching dense decode/prefill timing. Reject mismatched schemas, library fallback or unbound artifact hashes.

## Effort & Risk



## Standards



## Acceptance Criteria

A reproducible matched-baseline row matrix separates native coverage, correctness gaps, performance gaps and untested shapes.

## Notes

Execution detail and source matrix: docs/reviews/radiance-native-libr3d/ASSESSMENT.md. Do not infer support from patch applicability or an upstream row name.

Use tracked kbench-libr3d evidence as fixture history. The archived reports inspected use only libr3 and libref. They do not establish a native A/B baseline. Codeberg DNS/browsing did not provide a fresh HEAD during this assessment. Refresh and re-diff libr3d before taking implementation results.

## Change Log

- 2026-11-18T08:41:29.761201+00:00 (created-by): Created by agent
- 2026-11-18T08:49:00.279834+00:00 (updated-by): Updated: section:notes
