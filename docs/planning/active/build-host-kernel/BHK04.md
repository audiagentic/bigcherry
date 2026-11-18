---
id: BHK04
order: 4
plan: build-host-kernel
state: pending
created-at: '2026-11-18T01:29:48.426851+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P2
work: M
---

# Audit and repeat the RAM baseline after the four-DIMM upgrade

## Description

Use docs/evidence/2026-11-18-memory-baseline as historical evidence. Audit the STREAM-style harness byte accounting (Triad reads two arrays and writes one), size/alignment and run identity before reuse; retain original reported numbers without silently rewriting history. Record 2-versus-4-DIMM slots, capacity, BIOS settings, active timing source, compiler, affinity, governor and background load.

## Steps

- Preserve the historical three-run table and explicitly audit the harness before comparison. Correct Triad traffic accounting in a separate small change, recording both historical and corrected definitions rather than silently replacing the old 47.16 GB/s figure.
- Check bounds/alignment for the fixed default allocation; avoid generalizing the harness into a benchmark framework.
- Record exact DIMM slots/capacity, BIOS-selected speed/voltage, active timing source, kernel/compiler, CPU affinity, governor and background load before/after the upgrade.
- Use the same array size, repetitions and explicit thread settings for a short three-run sample. Add latency only if an existing available tool makes it cheap and comparable; bandwidth alone can satisfy the initial comparison.
- Calculate relative changes from rates using the same traffic definition. Label missing original configuration or changed power/load as comparison limitations.

## Detailed Solution & Technical Design

The saved stream.c computes n*sizeof(double)*(k==2 ? 3 : 2) bytes, but Triad a=b+3*c reads two arrays and writes one, so its conventional traffic multiplier is also three. The raw timings weren't retained; a derived rate multiplied by 3/2 is a conversion of the old convention, not a fresh measurement. Record this explicitly and obtain a corrected pre-upgrade sample when possible.

Keep the small local C harness and repeatable shell command; do not add a long synthetic stress suite or a new native benchmark subsystem. Initial comparison needs a short reproducible bandwidth sample and the memory configuration identity. Memory overclock stability qualification is a separate activity from baseline measurement.

## Code Samples & Guidance

Targeted correction guidance (not applied by this plan):
```c
const double traffic_arrays = (k >= 2) ? 3.0 : 2.0;
const double gbps = (double)n * sizeof(double) * traffic_arrays / seconds / 1e9;
```
Use unchanged traffic convention for both compared arms and retain the original report with an explanatory correction note.

## Files

- docs/evidence/2026-11-18-memory-baseline/stream.c
- docs/evidence/2026-11-18-memory-baseline/README.md

## Validation

Review the four kernel formulas and byte counts; run the corrected harness only when taking the actual before/after baseline under the container measurement lease. No full RAM stability or tuning campaign in this item.

## Effort & Risk

M: historical Triad byte counting and changed load/settings can invalidate comparison. Correct the method transparently, prefer a short fresh pre-upgrade sample, and label unsupported comparisons.

## Standards



## Acceptance Criteria

A short comparison report includes corrected methodology, comparable run identities and relative bandwidth changes. The old Triad value is not treated as directly comparable without correction, and unavailable timing fields are labeled.

## Notes

Plan-only follow-up to the earlier RAM baseline and timing investigation. BHK01 already owns kernel recovery; do not duplicate it or modify live host settings as part of planning.

Depends on BHK02 for active timings where available; basic configured-memory comparison can proceed with SMBIOS if active decode remains blocked. Do not block the whole RAM comparison on writer feasibility BHK03.

## Change Log

- 2026-11-18T01:29:48.426851+00:00 (created-by): Created by codex
- 2026-11-18T01:35:42.339274+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:44:14.536296+00:00 (updated-by): Updated: section:effort_risk
