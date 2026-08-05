---
id: HI09b
order: 0
plan: hip-autotune
state: pending
created-at: '2026-08-04T18:40:00+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: M
---

# Parse compiler resource reports and blacklist unviable geometries

## Description

Derived from prework plan section 17.2, which the HI01–HI16 decomposition
dropped. After the full-max compile audit (HI09), parse the compiler's
kernel-resource-usage output, identify geometries that spill registers or
exceed occupancy targets, and emit a blacklist the catalog consumes — **before**
any runtime tuning begins.

HI15 currently lists the blacklist as production hardening. That is too late.
A spilling MMVQ geometry that reaches the tuner costs a full measurement cycle
— warmup, screening, correctness validation — to establish something the
compiler already reported at build time. With a bounded matrix of ~1760 MMVQ
candidates per staged type, that is not a rounding error.

## Steps

- Build full-max with `GGML_HIP_EXPORT_METRICS=ON` (already an upstream option;
  it sets `-Rpass-analysis=kernel-resource-usage --save-temps`)
- Parse the resource remarks: per-kernel VGPR/SGPR/AGPR counts, scratch/spill
  bytes, LDS bytes, occupancy
- Map each mangled kernel symbol back to its candidate stable name — the
  generated MMVQ instances are one candidate per translation unit, which makes
  this tractable
- Classify: `spill` (scratch > 0), `low_occupancy` (below an architecture
  threshold), `lds_overflow`
- Write `artifacts/<rev>/resource-report.json` and a blacklist keyed by
  (stable_name, architecture)
- Feed the blacklist into `autotune_catalog.py` so blacklisted candidates are
  excluded from the manifest, not merely skipped at runtime
- Populate the existing `candidate_blacklist` SQLite table for tune builds
- Report totals per architecture, so a geometry that is fine on gfx1100 and
  spills on gfx1201 is visible as such

## Detailed Solution & Technical Design

Excluding at catalog level rather than at runtime is the important choice. A
runtime skip still pays the compile cost and still carries the candidate in the
registry; a catalog exclusion removes the translation unit entirely, which is
the actual control for the compile-explosion risk in plan 17.1.

The blacklist is per (stable_name, architecture) because resource pressure is
architecture specific — RDNA3 and RDNA4 have different register files, and CDNA
has AGPRs that RDNA does not.

## Code Samples & Guidance



## Files

- `tools/bigcherry/resource_report.py` — new
- `tools/bigcherry/autotune_catalog.py` — consume the blacklist
- `sql/dispatch-db.sql` — `candidate_blacklist` already exists

## Validation

- A geometry known to spill is present in the report and absent from the
  subsequent manifest
- Blacklist is architecture specific — the same geometry can be blacklisted on
  one target and kept on another
- Re-running generate with an unchanged blacklist produces an unchanged
  manifest hash
- Every blacklisted candidate has a recorded reason

## Effort & Risk

Low risk, entirely offline. The main uncertainty is the stability of the
compiler remark format across ROCm versions — parse defensively and fail loudly
rather than silently blacklisting nothing.

## Standards

HIP_AUTOTUNE_STANDARDS sections 2.5 (one source of truth), 6.3 (profiling
builds are separate), 12.4 (hard eligibility before launch)

## Acceptance Criteria



## Notes

Sequenced between HI09 and HI10 so the blacklist exists before the first
record/tune cycle.

## Ledger-events

- chg_20260805_053447_validated-the-tuning-fixes-on_5741
- 2026-08-05T05:34:47.710940+00:00 (updated-by): Updated: section:ledger-events

## Change Log

- 2026-08-05T04:29:13.986677+00:00 (updated-by): Updated: section:ledger-events

## Ledger-events

- chg_20260805_054935_kernel-variants-that-require-d_7547
- 2026-08-05T05:49:35.668059+00:00 (updated-by): Updated: section:ledger-events
