---
id: PNRO10
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:52:53.458006+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P1
---

# Add ctx_other model devices to speculative scheduler backends

## Description

Add ctx_other model devices to speculative scheduler backends -- orchestration correctness fix for shared tensors under speculative decoding. IMPLEMENTED-AS-PATCH: patches/1261_nro10_spec_ctx_other_devices exists (state=untested); the plan item's own ledger-events already show a 2026-09-20 note '1261-pnro10-verified-o...' indicating prior verification activity. Disposition: validate/qualify existing patch.

## Steps

- Reproduce target/draft device-list mismatch with shared tensors under speculative configuration.
- Enumerate model_other->devices, deduplicate by backend device handle, and initialize only missing backends after ordinary model devices.
- Preserve backend ordering relative to ACCEL/CPU and verify ownership/destruction of added instances.
- Test same-device no-op, subset/superset/disjoint lists, Meta-wrapped tensor split and single-GPU cases.
- Verify shared tensors schedule on an allocator-valid backend with output parity and no unexpected copies; initialization failure must be explicit.

## Detailed Solution & Technical Design

Device identity comes from the initialized other model, not CLI guesses. This is not a speed patch; backend lifetime/order and shared allocation validity are the acceptance boundary.

## Code Samples & Guidance



## Files

src/llama-context.cpp context backend construction; speculative ctx_other setup; scheduler/shared-tensor integration tests; device-list fixtures.

## Validation

Patch mechanics: `PYTHONPATH=tools python -m bigcherry patch-lint patches/1261_nro10_spec_ctx_other_devices`; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay 1261_nro10_spec_ctx_other_devices --source bigcherry-tuning`; package pytest offline. Fixtures: mismatched-device control vs subject, same-device no-op, subset/superset/disjoint lists, Meta-wrapped tensor split, single-GPU. Static/unit-level output/copy-topology and lifetime checks (no hardware required for most of this item since it's an orchestration/scheduling correctness fix, not a kernel). Hardware confirmation (Brutus, 2+ device speculative config): `python -m bigcherry.patch.validation_campaign --overlay 1261_nro10_spec_ctx_other_devices --arch gfx1100` verifying shared tensors schedule on an allocator-valid backend with output parity and no unexpected copies; initialization failure must be explicit.

## Effort & Risk



## Standards

Affirmative scheduler/output evidence; fail explicitly on missing required backend; no throughput claim.

## Acceptance Criteria

Shared tensors execute on valid allocation backends under mismatch; no duplicate backend or lifetime leak; controls remain unchanged.

## Notes

Supersedes: NRO11
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro11

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH. patches/1261_nro10_spec_ctx_other_devices exists, state=untested. Item's Files section still names raw src/llama-context.cpp paths (pre-patch-package phrasing) but the mapped package already exists (Successor key nro11 / Supersedes NRO11 in Notes, id verified in patches/1261*/patch.toml). No upstream absorption found relevant to this specific ctx_other device-list orchestration bug. Disposition: validate/qualify existing patch; no GPT design needed.

## Change Log

- 2026-09-09T10:52:53.458006+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:09:20.280241+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.094454+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.745657+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:44:11.869473+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024433_three-more-nasone-successors-n_7555
- 2026-09-10T02:44:33.151338+00:00 (updated-by): Updated: section:ledger-events
- chg_20260920_064350_patch-1261-pnro10-verified-o_7857
- 2026-09-20T06:43:55.564429+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:26:37.993604+00:00 (updated-by): Updated: section:description, section:validation, section:notes
