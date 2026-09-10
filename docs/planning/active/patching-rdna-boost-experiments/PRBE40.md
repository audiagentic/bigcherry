---
id: PRBE40
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:15.028499+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-KVPROJ-001: Fuse attention K and V projection into one MMVQ dispatch

## Description

Implement and qualify AMD-KVPROJ-001 for concatenated attention K/V projection in one MMVQ dispatch on compatible gfx1151/gfx1201 small-model paths. Preserve the two-projection fallback for incompatible layouts, metadata, or model sizes.

## Steps

1. Characterize the K/V weight and head shapes, quantization types, and metadata required for safe concatenation at model load. 2. Add a representation that concatenates compatible K and V weights without changing per-projection semantics, and emit one MMVQ dispatch for the paired projection. 3. Gate on supported gfx1151/gfx1201 hardware, small-model/head-dimension conditions, and compatible tied/untied metadata. 4. Add tests comparing individual K and V outputs to the two-matmul baseline, including incompatible-layout fallback. 5. Benchmark 0.5B-4B controls and 27B+ controls, recording occupancy, launch count, TG timing, memory use, and load complexity.

## Detailed Solution & Technical Design

Concatenate compatible attention K/V projection weights during model-load preparation and simplify graph emission to one MMVQ dispatch. The fused representation must retain separate logical K and V outputs, scales, offsets, quantization metadata, and head ordering so each output is individually identical to the baseline. Apply only where the source and destination layouts are exactly compatible; do not force the optimization on large FFN-dominated models or tied/incompatible metadata. Keep the original two-dispatch implementation as the fallback.

## Code Samples & Guidance

Trigger: small Qwen 0.5B-4B attention layers on gfx1151/gfx1201 with compatible K/V shapes and types. Controls: 27B+ FFN-dominated models, incompatible K/V layout, tied metadata, unsupported head dimensions, and non-target hardware. Boundary: model-size/head-dimension gate and memory/layout complexity.

## Files

Model-load K/V concatenation and attention projection graph emission; MMVQ dispatch and metadata tests; small/large-model benchmark manifests and evidence for AMD-KVPROJ-001.

## Validation

Correctness: K and V outputs separately match the two-matmul baseline across representative sequence lengths, dtypes, quantization, and head shapes. Safety: incompatible layout/metadata/hardware must use the two-dispatch fallback. Performance: report MMVQ occupancy, launch count, TG/kernel timing, load memory overhead, and variance for 0.5B-4B; confirm no regression on 27B+ controls. Acceptance: promote only if correctness is exact and memory/layout complexity remains low with a repeatable benefit.

## Effort & Risk

M; model-load representation and graph emission touch multiple layers, with risk of metadata/head-order mistakes. Retain fallback and test both output streams independently.

## Standards

Use the repository campaign/evidence protocol, preserve quantization metadata and logical K/V identity, and do not claim a large-model win without the specified 27B+ control.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: RD48
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd48

Supersedes RD48. Source is AMD PR Set 4/#59 (discussion #26378), with source status recheck required before implementation.

## Change Log

- 2026-09-09T10:56:15.028499+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:29.738621+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.310431+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.070903+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:03:05.042679+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- chg_20260910_030318_repaired-three-patching-succes_9681
- 2026-09-10T03:03:18.998203+00:00 (updated-by): Updated: section:ledger-events
