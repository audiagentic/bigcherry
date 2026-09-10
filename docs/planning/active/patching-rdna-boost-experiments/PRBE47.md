---
id: PRBE47
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:45.120291+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# UP-HIP-002: AMD DPP/native shuffle path

## Description

Evaluate AMD RDNA DPP/native shuffle replacements in targeted HIP reduction helpers, with exact output parity and no blanket intrinsic substitution.

## Steps

1. Recheck llama.cpp PR #26466 and identify reduction/shuffle helpers where RDNA DPP expresses the same lane pattern. 2. Implement architecture-guarded DPP paths for gfx1100/gfx1201 and preserve generic shuffle for non-RDNA or unsupported patterns. 3. Add exact output tests and ISA inspection. 4. Benchmark targeted Q6/Q8 decode kernels with instruction counts, DS usage, TG timing, and E2E decode controls. 5. Adopt only per-kernel paths where generated ISA and runtime both improve.

## Detailed Solution & Technical Design

Specialize selected reduction/shuffle helpers to native DPP instructions on RDNA, but keep the generic implementation whenever lane topology cannot be expressed exactly. Selection is per helper/kernel and architecture guarded; this is not a blanket intrinsic replacement.

## Code Samples & Guidance

Trigger: Qwen Q6/Q8 decode hot kernels on gfx1100/gfx1201. Controls: non-RDNA architectures and kernels where DPP cannot express the required lane pattern. Boundary: DPP versus generic per kernel; inspect ISA.

## Files

HIP reduction/shuffle helpers and architecture selectors; exact output tests; generated ISA reports; targeted decode benchmark manifests/evidence for UP-HIP-002.

## Validation

Correctness: exact output parity across targeted kernels and representative shapes. Performance: report instruction counts, DS usage, TG/kernel timing, and E2E decode with interleaved controls. Acceptance: enable only where generated ISA and runtime both improve; no blanket replacement.

## Effort & Risk

M; lane semantics and compiler lowering can differ by target. Keep generic fallback and inspect generated code.

## Standards

Preserve non-RDNA portability, exact reduction semantics, fail-closed selectors, and evidence provenance.

## Acceptance Criteria

Acceptance requires exact output parity, generated-ISA and DS-use evidence, repeatable targeted-kernel and E2E improvement, and generic fallback on unsupported architectures/patterns; no blanket replacement.

## Notes

Supersedes: RD56
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd56

Supersedes RD56.

## Change Log

- 2026-09-09T10:56:45.120291+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:58.680914+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.341721+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.115748+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:08:59.693148+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-10T03:09:19.555246+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20260910_030930_repaired-two-more-active-patch_7368
- 2026-09-10T03:09:30.744289+00:00 (updated-by): Updated: section:ledger-events
