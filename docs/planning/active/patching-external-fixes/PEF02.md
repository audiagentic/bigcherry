---
id: PEF02
order: 0
plan: patching-external-fixes
state: pending
created-at: '2026-09-09T10:47:44.194455+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Generated q1_0 MMVQ timing failure poisons HIP context during tuning

## Description

Maintain fail-closed tuning-context poisoning containment while the historical q1_0 candidate-level fault remains unresolved.

## Steps

- Preserve durable cold-path attempt attribution with candidate, signature digest and protocol stage before risky GPU work.
- Keep failures in HIP event timing, correctness copies and stream synchronization on the terminal poison path; suppress later tuning work and replay export.
- Use deterministic fault injection and fresh native/tune q1_0 runs as containment validation, not root-cause proof.
- Resume investigation only on a naturally recurring fully attributed failure, recovered historical candidate evidence, or a proposed containment-policy change.
- Do not add a broad eligibility quarantine or claim incident closure without generated-candidate and faulting-instruction evidence.

## Detailed Solution & Technical Design

Containment is scoped to the tuner measurement protocol and does not retroactively invalidate already cached dispatcher bindings or reset a poisoned HIP context. Keep that distinction explicit; any broader dispatcher quarantine is a separate decision.

## Code Samples & Guidance



## Files

hip-autotune-dispatch.cu; hip-autotune-tuner.cu; replay_cache.py; dispatch-safety/replay-promotion tests; FINDINGS.md; durable attempt journal fixtures.

## Validation

Native and tune q1_0 45/45 or equivalent current matrix; candidate-attempt trace; deterministic injected failure; poisoned experiment suppresses work and replay; no HIP/assertion/fatal errors; candidate-level reproduction remains open.

## Effort & Risk



## Standards

Fail closed after measurement failure; preserve candidate attribution; no broad quarantine from clean non-reproduction.

## Acceptance Criteria

Containment and attribution remain verified, but PEF02 stays open until candidate/fault identity is recovered or an explicitly accepted quarantine/monitoring disposition supersedes root-cause work.

## Notes

Supersedes: EX03
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-external-fixes-ex03

## Change Log

- 2026-09-09T10:47:44.194455+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:03:44.467456+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.763558+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.219141+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:41:17.952874+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024129_build-and-external-fix-success_1105
- 2026-09-10T02:41:29.994463+00:00 (updated-by): Updated: section:ledger-events
