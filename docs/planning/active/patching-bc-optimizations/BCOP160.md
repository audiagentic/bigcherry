---
id: BCOP160
order: 160
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-12T10:06:14+00:00'
created-by: agent
priority: P1
work: S
---

# PEC02: require cell-bound trigger and correctness proof

The VA26 qualification executor accepts missing trigger evidence and can substitute passing correctness when required checks are absent. Positive trigger proof is not matched to the expected qualification cell. These are source-level evidence-integrity risks, not reproduced GPU failures.

PEC02 owns the correction. Require per-cell proof, strict integer counts, provenance, and required correctness before promotion. Missing/foreign/duplicate evidence is INVALID. Preserve legacy direct-call compatibility and use existing VA14/VA26 machinery; do not create another evaluator or hardware queue. The next gate is a bounded host-only negative fixture matrix followed by re-evaluation of one existing artifact, if attributable. Recent Radiance, QFP15/48/53 and release work remains protected.
