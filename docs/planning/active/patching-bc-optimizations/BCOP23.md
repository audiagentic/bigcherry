---
id: BCOP23
order: 23
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T05:08:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: S
---

# Track single-source MMQ J ownership migration

## Description

Action/disposition ledger for the MMQ J allocation/launch ownership issue. QFP10 and patch `0300_mmq_forced_j` are the BigCherry technical owners; upstream MMQ owns native/default J selection. BCOP23 no longer carries a second implementation design.

The unresolved requirement is simple: once upstream resolves one native `J_best` before allocation, patch 0300 must consume that resolved value in native mode and retain only an experimental forced override with eligibility/tail-safety checks.

## Actions

1. Verify the current pin's upstream MMQ J ownership and #29941-equivalent MoE padding safety.
2. Use the existing 0300 host fixture to prove allocation-J equals launch-J and that native mode does not rescan policy.
3. If the upstream ownership model is present/adopted, remove 0300's lifted native selector and duplicate policy tests; keep forced override only.
4. Require host proof, patch apply/idempotence, real HIP build and gfx1100/gfx1201 execution before QFP10 performance tuning resumes.
5. Record terminal disposition: `upstream-owned`, `local-qualification-pending-upstream`, or `rejected-upstream-changed`.

## Consolidation

BCOP28 duplicates this follow-through and should be treated as superseded by this single ledger. No second selector, padding estimator or dispatch table is authorized.

## Related

QFP10; patch 0300; llama.cpp #29941/#29953 lineage; BCOP28 (superseded duplicate ledger).
