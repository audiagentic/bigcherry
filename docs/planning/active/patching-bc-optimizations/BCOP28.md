---
id: BCOP28
order: 28
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T10:10:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: S
---

# Track 0300 migration to upstream-owned single-source MMQ J

## Description

Action ledger for the BCOP23 re-audit. The current 0300 forced-J patch still lifts and owns the native J scan because that matched the older upstream MMQ design. Open upstream #29953 moves native J resolution before allocation and passes the resolved J through `mmq_args`; retaining 0300's scan after that change would recreate the allocation/launch split that #29953 fixes.

BCOP23 is the implementation authority for this migration. This item records disposition only and must not grow a second MMQ design.

## Actions

1. Verify the current composed pin contains merged #29941's MoE padding correction.
2. Qualify the #29953 ownership model on the current pin with the existing 0300 host-fixture surface before GPU tuning.
3. When adopting that model, remove 0300's `ggml_cuda_mmq_native_j_best`/lifted native scan and make native mode consume the upstream-resolved J; retain only forced override plus hard/tail eligibility.
4. Require host proof + patch apply/idempotence + real HIP build + gfx1100/gfx1201 lopsided-MoE/Flash-Next execution.
5. Record one terminal disposition here: adopted upstream/final #29953; locally qualified pending upstream; or rejected because final upstream ownership changed. Delete disposable prototype code on rejection.

## Acceptance Criteria

- No duplicate native J selector remains between upstream MMQ and patch 0300.
- Allocation and launch consume one resolved J for dense and MoE shapes.
- Existing forced-J experimentation remains possible without weakening padding/tail safety.
- QFP10 performance work is unblocked only after correctness/build/hardware gates pass.

## Related

BCOP23 (technical authority); QFP10; patch 0300; llama.cpp #29941/#29953.
