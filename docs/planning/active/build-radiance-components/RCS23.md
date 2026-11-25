---
id: RCS23
order: 0
plan: build-radiance-components
state: pending
created-at: '2026-11-25T00:54:41.130343+00:00'
breadth: ''
skill: advanced
created-by: lead
work: L
priority: P1
---

# Owned kernels as overlay plugins (libbc-<family>-<arch>.so) with an origin manifest and an upstream acceptance transaction

## Description

Take ownership of the kernels we have rewritten heavily (the IQR GEMM, prefill, head and embedding units) without forking libr4d and without substituting source units inside upstream's build. The engine's loader already composes kernel libraries by hierarchy (the libbcgfx12 plugin uses it), so owned families are built as overlay plugins placed first in the hierarchy; upstream's libraries stay what upstream builds. Measured upstream churn on the first shadowed files: 14 commits in 12 releases over 30 days (the row-table file, 29 commits, changes what an unchanged kernel means and is the main exposure).

## Steps

0. Baseline: per-architecture snapshot of build units, row catalogues, callback dependencies and selected behaviour (gfx1030 exclusions included). 1. Inventory: per libr4d unit, the share of lines we rewrite or patch, the patches touching it, and upstream commits per unit per release. 2. Manifest and tooling: overlay families with upstream origins (path, base commit, base blob, last integrated commit), build slots and rows overridden, architectures, absorbed patches and rewrites, integration strategy per origin (whole-unit: three-way merge applies; extracted-family: structural review); provenance verifier; dry-run merge; include closure from the compiler's dependency output; row-contract snapshot of the shadowed rows between pins; fixtures: unchanged file, changed file, wrong base blob, non-ancestor target, rename, split, moved header, changed indirect header, changed row priority, unchanged kernel under a changed layout contract, clean text merge failing qualification, upstream adoption. 3. Replay the twelve release transitions one by one; every one classified, no unexplained clean result. 4. First overlay: the embedding gather (row and export equivalence, byte identity, resource parity, cold and warm no regression, removing the overlay restores the base behaviour). 5. Second overlay: the fused tile on the traits seam, both architectures. 6. The large IQR GEMM, prefill and head families after the RCS15 decision, so the migration is not done twice. 7. Two real pin bumps through the new process with verified resume and rollback. 8. Naming: sources bc_<family>_<kernel>, introduced symbols bc_ prefix (op names and row identities keep their upstream spelling), plugins libbc-<family>-<arch>.so with a provider tag and the manifest digest in the identity; the base libraries keep their names; serving hierarchy, release asset inventory and bc-release verify updated; no engine change expected (its loader takes the hierarchy from configuration). 9. Bump acceptance as a transaction: verify provenance; classify origins (no change, candidate, blocked, unverified); include closure; row-contract snapshot; composition gates 1 to 3 then qualification per overlay; advance the integrated revision only after qualification (examined revision recorded at triage); resume binds to the manifest digest, owned-source digest and candidate. 10. Adoption: change-level (retire a change upstream took, after verifying equivalence) and unit-level exit (only when every retained change is gone).

## Detailed Solution & Technical Design

Design: docs/design/radiance-kernel-composition.md section 11 (overlay model), section 12 (naming).

## Code Samples & Guidance



## Files

engines/radiance (owned unit sources, manifest, drift tool), engines/radiance/components/libr3, patches, tools/bigcherry pin triage, docs/design

## Validation

Fixtures above fail closed with the expected classification; the twelve-transition replay reproduces the measured churn with every transition classified; each overlay: row and export equivalence against the base, resource records, byte identity at every affected shape, cold and warm replay no regression, removal restores base behaviour; gates 1 to 3 (RCS19) pass for both architectures.

## Effort & Risk

Large. Main risks: semantic drift through shared headers and the row table (include closure and row-contract snapshot), upstream restructuring (fail closed on rename, split, ambiguity), a false or stale base (verified provenance, examined versus integrated revision), an overlay row that silently loses to a base row or duplicates one (hierarchy tests, row-catalogue comparison).

## Standards



## Acceptance Criteria

The IQR families are owned overlay plugins with an origin manifest, retired patches and rewrites for the shadowed units, one source for gfx11 and gfx12, and an automated, transactional upstream check at each pin bump; no clean text merge is accepted without the gates; the base libraries and the engine are unchanged.

## Notes

Depends on RCS19 (gates, ownership records) and RCS21 (traits seam); sequenced in parallel with the RCS15 baseline, which decides whether a native gfx11 provider replaces the generated tree for the unowned units. Review by GPT only (code returned as text).

Revised after a design review (2026-11-25): the first version proposed whole-file substitution inside upstream's units; the loader's existing multi-library hierarchy makes overlays simpler and reversible. Depends on RCS19 (gates, ownership records) and RCS21 (traits seam); the large gfx11 migration is gated on RCS15. GPT is review only on this work.

2026-11-25 simplification (owner: no adversaries, test discretely): provenance means git commit ids and plain comparisons; no digest binding of the manifest or owned sources, no composite identity hashes, no resume-binding to digests. The base commit check is `the commit exists and is an ancestor of the target`. Overlay identity is a name, a version string and the upstream release it was built against. The row-contract comparison stays as a test between the base and overlay rows. Per-overlay checks use the discrete harness (RCS25), not the full plugin.

## Change Log

- 2026-11-25T00:54:41.130343+00:00 (created-by): Created by lead
- 2026-11-25T00:54:59.083112+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
- 2026-11-25T00:58:03.939770+00:00 (updated-by): Updated: section:steps
- 2026-11-25T01:08:26.587516+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:detailed_solution, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
- 2026-11-25T02:12:45.058840+00:00 (updated-by): Updated: section:notes
