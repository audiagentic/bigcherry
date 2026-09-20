# PA25: audit focal evidence / G4 + exact-composition G7 for the seven promoted/touched patches

Plan item: PA25
Status: completed
Owner: PA25 (patching-patch-system)
Question state: answered

## Question

For the seven promoted/touched patches (`1225`, `0840`, `1224`, `1200`,
`1001`, `1002`, `1232`), does the recorded focal evidence (G4) and
exact-composition identity (G7) still agree with the current patch catalog and
registry state? This is a read-only consistency audit, not a re-run.

## Inputs

- The seven focal patch IDs and four named experiment contexts
  (`hi155-hybrid-allreduce-gate`, `hi18-reduce-probe`, `rd19-only`,
  `hip-internal-allreduce-only`).
- `patches/catalog.toml`, the live patch registry, and campaign resolution.

## Outputs

`audit_receipt_20260919.json` — a per-patch G4/G7 consistency snapshot
(evidence authority, state, blockers, artifact hashes) at the recorded
BigCherry + llama.cpp revisions.

## Runtime

GPU required: no
Real compilation required: no
Mutates canonical BigCherry state: no (read-only audit)

## Safety

- Read-only: resolves and reports identities; never writes campaign records.
- Not production tooling or evidence authority — the receipt is a diagnostic
  snapshot for the PA25 audit.

## Disposition

Retained (TRANSITIONAL) as the PA25 audit driver + receipt; diagnostic-only,
not a maintained tool or evidence authority.
