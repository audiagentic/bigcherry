---
id: RDR02
order: 12
plan: run-docs-reference
state: completed
created-at: '2026-09-09T10:47:27.966718+00:00'
breadth: ''
skill: basic
created-by: capability-rebaseline-v3
work: M
priority: P3
---

# Migrate docs from hardcoded host facts to environment roles

## Description

The environment mechanism exists, but the source explicitly leaves the mechanical multi-file prose/path migration undone.

## Steps

1. Implement the still-valid future scope.
2. Run the stated acceptance and evidence gates.
3. Preserve predecessor provenance and record successor evidence under this ID.

## Detailed Solution & Technical Design

Capability owner: run

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

Mechanical prose/path pass over docs/** (scope determined by RDR01's closure audit -- live/maintained docs only, per the corrected notes above). config/environment.toml, tools/env/bigcherry-env.sh, docs/reference/ENVIRONMENT.md are the existing mechanism being pointed at, not files this item edits.

## Validation

grep -rc for the hostname, /home/, /mnt/vault, and hardcoded IPs across docs/ before and after -- RE-COUNT against current HEAD, the original DR01 baseline (83/64/39/15) is stale and must not be reused verbatim. `source tools/env/bigcherry-env.sh` then run a doc's command block verbatim and confirm it still works. Remaining occurrences must each be justifiable as genuinely host-identity-specific (immutable historical evidence, ssh config examples).

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: DR01
Migration: capability-rebaseline-v3-2026-09
Successor key: run-docs-reference-dr01

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): GO, after RDR01. Design (the 5-step prose/path migration already specified) is sufficient as-is. Add one explicit rule not previously stated: migration applies to MAINTAINED/LIVE docs only, not to immutable historical evidence/archive/campaign artifacts whose literal host/path/IP values are themselves provenance — those may legitimately keep Brutus-identity statements. No backward-compat aliases for old hardcoded paths (project doctrine: migrate up fully). Execution order: ranked #12, last — depends on RDR01 first establishing which files are live vs immutable evidence, so the mechanical pass doesn't touch the wrong set.

CORRECTION from deeper repo-validated dev-gpt review (2026-09-10): still valid after RDR01, but do NOT blindly reuse the original occurrence counts (83x 'brutus', 64x '/mnt/vault', etc.) quoted in DR01 -- those are stale. Recount against current HEAD before scoping the mechanical pass. Execution order stays #12 (unchanged), last.

RE-COUNTED 2026-09-10 against current HEAD, per the deeper review's explicit instruction not to reuse DR01's stale 83/64/39/15 baseline: those old counts included historical docs/evidence/**and docs/planning/** material that RDR01's closure audit already confirmed is correctly out-of-scope (immutable historical record, not live reference). Isolated to the actual in-scope corpus (docs/reference/**, matching RDR01's own classification boundary): raw 'brutus' appears 8 times across only 3 files (ENVIRONMENT.md, build/DEPENDENCIES.md, tooling/TOOL_DISPOSITION.md) -- not 83 across 40. Reviewed every one individually against DR01's own rule (replace when the sentence is about the ROLE; keep when genuinely about that one machine's identity/evidence):
- ENVIRONMENT.md:52 `ssh brutus '...'` -- NOT a violation, it is the deliberate BEFORE half of an existing before/after pair immediately followed by the correct `$BC_HOST` form; left as-is.
- ENVIRONMENT.md:90, TOOL_DISPOSITION.md:149/155 -- literal script filenames (tmp/brutus-probe.sh, tmp/h36-brutus-pipeline.sh) containing 'brutus' as part of their actual name; not host-identity prose, left as-is.
- DEPENDENCIES.md:56, TOOL_DISPOSITION.md:552 -- genuine historical/evidence statements ('RD87 built and executed a working client on Brutus') recording what was actually done on that real machine; kept per DR01's own exception for identity-specific statements.
- DEPENDENCIES.md:21 `/home/audumla/rocm-shim` -- FIXED to `$BC_HOME/rocm-shim`.
- DEPENDENCIES.md:33 'For Brutus, also verify...' -- FIXED to 'On the build server, also verify...' (procedure prose about the role, not the specific machine).
No hardcoded IPs or /mnt/vault occurrences found in docs/reference/** at all (the earlier full-docs/ IP count of 171 was almost entirely timestamps like '0.00.374.130' and legitimate '127.0.0.1' loopback references in testing/TEST.md -- not host-identity leaks). Ports (8080/18400) not found hardcoded in docs/reference/** either.

VERDICT: this item's ACTUAL residual scope, after RDR01 correctly narrowed what counts as live reference, was two small fixes -- both applied. DR01's original 'mechanical pass over 40 files' framing no longer matches current reality; the corpus shrank because RDR01's own classification work (and general doc hygiene since DR01 was filed 2026-09-06) already did most of what this item was for. CLOSED AS SATISFIED -- no further mechanical pass needed against docs/reference/**.

## Change Log

- 2026-09-09T10:47:27.966718+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:03:28.103636+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.745382+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:07:57.299621+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:08:10.126436+00:00 (updated-by): Updated: order=12, priority='P3'
- chg_20260910_000828_reviewed-and-re-planned-all-pe_3612
- 2026-09-10T00:08:28.911837+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.190357+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:19:10.765467+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:27:39.763516+00:00 (updated-by): Updated: section:files, section:validation
- 2026-09-10T00:32:24.971580+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:32:25.300266+00:00 (state-transition): State: pending → completed
- chg_20260910_003230_closed-the-host-facts-to-envir_3349
- 2026-09-10T00:32:30.672761+00:00 (updated-by): Updated: section:ledger-events
