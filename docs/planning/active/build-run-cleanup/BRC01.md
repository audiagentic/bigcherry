---
id: BRC01
order: 1
plan: build-run-cleanup
state: pending
created-at: '2026-09-11T06:44:31.138556+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Remove legacy build/run shims and consolidate bridging layers

## Description

GPT gateway audit found the toolset directionally coherent but still overly bridged. The primary defect is substantive legacy _legacy_* source/patch ownership in bigcherry.__main__, called by bigcherry.cli.source and bigcherry.cli.patch. Root facades inventory.py, patcher.py, and replay_cache.py are pure compatibility aliases and should be retained pending independent retirement proof. Root smoke tooling and duplicate rocprof paths require consolidation-first analysis.

## Steps

1. Inventory legacy shims and bridging modules. 2. Classify each as removable, retain-until-migration, or canonical. 3. Define file-level cleanup and consolidation sequence. 4. Implement only after design review and focused validation.

## Detailed Solution & Technical Design

Initial GPT assessment (audited main at 644b8207d11eaafa14c34f29fe4dc6c95388911d): Tranche 1 should invert cli -> __main__ ownership. Move source command behavior into cli/source.py or canonical lower-level owners; make cli/patch.py the sole CLI patch command owner while reusing bigcherry.patch.apply; reduce __main__.py to entrypoint compatibility wiring only; add static enforcement that bigcherry.cli.* cannot import bigcherry.__main__. Do not delete any shim immediately. Later work: establish canonical campaign equivalents before consolidating e2e_smoke_campaign.py/e2e_smoke_report.py; reconcile rocprof.py only after direct-consumer and parity scans; retire inventory.py, patcher.py, and replay_cache.py independently and last.

## Code Samples & Guidance



## Files

Tranche 1: tools/bigcherry/__main__.py; tools/bigcherry/cli/source.py; tools/bigcherry/cli/patch.py; tools/bigcherry/cli/main.py; tools/bigcherry/check.py; tools/tests/core/test_compatibility_facades.py; focused CLI/source/patch tests. Successor scans: tools/bigcherry/e2e_smoke_campaign.py; tools/bigcherry/e2e_smoke_report.py; tools/bigcherry/rocprof.py; tools/bigcherry/profiling/rocprof.py; tools/bigcherry/inventory.py; tools/bigcherry/patcher.py; tools/bigcherry/replay_cache.py; TOOL_DISPOSITION.md.

## Validation

Consumer proof across Python imports, direct script invocations, shell/CI, docs, and tests. Verify cli.source and cli.patch import without loading __main__. Preserve facade identity tests. CLI parity for python -m bigcherry and source/patch commands including help, aliases, exit codes, errors, streams, and cwd behavior. Compare evidence/receipt digests, SHAs, normalized paths/order, build/run identity, and failure receipts before/after. Exercise replay cache hit/miss and lock semantics separately. Run smoke campaign regression tests. Use disposable fixtures and assert HEAD, branch, index, and unrelated worktree state are unchanged.

## Effort & Risk



## Standards



## Acceptance Criteria

No immediate shim deletion is assumed. The first implementation tranche removes substantive source/patch ownership from __main__, makes cli handlers canonical, preserves supported entrypoint behavior and evidence/replay identity, and adds forbidden-backedge enforcement. Later shims are retired only after independent consumer, parity, CLI, and disposition evidence.

## Notes

Initial assessment and design to be supplied by GPT gateway review before implementation.

GPT gateway initial response: audited main at 644b8207d11eaafa14c34f29fe4dc6c95388911d; verdict directionally coherent but still overly bridged. Recommended order: (1) cli/__main__ ownership inversion; (2) reduce __main__ to compatibility shell; (3) enforce no cli -> __main__ back-edge; (4) consolidate smoke/report; (5) reconcile rocprof; (6) retire root facades last. Safe to remove now: none. Root facades are intentional module-identity aliases. The audit also noted BRC01 was not visible on the audited tip, which is why this plan must be committed and pushed before the second review.

## Change Log

- 2026-09-11T06:44:31.138556+00:00 (created-by): Created by agent

## Ledger-events


- chg_20260911_064900_added-a-tracked-cleanup-plan-f_3622
- 2026-09-11T06:49:00.213062+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-11T07:01:03.606121+00:00 (updated-by): Updated: section:description, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260911_070115_recorded-gpts-cleanup-assessm_2226
- 2026-09-11T07:01:15.274980+00:00 (updated-by): Updated: section:ledger-events
