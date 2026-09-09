# Agent prompt — BigCherry planning capability rebaseline v3

Repository: `audiagentic/bigcherry`
Source branch: `planning-refactor`
Frozen semantic source: `4ff1e93dd39997f9d9ec206273ad3f444eadda7e`

Execute the v3 planning rebaseline using this pack. Treat `SOURCE_LOCK.json`, `REVIEW_PROTOCOL.md`, and generated manifests as authoritative migration controls.

Rules:

1. Review **every** lifecycle plan item at the frozen commit; no sampling and no seven-item-only queue.
2. Continuing active work receives a new capability-native successor ID. Old IDs remain historical predecessors.
3. Final active namespaces must be semantically designed under `build-*`, `run-*`, `patching-*`, or `tuning-*`; do not merely prefix old folder names.
4. Split only at independent acceptance boundaries; merge only when several predecessors are fragments of one acceptance boundary.
5. Preserve historical reviews, evidence and ledger events on predecessors. Successors reference inherited history; never transplant provenance.
6. Classify references semantically. Rewrite active dependencies/followups/scope to successors; preserve historical evidence/review/decision references to predecessors. No global ID replacement.
7. Use `ag-planning` for plan create/update/state transitions. Do not directly move/create plan files for lifecycle operations.
8. Add reciprocal visible lineage: successor `Supersedes:`; predecessor `Superseded by:`; both include migration ID.
9. Do not terminally transition any predecessor until all successor definitions exist, lineage is reciprocal, active references are resolved, and `validate_manifests.py --phase preapply` passes.
10. Never delete predecessors.
11. Use `ag-ledger` for substantive migration events with affected old and new `plan-item-ids`.
12. Do not use git stash/reset/rebase. Preserve unrelated shared-worktree changes.
13. Do not invent GPU/performance evidence or reinterpret historical evidence under a new plan ID.

Order:

`generate inventory -> design namespaces -> semantic disposition of all items -> successor specs -> lineage/dependency/reference maps -> review validation -> create/allocate successors -> exact reference migration -> preapply validation -> predecessor lineage updates -> terminal transitions -> post-apply validation -> ledger synchronization`

Do not begin lifecycle mutation until the complete disposition/successor graph is reviewed.
