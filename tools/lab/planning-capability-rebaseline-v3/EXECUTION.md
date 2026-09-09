# Execution procedure

## 0. Source lock

The semantic source is fixed by `SOURCE_LOCK.json` at commit `4ff1e93dd39997f9d9ec206273ad3f444eadda7e`.

If newer planning changes must be included, stop and intentionally create a new migration source lock. Do not silently rerun against branch HEAD and mix epochs.

## 1. Generate inventory

Run `generate_inventory.py` from the repository. It emits:

- `PLAN_INVENTORY.csv`;
- `PLAN_REFERENCES.tsv`;
- pre-populated `DISPOSITIONS.csv` with `UNREVIEWED` decisions;
- template copies for namespaces, successors, lineage, dependency remaps and reference decisions.

The generator reads Git objects only. It does not mutate the worktree.

## 2. Review namespaces first

Fill `NAMESPACES.csv` before assigning successors.

Requirements:

- capability is exactly `build`, `run`, `patching`, or `tuning`;
- namespace starts with `<capability>-`;
- namespace is semantically meaningful, not merely the old folder prefixed with a capability;
- `id_prefix` is uppercase, unique, and appropriate for the namespace;
- every row must be explicitly approved.

This prevents allocating IDs against a namespace taxonomy that is still moving.

## 3. Review every predecessor

Fill every row of `DISPOSITIONS.csv`.

For continuing work, assign one or more stable pack-local `successor_keys`. These are not plan IDs. They let the complete graph be reviewed before consuming/creating repository IDs.

Write one successor spec per key in `successor-specs/<key>.md` using the template.

Populate:

- `SUCCESSORS.csv` with target namespace, title, acceptance boundary and spec path;
- `LINEAGE.csv` with every predecessor->successor edge;
- `DEPENDENCY_REMAP.tsv` for active plan dependencies;
- `REFERENCE_DECISIONS.tsv` for every reference that cannot be mechanically proven historical/active.

At this stage `allocated_id` may be blank.

## 4. Review gate

Run:

`validate_manifests.py --phase review`

It must prove:

- every source item has exactly one approved disposition;
- every continuing item has lineage;
- every successor has predecessor lineage;
- split/merge cardinality agrees with disposition;
- every target namespace is approved and capability-consistent;
- successor specs exist;
- no source snapshot drift;
- no duplicate successor keys;
- no unclassified/ambiguous forward dependency remains.

Only after this passes should IDs be allocated/created.

## 5. Allocate/create successors through ag-planning

Use `plan_create_item`; do not create plan files directly.

Create all successor items while predecessors are still unchanged/non-terminal. Record the returned IDs in `SUCCESSORS.csv`.

Then use `plan_update_item` to apply the reviewed successor specs and explicit `Supersedes:` lineage notes.

Do not claim inherited evidence as new successor evidence.

## 6. Resolve dependencies and references

Fill any target values that depended on allocated IDs.

Forward-looking dependencies/references move to successors. Historical evidence/review/decision references remain on predecessors.

Repository source/reference edits should be exact and reviewed; no global old-ID replacement.

## 7. Preapply gate

Run:

`validate_manifests.py --phase preapply`

Additional requirements:

- every successor has a syntactically valid allocated ID;
- allocated IDs are unique;
- every reference/dependency decision is resolved;
- no active dependency intentionally targets a predecessor marked for supersession without an explicit exception;
- old and new lineage is complete enough to render terminal operations.

Then render `PLAN_OPS.jsonl`.

## 8. Update predecessor lineage, then retire

For every continuing predecessor:

1. `plan_update_item` old item to add `Superseded by:` and migration note.
2. Verify successor exists and carries reciprocal `Supersedes:` note.
3. Apply exact non-plan repository reference rewrites.
4. Only then `plan_set_state(old_id, 'superseded')`.

For genuine retirements use the appropriate supported terminal state (`completed`/`deprecated`) through `plan_set_state`.

Never use `plan_delete_item` for migration cleanup.

## 9. Ledger

Record change events with `ag-ledger record_change_event`, `status: unreleased`, and all affected `plan-item-ids`.

Recommended commit/event boundaries inside the single migration branch:

1. pack/inventory + reviewed manifest;
2. successor creation/content + reciprocal lineage;
3. exact dependency/reference migration;
4. predecessor terminal transitions + semantic cleanup;
5. validation fixes if required.

Do not merge intermediate states to `main`.

## 10. Post-apply validation

Required invariants:

- every frozen source item has exactly one disposition;
- every continuing predecessor is terminal and has successor linkage;
- every successor has predecessor linkage;
- split/merge edges are reciprocal;
- every active plan lives in an approved `build-*`, `run-*`, `patching-*`, or `tuning-*` namespace;
- every active plan's `plan:` matches its namespace;
- no active dependency points to a superseded predecessor when a successor exists, unless explicitly documented as historical/exceptional;
- historical evidence/review references still resolve to predecessor identity;
- predecessor reviews/evidence/old ledger associations were not reassigned;
- no predecessor was deleted;
- normal planning/reference validation passes;
- repository tests required by changed reference/tooling files pass.
