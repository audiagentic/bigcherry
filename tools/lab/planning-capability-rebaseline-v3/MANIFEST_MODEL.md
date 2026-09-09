# Manifest model

The migration manifest is normalized across review-friendly files rather than one wide CSV with multi-valued fields.

## Authorities

| File | Cardinality | Authority |
| --- | --- | --- |
| `PLAN_INVENTORY.csv` | 1 row / frozen item | immutable source identity, state, path, content hash |
| `LIFECYCLE_NORMALIZATION.csv` | 1 row / frozen item | explicit continuing vs terminal/no-successor vs adjudicate gate before successor allocation |
| `NAMESPACES.csv` | 1 row / final plan namespace | capability ownership and ID prefix |
| `DISPOSITIONS.csv` | exactly 1 row / frozen item | one final semantic disposition |
| `SUCCESSORS.csv` | 1 row / new plan | target namespace, allocated ID, acceptance boundary, spec |
| `LINEAGE.csv` | 1 row / predecessor-successor edge | supports 1:1, 1:N and N:1 without lossy list fields |
| `DEPENDENCY_REMAP.tsv` | 1 row / explicit dependency decision | active dependency migration |
| `REFERENCE_DECISIONS.tsv` | 1 row / source occurrence requiring review | historical preserve vs active rewrite/remove; deterministic occurrence/context hashes and semantic action |
| `successor-specs/*.md` | 1 file / successor | reviewed future scope/content passed to `plan_update_item` |

`DISPOSITIONS.csv` is the coverage authority: every frozen item must appear exactly once. `LINEAGE.csv` is the graph authority: prose lineage is required for discoverability but does not replace graph validation.

## Disposition graph

`successor`:

`OLD --supersedes--> NEW`

`split`:

`OLD --split_from--> NEW-A`

`OLD --split_from--> NEW-B`

`merge`:

`OLD-A --merged_from--> NEW`

`OLD-B --merged_from--> NEW`

Retirement/history dispositions have no lineage edge.

## Stable pack-local keys

`successor_key` exists so semantic review can finish before repository IDs are consumed. Use a descriptive lowercase key such as `run-qualification-matrix` or `tuning-replay-qualification`. It must remain stable for the migration even after `allocated_id` is known.

References and dependencies may target a `successor_key` during review. After creation, `allocated_id` becomes the repository identity; `render_operations.py` resolves keys to allocated IDs.

## Source hashes

Each disposition row repeats the frozen plan's SHA-256 content hash. The validator compares it to both `PLAN_INVENTORY.csv` and `git show <source_commit>:<path>`. This prevents an agent from accidentally reviewing one revision and retiring a different one.

Each reference occurrence is pinned by a deterministic 16-hex occurrence ID and a SHA-256 hash of its frozen context. Semantic classes (`active_scope`, `historical_provenance`, `identity_declaration`, `literal_example`, or `unclassified`) and actions (`rewrite_to_successor`, `preserve_predecessor`, `remove`, `no_change`, or `unclassified`) are explicit. Unclassified or ambiguous occurrences block preapply.
