# BigCherry planning capability rebaseline v3

Migration ID: `capability-rebaseline-v3-2026-09`

Repository: `audiagentic/bigcherry`

Frozen semantic source: `4ff1e93dd39997f9d9ec206273ad3f444eadda7e` on `planning-refactor`.

## Goal

Produce one reviewed, capability-native planning epoch before any predecessor is retired:

- new active plan namespaces begin with `build-`, `run-`, `patching-`, or `tuning-`;
- continuing work is represented by newly created successor IDs;
- old items remain immutable historical predecessors and are moved to an appropriate terminal state only after their successors validate structurally;
- every replacement has reciprocal predecessor/successor lineage;
- historical reviews, evidence and ledger events remain attributed to the predecessor that created them;
- active dependencies and forward-looking references move to successors; historical references remain on predecessors;
- no lifecycle mutation occurs until the complete graph is reviewed and passes validation.

This is deliberately **not** the v2 structural move. v2's inventory/move concepts may be used as input if available, but v3 is authoritative for semantic disposition and lineage.

## Capability boundary

Read `REVIEW_PROTOCOL.md` before classifying anything.

- **Build**: source/vendor revision, recipes, compiler/toolchain, patch composition at build time, build artifact identity/provenance.
- **Run**: model/GPU/topology/runtime resolution, preflight, campaign/matrix orchestration, warmup/measurement, A/B, work-equivalence, correctness execution, diagnostics/profiling, evidence/reporting.
- **Patching**: patch identity/content/applicability/contracts/tests/validation semantics/evidence status/dependencies/conflicts/promotion/demotion.
- **Tuning**: BC-native boundary onward: dispatch recording, tuning signatures/candidates/search/ranking/cache/replay/qualification/execution attribution/winner provenance/counterfactuals/prebinding/overhead/composition.

Build and Run are shared infrastructure. Patching and Tuning define/evaluate experiments and consume Build/Run rather than embedding their implementation details.

## Pack contents

- `SOURCE_LOCK.json` — immutable migration source identity.
- `REVIEW_PROTOCOL.md` — semantic classification and lineage rules.
- `EXECUTION.md` — end-to-end migration procedure and commit/ledger boundaries.
- `AGENT_PROMPT.md` — compact prompt for an implementation/review agent.
- `templates/` — normalized review manifests; do not invent parallel formats.
- `successor-specs/` — put reviewed successor content here, one spec per `successor_key`.
- `scripts/generate_inventory.py` — reads the frozen Git object without checking it out; emits inventory, references, and pre-populated dispositions.
- `LIFECYCLE_NORMALIZATION.csv` — explicit continuing/terminal/adjudicate gate; active metadata is not treated as proof that a successor is required.
- `scripts/seed_review_manifests.py` — generates an explicitly unapproved draft disposition/successor/lineage graph from the reviewed v2 capability map; it never allocates IDs or mutates plan state.
- `scripts/analyze_lifecycle_cues.py` — emits review hints for active plans whose prose contains terminal-language cues; hints require human adjudication.
- `scripts/apply_gpt_lifecycle_review.py` — records dated GPT classifications as advisory lifecycle evidence; it cannot approve dispositions or allocate IDs.
- `scripts/reconcile_lifecycle_reviews.py` — compares independent reviewer CSVs and reports coverage/conflicts without silently selecting a winner.

To reproduce the current advisory review state after regenerating the frozen
inventory, seed the draft manifests, apply the dated GPT review with the
source-backed manual resolutions, then apply the independent Luna review:

```powershell
$env:PYTHONPATH = 'tools'
python tools/lab/planning-capability-rebaseline-v3/scripts/seed_review_manifests.py `
  --work artifacts/lab/planning-capability-rebaseline-v3 `
  --v2-map artifacts/lab/planning-capability-rebaseline-v3/V2_ACTIVE_PLAN_MOVE_MAP.csv
python tools/lab/planning-capability-rebaseline-v3/scripts/apply_gpt_lifecycle_review.py `
  --work artifacts/lab/planning-capability-rebaseline-v3 `
  --evidence tools/lab/planning-capability-rebaseline-v3/review-evidence/GPT_LIFECYCLE_REVIEW_2026-09-09.csv `
  --resolution tools/lab/planning-capability-rebaseline-v3/review-evidence/MANUAL_LIFECYCLE_RESOLUTIONS_2026-09-09.csv
python tools/lab/planning-capability-rebaseline-v3/scripts/apply_gpt_lifecycle_review.py `
  --work artifacts/lab/planning-capability-rebaseline-v3 `
  --evidence tools/lab/planning-capability-rebaseline-v3/review-evidence/LUNA_LIFECYCLE_REVIEW_2026-09-09.csv `
  --reviewer luna-high-agent
```
- `scripts/validate_manifests.py` — fail-closed graph/source/reference validator.
- `scripts/render_operations.py` — emits deterministic JSONL operations for an agent to execute through `ag-planning`, repository editing, and `ag-ledger`.

Generated working files belong under:

`artifacts/lab/planning-capability-rebaseline-v3/`

They are review working data, not plan/evidence authority.

## Start

From repository root:

```bash
python tools/lab/planning-capability-rebaseline-v3/scripts/generate_inventory.py \
  --pack tools/lab/planning-capability-rebaseline-v3 \
  --output artifacts/lab/planning-capability-rebaseline-v3
```

If the pack is used outside the repo, point `--pack` at this directory.

The generator uses `git ls-tree` and `git show` against the locked commit. It does not checkout, reset, rebase, stash, edit plans, or touch GPU state.

Then review and fill the generated manifests. Do **not** allocate/create successor IDs until namespaces, dispositions and lineage are complete enough to pass:

```bash
python scripts/validate_manifests.py --pack . --work <generated-dir> --phase review
```

After successor IDs/specs and reference decisions are complete:

```bash
python scripts/validate_manifests.py --pack . --work <generated-dir> --phase preapply
python scripts/render_operations.py --pack . --work <generated-dir> --out <generated-dir>/PLAN_OPS.jsonl
```

`PLAN_OPS.jsonl` is an execution plan, not a mutation script. Lifecycle mutations must be performed with BigCherry's `ag-planning` tools; ledger events with `ag-ledger`.

## Required disposition for every source item

Exactly one:

- `successor` — 1:1 continuation into the new planning epoch;
- `split` — one predecessor maps to two or more independently completable successors;
- `merge` — multiple predecessors map to one successor;
- `retire-completed` — genuinely finished; no artificial successor;
- `retire-deprecated` — no longer valid/desirable;
- `retain-history` — already-terminal historical item; no active migration;

Do not use `preserve-id` for continuing active work in this rebaseline. The purpose is to freeze the old planning epoch and create capability-native successor identities.

## Non-negotiable provenance rule

Never move an old review, evidence record, or ledger event to a successor to make the new plan look historically complete. Leave historical provenance attached to the predecessor. The successor may cite it as inherited evidence/constraint and must add new evidence under its new ID.

## Merge gate

The migration branch/PR may contain multiple commits, but no intermediate structural state is merged to `main`. Merge only after `preapply` validation, execution, post-apply validation, repository reference checks, and ledger synchronization all pass.
