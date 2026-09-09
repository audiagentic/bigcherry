# Semantic review protocol

## Review unit

Review every lifecycle plan item found at the frozen source commit. Do not limit semantic review to a small hand-picked queue: the rebaseline changes the ownership model, so every item needs an explicit disposition even if the result is `retain-history`.

For each source item, answer in order:

1. Is there still unfinished engineering work?
2. What is the independently completable acceptance boundary?
3. Which capability owns that boundary: Build, Run, Patching, or Tuning?
4. Does the item contain more than one independent acceptance boundary?
5. Is it duplicate/overlapping with another surviving item?
6. Which historical evidence/decisions remain valid, and where were they originally produced?
7. Which active dependencies must point to new successor work?
8. Which references are historical facts and must remain on the predecessor?

## Ownership tests

### Build

Own here when completion is primarily a statement about constructing a reproducible binary/artifact:

- llama.cpp/vendor/source revision;
- source checkout/materialization;
- compiler/toolchain/dependencies;
- build recipe/lane/CMake composition;
- applying a selected patch set while constructing an artifact;
- artifact hash/manifest/provenance/verification.

Build applies patches but does not own patch semantics, validation policy, promotion, or tuning winners.

### Run

Own here when completion is primarily a statement about executing a workload reproducibly and producing trustworthy observations:

- registered model resolution;
- physical GPU/device/topology selection;
- runtime profile resolution;
- whole-matrix preflight;
- execution scheduling/orchestration;
- warmup and measurement;
- paired A/B execution and work-equivalence;
- correctness execution;
- runtime diagnostics/profiling/log capture;
- evidence/report generation.

Run executes experiments but does not decide patch semantics or tuning-candidate meaning.

### Patching

Own here when completion is primarily a statement about a source modification and its lifecycle:

- patch identity/content/rationale;
- prerequisites/applicability;
- contracts and validation semantics;
- patch fixtures/tests;
- patch evidence requirements;
- dependencies/conflicts;
- promotion/demotion/status.

If a patch plan currently contains bespoke build/run machinery, keep only the patch requirement and make the shared Build/Run capability a dependency.

### Tuning

Own here when completion starts at or beyond BC-native selection and is about selecting better runtime implementations:

- dispatch recording/signatures;
- candidate catalog/search/ranking;
- tuning profiles/cache;
- replay and replay qualification;
- execution attribution;
- winner provenance/counterfactuals;
- prebinding and tuning overhead;
- composition of tuning winners.

A patch that merely enables/observes tuning remains Patching if the acceptance boundary is patch lifecycle; actual candidate selection/promotion behavior is Tuning.

## Split rule

Split only when scopes can be completed and accepted independently. A dependency alone is not a reason to split.

Example:

- "add declarative GPU/model matrix execution" -> Run successor.
- "define patch promotion rule using evidence produced by that matrix" -> Patching successor.

These may depend on each other but have separate acceptance boundaries.

## Merge rule

Merge when multiple predecessor items are now different fragments of one acceptance boundary. Do not merge merely because they share a capability or code directory.

Every merged predecessor remains individually discoverable and points to the shared successor.

## Successor rule

A continuing item gets a new ID in this migration even if its engineering question is mostly unchanged. The semantic reason is the new planning epoch and capability-native ownership, not the filesystem move by itself.

A successor spec must contain only still-valid future scope. Do not copy stale chronology, obsolete paths, completed steps, or historical conclusions into active scope.

## Lineage rule

Lineage is many-to-many and normalized in `LINEAGE.csv`:

- `supersedes` for 1:1 continuation;
- `split_from` when one predecessor produces multiple successors;
- `merged_from` when multiple predecessors produce one successor.

Every successor must have at least one predecessor. Every source item with `successor`, `split`, or `merge` disposition must have at least one lineage edge.

Visible plan text must also carry lineage:

New plan note:

`Supersedes: <predecessor IDs>`

Old plan note before terminal transition:

`Superseded by: <successor IDs>`

`Migration: capability-rebaseline-v3-2026-09`

Do not rely solely on prose: the CSV graph is the migration authority and validator input.

## Historical provenance rule

Keep on predecessor:

- reviews created against predecessor scope;
- evidence produced under predecessor identity/contracts;
- prior ledger events;
- historical decisions and completed implementation chronology.

Successor may cite those records under `Inherited evidence and constraints`. It does not claim them as newly generated successor evidence.

## Reference classification

Every old-ID/path occurrence requiring a decision is one of:

- `active_dependency` — forward execution/work prerequisite; normally rewrite to successor;
- `active_followup` — points to continuing future work; normally rewrite;
- `active_scope` — names the plan currently responsible for work; rewrite;
- `historical_evidence` — identifies where evidence was generated; preserve predecessor;
- `historical_review` — identifies a review/decision made against old scope; preserve predecessor;
- `historical_decision` — chronology/provenance; preserve predecessor;
- `ambiguous` — must be manually resolved before preapply.

Do not globally replace IDs or paths.

## Retirement rule

- `retire-completed`: implementation and validation are actually complete.
- `retire-deprecated`: scope is intentionally abandoned/invalid.
- `retain-history`: item is already terminal and remains historical.

Use BigCherry-supported lifecycle state operations. Never delete a predecessor simply to clean the tree.
