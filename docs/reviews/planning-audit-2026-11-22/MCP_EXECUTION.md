# Planning MCP execution handoff — BigCherry 2026-11-22

This handoff is **not** an instruction to patch `docs/planning/` files directly. Use `ag-planning` MCP with the explicit isolated audit task worktree and current tool schemas. AUDiaGentic PR #491 supplies a project-scoped external MCP accessor; it is not BigCherry source code. Do not replace an unavailable planning service with shell, GitHub contents API, generic file writes or YAML edits.

## Ordered change batches

### A. Re-read current main, inspect existing groups and preserve protected work

1. Re-fetch and merge (never reset/rebase/force) the latest origin/main into this task worktree if advanced, checking clean status before action.
2. Planning MCP: list all groups/items and linked reviews, compare against `ITEM_INVENTORY.csv`. Confirm if another session has already moved an item or created an owner since 7e4e7eae8. Do not modify stale inventory assumptions directly.
3. Protect all 32 in-progress items, with special attention to RUM06 (#1272), RHP04/11/16/20, RSC02/03/04/06/11/12, XDP01–09, PWR06–09, RSP01 and VTP42. Active state does not by itself prove progress or acceptance; explicitly validate any proposed closure.
4. Keep completed history immutable. Never delete plan/evidence/ledger files as the cleanup mechanism.

### B. Make ownership unique before retirement

0. Resolve the one verified active/completed collision: RHA07 exists as pending in active and completed in history. Compare the completed test record, actual four evaluator/caller implementations and RV159; retire the stale active RHA07 via Planning MCP while keeping the completed document. Confirm there are zero ID collisions afterward. The complete historical inventory is in `COMPLETED_INDEX.csv`.

1. Adopt **one** owner per subject: llama.cpp, Radiance or platform. When shared interfaces are involved, platform maintains the contract, the engine owns only its adapter and acceptance tests.
2. For BCOP target-active cases (98 proposed), use Planning MCP to create a review of the BCOP/primary overlap; compare unique acceptance/quality gates, fold only unique scope into its existing primary item; close review with rationale; then supersede the secondary BCOP item.
3. For BCOP target-completed cases (19 proposed), compare against immutable completed acceptance and evidence. Supersede BCOP if already covered; otherwise create one minimal successor in the real owner group and explicitly link the old completed gate. Never reopen an old completed record merely to reuse an identifier.
4. For the 20 BCOP titles with no resolved primary, choose a real owner before a state change; avoid creating a generic BCOP backlog owner.
5. Confirm each transition by re-reading the Planning MCP state, generated files and links. Do not count only the submitted tool call as a completed transition.

### C. Separate model-specific LTX from generic Radiance migration (registered 2026-11-22)

1. Read current PR #1262 and #1294 diffs and branch ancestry; #1294 currently targets the LTX branch and is not an independent main-target PR.
2. **Done through Planning MCP:** registered `LTX01–LTX08` in the existing evidence-proposed `build-radiance-ltx25-tp3` group. They map all LT3-01–15 requirements without creating fifteen extra plan items. Source and proof boundaries remain in `docs/evidence/ltx25-plan-registration/`. The production TP3 graph is **not** thereby accepted.
3. **Done through Planning MCP:** registered `RUM13` for model-independent AQL harness consolidation and exact mixed-card regression. **RUM06** remains the generic production engine-generation and collective lifetime owner. LTX02 consumes the RUM13 proof without adding a second collective or scheduler.
4. Keep PR #1272 and LTX PRs independently mergeable. Do not claim GPU/ABI acceptance from source or host-only evidence.

### D. Reconcile stale engine and platform programs

1. Review PRQ01–05 against the 2.2 current source, `docs/reference/build/PIN_BUMP.md`, completed RUM evidence and live patches. Create findings reviews before plan edits. Close verified historical source-apply tasks; keep **one** current patch-admission/qualification owner for remaining patches, with concrete file and test gates.
2. Review RSP01 release-asset/installer policy, RSC native runtime scope, MEN24/BU01–02/BPB01 build-bump work and BRL01 ledger work. Split by product ownership, not duplicate implementation paths. Pin current release naming (2020.x) and remove obsolete 1.17-only requirements from **active** acceptance, preserving historical notes.
3. Update RHP and VTP roadmaps and links to completed item locations, retaining genuinely open behavior in the right active owner; resolve RHP02 completion-vs-remaining-notes ambiguity explicitly rather than silently reopening it.
4. Review old `run-multi-engine` MEN entries individually: relocate engine-specific residuals to an existing Radiance owner, retain the shared cross-engine comparison contract only on platform. Earlier 2026-11-04 reopens must not be reversed without current proof.
5. Keep llama.cpp LCM01 and LCM02 and existing MET/QFP/FMTP/PNRO/PRBE/PGC implementation gates with their primary IDs. Keep BuildPlan/campaign/release/run-record core on platform. Mark dormant conditional work as pending/deferred with **explicit trigger**, not completed on age.
6. Relocate nonplan technical notes `RR10-AR.md` and `RAD50_SLICE_OPTIMISATION.md` to their evidence/design owner with reference preservation if warranted; these two are not lifecycle items.

### E. Evidence, validation, review and merge

1. For each survivor that lacks executable acceptance, supply files/functions, minimal fixtures, offline tests, leased hardware tests where needed, expected evidence location, and one unambiguous stop/rollback condition. Avoid open-ended “investigate/optimize” tasks without a measurable decision.
2. After Planning MCP mutations, regenerate inventory and dispositions against the **new** tree, revise this audit with actual transitions instead of proposals, and compare counts by group/owner. Detect duplicate active ID, broken local markdown path, invalid state/plan group, missing acceptance and unclosed reviews.
3. Run repository-specific quick documentation/plan checks, `git diff --check`, and scoped tests if changing more than documents; do **not** acquire GPUs solely for a planning restructure.
4. Ledger MCP: record a release change event if the planning edits are classified substantive by the installed workflow. Commit its generated event alongside the plan files. Do not create a hand-authored ledger event as a substitute.
5. Commit plan outputs in bounded batches (BCOP ownership, Radiance/LTX separation, engine/platform adjustments, documentation validation), no attribution lines or host names embedded in tracked content. Push the branch normally and open/update a **draft PR** for audit, then mark ready only when plan edits, review closures and checks are verified. Merge through branch protection; confirm merged SHA and clean only this worktree afterwards.

## Review gate

The final PR must show: actual before/after active-item count and per-owner counts; every moved/superseded ID and surviving owner; evidence for every completed item; no source/plan code duplication; preserved lifecycle and reviews; unchanged runtime code unless independently justified; absence of broken plan links; checked generated plan/ledger files; and no unverified performance claim.

The audit CSV is a **triage queue**, not a script to automatically retire items. Its “target” is extracted from a title, and its “owner_candidate” is a heuristic; both must be confirmed by the Planning MCP review.
