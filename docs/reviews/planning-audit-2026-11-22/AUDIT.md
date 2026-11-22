# BigCherry plan reconciliation audit — 2026-11-22

**Base:** `origin/main` at `7e4e7eae816889849362e6369335786882490beb`. **Branch:** `agent/planning-holistic-20261122`. The inventory and BCOP dispositions are **pre-registration review snapshots**; nine Radiance/LTX plan items were subsequently created through the canonical Planning MCP.

## Scope and previous review baseline

This audit starts with the 2026-11-20 stale-plan cleanup request, the subsequent measurement-owner and Strata/Radiance consolidation commits (#1243–#1248), and the 2026-11-22 instruction to keep generic Radiance migration (RUM) separate from LTX 2.5 TP3 and llama.cpp. Current code and plan state, rather than the conclusions of historical triage notes, determine decisions. In particular, MEN01 and many other items were previously marked superseded and later reopened on 2026-11-04; neither the earlier closure nor the reversal is proof of current delivery.

The completed-area history is indexed in `COMPLETED_INDEX.csv` to detect current/history collisions without rewriting old evidence. The source inventory is `ITEM_INVENTORY.csv`; `PROPOSED_DISPOSITIONS.csv` contains a proposed action for **every** direct item document, with original path, priority, state, recent Git touch, ownership candidate, target plan where identifiable, and reasoning. Both files are review aids, **not** planning-service writes. `BROKEN_RELATIVE_LINKS.csv` contains a direct-item Markdown-link check.

### Counts

| Measure | Count |
| --- | ---: |
| All active Markdown files, including readmes/reviews/supporting documents | 536 |
| Top-level plan-like Markdown documents | 497 |
| Registered canonical item names and populated state (pending or in_progress) | 495 |
| Completed-area item records (separate historical index) | 910 |
| Completed / superseded / deprecated historical records | 492 / 353 / 65 |
| Active/completed duplicate IDs | **1 (RHA07)** |
| Pending | 463 |
| In progress | 32 |
| Supporting technical notes incorrectly sitting alongside active items | 2 |
| Items created before 2026-11-01 (age signal only) | 375 |
| Direct items without a Git documentation touch since 2026-11-01 | 285 |
| Direct items missing a populated canonical acceptance section | 171 |

**Caveats:** The owner column is a group/title heuristic, not proof of runtime/source ownership. Age and document touch are **not** evidence of nonimplementation. The missing-acceptance count includes BCOP decision notes and two noncanonical supporting documents; it is not equivalent to 171 incomplete engineering implementations. No GPU measurements or broad builds were performed. Nine new pending items were created through Planning MCP; none of the pre-existing items was transitioned.

**Post-registration state (2026-11-22):** after merging updated `origin/main` and registering LTX01–LTX08 and RUM13 through Planning MCP, there are **506** direct active item documents (**nine new pending**). The 497-item CSV remains the preserved pre-registration comparison set. No pre-existing item was closed or modified through this registration slice.

## Ownership decisions

| Owner | Owns | Must not own |
| --- | --- | --- |
| **llama.cpp** | `engines/llamacpp/` patches, GGML/HIP/Vulkan backend kernels, Flash-Next/MoE/GDN experiments and llama.cpp run-record producer adapters; `LCM01`/`LCM02` | Radiance extension ABIs, native `bc-serve` session owners, shared ledger/release/campaign framework |
| **Radiance** | `engines/radiance/` patches, serving and runtime, native TP/collectives, RUM non-LTX migration, model-specific LTX 2.5 TP3, heat/placement, Radiance release asset content and native canaries | GGML patches, llama.cpp producer work, duplicate platform campaign, record, store or release service |
| **Platform** | Core records/facts/feature schema, BuildPlan and campaign orchestration, release metadata/ledger/event integration, repository worktrees, CI, power receipts and engine-neutral qualification/reporting contracts | Engine-specific policy/implementation, a second source of engine runtime configuration, bespoke model-specific launchers |

Engine-neutral infrastructure belongs to platform, with **engine-specific adapters/tests** kept at engine ownership. One owner per deliverable; cross-engine integration is an interface acceptance check, never a second implementation ticket. Within Radiance, `RUM` owns **non-LTX** generic migration and reuse, while a distinct LTX plan group should own LTX graph, TP3 and model-specific numerical/hardware acceptance; LTX may depend on generic AQL tests without copying them.

## Priority dispositions

### 0. Resolve a verified duplicate current/completed item: RHA07

`RHA07` appears in **both** `docs/planning/active/run-hip-autotune/RHA07.md` (`pending`) and `docs/planning/completed/run-hip-autotune/RHA07.md` (`completed`). The latter records 118 passing focused tests and a 78-pass regression suite; the actual `run_advisories.py` has all four evaluators, and `benchmark.py`, `cli/build.py`, `tuning/workflow.py` and `tuning/tune_promotion.py` call them. Retire the duplicate active item via Planning MCP after checking linked RV159/review state; preserve the completed record and its evidence. This is a concrete bug, not a speculative age-based closure.

### 1. Collapse secondary BCOP decision tickets after preserving their unique gates

`docs/planning/active/patching-bc-optimizations` has **137 pending** decisions, many titled `QFPxx: …`, `PRBExx: …`, `METxx: …`, `RNXxx: …`, `PGCxx: …`. They currently double the apparent execution backlog. An exact title-ID-to-filename comparison found **98** with an active primary plan, **19** with a completed primary, and **20** without a resolvable primary in the title. See the complete mapping in `PROPOSED_DISPOSITIONS.csv`.

- For the 98: review the BCOP ticket for a *genuinely unique* condition, copy its validation/acceptance/evidence to the owning active item through Planning MCP, then mark BCOP superseded and point to the surviving ID. Do not spawn a new implementation path.
- For the 19: check whether the completed primary acceptance covers the stated condition; if it does, retire the BCOP decision with that evidence. Otherwise open one narrowly bounded **successor under the primary owner**, not an independent BCOP runtime implementation.
- For the 20: explicitly select the real successor first. Examples `BCOP28` (self-titled superseded), `BCOP37` (Strata comparison), `BCOP33` (telemetry) are **not** automatically terminal merely because no primary ID appears in their title.

The BCOP count is title-based evidence of likely double ownership, **not** 117 approved closures. Existing engineering owners such as `QFP`, `PRBE`, `MET`, `PNRO`, `PGC`, `RNX` and `TRVP` should retain distinct hardware, correctness or runtime gates.

### 2. Radiance: preserve active ABI work; partition LTX; remove stale 1.17 qualification

- **Retain `RUM06`** as the active generic Radiance 2.2 remote-MoE generation/lifetime change (`patching-radiance-upstream/RUM06.md`, PR [#1272](https://github.com/audiagentic/bigcherry/pull/1272)). It explicitly reuses the existing admission ledger, rank coordinator and TP rendezvous; no new scheduler or parallel model registry.
- `RUM01–05` and `RUM07–12` are in `docs/planning/completed/patching-radiance-upstream/`, including recent #1291 (RUM08) and #1292 (RUM12). Do not recreate or re-open finished migration gates to carry unrelated LTX work.
- **Registered LTX01–LTX08** under `build-radiance-ltx25-tp3` using the existing `docs/evidence/ltx25-plan-registration/` proposals and Planning MCP. **Registered RUM13** under `patching-radiance-upstream` for the single model-independent native AQL lab (PR #1294). Generic production lifetime still belongs only to RUM06 (#1272); LTX02 owns the model-backed execution, not an alternate collective. PR #1262 landed on main in the intervening sync. No GPU run or new production qualification was done by this planning change.
- **Revalidate `PRQ01–05`** against current source and installed 2.2/Radiance release line before closing or rewriting: `PRQ01.md` still specifies a 26-patch v1.17 source-reconciliation gate and describes a prior false pin mismatch, whereas current `docs/reference/build/PIN_BUMP.md` describes Radiance 2.2 as 2020. `PRQ03.md` already records rad0005 as archived after upstream 2.2 FFmpeg 6 support. Rehome remaining *patch qualification* under one current-pin owner, with links to RUM02/07/08 evidence, not a 1.17 parallel system.
- **Review `RHP` and `VTP` roadmaps** in situ. `patching-radiance-heat-placement/README.md` refers to RHP02/17/18/19 as future/active work even though their documents now live in `completed/`. `run-dense-27b-virtual-tp/README.md` links `VTP01–14` under its own active directory although many are completed. Distinguish a completed component from genuinely outstanding end-to-end acceptance, fix navigational links, and preserve the actual live placement and dense TP gates.
- **Re-scope `RSC02/03/04/11/12`, `RSP01`, `RCS*`, `MEN*`** by concrete shipped 2.2 native configuration/runtime boundaries; one native serve owner and one release manifest, not old Python launcher follow-up paths. RSP01 owns release packaging policy; RSC owns the native engine start/session behavior; platform owns the generic release machinery. `MEN24` build/bump optimisations must be checked against `BU01/02`, `BPB01` and the actual pin-bump workflow before preserving duplicate tasks.
- `RAD50_SLICE_OPTIMISATION.md` and `RR10-AR.md` are detailed notes, not lifecycle item records (neither has canonical plan front matter). Rehome or index them as evidence without rewriting their measurements.

### 3. llama.cpp: preserve engine-specific chains, not platform/Radiance counterparts

- **Retain `LCM01` and `LCM02`**. The descriptions explicitly carry the llama.cpp-only portions excluded from completed Radiance `MRC02/MRC03/MRC06`. `LCM01` must eliminate its run_bench monkeypatch/redirect shim or retire that producer, never write a second result store. `LCM02` must use the platform schema rather than copy platform feature/receipt services.
- Preserve `MET02–12` and the appropriate `QFP`, `FMTP`, `PNRO`, `PRBE`, `PGC` hardware quality/optimisation gates where acceptance still differs; fold BCOP restatements into these plans.
- Put GGML/Vulkan shader/provider implementation in llama.cpp, while generic `BuildPlan` or `ArtifactStore` support remains platform. Do not create another engine-agnostic kernel-selection or graph-recipe registry under the llama owner.
- Treat Strata-related `BCOP37` and older comparison proposals as a **bounded evidence decision** under an existing llama.cpp benchmarking owner, not a second production engine adoption plan without an independently validated need.

### 4. Platform: retain only one shared framework per concern

- **`BRL01` is not safely completable on age.** `docs/releases/events/` has 331 tracked event files on this base, yet `docs/releases/README.md` still says `CURRENT_RELEASE_LEDGER.ndjson` is the tracked change ledger. Recheck release-notes loader, CI and stale-branch conversion gates; update BRL01 to the real residual, not another ledger implementation.
- Reconcile `BRBC*`, `PRBC*`, `RRBC*`, `TRBC*` against the existing BuildPlan/ArtifactStore/campaign pipeline. Keep the distinct build/run/tune contracts where they have independent observable gates; retire speculative duplicate orchestration.
- Preserve `BHK01`/GFXOFF host-kernel ownership and `PWR*` power/receipt experiments at platform/hardware level; move native `bc-serve` profile integration acceptance into the **one** Radiance serving owner rather than two power controllers.
- `WSL*`, `BRC*`, `RCD*`, `PA49/50`, `PVPS*`, `PEC02` require ownership review against current `audiagentic worktree`, event store, CI and pin-bump capabilities. Remove shims only after caller migration; preserve evidence and references.

## Quality issues and checks

- `RSC11.md` links to missing `../../evidence/2026-11-13-unpublished-branch-reconciliation/README.md` (the direct-item link check found one broken relative Markdown link). Verify whether its evidence exists under another retained path before changing the link.
- `RHP` and `VTP` group READMEs contain additional stale references to completed items; the direct-item link checker intentionally does not count supporting README files.
- 171 direct files lack a populated canonical `## Acceptance Criteria` section (mostly BCOP decision notes). Normalize **only surviving implementable plans**: real scope, affected files, proof gates, explicit stop conditions, linked evidence, work not duplicated.
- Evidence must distinguish source/patch applicability, build, runtime/ABI, numerical quality, hardware admission and throughput. Closing based on a documentation move or plan age is invalid.
- No plan should be rewritten solely to move it into a broad domain name. Prefer a single owner and cross-links; never rewrite historic completed evidence to manufacture present acceptance.

## Publication and lifecycle gate

The repository requires **ag-planning Planning MCP** for all plan/review mutations. AUDiaGentic PR #491 adds an external, project-scoped stdio accessor that invokes the **same** Planning and Ledger MCP servers, not hand-written file updates. Using that canonical boundary, this branch registered nine new `pending` item records (LTX01–08 and RUM13). The baseline inventory and proposed BCOP dispositions remain review-only; **no old item was closed, superseded or deleted**.

This review branch remains a **draft PR, not merge-ready broad cleanup**. The LTX/RUM registration slice is done; the remaining BCOP, RHA07, PRQ, RHP and platform reviews still require explicit Planning MCP transitions and verified acceptance.  No claims of 117 closures, 2.2 quality acceptance or LTX TP3 performance are made here.
