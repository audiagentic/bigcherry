# Kernel group qualification and profile selection (RCS26–28)

See also [GPU qualification code-level implementation guide](gpu-qualification-code-guide.md), especially section 7 for the native harness power-intent gate and section 9 for owner admission/recovery prerequisites. PWR13 owns independent workload expansion; RCS26–28 retain native selection and integration.

Extends [kernel composition](radiance-kernel-composition.md), [bump integration](radiance-bump-integration.md) and [AI power screen](../reference/tooling/AI_POWER_SCREEN.md). RCS19–25 delivered the foundations; do not reopen their completed records. RCS08/RCS10, KAB01, FKE01, MEN bump work and PWR07/08/10/12/PWR02 remain their existing owners. This design introduces no alternate build graph, engine runtime authority, profile format, launch manifest, cache or evidence database.

## 1. Inventory and ownership

The three grouping axes are deliberately different:

| Axis | Authority | Meaning |
| --- | --- | --- |
| Build family | libr3/r3_groups.cmake; generated r3_groups.tsv | One translation-unit compilation owner in the actual configured graph |
| Qualification family | tools/lab/radiance/unit_families.toml | A narrow set of units and support units with oracle, shape and replay coverage |
| Runtime row | upstream registry and overlay origins.toml; component.toml | Provider, architecture, layout and callback contract, not a build group |

The eight current build families are examples, not a second Python enum. Consume the existing CMake output, validate its header and uniqueness, and compare with the full composed source's unit list/exclusions using the existing coverage gate. The current TSV only contains units selected in that build; R3_ONLY and excluded overlay units must not be interpreted as full coverage. An overlay is mapped by origins.toml upstream unit, base/overlay rows and supported targets, with component.toml the build authority. The r4d_gemm_iqr_p* wrappers share the owner of the primary source; resolve this from actual include edges. Shared/extracted headers are dependencies, not additional independent row owners. A gfx12 native or upstream build need not copy gfx11's excluded unit set. Use canonical classification on the composed unit inventory when a target lacks a TSV, not a duplicated regex table. Validate any new unit defaulting to core during pin triage; do not guess its oracle.

Patch ids and units come from the registry, rewrites from actual component ownership, and allowed targets from build metadata (where a patch has no architecture metadata, distinguish known shared from unknown rather than inferring from its name). Naming-rules.toml remains the naming policy; use literal suffix expansion against validated owner identity and explicit target/family alias facts. A suffix permitted for a shared implementation is not evidence it builds everywhere. Unknown placeholders/layers/ambiguous ownership must fail. Lexical parsing is code: mask comments/literals including raw strings, account for preprocessing variants, namespace/linkage vs class/function scope, macro scope independent of braces, HIP attributes and ordinary helpers. Use composed context for fragment edits; comparing isolated snippets or suppressing a declaration merely because its name occurs in an anchor is not sound. Audit exceptions with source/owner/why, keeping historical decisions. Syntax host/device and duplicate-definition gates remain separate; naming is not a C++ compiler.

## 2. Resolution and selection

Inputs: exact candidate and matched baseline, architecture/provider, composed source, actual build inventory, selected build group(s) or affected units (from compose_check.affected_units / pin-triage), and requested qualification intent. Changed headers expand through actual compiler dependencies; unknown dependency scope widens to the full affected build or blocks. Never narrow based on a filename guess. Intersect selected units with each configured qualification family's primary units; expand to that whole narrow family and its declared support units. A support unit is a build dependency, not evidence its own rows were qualified. Resolve owned units to overlay cases; never run the excluded base as a substitute for the overlay. Keep uncovered units/rows visible and nonpass. Do not silently skip unknown group/architecture, empty selection or incomplete inventory. The initial offline planner may consume the existing TSV and report only coverage within it; until full inventory/overlay resolution exists, it must refuse to claim complete coverage.

The IQR example: gemm_quant is a broad compile group; iqr is a narrower test family, with four primary upstream units, int8 reference support units, a tiled selftest case, extra ragged/tail shapes and replay cases. bc-iqr owns actual IQR embed, vocabulary and GEMM rows through origins.toml; those are distinct overlay cases in unit_families.toml. They are not equivalent to every gemm_quant unit or every IQR encoding. The group name selects work, not an automatic pass. A new family should require data and an existing oracle implementation, not Python architecture branches. New arithmetic semantics need real oracle code, not a configuration string pretending to provide one.

## 3. Run lanes and location

Use existing tools in their current owners. All generated files and logs stay in the configured work root, under the current unit harness/plan evidence paths; compact reviewed proof belongs in docs/evidence. No local host paths, card addresses or credentials in tracked config.

| Intent | Existing executor | Required evidence |
| --- | --- | --- |
| Static/PR | ownership lint, compose gates, CTest group test | Valid names, ownership, anchors, complete scope, host/device syntax |
| Compile | existing CMake/Ninja r3_group targets or canonical full target | Actual objects, matched flags/target; compile-only not runtime-qualified |
| Discrete | unit_harness.py build / measure / run; overlay-build / overlay-measure | Resource records, selected row/layout, positive output hashes, independent selftest, cold and warm replay where required |
| Bump | pin-triage + existing composition/bump workflow | Dependency closure, exact source identity, relevant discrete gates; existing integrated build/Gate E |
| Power role | gpu-ai-screen plus existing native kernel/transfer and production role tests | Physical device, effective settings, stock/candidate/stock, hot/resident/burst coverage, energy, quality, driver health and verified restore |
| Release | existing full engine/plugin lane and release workflow | Full link, component tests, installed Gate E and affected serving/parity |

One build at a time per Ninja tree, heavy compile nice -n 19, not during another actor's measurement. GPU runs only in bounded queued leases. Separate compile from GPU measurements; select one physical BDF and verify architecture and actual provider. Keep full release R3_ONLY empty. No new engine startup test or runtime dependencies.

The first implementation is an offline selection and evidence-gating slice in the existing unit harness, not a second CLI or production scheduler. Subsequent integration uses existing platform campaign and pin tools. In a production build group the current command remains cmake --build BUILD_DIR --target r3_group_GROUP -j 1. For an exact family run use unit_harness.py run --family FAMILY --arch ARCH --release RELEASE; for an owned overlay use overlay-build and overlay-measure under the existing lease. Require explicit baseline when generating qualification plans. Do not reuse the harness's implicit newest release as a qualification identity.

## 4. Evidence and profiles

Separate a workload qualification profile (which required tests to run) from a power profile (authored hardware intent) and tuning calibration (measured execution policy). Workload profile is resolved from existing unit_families specs plus required gates for the intent. Shape selection comes from actual row contracts/workload traces and boundary cases: small decode, prefill tile edges, ragged dimensions, supported forms, layout refusal, fused/unfused where applicable, chaining, transfers/collectives when used. Do not invent coverage from a model name or resource equality. Compiler resource records alone cannot establish byte identity or timing; unchanged counts do not imply unchanged code. Match baseline architecture/provider/flags and logical operands; private packed bytes cannot be shared across incompatible providers.

Use existing platform run-record v1, calibration records where applicable, and current unit-harness output, not a new persistent outcome format. Record candidate and baseline source/pin/build/flags/toolchain, target, physical device and actual provider/layout/row, workload/shape/seed and oracle, profile controls/readbacks and thermal envelope when relevant, completion and artifact references. Reuse prior evidence only for exact relevant identity, gate coverage and compatible conditions. Invalidate affected coverage on source/dependency, target/flags/toolchain, layout/registry, workload/oracle or power envelope change. Unknown is not reusable.

Outcome policy (not a hardware setting):

| Outcome | Meaning / next action |
| --- | --- |
| diagnostic | Static/compile/resource-only; no profile qualification |
| inconclusive | Missing/unsupported gate, no or incomplete hash coverage, wrong identity, skipped/unconverged replay, thermal target not reached or interrupted; keep old profile and collect missing evidence |
| rejected | Numerical mismatch, required row refused/missing, confirmed regression, driver/restore failure; retain evidence and use existing recovery/owner workflow |
| eligible | All required gates and identities verified, no regressions; submit evidence to existing owner for review/promotion, never auto-activate |

Both reference and candidate must have positive matching coverage and pass their independent oracle; reference failure invalidates comparison. Required target row cannot be silently supplied by a base fallback. A timing window that does not converge is inconclusive even when the ratio looks favorable. Unchanged resources are not a reason to skip a required replay. Keep resource changes visible and require reason/review rather than universal zero-change threshold. Replay uses existing bounded ABBA methodology, no new handpicked cutoffs; actual thresholds from existing policy and measured uncertainty. A partial test can recommend more work, not declare the profile safe.

Power controls remain in engines/radiance/power-profiles.yaml with the existing platform reader, native owner and transaction/restore authority. PWR12 is independent screening, not proof of production kernels; use role-appropriate native rows and existing production quality/energy gates (PWR10/PWR02). For a candidate, use actual matched physical device, verified stock, existing transaction, named lease and restore before release. No GPU writes on an offline plan or automatic promotion. A clean arithmetic run is not evidence of all clocks, temperatures, cards or roles; keep scope exact. RCS10 remains tuning-default owner; do not set runtime flags from a test outcome.

## 5. Delivery and acceptance

RCS26: one family/ownership/naming resolution from actual build metadata; lexical and data-policy fixtures; audit exception removals, unknown scope and shared-header expansion. RCS27: existing harness plan selection, coverage gaps and fail-closed outcome gates; add missing family oracles and overlay mapping, then real matched builds/tests. RCS28: consume these outputs in existing bump/campaign/profile workflow and prove evidence reuse and refusal; physical profile testing stays with current authorized owners. Merge planning/design before implementing. Commit implementation and its ledger record together. Do not revise active related plans without coordinating; retain completed plans as evidence.

Acceptance: adding a synthetic group/target/qualification family through existing metadata does not require another classification table; exact source/build coverage, dependency and overlay gaps visible; fixtures include malformed/partial inventory, invalid naming rules, missing rows, wrong layout/provider, empty hashes, bad reference, skipped/unconverged replay and stale evidence; real selected family build, matched discrete GPU qualification and existing release gates separately reported. No new persistent authority or auto profile application. Unimplemented coverage stays pending rather than being described as qualified.
