# Composable Radiance kernel builds

Status: design, 2026-11-25. Plan items: RCS19 to RCS22 (build-radiance-components), supplementing RCS15 to RCS18 (native RDNA3 provider migration). Nothing here is implemented yet except where marked.

## 1. Why

Our gfx1100 kernels are a generated tree: libr3 compiles upstream's gfx12 units for gfx11 through a compatibility header (`r3_compat.h`) and a set of native rewrites (`native/*.rw`). Managed patches edit the same upstream source text before libr3 rewrites it. Three costs followed from that:

1. **Conflicts between owners of the same text.** The gfx1201 fused-tile patch passed every gate (byte identical, no spill, all 24 single-band points faster, 0.48 to 0.86 of shipped time) and still could not build: it edited, or inserted, the same lines the merged gfx1100 rewrites anchor on, so the anchors vanished and two definitions of the same type appeared. Nothing in the patch tests or lint could see it; it was found by a full lane build.
2. **Gate loops measured in a full plugin build.** Every candidate is built as the whole plugin (about 7,800 kernel records) before it can be hashed, replayed and resource-compared, even when one unit changed.
3. **No clean unit of release, rollback or upstream contribution.** The wins of the last two weeks (head tile, accumulate-once, native fragments, fused tile, embedding butterflies, NT4 head prefetch, chained long tiles) are each a small change to one kernel family, but ship and revert only as part of the whole plugin.

Goal: make a kernel change a self-contained, mechanically checkable unit that composes with every other change, builds and gates on its own family, and, where it is architecture-neutral, can be offered upstream.

## 2. Current structure (facts)

- Upstream's own unit list defines the compile units; libr3 reads it from upstream's build file. Rows whose unit is left out are dropped at load.
- Application order today: pinned source, managed patches, libr3 configure (rewrites, compat layer), compile, link into one `libr3.so`; a separate native plugin for gfx1201 (`libr4d`), built from the patched source without rewrites.
- An earlier effort (unmerged branch) grouped every upstream unit into eight families: `core`, `gemm_quant`, `gemm_dense`, `attention`, `moe`, `speculation`, `multimodal`, `collectives`, with compile-only per-family targets and a coverage test that each unit has exactly one family. It shares objects with the full build and produces no loadable plugin.
- The loader expects one plugin per architecture; helper contracts (remote MoE, n-gram) have their own ownership.
- Wins that exist only because of the shim (accumulate-once native layouts, native fragments, native head sums) are gfx11-specific by nature. Wins that are arithmetic-neutral (fused single-band tile, wave-local embedding butterflies, head prefetch lifetime, weighted-head ownership across ranks) are not.

## 3. Principles

- One owner per line of source: every line is edited by at most one layer (a managed patch or a rewrite), and the overlap is checked by a machine, not discovered by a build.
- Additive before invasive: new kernel entry points and guarded dispatch beat edits to shared bodies. A shared-body edit is a template parameter whose default reproduces the old code.
- Names carry their owner: a symbol introduced by a patch or rewrite carries an architecture/patch suffix, so two owners can never define the same name.
- Derive, never hand-pick: tile selection comes from architecture traits, form fields and band counts, as in the guards already merged.
- Evidence travels with the unit: resource table, byte identity, cold and warm replay, in the patch's own folder.
- Do not build machinery for a tree that may be retired: layers 1 and 2 pay off with the generated tree or with a native provider; layer 3 waits for the RCS15 baseline.

## 4. Layer 1: composition seams (RCS19, build first)

A design review (2026-11-25) showed that a textual anchor check proves less than this section first claimed. The rewrite checker verifies match counts and the paired-run rule; it does not parse C++, preprocess architecture macros, compile host and device passes, or link. So composition is checked by four separate gates, each saying only what it proves.

**Gate 1, provenance and anchors (seconds, offline).** Apply the full ordered patch set to the pinned source (the bc patches included), then run the real rewrite check against that patched source. It proves: every required anchor resolves exactly as declared; an optional alternative group has exactly one match (the default for mutually exclusive variants; a deliberate multi-match needs explicit metadata); declared dependencies exist; edited regions do not overlap. It records the digest of the resulting source. It does not prove the result is valid C++.

**Gate 2, compile.** Compile the affected translation units for gfx1100 and gfx1201 (host and device passes, the feature macros each architecture sets, explicit template instantiations of the touched kernels). It catches a duplicate type, a repeated template default argument, a device-only definition clash, and an architecture guard that selects the wrong branch. Run per pull request on affected units, with the compiler cache; not part of the seconds-scale gate.

**Gate 3, link and export.** Link the affected family (or a contract fixture) and inspect duplicate definitions, unresolved symbols, exported names and unexpected external device symbols.

**Gate 4, qualification.** Resource table, byte identity, selftests and replay for the affected family and for the units that include a changed shared header.

**Source ownership (replaces the coarse unit rule).** Three separate records: *affected units* (what must rebuild and retest, expanded through the transitive include graph, so a changed shared header invalidates its dependents); *owned regions* (the lines a transformation edits or inserts, to detect overlap); *consumed anchors* (a rewrite that relies on a patch's text names that patch and the anchor). Two changes may touch the same unit when their regions do not overlap, and may collide through a shared header when their units differ; the records handle both.

**Naming.** A name that two architectures' implementations must both define takes an architecture suffix (for example a gfx12 and a gfx11 fused tile). Suffixes are a convention, not a uniqueness proof: uniqueness comes from the symbol check in gate 3. Upstream-required names and ABI names are never renamed, and neutral algorithms intended for upstream carry no local patch identifier.

**Freshness.** The libr3 configure step depends on the rewrite rules and rules file but not on the patched upstream sources it rewrites, so an incremental build directory can keep an old generated copy after the upstream text changes. Build directories are content-addressed by source slice in our flow, which limits the exposure, but the dependency should be real: make the generation depend on a digest of the patched inputs, with a test that changing an upstream header regenerates the output without deleting the build directory.

**Layer order.** pinned source, arch-neutral managed patches, libr3 rewrites (gfx11 specifics only), compile. A rewrite may rely on a patch only through a declared anchor.

First customer, resolved 2026-11-25 on a branch (not yet general tooling): the gfx1201 fused-tile package keeps its shared-header edits; the five libr3 header blocks that anchored on the same text became find-1-or-0 with a paired identity block; the patch's own tile carries gfx12-suffixed names and is compiled out of the gfx11 build. Lane builds of both architectures confirmed the gfx1100 records unchanged and 11 fused kernels added on gfx1201 with no spill. Gate 1 would have caught the vanished anchors; the duplicate definition needed gate 2.

Deliverables: gate 1 with fixtures (vanished anchor, doubled anchor, exactly-one-of-pair), gate 2 with failing fixtures (duplicate type, repeated template default, device-only definition, conflicting conditional compilation), the ownership records in patch metadata, the freshness dependency and its test, and the patch-author guidance. Acceptance: every managed patch set applies to the pinned source; every required anchor resolves under its declared variant; invalid declarations, duplicate definitions and symbol conflicts are rejected for both architectures before a lane build; no family is qualified until the affected and dependent kernels pass their correctness and resource gates.

## 5. Layer 2: arch-neutral formulation and the upstream path (RCS21)

**Traits seam.** Write new tile code once, against a small compile-time traits type: wave size, native WMMA operand layout accessors, accumulator conversion (identity on gfx12, the 11-to-12 conversion on gfx11 today, a no-op on a native gfx11 provider), LDS capacity and granularity, register-file budget and maximum waves. Each architecture supplies its traits. The libr3 rewrites then shrink to the traits header and the few genuinely gfx11-only optimisations.

**What is neutral and what is not.**

| Change | Neutral | Note |
|---|---|---|
| Fused single-band IQR prefill tile | yes | gfx1100 and gfx1201 versions exist and are gated; unify on traits |
| Wave-local butterflies in the rotated embedding gather | yes | five wave-local stages, two shared-memory stages |
| Head tile prefetch lifetime for the widest token tile | mostly | guard from the activation staging ring, spill-driven |
| Weighted vocabulary head across uneven ranks (rad0063, rad0064) | yes | feature, not a kernel |
| Accumulate-once native sums, native fragments, native head sums, chained long tiles | no | exist because of the gfx11 shim; disappear or change on a native provider |

**Upstream.** Neutral changes are prepared as upstream-shaped patches: one per unit, a clear description, the correctness test, the resource table and replay evidence. They are offered only with the owner's approval; until upstream accepts one, it stays a local managed patch. When a pin bump brings the change in, the local patch is archived unchanged (never composed) and the rewrite blocks that depended on it are re-derived. Our text is in our own words and carries no reference to material outside the repository.

The fused tile is architecture-neutral algorithm and dispatch policy with architecture-specific data motion. The traits layer owns fragment representation, accumulator layout and the WMMA boundary, resolved at compile time and inlined: no runtime trait object, no conversion through a common accumulator layout (on gfx1100 the native accumulation survives to the existing conversion boundary; on gfx1201 the native fragments are used without a gfx11-shaped intermediate). The qualified dispatch certificate (computed trellis forms, L=16 and the one L=15, row-f32 and bf16 scale classes, single band) is preserved as is, not replaced by a broader traits-derived rule. Launch geometry, wave count, shared memory placement and launch bounds stay part of the qualified design. A traits layer is zero-cost only if the generated code is unchanged, so the gate compares instructions.

Deliverables: the traits header and the unified fused tile as the first port (gated on both architectures, so it proves the seam), then the embedding butterflies and the head prefetch, only after the first port shows zero regression. Acceptance: a shared fused-tile source retains each architecture's native operand and accumulator representation, qualified geometry and dispatch certificate; both architectures give byte-identical results with no measured performance regression (cold and warm) and unchanged control kernels; resource records and instruction counts (WMMA, lane exchanges, barriers, loads, accumulator conversions) are compared with the shipped per-architecture versions and any difference is reviewed.

## 6. Layer 3: loadable kernel families (RCS22, gated on the RCS15 baseline)

The review's central point: **the row contract, not the translation unit, is the unit of runtime composition.** Source units decide what to compile; an independently loadable family must own complete, validated kernel rows and their dependencies. The current libr3 link is built around a complete upstream row table: the declaration includes the upstream-wide host declarations, generated stubs stand in for missing entry points and the registry drops rows that point at a stub. That works for one intentionally incomplete monolith and does not establish independent families (a family whose kernel needs another family's layout or scratch callback would have it stubbed and its row silently removed).

**Row ownership model.** Every row has a stable identifier, one owning family, and a complete callback set (launch, layout, relayout, scratch, init and fini, tuning), plus its concurrency, collective and chaining metadata. A callback that belongs to another family is either moved into an engine-owned common library or declared as a versioned dependency; it is never stubbed. Row order and dispatch priority are deterministic and independent of load order, because they affect which kernel wins, row-indexed behaviour and plan-cache decisions without changing arithmetic.

**Plugin contract.** Each family exports one versioned descriptor entry point returning its row table, compatibility identities and ownership metadata; memory owned by the library, which the loader keeps loaded for every plan and in-flight operation that uses it. Libraries load with local symbol scope and hidden visibility by default, an explicit versioned export surface, and no correctness dependency on constructor order or on another family's private symbols.

**Identity, not one hash.** `host_abi_id` (structure sizes and offsets, callbacks, enums, calling conventions, relevant definitions), `row_contract_id` (row ids, schemas, layout and scratch contracts, priorities, concurrency and chaining semantics), `device_abi_id` (target architecture, code-object version, device-linking convention, external device-symbol policy), `artifact_id` (library and code-object digests, source revision, patch set, toolchain, flags) and `composition_id` (hash of the ordered, resolved per-architecture manifest). Equal ABI ids are a necessary compatibility condition, not proof of equal behaviour. The plan cache keys on the composition and dispatch identity, not on family version strings.

**Device code.** First version: no relocatable device code and no external device symbols; each family carries the code objects for every architecture it supports, and each is registered per selected device with explicit failure cleanup. Code objects are inspected at release (instantiated templates, device globals and constant symbols can still create linkage). No unloading while GPU work or captured graphs refer to a family; no live swapping in the first version (a quiesced engine and a fresh plan).

**Release and rollback.** One sealed, atomic manifest per release describing the qualified combination, with exact file and code-object digests. Whole-release rollback first. Per-family rollback only later, after compatibility and cache invalidation are shown. Mixed gfx1100 and gfx1201 starts may load different family sets under one globally coordinated plan; the matrix covers them.

**Build graph.** One link per family, a per-unit compiler cache as today, a family test target; a full all-family build for release. Changing a widely included header rebuilds most families; measure it. The compile-only family targets (RCS20) are a development accelerator, not evidence that separate libraries link.

**Staging.** 22a prototype one independent family (link, load, register, test with complete callback ownership); 22b compose several (dependency closure, row order, symbol isolation, HIP module lifetime, rejection cases); 22c qualify a sealed release against the monolithic provider and the whole-release rollback.

**Qualification matrix.** Selected row ids, order, metadata and refusals; missing-family behaviour; byte outputs; resources; cold and warm performance; mixed-architecture starts, each against the current monolithic release.

Decision gate: if the RCS15 baseline shows a native RDNA3 provider matches the generated tree, layer 3 is built for the native provider instead and the gfx11 rewrites retire with the generated tree. If not, layer 3 is built on the generated tree. Layers 1 and 2 stand either way.

## 7. Immediate step before layer 1: the family compile targets (RCS20)

Refresh the unmerged family branch against main, run the full build, and merge it. It is the cheapest dev-loop gain and layer 3's starting point. Record, once, the build time of a one-family compile against a full plugin so the later benefit is measured, not assumed.

## 8. Relation to the native RDNA3 migration (RCS15 to RCS18)

- The Stage 0 item of the migration brief (the additive gfx12 body with distinct identifiers, no removed or duplicated anchors, both architectures built and gated) is the first user of layer 1 and is already in progress as the gfx1201 fused-tile package.
- RCS15 (freeze and measure) runs next, GPU-free parts first: refresh the upstream pin, inspect the native provider's rows, inventory local optimisations and classify each (upstream equivalent, port required, obsolete, rejected).
- Layer 2's traits seam is also the port surface for a native provider.

## 9. Sequencing and ownership

| Step | Plan item | Owner | Gate |
|---|---|---|---|
| Anchor gate, lint, patch-author guidance | RCS19 | lead | gate catches the recorded collision |
| Merge family compile targets, time one-family builds | RCS20 | lead | full build and component tests pass |
| Finish the gfx1201 fused tile with distinct names; promote | RCS19 (first customer) | bigcherry-2 | both architectures built, bytes equal, controls unchanged |
| Traits header and unified fused tile | RCS21 | lead | one source builds on both, records equal |
| Upstream preparation for the neutral patches | RCS21 | lead, with owner approval | evidence per patch |
| RCS15 baseline, GPU-free first | RCS15 | lead, with a design reviewer | matrix committed |
| Family plugins | RCS22 | decided at the RCS15 gate | loader and registry tests, full qualification |

GPT is used for review only: design review of this document and of each step, and diff review; where code is wanted it is returned as text in the reply and applied, built and committed by the owner. Builds and GPU gates stay with the lead and bigcherry-2.

## 10. Acceptance for the whole programme

A kernel change is a self-contained patch or rewrite with declared units, suffixed names, evidence in its own folder and a passing anchor gate; it builds and gates on its own family; arch-neutral changes have a single source for both architectures; releases name their family versions; no silent provider fallback; and for each neutral change there is a documented decision to keep it local or offer it upstream.

## 11. Owned units: managing upstream change (RCS23)

The three layers above keep upstream's source as an untouched input and change it by patches and rewrites. For the families we have already rewritten heavily that stops paying off: the IQR GEMM unit carries 440 lines of rewrites over a 1,343-line upstream file, the prefill header 134 over 309, the vocabulary head 121 over 536, the embedding gather about 57 over 142; 19 of 48 patch packages touch libr4d. For those units we take ownership of the source and manage upstream change explicitly. We do not fork libr4d as a whole.

**Measured upstream churn (12 releases, 20 October to 19 November 2026).** 309 upstream commits, 52 of them in libr4d. The files that would be owned changed in 14 commits in that month (the IQR GEMM unit 6, its decoder header 3, the prefill header 2, the vocabulary head 2, the embedding gather 1), about three a week. The row-table source file changed 29 times and is never owned. libr4d has about 170 files and 115 build units; we own about five. So the merge surface is small and bounded, and a unit we do not own costs nothing.

**What owning means.**
1. An owned unit is a plain source file in our repository, split by kernel family (GEMV, 16-row tile, wide tile, plane prefill, fused tile, head, embedding), written against the traits seam so one source builds for gfx12 and for gfx11.
2. Its patches and rewrites are retired (single owner per line), except small patches that touch only unowned units.
3. A manifest lists every owned unit: the file, the upstream commit and blob it was based on, the family, and the last upstream commit merged. The build takes owned units from our repository and every other unit from the pinned upstream source.

**Tracking upstream (the drift check, run at every pin triage and bump).**
- For each owned unit: diff the upstream change on that file between our recorded base and the new pin. Empty: advance the base automatically. Not empty: try a three-way merge of upstream's change into our file; a clean merge is applied on a branch and sent through the gates; a conflict becomes a plan item with the upstream diff and ours side by side.
- Include closure: a changed upstream header included by an owned unit is reported even when the unit itself did not change (the row ABI and shared types are the usual cause). Compile and link gates 2 and 3 catch what the diff cannot.
- Unowned units are unchanged from today: pinned, patched, rewritten, with gate 1.
- Exit: when upstream adopts a change of ours, the drift check shows our file equal to upstream's and the unit leaves the owned set (any local patch is archived unchanged).
- Entry: a unit joins when it carries several independent changes or a restructuring and the drift check shows it is mostly ours anyway. This is a policy judgement recorded in the manifest, not a numeric threshold.
- Hot shared files (the row table) are never owned: changes there are absorbed through patches with gate 1 and ABI checks.

**Library structure after this work.** The engine's loader contract stays: one library per architecture (libr4d for gfx12, libr3 for gfx11, libr2 for gfx1030). We do not rename or merge them in this phase; the names and files stay what the engine expects. What changes is where their source comes from: the owned families are one source tree built for both architectures (libr3 and libr4d each compile the same owned units with their traits), the unowned units still come from upstream, and the generated gfx11 tree shrinks to the unowned units and retires if a native gfx11 provider replaces it (RCS15). Splitting each library into loadable family plugins is the separate, later decision in section 6. So: separate per-architecture libraries remain as build outputs; we manage the source of the families we have changed, and upstream manages everything else.

**Risks and answers.** The merge burden: bounded by the measured churn and automated by the drift check. Silent drift of an owned file from upstream behaviour: the manifest base commit and the per-bump drift report make every divergence explicit. Upstream ABI change: caught by the include closure and by gates 2 and 3, and by the row-contract identity if family plugins exist. A bad first decomposition: the split is staged one family at a time, each step byte identical and resource equal to the shipped build.

**Staging.** (1) The inventory: for every libr4d unit, the share of lines we rewrite or patch and the number of patches that touch it, plus the upstream commit count per unit per release. (2) The manifest and the drift check, run against the last twelve releases to prove it reproduces the churn above. (3) Take ownership of one family (the fused tile and the embedding gather are small and recent), gated like any kernel change. (4) The remaining IQR units. (5) Review the set at each pin bump.

## 12. Naming: `bc` marks what we own

Rule: anything carrying `bc` is ours; anything without it is upstream's, pinned and untouched. The convention already exists in the repository (the `bc_*` implementation headers added by our patches, `bc-serve`, `bc-release`, the `libbcgfx12` interposer plugin, the `bigcherryrad` release archive); this extends it to the kernel providers so that no reader mistakes one of our builds for native or upstream.

- **Source files.** An owned unit is `bc_<family>_<kernel>.hip` (or `.h`). The manifest (section 11) records which upstream unit slot it replaces, so the build knows to leave that upstream file out.
- **Symbols and types.** Names we introduce carry a `bc_` prefix (kernels, tile types, certificates). Names the engine's row table requires of the unit slot keep their upstream spelling; they are the ABI.
- **Libraries.** Our builds of the architecture providers are `libbcr3.so` (gfx11) and `libbcr4d.so` (gfx12), once they contain owned units; the later family plugins are `libbc-<family>-<arch>.so`. A stock upstream provider, where one is loaded at all (a comparison arm, a reference), keeps its upstream name, so the RCS15 arms read unambiguously: arm A `libbcr3`, arm B upstream's native library. The old names are not kept as aliases.
- **Provenance in the identity.** The plugin identity string and the artifact identity carry a provider tag (`bc` or `upstream`) and the owned-unit manifest digest, so serving manifests, logs and the plan cache show which provider ran.
- **What the rename touches (all ours).** `archs.toml` (the `plugin` field is data), the component outputs, the plugin name string in the registry, the hard-coded `libr4d.so` search in the `libbcgfx12` plugin, the helper plugin default and its override setting, the serving layer's plugin field, the release asset inventory and `bc-release verify`. The engine's own plugin loader takes the path from configuration, so no change to upstream core is expected; stage 0 of RCS23 verifies this with a start of both architectures before anything else moves.
- **When.** One coordinated change at the start of RCS23, not piecemeal, with a release that restores cleanly (releases are self-contained, so the previous release keeps its old names). Until then `libr3.so` is already a generated build rather than upstream's, and the rename is the honest label.

## 13. Open questions

- Which of the neutral patches the owner wants offered upstream, and when.
- Whether `gemm_quant` is split before layer 3.
- Whether per-family rollback is wanted or whole-release rollback is enough.
- Whether the native provider replaces the gfx11 tree (decided by RCS15).
