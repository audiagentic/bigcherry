# Composable Radiance kernel builds

> **Re-review (2026-11-25):** [Upstream comparison and implementation decisions](radiance-integration-review-20261125.md) compares the shipping pin with upstream main, corrects the provider/ABI and roster assumptions, and distinguishes discrete development gates from full migration and release acceptance. It controls where this design disagrees; no runtime changes or candidate repin.

> **Bump integration and implementation order:** [Radiance upstream bump integration](radiance-bump-integration.md) defines the shared MEN24 sequence, candidate identities and acceptance gates. Domain ownership remains unchanged; this is a target design, not an implemented bump command.

Status: layer 1 and the test harness are built (RCS19, RCS20, RCS25: composition gates, family targets, unit harness, pin-bump wiring). Layers 2 and 3 (traits seam, family plugins, overlay plugins: RCS21 to RCS23) are deferred, each with its reopen trigger in its plan item. The native provider was not adopted (RCS15 to RCS18). 2026-11-25.

## 0. Terms (one meaning each)

| Term | Meaning |
|---|---|
| upstream | Radiance as released by its maintainers; we pin one release (recipes.toml) and never edit it in place |
| libr4d | upstream's gfx12 kernel library source (units named `r4d_*`); built for the R9700 as the `libr4d.so` plugin |
| libr3d | upstream's own native RDNA3 kernel library, a different upstream library from libr4d; status in our builds is what RCS15 measures; not used in production today |
| libr3 | our gfx11 build of upstream's units: libr4d's units compiled for gfx11 through the compatibility header plus our native rewrites; built as `libr3.so`; upstream-derived, so it keeps its name |
| libr2 | our gfx1030 helper library (no matrix instructions) |
| shim | `r3_compat.h`: emulates gfx12-only instructions on gfx11 |
| native rewrite | a `native/*.rw` block that replaces text in a libr4d unit for the gfx11 build |
| managed patch | a package under `engines/radiance/patches` that edits the pinned source for every architecture, with a lifecycle state (untested, validated) |
| overlay plugin | a library of kernels we own (`libbc-<family>-<arch>.so`, sources `bc_*`), placed first in the engine's kernel hierarchy so its rows override the base libraries' rows (RCS23); the existing precedent is `libbcgfx12` |
| family | a group of units by purpose (`core`, `gemm_quant`, `gemm_dense`, `attention`, `moe`, `speculation`, `multimodal`, `collectives`) |
| `bc` | prefix and suffix meaning "ours"; no `bc` means upstream |

## 1. Why

Our gfx1100 kernels are a generated tree: libr3 compiles upstream's gfx12 units for gfx11 through a compatibility header (`r3_compat.h`) and a set of native rewrites (`native/*.rw`). Managed patches edit the same upstream source text before libr3 rewrites it. Three costs followed from that:

1. **Conflicts between owners of the same text.** The gfx1201 fused-tile patch passed every gate (byte identical, no spill, all 24 single-band points faster, 0.48 to 0.86 of shipped time) and still could not build: it edited, or inserted, the same lines the merged gfx1100 rewrites anchor on, so the anchors vanished and two definitions of the same type appeared. Nothing in the patch tests or lint could see it; it was found by a full lane build.
2. **Gate loops measured in a full plugin build.** Every candidate is built as the whole plugin (about 7,800 kernel records) before it can be hashed, replayed and resource-compared, even when one unit changed.
3. **No clean unit of release, rollback or upstream contribution.** The wins of the last two weeks (head tile, accumulate-once, native fragments, fused tile, embedding butterflies, NT4 head prefetch, chained long tiles) are each a small change to one kernel family, but ship and revert only as part of the whole plugin.

Goal: make a kernel change a self-contained, mechanically checkable unit that composes with every other change, builds and gates on its own family, and, where it is architecture-neutral, can be offered upstream.

## 2. Current structure (facts)

- Upstream's own unit list defines the compile units; libr3 reads it from upstream's build file. Rows whose unit is left out are dropped at load.
- Application order today: pinned source, managed patches, libr3 configure (rewrites, compat layer), compile, link into one `libr3.so`; a separate native plugin for gfx1201 (`libr4d`), built from the patched source without rewrites.
- An earlier effort (PR 1545, RCS20) groups every upstream unit into eight families: `core`, `gemm_quant`, `gemm_dense`, `attention`, `moe`, `speculation`, `multimodal`, `collectives`, with compile-only per-family targets and a coverage test that each unit has exactly one family. It shares objects with the full build and produces no loadable plugin.
- The engine's own plugin loader already supports several kernel libraries at once: it loads them with local symbol scope, orders them by an explicit hierarchy (`--kernels` or `RADIANCE_KERNELS`), and selects each op's row from the earliest library in that order; an override plugin may replace selected rows of the libraries below it, and the loader checks the combined set. Our `libbcgfx12` plugin already uses this to override selected gfx12 rows. What assumes one primary library per architecture is our deployment data (`archs.toml`, the helper configuration) and some of our code that finds `libr4d.so` by file name. Helper contracts (remote MoE, n-gram) have their own ownership.
- Wins that exist only because of the shim (accumulate-once native layouts, native fragments, native head sums) are gfx11-specific by nature. Wins that are arithmetic-neutral (fused single-band tile, wave-local embedding butterflies, head prefetch lifetime, weighted-head ownership across ranks) are not.

## 3. Principles

- One owner per line of source: every line is edited by at most one layer (a managed patch or a rewrite), and the overlap is checked by a machine, not discovered by a build.
- Additive before invasive: new kernel entry points and guarded dispatch beat edits to shared bodies. A shared-body edit is a template parameter whose default reproduces the old code.
- Names carry their owner: a symbol introduced by a patch or rewrite carries an architecture/patch suffix, so two owners can never define the same name.
- Derive, never hand-pick: tile selection comes from architecture traits, form fields and band counts, as in the guards already merged.
- Evidence travels with the unit: resource table, byte identity, cold and warm replay, in the patch's own folder.
- Test the unit, not the world: a change is checked with a small harness on the unit that changed (section 4a); full-plugin, release and serving runs are for releases.
- Checks catch mistakes, not attackers: there is no adversary in this repository, so no cryptographic digests, fingerprints or tamper checks are added. Git commit ids, plain comparisons and version strings are the identity machinery.
- Do not build machinery for a tree that may be retired: layers 1 and 2 pay off with the generated tree or with a native provider; layer 3 waits for the RCS15 baseline.

## 4. Layer 1: composition seams (RCS19, build first)

A design review (2026-11-25) showed that a textual anchor check proves less than this section first claimed. The rewrite checker verifies match counts and the paired-run rule; it does not parse C++, preprocess architecture macros, compile host and device passes, or link. So composition is checked by four separate gates, each saying only what it proves.

**Gate 1, anchors (seconds, offline).** Apply the full ordered patch set to the pinned source (the bc patches included), then run the real rewrite check against that patched source. It proves: every required anchor resolves exactly as declared (an alternative group needs at least one match, the rule the rewriter already has); declared dependencies exist; edited regions do not overlap. It does not prove the result is valid C++.

**Gate 2, compile.** Compile the affected translation units for gfx1100 and gfx1201 (host and device passes, the feature macros each architecture sets, explicit template instantiations of the touched kernels). It catches a duplicate type, a repeated template default argument, a device-only definition clash, and an architecture guard that selects the wrong branch. Run per pull request on affected units, with the compiler cache; not part of the seconds-scale gate.

**Gate 3, link and export.** Link the affected family (or a contract fixture) and inspect duplicate definitions, unresolved symbols, exported names and unexpected external device symbols.

**Gate 4, qualification.** Resource table, byte identity, selftests and replay for the affected family and for the units that include a changed shared header.

**Source ownership (replaces the coarse unit rule).** Three separate records: *affected units* (what must rebuild and retest, expanded through the transitive include graph, so a changed shared header invalidates its dependents); *owned regions* (the lines a transformation edits or inserts, to detect overlap); *consumed anchors* (a rewrite that relies on a patch's text names that patch and the anchor). Two changes may touch the same unit when their regions do not overlap, and may collide through a shared header when their units differ; the records handle both.

**Naming.** A name that two architectures' implementations must both define takes an architecture suffix (for example a gfx12 and a gfx11 fused tile). Suffixes are a convention, not a uniqueness proof: uniqueness comes from the symbol check in gate 3. Upstream-required names and ABI names are never renamed, and neutral algorithms intended for upstream carry no local patch identifier.

**Freshness.** The former missing patched-input dependency was fixed in #1546. Reuse that CMake dependency and verify upstream-header regeneration in an incremental directory; no additional digest mechanism.

**Layer order.** pinned source, arch-neutral managed patches, libr3 rewrites (gfx11 specifics only), compile. A rewrite may rely on a patch only through a declared anchor.

First customer, resolved 2026-11-25, initially merged as an untested package and subsequently promoted by #1554 (the general gates are the RCS19 tooling, in review): the gfx1201 fused-tile package keeps its shared-header edits; the five libr3 header blocks that anchored on the same text became find-1-or-0 with a paired identity block; the patch's own tile carries gfx12-suffixed names and is compiled out of the gfx11 build. Lane builds of both architectures confirmed the gfx1100 records unchanged and 11 fused kernels added on gfx1201 with no spill. Gate 1 would have caught the vanished anchors; the duplicate definition needed gate 2.

Deliverables: gate 1 with fixtures (vanished anchor, doubled anchor, missing required alternative group; multiple matches only fail explicitly exclusive groups), gate 2 with failing fixtures (duplicate type, repeated template default, device-only definition, conflicting conditional compilation), the ownership records in patch metadata, the freshness dependency and its test, and the patch-author guidance. Acceptance: every managed patch set applies to the pinned source; every required anchor resolves under its declared variant; invalid declarations, duplicate definitions and symbol conflicts are rejected for both architectures before a lane build; no family is qualified until the affected and dependent kernels pass their correctness and resource gates.

## 4a. Testing discretely

A kernel change should be provable in minutes on one card without composing the whole system. Build the small harnesses once, reuse them everywhere, and run the large checks only at release.

| What changed | Check | Cost |
|---|---|---|
| a patch or rewrite anchor | gate 1 | seconds, no GPU |
| a header, template or declaration | gate 2a (syntax, host and device, both architectures) | about a minute, no GPU |
| one kernel unit or family | unit harness: build just that unit (or the family) into a small test plugin, then on one card: compiled resource record against the shipped build, output hash against the shipped build over the shapes and row counts the unit serves, cold and warm replay of the affected rows | minutes |
| a row's selection or contract | row-catalogue comparison between the base and the overlay (a test, no GPU) | seconds |
| a whole release | Gate E, then serving and parity only for the rows whose kernels changed (carry the rest) | release time |

**The unit harness** is the piece to build, once: a small tool that compiles one unit or family into a loadable test plugin with a loader that tolerates the absent rows, hashes selected rows' outputs and replays them with the adaptive launch rule (stop when two windows agree). The pieces already exist in scratch form (the single-unit plugin builds, the cross-plugin output-hash selftest, the replay tool with a cold flush); the work is to make them one repository tool with a stable command. Tracked as RCS25.

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

The loader already composes several kernel libraries by hierarchy, so a family plugin is not a new loading mechanism: it is an override library in the existing hierarchy (section 11 makes that the primary way to own kernels). What the loader does not give us, and this layer adds, is a sealed manifest, complete callback ownership per family, stable row identities across libraries, a composition identity and a lifecycle policy.

The review's central point: **the row contract, not the translation unit, is the unit of runtime composition.** Source units decide what to compile; an independently loadable family must own complete, validated kernel rows and their dependencies. The current libr3 link is built around a complete upstream row table: the declaration includes the upstream-wide host declarations, generated stubs stand in for missing entry points and the registry drops rows that point at a stub. That works for one intentionally incomplete monolith and does not establish independent families (a family whose kernel needs another family's layout or scratch callback would have it stubbed and its row silently removed).

**Row ownership model.** Every row has a stable identifier, one owning family, and a complete callback set (launch, layout, relayout, scratch, init and fini, tuning), plus its concurrency, collective and chaining metadata. A callback that belongs to another family is either moved into an engine-owned common library or declared as a versioned dependency; it is never stubbed. Row order and dispatch priority are deterministic and independent of load order, because they affect which kernel wins, row-indexed behaviour and plan-cache decisions without changing arithmetic.

**Plugin contract.** Each family uses the existing native plugin info, schema and row exports, plus the native ABI version and layout fingerprint; no new descriptor or loader. The library owns its callbacks and stays loaded for each plan and in-flight operation that uses them. Libraries load with local symbol scope and hidden visibility by default, an explicit versioned export surface, and no correctness dependency on constructor order or on another family's private symbols.

**Identity.** Keep it simple: each family plugin reports a name, a version string and the pinned upstream release it was built against; the loader already checks both ABI version and layout fingerprint, rejects duplicate plugin names and inconsistent schemas, and allows alternative rows for an op. A managed release must verify actual chosen providers/rows and reject unapproved shadows and missing required coverage. Do not add new hashes. The row contract (row ids, schemas, layout and scratch contracts, priorities, concurrency and chaining semantics) is compared by a test between the base and the overlay, not hashed. The plan cache keys on the loaded plugin names and versions.

**Device code.** First version: no relocatable device code and no external device symbols; each family carries the code objects for every architecture it supports, and each is registered per selected device with explicit failure cleanup. Code objects are inspected at release (instantiated templates, device globals and constant symbols can still create linkage). No unloading while GPU work or captured graphs refer to a family; no live swapping in the first version (a quiesced engine and a fresh plan).

**Release and rollback.** One manifest per release listing the family plugins and their versions; installing a release replaces the set as a whole. Whole-release rollback first (the previous release root is kept). Per-family rollback only later. Mixed gfx1100 and gfx1201 starts may load different family sets under one plan; the test matrix covers them.

**Build graph.** One link per family, a per-unit compiler cache as today, a family test target; a full all-family build for release. Changing a widely included header rebuilds most families; measure it. The compile-only family targets (RCS20) are a development accelerator, not evidence that separate libraries link.

**Staging.** 22a prototype one independent family (link, load, register, test with complete callback ownership); 22b compose several (dependency closure, row order, symbol isolation, HIP module lifetime, rejection cases); 22c qualify a sealed release against the monolithic provider and the whole-release rollback.

**Qualification matrix.** Selected row ids, order, metadata and refusals; missing-family behaviour; byte outputs; resources; cold and warm performance; mixed-architecture starts, each against the current monolithic release.

Decision gate: if the RCS15 baseline shows a native RDNA3 provider matches the generated tree, layer 3 is built for the native provider instead and the gfx11 rewrites retire with the generated tree. If not, layer 3 is built on the generated tree. Layers 1 and 2 stand either way.

## 7. Immediate step before layer 1: the family compile targets (RCS20)

The family branch is refreshed on current main and in review as PR 1545 (RCS20): eight family targets and the coverage test, compile-only. It is the cheapest dev-loop gain and layer 3's starting point. Once merged, record the build time of a one-family compile against a full plugin, cold and warm, so the benefit is measured, not assumed.

## 8. Relation to the native RDNA3 migration (RCS15 to RCS18)

- The Stage 0 item of the migration brief (the additive gfx12 body with distinct identifiers, no removed or duplicated anchors, both architectures built and gated) is the first user of layer 1 and is already in progress as the gfx1201 fused-tile package.
- RCS15 (freeze and measure): part one is done (docs/evidence/2026-11-25-rcs15-libr3d-inventory): upstream's native RDNA3 library has no row for the IQR or the local int8 form, so the overlay model is needed whichever way the provider decision goes. Part two, the matched arms on one card for the remaining rows (MXFP4, E4M3, GDN, MoE, DFlash, attention, helpers), decides the provider.
- Layer 2's traits seam is also the port surface for a native provider.

## 8a. Programme map (plan items)

| Item | What | Depends on |
|---|---|---|
| RCS15 | baseline: upstream libr3d against our libr3, matrix of rows | none |
| RCS16 | native compute migration per family (only if RCS15 says libr3d is at least as good) | RCS15 |
| RCS17 | transport and helper integration for a native provider | RCS16 |
| RCS18 | production cutover and retirement of the generated gfx11 tree | RCS15 to RCS17 |
| RCS19 | composition gates (anchors, compile, link, qualification), ownership records, freshness | none; first customer the gfx1201 fused tile |
| RCS20 | merge the family compile targets | none |
| RCS21 | traits seam, unified fused tile, upstream path | RCS19 |
| RCS22 | loadable family plugins | RCS15 gate, RCS19 |
| RCS23 | owned kernel families as overlay plugins, manifest, drift check, `bc` naming | RCS19, RCS21 |
| RCS24 | cleanup pass: stale docs, tests, dead code | none |
| RCS25 | discrete kernel test harness (section 4a) | RCS20 |

RCS15 to RCS18 decide the gfx11 provider; RCS19 to RCS23 make whichever provider we keep composable and manageable. Documents they supersede or qualify are listed in section 8b.

## 8b. Related documents and their status

- `docs/reviews/radiance-native-libr3d/ASSESSMENT.md`, `INTEGRATION.md`, `native-rows.tsv`: written against pin 1.17 and an earlier upstream snapshot; input to RCS15, not a statement of current facts.
- `engines/radiance/components/libr3/README.md`: component description; its terms follow section 0.
- `docs/reference/build/PIN_BUMP.md` and the bump-radiance skill: the pin-bump flow; the planned drift check for owned kernel overlays (RCS23) is noted there and is not implemented yet.
- `engines/radiance/components/libr3/exclude.txt`: units our gfx11 build leaves out; currently none.
- `engines/radiance/components/libr3/native-shelved/`: rewrite files kept as source but not built (each reason is in `component.toml`).

## 9. Sequencing and ownership

| Step | Plan item | Owner | State | Gate |
|---|---|---|---|---|
| gfx1201 fused tile de-conflicted (first customer of the seams) | RCS19 | bigcherry-2 | merged as an untested package (rad0065); promotion next | both architectures built, bytes equal, controls unchanged |
| Composition gates 1 and 2 (`patch-compose-check`) | RCS19 | bigcherry-2 | in review | fixtures fail with the expected diagnostic; merged rad0065 passes |
| Patched-source freshness dependency | RCS19 | lead | merged #1546 | regeneration test |
| Ownership records in patch metadata | RCS19 | lead | after the gates exist | records checked in the gates |
| Family compile targets | RCS20 | lead | in review | full build and component tests; one-family time recorded |
| RCS15 part one (inventory) | RCS15 | lead | done | evidence note |
| RCS15 part two (matched arms) | RCS15 | bigcherry-2 for the build and runs | not started | matrix committed; decides the provider |
| Traits header and unified fused tile | RCS21 | lead | after RCS19 gates | one source builds on both, records equal, instructions compared |
| Owned overlays: manifest, drift check, embedding gather first | RCS23 | lead | design merged | fixtures, twelve-transition replay, first overlay equivalent to the base |
| Family plugin contract | RCS22 | decided at the RCS15 gate | not started | loader and registry tests, full qualification |
| Discrete kernel test harness (build one unit, hash, replay) | RCS25 | bigcherry-2 | not started | reproduces the recorded fused-tile resource, hash and replay results in minutes |
| Stale docs, tests and dead code | RCS24 | bigcherry-2 (build GC), lead (rest) | build GC cleanup in review | grep finds no live reference to removed concepts |

GPT is used for review only on this work: design review of this document and of each step, and diff review; code it supplies comes back as text and is applied, built and committed by the owner. Builds and GPU gates stay with the lead and bigcherry-2.

## 10. Acceptance for the whole programme

A kernel change is a self-contained patch, rewrite or overlay with evidence in its own folder; it passes the composition gates (provenance and anchors, compile, link and export, qualification) before a lane build; it builds and gates on its own family; arch-neutral changes have a single source for both architectures; owned kernels are overlay plugins with an origin manifest and a transactional upstream check at each pin bump; releases name their overlay and family versions; there is no silent provider fallback; and for each neutral change there is a recorded decision to keep it local or offer it upstream.

## 11. Owned kernels as overlay plugins: managing upstream change (RCS23)

Review outcome (2026-11-25), recorded because it changed the design: the first version of this section proposed taking whole source files into our tree and substituting them for upstream's units inside the upstream build. A review pointed out that the engine's loader already composes kernel libraries by hierarchy, and that unit substitution needs a new build-system seam (the unit list is parsed from upstream's build file and compiled by wrappers that always include upstream's file) plus fail-closed handling of upstream renames and splits. The overlay model needs neither.

**The model.** Upstream's libraries stay byte-for-byte what upstream builds: libr4d for gfx12, and for gfx11 our build of the same units through the compatibility layer (libr3). The kernels we own live in our repository as families (`bc_<family>_<kernel>` sources written on the traits seam) and are built as separate **overlay plugins**, `libbc-<family>-<arch>.so`, placed first in the kernel hierarchy. For an op where an overlay row exists, the loader selects it; everything else falls through to the base library. Removing an overlay restores upstream's behaviour exactly. `libbcgfx12` is the existing precedent.

**What this buys over unit substitution.**
- No change to upstream's build list, no source resolver, no per-architecture include ordering trick, no risk of two definitions of one exported kernel.
- The base libraries stay upstream-derived and keep their names; our code is recognisable by `bc` alone (section 12).
- Each overlay builds, gates, versions and rolls back on its own, which is the layer-3 benefit (section 6) reached incrementally on a loader that exists.
- Ownership is per family and reversible: a family that upstream catches up with is deleted.

**What we must own for an overlay row.** The row descriptor (op, schema, layout, relayout, scratch, init and fini, tuning, concurrency, collective and chaining metadata) and the kernel. Upstream decoders and shared headers are compiled in from the pinned source, so a changed upstream header changes our overlay too: that exposure is what the drift check watches.

**Measured scale.** Over 12 upstream releases (20 October to 19 November 2026): 309 commits, 52 in libr4d, 14 on the files we would shadow first (the IQR GEMM unit 6, its decoder header 3, the prefill header 2, the vocabulary head 2, the embedding gather 1); the row-table source changed 29 times. libr4d has about 170 files and 115 build units; the first overlays shadow about five units. File-level churn is low; the exposure that matters is shared headers and the row table, because they can change what an unchanged kernel means.

**Origin mapping.** Each overlay family records, in a manifest, its upstream origins and what it replaces: `origins = [{ path, base_commit, last_integrated_commit }]`, the build slots and the rows it overrides, the architectures it covers, the patches and rewrites its source absorbed, and an integration strategy per origin: `whole-unit` (our file is upstream's file plus our changes: three-way merge applies) or `extracted-family` (our file is a carved-out part of upstream's unit: an upstream change to the origin needs structural review, never a text merge). One origin may feed several owned files; one owned file may rest on several origins.

**Pin triage and bump: an acceptance transaction, not a merge.**
1. Verify provenance: the base commit exists and is an ancestor of the target; compare the origin path and source changes using git; otherwise the unit is `unverified`, never clean.
2. Classify each origin: unchanged; changed and cleanly mergeable (candidate diff only); conflicting, renamed, split, deleted or ambiguous (blocks automatic integration, opens a reconciliation item).
3. Include closure: report changed headers reachable from any owned source for each architecture and pass, preferring the compiler's dependency output over text search; an incomplete closure is reported as incomplete and widens the compile set.
4. Row contract snapshot: compare row presence, schemas, constraints, priority, callback ownership and chaining semantics of the shadowed rows between the old and new pin; an unchanged owned kernel under a changed row or layout contract is the failure this catches.
5. Gates: composition gates 1 to 3 (section 4), then qualification (resource, byte identity, replay) per overlay.
6. The manifest advances only after the checks pass: the examined upstream revision is recorded at triage, the integrated one at the end, as plain git commit ids. A stopped bump leaves the integrated revision where it was.
Outcomes per origin: no change (base verified), candidate (diff plus gates required), blocked (conflict, rename, split, uncertain), unverified (tool failure, missing history, incomplete closure).

**Adoption and exit.** Two levels: change-level adoption (upstream took one of our changes: retire that change after verifying equivalence) and unit-level exit (the whole overlay row set can be dropped only when every retained local change is gone and upstream's rows are qualified in its place).

**Library structure at the end.** Unchanged for the engine: libr4d (gfx12), libr3 (gfx11, generated from upstream units), libr2 (gfx1030) keep their names and loading. Added: a handful of `libbc-<family>-<arch>.so` overlays, first in the hierarchy. libr2 and the native gfx11 provider (libr3d, if RCS15 selects it) stay as they are unless an overlay explicitly covers their architecture. Splitting the base libraries themselves into families is not part of this plan.

**Where a shared-body edit used to be a patch.** An overlay compiles its own copy of the body it changes (for example the prefill body for the fused tile); the copy is an owned origin with an `extracted-family` or `whole-unit` strategy. That is the price of not editing upstream's text, and it is bounded by the manifest.

**Staging (gated on RCS15 for the large gfx11 work).**
0. Baseline: per-architecture snapshot of build units, row catalogues, callback dependencies and selected behaviour, including the gfx1030 exclusions.
1. Inventory: every libr4d unit's share of lines we rewrite or patch, the patches touching it, and upstream commits per unit per release.
2. Manifest, provenance verifier, dry-run merge, include closure and row-contract snapshot, with fixtures: unchanged owned file, changed file, wrong base blob, non-ancestor target, rename, split, moved header, changed indirect header, changed row priority, an unchanged kernel under a changed layout contract, a clean text merge that fails qualification, an upstream adoption.
3. Replay the twelve release transitions one by one (not first against last: a change introduced and later reverted disappears in an endpoint diff); every transition classified, no unexplained clean result.
4. First overlay: the embedding gather (a distinct source boundary): row and export equivalence, byte identity, resource parity, cold and warm no regression, removal restores the base behaviour.
5. Second overlay: the fused tile on the traits seam, both architectures.
6. The large IQR GEMM, prefill and head families, after the RCS15 decision, so the migration is not done twice.
7. Two successive real pin bumps run through the new process with complete reports, verified resume and rollback.

## 12. Naming: `bc` marks what we own

Rule: anything carrying `bc` is ours; anything without it is upstream's or built from upstream's units, pinned and untouched. The convention exists in the repository (the `bc_*` implementation headers added by our patches, `bc-serve`, `bc-release`, the `libbcgfx12` plugin, the `bigcherryrad` archive); this extends it to the owned kernels.

- **Source files.** An owned kernel source is `bc_<family>_<kernel>.hip` (or `.h`).
- **Symbols and types.** Names we introduce carry a `bc_` prefix. Names an overlay row must keep for the engine (op names, row identities) keep their upstream spelling: they are the interface.
- **Plugins.** Our overlays are `libbc-<family>-<arch>.so`. The base libraries keep their names (`libr4d.so`, `libr3.so`, `libr2.so`): they are upstream's units, so they carry no `bc`. There is no rename of the base libraries and no alias.
- **Identity.** Each overlay reports a plugin name, a version string and the upstream release it was built against, so logs and serving manifests show which overlays ran. The base libraries report as upstream-derived.
- **What a new overlay touches.** The kernel hierarchy in the serving configuration (our overlays first), the release asset inventory and `bc-release verify`, the helper plugin default where a helper loads an overlay, and nothing in the engine: its loader takes the hierarchy from configuration.

## 13. Open questions

- Which of the neutral patches the owner wants offered upstream, and when.
- Whether `gemm_quant` is split before family plugins.
- Whether per-family rollback is wanted or whole-release rollback is enough.
- Which provider serves the remaining gfx11 rows (decided by RCS15 part two); the IQR and IQ8R rows are ours under either answer.
- Whether libr2 (gfx1030) shares any owned overlay source, or keeps its own.
