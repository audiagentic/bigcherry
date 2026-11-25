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

## 4. Layer 1: composition seams (RCS19, build first, small)

**S1. Unit ownership.** Each managed patch declares the upstream units it edits (`units = [...]` in its metadata). Each native rewrite file is named after one unit already. A new check lists every unit edited by more than one layer.

**S2. Anchor discipline.** No patch edits a line that a rewrite block finds or replaces. Rewrite blocks that depend on a patch's text name that patch (a `consumes` note) so a patch change shows its dependents.

**S3. Anchor gate (the mechanical check).** For each patch PR: apply the patch set to the pinned source, then run the rewrite check with the real rules against that patched source. Failure means a conflict. It needs no GPU and takes seconds, and goes in the offline patch job. The only allowed failures are the known `bc_*` sources that exist after the bc patches (the gate applies them first).

**S4. Naming rule.** A patch or rewrite introducing a type, kernel or constant uses the suffix `<arch><patch-id>` (for example a gfx12 fused tile is `…12`, its gfx11 counterpart `…11`). Lint rejects an introduced name without the suffix, and the anchor gate rejects duplicate definitions through the compile of the patched source for both architectures' host passes.

**S5. Layer order.** pinned source, arch-neutral managed patches, libr3 rewrites (gfx11 specifics only), compile. A rewrite may rely on a patch only through a declared anchor.

First customer, resolved 2026-11-25: the gfx1201 fused-tile package keeps its shared-header edits; the five libr3 header blocks that anchored on the same text became find-1-or-0 with a paired identity block (the rewriter's paired-run rule keeps upstream drift failing); the patch's own tile carries gfx12-suffixed names and is compiled out of the gfx11 build. Lane builds of both architectures confirmed the gfx1100 records unchanged (0 added, 0 removed, 0 changed) and 11 fused kernels added on gfx1201 with no spill. This is the pattern S2 and S4 formalise.

Deliverables: the gate script and its test (including a fixture reproducing the fused-tile collision), lint rules for S1 and S4, a section in the patch-author skill, and the check wired into CI. Acceptance: the collision that blocked the gfx1201 patch is caught by the gate in seconds before any lane build.

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

Deliverables: the traits header and the unified fused tile as the first port (it is gated on both architectures, so it proves the seam), then the embedding butterflies and the head prefetch. Acceptance: one source for the fused tile builds on both architectures; resource records and bytes equal the shipped per-architecture versions.

## 6. Layer 3: loadable kernel families (RCS22, gated on the RCS15 baseline)

Replace "one plugin per architecture" by "one small plugin per family per architecture", composed at load from a manifest.

- **Families:** the eight above (finer splits later: `gemm_quant` into IQR and int8 if it pays).
- **Plugin contract:** each family plugin exports its row table, a family version and an ABI hash of the shared header set. Shared decoders and helpers stay header-only so families have no cross-plugin symbols.
- **Registry and loader:** the engine loader reads a family manifest per architecture and merges the row tables; a row present in two families is an error; a family required by the configuration but missing is an explicit refusal, never a silent fallback to a reference kernel.
- **Packaging and verification:** the release bundle carries the family plugins and the manifest; install and verify check each family's ABI hash and code objects. Rollback can swap a whole release first; per-family rollback is allowed only between releases with equal ABI hashes.
- **Build graph:** one link per family, a per-unit compiler cache as today, a family-level test target (family selftest and resource table), and a full all-family build only for release. Ninja stays one command per build directory.
- **Plan cache and identity:** the static identity of a start includes the loaded family versions; a changed family invalidates only what depends on it.
- **Risks:** an engine loader change as a managed patch; load time for several plugins (expected small); the n-gram and remote MoE helper contracts must not change meaning; the registry rule for duplicate rows.

Decision gate: if the RCS15 baseline shows a native RDNA3 provider matches the generated tree, layer 3 is built for the native provider instead and the gfx11 rewrites retire with the generated tree. If not, layer 3 is built on the generated tree. Either way layers 1 and 2 stand.

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

## 11. Open questions

- Which of the neutral patches the owner wants offered upstream, and when.
- Whether `gemm_quant` is split before layer 3.
- Whether per-family rollback is wanted or whole-release rollback is enough.
- Whether the native provider replaces the gfx11 tree (decided by RCS15).
