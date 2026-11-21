---
id: RUM02
order: 4
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-18T01:28:00.655993+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P1
work: L
---

# Extract injected BigCherry implementations into owned components and narrow hooks

## Description

Move large source-creation literals and duplicated policy from patch.py into ordinary owned source files. Keep only concrete typed integration hooks; reuse RCS13 callable inventory and RCS01 ownership mapping. Preserve behavior before mixing extraction with upstream ports.

## Steps

- Complete RUM04's native-boundary decision and removal first; do not extract rad0009/rad0011 bodies whose current purpose is obsolete.
- Build the keep/delete map from actual callers and RSC13 inventory: keep remote execution/admission, measured memory/plan checking, exact rank constraints, shared group adapter and qualified diagnostics.
- Move only surviving CPU implementation and tests from raw patch strings into existing serving/moe-remote owners and declared CMake targets.
- Keep necessary architecture template adapters as small owned headers with typed operand/handle views; preserve RAD_OP dependency declarations and fused route/combine semantics.
- Reduce patch.py to concrete linkage/include/call-site changes. Remove the old created payload in the same change, and hash/link all owned source inputs.
- Separate moved lines from real deletion and verify a no-behavior-change slice only where extraction is still warranted.

## Detailed Solution & Technical Design

Extraction is not the first objective: preserve fewer concepts. rad0009 startup translation should mostly disappear through upstream YAML; measuring/expected-plan hooks from rad0011 may survive. Classify first, delete native duplication second, extract actual BigCherry behavior third.

Retain ordinary direct typed calls and the existing extern-C plugin contract where needed. Do not add an extension event bus, service locator, callback factory or generic lifecycle manager. The first supported case is one engine and disjoint helpers, so full multi-engine helper ownership is not a prerequisite.

Source movement improves patch review but does not shrink compiled code. Net reduction comes from removed startup bridge/projections, proven upstream replay/governor reuse and redundant internal callers, not reorganizing the same text.

## Code Samples & Guidance

Sequence:
```text
native ownership classification -> remove obsolete bridges/mirrors
-> enumerate actual surviving callers -> extract surviving implementation
-> minimal target attachment sites -> focused boundary proof
```
See Astra removal map for exact surviving versus removable source clusters.

## Files

- engines/radiance/patches/rad0002_moe_remote_handoff/patch.py
- engines/radiance/patches/rad0009_serving_manifest/patch.py
- engines/radiance/patches/rad0011_plan_memory/patch.py
- engines/radiance/components/serving/bc_serving_v2.cmake
- engines/radiance/components/moe-remote/CMakeLists.txt
- engines/radiance/engine.toml

## Validation

Canonical patch apply/idempotence for the changed hooks, one touched-component build and the existing callback-boundary tests. Compare old/extracted semantic projections or outputs for one fake-engine case; do not run GPU performance tests for source relocation alone.

## Effort & Risk

L: template adapters and callback visibility can create hidden coupling. Extract in small component-bound slices, preserve symbol/ABI boundaries, and avoid introducing a new abstraction merely to relocate code.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

Surviving large implementations in changed packages move only where doing so reduces repeated upstream attachments or build drift. Obsolete implementations are deleted first. Owned files participate in build identity/packaging; typed hooks preserve behavior. Moving every payload or relocating on the old pin is not an initial-start gate.

## Notes

Prerequisites RUM01/RUM03/RUM04. The Astra source review ASTRA-CONSOLIDATION.md supersedes extraction-first ordering. Coordinate RCS13 inventory and RCS01/RCS08 component ownership. Remove obsolete native bridges/replay/policy before moving surviving code; movement is not net deletion.

2026-11-20 owner direction: convert every BigCherry patch and extension to 2.2 and leave the next bump as easy as possible (document, simplify, consolidate, reuse). State on the candidate (patch-rebase-check per source): production 32/32, radiance-diag 33/33, routing-evidence 33/33, vtp44-evidence 36/36 clean; all six components built-against the 2.2 commit. Scope added to this item: (1) obsolete first: archive, unchanged, per PATCH_SYSTEM 'Retired patches' the patches 2.2 superseded: rad0005 (upstream FFmpeg 6 writer), rad0037/rad0038/rad0040/rad0041 (replaced by rad0048-rad0053); rad0013 follows RUM08's retirement proof. `patch-rebase-check --all` must then be clean, so a future bump has one probe. (2) Footprint: record anchors and edits per patch (docs/reviews/radiance-upstream-2026-11-18/patch-footprint.json) before and after; consolidate where it cuts repeated upstream attachments: the largest are rad0009 serving manifest, rad0017 rendezvous, rad0051/rad0053 KV kernels and layouts; move injected bodies into owned components behind narrow typed hooks, so that a bump re-anchors hooks, not bodies. (3) Reuse: one shared row-registration helper for KV geometry rows so a new model geometry is one table entry (RUM11 was a missed geometry). Acceptance adds: no superseded patch outside archive/, --all clean, footprint report committed with the before/after anchor counts.


### 2026-11-21 RUM02 baseline footprint (analysis only)

Generated from the declared source compositions on pinned Radiance 2.2.0 `b6b8751cccb0e834a13ae9e5afca824ee9a487aa` with `bigcherry --engine radiance patch-footprint`; verify after any recipe/patch change with `--check`. Committed artifacts: `docs/evidence/radiance-patch-footprint/{footprint.json,footprint.md}`. Union of five declared sources (including the stock source): **38 composed packages, 706 edits, 679 non-create anchors** (666 upstream-owned, 13 on patch-created files), 714 expected regex match sites, 27 created-file edits, **80 upstream-owned anchor files**, 84,957 regex anchor bytes and 503,500 inserted/replacement text bytes. Every number is lexical definition footprint, not actual source matching, changed net bytes or source/build parity. Historical 2026-11-18 patch-footprint.json remains untouched; this is the 2.2 starting baseline, not a completed before/after extraction result.

**Ranked conditional consolidation proposals** (anchor reductions are per-FilePatch estimates, not a committed code change or a promise of ABI/semantic parity):

| Priority | Package; measured anchor edits | Candidate existing attachment sites -> typed hooks | Estimated reduction | Constraint |
| --- | --- | --- | ---: | --- |
| 1 | `rad0051_b16_kv_kernels`; 97 | 25 prefill + 29 decode kernel edits in two `libr4d/r4d_attn_*h256_gqa6.hip` files -> one owned typed implementation hook each | **52** | Preserve cache byte layout and ABI44 kernel parity; test host shape and actual compiled HIP targets before extending to other 43 edits |
| 2 | `rad0053_v4_kv_layouts`; 62 | `libr4d/r4d_rows.cpp` 9 row/chain edits -> 1 registration hook (8); `core/mem/packed_kv.h` 7 descriptor/geometry edits -> 1 owned descriptor hook (6) | **14** | Do not paper over upstream-owned ABI paths; require native row and packed-layout tests |
| 3 | `rad0009_serving_manifest`; 11 | Nine `core/main.cpp` edits -> at most two managed native-v3 startup call sites | **7** | **RUM04 removal first:** delete superseded v2 argv/manifest duplication, then move only surviving 37,503 authored bytes of created manifest .h/.cpp to `components/serving` behind CMake; moving create payloads alone saves zero anchors |
| 4 | `rad0017_vtp14_n3_shared_rendezvous`; 14 | Nine `libr4d/r4d_ar_entry.hip` rendezvous/lifecycle edits -> two typed calls to an owned shared-slot implementation | **7** | Keep rank-buffer publication, poisoning, teardown and concurrency semantics; component linkage and lifecycle proof first |
| 5 | `rad0052_b16_kv_geometries`; 24 | Five `libr4d/r4d_rows.cpp` constraints, rows and chain edits -> one shared KV-row helper | **4** | Coordinate with rank 2: avoid re-counting its eight row-site savings; preserve mixed per-rank geometry |

The five disjoint candidate slices total **up to 84 anchor edits removed** from this 679-anchor baseline (roughly 12.4%), conditional on replacing the listed edits by the enumerated hooks. Other high-anchor packages, including `rad0016` (39) and `rad0045` (36), need separate caller/lifetime classification; high count by itself does not justify extraction. Source movement is not net code deletion.

**RUM11 lesson:** make one shared, typed KV row registration path consumed by the 0052/0053 row/chain sets (and related Qwen geometry), with per-layout capability and per-rank geometry in declarative data. A new supported shape should require one row, not repeated source edits. Constrain actual kernel requirements (e.g. BF16 query, head_dim=256, block_size=16, E4M3 cache for that resident row), **not** arbitrary local `n_head`, `n_head_kv` or rotary-width equalities; keep any real GQA restriction where its kernel demonstrably needs it. This is a proposal, not implemented in this analysis-only slice. The next extraction requires RUM04 classification/deletion, exact-recipe rebase proof, component build, CPU geometry tests, then leased-GPU parity separately.

This analysis-only change does not edit the patch packages: observed anchor count before/after is **679 -> 679**; the up-to-84 reduction is for future guarded extraction slices. Host validation: 5/5 generator tests, 25/25 Radiance patch-package tests (executable scratch inside the worktree because /tmp is noexec), 10/10 tooling-boundary tests, report --check, ruff --no-cache and git diff --check. No GPU jobs.

Planning MCP progress and `ag-ledger record_change_event` remain pending: neither MCP action is available in this session. This worktree note records the technical result without pretending to submit an external plan update or ledger event.

2026-11-21: footprint generator `bigcherry --engine radiance patch-footprint [--check]` and the 2.2 baseline report (docs/evidence/radiance-patch-footprint/) merged into node/radiance-latest-bump (5dedbaf53, regenerated b9072bcef): 38 patches, 679 anchors, 80 upstream files. Next slices in rank order: shared KV-row registration helper (rad0052/0053, the RUM11 lesson), rad0051 kernel hooks, rad0017 rendezvous, rad0009 after the RUM04 v2 removal. Each needs a rebase check, a component build and CPU geometry tests, then leased parity.

### 2026-11-21 shared KV-row registration slice (RUM02 slice 2)

**Current frozen 2.2 source:** `b6b8751cccb0e834a13ae9e5afca824ee9a487aa`.
`rad0049` owns a single patch-created `libr4d/bc_kv_rows.h` with a phased
declaration/row/chain inventory. Its one upstream `r4d_rows.cpp` attachment
includes this header in the existing `kKernels` registry and extends chained
eligibility. `rad0051`, `rad0052` and `rad0053` now add declarations, rows
and chain memberships to this same owned header, through their existing
explicit dependency chain; none attaches to upstream `r4d_rows.cpp` anymore.
The native E4M3 page-4 layouts are generated from a single declarative
geometry X-macro row per supported shape with query dtype, head dimension,
required GQA, page size, KV dtype and decode bound; the RUM11 page-16 fused
row keeps its intentionally unpinned local head counts/rotary width.
No kernel ABI, geometry admission policy or runtime source selection changed.

**Regenerated footprint (current patch definitions):** 38 composed patches,
`679 -> 665` non-create anchor edits (**14 fewer**, -2.1%). Upstream-owned
anchor edits `666 -> 641` (**25 fewer**); 11 attachments have moved onto the
one patch-created row header (`13 -> 24` created-target anchors), while
created-file edits `27 -> 28`. Total edits `706 -> 693`; upstream-owned
files carrying anchors remain 80. These counts describe source-definition
maintenance, not lines of binary code removed. Generated JSON/Markdown:
`docs/evidence/radiance-patch-footprint/`.

**Frozen-pin correctness:** immutable before-projection fixture
`tools/tests/patch/fixtures/rum02-kv-row-baseline.json` was captured before
the helper migration by applying the four original row-edit packages to the
exact pinned source. The regression test compares the full preprocessed
registration: **638/638 identical row initializers**, **444/444 identical
constraint declarations** and **156/156 identical chained names**; zero
lost/added names or modified bodies. Declaration indices move as the table is
collected under one hook. For every changed-order pair with the same op and
priority (285 pairs), constraints are provably mutually exclusive on a
required, non-derived field; therefore native declaration-order tie selection
cannot change. The test fails closed on unknown/overlapping constraints.
Missing/ambiguous hook anchors also fail without partial writes.

**Proof gates:** `patch-rebase-check --all` 38/38 clean, focused patch and
helper tests (13 total, 12 pass and one optional pre-RUM11 historical fixture
skipped); native C++ `-fsyntax-only` of the replayed registry and libr3 native
rewrite `--check` through the focused suite; `patch-footprint --check`,
Ruff and whitespace guards. No GPU job or full engine build. Owner follow-up:
component builds, numerical/TP3/Flash-Next selftests and production starts
on an appropriate leased GPU before concluding hardware parity. Planning MCP
and ag-ledger event are not available in this agent session; bigcherry-1
records external progress/ledger after integrating this commit.

### 2026-11-21 RUM02 slice 3: resident B16 kernel hooks (rad0051)

Completed on frozen Radiance 2.2 pin `b6b8751cccb0e834a13ae9e5afca824ee9a487aa`.
The 25 individual prefill and 29 individual decode edits on the two upstream
`libr4d/r4d_attn_*_h256_gqa6.hip` sources were replaced by **one typed
include hook per upstream file**. The complete original B16-aware template
bodies are now rad0051-owned, hash-bound `bc_b16_prefill_impl.h` and
`bc_b16_decode_impl.h` patch-created headers. Three rad0053 descriptor
edits per file now target those owned headers. The gfx11 native rewrite
retains the original O_R3NATIVE option hook and moves the eleven template
rewrites into `native/bc_b16_prefill_impl.h.rw`. No upstream vendor edits,
GPU jobs, full engine build or new adapters.

**Measured before -> after**, using the current parent commit's checked-in
footprint as the before state (not the earlier slice-1 historical baseline):

| Item | Before | After | Effect |
| --- | ---: | ---: | --- |
| rad0051 upstream prefill anchors | 25 | 1 | -24 |
| rad0051 upstream decode anchors | 29 | 1 | -28 |
| rad0051 all anchors | 96 | 44 | -52 |
| rad0051 upstream-owned anchors | 93 | 41 | -52 |
| rad0053 upstream-owned anchors | 41 | 35 | -6 (relocated to owned headers) |
| All selected patches, non-create anchors | 670 | 618 | -52 |
| All selected patches, upstream-owned anchors | 646 | 588 | -58 |
| All selected patches, edits including created files | 698 | 648 | -50 |

**Verification:** `patch-rebase-check --all` 38/38 CLEAN;
`patch-footprint` regenerated and `--check` clean; 32-package
production source composition applied clean; repeated rad0051/rad0053
application idempotent; `r3_rewrite.py --check` reports no failures on
the composed source. Original and extracted templated definitions after
rad0053 are byte-identical (prefill 33,446 bytes; decode 34,703 bytes),
including all template layout variants; preprocessed kernel token
projections matched for the target device and host passes. Both kernels
passed HIP `-fsyntax-only` before and after for gfx1100 and gfx1201
using `/opt/rocm-7.2.4/bin/hipcc` at nice 19 (8/8); the native gfx11
rewritten prefill body was byte-identical (38,956 bytes). Relevant frozen
source, registry, footprint and native-rewrite unit tests executed with
`RADIANCE_SRC` pointing to the pinned checkout and an executable worktree
`TMPDIR`. These host-only tests do not replace hardware selftests or
Flash-Next/TP3 starts; those belong to the owner GPU validation.

## Change Log

- 2026-11-18T01:28:00.655993+00:00 (created-by): Created by codex
- 2026-11-18T01:37:27.808050+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:42:49.165174+00:00 (updated-by): Updated: section:effort_risk
- 2026-11-18T01:56:30.966408+00:00 (updated-by): Updated: section:code_samples
- 2026-11-18T02:16:37.229196+00:00 (updated-by): Updated: order=4, section:steps, section:detailed_solution, section:code_samples, section:notes
- 2026-11-18T02:22:18.965936+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:25:00.590374+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:25:21.136200+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:28:04.354691+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
- 2026-11-20T12:44:25.653062+00:00 (updated-by): Updated: section:notes
- 2026-11-21T01:48:35.924734+00:00 (updated-by): Updated: section:notes
