---
proposed-id: LTX01
group: build-radiance-ltx25-tp3
proposed-state: pending
engineering-status: acceptance-evidenced-on-main
priority: P0
work: M
skill: advanced
owner: Planning MCP registration required
---

# Qualify source-backed IQ8R TP3 declarations and finish the foundation

**Planning status:** Candidate for registration through Planning MCP. This file is NOT an authoritative item. Proposed owning group: `build-radiance-ltx25-tp3` (check uniqueness first). Do not hand-edit `docs/planning/` or state files. Register with the same sections, priority and dependencies when the service becomes available.

**Baseline:** Radiance 2.2.0 / ABI44; BigCherry patch `rad0060_ltx2_tp3_linear_declare` is unselected and the production `ltx2` builder rejects TP3. Shared native TP3, AQL stream dispatch and exact collectives already exist. Reuse them. The reference model is the licensed local LTX-2.5 IQ8R RAD2 checkpoint. Never substitute synthetic-only numeric checks for real-model or live GPU proof.

**Working files:** `engines/radiance/patches/rad0060_ltx2_tp3_linear_declare/`, `tools/lab/radiance/ltx-tp3/`, `tools/tests/lab/test_ltx_*.py`, `docs/design/ltx25-iq8r-tp3.md`. Use isolated task worktrees from current `origin/main`, coordinate with PR #1262, no redundant patch or forked scheduler.

**General gates:** patch applicability/idempotence against the pinned source, exact ABI/versioned kernel/build identity, CMake/CTest of the real source-backed native test (not just synthetic fixtures), focused pytest and tooling guards, no GPU process without a short `agent-lease` lease; reject rather than fallback if an unqualified world/geometry is selected. Keep results and raw receipts in plan-owned work-root directories and concise stable evidence in `docs/evidence/`.


## Description

Finish only the diagnostic LTX-specific model-source foundation; the generic AQL lab cleanup is RUM13-owned. The C01-C06 declarations already reuse Radiance sharding; the C08 AQL result is the correct native collective proof. Consolidate model-derived uneven slices with strictly legal IQ8R K offsets, preserving the existing production TP1 behavior.

**Existing engineering gap coverage:** LT3-01, LT3-03, LT3-04, LT3-07. **Dependencies:** Radiance 2.2 ABI44; candidate PR #1262; none of the downstream LTX plans.

## Shared native qualification dependency (RUM13)

The model-independent mixed-architecture AQL test and its direct-HIP
duplicate are now **owned exclusively by proposed RUM13**, not LTX01.
See [RUM13](../rum13-native-qualification/README.md). RUM13 removes
the duplicate launch and build path, retains native source/ABI
validation, and publishes one reusable mixed-card exact-collective
receipt. LTX01 consumes this evidence; it must not change shared
AQL dispatcher, `rad0017`, engine device admission, group lifecycle
or shared test tool. RUM05 and RUM06 retain those shared runtime
implementation responsibilities.

## Steps

- 1. Reconcile the latest `main` and PR #1262; inspect the complete `rad0060` patch, all fragment `src/*.inc` files and native reference/test paths. Confirm one code path for geometry in the actual LTX builder and document any dead duplicate declarations.
- 2. Verify every one of the 48 DiT blocks and 960 model IQ8R projection names; assert `RadFile` code/U8-group/BF16-row scale planes, fused QKV/KV bank offsets and bias sources are exact. Use native `LinearFP8::span_parts` and `rad_weight_shard_span_parts`; do not add a second slicing helper.
- 3. Verify 8/8/16 and 12/12/8 whole-head partitions for video H128 and audio H64, and reject unsupported audio 10/10/12 group-256 K columns before allocation. Check zero-width, alternate rank order and duplicate-head refusal.
- 4. Keep `ltx2` TP1 behavior byte/graph-identical. In `rad0060` remove stale comments and reconcile README/SUMMARY/catalog, source hashes and the exact pinned patch inventory. Keep the code unselected until LTX02–05 complete.
- 5. Run all synthetic, model-backed native C01/C02/C04/C05/C06 CTests, the five focused pytest modules and the repository docs/catalog/patch/forbidden-term gates; record actual checks, source pin and skips.

## Detailed Solution & Technical Design

One existing `C01Slice` (rank-local shape, bank offsets and aligned K ownership) should supply all declarations. Do not create another local metadata/loader or an extra per-attention rank table. Reuse current `RadBuilder` and native `Geom`. The C04 bias defer remains a flag on `Lin`, not a new postprocessor service.

## Code Samples & Guidance

```text
main + pinned Radiance 2.2/ABI44 -> existing rad0060 + native CTest
-> narrow model-backed issue -> AQL three-rank numerical proof
-> safe owner/conditioning acceptance -> measured release decision
```

Reuse the exact current source-managed extension and existing tests; no side-path implementation of graph, device collective, kernel discovery, model parsing or config. All GPU jobs require an acquired lease, real build/source identity and exclusive card ownership. The first failing native/assertion test is required before editing a production path.

## Files

- `engines/radiance/patches/rad0060_ltx2_tp3_linear_declare/patch.py`
- `engines/radiance/patches/rad0060_ltx2_tp3_linear_declare/src/{linear,bias,preflight,attn_geometry}.inc`
- `tools/lab/radiance/ltx-tp3/native-c01-gate/`
- `tools/tests/lab/test_ltx_{tp3_builder_patch,iq8r_numeric,iq8r_planes,iq8r_shards}.py`
- `docs/reference/engines/radiance/ENHANCEMENTS.md; README.md`

## Validation

Model-backed C01/C02/C04/C05/C06 native CTest; 11 synthetic CTests; full 48-block real model plane and code/scale/row checks; patch replay and changed-file/catalog guards. Real checkpoint path is read-only and tests with no model must explicitly skip rather than claim pass.

## Effort & Risk

**P0, M.** M: host declarations can appear green while emitted device code fails. Separate schema/BF16-GEMM/device execution assertions.

## Standards

BigCherry `docs/reference/agents/WORKFLOW.md`, patch and tooling dispositions, stable exact native ABI, source/weight identity, tests before production code, measured proof before speedup claims. No host paths, hand-picked model constants, new config/parser, shims or duplicate path. Planning/ledger state is MCP-owned.

## Acceptance Criteria

RUM13's single-AQL-harness cleanup and exact GPU evidence are external shared capabilities consumed by LTX01, not implementation work in this item. Measure deletion only of redundant **LTX-specific** metadata/geometry code. All supported slices have unique full head coverage and exact model source views. At least the previously demonstrated 3,024 row/biasless, 2,640 deferred, 2,016 reduction and 576 attention declarations are rerun at the candidate commit, with zero mismatches and no invalid audio K cuts admitted. No TP3 production activation, duplicate loader or unowned test script. No release claim from these tests.

## Notes

**LTX01 engineering closure (2026-11-22):** the entire original
source/geometry/declaration acceptance set has been rerun against the
exact Radiance 2.2/ABI44 frozen source and the real IQ8R checkpoint at
the merged LTX02 baseline. The 22 anchored source edits apply
idempotently and TP1 remains unchanged/TP3 unselected. Native
declaration receipts: **3024** fused/biasless, **2640**
deferred-bias, **2016** exact output-reduction, **1728**
global Q/K graph cases and **576** video/audio self-attention
graph declarations. **384** misaligned audio K partition
attempts fail closed. The real RAD2 inventory checks all
**960 DiT IQ8R projection entries across 48 blocks**, all
**1306** IQ8R tensor plane structures, and 1920 valid
raw-plane rank roundtrips; no separate model parser/metadata
authority is added. A single `C01Slice` header remains
authoritative for LTX-specific shard geometry; no redundant
LTX geometry implementation warranted deletion.

New C01 fixture negatives explicitly test zero-width, duplicate
head ownership, gap and reversed rank enumeration. LTX02's
already-merged native AQL Q/K evidence is a **dependency**,
not an LTX01 implementation change. **11/11 synthetic CTests**,
**6/6 source-backed native CTests**, **12/12 real-source
and real-model pytest tests**. [Full qualification and
source fingerprint](../../evidence/2026-11-22-ltx25-ltx01-foundation-closure/README.md).
This completes the **engineering acceptance** of LTX01;
the Planning MCP item remains **unregistered**, not falsely
marked closed in generated `docs/planning/`. Register and close
it against the same evidence when the authoritative service
is connected. LTX02–LTX08 are **not** completed by this.

C01-C06 source-backed host proof exists, C06 real GPU kernels were separately tested, but fresh current-commit revalidation is required. RUM13 owns retiring the C07 direct-HIP path; its crash is not production Radiance failure. C08 native AQL mixed-device proof passed. Full device/weight issue is LTX02, not a LTX01 deliverable.

References: `docs/design/ltx25-iq8r-tp3.md`, `docs/design/ltx25-tp3-component-test-ladder.md`, `docs/evidence/2026-11-22-ltx25-c08-aql-mixed-collective/README.md`, PR #1262. This item is a **registration candidate** until Planning MCP creates it.
