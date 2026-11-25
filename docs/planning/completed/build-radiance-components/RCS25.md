---
id: RCS25
order: 0
plan: build-radiance-components
state: completed
created-at: '2026-11-25T02:12:11.908255+00:00'
breadth: ''
skill: intermediate
created-by: lead
priority: P1
work: M
---

# Discrete kernel test harness: build one unit or family, hash and replay its rows on one card without the full plugin or engine

## Description

Today a kernel change is proved by building the whole plugin (about 7,800 records) and replaying with the engine's own runner. A small reusable harness lets one unit or family be built, compared and timed on one card in minutes, with no engine, no model and no full composition. The pieces exist as scratch tools (single-unit plugin builds with a loader that tolerates absent rows, a cross-plugin output-hash selftest, a cold-flush replay tool); this makes them one repository tool.

## Steps

1. One command that compiles a named unit or family (the RCS20 family targets name them) for a chosen architecture into a small test plugin, for libr3 (gfx11, through the rewritten copy) and libr4d (gfx12). 2. A test loader that accepts the missing rows instead of refusing, and runs on one chosen card. 3. Resource comparison: the compiled record of every symbol in the test plugin against the shipped release, listing changed, added and removed. 4. Output hash comparison: hash the selected rows' outputs against the shipped library over the shapes and row counts the unit serves (tails and ragged N included), reporting any difference. 5. Replay with the adaptive rule (windows of launches until two consecutive medians agree within their spreads, cold flush), changed rows only, reporting wins, regressions and neutral. 6. Documentation and a short fixture (a known-good and a known-bad unit). Reuse r3_selftest and r4d_selftest where they already provide a check; add only what is missing.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

tools/lab/radiance (harness command), engines/radiance/components/libr3 (test target), docs

## Validation

On the merged fused IQR tile: the harness reproduces the recorded resource table, 491 hashes and replay ratios in minutes, without the full plugin or the engine.

## Effort & Risk

Medium; mostly assembling existing scratch tools. Risk: a test plugin whose module is too large for the device loader (the earlier multi-unit module hit an out-of-memory launch error): build per unit or per small family.

## Standards



## Acceptance Criteria

A kernel change can be built, hashed against the shipped build, and replayed from one command in minutes; the full plugin build, Gate E and serving runs are needed only for a release.

Native provider vs overlay fixture independently verifies real supported forms, logical output equality and refused forms with actual layouts, records selected row and compile resource record; target row missing or unwritten outputs fail. Bounded replay and no disguised reference fallback. RCS25 discrete tests before full release, do not require full plugin per kernel.

## Notes

Design: docs/design/radiance-kernel-composition.md section 4a. No tamper or digest machinery: git commit ids and plain comparisons only.

2026-11-25 MEN24 bump integration: this is the sole discrete kernel harness owner. RCS19 supplies the actual composed input and affected-unit list; RCS20 supplies family targets; RCS23 uses this harness for overlay resource, byte-output and replay qualification before advancing integrated revision. Compare exact git revisions, plugin name/version/upstream release and actual rows; no additional hash/identity machinery apart from existing output hashes. Run one card under queued lease, unknown dependency scope widens checks; isolated harness tolerates missing unrelated rows only, never suppresses required target-row failures. Full integrated lane, installed Gate E and MEN17 serving/parity checks remain release gates. See docs/design/radiance-bump-integration.md.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. Reuse one discrete harness for native libr3d, generated libr3 and owned overlay comparisons where supported, exact source/build/target. Native packed IQR private raw layout differs from local; build actual layout/relayout from same logical source or canonical inputs for each arm, never reuse incompatible packed bytes. Test omitted target row, wrong provider/layout, altered scale, untouched output sentinel, OOM/timeout/wrong card. Missing unrelated rows test-only; never production fallback. Timing adaptive windows bounded, inconclusive if not converged; preserve bit-identity for existing arithmetic changes. Full plugin and engine not prerequisite.

2026-11-25 lean build sweep at 79c61a758. One existing CMake/Ninja build graph, one canonical matched release lane, existing tests and release records. No new cache, build system, manifest, hash scheme or generic framework. Reuse actual evidence; only test changed behavior and dependency closure. BU03 #1571 cold-build records now baseline for five plane-prefill IQR forms (shipped resources differed, measured replay and output neutral). Use existing cold exact-source reference and recorded output baseline; no full cold plugin build per discrete test, compile only affected unit with matched flags and actual layouts. Distinguish real performance/resource regression from baseline difference or compiler drift. No duplicate GPU run if same row/shape/source proof already recorded. Existing full plugin/engine gate at release stays.

2026-11-25 implementation: production libr3 component explicitly resets R3_ONLY empty; partial test builds remain isolated. No changes to unit harness under #1565, no duplicate harness. Full production row/link and target-row proof remain with actual builds.

2026-11-25 closed: docs/evidence/2026-11-25-rcs20-rcs25-family-harness/README.md. The harness (merged #1565, Release build type #1581) builds the IQR family for gfx1100 in 155 s (225 s with a cold selftest cache) against the shipped 2020.13.0, hashes 555 outputs (0 differ), 0 selftest failures, compares 3,521 resource records and replays 52 rows, without the full plugin or the engine. Known difference: 5 iqr_prefill_kernel forms (236 to 240 VGPR, 2 to 3 spills) between a cold build of the source and the shipped library, the same in two runs, output hashes equal; carried by the BU03 follow-up. Replay rows near 1.0 flip between regress and same across runs, so hashes and resource records are the stable checks.

## Change Log

- 2026-11-25T02:12:11.908255+00:00 (created-by): Created by lead
- 2026-11-25T02:12:35.487166+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:effort_risk, section:acceptance_criteria, section:notes

## Reviews

- RV4625
- 2026-11-25T02:18:13.454037+00:00 (updated-by): Updated: section:notes
- RV4636
- 2026-11-25T02:42:19.951361+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
- 2026-11-25T05:21:34.496954+00:00 (updated-by): Updated: section:notes
- 2026-11-25T05:32:38.631650+00:00 (updated-by): Updated: section:notes
- 2026-11-25T08:45:02.912213+00:00 (updated-by): Updated: section:notes
- 2026-11-25T08:45:21.158894+00:00 (state-transition): State: pending → completed
