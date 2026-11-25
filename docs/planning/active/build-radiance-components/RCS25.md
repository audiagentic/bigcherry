---
id: RCS25
order: 0
plan: build-radiance-components
state: pending
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

## Notes

Design: docs/design/radiance-kernel-composition.md section 4a. No tamper or digest machinery: git commit ids and plain comparisons only.

## Change Log

- 2026-11-25T02:12:11.908255+00:00 (created-by): Created by lead
- 2026-11-25T02:12:35.487166+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
