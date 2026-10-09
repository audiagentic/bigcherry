---
id: MEN09
order: 3
plan: run-multi-engine
state: pending
created-at: '2026-10-09T00:16:08.957474+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P0
work: M
---

# Per-engine pins and release lines (own numbering per engine, platform released separately)

## Description

Owner direction 2026-10-09: BigCherry becomes a test platform for several engines; each engine is pinned individually and needs its own release numbering. Today there is one release-please component (`bc`, package path `.`) whose version is `<llama.cpp build>.<minor>.<patch>` (current tag bc-11474.2.0), one `pinned = "b11474"` in config/recipes.toml and one releases/pin-transition.json. A radiance release would have to share that number line, and a radiance pin move would look like a llama.cpp release.

## Steps

1. Three kinds of release line, each its own release-please component with its own version file, changelog and tag prefix:
   - `bc-llamacpp`: continues the existing line, version `<upstream build>.<minor>.<patch>` (next: bc-llamacpp-11474.3.0). Existing bc-* tags stay as history; no re-tagging.
   - `bc-radiance`: version `<encoded upstream>.<minor>.<patch>`, where upstream 1.3.0 encodes as 10300 (major*10000 + minor*100 + patch), so the first is bc-radiance-10300.0.0. Minor counts promotion batches at that pin, patch counts fixes, exactly as for llama.cpp. If radiance is pinned to an untagged commit, the release notes carry the commit and the encoded number stays at the last upstream tag.
   - `bc-platform`: plain semver for what is shared (patch engine, lab queue, CI, slices, planning), so a tooling change does not bump any engine's line.
2. release-please attributes a commit to a component by path, so each engine's files live under one root: engines/<engine>/ (patches, overlay, engine config, release records, lab topics). Everything else is the platform package. This is the layout MEN08 must produce.
3. Pin per engine: engines/<engine>/engine.toml holds upstream URL, pin (tag or commit), vendor location and the transition marker. pin-status / pin-bump / repin take --engine; a bump of one engine changes only that engine's root and so only that engine's release line.
4. A pin move starts a new major on that engine's line (11474 -> 11490, 10300 -> 10400); promotions at a pin bump the minor; fixes bump the patch.
5. Commit convention: the conventional-commit scope names the engine where it matters (`feat(llamacpp): ...`, `fix(radiance): ...`); CI's title check accepts the engine scopes.
6. Ledger and evidence records carry the engine so release notes are generated per line.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

release-please-config.json, .release-please-manifest.json, releases/bc-version.txt, docs/releases/CHANGELOG.md, config/recipes.toml, releases/pin-transition.json, tools/bigcherry/release/pin_status.py, tools/bigcherry/release/pin_bump.py, .github/workflows/

## Validation

A dry run of release-please on a branch with one commit under each root proposes three separate release PRs with the expected next versions; pin-status lists each engine with its own pin and state.

## Effort & Risk



## Standards



## Acceptance Criteria

A promotion on radiance and a pin bump on llama.cpp can be released on the same day without either changing the other's version or changelog.

## Notes

No radiance release should be cut before this lands: with the current single component it would take the next number on the llama.cpp line. Depends on the directory layout in MEN08; the release configuration and the layout are best done as one change.

## Change Log

- 2026-10-09T00:16:08.957474+00:00 (created-by): Created by agent
