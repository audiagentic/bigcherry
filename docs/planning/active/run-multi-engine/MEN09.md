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

2026-10-09 owner decision on tags (replaces the encoded-upstream-number scheme in the steps above): every engine release carries two tags on the same main commit.
- Readable tag, the one people use: bc-<engine>-<engine build>-r<N>. The engine build is upstream's own identity (llama.cpp b11474; radiance 1.3.0; for a commit pin the last upstream tag plus the short commit, e.g. 1.3.0-g89cee7c). N restarts at 0 on each pin and counts our releases at that pin. Examples: bc-llamacpp-b11474-r3, bc-llamacpp-b11490-r0, bc-radiance-1.3.0-r0, bc-radiance-1.3.0-g89cee7c-r0.
- Numeric tag, the one release-please needs: bc-<engine>-<pin counter>.<batch>.<fix>. For new engines the first number is our own pin counter (1 for the first pin, 2 for the next). llama.cpp keeps its existing line with the build number as the first number (bc-llamacpp-11474.3.0), since it is already a single monotonic integer.
- A step in the release workflow adds the readable tag after release-please publishes, reading the engine build from engines/<engine>/engine.toml at that commit, and sets the GitHub release title to the readable tag. Both tags derive from the same commit and pin file, so they cannot disagree.
- bc-platform-<semver> for the shared tooling; it has one tag.
- No engine branches: all lines live on main and the tags distinguish them. The only branch exception is a short-lived release/<engine>-<build> cut from a tag to fix an older pin after main has moved on.
- Mapping to r<N>: each numeric release at a pin gets the next N in release order, so r-numbers are dense and ordered.
Still to verify with a dry run: that the platform package can exclude engines/ so an engine-only commit does not bump the platform line.

2026-10-09 first build step done on branch feat/men09-release-lines (stacked on the engine-layout move, PR #88): release-please-config.json has two packages, engines/llamacpp (component bc-llamacpp, continuing from 11474.2.0) and '.' (component bc-platform, exclude-paths ['engines']), with separate-pull-requests and last-release-sha set to the bc-11474.2.0 commit; the changelog of the old single line moved to engines/llamacpp/CHANGELOG.md; config/release.toml tag-prefix is bc-llamacpp-; pin_release reads the engines/llamacpp package; the release workflow uses per-package outputs, adds the readable tag bc-llamacpp-b<build>-r<N> (N = releases at that build before this one) and titles the release with it. Dry run of release-please 16 against the branch: it would open two separate release PRs, 'bc-llamacpp 11474.3.0' (5 commits) and 'bc-platform 1.0.0' (22 commits). Before merging: push alias tags bc-llamacpp-11474.0.0 / .1.0 / .2.0 onto the commits of bc-11474.0.0 / .1.0 / .2.0, so the tooling's count of releases at this build and the readable r-number continue (next = r3). Not yet verified: that the platform line's changelog leaves out engine-only commits (check the first real release PR). Not in this step: the per-engine pin in engines/<engine>/engine.toml (pin is still `pinned` in config/recipes.toml), and a radiance package (added with its first release). Owner direction the same day: the RDNA3 (gfx1100) kernel library and Flash-Next on radiance are deferred; keep building the platform plans (MEN09, MEN04, MEN05, MEN03, MEN06).

## Change Log

- 2026-10-09T00:16:08.957474+00:00 (created-by): Created by agent
- 2026-10-09T00:46:01.548384+00:00 (updated-by): Updated: section:notes
- 2026-10-09T03:34:03.450100+00:00 (updated-by): Updated: section:notes
