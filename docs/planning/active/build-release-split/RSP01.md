---
id: RSP01
order: 0
plan: build-release-split
state: pending
created-at: '2026-11-16T06:37:56.515652+00:00'
breadth: ''
skill: advanced
created-by: claude
work: L
priority: P1
---

# Release only what runs the model: separate run-side from build-side tooling

## Description

Owner direction (2026-11-16): the BigCherry release must not carry build or worktree tooling; that is build work, not LLM model execution work.

Today the release is wider than the runtime. The radiance runtime bundle (bin, radiance_home, runtime-prefix) is already run-side only, but the release tag is the whole repository: `bc-platform` is component `.`, and `release/install.py` installs by creating a detached git worktree of the tag. The same tag therefore carries the build, patch, campaign, source, slice and worktree tooling, and `release/ci_bundle.py` BUILD_INPUTS folds that tooling into the release identity (bundle reuse key).

Goal: a release contains and versions only what is needed to run and serve a model; build and development tooling is not part of it and does not change its identity.


## Steps

1. Inventory: classify every module under `tools/bigcherry/` and every top-level path as run-side (serve, runtime, launchers, runtime-assets restore, serving config, engines' runtime data) or build-side (build, campaign, source, patch, slice, worktree, jobs, release tooling, lab). Record the table in this item; flag shared modules.
2. Decide the boundary mechanically (a directory/package split or a manifest of run-side paths), not by convention; one test fails when a run-side module imports a build-side one.
3. Change release-please components so `bc-platform` no longer releases `.`; release only the run-side paths (and keep bc-radiance/bc-llamacpp as the engine components).
4. Change `release/install.py` to install the verified bundle plus serving config and launcher without a git worktree of the tag.
5. Re-scope `ci_bundle.BUILD_INPUTS` so the bundle reuse key covers what shapes the binary (engine patches, recipes, build image) and nothing that is dev tooling or release ceremony; keep the safe side (a rebuild is allowed, a stale bundle is not).
6. Migrate callers in the same change (no compatibility shims).


## Detailed Solution & Technical Design

Open design points, to decide before step 3:
- Where build-side tooling lives afterwards (a separate package in this repository that is not a release component, or a separate repository).
- How a serving host reproduces a release: bundle + config pinned by tag, restored without git.
- Which shared modules (core/, engine_adapters/) the runtime really needs, and whether they split.
- Radiance tag naming and the release record (`releases/*.json`) are kept; only their contents' scope changes.


## Code Samples & Guidance



## Files

release-please-config.json; .release-please-manifest.json; tools/bigcherry/release/install.py; tools/bigcherry/release/ci_bundle.py; .github/workflows/radiance-release-bundle.yml; tools/bigcherry/ package layout; docs/reference describing releases.


## Validation

- A CI test asserts the run-side manifest imports nothing from build-side modules.
- Install a release on a clean host path with no git checkout of the repository and launch flashnext-4gpu.
- Bundle reuse key is unchanged by edits to slice/worktree/patch-authoring tooling and changes when an engine patch or the build image changes.
- Release notes list only run-side changes.


## Effort & Risk

Large blast radius: release-please wiring, install path and CI bundle reuse all change; shared modules may need splitting. Risk: breaking installs from older tags; keep the previous install path for already published tags only if named as an explicit exception, otherwise migrate.


## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-11-16T06:37:56.515652+00:00 (created-by): Created by claude
