---
id: BRVP06
order: 0
plan: build-rocm-vulkan-provider
state: pending
created-at: '2026-11-04T22:49:46.482473+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# Explicit ROCm SDK selection for every build: named stack per lane, recorded and hashed, CI image and runtime bound to it

## Description

Owner direction 2026-11-04: we can build against any ROCm version, so the SDK must be an explicit, recorded input of every build, not whatever the host resolves. Today platform.linux-radiance (engines/radiance/recipes.toml) names no stack, so the Radiance lane takes the host default: /opt/rocm, a symlink to /opt/rocm-7.2.4. The mechanism exists (host rocm-sdks table, stack.sdk, engine.toml build.rocm-paths passing explicit compiler/HIP/HSA paths, rocm_root and rocm_version in the build metadata, RUNPATH baked to the SDK lib dir, runtime_assets checking rocm_version) but it is not required. Gaps verified 2026-11-04: (1) the lane does not name a stack; (2) the CI image pins ROCM_VERSION=7.2.4 in its own Dockerfile ARG, unconnected to the stack table; (3) it is unverified that the SDK identity feeds build_plan_id / the content-addressed build id and the CI prebuilt-layer key (ci_build_layers.input_keys uses the compiler image digest only); (4) /opt/rocm-7.14 is an empty directory on the dev host and the local config maps 7.14 to a path that does not exist, so no 7.14 comparison is possible; (5) docs/reference/build/DEPENDENCIES.md already says never to silently substitute an unversioned /opt/rocm for a requested version, nothing enforces it. Builds on BRVP01 (toolchain identity in the BuildPlan, Vulkan side) and RCS11.

## Steps

1. Inventory every path that resolves a ROCm SDK: platform/lane definitions in every engine's recipes.toml (radiance, llama.cpp, any others), tools/bigcherry/core/config.py _sdk_root and stack resolution, tools/env/rocm-env.sh, the CI Dockerfile ARG and .github/workflows/radiance-build*.yml, ci_build_layers.py, tools/lab scripts that hardcode /opt/rocm, the launchers' library paths. Record the table in this item.
2. Contract: every platform/lane that compiles HIP names a stack (sdk = a version in the host rocm-sdks table); a missing stack is a configuration error, no implicit system default for a build. rocm-system stays only when named explicitly, resolves by realpath to a versioned directory, and records that version. Migrate EVERY lane in the same change (no compatibility default).
3. Identity: the SDK identity (version file, realpath, clang --version, a content hash of the compiler and the HIP/HSA runtime library files) goes into build_plan_id and the content-addressed build id (with BRVP01), and into the CI prebuilt-layer key. Prove that two builds of one source under two SDKs get different ids and never share a cache entry, and that an in-place SDK change invalidates reuse.
4. CI: the build image's ROCm version is derived from or validated against the same stack table (a test fails if the Dockerfile ARG differs from the version of the stack the CI lane names); the layer fingerprint includes the SDK identity, not only the image digest.
5. Runtime: binaries run against the SDK recorded in their build metadata (RUNPATH already points at it); the launchers and bc-serve resolve libraries from the recorded rocm_root and refuse to start on a mismatch with the environment (no silent LD_LIBRARY_PATH override).
6. Tests (card-free): a lane without a stack fails; recorded SDK differs from the requested stack fails; ids differ across SDKs; Dockerfile ARG and stack agree; a hardcoded /opt/rocm in a build path is flagged (extend the RCS12 guard pattern list or a sibling guard with an allowlist).
7. Separately (not part of the plumbing change): acquire ROCm 7.14 per docs/reference/build/DEPENDENCIES.md into vendor/rocm/7.14 (record URL and sha256, no network side effects on the host), add it to the host rocm-sdks table, build the Radiance lane under both SDKs and A/B the gfx1100 and gfx1201 kernels with kbench_ab (compiler quality is the real question), then decide whether the default moves.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

engines/*/recipes.toml, config/recipes.toml, tools/bigcherry/core/config.py, tools/bigcherry/build/*, tools/env/rocm-env.sh, .github/docker/radiance-build/Dockerfile, .github/workflows/radiance-build*.yml, .github/scripts/ci_build_layers.py, tools/bigcherry/release/runtime_assets.py, docs/reference/build/DEPENDENCIES.md, docs/reference/ENVIRONMENT.md, tools/tests

## Validation

Card-free tests in tools/tests (lane without stack, SDK mismatch, id and cache-key separation across two named SDKs, Dockerfile ARG against the stack table); a build of the Radiance lane under the named 7.2.4 stack whose metadata, build id and CI layer key all carry the SDK identity; the full existing build, release and CI-layer tests pass; docs/reference/build/DEPENDENCIES.md and ENVIRONMENT.md updated.

## Effort & Risk



## Standards



## Acceptance Criteria

No build path resolves ROCm without naming a stack; every build records and hashes its SDK identity; CI image and runtime are bound to the same stack; two SDKs never share a build id or cache entry; no unversioned /opt/rocm in a build path outside an explicit allowlist.

## Notes

Evidence for the gaps: lab build ab39ffd2 CMake cache CMAKE_HIP_COMPILER=/opt/rocm-7.2.4/lib/llvm/bin/clang++, binary RUNPATH /opt/rocm-7.2.4/lib, metadata toolchain rocm_root=/opt/rocm-7.2.4 rocm_version=7.2.4; engines/radiance/recipes.toml platform.linux-radiance has no stack; .github/docker/radiance-build/Dockerfile ARG ROCM_VERSION=7.2.4; config/environment.local.toml rocm-sdks 7.14 = a path under the user home that does not exist here (absent), /opt/rocm-7.14 empty. Owner: bigcherry-2 (delegated 2026-11-04).

## Change Log

- 2026-11-04T22:49:46.482473+00:00 (created-by): Created by agent

## Ledger-events


- chg_20261104_224956_planned-making-the-rocm-sdk-ve_3537
- 2026-11-04T22:50:00.649519+00:00 (updated-by): Updated: section:ledger-events
- chg_20261104_230800_builds-now-always-name-the-roc_2582
- 2026-11-04T23:08:04.294463+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-04 (bigcherry-2): items 1-6 implemented in PR #611 except the bc-serve part of item 5. Open follow-up kept here: bc-serve refuses to start when the build's recorded rocm_root differs from the environment (the launcher check, lib/sdk-check.sh, is in #611). Item 7 (7.14 acquisition and A/B) is on hold until the owner confirms the download.
