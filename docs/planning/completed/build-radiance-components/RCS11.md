---
id: RCS11
order: 0
plan: build-radiance-components
state: completed
created-at: '2026-10-29T04:30:07.331128+00:00'
breadth: ''
skill: basic
created-by: agent
priority: P2
work: S
---

# Radiance builds in the agent-dev container: find the ROCm runtime without /opt/rocm

## Description

In the agent-dev container ROCm is installed as /opt/rocm-7.2.4 (and /opt/rocm-7.14) with no /opt/rocm link. Radiance's top-level CMakeLists finds the runtime with find_library(RAD_HIP_RUNTIME amdhip64 HINTS /opt/rocm/lib ENV ROCM_PATH) (and RAD_HSA_RUNTIME the same way); ROCM_PATH=/opt/rocm-7.2.4 is searched as a directory itself, not its lib/, so both are NOTFOUND and every in-container `bigcherry build --lane radiance-lab:...` fails at configure (first seen 2026-10-29, run rad32-r3-selftest-10290421, build b7f1b308.../e3f867530a95ff3d7571cdd51ea761ed). Builds made on the host work because the host's /opt/rocm is 7.2.4. Also: config/environment.local.toml [host.dev] rocm-root = "/opt/rocm" is a host path that does not exist in the container, while [host.dev.rocm-sdks] "7.2" = "/opt/rocm-7.2.4" does.

## Steps

1. Decide the fix: (a) the container image provides /opt/rocm -> /opt/rocm-7.2.4 (matches the host), or (b) the bigcherry radiance build passes the environment's ROCm SDK root to CMake (CMAKE_PREFIX_PATH or the runtime library paths), so no build depends on /opt/rocm.
2. Remove the quick fix (export CMAKE_PREFIX_PATH=/opt/rocm-7.2.4 in lab job scripts) once the chosen fix is in.
3. Verify: an in-container radiance-lab build configures and links (RAD_HIP_RUNTIME/RAD_HSA_RUNTIME resolved to the 7.2.4 libraries), and a host build is unchanged.

## Detailed Solution & Technical Design

Both options, as the lead asked (2026-11-02).

(a) The agent-dev image links /opt/rocm -> /opt/rocm-7.2.4 (audia-ci-dashboard #29). Confirmed: an in-container radiance-lab moe-remote-handoff build with no ROCm overrides configured and linked (build f0fadb596b523ebbcf8bc7c7c1919da7, with radiance finding /opt/rocm/lib/libamdhip64.so through its HINTS).

(b) The build no longer depends on /opt/rocm. engine.toml [build] rocm-paths declares each ROCm lookup radiance's CMake would run (RAD_HIPCC = bin/hipcc, CMAKE_HIP_COMPILER = lib/llvm/bin/clang++, RAD_HIP_RUNTIME = lib/libamdhip64.so, RAD_HSA_RUNTIME = lib/libhsa-runtime64.so) as a path under the SDK root. campaign/build.py resolves the selected root (rocm-root, BC_ROCM_ROOT overriding, symlinks resolved), fails closed when it is unset, missing or lacks a declared path (no /opt/rocm fallback), and passes each path as a CMake cache value. find_program/find_library skip their search when the cache value is set, so the HINTS never run. RAD_HIPCC also fixes radiance's include root (RAD_ROCM_ROOT is derived from it). The values flow through _backend_configure_options, so the BuildPlan's cmake_options and the real configure argv come from one map. They also reach the bump gate (radiance_bump) and every component's configure (runtime_bundle). Engines that declare no rocm-paths (llama.cpp) are unchanged.

Identity: before this, a radiance build (GCC host compiler, so no ROCm probe) had the same build_plan_id whichever SDK /opt/rocm pointed at, as long as PATH and ROCM_PATH were unchanged. That was a defect. Now the pinned paths are in cmake_options, and toolchain_request carries rocm_root and the root's .info/version, so two SDKs, or one SDK before and after an in-place upgrade, never share a build_plan_id. The lab's rad_build reuse key (tools/lab/radiance/lab-build.sh) now includes the SDK root and version too, and rad32-r3-selftest.sh forwards the build's pinned paths into its libr3 reconfigure.

The selected SDK is the host rocm-root because no lane selects a stack yet ([stack.*].sdk is parsed but consumed by nothing). Lane-level stack selection belongs to BRVP04 (slice 1); selected_rocm_root() is the single point it would replace.

## Code Samples & Guidance



## Files

engines/radiance/engine.toml, engines/radiance/recipes.toml, tools/bigcherry/core/engines.py, tools/bigcherry/campaign/build.py, tools/bigcherry/campaign/lane.py, tools/bigcherry/campaign/workers.py, tools/bigcherry/campaign/runtime_bundle.py, tools/bigcherry/engine_adapters/radiance_bump.py, tools/lab/radiance/lab-build.sh, tools/lab/radiance/rad32-r3-selftest.sh, docs/reference/ENVIRONMENT.md, tools/tests/campaign/test_campaign_lane.py, tools/tests/campaign/test_campaign_build.py, tools/tests/release/test_bump_gates.py

## Validation

In-container `bigcherry build --lane radiance-lab:stock:linux-radiance --experiment moe-remote-handoff` with ROCM_PATH, HIP_PATH and CMAKE_PREFIX_PATH unset succeeds with no extra environment; CMakeCache shows RAD_HIP_RUNTIME and RAD_HSA_RUNTIME under the configured ROCm SDK.

Results (2026-11-02, agent-dev container, environment rocm-root /opt/rocm-7.2.4):
- (a) before this change: build f0fadb596b523ebbcf8bc7c7c1919da7 configured and linked through the image's /opt/rocm link (RAD_HIP_RUNTIME=/opt/rocm/lib/libamdhip64.so, found by HINTS).
- (b) with this change: build 6428b65c4a9d7b1356d753c3628ce7d8 succeeded. RAD_HIPCC, CMAKE_HIP_COMPILER, RAD_HIP_RUNTIME and RAD_HSA_RUNTIME are under /opt/rocm-7.2.4 in the engine's CMakeCache and in every component's (bc-gfx12, libr2, libr3, moe-remote, ngram-helper). No CMakeCache names /opt/rocm/, and bin/radiance links /opt/rocm-7.2.4/lib/libamdhip64.so.7 and libhsa-runtime64.so.1. The new plan id is expected: the SDK is now part of the identity.
- Card-free tests: test_campaign_lane.SelectedRocmSdkTests (the selected SDK reaches the BuildPlan and the configure argv; a symlinked root resolves; two SDKs never share a build_plan_id; an in-place upgrade changes it; a missing pinned library and an unselected SDK fail before any build) and test_bump_gates (the engine and every component configure pin the four paths; a missing path fails the gate before configure).
- Full tools/tests under the CI environment (environment.example.toml, BC_ROCM_ROOT pointing at no SDK): no new failures against origin/main.

## Effort & Risk



## Standards



## Acceptance Criteria

- An in-container radiance-lab build with no ROCm overrides configures and links; CMakeCache shows RAD_HIPCC, CMAKE_HIP_COMPILER, RAD_HIP_RUNTIME and RAD_HSA_RUNTIME under the selected SDK root, in the engine and in every component.
- A selected root that is missing, or lacks a declared path, fails before configure; nothing falls back to /opt/rocm.
- Different SDKs, or an in-place upgrade at one root, give different build_plan_ids.
- Engines without rocm-paths (llama.cpp) build exactly as before.
- The CMAKE_PREFIX_PATH quick fix is removed from the lab job scripts.

## Notes

Quick fix in use since 2026-10-29 (RAD32 libr3 GEMV K-slab work): export CMAKE_PREFIX_PATH=/opt/rocm-7.2.4 in the job (<scratch>/rad32-gemv/job.sh). Also tracked in the team task list /instances/.teams/bigcherry/TASKS.md. Option (b) fits the no-machine-paths rule better; option (a) is a container change owned outside this repo.

## Change Log

- 2026-10-29T04:30:07.331128+00:00 (created-by): Created by agent
- 2026-11-02T07:16:37.701238+00:00 (state-transition): State: pending → in_progress
- 2026-11-02T07:16:57.811059+00:00 (updated-by): Updated: section:detailed_solution, section:files, section:acceptance_criteria
- 2026-11-02T07:26:00.711213+00:00 (updated-by): Updated: section:validation

## Ledger-events

- chg_20261102_072607_radiance-builds-now-compile-an_5297
- 2026-11-02T07:26:11.706389+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-02T07:28:10.842496+00:00 (state-transition): State: in_progress → completed
