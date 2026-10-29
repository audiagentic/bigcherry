---
id: RCS11
order: 0
plan: build-radiance-components
state: pending
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



## Code Samples & Guidance



## Files



## Validation

In-container `bigcherry build --lane radiance-lab:stock:linux-radiance --experiment moe-remote-handoff` succeeds with no extra environment; CMakeCache shows RAD_HIP_RUNTIME and RAD_HSA_RUNTIME under the configured ROCm SDK.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Quick fix in use since 2026-10-29 (RAD32 libr3 GEMV K-slab work): export CMAKE_PREFIX_PATH=/opt/rocm-7.2.4 in the job (/work/development/tmp/bigcherry/rad32-gemv/job.sh). Also tracked in the team task list /instances/.teams/bigcherry/TASKS.md. Option (b) fits the no-machine-paths rule better; option (a) is a container change owned outside this repo.

## Change Log

- 2026-10-29T04:30:07.331128+00:00 (created-by): Created by agent
