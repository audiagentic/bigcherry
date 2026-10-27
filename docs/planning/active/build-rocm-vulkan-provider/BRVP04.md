---
id: BRVP04
order: 0
plan: build-rocm-vulkan-provider
state: pending
created-at: '2026-10-26T00:27:55.399283+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P1
---

# ROCm SDK stack as a real build dimension: one resolved SDK drives build, identity, runtime and verification

## Description

Owner 2026-10-26: resolve ROCm selection properly in the build system, no hacks or workarounds. Today [stack.*] (sdk -> host rocm-sdks -> root, compilers, runtime_library_dirs, environment) is parsed in core/config.py (BackendStack) but consumed by nothing; builds take ROCm from whatever PATH/ROCM_PATH the caller exported, and radiance's CMake finds hipcc with HINTS /opt/rocm/bin ahead of $ROCM_PATH and takes HIP headers from it. A ROCm 10.1 build therefore compiled with 10.1's clang against /opt/rocm (7.2.4) headers and failed (__ocml_* undeclared). Scripts carry ad-hoc 'export ROCM_PATH=/opt/rocm-7.2.4' lines.

## Steps

1. Lane identity names a stack (source:build:platform[:stack]); each platform declares its default stack; stack resolves through the host rocm-sdks table to one SDK root (fail closed when missing).
2. The build derives its entire toolchain from that root: ROCM_PATH/HIP_PATH/PATH for subprocesses, C/C++/HIP compilers, CMAKE_PREFIX_PATH, and engine-declared CMake hooks for the SDK (radiance: its hipcc/include discovery) - never from the caller's shell or /opt/rocm.
3. SDK identity (version from <root>/.info/version + realpath) is part of build_plan_id (with BRVP01), so builds for different SDKs get separate build dirs.
4. Build metadata records sdk version/root/runtime_library_dirs; every launcher (ServerRunner, ab-benchmark sidecars/helpers, launch-flashnext-best.sh, run_bench) sets the runtime library path from the build's metadata so a binary runs against the ROCm it was built with.
5. Post-build verification: compile_commands include dirs and the linked libamdhip64/libhsa-runtime64 resolve inside the stack root; mismatch fails the build.
6. Start-up environment check (owner rule): warn when the host's ROCm/kernel/firmware differ from the validated set.
7. Remove every ad-hoc ROCM_PATH export/workaround in tools/lab and docs (no legacy paths).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

tools/bigcherry/core/config.py (BackendStack), core/environment.py, campaign/build.py, campaign/lane.py, build/builds.py, tuning/server_runner.py, engine_adapters/radiance.py, engines/radiance/recipes.toml, config/recipes.toml, tools/lab/radiance/*.sh, docs/reference/build/BUILD.md

## Validation

Unit tests: lane parsing with stack, plan id differs by SDK, toolchain env derived only from the stack (shell ROCM_PATH ignored), verification rejects a /opt/rocm include in a 10.1 build. Hardware: radiance-lab lane builds with stack rocm-7-2 and rocm-10 side by side; both pass verification; each runs (helper + engine) against its own runtime; then the ROCm 10.1 vs 7.2.4 A/B on Flash-Next.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Related: BRVP01 (build identity for toolchains) - this item supplies the HIP/ROCm half of the compile identity. Triggered by RAD20's ROCm 10.1 comparison.

GPT design review 2026-10-26 (req_bigcherry_66654d823011): APPROVE WITH CHANGES. Invariant: ResolvedStack -> BuildPlan -> configure env -> CMake cache -> artifact -> runtime env -> verification; one SDK, no implicit fallback, no mixed installs. Changes: (1) stack is a first-class BuildPlan field: platform default-stack, lane override (named field internally; ':stack' only CLI syntax); fail closed. (2) remove ambient discovery: resolve_build_environment fingerprinting PATH/ROCM_PATH/HIP_PATH/LD_LIBRARY_PATH, _find_rocm_prefix compiler-ancestry, compiler-inferred CMAKE_PREFIX_PATH, 'ROCm from PATH' platform; missing probes must fail, not pass. (3) engine SDK hooks declarative in engine recipes with typed placeholders from a fixed allowlist ({sdk.hipcc}, {sdk.hip_include}, ...), no engine branches in core; where upstream CMake ignores cache vars (radiance find_program HINTS /opt/rocm), fix by a maintained engine patch making SDK inputs authoritative. Host GCC stays separate from the HIP toolchain. (4) runtime: RUNPATH-first (absolute SDK paths for fixed-host builds, $ORIGIN for relocatable), artifact-owned runtime manifest, verify full dependency closure incl. dlopen'd GPU libs; no inherited LD_LIBRARY_PATH; kernel/driver interfaces exempt. (5) identity: BRVP01 owns build_plan_id; BRVP04 supplies SDK identity = requested+detected version, canonical root, compiler realpath+version, hip-config identity, runtime library manifest, device library identity, manifest digest; reuse needs identity match + artifact integrity. Host kernel/firmware NOT in build id (execution qualification). (6) qualification records separate: config/qualification/rocm/<id>.toml (SDK digest, GPU archs/IDs, kernel identity, amdgpu version, firmware hashes, topology constraints, evidence id), probed generically; outcomes validated/unvalidated/incompatible (dev warns on unvalidated, incompatible fails closed). PR slices: 1 stack schema+resolution+CLI; 2 SDK-derived toolchain + declarative hooks (radiance and llama.cpp build with selected SDK); 3 BRVP01 identity; 4 RUNPATH + runtime manifest; 5 all launchers via one shared runtime resolver (ServerRunner, ab-benchmark sidecars, engine adapters, run_bench, launch scripts); 6 build/runtime dependency verification; 7 qualification records + probes; 8 remove ad-hoc exports/obsolete paths. Tests listed in the review (two SDKs side by side, mixed-SDK rejection, in-place SDK replacement, hostile env, unknown topology, Vulkan unaffected).

## Change Log

- 2026-10-26T00:27:55.399283+00:00 (created-by): Created by agent
- 2026-10-26T00:50:19.211059+00:00 (updated-by): Updated: section:notes
