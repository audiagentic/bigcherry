# Toolchest build publication

BigCherry is the build authority for formal llama.cpp comparisons. BigCherry materializes the source, applies the exact source/patch composition, fingerprints the toolchain/environment, builds into a content-addressed BuildPlan directory, verifies effective CMake configuration and runtime artifacts, and publishes an immutable runtime bundle. Toolchest registers and runs that verified output; it does not rebuild or copy it.

## Enable live publication

Set the Toolchest base URL in the environment used to run BigCherry:

    export BIGCHERRY_TOOLCHEST_URL=http://127.0.0.1:3000

If the Toolchest API is later protected by a bearer token, optionally set:

    export BIGCHERRY_TOOLCHEST_API_KEY=...

Then use the normal canonical build command. For example:

    bigcherry build --profile standard

When `BIGCHERRY_TOOLCHEST_URL` is set, `bigcherry build` changes the historical default primary target from `bin/llama-bench` to `bin/llama-server` and adds `llama-bench` as an extra CMake target. Both executables are produced by the same configure/build identity and published into the same verified runtime bundle.

A successful lane is registered immediately with Toolchest through:

    POST /api/builds?external=1

No Toolchest restart is required. A registration failure makes the requested build command return non-zero; the immutable BigCherry artifact remains valid and can be registered again later.

## What is registered

The Toolchest build ID is:

    <source>-<build>-<platform>-<full-build-plan-id>

Examples naturally distinguish an upstream control from a BigCherry lane because the source identity is part of the ID:

    llama-native-stock-linux-multi-...
    bigcherry-native-control-linux-multi-...
    bigcherry-replay-linux-multi-...

The registration records:

- Toolchest profile: BigCherry `hip` maps to `rocm`; `vulkan` remains `vulkan`.
- Binary path: the immutable ArtifactStore `llama-server` copy, not the mutable CMake build directory.
- Git ref: exact resolved upstream llama.cpp revision.
- Git SHA: exact BigCherry producer revision from artifact provenance.
- Tag: short effective-build and runtime-bundle identities (`eb-...-rb-...`).
- Replace: true, making publication idempotent for the same content-addressed BuildPlan ID.

The full `build_plan_id` stays in the Toolchest ID. The BigCherry artifact provenance remains the authoritative detailed provenance and includes source slice, patch composition, effective build, runtime bundle, inputs, toolchain and workload lineage.

## Formal A/B rule

Do not use a Toolchest-native compiled build as one side of a formal BigCherry performance comparison. Build both upstream/native and patched candidates through BigCherry so the compiler, ROCm/Vulkan stack, CMake options, requested targets and build-environment identity are controlled by the same pipeline.

`campaign.standard` already contains a `llama-native:stock:linux-multi` lane as well as BigCherry lanes. This allows stock upstream and BigCherry candidates to be compared without changing build systems.

Toolchest's native builder remains useful for convenience and exploratory testing; it is not the formal baseline authority.

## Filesystem requirement

Toolchest launches the registered absolute ArtifactStore path. BigCherry and Toolchest therefore need the same filesystem view of the BigCherry work root.

If both run on the host, no extra setup is needed. If Toolchest runs in a container, bind-mount the BigCherry work/artifact-store path at the same absolute path inside the container. Keeping the path identical avoids a second path-remapping authority and preserves the exact registered path.

## Ownership and deletion

BigCherry owns build artifacts and their lifecycle. Toolchest owns only the registry entry, launch state, benchmark configuration and benchmark results.

Deleting an external build in Toolchest unregisters it but does not remove the BigCherry ArtifactStore binary/runtime bundle.

## Benchmark flow

1. Build upstream and/or BigCherry lanes through `bigcherry build` with `BIGCHERRY_TOOLCHEST_URL` set.
2. Successful builds appear immediately in Toolchest.
3. Select those Build IDs in one Toolchest benchmark job.
4. Run the existing `models x builds x presets x sweeps` matrix, including tensor split, GPU assignment, KV type, batch/ubatch and `draft-mtp` settings.
5. Use Toolchest's stored build snapshots and BigCherry's immutable provenance to interpret and reproduce results.
