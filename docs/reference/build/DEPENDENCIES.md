# Toolchain and dependency inventory

This is the durable inventory for BigCherry build/toolchain dependencies. It records what the repository has actually observed or configured, how to acquire unusual dependencies, and what still requires host-local verification. It is not a claim that every listed path still exists on a host today.

Use `config/recipes.toml` for source/build/platform identities, `config/environment.toml` for host facts, `tools/bigcherry/build/toolchain.py` for per-build toolchain capture, and `docs/evidence/<run-id>/` for evidence from a specific run. Do not duplicate those authorities here.

## Current source and backend identities

- `config/recipes.toml` currently pins llama.cpp at `b10705`.
- HIP is the production backend. `platform.linux-multi` enables HIP/RCCL and targets `gfx1100`, `gfx1201`, and `gfx1030`; `platform.windows-gfx1100` is the Windows HIP platform.
- Vulkan is a real configured backend, not a hypothetical: `source.vulkan-stock`, `build.vulkan-stock`, `platform.vulkan-linux`, and the `vulkan-radv` / `vulkan-amdvlk` stack identities exist. The Vulkan platform explicitly still has placeholder device identity and requires real device/ICD evidence before hardware claims.

The llama.cpp pin and ROCm version are independent identity axes. A pin does not imply a ROCm version, and the repository does not currently enforce a pin x ROCm compatibility matrix. Record the actual toolchain and source revision in the evidence for each build/run; do not infer compatibility from a successful run on another version.

## Observed/configured ROCm inventory

| Host / role | Identity | Evidence status | Use / caveat |
| --- | --- | --- | --- |
| Windows workstation | ROCm 7.1 | Observed in the 2026-08-23 TO01/RD87 investigation; `platform.windows-gfx1100` also names ROCm 7.1 compiler paths | Recheck the local install/vendor path before use; this update did not access the workstation |
| build-server (`brutus`) | `/opt/rocm` | Configured in `config/environment.toml` | System/default path; resolve its real target before treating it as a version identity |
| build-server | `/home/audumla/rocm-shim` | Configured in `config/environment.toml` | Compiler-name compatibility shim; not a distinct ROCm release |
| build-server | `vendor/rocm/7.2.4` | Directly observed in TO01 on 2026-09-02 | Complete library tree used as the 7.2.4 reference |
| build-server | `vendor/rocm/7.14` | Directly observed in TO01 on 2026-09-02 | Keep version-qualified; do not replace with an unversioned ad-hoc path |
| build-server | `vendor/rocm/7.2.4-merged` | Created and verified in TO01 on 2026-09-02 | Combines the compiler-layout convenience of `artifacts/va15-rocm-merged` with the complete 7.2.4 libraries, including RCCL |
| build-server | `vendor/rocm/10.0` | Installed and observed in TO01 on 2026-09-02 | ROCm 10.0.0 / RCCL 2.30.4 comparison tree; installer changed system `update-alternatives`, which was reverted |

`vendor/` is intentionally gitignored. Inventory entries therefore describe observed host state, not files guaranteed by a checkout. On a host, use the existing environment helper first:

```bash
tools/rocm-env.sh --list
```

For Brutus, also verify the shared defaults before a version-sensitive run:

```bash
readlink -f /usr/bin/hipcc
readlink -f /usr/bin/amdclang
readlink -f /opt/rocm
ls -1 vendor/rocm/
```

Do not silently substitute an unversioned `/opt/rocm` for a requested version.

## Acquisition rules

### Normal ROCm toolchains

Keep versioned SDKs under `vendor/rocm/<version>/` when a host-local vendored copy is required. Preserve a complete compiler/header/library layout; partial copies are not interchangeable with a complete SDK. Prefer configured/version-qualified paths in repeatable work.

The ROCm 10.0 runfile installation exposed a shared-host hazard: even with a custom target directory, post-install steps can change system-wide `update-alternatives` entries. Before and after any such install, capture the resolved shared compiler/tool paths and restore only changes introduced by that install. Never assume `target=` confines all side effects.

`rccl-tests` uses `ROCM_PATH` to select its ROCm tree. Supplying only similarly named variables can fall through to the host default and invalidate a version comparison.

### RD87: `hipblaslt-bench`

The prior BUILD note that the hipBLASLt client was simply "not present" is stale. RD87 built and executed a working client on Brutus. The reproducible source is the ROCm `rocm-libraries` monorepo, not a standalone hipBLASLt clone for the tested tree.

Observed rebuild recipe from RD87:

1. Sparse-clone `ROCm/rocm-libraries` under the ignored `vendor/rocm-libraries` tree, including `projects/hipblaslt` and required `shared/` siblings (`origami`, `rocroller`, `tensile`, `mxdatagenerator`, `stinkytofu`, `ctest`, `primbench`).
2. Required system packages observed during the build: `gfortran`, `libboost-dev`, `libmsgpack-cxx-dev`.
3. Provide GTest/GMock; RD87 used a userspace extraction of `libgtest-dev` and `libgmock-dev` and pointed `CMAKE_PREFIX_PATH` at the extracted `usr/` tree.
4. Use the monorepo's `invoke` build interface, not the deprecated `install.sh` wrapper. Keep the invoke venv outside the generated `build/` directory.
5. Install Tensile's Python requirements into that venv and put its `bin/` first on `PATH` so CMake resolves the intended Python interpreter.
6. From `projects/hipblaslt`, build clients for the required architecture with the `invoke build --clients --architecture ... --cpu-ref-lib lapack` interface.
7. At runtime, ensure the matching ROCm `libomp.so` is on `LD_LIBRARY_PATH`; RD87 needed the version-qualified 7.2.4 LLVM library path.

RD87's 2026-08-23 plan record contains the exact commands and the successful real-hardware smoke result. This document records the acquisition path only; it does not promote that historical benchmark into fresh evidence for another run.

## Vulkan prerequisite

Vulkan is already wired through the standard source/build/platform configuration, so TO01's old "is Vulkan real or hypothetical?" question is resolved. What remains external to this inventory is hardware qualification: `platform.vulkan-linux` still says its device/ICD identity is TBD. Before relying on Vulkan results, capture the actual device, driver/ICD, source revision, and command in `docs/evidence/<run-id>/` according to `docs/evidence/README.md`.

## Freshness rule

Treat an inventory row as one of:

- **configured**: declared by tracked config, but not proof the path exists now;
- **observed**: directly recorded by a prior host investigation, with its observation date;
- **verified for a run**: captured in a specific evidence bundle.

Only the last category proves the environment used by a result. Re-run host checks when a claim depends on current availability or version identity.
