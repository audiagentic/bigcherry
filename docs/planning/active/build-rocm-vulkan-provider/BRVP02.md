---
id: BRVP02
order: 0
plan: build-rocm-vulkan-provider
state: pending
created-at: '2026-09-26T00:48:51.745817+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P3
work: M
---

# ROCm toolchain images pinned by digest (builds first, measurement only after A/A qualification)

## Description

Multi-ROCm is a standard build dimension, currently served by hand-installed prefixes (/opt/rocm-7.2.4, vendor/rocm/10.1.0rc2 + -campaign wrapper, ~/rocm-7.14-multiarch). Replace with official ROCm container images pinned by digest so each job's toolchain is an exact reproducible identity and adding a version is an image pull. Runs under the Slurm-based job service (OCI job steps or Podman), never containerizing Slurm itself.

## Steps

1. Select images (rocm/dev-ubuntu-24.04:<ver>) per supported ROCm version; record digests in tracked config by logical toolchain name.
2. Build path in container: bind-mount worktrees, builds, ccache, TMPDIR; pass /dev/kfd and /dev/dri/renderD*; CCACHE_COMPILERCHECK=content.
3. Toolchain identity = image digest (+ hipcc version) in the attempt manifest.
4. Measurement stays on host binaries until an A/A qualification (host vs container execution of identical binaries, paired, pre-declared +-0.10% equivalence bound) passes.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Byte-identical build outputs host-prefix vs image for one pinned version (or documented differences); A/A qualification record before any container-measured evidence.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner question 2026-09-26 'should we package this in docker': decision = not for Slurm/scheduler; optional for toolchains later. Depends on the job-orchestrator Slurm decision (docs/design/JOBS_ORCHESTRATOR.md).

## Change Log

- 2026-09-26T00:48:51.745817+00:00 (created-by): Created by agent
