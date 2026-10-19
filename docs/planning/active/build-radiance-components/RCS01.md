---
id: RCS01
order: 1
plan: build-radiance-components
state: pending
created-at: '2026-10-18T12:57:29.818029+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Capability map and interfaces for splitting libr3 (design gate)

## Description

libr3 is one component holding five separable capabilities, and the MoE helper is gfx1100-specific only in its kernels. Fix the target map and the interfaces before moving code, so RDNA2 (gfx1030) is a new backend and not a fork.

Target components (each with its own component.toml, settings, evidence and tests):
1. libr3 (engine kernel overrides, gfx1100): r3_compat.h, native/*.rw, rewrites.txt, r3_rewrite.py, r3_registry, r3_mxfp4_*.h, r3_selftest, r3_wmma_perf. Settings R3_ATTN_NATIVE, R3_GDN_NATIVE. Output kernels/libr3.so.
2. moe-remote (arch-neutral): moe_remote_shm.h (+test), moe_remote_hook, moe_mailbox_plugin (libr3moe.so), moe_mailbox_ops.hip, and the helper process (options, ownership/heat/balance, serve loop). Output helpers/r3_moe_helper.
3. moe-harness (diagnostic): capture, replay, replay-sweep, trace dump, plus the tools/lab scripts that drive them. Never in a production profile.
4. moe-backend-gfx1100: lane kernels (front end, small GEMV lane, large grouped WMMA lane, reduce/publish, FP16/AHALF/VECIO/FUSEROUTE/FUSESTAGE). Output helpers/moe_backend_gfx1100.so.
5. ngram-helper: r3_ngram_* (evaluated, default off; owner decision 2026-10-13: lookup work belongs on host RAM/CPU, not the R9700).

Interface: a C header moe_backend.h with a small vtable (arch id, capability bits, init, front_end, small_lane, large_lane, reduce, trace stamps). The helper picks the backend at runtime from gcnArchName and dlopens it; no match is a clear startup error. Mailbox ABI (kVersion) stays in moe-remote and is bumped per its static_assert.

## Steps

1. Confirm the five-way map against the code (who includes what; moe_remote_gpu.h and r3_moe_helper.cpp cross-references).
2. Specify moe_backend.h (entry points, ownership of device buffers and streams, what is per-arch: kernels and tile geometry; what is shared: routing, ownership, protocol).
3. Decide setting ownership per component and keep env names unchanged (they are runtime identifiers, not legacy shims); evidence mechanism ids move with their setting.
4. Decide build wiring: engine.toml [components] lists every component; each declares targets and built-against; the pin-bump components-build gate builds all.
5. Review the design with GPT (commit and push first; ask for better approaches and prior art, e.g. how other engines separate per-arch MoE backends).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Design note reviewed; every file in libr3 has exactly one owner component; moe_backend.h covers all current call sites.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Gate for RCS02-RCS08. Do not start moves until the in-flight RAD06 evidence PR merges, so evidence paths and mechanism ids do not shift mid-run.

## Change Log

- 2026-10-18T12:57:29.818029+00:00 (created-by): Created by agent

## Design decision (2026-10-19, GPT review after the Qwen3.6-35B-A3B FP8 helper attempt)

Evidence: the helper refuses the 35B model at its first expert ("cannot be laid out as w4nl64a8h"); the sizing in `r3_moe_helper.cpp::select_owned` and `moe_remote_gpu.h::expert_bytes` is W4-specific, so changing GPU kernels alone does not fix loading.

Boundary: **moe-remote** is the model-independent expert-worker runtime (shared-memory protocol and mailbox, lanes, scheduling, route mapping, expert ownership, heat and balance, request lifecycle, the worker process, the engine-side adapter). **A backend per architecture** (gfx1100 now, gfx1030 later) owns format-specific expert loading and relayout, device scratch, GEMV/GEMM and quantisation kernels, reductions and launch configuration; one backend may implement several weight formats. **moe-harness** owns synthetic experts, capture/replay, CPU references and differential benchmarks. libr3 keeps only the engine-specific RDNA3 kernel overrides.

Contract: `helper/moe_backend.h` (ABI 1): `MoeShape`, `MoeWeightDesc`, `MoeExpertDesc`, `MoeDeviceCaps`, checked `expert_bytes()` that needs no GPU allocation, and `probe()` that refuses an unsupported (arch, format, activation, scale, geometry) combination before ownership is chosen. A later step adds the C `MoeBackendV1` function table (load_expert, enqueue, synchronize, destroy) once the gfx1100 W4 kernels are adapted behind it.

Order (do not combine rename, ABI change, component move and FP8 in one PR):
1. Backend contract and format-independent sizing (header and offline test) - this PR.
2. Adapt the existing gfx1100 W4 path behind it and make `select_owned` use `expert_bytes()` instead of W4 sizing. Gate: Flash-Next replay bitwise unchanged, benchmarks within noise.
3. RCS09: shared-memory header v14 and shape from the container (checked 64-bit capacity arithmetic, no `kHidden`/`kTopK`/`kExperts`/`kMaxLayers` model constants; runtime shape arguments to kernels, template instantiations only for tile geometry and formats). Gate: Flash-Next plus a second synthetic shape.
4. FP8 block-scaled CPU reference and synthetic container (exact E4M3FN decode and 128x128 scale lookup).
5. gfx1100 FP8 path: E4M3 to FP16 decode with the block scale feeding the existing FP16 WMMA tiles, and a separate bandwidth-oriented GEMV for tiny routed batches. Benchmark A (resident FP8, fused decode+WMMA) against B (load-time FP16 expansion).
6. RCS04: relocate the components out of libr3 as a mechanical move after backend selection works.
7. Qwen3.6-35B-A3B end to end, then the gfx1030 backend (RCS07).

Replay capture v3 carries the shape, activation, weight descriptor and backend identity; the v2 reader stays for existing captures. GEMM outputs are compared within recorded tolerances, never bitwise across different numerical implementations; routing, ownership, FP8 decode and scale lookup are compared exactly.
