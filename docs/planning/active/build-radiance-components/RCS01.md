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
