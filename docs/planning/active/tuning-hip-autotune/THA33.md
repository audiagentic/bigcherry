---
id: THA33
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-12T08:11:34.403381+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: null
---

# Intermittent build-sensitive SIGSEGV in production mul_mat_vec_f_cuda on gfx1201

## Description

Found while running THA02's schema-2 hardware validation on gfx1201 (dev device 2, not the patches' target arch). Real, reproducible-per-build-artifact SIGSEGV in production mul_mat_vec_f_cuda -> ggml_cuda_mul_mat_vec_f -> ggml_cuda_graph_evaluate_and_capture (libggml-hip.so.0), confirmed unrelated to any BigCherry patch (crash stack never touches patched code). Two builds from byte-identical materialized source produced genuinely different binaries (confirmed via sha256sum, including stripped .text section hashes) with different crash behavior: one crashes 100% reproducibly (isolated single-GPU, no topology confounds), the other never crashes. No kernel-level GPU fault/reset events (journalctl -k clean), no ccache indirection, identical CMakeCache.txt. Root cause is either compiler/codegen nondeterminism exposing a latent UB/race bug in the production MMVF kernel path on gfx1201, or some other build-artifact difference not yet identified.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

On the exact gfx1201 host when investigation is authorized, capture immutable provenance for each build: materialized source/submodule hashes, compiler and ROCm versions, build flags, CMakeCache, environment, cache state and complete build commands. Produce at least three clean isolated builds from byte-identical inputs; record whole-binary and stripped .text hashes plus GPU_DUMP_CODE_OBJECT=1 output for mul_mat_vec_f_cuda. For each distinct artifact, run the isolated HIP_VISIBLE_DEVICES=2 reproducer across repeated fresh processes and retain success/fault counts, symbolized backtrace, process/runtime logs and kernel-fault/reset logs. Compare device code objects first; if identical, investigate host-side runtime/loader inputs. Confirm whether the stack touches any patched code. Keep gfx1100 as a separate unaffected control boundary; no inference from a different architecture.

## Effort & Risk



## Standards



## Acceptance Criteria

THA33 remains pending until either (a) the artifact-dependent gfx1201 failure mechanism is identified and a source/build/runtime correction is verified across repeatable clean builds and isolated reruns, or (b) a reproducible fail-closed gfx1201 regression gate with exact artifact/toolchain identity and preserved crash evidence is accepted by the technical owner. Different behavior between hashes must be explained or explicitly retained as unresolved; a single non-crashing build is not a pass. Record exact runtime/build/device provenance and symbolized fault evidence, prove the BigCherry patch path is or is not implicated, and keep gfx1100 production scope separate. Do not claim resolution from compile success, a clean source hash, or unrelated architecture results.

## Notes

Filed 2026-09-12 during THA02's real hardware pass. GPT-reviewed across 3 deep-analysis rounds (req_f2d1d0cf12234313, req_6a56ad440ffa482f, req_5c5b77a770a44a75). Evidence trail: bisection ruled out each of 1236/1238/1239/1240 individually; full-chain rebuild from identical source hash did not reproduce; isolated single-GPU (HIP_VISIBLE_DEVICES=2) A/B/A ruled out multi-GPU/driver contamination; gdb backtrace on the archived crashing binary shows the fault entirely within production mul_mat_vec_f_cuda, never touching test-harness code; stripped .text section hashes of libggml-hip.so.0 genuinely differ between the two builds. Next steps if picked up: compare gfx1201 device code objects directly (GPU_DUMP_CODE_OBJECT=1) between the two builds to determine if AMDGPU codegen for the kernel itself differs; if identical, investigate host-side runtime/loader differences. Out of scope for gfx1100 (dev's actual production topology), which is unaffected.

Review RV4251 added explicit future validation and acceptance gates for the existing build-sensitive production SIGSEGV evidence. No hardware run or root-cause conclusion was made in this planning cleanup; the item remains pending.

## Change Log

- 2026-09-12T08:11:34.403381+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260912_081232_confirmed-the-moeglu-correctn_6156
- 2026-09-12T08:12:32.130115+00:00 (updated-by): Updated: section:ledger-events

## Reviews

- RV4251
- 2026-11-20T03:20:42.219523+00:00 (updated-by): Updated: section:validation, section:acceptance_criteria, section:notes
