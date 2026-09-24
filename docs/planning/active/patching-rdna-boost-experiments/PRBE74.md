---
id: PRBE74
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:58:43.924628+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Q4_K MMVQ wide-load / x4 activation / VDR candidate family (AMD #84 + #88 as one experiment)

## Description

TODO. No existing patch implements the AMD #84/#88 Q4_K MMVQ wide-load/x4-activation/VDR candidate family (grep of patches/*/patch.toml for RD98 finds nothing). At b11126, `vec_dot_q4_K_q8_1` (ggml/src/ggml-cuda/vecdotq.cuh, verified present) and its dispatch via `case GGML_TYPE_Q4_K:` in mmvq.cu (multiple arms, verified present) use `VDR_Q4_K_Q8_1_MMVQ = 2` (vecdotq.cuh, verified). The candidate family adds x4-wide-load/VDR=8 variants as gated, causally-isolated knobs, following the same structure/precedent as the rejected-but-instructive patches/1204_rd08_q6k_mmvq_vdr2 package (VDR-tuning precedent for a different quant).

## Steps

1. Read the full body of `vec_dot_q4_K_q8_1` (vecdotq.cuh, starts ~line 918) and the mmvq.cu kernel loop that calls it via VDR-based unrolling (grep `VDR_Q4_K_Q8_1_MMVQ` usage sites in mmvq.cu) to identify the exact load/unroll structure before writing any Edit anchor.
2. Design one gated env var (e.g. GGML_CUDA_MMVQ_Q4K_VARIANT=native|x4_layout|unroll|barrier|vdr8|best_nwarps) read once via the same static-lambda getenv pattern used elsewhere (see patch 1203/1215 precedent), default 'native' (= current VDR=2 behavior, zero overhead when unset).
3. Implement each candidate as a separately gated code path within `vec_dot_q4_K_q8_1` / its mmvq.cu call site, each with an explicit `can_execute`-style legality assertion (e.g. x4_layout requires the block_q8_1 producer to ALSO be x4-aware -- assert this at compile time via a paired producer/consumer flag, never silently mixing a wide-load producer with the plain consumer) and native fallback if the legality check fails.
4. VDR=8-ownership variant: redefine the per-thread element ownership stride consistent with `VDR_Q4_K_Q8_1_MMVQ` doubled to 8, requiring matching changes to whatever unroll/loop bound in mmvq.cu currently hardcodes assumptions from VDR=2 (identify these from step 1's read).
5. Correctness: repeated fixed-seed direct MUL_MAT_VEC test-backend-ops cases for Q4_K at long K (matching the existing HI70-style direct-op corpus precedent used by patch 1203/1204), run on gfx1201 (primary) and gfx1100 (holdout), plus an unaffected-quant control (Q8_0, unaffected by this Q4_K-only change) proving no cross-quant regression, each variant run multiple times with the same seed to catch nondeterminism from the new load pattern.
6. Only after correctness passes for a variant: kernel time and end-to-end decode t/s, reported per variant separately from native, on gfx1201 primary/gfx1100 holdout.

## Detailed Solution & Technical Design

This must never expose a wide (x4) load producer to a plain (VDR=2-shaped) consumer -- that is the item's own explicit correctness hazard. The cleanest structure is a single patch with each variant behind its own Edit id, all sharing one selector env var, default off/native, so co-tenant patches touching the same file (if any -- check via `grep -l 'vecdotq.cuh\|mmvq.cu' patches/*/patch.py`) are not silently broken by a variant that isn't even selected.

## Code Samples & Guidance

Confirmed real anchors at b11126: `vecdotq.cuh:504: #define VDR_Q4_K_Q8_1_MMVQ 2` and `vecdotq.cuh:918: static __device__ __forceinline__ float vec_dot_q4_K_q8_1(...)`; mmvq.cu dispatch arms at lines 21, 53, 82, 167, 205, 233, 251, 274, 338, 360 (all `case GGML_TYPE_Q4_K:`, confirmed via grep). Exact before/after Edit text is NOT given here since the full function body (beyond the signature) was not read this session -- the plan's step 1 is a hard prerequisite before any patch.py is written; do not skip it or invent anchor text.
Patch package sketch: `patches/12xx_rd98_q4k_mmvq_wideload_family/patch.toml` (backend="hip", state="untested", experiment-contracts=["RD98-Q4K-MMVQ-WIDELOAD-FAMILY"], external-source pointing at AMD PR #84/#88 if those are the same 'AMD ecosystem' external-source already used by patch 1215 -- check external-sources.toml for an existing entry before adding a new one).

## Files

ggml/src/ggml-cuda/vecdotq.cuh; ggml/src/ggml-cuda/mmvq.cu; tests/test-backend-ops.cpp (new direct-op long-K correctness cases); patches/12xx_rd98_q4k_mmvq_wideload_family/{patch.toml,patch.py,SUMMARY.md}; config/external-sources.toml (if a new source entry is needed).

## Validation

PYTHONPATH=tools python -m bigcherry patch-lint; patch-rebase-check --focal-overlay rd98_q4k_mmvq_wideload_family --source bigcherry-tuning. Repeated fixed-seed MUL_MAT_VEC(Q4_K) test-backend-ops correctness at long K, same-seed repeats, on gfx1201 primary/gfx1100 holdout, Q8_0 unaffected-quant control -- required to pass BEFORE any performance claim. Hardware performance (not run here) via python -m bigcherry.patch.validation_campaign: kernel time + decode t/s per variant vs native.

## Effort & Risk

M; kernel-level correctness risk is real (the item's own acceptance criteria explicitly warn against x4-producer/plain-consumer mismatches) -- step 1's full read of the current code is non-negotiable before implementation.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Promote only candidates with strict direct-op parity, matching x4 producer/consumer, legal narrow eligibility, and repeatable kernel/E2E evidence; native remains default otherwise.

## Notes

Supersedes: RD98
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd98

2026-09-24 relevance at b11126: TODO confirmed, no existing patch (grep patches/*/patch.toml for RD98 finds nothing), target functions/constants verified present (vec_dot_q4_K_q8_1 at vecdotq.cuh:918, VDR_Q4_K_Q8_1_MMVQ=2 at vecdotq.cuh:504, Q4_K dispatch arms in mmvq.cu). GPT design request: gateway rejected all submissions this session (VAL-AGW-025 / EXT-GPTAUTO-003, seen in agent_task_gateway_overview); plan authored directly from verified anchors -- no GPT request id. Function BODY (beyond signature/constants) was not read this session -- flagged as a hard step-1 prerequisite rather than guessed.

## Change Log

- 2026-09-09T10:58:43.924628+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:54.994935+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.462729+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.305180+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:20:34.632180+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032047_repaired-five-more-active-succ_6361
- 2026-09-10T03:20:47.972545+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:37:26.479126+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
