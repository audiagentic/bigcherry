---
id: RAD04
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-14T04:55:19.264169+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# 27B prefill: overlap the cross-card sum with the next GEMM (two half-steps, second queue)

## Description

Measured 2026-10-14 (27B mxfp4, two XTX, 7.3K prompt, --profile-ops, rank 0): ar_rmsnorm_quant_fp8 (fused cross-card sum + norm + quant) 27.8% of prefill; MXFP4 GEMMs ~54% at ~102 TFLOP/s (gate_up N=17408 K=5120: 1.59 ms per ~913-token call) vs 142 measured peak. libr4d marks every ar-family kernel non-concurrent (r4d_rows.cpp rad_kernel_concurrent: stream-ordered scratch/counters), so the sum cannot overlap anything today.

Plan: split a prefill step into two halves A/B; issue A's all-reduce on the engine's second queue while B's GEMMs run; give each half its own all-reduce scratch/counters (libr4d AR variant) so two can be in flight. Engine patch (step scheduling) + libr3 AR variant. Target: hide most of the ~28%, which with a GEMM uplift to ~125 TFLOP/s reaches ~3,000 tok/s vs one R9700 3,104-3,214.

Validation: KL vs exact reference (must be exact or within wire tolerance), ab-benchmark balanced A/B on 27B prefill, both ranks clean shutdown.

## Steps



## Detailed Solution & Technical Design

Engine structure (pinned radiance, arch/qwen35_fp8/qwen35_fp8.cpp step()): a flat per-layer loop -- attn.step or gdn.step, then mlp.step -- each block issuing the whole step's rows. The cross-card sum is fused into the NEXT block's norm (rad_fp8.h ar_rmsnorm_quant_fp8, declared with reads {src,res} writes {q,s,x,res}); libr4d marks every ar-family kernel non-concurrent, so it serialises against everything.

Two-half pipeline: split a prefill step's rows into A=[0,T/2) and B=[T/2,T) and, per layer, issue A.mix, A.mlp, B.mix, B.mlp. Ordering is valid: B's attention reads A's KV for that layer (written by A.mix), B's delta-net continues A's final recurrent state, and each half's next-layer input is its own previous output. A's fused sum (in the next block's norm) goes to the engine's second queue with rad_event_record/rad_event_wait so B's mix/GEMMs on the main queue run under it; B's sum likewise overlaps A's next layer.

Work: (1) row-range step entry points for the attention, gated delta-net and MLP blocks (r0, rows) incl. rope/KV/state offsets; (2) a per-half ar_rmsnorm_quant_fp8 instance with its own scratch/counters so two can be in flight (libr3 variant; ar kernels stay non-concurrent within a queue); (3) step() orchestration for prefill steps above a row threshold, decode unchanged; (4) engine patch package rad0004 + libr3 kernel, default off behind a component setting during qualification.

## Code Samples & Guidance



## Files



## Validation

KL vs exact reference must be exact (same arithmetic, reordered issue); ab-benchmark balanced A/B on 27B prefill (pp at 2K/7K/16K) on two XTX; clean shutdown both ranks; decode unaffected (tg identical).

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

2026-10-14 source survey (radiance 1.3.0, 89cee7c): the runtime already has the second lane this item needs -- Ctx::set_lane / lane_join (abi rad_lane, rad_lane_join), rad_buf_concurrent in the builder, and ctx.h documents the rules: every collective on lane 1 (ar seq/double-buffer parity), device-wide counters keyed per stream (split-K split_ctr already per stream in libr4d), concurrent buffers named via rad_buf_concurrent, and slices must stay in the same bucket band (a 4096 chunk halves to 2048+2048 = same band; no more than two slices). The comment budgets ~15% prefill from overlapping the AR. No arch uses the lane yet (grep: only abi/core). Remaining work is the arch side: the blocks (rad_block_attn_gated_fp8 / gdn_fp8 / mlp_fp8) take a whole RadBatch and fixed brows(b_x, n_tok) from row 0, so a half step needs either a row-offset sub-batch path in each block or per-half residual buffers; B's attention reads A's KV and B's GDN continues A's state, so per-layer order A.l then B.l is valid. Also check libr3 (gfx1100) ar kernels and any libr3 device-wide counters against rule TWO before putting anything on lane 1. GPT request req_d2210f41d8004623 was orphaned 14 h with no output and was abandoned; branch feat/rad04-two-half holds only the seed commit.

## Change Log

- 2026-10-14T04:55:19.264169+00:00 (created-by): Created by agent
- 2026-10-14T05:01:27.920591+00:00 (updated-by): Updated: section:detailed_solution, section:validation
- 2026-10-14T08:41:26.958185+00:00 (updated-by): Updated: section:notes
