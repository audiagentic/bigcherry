---
id: RAD36
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-29T05:25:15.204887+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# libr3 MXFP4 GEMV at 2-8 rows: close the drafter verify gap on gfx11 (27B dflash2 decode)

## Description

Qwen3.8-27B MXFP4 on two XTX (tp2) with radiance's dflash2 drafter (depth 7, verify step M=8) decodes at about 58-91 tok/s (typically ~80) on radiance 1.13.0 + libr3 (PR #451 at f6ce657d4); radiance 1.3.0 reached 103-109 tok/s at the same context (runs p4-tp2spec, 2026-10-10). Acceptance is not the problem (~33-36% of 7 drafts, ~3.4 tokens a step); the step cost is: a drafter step ~48 ms vs ~22 ms for an M=1 step (profiles rad32-drafter-prof-10290408 / rad32-nodraft-prof-10290409), almost all in the target's gemm_nt_q / gemm_nt_q_gated_mx at M=8. --perfmxd on the libr3 WMMA GEMV (r4d_gemm_mxfp4a8_gemv rewrite) at M=8 vs M=1 after #451's fixes: plain 198-242 vs ~400 GB/s, gated 242-288 vs 520-555 GB/s. The upstream split-K decode row runs M=8 at ~490 GB/s, but upstream keeps M<=8 on the gemv row for its K-only sum order (a row's bits independent of M), so routing 2-8 rows to it is not an option. (ID RAD36: RAD32-RAD35 are reserved by the radiance 1.13.0 bump work.)

## Steps

1. GPT design review of the current kernel (consult text saved at /work/development/tmp/bigcherry/rad32-gemv/gpt-consult.md; the gateway rejected it with RES-EXP-004 on 2026-10-29: execution profile surface gpt-auto-cdp not validated).
2. Find what still costs at M=8: every block re-stages and re-widens the whole M x K activation (two column tiles a block halve it; four spill), the per-row scale fmaf, the sP cross-wave reduction at 8 live rows. Measure each with --perfmxd at the 27B shapes.
3. Candidate designs keeping a K-only sum order: a bf16 activation pre-pass shared by all blocks (one widen, not one per block); activation held in VGPRs across column tiles; a TN=4 form that does not spill (fewer live weight fragments).
4. A/B against f6ce657d4: --perfmxd M=1/8, 27B drafter decode (2.3K/6.5K, exact and wht6), 27B no-drafter decode/prefill, selftest rows byte-identical.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

27B dflash2 decode at 2.3K/6.5K context >= 100 tok/s on two XTX; --perfmxd M=8 within ~20% of M=1 bandwidth; libr3 selftest 0 failures with the gemv rows byte-identical; 27B no-drafter decode and prefill not worse than PR #451's gate (prefill 2,580-2,792, decode 46.1-46.2).

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

PR #451 already carries: K-slab staging (any K at M=1: K=25600 207 -> 417-553 GB/s), 32-byte row pad, TN=2 at 2-16 rows incl. the one-row-tile launch. TN=4 from three rows was tried and reverted (spills: 86-107 GB/s at M=8, drafter 41-60 tok/s). Runs and logs: /work/development/tmp/bigcherry/rad32-gemv/ (base.txt, after-*.txt, drafter-*.log).

GPT review 2026-10-29 (reviewer-gpt-agent, req_main_7ee5e132f3b2): conditional approval of the #451 changes, no defect
found; the one check it asked for (WMMA forms only at K % 128 == 0) holds. Its order for this item: (1) resource
baseline first -- VGPRs, spills, waves/CU, LDS and kernel time at M=1/2/4/8 x TN=1/2, real DRAM bytes (perfmxd GB/s is
logical bytes over time), and rank the whole drafter step's kernels; (2) a bf16 activation pre-pass shared by all blocks
(same conversion, scale/FMA and chunk order; scratch buffer; dispatched only where it wins); (3) register-lean pipelined
TN=2; (4) a TN=4 that keeps one s fragment live, only if the resource data says it can avoid spills. No split-K (it
changes the sum order). Full text: /work/development/tmp/bigcherry/rad32-gemv/gpt-review-req_main_7ee5e132f3b2.md.

## Change Log

- 2026-10-29T05:25:15.204887+00:00 (created-by): Created by agent
