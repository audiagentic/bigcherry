---
id: RCS27
order: 0
plan: build-radiance-components
state: pending
created-at: '2026-11-26T01:36:30.551307+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Model-driven discrete qualification selection and evidence gates

## Description

Extend RCS25 unit_harness.py and unit_families.toml. Resolve actual build groups to narrower qualification families and owned rows, support dependencies and gaps. Explicit arch and matched reference. Add read-only plan, fail closed on missing hashes, reference failures, missing rows, skipped/unconverged replay and resource-only PASS ambiguity. Extend uncovered families with real oracle and workload data.

## Steps

1. Add offline group-to-family selection in existing harness; report incomplete inventory and coverage. 2. Map overlays via origins/targets and actual row contracts. 3. Gate exact reference and candidate evidence, missing/zero/skipped/unconverged cases. 4. Validate selected family compile and matched leased measurements; add missing oracles incrementally.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

See docs/design/radiance-kernel-composition.md; extend existing owners.

## Validation

Synthetic groups, malformed/partial inventory, missing coverage, zero hashes, stale/failed reference, skipped and unconverged replay. Actual selected build, then discrete GPU tests under bounded lease.

## Effort & Risk



## Standards



## Acceptance Criteria

Existing harness and outputs only; every required unit covered or explicitly nonpass; compile-only is diagnostic, no automatic promotion.

### Replacement proof (2026-11-27)

[Enhancement parity guide](../../../design/radiance-enhancement-parity.md) specifies exact patch groups, code seams and current named harnesses. Native incomplete by source -> port, no redundant GPU comparison. A prior qualified contract, B same-target native focal delta absent, C retained/reconciled delta (only when needed); no incompatible ABI dlopen, independent oracle not B==C alone. Small executable tests only where text/compile cannot prove behavior. Preserve existing required form coverage, use targeted replay and no full engine test for isolated code. Explicit capability parity is independent of current model use.

### Small tests first (2026-11-27)

Use [fast-gate review](../../../evidence/2026-11-27-radiance-230-bump/FAST-GATES.md). Existing unit harness is IQR-centric and gfx1100/gfx1201 only; update stale support-unit name r4d_gemm_fp8a8 -> r4d_gemm_i8a8 for target when needed, add only actual missing small oracles. Use existing libr2/worker/collective tests for other roles; do not universalize. --unit is resource-only; --model-graph declared, not execution proof. Exact same-target compatible plugin references; never load old ABI in new harness. Narrow build and test to affected shapes/forms with boundary/control, expand only for changed arithmetic/layout, unknown coverage, failure or noisy replay; preserve required hash/oracle and fail-closed rules. No engine build to validate an isolated kernel.

### 2.3.0 compatibility closure (2026-11-27)

Use unit_harness.py --model-graph with actual baseline and candidate saved /graph, unit_families.toml, compose_check and origins; join selected row, layout, operand geometry, rank/architecture and execution evidence, including helper-worker and speculative-draft paths not represented by base graph alone. Missing M64/gated IQR, group256 IQ8R, U6/U7 run256/512, mixed-bit head, U2 draft head, U4 I8-table drafter, Flash-Next w4nl64a8i8h and protected BF16 experts, E4M3 HC/MTP/ngram and packed cache rows must be explicitly covered where used. Provider substitution must pass real model layout and both graph selection and measured issue; native libr3d lacks chained/collective paths and does not cover all required forms. Test refusal as well as success, use existing byte/resource/replay gates; a row name or metadata match is not proof. No retirements on missing/unresolved graph coverage. MEN17 owns integrated generation and quality; no duplicate harness.

## Notes

Detailed design and boundaries: docs/design/radiance-kernel-group-qualification.md. No active related item edited.

2026-11-26: Extended existing unit harness with exact rows and saved Radiance /graph model-band selection; provider, geometry, rank and gaps retained. Require leased physical BDF and actual selftest device identity, artifact digests and observed controls; strict duplicate/missing/skipped hash and per-round timing/layout checks, cold/warm replay. 280 focused tests passed, one skipped. Real configured inventory and explicit IQR row plan resolved embed/tiled, M64/gated/unknown row remain uncovered. Graph importer fixture-tested; no new live graph or revised physical replay due to active GPU cohort. Remaining native oracles and matching reference/leased build validation.


2026-11-28 on the 2.3.0 pin (bc-radiance-2030.0.0): the harness's selftest did not build (libr3's selftest rewrite still added --pin-mw/--pin-nt, whose plugin hooks upstream removed in 2.3.0; release gates do not build the selftest). Retired those rewrite blocks and re-anchored the three that chained on them; all libr3 rules apply. All nine existing IQR/embed/vocab selftest edit anchors still match 2.3.0 once. Added family iq8r (unit r4d_gemm_i8a8 with r4d_quant_act_fp8, r4d_fused_quant, r4d_model_bf16, r4d_keyed; case gemm_iq8r_tiled), the Qwen 27B IQ8R trunk rows, with two selftest edits: extra 27B trunk shapes from UH_HASH_CASES (16-token row to 64 rows, tiled above; host fold skipped for extra shapes, the hash against the shipped library is their oracle) and an output hash per shape. Result on gfx1100: 580 kernels resource-unchanged, 75 shapes hashed and equal, selftest failures 0/0, checks_passed. Replay finding: against the shipped library the slim plugin timed -14% to +65% on byte-identical tiled code (same compiler), so release-vs-harness replay is not a code comparison. The build now digests each family kernel's instructions and replays only when the family's code differs; identical code needs no replay. Still uncovered in gemm_quant: the int8 head (r4d_logits_gemm_i8), MXFP4/w4a8 Flash-Next rows and the other 18 units; next is the head row.


2026-11-28 model-driven coverage: saved the qwen27b-tp3 /graph from bc-radiance-2030.0.0 (rank 0 of 3, 111 declared rows). Families now name every row their case checks (`rows`), so iq8r covers both GEMMs, the quantiser and the three fused producers. Added family head_i8 (r4d_logits_gemm_i8 + r4d_rowtopk_bf16; rows logits_gemm_i8, logit_rerank_i8), hashed at the three 248320-token vocabulary shards (82816/82688 x 5120) and decode/draft/verify row counts: 166 kernels resource-unchanged, 14 shapes hash-equal, selftest 0/0, checks_passed on gfx1100. A family without a timing command passes only on identical kernel code; changed code is inconclusive until a replay exists (--perfvocab needs the bc-iqr rows too). Remaining qwen27b rank-0 device rows without a harness family: attention (h256 gqa6 / h128 gqa4 fp8kv decode and prefill, kv_store_fp8, qk_norm_rope_gate_kv_store), GDN, the 3-rank all-reduce, DFlash, embed/gather/scatter, rope, rmsnorm/silu/add, cast and sampling. Per this item's notes those keep their existing selftest/collective tests rather than harness families unless a kernel change needs one. FlashNext /graph captured on 2030.1.0 for its rows next.


2026-11-28 Flash-Next: saved the flashnext-4gpu engine /graph on bc-radiance-2030.1.0 (gfx1201, world 1, 77 device rows). Its int8 trunk runs gemm_i8a8_nt_m16 (M <= 64) and gemm_i8a8_tiled (above) at group 128 on six (N, K) shapes. Added family i8a8 (r4d_gemm_i8a8 with the quantiser and keyed blocks; case quant_act_i8g, which runs every int8 trunk check) with those shapes and an output hash per GEMM; replay through --perfw8. gfx1201 on bc-radiance-2030.2.0: 218 kernels unchanged, 71 shapes hash-equal, selftest 0/0, checks_passed. Remaining Flash-Next engine rows without a family: the w4nl64a8i8h expert GEMM (moe_gemm_w4a8i8), bc-gfx12 HC rows, MTP/ngram/QSA rows, attention, GDN, router and sampling.


2026-11-28 Flash-Next experts: added family moe_w4a8i8 (r4d_moe; case moe_gemm_w4a8i8) with Flash-Next's expert geometry from its /graph (gate/up 1280 x 2560 in two parts, down 2560 x 640 over sorted rows, w4nl64a8i8h, top 10, 16 experts to keep the host decode affordable) at 8, 64 and 600 tokens. Families may now pass their own case format (`cases`) and name skips that are the case's contract (`expected_skips`; here the 640-expert shapes' 'no run map past 512 experts'); any other skip still fails. gfx1201 on bc-radiance-2030.2.0: 197 kernels unchanged, 399 outputs hash-equal (including the selftest's 4096/8192-token down passes), selftest 0/0, checks_passed. A SIGSEGV seen while writing the family was the injected case's own bug (routed rows passed as tokens at top 10); buffered selftest output made it look like upstream's large pass, which passes in every environment tried.


2026-11-28 bc-iqr M64/gated rows (qwen27b selects bc_gemm_iqr_nt_m64 and bc_gemm_iqr_gated): tried running the iqr_gemm overlay through the selftest's broader `gemm_iqr` case on gfx1100 against a harness-built base IQR reference (bc-radiance-2030.4.0 tree). Not shippable as is: (1) the gated check needs gated_had_quant_i8g, so r4d_fused_quant must be in both the base reference and the overlay's base stage (every gated case reported rows absent); (2) `gemm_iqr` runs every upstream IQR form (about 1000 hashed outputs on the reference), while the owned rows serve only the production forms, so the overlay hashed 21 and the rest read as missing. Tiled replay over the run was level (0.996 to 1.007), and no compared hash differed. Next: a selection restricted to the owned forms (V4 trellis-mad B=6/7) for the M64 and gated checks, plus r4d_fused_quant in the overlay base; the code change was not merged.


2026-11-29 gap (3), the draft_u2 family: gemm_w2a8_nt (the DFlash2/MTP draft head) had only the selftest's execution-only gemm_case. The harness now injects a numeric case:
- U2 codes, an f16 scale and a u8 zero per 128 of K (the IQ6R container's drafter head encoding), stored through the row's own layout hook;
- an exact int8 activation written in quant_act_i8's fragment order, with its f32 row scale and [K/128][Mpad] group sums;
- a double-precision oracle over sampled columns, and a hash of every output;
- the row's refusals: no group sums, N not a multiple of 16, and a plain bf16 weight at the layout hook.
Shapes: the case's own 512 x 1024 at M 1/4/16, plus the 27B drafter head's TP3 shards of the 248,320-token vocabulary (82816 and 82688 x 5120) at M 1, 2, 3, 4 and 16.
Results against bc-radiance-2030.5.0, on gfx1201 (R9700) and gfx1100 (XTX, libr3's port through the int8 WMMA emulation):
- 48 kernels resource-unchanged, 10 outputs hash-equal, selftest failures 0/0, checks_passed;
- rel_l2 1.5e-3 to 1.9e-3 against the oracle (bf16 rounding);
- negative control: the same oracle without the zero points misses by 0.70 to 1.24, so it tells a wrong decode apart;
- the gfx1100 and gfx1201 output hashes are identical. This settles libr3's note that the emulated w2a8 byte layout was assumed.
No replay: the family's kernel code is identical to the release, and there is no timing command for the head yet.

2026-11-29 gap 1, IQR 512-row runs (Qwen 27B IQR-V4 ffn_gate_up) in the unit harness:
- forms: the iqr case gains a run length (256 by default, existing hash names unchanged) and the 6/7-bit 512-row trellis forms; the gated row gains 6 and 7 bits at M 1/4/16, n 6656, K 5120 at 512-row runs; UH_IQR_RUN sets the run of the --perfiqr planes.
- U6/U7: BF16-scale storage is covered by the case's existing bf16s forms.
- family iqr, gfx1201 against bc-radiance-2030.5.0: 655 hashes equal (100 at run=512), selftest failures 0/0.
- family iqr, gfx1100: the release's gfx1100 base library has no IQR rows (bc-iqr owns them), so there is no release reference. The candidate has 0 selftest failures and all 655 output hashes equal gfx1201's, including the 100 at run=512.
- overlay iqr_gemm_m64, gfx1100 (bc-iqr of main with #1780, against that family build): 480 hashes equal, 18 gated, 6 of them at run=512, byte-equal to the unfused chain; selftest failures 0/0; checks_passed.
- run length is a runtime value in bc-iqr (same kernel instantiations). Cold --perfiqr 13312x5120, XTX, stock, 256-row against 512-row runs, ABBA: gated and plain times within 0.8 us at B 6/7, M 1/4/16 (gated 6-bit: 67.1, 76.8, 79.9 us).

## Change Log

- 2026-11-26T01:36:30.551307+00:00 (created-by): Created by agent
- 2026-11-26T01:37:56.167096+00:00 (updated-by): Updated: section:steps, section:notes
- 2026-11-26T05:07:03.703862+00:00 (updated-by): Updated: section:notes
- 2026-11-28T06:21:30.682022+00:00 (updated-by): Updated: section:notes
- 2026-11-28T07:25:59.101118+00:00 (updated-by): Updated: section:notes
- 2026-11-28T08:33:47.490967+00:00 (updated-by): Updated: section:notes
- 2026-11-28T09:24:04.635107+00:00 (updated-by): Updated: section:notes
- 2026-11-28T10:32:11.511099+00:00 (updated-by): Updated: section:notes

## Notes Append

2026-11-28 coverage map of the required forms on 2.3 (GPT read-only review). Verified: M64/gated IQR (iqr_gemm_m64), group256 IQ8R (iq8r), Flash-Next w4nl64a8i8h engine form (moe_w4a8i8). Real tests exist but need linking to selected provider/device: E4M3 HC/MTP/ngram (hc_read_e4m3, mtp_enter, ngram_*, r3_ngram_test --selftest, r3_e4m3_f16_test), protected BF16 experts (moe_gemm_bf16 F64 oracle plus worker topology tests). Remaining gaps, smallest additions, in order: (1) U6/U7 run512 and the BF16-scale variant in family.iqr (the selftest's trellis uses 256-row runs); (2) the DFlash U4 I8-table drafter at real draft M/N/K in the IQR overlay's form matrix; (3) a draft_u2 family: gemm_w2a8_nt is execution-only today, needs an output oracle and hashes for the U2/F16/U8 layout; (4) iqr_vocab with model-derived class partitions and rank spans; (5) a bc-gfx12 HC overlay entry on hc_read_e4m3 (decode; prefill delegates upstream); (6) one assertion in the existing worker tests that protected BF16 experts stay on their provider; (7) packed-cache rows (fp8v4/k6v4 stores, fused vs standalone, page boundaries, refusal) by extending the selected selftest cases. Items 1 to 3 sit on the IQ6R decode path that ENG09 found 15% slower than IQ8R, so they go first.
- 2026-11-28T13:25:13.949344+00:00 (updated-by): Updated: section:notes_append
- 2026-11-28T15:17:04.940789+00:00 (updated-by): Updated: section:notes
