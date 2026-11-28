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

## Change Log

- 2026-11-26T01:36:30.551307+00:00 (created-by): Created by agent
- 2026-11-26T01:37:56.167096+00:00 (updated-by): Updated: section:steps, section:notes
- 2026-11-26T05:07:03.703862+00:00 (updated-by): Updated: section:notes
- 2026-11-28T06:21:30.682022+00:00 (updated-by): Updated: section:notes
- 2026-11-28T07:25:59.101118+00:00 (updated-by): Updated: section:notes
- 2026-11-28T08:33:47.490967+00:00 (updated-by): Updated: section:notes
- 2026-11-28T09:24:04.635107+00:00 (updated-by): Updated: section:notes
