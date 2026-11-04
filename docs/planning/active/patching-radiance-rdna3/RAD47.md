---
id: RAD47
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-11-04T11:41:29.284540+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# gfx1100 MXFP4 GEMM row selection: the tiled kernel beats the skinny rows from M=9 up on the 7900 XTX (27B TP2 prefill and multi-stream decode)

## Description

Found with the kernel A/B harness (KAB01) on 2026-11-04 while explaining the 27B prefill on two 7900 XTX cards (qwen27b-2xtx, MXFP4 container, radiance v1.16.0, libr3 = our gfx11 port of the libr4d kernels). rad-kbench with --all-rows times every candidate row of gemm_nt_q on one XTX at the 27B shapes. The big prefill GEMM (tiled row) is already near the card: 85-107 TFLOP/s at M=2048 against about 123 for bf16 WMMA. But the row priorities libr3 inherits from libr4d (tuned on gfx1201, the R9700) pick the SLOWEST candidate in the middle bands on gfx1100: M=9..64 -> nt_m64 / gemv_wide and M=65..128 -> decode, while the tiled row, which is valid for every M >= 1, is far faster wherever it was measured. Measured (microseconds, one XTX, cold weights, 256 MiB rotation): N=34816 K=5120: M=64 nt_m64 3269 vs tiled 530 (6.2x); M=65 decode 9363 vs tiled 539 (17x); M=128 decode 3662 vs tiled 571 (6.4x). N=12288 K=5120: M=64 nt_m64 1129 vs tiled 187 (6.0x); M=128 decode 1397 vs tiled 199 (7.0x). N=5120 K=17408: M=64 nt_m64 1618 vs tiled 389 (4.2x); M=128 decode 1771 vs tiled 403 (4.4x). N=5120 K=25600: M=64 nt_m64 2376 vs tiled 607 (3.9x); M=128 decode 2615 vs tiled 629 (4.2x). The engine reaches these bands with several concurrent streams plus speculation (for example 4 streams x (1 + 7 drafts) = 32 rows), with the last chunk of a prefill (65-128 tokens), mixed prefill/decode steps and prefix-cache resumes. Not yet measured: the tiled kernel below M=64 (the recorded fixture has no cases there and the rows' constraints did not exercise it in the all-rows pass), and Flash-Next, which runs on the R9700 (libr4d, gfx1201 priorities, where the upstream tuning applies) and so is not affected.

## Steps

1. Measure the gap in the 9..63 band: record a fixture that reaches those rows (rad-kbench --record-rows moves the fixture cases into the constraints of every row of the libraries under test) and time tiled against nt_m64 and gemv_wide at M = 9, 16, 24, 32, 48, 63 for the 27B shapes, with the kernel A/B harness (kbench_ab, KAB01) under the gpu lease; derive the crossover per shape class from the measurements, as RAD46 did, with no hand-picked cutover.
2. Settle bit-invariance. libr4d documents that a row's bits are the same in gemv and gemv_wide at every M (one 16-row tile chain), which speculative verification relies on. Check whether the tiled kernel's per-row sum order equals that chain; if it does not, a cutover changes a row's bits when the batch size crosses it. Quantify with the correctness section of rad-kbench (rel_l2 against the reference per row) and with the decode-vs-verify identity used by the engine's own tests; decide whether the cutover must be exact or whether a documented bounded difference is acceptable (KLD and draft acceptance gate).
3. Implement as a managed change in the libr3 component (row constraints or priority for gfx1100 only; libr4d priorities for gfx1201 stay untouched), keyed on the measured crossover and on gfx1100, not on the model. No model constants.
4. Validate: kbench replay correctness for the changed rows (all fixtures), then end to end on qwen27b-2xtx: decode ms/step at 1, 4 and 8 streams with and without speculation, prefill of a 7K-token prompt cold (baseline 2.57 s), the 27B KLD against the BF16 reference, and the Flash-Next production layout unchanged (it does not load libr3 for the engine).
5. Record the result in this item.

## Detailed Solution & Technical Design

Where the priorities live: vendor libr4d r4d_rows.cpp, constraint blocks cGemmMxfp4A8Decode (M <= 128, N <= 36864), cGemmMxfp4A8Gemv (M <= 8), cGemmMxfp4A8GemvWide (9 <= M <= 64, K % 128), cGemmMxfp4A8DecodeShortK and cGemmMxfp4A8Tiled (M >= 1); the comments state the tiled row 'starts at 1 rather than at a hand-picked crossover' because the decode and skinny rows outrank it by priority. Those priorities were chosen from gfx1201 measurements (for example '19456 x 2560 at 64 rows: 93.3 us against gemv_wide's 131.8 on gfx1201'). libr3 ports the kernels to gfx11 and registers the same rows (components/libr3/r3_registry.cpp and native/*.hip.rw), so the gfx11 behaviour of the skinny kernels (they use the BF16 WMMA path, with 15 of 16 fragment rows idle at M=1 and weaker reuse at larger M) was never part of the selection. A gfx1100-specific constraint set is the natural place for the fix. Risk to weigh: the skinny rows exist partly so that a token's logits do not depend on how many other rows share the step; replacing them with a kernel of a different accumulation order can change draft acceptance and, at the bit level, outputs. The M <= 8 band (single-stream decode with up to 7 drafts) is deliberately NOT in scope: it is memory bound there and gemv is right; the item concerns M >= 9 only (first measured gain at M=64/65).

## Code Samples & Guidance



## Files

engines/radiance/components/libr3/r3_registry.cpp and the native/*.hip.rw rewrites (row constraints/priority for gfx1100), engines/radiance/components/libr3/component.toml if a setting is needed, tools/lab/radiance/kbench_ab.py (reused), new or extended kbench fixture recipes, docs/reference/engines/radiance/models/qwen3.8-27b.md for the measured effect.

## Validation

Card-free: row-selection unit tests for gfx1100 over M = 1..4096 and the shapes of the 27B (selection equals the measured-fastest row at every measured point). Hardware under the gpu lease: kbench --all-rows before/after on the 27B fixtures (correctness section clean for every changed row); qwen27b-2xtx end-to-end decode ms/step at 1/4/8 streams (with and without speculation), cold prefill of the same 7K-token prompt, and KLD against the BF16 reference on the standard harness; Flash-Next production layout decode and prefill unchanged.

## Effort & Risk

Medium. Low risk to Flash-Next (different engine card and library). The risk is numerical: kernels with different accumulation order change bits across the cutover, which can change speculative acceptance and outputs slightly, so it needs the KLD and acceptance gates. The measurement is cheap (minutes per sweep with the harness).

## Standards

Managed component change only, thin and keyed on the measured crossover (derive from the sweep, no hand-picked cutover); no model constants; no compatibility shim; Flash-Next and gfx1201 priorities untouched; GPU runs under the lease; report decode and prefill separately.

## Acceptance Criteria

1. A measured crossover table for gemm_nt_q (and the gated variants if they show the same pattern) on gfx1100, from the A/B harness. 2. The bit-invariance question answered with data and the accepted policy recorded. 3. If a cutover ships: 27B prefill and multi-stream decode improve beyond run-to-run variance, KLD within the standard limits, draft acceptance not worse, Flash-Next unchanged. 4. If it does not ship: the reason is recorded here.

## Notes

Evidence (kept in the work root, not in git): libr3 gemm_nt_q all-rows timing report on the 2048-row fixture of the 27B MXFP4 container, one 7900 XTX, 2026-11-04 (729 rows, 5 of 770 declared rows ran). Related: RAD04 (overlap the all-reduce with the next GEMM; reopened), KAB01 (kernel A/B harness), RAD46 (a derived row-count crossover on gfx1030, same method), RAD37 (the libr3d A/B it asked for). The libr3d (upstream RDNA3 plugin) comparison is still open: its gemm_nt_q produced no rows in 15 minutes while the GPU was busy; given the tiled row is already at 85-107 TFLOP/s on large M, the larger lever is row selection, not a different GEMM kernel.

## Change Log

- 2026-11-04T11:41:29.284540+00:00 (created-by): Created by agent
- 2026-11-04T11:41:58.549037+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20261104_114203_measured-that-qwen38-27b-on-t_7412
- 2026-11-04T11:42:07.734096+00:00 (updated-by): Updated: section:ledger-events
