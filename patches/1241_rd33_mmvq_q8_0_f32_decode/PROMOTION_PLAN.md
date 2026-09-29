# RD33 scoped promotion follow-up

## Supportable claim

PASS, scoped only: **dual Radeon RX 7900 XTX (gfx1100), `-sm tensor`, dense Q8_0 Qwen3.8-27B plain non-speculative decode: RD33 improves fixed-length generation throughput about 4.4-4.5% at tg512/tg2048 in an 8-pair order-balanced A/B.** MTP/speculative decoding is excluded.

Correction: shipped PRBE26 eligibility is `ne1 >= 1 && ne1 <= 8`, not single-column-only. Dual-XTX trace evidence confirms ncols 1/2/4 hits, and MTP n_max=4 verify width 5 is eligible. This directly explains why MTP verification numerics/acceptance can change.

## Correctness contract by gate width

Common rule: replace stock-relative `max_abs_logprob_diff <= 5e-4` with independent backend/F32-activation reference accuracy. Require normalized error/NMSE <=5e-4, no NaN/Inf, and report stock and RD33 against the same reference. Keep a fixed >=100k-token `llama-perplexity -b 1 -ub 1` quality guard (RD33 relative PPL <=+0.25%, chunk/bootstrap upper CI <=+0.5%) and >=32 deterministic prompts reporting top-1 agreement/divergence. Full-vocabulary KL is optional only if existing tooling can emit full logits.

**Current 1..8 gate:** backend-reference shapes must cover widths 1..8, emphasizing production widths 1 and MTP verify 5. MTP acceptance, greedy behavior, and end-to-end performance are in-scope correctness/behavior evidence because RD33 modifies verify. Current acceptance 0.90101 -> 0.95580 and tg2048 -5.65% block an MTP promotion claim.

**Proposed ne1==1 gate:** backend-reference positive scope is width 1 only. MTP becomes a negative-activation/control lane: traced subject must hit width 1 but not width 5; MTP acceptance/output should remain control-equivalent under the unchanged verify path. Plain non-MTP tg512/tg2048 remains the positive performance lane; pp and MTP are no-regression controls.

## Next experiment / patch structure

Run `ne1==1` as the next isolated experiment. If it retains the plain-decode gain and eliminates MTP perturbation, **narrow patch 1241 itself and migrate callers/compositions**. Do not add a permanent runtime option/backward-compat shim for 1..8: that would combine two materially different hypotheses under one patch/contract.

Only create a separate patch for the 1..8 behavior if it is intentionally retained as an independently useful multi-column/MTP optimization with its own positive evidence. That patch would require widths 1..8 reference validation plus MTP acceptance/quality/performance gates.

## Minimum hardware work after narrowing

1. Trace activation: subject hit at width 1; explicit no-hit at MTP verify width 5; control no-hit.
2. Backend/F32 reference accuracy for representative production Q8_0 width-1 shapes.
3. MTP control check: acceptance/output behavior returns to stock/control behavior; this is a non-regression check, not a performance qualification campaign.
4. Contract-admitted order-balanced plain-decode A/B (tg512/tg2048 positive; pp <=1% regression), reusing prior evidence only if the evidence system permits it after the gate/contract hash changes.
5. Fixed >=100k-token PPL + >=32-prompt distribution guard for validated correctness evidence.

Architecture scope may remain gfx1100-only while the runtime eligibility is gfx1100-exact.