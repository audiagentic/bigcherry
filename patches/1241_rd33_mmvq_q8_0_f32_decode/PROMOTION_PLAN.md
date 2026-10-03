# RD33 scoped promotion follow-up

## Supportable claim

PASS, scoped only: **dual Radeon RX 7900 XTX (gfx1100), `-sm tensor`, dense Q8_0 Qwen3.8-27B plain non-speculative decode: RD33 improves fixed-length generation throughput about 4.4-4.5% at tg512/tg2048 in an 8-pair order-balanced A/B.** MTP/speculative decoding is excluded.

1241 is now `ne1==1` only. The superseded 1..8 widening is historical evidence explaining the MTP perturbation; it is not a supported gate or compatibility mode.

## Correctness contract

Replace stock-relative `max_abs_logprob_diff <= 5e-4` with independent backend/F32-activation reference accuracy. Positive reference scope is representative production Q8_0 `ne1==1` shapes only. Require normalized error/NMSE <=5e-4, no NaN/Inf, and report stock and RD33 against the same reference.

Keep a fixed >=100k-token `llama-perplexity -b 1 -ub 1` quality guard (RD33 relative PPL <=+0.25%, chunk/bootstrap upper CI <=+0.5%) and >=32 deterministic prompts reporting top-1 agreement/divergence. Full-vocabulary KL is optional only if existing tooling emits full logits.

MTP is a negative-activation/control lane: traced subject must not hit widths 2..8, especially verify width 5; control must not hit at all. MTP acceptance/output should remain control-equivalent because verify stays on stock. Plain non-MTP tg512/tg2048 is the positive performance lane; pp and MTP are no-regression controls.

## Minimum hardware work

1. Activation: subject width-1 hit; subject no-hit widths 2..8/MTP verify; control no-hit.
2. Backend/F32 reference accuracy for representative production Q8_0 width-1 shapes.
3. MTP control: acceptance/output behavior remains stock/control-equivalent; no RD33 verify activation.
4. Contract-admitted order-balanced plain-decode A/B, tg512/tg2048 positive and pp <=1% regression.
5. Fixed >=100k-token PPL + >=32-prompt distribution guard.

Architecture scope may remain gfx1100-only while runtime eligibility is gfx1100-exact. Only contract-campaign evidence accepted by `patch-verify-evidence` can promote the patch.