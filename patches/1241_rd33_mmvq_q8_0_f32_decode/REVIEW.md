# Independent review — 1241 RD33 MMVQ Q8_0 F32 decode

## Correction: shipped eligibility

The shipped PRBE26 gate is **not** single-column-only. It is dense Q8_0, gfx1100/RDNA3_0, `!forced.requested()`, with `ne1 >= 1 && ne1 <= 8`. Activation evidence on dual XTX confirms hits at ncols 1, 2, and 4. Therefore an MTP `n_max=4` verify batch (`ne1=5`) is eligible for the same F32-activation path. Earlier statements in this review that RD33 directly touched only `ncols_dst==1` were stale.

## Scoped verdict

I sign off on the narrow measured claim that dual-gfx1100 Qwen3.8-27B-Q8_0 `-sm tensor` **plain non-MTP decode** is faster with the current RD33 build: the order-balanced 8-pair follow-up reports tg512 +4.52% (CI +4.36..+4.67) and tg2048 +4.42% (+4.33..+4.52), 8/8 faster. This is not an MTP promotion claim.

The MTP result is now mechanistically less ambiguous: because the widened gate includes the 5-column verify batch, RD33 changes target verification numerics directly. Acceptance changes deterministically from 0.90101 to 0.95580 and tg2048 is -5.65%. Different generated text/acceptance still prevents a fixed-work kernel attribution, but the widened RD33 path is a direct causal candidate rather than an incidental single-column-only effect.

## Correctness/quality

Stock-relative `max_abs_logprob_diff <= 5e-4` remains scientifically mismatched: RD33 intentionally removes Q8_1 activation quantisation and changes accumulation arithmetic. Use an independent F32-activation/backend reference. For the **current 1..8 gate**, reference validation must cover every eligible width 1..8, with production emphasis on widths 1 and 5; quality/acceptance behavior under MTP is part of the contract because verify is modified. For a future **ne1==1-only gate**, the primary backend-reference matrix narrows to width 1 and MTP becomes a non-activation/control requirement: prove no RD33 marker at verify width 5 and require MTP acceptance/output behavior to match the control within the normal unchanged-path contract.

The existing 4096-token batch-1 PPL result (11.5989 RD33 vs 11.6132 control) is supportive, not sufficient promotion evidence. Keep the broader fixed-corpus PPL/distribution guard proposed in `PROMOTION_PLAN.md`.

## Recommendation on gate narrowing

Treat `ne1==1` as a **narrowing/migration of patch 1241**, not a permanent option or compatibility shim, if the project decision is that RD33's supported product is plain non-MTP decode. The 1..8 widening has no demonstrated production MTP benefit and has a material tg2048 regression; keeping two selectable behaviors would create two hypotheses/contracts under one patch. First run the `ne1==1` change as an isolated experiment (temporary experiment branch/variant is fine for evidence). If it preserves the plain-decode win and removes MTP perturbation, replace 1241's 1..8 gate with `ne1==1`, update its docs/contract, and migrate callers/compositions directly.

Create a **separate patch** only if the 1..8 behavior remains scientifically/product-useful enough to carry independently (for example, a later reference-valid MTP optimization with positive end-to-end evidence). In that case it is a distinct hypothesis: multi-column/MTP F32-activation MMVQ, with its own contract and acceptance/performance gates. Do not expose a runtime width option merely to preserve the old behavior.

## Promotion conditions

For the intended ne1==1-only 1241: (1) activation subject hit at width 1 and explicit non-hit for MTP verify width 5; (2) width-1 backend/reference accuracy; (3) broader PPL/distribution quality guard; (4) contract-admitted order-balanced plain-decode gain; and (5) MTP control proving the narrowed patch no longer changes verify acceptance/output materially. gfx1100-only promotion does not require other architectures while the hardware gate remains gfx1100-exact.