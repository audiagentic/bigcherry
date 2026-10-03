# Independent review — 1241 RD33 MMVQ Q8_0 F32 decode

## Current eligibility

1241 is now narrowed to dense Q8_0, gfx1100/RDNA3_0, `ne1 == 1`, `!forced.requested()`. The superseded PRBE26 `ne1=1..8` experiment directly modified MTP verify width 5, changed acceptance 0.90101 -> 0.95580, and measured tg2048 -5.65%; that widening is no longer part of 1241. Narrowed-build activation evidence shows only ncols=1.

## Scoped verdict

The evidence supports only: dual-gfx1100 Qwen3.8-27B-Q8_0 `-sm tensor` plain non-MTP decode is faster with RD33; the order-balanced 8-pair experiment measured tg512 +4.52% (CI +4.36..+4.67) and tg2048 +4.42% (CI +4.33..+4.52), 8/8 faster. MTP is explicitly excluded from the positive claim.

## Correctness/quality

Stock-relative logprob identity is scientifically mismatched because RD33 intentionally removes Q8_1 activation quantisation. The correctness contract should use an independent F32-activation/backend reference for explicit production-representative `ne1==1` Q8_0 shapes. Retain the broader fixed-corpus PPL/distribution guard in PROMOTION_PLAN.md.

For MTP, correctness is now a **non-activation/control property**: subject must not emit an RD33 marker for ncols 2..8 (especially verify width 5), and the narrowed build should preserve normal MTP acceptance/output behavior because verify remains on stock. The finishing `rd33n1-ab1` run is relevant control evidence, not the positive performance lane.

## Promotion conditions

Promotion requires contract-produced validated evidence for: (1) subject hit at width 1/control miss plus explicit subject non-hit widths 2..8; (2) width-1 backend/reference accuracy; (3) broader PPL/distribution quality guard; (4) contract-admitted order-balanced plain-decode gain; and (5) MTP non-activation/no-regression control. gfx1100-only promotion does not require other architectures while the hardware gate remains gfx1100-exact.