# Testing — RD33 Q8_0 F32 single-column decode

Target positive lane: Qwen3.8-27B-Q8_0, dual 7900 XTX/gfx1100, `-sm tensor`, plain non-speculative decode. MTP n_max=4 is a non-activation/control lane.

## Activation
Under `BIGCHERRY_PATCH_TRACE=1`, require subject hit/control miss for `BIGCHERRY_PATCH_HIT patch=1241_rd33 path=q8_0_f32_decode ncols=1`. On subject, explicitly require no marker with `ncols=2..8`; MTP verify width 5 must stay on stock.

## Correctness
Run Q8_0 MUL_MAT backend/CPU-reference checks with explicit production-representative `ne1==1` shapes. Output is intentionally not stock-bit/logprob identical because activation Q8_1 quantization is removed. Use reference-based numerical accuracy plus the fixed-corpus PPL/distribution quality guard described in PROMOTION_PLAN.md. MTP control must preserve stock verification behavior because RD33 must not activate there.

## Performance
Use order-balanced fixed-work plain non-MTP A/B; report tg512/tg2048 positive effect and pp1024/pp4096 no-regression controls. One variable = 1241. MTP performance cannot satisfy the positive lane; run it only to prove non-activation/no collateral regression and acceptance parity after narrowing.

## Promotion
Only contract-campaign evidence accepted by `patch-verify-evidence` is promotion evidence. Ad-hoc lab A/B establishes the hypothesis but does not replace validated evidence.