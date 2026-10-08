# rd33 (1241) vs control, plain decode (no MTP), 8 order-balanced pairs

Dual 7900 XTX, Qwen3.8-27B-Q8_0, `-sm tensor`, no speculative decoding, bench set mtp-dual, sigint shutdown, all 16 cells rc=0.
Single variable: build (control 7f701510/80a35ed7 vs rd33 89722ca1/081c8574).

| metric | effect | 95% CI | pairs > 1 |
|---|---|---|---|
| tg512 | +4.52% | +4.36 .. +4.67 | 8/8 |
| tg2048 | +4.42% | +4.33 .. +4.52 | 8/8 |
| pp1024 | -0.32% | -0.51 .. -0.11 | 1/8 |
| pp4096 | -0.13% | -0.20 .. -0.04 | 1/8 |

Decode gain reproduces the fixed-work result (rd33-divergence, +4.9%) under order-balanced pairing. Prefill is ~0.1-0.3% slower
(rd33 only takes ncols_dst==1, so prefill should be unaffected; the small loss is within typical drift/noise but CIs exclude 0).

Not attributed: activation of the rd33 path was not captured in these cells (no BIGCHERRY_PATCH_TRACE); numerics differ from stock
(greedy divergence 2/4 prompts, max |dlogprob| 0.017-0.075, batch-1 PPL 11.599 vs 11.613). Work equivalence is not applicable
without MTP, but generated text differs after divergence, so throughput compares different token streams of equal length.
MTP lane: no win (see ../rd33-ab2). Applies to plain decode only.
