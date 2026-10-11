# 1355_hc_post_gate_fuse

**Status:** validated
**Plan item:** QFP35

## Mechanism

The production graph already contains 1313's F32 `SCALE -> SIGMOID -> SCALE` hyper-connection gate immediately before
`DSV4_HC_POST`. This package recognizes that exact single-consumer four-node chain and evaluates the gate inside a
3-D-grid HC_POST kernel instead of materializing the F32 gate tensor.

It preserves HC_POST's per-output arithmetic and residual/comb reduction order. The only eliminated work is the
intermediate gate write/read and one launch. `BIGCHERRY_HC_POST_GATE_FUSE=0` restores the composed 1313 + 1344 path.

Activation: `BIGCHERRY_PATCH_HIT patch=1355_hc_post_gate_fuse`.

## Qualification

**Measured historical b11474 A/B (2026-10-09/10):** Qwen3.8-Flash-Next pooled decode means 94.7 vs 94.0 t/s at 8K (+0.7%), 83.8 vs 83.4 at 24K (+0.5%), 64.6 vs 64.3 on a second 24K request (+0.5%). Greedy output and MTP acceptance unchanged; prefill and non-Qwen4Exp controls did not regress. A/B ranges overlap; this is a small profile-tier win, not a >=3% gain. No isolated b11552 HC result; see README and completed QFP35.
