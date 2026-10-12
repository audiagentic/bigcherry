# 1355_hc_post_gate_fuse

**Status:** untested
**Plan item:** QFP35

## Mechanism

The production graph already contains 1313's F32 `SCALE -> SIGMOID -> SCALE` hyper-connection gate immediately before
`DSV4_HC_POST`. This package recognizes that exact single-consumer four-node chain and evaluates the gate inside a
3-D-grid HC_POST kernel instead of materializing the F32 gate tensor.

It preserves HC_POST's per-output arithmetic and residual/comb reduction order. The only eliminated work is the
intermediate gate write/read and one launch. `BIGCHERRY_HC_POST_GATE_FUSE=0` restores the composed 1313 + 1344 path.

Activation: `BIGCHERRY_PATCH_HIT patch=1355_hc_post_gate_fuse`.

## Qualification

**Hardware-screened, not contract-validated.** PR #54 merged 2026-10-09. Historical three-round 8K/24K/98K ABBA reported approximately +1% pooled decode, identical greedy text, and no prefill separation. This is not a four-independent-session full-logits/MTP/graph correctness gate. The package remains experiment-only in config/recipes.toml and metadata state remains untested.

**New audit risk:** bc_dsv4_hc_post_gate_grid_f32 evaluates expf for each embedding-output thread although the gate is invariant over embedding. Relative to the materialized 1313 producer this is n_embd times as many source-level sigmoid evaluations; physical wave/SFU cost is unmeasured. Profile/disassemble before trying a wave-broadcast discriminator. Require backend-op/logits/MTP/graph parity, full activation attribution and independent paired sessions before production adoption. Technical owner QFP35; action ledger BCOP170. No new performance measurement in this audit.
