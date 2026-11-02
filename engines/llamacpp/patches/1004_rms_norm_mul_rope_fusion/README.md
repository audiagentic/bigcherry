# 1004_rms_norm_mul_rope_fusion

## Summary

Was intended to fuse the per-layer rms_norm/mul(scale)/rope(+optional view/set_rows) chain into a single kernel launch instead of upstream's four-to-five separate launches, entirely outside the MUL_MAT-family dispatch-engine work.

## Outcome

Recorded state: `rejected` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
