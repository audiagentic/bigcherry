# 1301_prbe115_q8_f32_mtp_widths

## Summary

PRBE115: widen 1241's Q8_0 raw-F32-activation MMVQ path (no q8_1 activation quantize) from ncols_dst == 1 to the MTP verify widths, and optionally to RDNA4. Opt-in for screening: `BIGCHERRY_Q8_F32_MAXCOLS=N` (1..8) and `BIGCHERRY_Q8_F32_RDNA4=1`. The 1241 kernel branch is already column-generic; only the host gate and the per-width compile-time dispatch change.

## Outcome

Recorded state: `rejected` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
