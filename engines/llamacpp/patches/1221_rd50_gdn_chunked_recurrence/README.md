# 1221 RD50 GDN chunked recurrence

## Summary

Folds the two new upstream-fork files (gated_delta_net_chunked.cu/.cuh) into gated_delta_net.cu, adding a GDN_CHUNK=32-token chunked kernel that turns the token-by-token delta-rule recurrence into dense matmuls over LDS on RDNA3.5, narrowly gated (scalar gate only, S_v==128, n_tokens>32, RDNA3_5) with an env-var force-disable for A/B benching.

## Outcome

Recorded state: `untested` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

This package owns the RD50 implementation and its activation-gate evidence.
The evidence records the compiled RDNA3.5 eligibility result and the verified
non-RDNA3.5 fallback path; the RDNA3.5 performance claim remains unvalidated.


## Static validation-package repair (2026-10-08)

The existing Experiment Contract is explicitly bound and `validation.toml` declares apply, build, correctness, activation, performance, and controls. This repairs the former `patch-lint` missing-binding/missing-adapter errors; it does **not** certify a validation campaign or introduce measurements. The declared activation marker does not yet exist in `patch.py`, so a new validation attempt must remain BLOCKED until real subject-hit/control-miss instrumentation and evidence are added. This patch remains untested/deferred on the unavailable gfx1151 hardware.
