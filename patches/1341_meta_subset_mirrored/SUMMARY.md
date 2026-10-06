# 1341_meta_subset_mirrored

**Status:** untested
**Plan item:** MSM03

Kind: enhancement, flag `BIGCHERRY_META_SUBSET_MIRROR` (default 0).

Adds `active_mask` to Meta split state (0 = legacy/all devices), propagates it across MIRRORED operations, and honors it in simple-tensor creation and Meta transfers. This first step seeds only Qwen4Exp `cache_idx_(k|v)_l*` from nonzero `BIGCHERRY_ATTN_TS` entries. KQ/kpool/QSA compute inputs are intentionally not seeded yet.

## Evidence

- Offline mechanics test: `tools/tests/patch/test_1341_meta_subset_mirrored.py`.
- Compile, runtime fidelity and hardware memory measurements: pending.
