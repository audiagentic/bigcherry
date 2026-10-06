# 1341_meta_subset_mirrored

**Status:** validated
**Plan item:** MSM03

Kind: enhancement, flag `BIGCHERRY_META_SUBSET_MIRROR` (default 0).

Adds `active_mask` to Meta split state (0 = legacy/all devices), propagates it across MIRRORED operations, and honors it in simple-tensor creation and Meta transfers. This first step seeds only Qwen4Exp `cache_idx_(k|v)_l*` from nonzero `BIGCHERRY_ATTN_TS` entries. KQ/kpool/QSA compute inputs are intentionally not seeded yet.

## Evidence

- Offline mechanics test: `tools/tests/patch/test_1341_meta_subset_mirrored.py`.
- Hardware: see README.md (identical output, no speed change, 1,440 MiB less on the R9700 at ctx 245760).
