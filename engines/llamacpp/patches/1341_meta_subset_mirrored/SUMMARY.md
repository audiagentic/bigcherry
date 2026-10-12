# 1341_meta_subset_mirrored

**Status:** validated
**Plan item:** MSM03

`BIGCHERRY_META_SUBSET_MIRROR=1` seeds Qwen4Exp `cache_idx_(k|v)_l*` full replicas only on devices with nonzero `BIGCHERRY_ATTN_TS`; default 0 preserves legacy. `active_mask=0` means all devices. Existing Meta allocation, transfers and MIRRORED propagation honor the subset.

**Current source scope (2026-10-12):** `patch.py::_N_INPUT_RULE` also marks `GGML_OP_NONE` KQ-mask graph inputs by name, including scheduler-prefixed copies. `BIGCHERRY_META_SUBSET_MIRROR_INPUTS` defaults to **1** when subset mirroring is enabled; 0 retains legacy input mirroring. This corrects the original "indexer only" summary. Independent kpool or computed QSA-mask seeding is not proven.

**Open conditional lifecycle gate:** `bc_meta_attn_input_mask` is mutable process-global storage set during cache classification and read by graph split inference without synchronization/reset/per-instance identity. No failure has been reproduced in the supported single-policy production layout. MSM03 specifies a CPU-first instance-isolation discriminator and minimal callback-based binding; do not expand mask coverage before resolving it.

**Existing measurement:** 1,440 MiB less indexer cache on zero-attention-share R9700 at ctx 245760 with identical greedy/probes. See README for exact controls. No new GPU result.
