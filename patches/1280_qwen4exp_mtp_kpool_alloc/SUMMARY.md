# 1280_qwen4exp_mtp_kpool_alloc

**Status:** rejected
**Plan item:** QFN01

## What it does

Forward-expands the qwen4exp k-pool `k_idxs` and `new_pool_*` inputs so the MTP draft graph (no QSA layer) still allocates them. Without it, `-sm tensor` + MTP aborts at server load with `GGML_ASSERT(buffer)` in `set_input_k_idxs` / `set_input_kpool`.

## Rejected (2026-10-02)

Root cause was the legacy unsloth MTP sidecar, not the k-pool allocator: it writes `compress_ratios[48] = 0` (dense MTP), while c061df198 (#29761) expects the MTP layer to be QSA (ratio 4). The k-pool inputs were then built but dead. With the sidecar's ratio corrected to 4, pristine c061 runs MTP under `-sm tensor` with greedy parity (decode 28.3 -> 41.8/42.9 t/s at depth 2/3). This patch only masked the metadata mismatch.
