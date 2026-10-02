# 1280_qwen4exp_mtp_kpool_alloc

**Status:** untested
**Plan item:** QFN01

## What it does

Forward-expands the qwen4exp k-pool `k_idxs` and `new_pool_*` inputs so the MTP draft graph (no QSA layer) still allocates them. Without it, `-sm tensor` + MTP aborts at server load with `GGML_ASSERT(buffer)` in `set_input_k_idxs` / `set_input_kpool`.
