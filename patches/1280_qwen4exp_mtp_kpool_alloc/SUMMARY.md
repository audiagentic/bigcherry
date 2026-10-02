# 1280 qwen4exp MTP k-pool alloc

Forward-expands the qwen4exp k-pool `k_idxs` input so the MTP draft graph (no QSA layer) still allocates it. Without it, `-sm tensor` + MTP aborts at load with `GGML_ASSERT(buffer)` in `set_input_k_idxs`. Plan: QFN01.
