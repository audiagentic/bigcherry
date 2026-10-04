# 1330_qsa_mask_inplace

**Status:** untested
**Plan item:** QFP17

## What it does

With `BIGCHERRY_QSA_MASK_INPLACE=1`, build_qsa_sel adds the causal kq_mask into the QSA selection mask in place
(`ggml_add_inplace` on the view of the set_rows result) instead of allocating a second dense [n_kv, T] F16 tensor.
Same values; removes one full-context x ubatch mask per QSA layer (~242 MiB at ub512, ~484 MiB at ub1024 at 240K,
mirrored on every Meta rank) from the compute buffer.
