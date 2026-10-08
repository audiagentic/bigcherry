# 1330_qsa_mask_inplace

**Status:** superseded
**Plan item:** QFP17

## What it does

With `BIGCHERRY_QSA_MASK_INPLACE=1`, build_qsa_sel adds the causal kq_mask into the QSA selection mask in place
(`ggml_add_inplace` on the view of the set_rows result) instead of allocating a second dense [n_kv, T] F16 tensor.
Same values; removes one full-context x ubatch mask per QSA layer (~242 MiB at ub512, ~484 MiB at ub1024 at 240K,
mirrored on every Meta rank) from the compute buffer.

## Supersession

Superseded by `1332_qsa_token_chunk` at pin b11474. The production composition already contains 1332, and `patch-rebase-check --experiment deploy-v6-plus-1330` refuses the experiment because 1330 conflicts with 1332. Therefore 1330 cannot be built on the production set and is retired rather than carried as an unbuildable experiment.
