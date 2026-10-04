# 1327_qsa_host_remap

**Status:** evaluated
**Plan item:** QFP13

## What it does

Pin 0504396 (upstream #29819) remaps dead QSA selection slots to private dump rows with ~12 small graph ops per QSA
layer; the kernel census after the bump showed +~54 kernels per token per XTX and decode -2.3% at ~8K. With
`BIGCHERRY_QSA_HOST_REMAP=1` the two terms that depend only on host data - `live_tail` (from the host input tail_idxs)
and `dump` (n_kv + slot) - are computed once per graph on the host in the kpool input's set_input and fed to every QSA
layer, removing six ops per layer. Values are bit-identical; the device live_pool and final remap are unchanged.

## Hardware result (2026-10-04, pin 0504396, env screen on one v5+1327 build)

~24K 41.4 vs 42.1/41.4 ms/step, ~80K 46.8 vs 47.8/47.1 (0..-1%); greedy identical at both depths. Census: kernels/token 1053 -> 1027 per XTX (elementwise 312 -> 292); the old pin was ~1000, so ~half of the #29819 cost is removed. The rest (device live_pool/get_rows/repeat/concat/final remap) needs a fused remap kernel. Small win, no regression: include in the next profile.
