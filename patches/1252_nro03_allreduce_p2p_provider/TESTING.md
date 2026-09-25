# Testing — NRO03 P2P AllReduce provider

Static: default-off flag, source-current set_device before every peer copy, no issuer, probe gates enablement (tools/tests/patch/test_nro_patch_packages.py).

Hardware (Brutus 2x gfx1100, `-sm tensor`): subject = `GGML_CUDA_AR_P2P=1`, control = unset, same binary. Require the probe to pass (log shows P2P on) and the marker in the subject log. Correctness: greedy output and full-vocab logprobs identical to control (P2P is exact). Perf: tg128/tg512/pp4096 on Qwen3.8-27B Q8_0 plus a dense control model. Retain negative evidence if host staging wins.
