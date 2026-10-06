# 1335_tiled_lightning_indexer

**Status:** validated
**Plan item:** QFP17

Kind: enhancement (upstream backport), on by default; `BIGCHERRY_INDEXER_TILE=0` is the off switch.

Backport of upstream llama.cpp #29901 (commit 1b43d3116, merged after the b11402 pin): the QSA lightning indexer for
4 heads scores a tile of 64 keys against 8 tokens per block, staging the keys once in shared memory instead of
re-reading every key from global memory for every token. Batches under 8 tokens (decode) keep the vector kernel.
The code is upstream's, plus the off switch and an activation marker.

Not bit-identical to the vector kernel (non-F16 keys pass through F16; different summation order).

Superseded when the pin reaches a llama.cpp release that contains #29901.

## Evidence

Motivation (b11402, rocprofv3 prefill trace): the indexer is 7.6 ms of each 512-token ubatch averaged over 32K
tokens and 22 ms averaged over 100K. Upstream reports the indexer 16.4 -> 6.4 ms at kv 65536, 2048 tokens on CUDA.

Result (b11402, Brutus, 2026-10-06; details in README.md): prefill +2.6% at ~99K tokens and +5.6% at ~202K, decode
time per step unchanged, backend tests equal to CPU on all four GPUs with the kernel proven active, next-token
fidelity inside the envelope of the other attention-path changes.
