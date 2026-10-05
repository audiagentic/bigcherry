# 1334_hip_sparse_flash_attn

**Status:** untested
**Plan item:** QFP25/QFP17

Kind: enhancement, flag `BIGCHERRY_FA_SPARSE` (default 0).

Enables upstream's sparse flash attention on RDNA3/RDNA4 WMMA. Upstream compacts the attention mask into one index
list per tile of queries and gathers only those K/V cells in the MMA kernel, but compiles the path out for HIP and
selects it on NVIDIA only. On AMD, Qwen4Exp QSA attention therefore reads the whole KV cache although each query can
see about 2048 cells.

The patch adds a HIP version of the index kernel (no warp ballots, wave-size independent), compiles the host side and
the two dispatch sites for HIP, accepts RDNA WMMA in the selection when the flag is set, and makes the RDNA tile-shape
choice pick ncols2 = 8 (the only shape with sparse kernels) when the sparse path would be taken.

## Evidence

Motivation (b11402, production config, rocprofv3 kernel trace of one uncached prefill): flash attention is 1.96 s of
23.5 s kernel time per XTX over 31.8K tokens and 18.9 s of 89.5 s over 99.3K tokens (32 -> 97 ms per 512-token
ubatch); the R9700 holds no attention and spends the equivalent time waiting in the all-reduce.

- Build on HIP: pending.
- Correctness (test-backend-ops flash attention with a sparse mask, both architectures): pending.
- Prefill ABBA at ~100K and ~200K, output vs the CPU f32 reference: pending.
