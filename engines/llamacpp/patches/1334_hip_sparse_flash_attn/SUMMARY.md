# 1334_hip_sparse_flash_attn

**Status:** validated
**Plan item:** QFP25/QFP17

Kind: enhancement, on by default; `BIGCHERRY_FA_SPARSE=0` is the off switch.

Enables upstream's sparse flash attention on RDNA3/RDNA4 WMMA. Upstream compacts the attention mask into one index
list per tile of queries and gathers only those K/V cells in the MMA kernel, but compiles the path out for HIP and
selects it on NVIDIA only. On AMD, Qwen4Exp QSA attention therefore reads the whole KV cache although each query can
see about 2048 cells.

The patch adds a HIP version of the index kernel (AMD wave ballot and popcount, correct for 32- or 64-lane waves), compiles the host side and
the two dispatch sites for HIP, accepts RDNA WMMA in the selection when the flag is set, and makes the RDNA tile-shape
choice pick ncols2 = 8 (the only shape with sparse kernels) when the sparse path would be taken.

## Evidence

Motivation (b11402, production config, rocprofv3 kernel trace of one uncached prefill): flash attention is 1.96 s of
23.5 s kernel time per XTX over 31.8K tokens and 18.9 s of 89.5 s over 99.3K tokens (32 -> 97 ms per 512-token
ubatch); the R9700 holds no attention and spends the equivalent time waiting in the all-reduce.

Results (b11402, Brutus, 2026-10-05/06; details and the promotion rationale in README.md):

- Prefill, ABBA, production config: ~99K tokens 862.4 / 872.1 -> 978.7 / 982.5 t/s (+13%); ~202K tokens
  658.2 / 663.5 -> 846.1 / 847.8 t/s (+28%). 24K MTP decode unchanged (41.5 / 41.6 vs 41.2 - 41.4 ms/step).
- Correctness: test-backend-ops sparse-mask flash attention matches the CPU backend on all four GPUs with the flag
  off and on, with the activation marker proving the 8x8 sparse kernel ran.
- Fidelity: not bit-identical to the dense path (different summation order feeding a discrete top-k selection);
  against a CPU f32 reference the sparse path is no further away than the dense path.
