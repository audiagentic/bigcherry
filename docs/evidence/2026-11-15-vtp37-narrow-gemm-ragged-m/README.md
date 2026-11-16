# VTP37 step 3a: gfx1100 narrow 8-bit GEMM at 17-31 tokens

2026-11-15. Layout `qwen27b-tp3-int8-556-long` (TP3: 7900 XTX, 7900 XTX, R9700; split 5,5,6; fp8 KV cache; DFlash2
depth 3; 8 sequences). Engine argv from the stored manifest; both arms load a libr3 first through `--radiance-home`:
R is main's libr3 (RAD52 + RAD53), V is R plus this change (`native/r4d_gemm_fp8a8.hip.rw`).

## Finding

Decode at 5-7 streams issues 17-31 token rows to the narrow GEMM (`gemm_i8a8_nt_m16`, two 16-row token tiles). libr4d
instantiates its unguarded form (`Full`) only when N, K and the token count are all whole tiles, and `Full` drops the
weight-row, K-segment and token guards together. A ragged token count therefore ran the fully guarded kernel, which on
gfx1100 costs more than 32 tokens unguarded. Pinning the wave count or token-tile count (`--pin-mw`, `--pin-nt`) did not
recover it; the guard form did.

## Change

The token guard is always kept (one scalar compare per staged row, per scale read and per output tile), and `Full` is
decided from N and K alone, so a ragged token count no longer brings back the weight-row and K guards. The same
products are summed in the same order; only rows past M, which were never stored, are skipped.
`-Rpass-analysis=kernel-resource-usage`: VGPRs and occupancy unchanged in every instantiation; scratch 24 -> 28 bytes a
lane in one (Full, MW 4, FK 256).

## Kernel, one XTX

`r3_selftest --perfw8 N K M 20 --cold 256` (weights cold, as in decode), median of 3, rule's own choice (no pins), us:

| N x K | M 1 / 8 / 16 | M 17-31 | M 32 | M 33-48 |
|---|---|---|---|---|
| 11264 x 5120 | 81 / 82 / 82 -> same | 137-140 -> 125-127 | 127 -> 128 | 181-193 -> 185-186 |
| 5120 x 5632 | 60 / 60 / 60 -> same | 110-112 -> 90-91 | 90 -> 91 | 194-200 -> 197-198 |
| 5120 x 5120 | 55 / 55 / 55 -> same | 100-102 -> 83-84 | 83 -> 84 | 177-183 -> 180-181 |
| 5120 x 1920 | 37 / 37 / 34 -> 39 / 39 / 36 | 42-43 -> 33 | 33 -> 33 | 68-70 -> 67-68 |

M 17-31 now costs what M 32 does. The always-on token guard costs about 2 us (5%) at M <= 16 on the smallest shape and
up to 2% at M 32-48. The fp8 trunk's row (`gemm_fp8a8_nt_m16`, not served on gfx1100) moves within its run-to-run noise.

## Correctness

- `r3_selftest --case`: `gemm_i8a8_nt_m16` 13 ok, `gemm_fp8a8_nt_m16` 2 ok, `gemm_fp8a8_bias_nt_m16` 9 ok, 0 FAIL, on
  both builds. The rows cover ragged M (1, 8, 24, 40, 48 with split-K) against a host reference and byte identity
  across every wave and token-tile count.
- Engine identity, fp8 `-long` argv, long corpus (5 documents, 276,900 positions, one sequence, no speculation: M = 1,
  which now takes the `Full` form), against the RAD52 reference that main's libr3 already matched: perplexity equal to
  every digit (3.0921531658), top-1 agreement 1.0, KLD max 2.7e-8.
- Stress: 12 fresh starts of V, each a 4/5/6/7/8-stream bench of 512-token prompts, then 6 streams generating 1500
  tokens beside a 60k-token prompt: 12/12 up and alive, 0 fault lines, every bench ok.

## Engine, decode-only

Order R V V R, twice (4 fresh starts per arm), 256 new tokens a stream, chunks a second after the last stream's first
chunk:

| streams | R (main) | V | change |
|---|---|---|---|
| 1 | 36.26 (36.2-36.5) | 35.94 (35.9-36.0) | -0.9% |
| 4 | 101.3 | 101.1 | -0.3% |
| 5 | 100.3 | 110.2 | +9.9% |
| 6 | 107.5 | 119.6 | +11.3% |
| 7 | 122.5 | 132.5 | +8.2% |
| 8 | 134.5 | 136.1 | +1.2% (noise: 119-154 in both arms) |

The 6-stream dip noted since VTP14 is gone: throughput now rises with every stream from 4 to 8. The 1-stream cost is
the token guard at M = 4 (one stream at depth 3), consistent across starts.
