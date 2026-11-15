# RAD53: gfx1100 fp8 KV attention - direct E4M3 conversion and no late K prefetch in the native prefill

2026-11-15. Layout `qwen27b-tp3-int8-556-long` (TP3: 7900 XTX, 7900 XTX, R9700; split 5,5,6; fp8 KV cache). Engine
argv from the stored manifest; the RAD53 arm adds only `--radiance-home` pointing at a libr3 built from this branch
(RAD52 already on main).

## Change

1. `r3_e4m3_f16.h`: E4M3 is exact in binary16, so 8-bit KV widens to f16 by shift and add instead of the software
   f32 conversion; the attention kernels use it. Exhaustive host test (`r3_e4m3_f16_test`, card-free): all 256 codes
   and every packed pair against an independent reference, and the old emulation restated and checked against it.
2. The native gfx1100 prefill attention drops `O_KLATE` (tile ti+1's K fetched during tile ti's last matrix tile), for
   the plain and the KV-split launch. With the native fragments and accumulator live, KLATE took the fp8 and k6_v4
   kernels' spill from 336 / 376 to 816 / 832 bytes a lane (`-Rpass-analysis=kernel-resource-usage`; 240 VGPRs and
   6 waves either way). bf16 and fp8_v4 never take KLATE and compile unchanged.

## Correctness

- Identity against RAD52 alone (main before this change), fp8 `-long` argv, long corpus (5 documents, 276,900
  positions, one sequence, no speculation): perplexity equal to every printed digit (3.0921531658), top-1 agreement
  1.0, KLD max 2.7e-8 (the tool's float floor). No arithmetic changes.
- Against stock (before RAD52 and RAD53): KLD mean 0.00461, p99 0.0104, top-1 98.30%; noise floor (stock, serial
  prefill) 0.00416, p99 0.0103, top-1 98.36%. The difference is RAD52's split summation order (see the RAD52
  evidence), at the floor.
- `r3_selftest` on gfx1100: attention prefill rows for all four caches (229 / 240 / 240 / 235 ok) and the KV-split
  rows (2382 ok), 0 fail.
- Stress: 12 fresh starts, each 4- and 8-stream benches and 4 decoding streams beside a 60k-token prompt: 12/12 up and
  alive, 0 fault lines, every bench ok.

## Performance

Kernel, `r3_selftest --perfpaged` at 100k keys on one 7900 XTX (two KV heads), ms per launch, mean of 2 interleaved:

| cache | rows | stock | RAD52 | RAD52 + RAD53 |
|---|---|---|---|---|
| fp8 | 512 / 1024 / 2048 | 40.5 / 79.6 / 119.1 | 25.4 / 53.5 / 119.6 | 10.2 / 20.3 / 55.7 |
| k6_v4 | 512 / 1024 / 2048 | 43.8 / 86.7 / 114.5 | 25.9 / 52.6 / 114.1 | 9.8 / 20.3 / 51.3 |
| fp8_v4 | 512 / 1024 / 2048 | 21.5 / 44.6 / 53.2 | 9.9 / 20.6 / 52.5 | 9.9 / 20.2 / 53.0 |
| bf16 | 512 / 1024 / 2048 | 23.6 / 48.6 / 64.1 | 12.2 / 25.5 / 63.8 | 12.1 / 25.4 / 65.3 |

Engine, ABBA (2 starts per arm), fp8 `-long`, stock against RAD52 + RAD53:

| metric | stock | RAD52 + RAD53 | change |
|---|---|---|---|
| TTFT 100k tokens | 65.18 s (1,540 t/s) | 45.45 s (2,208 t/s) | -30.3% |
| TTFT 60k | 31.72 s | 24.77 s | -21.9% |
| TTFT 30k | 13.03 s | 11.34 s | -13.0% |
| TTFT 8k | 2.96 s | 2.89 s | -2.4% |
| decode, 1 / 8 streams (chunks/s) | 32.92 / 79.27 | 33.39 / 79.93 | +1.4% / +0.8% |
| 100k prompt beside 4 decoders: long TTFT | 73.35 s | 53.19 s | -27.5% |
| decode beside it (chunks/s) | 4.19 | 5.83 | +39% |
| longest decode gap | 2025 ms | 1214 ms | -40% |

RAD52 alone measured -18.2% at 100k on the bf16 config, where RAD53 changes nothing. fp8 TTFT at 60k is now 24.77 s
against 24.11 s for bf16 with RAD52 (+2.7%, separate runs on the same libr3 code for bf16).

Not measured here: the fp8 decode step at 60k (the item's other premise, 39.4 vs 32.6 ms bf16); the direct conversion
is on the decode K read too.
