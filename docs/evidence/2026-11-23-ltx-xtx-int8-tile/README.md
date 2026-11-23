# gfx1100 IQ8R tile, K-depth and prefetch investigation

**Date:** 2026-11-23
**Status:** scratch investigation; no kernel promotion or shipping
**Target:** one gfx1100 RX 7900 XTX with a separate GPU lease for each measurement arm.

## Decision

**Do not promote any variant.** The only consistently positive experiment, K128, improved the isolated prefill FFN sweep by an average **2.22% (up)** and **2.47% (down)** across M=2048, 6144 and 13376, but it did not reach the required >5% video-FFN gate. An independently leased `rad-kbench` run at M2048 measured only **+0.09%** on video FFN up and **+0.34%** on video FFN down (versus the stock kernel); the 40 credible large-case comparisons showed just **+0.156% geometric mean**. Hence even K128 is not a robust full-engine optimization.

The managed production rewrite remains unchanged. No full-plugin build or 2048-token one-card issued-block timing was undertaken because no candidate cleared the gate.

## Isolated timing results

Six experiments changed only one of wave count, strip depth, tile dimensions or weight staging relative to the existing native IQ8R implementation. All reported throughputs are logical operations (2MNK), not a direct comparison to the plain-int8 rocBLAS operation, which does not perform the same scale folding.

| Variant | FFN up mean vs stock | FFN down mean vs stock | Disposition |
|---|---:|---:|---|
| Stock 128×128, 8 waves, K64 | baseline | baseline | retained |
| 128×128, **4 waves** | −46.34% | −45.13% | reject: severe register spill |
| 128×128, **K32** | −61.88% | −63.45% | reject: severe register spill |
| 128×128, **K128** | **+2.22%** | **+2.47%** | below threshold; don't promote |
| **256×128**, 16 waves | −15.67% | −22.37% | reject: doubled activation tile, doubled LDS |
| **128×256**, 8 waves | −41.00% | −41.32% | reject: spills and N=384 unsupported |
| Weight operand double-buffered through LDS | −24.87% | −26.16% | reject: extra LDS traffic/footprint |

Means above are arithmetic means of the three per-M speedup percentages, where each percentage is 100×(stock time / variant time − 1). The exact **63 timed cases** at M=2048, 6144 and 13376 for each of FFN-up (N30720,K3840), FFN-down (N3840,K15360) and attention projection (N4096,K3840) are in `timings.csv`. Every arm used the same standalone comparator's `--perfw8 N K M 6 --cold 256` harness with 256 MiB cold-data rotation, one gfx1100 and separate GPU leases.

For K128, the nine-shape timing sweep's video FFN gains were:

| M | Up stock → K128 (TOPS) | Up change | Down stock → K128 (TOPS) | Down change |
|---|---:|---:|---:|---:|
| 2048 | 100.9 → 102.7 | +1.77% | 98.8 → 99.6 | +0.74% |
| 6144 | 105.7 → 108.2 | +2.40% | 101.0 → 104.4 | +3.46% |
| 13376 | 101.2 → 103.7 | +2.49% | 104.9 → 108.3 | +3.21% |

The separate `rad-kbench` baseline versus K128 comparison ran **369 timed cases in each arm**, with 159 matched IQ8R shape rows and zero outputs checked by that timing-only tool. At the actual model's M2048 video FFN shapes: up stock **5573.54 µs**, K128 **5568.31 µs** (+0.09%); down stock **2829.53 µs**, K128 **2819.84 µs** (+0.34%). Unlike the isolated harness, the `rad-kbench` LTX fixture does not cover exact M6144 or M13376, so no estimates have been substituted.

## Compiled resource analysis

Values below were read from the `gfx1100` device code object AMDGPU kernel metadata for the IQ8R `gemm_8a8_prefill_kernel<true,0,256>` instantiation, not inferred from source. The values are in `resources.csv`.

| Variant | Waves/block | VGPR | VGPR spills | Per-thread scratch | LDS/block | LDS-limited blocks/WGP |
|---|---:|---:|---:|---:|---:|---:|
| Stock | 8 | 240 | 4 | 20 B | 19,456 B | 6 |
| 4-wave | 4 | 240 | 231 | 844 B | 19,456 B | 6 |
| K32 | 8 | 240 | 331 | 1,104 B | 11,264 B | 11 |
| K128 | 8 | 240 | 3 | 16 B | 35,840 B | 3 |
| 256×128 | 16 | 240 | 4 | 20 B | 38,912 B | 3 |
| 128×256 | 8 | 240 | 217 | 804 B | 19,456 B | 6 |
| Weight LDS | 8 | 254 | 0 | 0 B | 60,416 B | 2 |

The last column is **an LDS-only capacity ceiling**, computed as floor(128 KiB WGP LDS / LDS/block). It is *not* achieved occupancy: VGPR and thread/WGP scheduling can reduce it, and active-wave occupancy was not profiled. All code objects use wavefront size 32. 4-wave, K32 and 128×256 are spill-bound; weight staging consumes 60 KiB LDS per block. The compiler did not find a resource advantage sufficient to outperform the stock layout.

## BF16 byte-parity and tile constraints

The scratch numerical comparator invokes the stock and variant plugin with identical quantized activations, weights and output buffers and compares the complete BF16 output bitwise. **4-wave, K32, K128, 256×128, and double-buffered LDS weights each passed 17/17 shape comparisons with zero differing output elements**, including the large tiled M2048,N3840,K4096 output (7,864,320 BF16 values). The scale-fold ordering remained unchanged.

The **128×256** variant produced identical BF16 outputs on the 16 supported test cases reached, but its new N=256 tiling fails the existing N=384 tiled case at M300,K5120 (launch rejected with return −1). It therefore **does not pass the full numerical gate** and is not a drop-in replacement. Non-IQ8R instantiations and the changes to shared wave-scale assumptions were not regression-qualified; this is an additional reason not to ship the scratch tile rules.

See `parity.csv` for variant counts and the N constraint.

## Requested features already present upstream

* **Global-load prefetch:** The stock `tile` body issues the next activation and weight `stage8_load` *before* running the current strip's WMMA, using alternating registers and an unconditional last-strip request. This already is a one-strip-ahead, PGR1-like global prefetch. A no-op renaming is not a distinct performance arm.
* **Wider activation loads:** Each `Stage8` activation transfer uses **two 16-byte `uint4` vector loads**, which already satisfies the requested dword4 access width. The staged LDS write subsequently uses 8-byte granules for bank organization. No additional dword4-width change was necessary.
* **Scale folding:** The stock per-256-K fold was retained in all tested variants. The part-1 native-fold experiment, which changes scale layout, was deliberately not combined with any geometry arm to keep attribution one-variable-at-a-time.

## Limits and recommendation

These results cover isolated IQ8R GEMM operation timing and scratch-parity tests, not full TP3 inference. GPU leases were acquired/released separately; only one gfx1100 card was visible. The available toolchain metadata permits an LDS residency **bound** but does not establish achieved occupancy. There is no evidence for a >5% video FFN improvement; leave the existing managed rule untouched and do not allocate a full-plugin / issued-block gate to these candidates.

**Evidence:** `timings.csv`, `resources.csv`, `parity.csv`. Scratch source and full raw timing/build logs remain outside the repository, so no host-specific paths or binaries were committed.
