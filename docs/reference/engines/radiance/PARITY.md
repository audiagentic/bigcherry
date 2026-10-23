# Radiance on two RX 7900 XTX against RDNA4: where it stands and how to close the gap

Assessment of the goal "optimise radiance's RDNA3 path to match or exceed RDNA4 over the two XTX on dev,
Qwen3.8-27B and Flash-Next first" (2026-10-12/13). Evidence and run names: [engines/radiance/README.md](../../../../engines/radiance/README.md).

## Where it stands

| Model, two XTX (libr3) | Measured | Target | Status |
|---|---|---|---|
| 27B decode | 46.5-47 tok/s (no draft); 131-142 with drafter | R9700 38.3 (114.8 with drafter) | exceeded |
| 27B prefill (7.2K) | ~2,250 tok/s, wht6 wire, native attention; ~2,380 with the 4-bit wire | R9700 3,104-3,214 | 72-76% |
| Flash-Next prefill | 1,150-1,320 tok/s (4K chunks, prefill stager, exact wire, n-gram in RAM) | llama.cpp three cards 1,012-1,469; radiance on two R9700 ~6,500 | at llama.cpp parity; far from two R9700 |
| Flash-Next decode, MTP 3, prose | 53.6 / 51.6 tok/s (3.3K / 6.8K) | llama.cpp 72.2 / 55.6 | 74-93% |

What moved the numbers: native gfx11 prefill attention (+5%), the lossy wht6 wire on the 27B (+35%), the prefill
stager armed (`--max-num-seqs 6`, ~2x Flash-Next prefill) with engine patch rad0001 fixing its shutdown hang, 4K prefill
chunks (+50-60% at short prompts), the n-gram table off disk (+11% decode, 1.6-2.1x prefill), a smaller expert pool
(+15% long-prompt decode). What did not: RR08 GEMM forms, the pipelined all-reduce (-6%), the native delta-net scan
(wrong output), the 2-bit draft head (not the cause of low acceptance).

## Why the gap remains

1. 27B prefill: the tiled MXFP4 GEMM runs at ~100 of a measured 142 TFLOP/s per card (70% of prefill), and the
   cross-card sum costs ~20%: it is link time for its bytes (7.2 MB at ~3.5 MB/ms, out then in), on PCIe 4.0 x8
   with no P2P. Two XTX have more FP16 throughput than one R9700 but pay that sum on every layer.
2. Flash-Next: 38 GiB of experts a rank against 24 GB cards. A third of each rank's experts live in host memory and
   cross the x8 link at prefill; the R9700 reference (two of them, 64 GB, x16, P2P-capable topology) holds far more.
3. MTP acceptance 0.63 against llama.cpp's 0.79 at short context: the container's fp8 MTP layer as radiance runs it
   (same on the R9700 with libr4d, same with an int8 head), not libr3.

## How to close it, ranked

1. R9700 expert helper (Flash-Next prefill, decode at long context): keep the cold experts resident on the R9700 and
   compute them there; only activations and partials cross host memory. Engine patch rad0002 (hand-off point) is
   merged; the helper process is the remaining step. Expected 1.5-3x prefill per the measured pooled-vs-resident
   GEMM (47 vs 5 ms a 2K chunk).
2. 27B cross-card sum: fewer bytes (4-bit wire measured +5.6% at a quality cost, owner's call) or overlap with the
   next GEMM (engine scheduling change). Pipelining inside one kernel was measured slower.
3. 27B GEMM: 100 -> ~125 TFLOP/s needed with the sum halved to reach ~3,000 tok/s.
4. MTP acceptance: re-convert the container with a higher-precision MTP layer (needs the 335 GiB source checkpoint).
5. Deeper prefill stager / 8K chunks with a lower stager threshold (Flash-Next long prompts: 8K chunks gave
   1,386 tok/s at 7.2K).

Infrastructure for all of it is in place (RAD01): every experiment above is now a patch package, component setting or
profile, built by `bigcherry build` and measured by `ab-benchmark` with quality gates.
