# Flash-Next speed, capacity and retrieval of the packed KV caches (VTP44)

## Summary
On Flash-Next both packed caches serve and run at the speed of the caches they sit between. bf16_fp8 meets the 1.3x
capacity bar (1.307x bf16); bf16_i8 misses it (1.283x) because Flash-Next carries a second, unquantised cache group.
A 9-question retrieval probe at about 48K tokens shows no loss for either packed cache. The R9700 helper arm was not run.

## Outcome
Two 7900 XTX, TP2 exact wire, one run per arm, 65,536 context, 2,048-token chunks, one request at a time except where
noted. Source: `qwen3.8-next-flash-hcbf16`, build `kv-bf16-packed` (rad0037, 38, 40, 41, 42).

| | bf16 | bf16_i8 | bf16_fp8 | fp8 |
|---|---|---|---|---|
| TTFT, 8K tokens | 7.46 s | 7.29 s | 7.68 s | 7.23 s |
| TTFT, 48K tokens | 44.5 s | 44.9 s | 47.1 s | 44.3 s |
| Decode, 1 stream (chunks/s) | 28.86 | 28.95 | 30.78 | 30.66 |
| Decode, 4 streams (chunks/s) | 67.06 | 67.39 | 68.30 | 67.78 |
| Needle hits (of 9, 48K tokens) | 8 | 9 | 8 | 7 |
| attention page (K+V, 4 tokens) | 48.00 KiB | 36.75 KiB | 36.00 KiB | 24.00 KiB |
| cache tokens in 3.02 GiB (fp8: 2.85 GiB) | 222,108 | 284,968 | 290,344 | 393,296 |
| vs bf16 | 1x | 1.283x | 1.307x | 1.77x |

- Against the original bar (decode and TTFT within 3% of fp8): bf16_fp8 decode is +0.4% (1 stream) and +0.8% (4
  streams); its 48K TTFT is 6.3% slower. bf16_i8 decode is 5.6% slower than fp8 on one stream and 0.6% on four; its 48K
  TTFT is 1.3% slower. The single-stream decode figures fall in two groups (bf16 and bf16_i8 near 28.9; fp8 and bf16_fp8
  near 30.7), so V as int8 costs about 6% on one-stream decode against V as e4m3 here. These are single runs; the run to
  run spread was not measured, treat differences under about 2% as noise.
- Capacity: the attention group shrinks 1.306x (bf16_i8) and 1.333x (bf16_fp8). Flash-Next also holds a QSA block-key
  group (3.00 KiB a page, 12 layers) that no packed layout touches, so the whole-cache ratio is 1.283x and 1.307x.
- The bf16_fp8 single-wave decode variant spilled 45 registers against at most 20 for the existing variants (device
  compile, device-compile-b.md); it did not cost decode speed here.

## Detail
Setup limits, all deliberate: the R9700 remote-MoE helper was not used (it needs a measured plan from the launcher's
measuring start; the manifest written here is unmeasured and the helper refuses it); the host pool is 14,336 MiB and
speculation is off with no n-gram table (its 47.7 GiB pinned copy did not fit beside the pool), as in the KLD runs.
Absolute speeds are therefore below production, and the arms are comparable with each other only.

Needle probe: a passphrase sentence at 10, 50 and 90% depth (three repeats each) in a context of about 48,000 tokens,
asked through the chat endpoint with thinking off, temperature 0. The misses are refusals, not wrong codes: bf16 and
bf16_fp8 each missed depth 0.9 repeat 1; fp8 missed depth 0.1 and 0.9 repeat 1. The same prompt is the one every arm
misses, so a prompt-level effect is likelier than a cache one; nine questions cannot separate arms that differ by one.
The probe is retrieval at one length, not a quality measure.

KLD for these arms: FLASHNEXT-KLD.md (inside Flash-Next's arithmetic-noise band, so it cannot judge KV error).

## Follow-up
Per-token expert-id dump (a generic engine option) to confirm or refute routing flips as the cause of the KLD floor;
the helper arm once a measured plan can be made; a longer or harder retrieval probe (100K, several needles) if the
retrieval claim is to carry weight.
