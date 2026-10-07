# 1295_qsa_gather_decode

**Status:** validated
**Plan item:** RNX02

## What it does

For batches of at most 8 tokens, Qwen4Exp QSA attention gathers each token's selected KV cells (padded to a
multiple of 256 with a masked sentinel) and runs flash attention over them, instead of masking the full
cache. Prefill keeps the masked path. On by default; `BIGCHERRY_QSA_GATHER=0` restores the masked path.

## Why

On HIP the QSA mask does not reduce work: decode attention per token grew 7.6x from 10K to 80K context
(target 0.44 -> 3.36 ms/token, draft 0.18 -> 1.41 ms/token summed over GPUs). Upstream's sparse attention is
NVIDIA-only. See RNX02 review RV4213. Compared against 1296 (HIP port of the upstream sparse kernel path).

Validation: decode ms/step at 10K/80K/160K, greedy parity without MTP at 10K and 80K against the masked path.

## Results (2026-10-03)

Accuracy: greedy text differs from the masked path, but vs an f32 CPU reference (FA off, no tensor split) the
gathered path is closer (max |dp| over 8 decoded tokens at 10K: gathered 0.056, masked FA 0.073; masked vs
gathered 0.023). The masked HIP tile path rescales its half-precision accumulators over every KV tile, so its
error grows with n_kv; a 2-GPU run (one KV head per rank) showed the same difference, ruling out the meta split.
v1 (no threshold): 10K +3% ms/step, 80K -5.7%, no-MTP decode at 80K +16%.
v2 (BIGCHERRY_QSA_GATHER_MIN=32768, cache view reshaped directly, sentinel rows clamped), full ABBA on the
deployment candidate (flashnext-gather-v2-ab): 10K identical (threshold); 80K 56.5/55.6 -> 53.8/53.7 ms/step
(-3.7%, +7% t/s); 160K (192K tier) 70.3 -> 61.7 on one pair (-12%), the other 1295 arm ran out of memory on the
R9700 at runtime (that tier has ~0.3 GB headroom). Scope: tiers with headroom (96K / 144K / 160K-q8_0 at 4,4,3);
in the 192K tier cap context near 184K or lighten the R9700 share before enabling.
