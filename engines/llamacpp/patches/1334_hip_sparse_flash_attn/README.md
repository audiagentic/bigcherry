# 1334_hip_sparse_flash_attn

Promotion record (QFP18 lightweight evidence-reuse tier, pin b11402 / d89651a7). Mechanism is in SUMMARY.md; this
file records why the patch is promoted into `[patch-set.validated-enhancements]`. The path is on by default
(`BIGCHERRY_FA_SPARSE=0` is the off switch): it is taken only when a graph sets `n_kv_max` on its attention, which
only Qwen4Exp does today, so every other model runs the dense kernels exactly as before. The evidence below was taken
with the flag set explicitly, before the default changed.

## Evidence

Brutus, 2x 7900 XTX + R9700 tensor split, 6900 XT MTP drafter, ctx 245760, f16 KV, `flashnext` profile, ub 512.
Each comparison is one binary with the flag off (A) and on (B).

- Mechanics: offline patch test (`tools/tests/patch/test_1334_hip_sparse_flash_attn.py`) and patch-lint pass on pin
  b11402; builds for gfx1100, gfx1201 and gfx1030.
- Activation: `BIGCHERRY_PATCH_HIT patch=1334_hip_sparse_flash_attn n_kv=8192 n_queries=64 n_kv_max=512` under
  `BIGCHERRY_PATCH_TRACE=1` with the flag on, absent with it off (`sparseproof-b11402c-backend`).
- Kernel correctness: upstream `test-backend-ops` FLASH_ATTN_EXT cases with `n_kv_max` (a different random cell
  selection per query, the adversarial case for the per-tile union list) against the CPU backend: 18/18 on each of
  ROCm0-3 with the flag off and on, in the run that shows the marker (`tools/lab/flash-next/fa-sparse-backend-test.sh`).
- Prefill ABBA, complete separation (`sparsefa-b11402`, rebuilt `sparsefa-b11402b`):

  | Prompt | dense (t/s) | sparse (t/s) | change |
  |---|---|---|---|
  | ~99K tokens | 862.4 / 872.1 (rebuilt 862.7 / 871.0) | 978.7 / 982.5 (rebuilt 980.5 / 981.6) | +13% |
  | ~202K tokens | 658.2 / 663.5 | 846.1 / 847.8 | +28% |

- Decode: 24K MTP ABA 41.5 / 41.6 ms/step sparse against 41.2 - 41.4 dense - unchanged (single-query batches stay on
  the dense kernel: the WMMA sparse kernel needs 16 columns).
- Fidelity (`tools/lab/flash-next/flash-fidelity.sh`, `sparseproof-b11402c`): 24 next-token distributions of
  natural-text continuations over one cached fill, no MTP. D = dense, D2 = dense again, S = sparse, C = CPU f32.

  | Fill | pair | top-1 agree | TV mean | TV max |
  |---|---|---|---|---|
  | 38.7K tokens | D vs C | 22/24 | 0.146 | 0.633 |
  | 38.7K tokens | S vs C | 22/24 | 0.128 | 0.520 |
  | 38.7K tokens | S vs D | 20/24 | 0.127 | 0.624 |
  | 38.7K tokens | D2 vs D | 24/24 | 0 | 0 |
  | 99.3K tokens | S vs D | 20/24 | 0.110 | 0.609 |
  | 99.3K tokens | D2 vs D | 24/24 | 0 | 0 |

## Why greedy identity is not required

The sparse kernel sums the same cells in a different order, so its output differs from the dense kernel in the last
bits. Qwen4Exp then feeds attention outputs into a discrete top-k cell selection (the QSA indexer) in every later
layer, which turns last-bit differences into different selected cells. The dense GPU path already sits that far from
a CPU f32 run of the same weights (TV mean 0.146), and the sparse path is no further from f32 than the dense path is
(0.128), with the same top-1 agreement. The path is deterministic run to run. The kernel itself is verified against
the CPU backend by test-backend-ops.

The size of the dense-vs-f32 distance is a property of the model on this stack, not of this patch; it is tracked
separately (QFP28).

## Native llama.cpp comparison

`tools/lab/flash-next/queue-native-flash.sh b-sparsefa-b11402c b-native-b11402 native-sparse` with
`BIGCHERRY_FA_SPARSE=1` (2026-10-06): the production set with this patch on against native llama.cpp b11402 (source
`llama-native`, no patches; native compiles the sparse path out for HIP and runs the dense kernel). Flash-Next, 64K
context f16 (native cannot load the 240K f16 deployment), ABA.

| Depth | production + 1334 | native |
|---|---|---|
| 63.1K-token prefill (t/s) | 1029.4 / 1030.1 | 907.6 |
| decode at 8K (ms/step) | 37.0 / 36.9 | 50.4 |
| decode at 48K (ms/step) | 45.2 / 44.2 | 67.6 |

The decode gap is the existing production set (it measured 37.1 / 43.9 against 49.9 / 67.3 before this patch); the
prefill gap at this depth is +13% over native.

## Prefill-only gate (2026-10-10)

The sparse path is taken only for batches of at least `BIGCHERRY_FA_SPARSE_MIN_Q` queries (default 64, floor 5);
smaller batches keep the dense path and the tile shape they have with the flag off. Reason: on the production
Flash-Next build at 24K, four requests, switching the patch off raised MTP decode by 7.5% pooled (draft acceptance
57.0% -> 62.4%) for 1.3% less prefill. The sparse sums are no further from a CPU f32 reference than the dense ones,
but in decode and MTP verification batches the drafter agrees with them less often. Prefill batches still write
sparse-path values into the KV cache, so the change must be re-confirmed on hardware (ABBA and the four-request
decode check) before the decode cost is called closed.
