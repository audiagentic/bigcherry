# 1335_tiled_lightning_indexer

Promotion record (QFP18 lightweight evidence-reuse tier, pin b11402 / d89651a7). Mechanism is in SUMMARY.md; this
file records why the patch is promoted into `[patch-set.validated-enhancements]`. The path is on by default
(`BIGCHERRY_INDEXER_TILE=0` is the off switch); it only applies to 4-head QSA indexers with 8 or more tokens in the
batch, which today means Qwen4Exp prompt batches, so other models and single-token decode are unaffected.

## Evidence

Brutus, 2x 7900 XTX + R9700 tensor split, 6900 XT MTP drafter, ctx 245760, f16 KV, `flashnext` profile, ub 512,
sparse flash attention (1334) on in both arms. Build `b-ixtile-b11402f` (production set + this patch), run
`tools/lab/flash-next/queue-indexer-tile.sh b11402f 81920 163840`. One binary; A = default (tile on),
B = `BIGCHERRY_INDEXER_TILE=0`; ABBA.

- Mechanics: offline patch test (`tools/tests/patch/test_1335_tiled_lightning_indexer.py`) and patch-lint pass on pin
  b11402; the generated edits reproduce upstream commit 1b43d3116 apart from the off switch and the marker.
- Activation: `BIGCHERRY_PATCH_HIT patch=1335_tiled_lightning_indexer n_kv=256 n_batch=512` under
  `BIGCHERRY_PATCH_TRACE=1` with the tile on, absent with it off.
- Kernel correctness: upstream `test-backend-ops -o LIGHTNING_INDEXER` against the CPU backend, 192/192 on each of
  ROCm0-3 with the tile off and on, in the run that shows the marker
  (`tools/lab/flash-next/indexer-backend-test.sh`).
- Prefill, complete separation at both depths:

  | Prompt | vector kernel (t/s) | tiled (t/s) | change |
  |---|---|---|---|
  | ~99K tokens | 984.2 / 982.0 | 1003.9 / 1015.4 | +2.6% |
  | ~202K tokens | 847.9 / 846.7 | 888.1 / 902.8 | +5.6% |

  The gain grows with depth because the indexer's share does (6% of kernel time at ~99K, 11% at ~202K in the b11402
  sparse-path profile).
- Decode: single-token batches keep the vector kernel. Tokens per second differ with the generated text
  (~99K: 62.2 / 63.4 against 57.3 / 57.4 with acceptance 346/493, 348/487 against 333/532, 332/535); time per
  decode step is 16.7 - 16.8 ms in both arms at ~99K and 21.9 - 22.1 ms at ~202K.
- Fidelity (`flash-fidelity.sh`, 24 next-token probes over one cached 99.3K-token fill, no MTP): tile off against
  tile on top-1 agree 22/24, TV mean 0.086, max 0.421; tile on twice is identical. Inside the distance of the dense
  GPU path from a CPU f32 run on this model (TV mean 0.146, 1334 README). Not bit-identical for the reason upstream
  gives (keys pass through F16, different summation order) and, as for 1334, the discrete QSA top-k selection
  amplifies last-bit differences; greedy identity is therefore not required.
- No-MTP fill in the fidelity run: 1251.9 / 1252.8 t/s tiled against 1201.9 t/s vector (+4.2%).

## Native llama.cpp comparison

Native llama.cpp b11402 does not contain #29901 (it merged after the tag) and cannot load the 240K f16 deployment,
so there is no native arm at these depths. The base this patch is measured on (production set with 1334) is compared
with native at 64K context in the 1334 README: 63.1K-token prefill 1029.4 / 1030.1 t/s against 907.6 native. This
patch adds its gain on top of that base, and becomes redundant when the pin reaches a release that contains #29901.

## Superseded at b11474

Upstream llama.cpp merged the tiled lightning indexer as `1b43d3116` (#29901), the commit this patch backported. At
pin b11474 (b9acf138) the kernel and its dispatch are native; the patch's dispatch anchor no longer exists. Out of the
production set as of the b11402 -> b11474 bump (2026-10-08). What does not carry over: the `BIGCHERRY_INDEXER_TILE=0`
off switch and the activation marker.
