# 1332_qsa_token_chunk

Promotion record (QFP18 lightweight evidence-reuse tier, pin b11402 / d89651a7). Mechanism is in SUMMARY.md; this
file records why the patch is promoted into `[patch-set.validated-enhancements]`. The patch is an enabler: it is off
unless `BIGCHERRY_QSA_CHUNK` is set, and it exists so that `-ub 1024` fits the 240K f16 deployment. It is not set by
the `flashnext` profile because it only pays together with the server's `-ub 1024`, which a profile cannot set; the
deployment sets both.

## Evidence

Brutus, 2x 7900 XTX + R9700 tensor split, 6900 XT MTP drafter, ctx 245760, f16 KV, `flashnext` profile,
`BIGCHERRY_FA_SPARSE=1` (1334) in both arms, build `b-chunk-b11402d`, 2026-10-06
(`tools/lab/flash-next/queue-chunk-ub1024.sh b11402d 1 81920 163840`). One binary; A = `-ub 512`, B = `-ub 1024`
with `BIGCHERRY_QSA_CHUNK=256`; ABBA.

- Mechanics: offline patch test (`tools/tests/patch/test_1332_qsa_token_chunk.py`) and patch-lint pass on pin b11402.
- Activation: the ub 1024 arm loads and runs at 240K f16 only with the chunked masks (the unchunked ub 1024 compute
  buffer peak is the two QSA masks, 484 + 480 MiB, next to the 480 MiB kq_mask input - 1331 peak trace, QFP17).
- Prefill, complete separation at both depths:

  | Prompt | ub 512 (t/s) | ub 1024 + chunk 256 (t/s) | change |
  |---|---|---|---|
  | ~99K tokens | 965.1 / 980.3 | 1011.7 / 1013.7 | +4% |
  | ~202K tokens | 838.5 / 847.7 | 874.3 / 874.2 | +4% |

- Decode: tokens per second differ with the generated text (99K: 57.3 / 57.9 vs 59.1 / 59.9; 202K: 46.9 / 46.9 vs
  44.0 / 43.7) because draft acceptance differs (202K: 345/495 and 344/498 vs 334/530 and 332/536 accepted/steps).
  Time per decode step is unchanged: 99K 16.8 vs 16.7 ms/step, 202K 22.1 vs 22.0 ms/step.
- Fidelity (`flash-fidelity.sh`, 24 next-token probes over one cached 99.3K-token fill, no MTP): ub 1024 + chunk
  against ub 512 top-1 agree 23/24, TV mean 0.092, max 0.388; ub 512 twice is identical. That is inside the distance
  the dense GPU path has from a CPU f32 run on this model (TV mean 0.146, see 1334 README), so greedy identity is
  not required for the same reason as 1334: a different batch shape changes summation order, which the discrete QSA
  top-k selection amplifies.
- Without MTP the fill in the fidelity run was slower in the ub 1024 arm (1180.1 vs 1201.8 / 1203.6 t/s, one sample):
  the gain is for the production configuration, where the drafter also processes the prompt. A no-MTP deployment
  should stay on ub 512.

## Native llama.cpp comparison

Native llama.cpp b11402 cannot load the 240K f16 deployment, so there is no native arm at these depths. The base
this patch is measured on (production set with 1334) is compared with native at 64K context in the 1334 README:
63.1K-token prefill 1029.4 / 1030.1 t/s against 907.6 native. This patch adds its +4% on top of that base.
