# 1344_dsv4_hc_grid_index

Promotion record (QFP18 lightweight evidence-reuse tier, pin b11402 / d89651a7, 2026-10-07). Mechanism is in
SUMMARY.md: the Qwen4Exp hyper-connection PRE / POST kernels take their coordinates from a 2-D / 3-D launch grid
instead of 64-bit `%` and `/` per output element. On by default; `BIGCHERRY_HC_GRID_INDEX=0` restores the flat kernels.
The kernels are Qwen4Exp-only ops (DSV4_HC_PRE / DSV4_HC_POST), so the default reaches no other architecture.

## Evidence

Flash-Next UD-IQ4_XS, production topology (2x RX 7900 XTX gfx1100 + R9700 gfx1201 tensor split, MTP drafter on the
RX 6900 XT), ctx 245760, f16 KV, ub512, build `b-metamem-pn2` (production set + 1343 with its flag off + 1344).

- Activation (`BIGCHERRY_PATCH_TRACE=1`, run `metamem-pn2`): `path=pre n_embd=2560 tokens=512` and
  `path=post n_embd=2560 hc=4 tokens=2`, once each.
- Identity: greedy text identical with the grid kernels (A) and the flat kernels (B) at 8K and 24K depth (one md5 per
  depth); probes (`flash-fidelity.sh`, 24 probes, 8K): top-1 24/24, TV mean 0.0000, max 0.0000.
- Speed (`tools/lab/flash-next/queue-env-ab.sh`, run `hcgrid-ab`, ABBA on one binary, A = grid / B = flat):

| Depth | Prefill A (t/s) | Prefill B (t/s) | Decode A (t/s) | Decode B (t/s) | Acceptance A / B |
|---|---|---|---|---|---|
| 8K | 1080.9 / 1092.3 | 1062.4 / 1075.7 | 83.7 / 85.4 | 84.4 / 85.2 | 348/488, 349/485 / 349/485, 350/482 |
| 24K | 1088.6 / 1090.6 | 1077.3 / 1073.4 | 75.4 / 76.0 | 75.7 / 75.7 | 345/498, 346/495 / 345/498, 346/495 |

  Prefill is higher with the grid kernels on every run at both depths (complete separation, n = 2 per arm):
  +1.6% at 8K, +1.3% at 24K. Decode is unchanged.
- Offline: package tests (apply, idempotence, the grid kernels contain no division and repeat the flat kernels'
  arithmetic text, composition after 1311, fail-closed), patch-lint, production composition check.

## Native llama.cpp comparison

The flat kernels of arm B are native llama.cpp b11402's `dsv4_hc_pre_f32` / `dsv4_hc_post_f32`, unmodified, so the
ABBA above is the native kernels against the grid kernels inside one BigCherry binary: +1.3% to +1.6% prefill with
identical output. No separate run against a fully native binary was made for this patch; BigCherry production's
standing against native is recorded with the patches that produce it.
