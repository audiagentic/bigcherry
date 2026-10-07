# 1345_moe_ids_multiwarp

Promotion record (QFP18 lightweight evidence-reuse tier, pin b11402 / d89651a7, 2026-10-07). Mechanism is in
SUMMARY.md: for batches of 128 tokens and more the MoE routing helper runs 8 warps per expert where the native helper
runs one, with byte-identical row lists. On by default; `BIGCHERRY_MOE_IDS_MULTIWARP=0` restores the native helper.

## Evidence

Flash-Next UD-IQ4_XS, production topology (2x RX 7900 XTX gfx1100 + R9700 gfx1201 tensor split, MTP drafter on the
RX 6900 XT), f16 KV, ub512, build `b-metamem-mw3` (production set + 1345).

- Gate (prefill kernel profile of the production build, run `gate0-d24576`): `mm_ids_helper<10>` 3.23 s of kernel
  time over the three target cards in a 38.7K-token fill, about 3.8% of their kernel time.
- Activation (`BIGCHERRY_PATCH_TRACE=1`): `used=10 experts=512 tokens=512 warps=8 inverse=1` in the production row
  split (`metamem-mw3`), `used=10 experts=158 tokens=512 warps=8 inverse=1` with expert-parallel on (1283, local
  expert ids from 1281; `mw3-ep-ab2`).
- Op level (`tools/lab/flash-next/mmid-range-test.sh`, run `mw3-optest`): the range op against the CPU reference,
  0 failures in every band on CPU and on the GPUs, including the 300-token band that takes the new helper.
- Identity: greedy text identical with the multi-warp helper (A) and the native helper (B) in every ABBA below (one
  md5 per depth and layout); probes (`flash-fidelity.sh`, 24 probes, 8K): top-1 24/24, TV mean 0.0000, max 0.0000.
- Speed (`tools/lab/flash-next/queue-env-ab.sh`, ABBA on one binary, A = multi-warp / B = native, prefill t/s):

| Layout, ctx | Depth | Prefill A | Prefill B | Gain | Run |
|---|---|---|---|---|---|
| production, 245760 | 8K | 1077.9 / 1120.9 | 1095.5 / 1098.5 | not separated | `mw3-ab` |
| production, 245760 | 8K | 1121.4 / 1120.6 | 1100.6 / 1099.1 | +1.9% | `mw3-ab2` |
| production, 245760 | 24K | 1111.8 / 1120.5 | 1096.5 / 1095.1 | +1.8% | `mw3-ab` |
| production, 245760 | 49K | 1084.5 / 1101.5 | 1075.9 / 1075.9 | +1.6% | `mw3-ab2` |
| expert-parallel, 49152 | 8K | 1139.7 / 1138.8 | 1114.7 / 1117.1 | +2.1% | `mw3-ep-ab2` |
| expert-parallel, 49152 | 24K | 1139.8 / 1140.5 | 1116.4 / 1116.2 | +2.1% | `mw3-ep-ab2` |

  Complete separation (n = 2 per arm) in five of the six comparisons; in the first 8K comparison the first run of the
  pair was low (1077.9), the repeat separates. Decode is unchanged in all of them (the helper only takes batches of
  128 tokens and more).
- Offline: package tests (apply, idempotence, the per-warp scan repeats the native lines, composition after 1281,
  fail-closed), patch-lint, production composition check.

## Native llama.cpp comparison

Arm B is native llama.cpp b11402's `mm_ids_helper`, unmodified, so the ABBAs are the native helper against the
multi-warp helper inside one BigCherry binary: +1.6% to +2.1% prefill with identical output. No separate run against
a fully native binary was made for this patch.
