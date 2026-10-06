# 1341_meta_subset_mirrored

Promotion record (QFP18 lightweight evidence-reuse tier, pin b11402 / d89651a7). Mechanism is in SUMMARY.md. This is
a memory-layout change of the tensor split, not a speed patch: with `BIGCHERRY_META_SUBSET_MIRROR=1` the Qwen4Exp
lightning-indexer cache (`cache_idx_*`) is a full copy on the devices that run attention and absent on a device whose
attention share is zero (1303's `BIGCHERRY_ATTN_TS`), where it was a full copy on every device. It is promoted because
the output is proven identical and nothing gets slower. First step of MSM03; the context-shaped compute masks follow.

## Evidence

Flash-Next UD-IQ4_XS, production topology (2x RX 7900 XTX gfx1100 + R9700 gfx1201 tensor split, attention and KV on
the XTXs: `BIGCHERRY_ATTN_TS=1,1,0`; MTP drafter on the RX 6900 XT), f16 KV, build `b-metamem-b11402f2`
(production set + 1339 + 1340 + 1341, the 1340 flag off).

- Memory, per device, from 1339's report (`BIGCHERRY_META_MEM=1`, run `metamem-b11402f2`), indexer cache in MiB on
  XTX / XTX / R9700: ctx 49152: 288 / 288 / 288 with the flag off, 288 / 288 / 0 with it on; ctx 245760:
  1440 / 1440 / 1440 off, 1440 / 1440 / 0 on. The same in the production row split and in the expert-split layout
  (1283). Nothing else in the report changes.
- Identity: greedy text identical with the flag on and off in both layouts at both contexts (one md5 per layout),
  and in the ABBA below at 8K and 98K depth. Probe distributions (`flash-fidelity.sh`, 24 probes, 8K depth, ctx
  245760): flag on against flag off top-1 24/24, TV mean 0.0000, max 0.0000 - the same as flag off against itself.
- No regression (`tools/lab/flash-next/queue-env-ab.sh`, run `msm03-f2`, ctx 245760, ABBA on one binary,
  A = production, B = flag on):

| Depth | Prefill A (t/s) | Prefill B (t/s) | Decode A (t/s) | Decode B (t/s) | Acceptance A / B |
|---|---|---|---|---|---|
| 8K | 1060.9 / 1048.3 | 1070.2 / 1070.3 | 84.5 / 84.7 | 84.2 / 84.7 | 349/485 / 348/488, 349/485 |
| 98K | 972.6 / 989.1 | 991.1 / 990.1 | 58.1 / 58.3 | 58.3 / 58.2 | 346/493 / 346/493 |

- Activation: the report line for the indexer cache on the zero-share device goes to 0.0 MiB only with the flag on.
- Mechanics: `tools/tests/patch/test_1341_meta_subset_mirrored.py` and patch-lint.

## Native llama.cpp comparison

Native llama.cpp b11402 mirrors the indexer cache on every device of the tensor split and has no attention split
(1303) at all, so there is no native counterpart; with the flag off this build behaves as before. Speed is unchanged
against the BigCherry production build, whose standing against native is recorded with the patches that produce it
(see 1334's README). The gain is memory: 1,440 MiB on the R9700 at the full context.

## What is not claimed

- Only `cache_idx_*` is covered. The context-shaped attention inputs (KQ mask, kpool tables, QSA masks) are compute
  tensors and still cost every device their share of the common compute arena (MSM02 / later steps of MSM03).
- The flag only acts when the attention split leaves a device with a zero share; with attention on every device
  nothing changes.
- Measured on Flash-Next only (the indexer cache exists for the Qwen4Exp family); gfx1100 + gfx1201 in one split.
