# 1254_nro05_gdn_mtp_prefix_tail

**Status:** untested
**Plan item:** NRO05

## What it does

For MTP (K > 1) prefill of one long sequence, runs 1253's BF16 chunked GDN on the first n_tokens - K tokens and the sequential kernel on the last K, so the K snapshot slots stay exact.

## Why

MTP models otherwise take the fully sequential GDN path for the whole prefill.

## Upstream

From nasone commit `4169fbbf50d24beb6d269a2350e7f780b85369e6` (block 02), BF16 path only.
