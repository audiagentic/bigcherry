# 1254_nro05_gdn_mtp_prefix_tail

**Status:** untested
**Plan item:** NRO05

## What it does

Adds the exact eligibility/prefix-length scaffold for chunking the history before the final K MTP snapshot tokens. It does not redirect live GDN execution yet.

## Why

A fully chunked K>1 recurrence does not automatically materialize all speculative snapshot states. Prefix chunking can preserve the final K sequential state transitions.

## Upstream

Local child of NRO04 informed by nasone commit `4169fbbf50d24beb6d269a2350e7f780b85369e6`.
