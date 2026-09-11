# 1253_nro04_gfx1100_bf16_chunked_gdn

**Status:** untested
**Plan item:** NRO04

## What it does

Adds a disabled gfx1100 BF16/WMMA eligibility scaffold on top of RD50. It does not yet include the large WMMA kernel body or live dispatch.

## Why

Current nasone GDN work includes a real gfx11 path that BigCherry's RDNA3.5-only RD50 does not cover. The kernel body must be ported only after gfx1100 WMMA fragment fixtures are ready.

## Upstream

Local staged adaptation of nasone commit `4169fbbf50d24beb6d269a2350e7f780b85369e6`.
