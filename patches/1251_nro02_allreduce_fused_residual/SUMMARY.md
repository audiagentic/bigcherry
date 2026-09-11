# 1251_nro02_allreduce_fused_residual

**Status:** untested
**Plan item:** NRO02

## What it does

Adds draft residual-capable finish kernels for exact/lower-precision and Q8 AllReduce output. Meta graph matching and node skipping are intentionally not wired yet.

## Why

A safe fused finish can remove the post-AllReduce residual ADD launch, but graph ownership must be proven before the ADD can be skipped.

## Upstream

Local atomic child of NRO01, informed by nasone commit `e06dcf6300718227cb8cfda9e61fb12ccb693418`.
