# 1250_nro01_allreduce_q8_wire

**Status:** untested
**Plan item:** NRO01/NRO02

## What it does

Extends `1252_nro03_allreduce_p2p_provider` with Q8_0 transport using the codec and finish primitives owned by `1272_ar_host_compressed_wire`. When P2P is available, 1272's Q8 copy-engine route uses peer copies; otherwise it retains 1272's host-staging fallback. PNRO02 optionally folds a following mirrored F32 ADD into the same shared finish.

## Ownership

1250 no longer defines `GGML_CUDA_AR_WIRE`, Q8 quant/dequant kernels, or duplicate finish kernels. It requires 1272 and reuses those definitions directly. The old duplicate implementation and its private Q8 threshold/profile plumbing are removed rather than retained as compatibility paths.

## Why

Two-GPU `-sm tensor` decode can reduce PCIe bytes with Q8_0 while P2P avoids host staging when the provider probe succeeds. Residual fusion removes the separate ADD launch when its strict graph/type checks pass.

## Upstream

Nasone commit `e06dcf6300718227cb8cfda9e61fb12ccb693418`, migrated from a monolithic fork diff to the BigCherry 1252 transport + 1272 codec ownership split.
