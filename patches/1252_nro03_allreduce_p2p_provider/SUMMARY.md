# 1252_nro03_allreduce_p2p_provider

**Status:** rejected
**Plan item:** NRO03

## What it does

Opt-in (`GGML_CUDA_AR_P2P=1`) two-GPU peer-to-peer copy path for the internal HIP AllReduce, replacing host staging when a startup probe proves both directions copy correctly.

## Why

Host-staged AllReduce crosses PCIe twice; direct peer copies cross once.

## Upstream

Port of nasone `7c5bb5cb`, adapted: no fixed issuer device, source-current push per direction, content-checked startup probe.

## Rejected (2026-10-08, pin b11474)

P2P AllReduce was re-tested on the lab topology on 2026-10-07 and faults; it does not compose at b11474 either.
Experiment `ar-p2p` and its lab script removed.
