# 0860_allreduce_provider_cli

**Status:** untested
**Plan item:** PGC04

## What it does

Adds `--allreduce <auto|ccl|host|adaptive|p2p|root3|butterfly>` and `--allreduce-wire <native|q8>` so the multi-GPU AllReduce implementation is chosen by CLI argument instead of the GGML_CUDA_ALLREDUCE / GGML_CUDA_AR_WIRE environment variables.

## Why

Several AllReduce implementations (stock NCCL/RCCL, host, butterfly, plus 0840/1244/1252 providers) need to be selectable per run without rebuilding. Unknown or unsupported provider/wire combinations are a startup error; `auto` never silently selects p2p.

## Upstream

BigCherry-original base framework; no promotion gate.
