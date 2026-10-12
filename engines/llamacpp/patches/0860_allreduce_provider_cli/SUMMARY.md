# 0860_allreduce_provider_cli

**Status:** validated
**Plan item:** PGC04

## What it does

Adds `--allreduce <auto|ccl|host|adaptive|p2p|root3|butterfly>` and `--allreduce-wire <native|q8>` so the multi-GPU AllReduce implementation is chosen by CLI argument instead of the GGML_CUDA_ALLREDUCE / GGML_CUDA_AR_WIRE environment variables.

## Why

Several AllReduce implementations (stock NCCL/RCCL, host, butterfly, plus 0840/1244/1252 providers) need to be selectable per run without rebuilding. Unknown or unsupported provider/wire combinations are a startup error; `auto` never silently selects p2p.

## Upstream

BigCherry-original base framework; no promotion gate.

## PA44-B owned definitions

The three file-scope configuration/helper blocks now live in BigCherry-owned include files beside their translation
units: `ggml/src/ggml-cuda/bc-allreduce-provider-cli.cuh`, `common/bc-allreduce-provider-cli.h`, and
`tools/llama-bench/bc-allreduce-provider-cli.h`. The option registration, parse/apply hooks, backend proc export,
state, defaults and behavior are unchanged.

## Identity proof

The moved blocks contain 142 physical lines / 130 non-blank whitespace-stripped lines,
preserved exactly in the owned files. Replacing the three new include lines with those files reconstructs the
pre-refactor composed source; the only multiset delta is the three include lines (owned-definition multiset
130/130, include delta +3).
