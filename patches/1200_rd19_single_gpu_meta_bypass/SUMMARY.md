# 1200_rd19_single_gpu_meta_bypass: Skip the Meta device wrapper when tensor-splitting a single GPU (RD19)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD19

## What it does

Uses the plain device instead of the Meta wrapper in llama_prepare_model_devices when n_devices == 1 (both the explicit-device-list and default-selection branches), leaving the multi-GPU Meta path untouched.

## Why

With one device, -s tensor still wraps the graph in a Meta device that splits it into extra subgraphs and compute calls even though no splitting is possible, adding launch overhead and clearing the Q8_1 quantize cache between subgraphs; the fork reports +1.4-1.8% tg64 from removing this.

## Upstream / provenance

Ported verbatim from stew675-rdna-boosts fork commit 3c48ecd63 (https://github.com/stew675/llama.cpp). Not merged into ggml-org/llama.cpp master.
