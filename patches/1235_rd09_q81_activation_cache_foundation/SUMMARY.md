# 1235_rd09_q81_activation_cache_foundation: Per-graph Q8_1 activation-quantization cache, foundation only (RD09 stage 1)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD09

## What it does

Adds a generation-scoped Q8_1 quantization cache implementation (stable-slab allocator, find/reserve/publish API, GGML_HIP_Q8_1_CACHE_MODE env gate) as pure additions to three vendor files, and gives ggml_backend_cuda_context an opaque pointer to own one instance; no caller references it yet.

## Why

Reusing one Q8_1 quantization of an activation across every MMVQ consumer that needs it within a graph avoids re-quantizing per node, but this foundation stage deliberately adds no caller so it can be reviewed and tested with zero behavioral risk before stage 2 wires it into ggml_cuda_mul_mat_vec_q().

## Upstream / provenance

Reimplemented (not verbatim-ported) from stew675-rdna-boosts fork commit 299f6eaf7 (https://github.com/stew675/llama.cpp), with two required adaptations found in design review: a cache key that includes view offset (the fork's key collides on same-root different-offset views), and never-relocated slabs instead of a growing arena (a relocating arena would corrupt baked-in pointers in captured HIP graphs).
