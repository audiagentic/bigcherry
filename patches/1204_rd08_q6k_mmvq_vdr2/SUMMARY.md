# 1204_rd08_q6k_mmvq_vdr2: Q6_K mmvq VDR=2 decode kernel (RD08)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD08

## What it does

Adds a vdr2 Q6_K vec_dot entry point that processes both 8-element chunks of a Q6_K dot product in one call, amortizing ql/qh/scales/d8 loads over 4 dp4a ops instead of 2, and switches get_vec_dot_q_cuda's Q6_K case to it. Also makes GGML_CUDA_OP_TIMING disable CUDA graph capture instead of aborting, plus adds decode-shaped perf test cases.

## Why

Decode is DRAM-bound, so halving loop iterations for the same row gives modest tg64 gains (fork: 23.35 -> 23.48 t/s d8192 on gfx1201); the fork reports the kernel is bit-identical to the VDR=1 path.

## Upstream / provenance

Ported from stew675-rdna-boosts fork commit 4591cc980 (https://github.com/stew675/llama.cpp). Not merged into ggml-org/llama.cpp master.
