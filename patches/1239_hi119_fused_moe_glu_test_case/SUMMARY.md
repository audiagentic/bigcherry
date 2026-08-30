# 1239_hi119_fused_moe_glu_test_case: New registered test_case for the fused MUL_MAT_ID(gate)+MUL_MAT_ID(up)+GLU subgraph (HI119)

**Status:** untested
**Group:** core
**Plan item:** HI119

## What it does

Adds test_bigcherry_moe_glu_fusion, a registered multi-node test_case that builds the same gate/up MUL_MAT_ID pair sharing one activation and ids tensor object (by pointer) that a production MoE FFN produces, with the GLU/swiglu_oai as the terminal output, so ggml-cuda's real graph-fusion detector fires on it.

## Why

HI108 found real dispatch signatures stuck at rejected_no_correctness_evidence because test-backend-ops' single-op-per-line --test-file mapper cannot build a multi-node fused subgraph, and every GLU-op dispatch this tuner records is a fused MUL_MAT_ID+GLU epilogue.

## Upstream / provenance

Local design, part of this project's own correctness-evidence work (HI119), verified via dev-gpt-agent deep design review against the real ggml_cuda_should_fuse_mul_mat/ggml_can_fuse_subgraph requirements. Requires patch 1238's deterministic routing.
