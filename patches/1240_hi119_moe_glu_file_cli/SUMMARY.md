# 1240_hi119_moe_glu_file_cli: --moe-glu-file CLI hook for the fused MoE GLU test case (HI119)

**Status:** untested
**Group:** core
**Plan item:** HI119

## What it does

Adds a --moe-glu-file argument (a new moe_glu_file_path parameter on test_backend(), independent of --test-file) that lets a Python evidence producer instantiate patch 1239's test_bigcherry_moe_glu_fusion class with runtime-derived parameters via a one-line `type glu_op k n m n_mats n_used broadcast` format, without recompiling test-backend-ops.

## Why

The two static-corpus instances patch 1239 registers prove the harness design works, but are not a general mechanism for arbitrary observed dispatch shapes; the existing --test-file/test_generic_op escape hatch cannot build a multi-node fused subgraph either.

## Upstream / provenance

Local design, part of this project's own correctness-evidence work (HI119), mirroring make_test_cases_from_file()'s existing line-format convention.
