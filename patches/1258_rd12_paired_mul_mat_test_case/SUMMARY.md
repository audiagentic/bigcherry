# 1258_rd12_paired_mul_mat_test_case

**Status:** untested
**Plan item:** RD12

## What it does

Adds a deterministic whole-graph `test-backend-ops` case containing two
distinct Q6_K `MUL_MAT` projections over one shared F32 activation so
RD12's real paired-MMVQ dual-output fusion can be correctness-tested.

## Why

RD12's production selector requires two distinct adjacent `MUL_MAT`
nodes sharing the same `src1` activation tensor -- a shape the ordinary
`test-backend-ops --test-file` single-op path cannot represent.
