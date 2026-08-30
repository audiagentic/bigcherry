# 1222_hi67_deterministic_test_backend_ops_seed: Deterministic tensor init for CPU-reference correctness evidence (HI67 slice 2a)

**Status:** untested
**Group:** core
**Plan item:** HI67

## What it does

Adds an opt-in deterministic mode (BIGCHERRY_TEST_DETERMINISTIC_SEED) to test-backend-ops' init_tensor_uniform(): single-threaded, seeded generation with a monotonic per-call counter folded into the seed, plus a machine-parseable FNV-1a digest printed on stderr; unset, every existing caller is byte-for-byte unchanged.

## Why

HI67's correctness contract requires comparing a candidate's GPU output and native's GPU output against the same CPU reference, which requires two separate process invocations to have generated identical input tensors -- confirmed by source read that init_tensor_uniform seeds from std::random_device, so this did not hold before this patch.

## Upstream / provenance

Local design, part of this project's own correctness-evidence work (HI67 slice 2, adjudicated by GPT in RV77).
