# 1238_hi119_deterministic_init_mul_mat_id_tensors: Deterministic expert-ID routing for the registered test_case classes' shared initializer (HI119)

**Status:** untested
**Group:** core
**Plan item:** HI119

## What it does

Fixes init_mul_mat_id_tensors() -- the shared initializer every registered MUL_MAT_ID-family test_case (test_mul_mat_id, test_mul_mat_vec_fusion, and HI119's new class) uses -- to shuffle its full [0, n_mats) index range with a seeded engine when BIGCHERRY_TEST_DETERMINISTIC_SEED is set, reusing patch 1222's helpers.

## Why

HI119 needs to reuse test_mul_mat_vec_fusion's use_id=true path as a template, but init_mul_mat_id_tensors() is a third, separate std::random_device site distinct from the two sites patches 1222 and 1236 already cover, so it was still genuinely non-deterministic for every registered caller.

## Upstream / provenance

Local design, part of this project's own correctness-evidence work (HI119), found via dev-gpt-agent deep design review. Requires patch 1222's deterministic-mode helpers.
