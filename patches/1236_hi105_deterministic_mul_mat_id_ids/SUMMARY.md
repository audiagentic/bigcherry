# 1236_hi105_deterministic_mul_mat_id_ids: Deterministic, full-expert-range routing for test_generic_op's MUL_MAT_ID initializer (HI105)

**Status:** untested
**Group:** core
**Plan item:** HI105

## What it does

Fixes test_generic_op's GGML_OP_MUL_MAT_ID branch to use a seeded shuffle (reusing patch 1222's helpers) of the full [0, n_expert) index range truncated to n_expert_used, instead of its own std::random_device engine and a range confined to [0, n_expert_used) that could never select experts from the rest of the pool.

## Why

HI105 extends correctness-evidence tooling to MoE-routed MUL_MAT_ID signatures via the --test-file path; two independent invocations of the same test case previously saw different random expert routing, and the routing range was structurally unable to reach most of a real MoE model's expert pool.

## Upstream / provenance

Local design, part of this project's own correctness-evidence work (HI105), found via dev-gpt-agent review and verified against the vendored source. Requires patch 1222's deterministic-mode helpers.
