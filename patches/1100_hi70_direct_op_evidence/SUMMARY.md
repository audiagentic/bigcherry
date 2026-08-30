# 1100_hi70_direct_op_evidence: Deterministic direct-op correctness corpus for hard-to-reach candidates (HI70)

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Adds a small, deterministic, CPU-referenceable test_mul_mat corpus to test-backend-ops that constructs the exact tensor shapes (fallback MMQ M%128!=0, and MMF f16 batch widths 1-16) that well-formed production models structurally never produce.

## Why

24 of gfx1201's 100 candidates never got correctness evidence from any real-model workload because reaching them requires shapes production models don't naturally hit; hunting for a lucky real model is not a reliable evidence strategy and ran out of local models for q5_k entirely.

## Upstream / provenance

Local design, part of this project's own correctness-evidence work (HI70), per gpt-auto-agent's deep-dive recommendation.
