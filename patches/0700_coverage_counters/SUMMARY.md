# 0700_coverage_counters: Family-entry instrumentation and coverage counters (HI13)

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Adds counters at every real family entry point (not just the dense selector) to measure what fraction of matmul launches actually reach measured dispatch, since the graph optimizer calls MMVQ/MMVF directly for fused patterns, bypassing the dense selector.

## Why

Without this number, a tuning run's coverage of real model work is unknown, and 'we tuned the model' is an unverified assumption; test-backend-ops cannot produce this figure since it bypasses the graph optimizer entirely.

## Upstream / provenance

Local design, not in the original plan; added to answer a coverage question no other patch answers (HI13).
