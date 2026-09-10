# 0700_coverage_counters: Family-entry instrumentation and coverage counters (HI13)

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Adds counters at every real family entry point (not just the dense selector) to measure what fraction of matmul launches actually reach measured dispatch, since the graph optimizer calls MMVQ/MMVF directly for fused patterns, bypassing the dense selector.

HI168: counter calls and their reentrancy probes are compiled only with
`GGML_HIP_DISPATCH_DIAGNOSTICS`. Production retains family dispatch collection
without diagnostic counting. The upgrade edit also guards previously applied
hooks. This removes known instrumentation work; throughput parity still requires
a controlled hardware comparison.

## Why

Without this number, a tuning run's coverage of real model work is unknown, and 'we tuned the model' is an unverified assumption; test-backend-ops cannot produce this figure since it bypasses the graph optimizer entirely.

## Upstream / provenance

Local design, not in the original plan; added to answer a coverage question no other patch answers (HI13).
