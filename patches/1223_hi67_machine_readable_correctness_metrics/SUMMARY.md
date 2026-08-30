# 1223_hi67_machine_readable_correctness_metrics: Machine-readable per-tensor correctness metrics from test-backend-ops (HI67 slice 2b)

**Status:** untested
**Group:** core
**Plan item:** HI67

## What it does

Prints an unconditional, machine-readable BIGCHERRY_CORRECTNESS_METRIC line from test-backend-ops' existing NMSE computation (already computed internally, previously only surfaced as a truncated pass/fail message), plus a new max_abs(backend1, backend2) computation in the same element loop.

## Why

test-backend-ops was already computing E_N/E_C (the exact numbers HI67's contract needs) internally but never exposed them in parseable form; RV49's contract also needs max_abs(C,R) <= max_abs(N,R) as a second bound, which nothing computed before.

## Upstream / provenance

Local design, part of this project's own correctness-evidence work (HI67 slice 2). Requires patch 1222's deterministic-mode gate.
