# rad0055_step_phase_trace

**Status:** validated
**Plan item:** RUM09

Scope: Per-rank, per-op decode timing for a native serving start, selected by `--step-phase-trace`.

## Summary

Off by default. The serving owner hands the engine a run-directory path only for a start given `--step-phase-trace`; the engine writes per-pass HIP-event timing per rank. Timing is perturbed when on, so throughput comes from an untraced run.

## Outcome

The untraced paths are the upstream calls. Ownership, rank order and the recorded tape are unchanged when tracing, and rank-local files are never overwritten.

## Detail

The trace uses the engine's HIP event API and the sealed owner's path, with no environment input.

[Patch implementation](patch.py).
