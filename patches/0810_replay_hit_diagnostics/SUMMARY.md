# 0810_replay_hit_diagnostics: Optional, compile-time replay hit diagnostics

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Gates a hit-recorder and aggregated JSONL hit log behind GGML_HIP_REPLAY_DIAGNOSTICS/GGML_HIP_DISPATCH_HIT_LOG so production replay builds compile out the diagnostics branch and its synchronization cost entirely.

## Why

Diagnosing replay dispatch-table hits needs visibility, but that visibility must not cost anything in production replay builds.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework.
