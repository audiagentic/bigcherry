# 0820_measurement_signature_shapes: Persist canonical signature shapes in tuning measurements

**Status:** superseded
**Plan item:** none

## What it does

Stores the canonical signature shape alongside each tuning measurement record.

## Why

Downstream tooling needs the canonical shape associated with a measurement, not just its raw dimensions, to correctly group and replay candidates.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework.

## SUPERSEDED (2026-09-24)

The sections above describe the original change and are preserved. All four
edits (`result-canonical-json`, `set-result-canonical-json`,
`emit-result-canonical-json`, `format-result-canonical-json`) target the
BigCherry overlay file `src/ggml/src/ggml-cuda/hip-autotune-tuner.cu`, which
already contains their output: `patch-rebase-check` reports this patch as
CLEAN_NOOP at both b10901 and b11126, so applying it changes nothing. The
capability now lives directly in the overlay. No patch requires 0820; it was
removed from `[patch-set.campaign-support]`. `patch.py` and prior evidence are
kept unchanged for history.

