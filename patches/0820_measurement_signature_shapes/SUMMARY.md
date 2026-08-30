# 0820_measurement_signature_shapes: Persist canonical signature shapes in tuning measurements

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Stores the canonical signature shape alongside each tuning measurement record.

## Why

Downstream tooling needs the canonical shape associated with a measurement, not just its raw dimensions, to correctly group and replay candidates.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework.
