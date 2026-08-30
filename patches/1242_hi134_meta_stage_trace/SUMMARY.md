# 1242_hi134_meta_stage_trace: Attribute META reduction copy activity to its transfer stages (HI134)

**Status:** untested
**Group:** core
**Plan item:** HI134

## What it does

Adds a bounded stage trace on top of the existing HI58/0830 telemetry: the generic META fallback reports the two copy sites it owns, and the HIP telemetry sink stores and serializes those stage records within the same per-reduction observation event.

## Why

HI58/0830's existing telemetry records one observation per reduction but does not attribute time within that reduction to its individual transfer stages, which HI134 needs for finer-grained diagnosis of META reduction copy activity.

## Upstream / provenance

Local design, part of this project's own telemetry work (HI134), building on the HI58/0830 observation-event bridge between generic META and the HIP owner.
