# 0830_split_reduce_telemetry: Observe actual SPLIT_REDUCE provider and meta handoff (HI58)

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Instruments the SPLIT_REDUCE path to record which reduction provider (RCCL/meta) actually handled a given reduction and the handoff between them.

## Why

Needed to verify, from real telemetry rather than assumption, which reduction path a given multi-GPU run actually used.

## Upstream / provenance

Local design, part of this project's own telemetry work (HI58).
