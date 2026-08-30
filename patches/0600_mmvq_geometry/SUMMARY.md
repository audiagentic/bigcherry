# 0600_mmvq_geometry: Explicit MMVQ geometry variants (HI09 part 1)

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Adds two defaulted template parameters (nwarps_explicit, rows_per_block_explicit) to the MMVQ kernel template; zero means derive geometry as upstream does (native instantiations unchanged), non-zero compiles a new geometry instance. Bounds are static_assert-checked in-kernel as a backstop.

## Why

MMVQ derives its geometry from calc_nwarps/calc_rows_per_block at compile time, so an alternative geometry needs genuinely new compiled code rather than a runtime switch, unlike MMQ/MMVF/MMF.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI09).
