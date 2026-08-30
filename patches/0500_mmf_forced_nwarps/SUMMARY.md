# 0500_mmf_forced_nwarps: MMF forced-nwarps dispatch (HI08)

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Same explicit-appended-defaulted-parameter shape as HI07, applied to MMF's three dispatchers (which share an identical signature/call tail); shared-memory sizes are recomputed from the forced nwarps immediately after the scan so allocation stays correct.

## Why

Needed so the tuner can force and measure a specific MMF nwarps value while leaving the native path byte-identical to upstream, without under-allocating shared memory for a forced value larger than native's choice.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI08).
