# 0400_mmvf_forced_block: MMVF forced block-size and accumulator-mode dispatch (HI07)

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Threads a forced block-size/accumulator-mode value down through appended, defaulted parameters from ggml_cuda_mul_mat_vec_f to its launcher, touching only the call chain a forced value actually travels (replace_all edits with asserted match counts).

## Why

An earlier thread-local-override design was rejected because production replay builds would pay a per-launch read on the hottest path for a value that's always zero in production; an explicit parameter keeps the native path byte-identical to upstream.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI07).
