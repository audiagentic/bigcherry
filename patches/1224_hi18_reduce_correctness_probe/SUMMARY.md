# 1224_hi18_reduce_correctness_probe: Standalone SPLIT_REDUCE correctness probe, test-hip-reduce (HI18)

**Status:** untested
**Group:** core
**Plan item:** HI18

## What it does

Wires a standalone host executable (source in the overlay) into the build that drives the real production META backend through GGML_HIP_REDUCE_PLAN=auto|rccl|meta with a minimal 2-node graph and reports machine-readable execution facts for tools/bigcherry/reduce_correctness.py.

## Why

HI18 needs a native-half correctness-comparison gate that exercises the real META split-state/reduce-plan machinery, not a Python reimplementation that could disagree with it; this is the missing tool HI15/HI16's review update assigned.

## Upstream / provenance

Local design, part of this project's own correctness-evidence work (HI18). D=2 device-count slice only.
