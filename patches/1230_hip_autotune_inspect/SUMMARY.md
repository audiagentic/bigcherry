# 1230_hip_autotune_inspect: Offline cache and registry inspector, hip-autotune-inspect (HI15/HI16)

**Status:** untested
**Group:** core
**Plan item:** HI15/HI16

## What it does

Wires a host executable (source in the overlay) into the ggml-hip backend build that links the same registry functions and replay loader a production process uses, so its answers can be checked against a Python reimplementation of the same logic.

## Why

HI16's catalog/registry agreement tests and HI15's review both needed a C++-side check that a Python reimplementation of the loader could not itself introduce disagreement with the real registry.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI15/HI16).
