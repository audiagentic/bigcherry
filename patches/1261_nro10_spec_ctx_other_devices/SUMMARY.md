# 1261_nro10_spec_ctx_other_devices

**Status:** untested
**Plan item:** PNRO10

## What it does

Adds the `ctx_other` (target context) model devices to the draft context's
scheduler backend list in `llama-context.cpp`, so shared tensors between a
speculative draft context and its target context can be scheduled on valid
backends. Devices are deduplicated by backend device handle, preserving the
existing ordering relative to ACCEL/CPU backends.

## Why

When a speculative context (MTP/draft) has a `ctx_other` target, shared
tensors between the two contexts must be schedulable on backends both
contexts can use. The target context's model devices may include backends the
draft context does not have (e.g., a different GPU partition); without adding
them to the draft's backend list, shared tensors could be un-schedulable.

## Upstream

Local (origin `local`); PNRO10. Anchored on the `backends.emplace_back(backend)`
site in `llama-context.cpp`.
