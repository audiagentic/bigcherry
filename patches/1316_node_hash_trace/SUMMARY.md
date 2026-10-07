# 1316_node_hash_trace

**Status:** untested
**Plan item:** QFP15

## What it does

Diagnostic only. `BIGCHERRY_NODE_HASH=from:count` installs a scheduler eval callback (when the caller set none) that,
for each llama_context's graph_compute calls `from .. from+count-1`, hashes observable simple-buffer node outputs and
logs `BIGCHERRY_NODE_HASH ctx g i name op ne h=<FNV-1a>`.

Accepted value format is exactly `from:count`, with `from >= 0` and `count > 0`; for example `0:4` observes the
first four graph_compute calls.

Meta-owned tensors are deliberately not passed to `ggml_backend_tensor_get()`. A Meta logical tensor may be
split/partial across devices and its flat generic getter can abort. Those nodes instead log `skip=meta` and graph
execution continues. Simple-buffer nodes are still hashed normally.

The callback forces per-node synchronization and disables graph replay, so overlap-dependent races may hide under it;
use 1315's unperturbed trace to choose the graph window.
