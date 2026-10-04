# 1316_node_hash_trace

**Status:** untested
**Plan item:** QFP15

## What it does

Diagnostic only. `BIGCHERRY_NODE_HASH=from:count` installs a scheduler eval callback (when the caller set none) that,
for each llama_context's graph_compute calls `from .. from+count-1`, copies every computed node to host and logs
`BIGCHERRY_NODE_HASH ctx g i name op ne h=<FNV-1a>`. Diffing two runs of the same build/prompt finds the first node whose
output is not bit-reproducible. The callback forces per-node sync and disables graph replay, so overlap-dependent races
may hide under it; use 1315's unperturbed trace to choose the window.
