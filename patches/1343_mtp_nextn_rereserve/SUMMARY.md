# 1343_mtp_nextn_rereserve

**Status:** validated
**Plan item:** QFP32

## What it does

With `BIGCHERRY_MTP_RERESERVE=1`, `llama_context::set_embeddings_nextn` marks the scheduler for a reserve when the
NextN (MTP) output mode really changes. The next `process()` then reserves the worst-case graph of the shape that will
run, once.

## Why

The context reserves its worst-case graph in the constructor, before the MTP driver switches the NextN outputs on. The
graphs that run afterwards have another node count, so the reserved plan does not apply to them: the first one is
planned from its current sizes, and every input sized by the filled context then stops fitting as it grows. That is
one allocator re-plan with a synchronize per prefill chunk. Measured with 1340's counters on Flash-Next at ctx 245760:
"no reserve plan nodes=7204" on the first graph, then 11 re-plans per device for a 2K fill and 246 for a 98K fill.

## Scope

Allocation and scheduling only; no kernel, fusion or arithmetic change, so the output must be identical. Topology
neutral (the setter is above the backends). A repeated call with the same mode asks for nothing.

## Activation

`BIGCHERRY_PATCH_TRACE=1` prints `BIGCHERRY_PATCH_HIT patch=1343_mtp_nextn_rereserve value= masked=` once per real
mode change (target and draft context each).
