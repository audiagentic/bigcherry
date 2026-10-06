---
id: MSM01
order: 1
plan: patching-meta-split-memory
state: pending
created-at: '2026-10-06T11:47:52.344928+00:00'
breadth: ''
skill: basic
created-by: agent
priority: P1
work: S
---

# Per-device memory report for the tensor-split backend (1339)

## Description

The server logs one size per Meta buffer, so what each card of the tensor split holds is invisible. Patch 1339_meta_memory_report (BIGCHERRY_META_MEM=1, diagnostic) prints the real per-device size of every meta buffer: compute arenas and static buffers (weights, KV, indexer state) with the first tensor's name. This is the measurement base for MSM02 and MSM03.

## Steps

1. Patch 1339 authored with offline test (done). 2. Run tools/lab/flash-next/queue-meta-mem.sh: production row split and the owner's layout (dense + attention + KV on the XTXs, usage-placed experts) at ctx 49152 and 245760. 3. Record MiB per device by class (compute arena, KV cache, indexer cache, weights). 4. Extend the report to the per-device compute allocator of MSM02 and to the AllReduce scratch (n_reduce_steps * max_tmp_size).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

patches/1339_meta_memory_report/, tools/tests/patch/test_1339_meta_memory_report.py, tools/lab/flash-next/queue-meta-mem.sh, config/recipes.toml (experiment meta-mem)

## Validation

Offline mechanics test + patch-lint; hardware: the report prints for every device and the per-device sums match rocm-smi VRAM within the driver overhead.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Base framework / diagnostic: no promotion evidence needed beyond 'prints and changes nothing' (identical greedy text with the flag on and off).

## Change Log

- 2026-10-06T11:47:52.344928+00:00 (created-by): Created by agent
