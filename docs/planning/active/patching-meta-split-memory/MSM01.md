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

2026-10-06 context dependence of the R9700 measured from rocm-smi peaks, owner's layout (EP_TS 128,128,256, -ts 1,1,0, attention + KV on the XTXs, reordered UD file, build b-moeep-b11402r): ctx 49152 -> 30.9 GB; 147456 -> 31.9 GB; 196608 -> 32.4 GB; 245760 -> card full (32.6 GB), OOM at run time. About 0.5 GB per 49K of context, i.e. roughly 2.4 GB at 245K on a card that holds no KV - the upper bound of what MSM02 + MSM03 can return on the R9700 (about 22 experts per layer at 0.108 GiB each). XTX peaks in the same layout: 21.4 GB (147K), 22.5 GB (196K), 23.3 GB (245K, failed load). Speed at the reduced contexts (A = row split on the same file): ctx 196608: 8K prefill 1071.7/1076.0 vs 1075.4/1073.2, decode 86.2/85.1 vs 82.1/82.3; 98K prefill 975.1/975.5 vs 989.9/988.6 (-1.4%), decode 60.4/60.6 vs 58.5/59.9. ctx 147456: 8K prefill 1075.3/1074.7 vs 1074.2/1068.3, decode 85.3/85.2 vs 82.4/82.7; 98K prefill 976.1/976.7 vs 989.0/987.8 (-1.2%), decode 61.3/61.0 vs 60.1/60.2. Per-device report (1339) building as b-metamem-b11402a2.

2026-10-06 FIRST PER-DEVICE REPORT (1339, build b-metamem-b11402a2, reordered UD file; dev0/dev1 = XTX, dev2 = R9700; MiB). Production row split, ctx 49152: weights 17288 / 17621 / 28740; KV cache 708 / 708 / 188; indexer cache 288 / 288 / 288; compute arena 285 / 285 / 285. Production, ctx 245760: weights unchanged; KV 3012 / 3012 / 188; indexer cache 1440 / 1440 / 1440; compute arena 1021 / 1021 / 1021. Owner's layout (EP_TS 132,132,248, -ts 1,1,0), ctx 49152: weights 17471 / 17471 / 28708; KV 802 / 802 / 1.4; indexer 288 x3; arena 285 x3. Owner's layout, ctx 245760: KV 3106 / 3106 / 1.4; indexer 1440 x3; arena 1021 x3. FINDINGS: (1) confirmed - the compute arena is the same size on every device and the indexer cache is a full copy on every device, in both layouts. (2) On the R9700 at 245K the context-dependent memory is 1440 (indexer) + 1021 (arena) = 2461 MiB, matching the 2.4 GB inferred from rocm-smi; neither is needed there in the owner's layout (no attention share). (3) The indexer cache is the larger half: MSM03 is worth 1440 MiB on the R9700 at 245K, MSM02 up to 1021 MiB there and an unknown part on the XTXs. (4) In production the R9700 also holds a 188 MiB static block classed as KV cache that does not scale with context (first tensor not yet identified - probably the recurrent state). The first run printed the arena under 'other static' (log prefix shifted the field; lab script fixed).

## Change Log

- 2026-10-06T11:47:52.344928+00:00 (created-by): Created by agent
- 2026-10-06T12:06:15.621273+00:00 (updated-by): Updated: section:notes
- 2026-10-06T12:18:19.352114+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20261006_144832_diagnostics-and-experimental-p_1077
- 2026-10-06T14:48:36.063535+00:00 (updated-by): Updated: section:ledger-events
