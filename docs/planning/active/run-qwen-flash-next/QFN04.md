---
id: QFN04
order: 0
plan: run-qwen-flash-next
state: pending
created-at: '2026-10-03T15:21:12.489205+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: S
---

# 248K prefill collapse on profile v2 (prefill ~1100 -> 85 t/s, decode unchanged)

## Description

On production profile v2 (QFP07, KV on both XTX) both f16/q8_0 and f16/f16 load at -c 253952 but prefill of a 10K prompt collapses to 85-88 t/s while decode stays ~72 t/s; at 245760 prefill is ~1100 t/s. The R9700 sits at 34.06 of ~34.2 GB in both cases, although f16 V needs more KV memory than q8_0 V, so the identical threshold suggests either the R9700 compute-buffer/runtime paging or a context-length threshold in some kernel (e.g. mask/index sizes) rather than KV size.

## Steps

1. Repeat 248K with the R9700 share reduced (more free VRAM): if prefill recovers it is paging. 2. rocprofv3 the slow prefill: which kernel ballooned. 3. Check whether 1302 eviction fires during prefill.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Root cause identified; either a fix to reach 256K or a documented ceiling.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Runs: flashnext-attn-maxctx-110-hi, flashnext-attn-maxctx-110-f16 (c253952-t0). GPT asked in req_ee409a9b21e74e86.

## Change Log

- 2026-10-03T15:21:12.489205+00:00 (created-by): Created by agent
