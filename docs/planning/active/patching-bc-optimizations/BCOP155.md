---
id: BCOP155
order: 155
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-12T05:08:32+00:00'
created-by: agent
priority: P1
work: S
---

# MSM03/1341: bind KQ-mask subset policy to the Meta instance

## Discovery / change

1341 already seeds `GGML_OP_NONE` KQ-mask inputs (default-on with subset mirroring), contrary to its stale summary. The rule uses unsynchronized process-global `bc_meta_attn_input_mask` without reset, per-instance identity or cache-policy generation. Source-level conditional lifetime/data-race risk; **no reproduced GPU failure**. Compiled C++17 host ownership fixture: 169 assertions passed over 42 different-mask context pairs; atomic global alone does not isolate contexts. No new performance measurement.

## Owners / existing work / 12-hour exclusions

MSM03/1341 owns subset residency and graph-input policy. MSM02/1340 owns physical arena; MSM01/1340 (formerly 1339) owns existing `BIGCHERRY_META_MEM` receipts; FKE01 owns fusion numerical equivalence. Reuse 1303 `BIGCHERRY_ATTN_TS`, pinned b11552 Meta callback/userdata and buffer-local cache. No new scheduler/allocator/registry/telemetry. Last substantive MSM03/1341 development Oct 7; Oct 9 packaging mechanical. Twelve-hour and BCOP145-154 novelty checks passed. Active Radiance RR08/MXFP4/GDN/paged prefill/wire, PA44/PA49, QFP15, QFP53/1362 and QFP49/1361 protected and untouched.

## Bounded action / terminal gate

First extend 1341 mechanics tests for two Meta contexts A/B/A, concurrent callbacks, unseeded/legacy reset, exact KQ names and cache lifetime. Prefer narrowly invoking existing model `get_split_state/get_split_state_ud` for `GGML_OP_NONE` mask inputs before the model weight-name parser; remove global setter, preserve MIRRORED fallback. If supported execution is proven single immutable policy, document and close without new code. Otherwise instance-bind before expanding masks. Only after CPU gates, run multi-request gfx1100/gfx1201 correctness/graph/MTP plus existing per-device memory accounting. Reject stale masks, wrong-device work, numerical regression or cache mismatch. Full design remains solely in MSM03.

## References

MSM01/MSM02/MSM03; 1303; 1340/1341; pinned llama.cpp b11552 `ggml-backend-meta.cpp`; upstream llama.cpp #26395; vLLM BlockPool; SGLang memory_pool.
