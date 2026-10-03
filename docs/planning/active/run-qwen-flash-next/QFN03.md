---
id: QFN03
order: 0
plan: run-qwen-flash-next
state: pending
created-at: '2026-10-03T05:45:51.897055+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# VRAM overhead reduction for long context: in-place QSA mask, token_embd on CPU, batch and runtime reserves

## Description

Every long-context load failure on Brutus (2x 7900 XTX + R9700 tensor split, MTP draft on the 6900) has been a compute-buffer allocation (1.5-3 GB on one GPU at 256K or ub1024), not weights or KV. Reduce non-weight, non-KV VRAM so the context tiers grow and ub1024 (about +10% prefill) fits at long context.

## Steps

1. QSA mask in place (patch, proposed 1298): build_qsa_sel builds mask_all (repeat of a -inf seed over n_kv+1 x n_tokens), scatters the selected cells (set_rows) and adds kq_mask - three tensors of n_kv x ubatch each (~200 MB each in f16 at 192K x ub512; upstream marks it 'TODO: figure out to reduce the large compute buffer'). Replace with one tensor: copy kq_mask, then scatter -inf into the unselected cells or build the selection mask with a single fused op; or remove the mask entirely where 1295/1296 gather paths apply.
2. token_embd (Q8_0, ~675 MB) to CPU with -ot 'token_embd\.weight=CPU' (per_layer_token_embd is already on CPU); measure decode/prefill cost of the host gather.
3. Batch: -b equal to -ub (today -b 2048, -ub 512) and measure whether any per-batch staging is freed.
4. Runtime reserves: per-GPU ROCm context, HIP graph memory, RCCL buffers (NCCL_BUFFSIZE, channel count), cpu-root pinned buffers; measure each with rocm-smi before/after.
5. Re-run the adaptive fit search (tools/lab/flash-next/maxctx-search.py) with the winners.

## Detailed Solution & Technical Design

Measure first: per-device compute buffer sizes from sched_reserve in the server log at 192K ub512/ub1024 with and without each change. For step 1 the target is replacing ggml_fill+ggml_repeat_4d+ggml_set_rows+ggml_add over [n_kv, n_tokens] with at most one [n_kv, n_tokens] tensor, keeping f16 and identical mask values (bit-exact attention).

## Code Samples & Guidance



## Files



## Validation

Greedy text identical without MTP at 10K and 80K (1294 makes long-context runs deterministic); per-GPU VRAM and compute buffer size before/after; max context from maxctx-search.py before/after for f16-K/q8_0-V and q8_0; prefill/decode at the tiers (no regression).

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Evidence: flashnext-long-ctx-fit-1, flashnext-ctx-fit-2, flashnext-ub-sweep-1 (ub1024 fails at 96K f16 / 160K q8_0 / 192K q8_0), flashnext-maxctx-f16k-1. Owner rule: KV never q4; f16 preferred, q8_0 minimum. Deferred by the owner behind more critical performance work (2026-10-03).

## Change Log

- 2026-10-03T05:45:51.897055+00:00 (created-by): Created by agent

2026-10-03 PRODUCTION RISK (top QFN03 item): at -c 196608 -ts 2,2,3, a 131K-deep request (164K tokens filled) OOMs in hipGraphInstantiate on device 2 (R9700) -- base-b (production build) as well as both 1295 arms (flashnext-gather-ab-2/d131072). Cause: HIP graph instances (one per new graph shape) take device memory late, after KV/compute buffers fill the R9700. Fix options: cap/evict cached graph instances (LRU, or free on OOM and retry), reserve headroom for instantiation, or fall back to eager launch for that shape on hipErrorOutOfMemory instead of aborting (graphs-off costs ~6% decode). 1295's 192K OOM is this, not its gather workspace.

2026-10-04 1302_cuda_graph_oom_evict verified (flashnext-1302-deepfill, deployment candidate + 1302, -c 196608 -ts 2,2,3 q8_0): 164,470-token fill completes (prefill 636 t/s, decode at 131K depth 46.4 t/s, 178/230 accepted); at the would-be crash point the R9700 context evicted 193 cached graphs. Follow-up lever: ~193 live graph instances per GPU is a large standing VRAM reserve -- a capped (LRU, e.g. 16-32) graph cache could return that memory permanently to KV/compute and may let 2.2,2.2,2.6 / 2.3,2.3,2.4 fit 192K (need ~1.2 GB + margin on XTX1). Measure per-instance memory first.

## Ledger-events


- chg_20261003_141338_long-conversations-near-the-19_8442
- 2026-10-03T14:13:41.688716+00:00 (updated-by): Updated: section:ledger-events
- chg_20261003_151601_flash-next-now-runs-240k-conte_5065
- 2026-10-03T15:16:04.804500+00:00 (updated-by): Updated: section:ledger-events
