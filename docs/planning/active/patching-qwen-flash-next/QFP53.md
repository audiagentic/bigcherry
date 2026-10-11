---
id: QFP53
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-11T21:08:26.013082+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Prompt batches rebuild the graph every time: reuse it across batches (coarser KV padding) or build the next one while the cards run

## Description

QFP48's re-measurement on pin b11552 found the XTXs 91-92% busy in a median step; the idle that matters is per prompt batch: about 6.5 gaps of ~3.3 ms in every 512-token batch (1.63 s of a 28 s Flash-Next 24K prefill, ~6%), identical with context checkpoints off and present without a drafter.

Cause (BIGCHERRY_SUBMIT_TIMING=1 on build b-pq11552, 8K-depth prefill, run c48mt2): a prompt batch never reuses the previous graph.
- llama side, 40 submits of 512 tokens: reused 0; graph build 7.4 ms (max 21.9), inputs 3.2 ms per batch.
- Meta backend, 30 prompt-sized graph_compute calls: 97 subgraphs each, rebuild=1 on 27 of 30; subgraph rebuild 9-14 ms when it happens (6.0 ms mean); launch 73.5 ms and AllReduce enqueue 3.7 ms of host time per call (these overlap card work).
- Sum of the serial host work between batches: about 20 ms, which matches the ~21 ms of idle per batch seen in the kernel trace.

Why no reuse: `llama_kv_cache::get_n_kv` pads the attended range to max(n_pad, 256) cells 'so that the graph remains constant across batches and can be reused'. A 512-token batch moves n_kv by 512 every time, so `can_reuse_kq_mask` (kq_mask->ne[0] == n_kv) fails, the graph is rebuilt and its uid changes, and the Meta backend rebuilds its per-device subgraphs. In MTP decode (4-token verify) reuse is 98%.

## Steps

1. Patch (opt-in, default off): BIGCHERRY_KV_PAD in llama_kv_cache::get_n_kv; ENV_DOCS; mechanics test.
2. Hardware, Flash-Next production profile, depths 8K / 24K / 98K, N in {2048, 4096, 8192}: prefill t/s ABBA against default, SUBMIT_TIMING reuse count and rebuild count, greedy identity, fidelity probes vs CPU reference unchanged.
3. Same on Qwen3.8-27B dual XTX (dense attention: the masked work is fully paid there).
4. If A wins at every depth for a model, make it that profile's setting; if it loses at depth, cap N by depth or do B.
5. Fix 1320's ENV_DOCS: the switch the code reads is BIGCHERRY_SUBMIT_TIMING, the help row says BIGCHERRY_META_TIMING.

## Detailed Solution & Technical Design

Two candidates, not exclusive:

A. Coarser padding while a prompt is processed. Pad n_kv to a multiple of N cells (N a multiple of the batch size, e.g. 4096 or 8192) so that N/512 consecutive batches share one graph and the cached Meta subgraphs. Cost: attention and the lightning indexer run over up to N-1 masked cells more (on average N/2). Flash-Next's QSA attends top-k pooled blocks, so the extra cost may be well under proportional; flash_attn_ext_f16 is 17% of kernel time at 24K. Opt-in flag first (BIGCHERRY_KV_PAD=<cells>, default = upstream behaviour); the padded cells are masked, so output must be bit-identical.

B. Build the next batch's graph and Meta subgraphs while the cards run the current batch. No extra card work, but build must be separated from scheduler alloc/submit (the compute buffer is shared between consecutive graphs) and the Meta subgraph cache must take a second entry. Larger change; only worth it if A's masked work costs more than it saves at depth.

## Code Samples & Guidance



## Files



## Validation

Bit-identical greedy text with the flag on; SUBMIT_TIMING shows reused=1 on (N/512 - 1) of every N/512 prompt batches and rebuild=0 on the same; adoption ABBA with complete separation on prefill t/s at 24K and no regression at 98K or in decode.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Measured upper bound of the gain: ~20 ms of ~374 ms per batch at 24K, about 5%, less whatever the masked cells cost. Tools: tools/lab/flash-next/kernel-step-census.py (idle by gap size), gap-api-calls.py, queue-boundary-arms.sh, queue-checkpoint-arms.sh (PR #169). Ruled out on the way: context checkpoints (only 2 per prompt, idle unchanged with them off), the outer scheduler (2 splits per batch), the mul_mat_q -> convert pair (median gap 9 us; the mean was two code object loads).

## Change Log

- 2026-10-11T21:08:26.013082+00:00 (created-by): Created by agent
