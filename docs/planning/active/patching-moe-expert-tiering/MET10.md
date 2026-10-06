---
id: MET10
order: 10
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-06T16:28:05.477298+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Host tail for the expert split: GPUs hold experts [0,K) of every layer, [K,N) stay in host RAM and are uploaded selectively

## Description

For quants whose routed experts do not fit in VRAM. Today that case runs with whole layers' experts on the host (--n-cpu-moe) plus 1336's selective upload, which is slow in prefill. Design answer from GPT (owner-relayed 2026-10-07; my two gateway requests for it had failed): option (b), a GPU-computed host tail - keep 1283's GPU-resident partition for experts [0,K), add a second MoE branch (gate / up / GLU / down range ops) over host-resident experts [K,N) that runs on a GPU with selective uploads, sharing the original activations, global expert ids and routing weights; aggregate each branch and combine, preferably merging the tail's contribution into one device's partial result BEFORE the existing AllReduce so the block keeps one collective. Rejected: (a) CPU as a meta device (heterogeneous ownership + collective changes), (c) CPU-computed tail (no uploads but CPU expert compute becomes the prefill bottleneck).

## Steps

FIRST DECISIVE EXPERIMENT (GPT): fixed contiguous K, no frequency permutation. Baseline = --n-cpu-moe + 1336; experimental = 1283 with [0,K) resident + [K,N) selective host tail. Same quant, 4096-token prefill, KV settings, GPU memory budget and routing inputs. Measure prefill t/s, host-to-device bytes and time, peak VRAM, graph synchronisation time; validate logits against a full-resident reference. Decision rule: if cutting upload bytes substantially does not raise prefill throughput, investigate synchronisation and tail-branch execution before building frequency placement. THEN: frequency placement (profile from 1338's recorder, reorder with expert-place.py), per-device ownership rule for the tail, 1336 range awareness, one-collective merge.

## Detailed Solution & Technical Design

Correctness traps named by GPT: (1) 1336 is not range-aware: sched_copy_experts assumes 0 <= id < src->ne[2]; it needs id_base = K, filtering of global ids, translation to local offsets, and empty selections handled. (2) 1336 recognises a host weight only when graph->nodes[0] is its MUL_MAT_ID; a second branch needs that detection generalised and the ids available. (3) Meta replication: a host-weight branch would execute on every GPU - enforce single-device ownership with zero contribution elsewhere, or uploads are redundant and sums wrong. (4) AllReduce boundary: adding the same tail contribution to every device's partial multiplies it by the device count; adding after the AllReduce needs the tail output distributed. (5) Keep quantisation alignment, MMQ padding and staging-buffer lifetime; reserve temporary VRAM. PLACEMENT: GPT says no offline repack is needed (per-layer load-time permutation of the expert slabs plus an old->new id map applied after top-k). NOTE (mine): we already have the offline form working and lossless - tools/lab/flash-next/expert-place.py permutes gate / up / down expert blocks and the router rows together, so no id remap is needed at run time; the cost is a second copy of the model on disk. Either works; the offline tool exists today. Caveat from MET09: the GPU top-k has no expert-id tie-break, so any reordering can change selection among equal scores. GPT also notes that large prefill selects nearly all experts, so frequency ordering mainly helps decode and small batches - consistent with our UD measurement (a 512-token batch touches 61% of a layer's experts on average, a 2048-token batch 79%).

## Code Samples & Guidance



## Files

patches/1283_qwen4exp_expert_parallel, patches/1281_moe_mul_mat_id_range, patches/1336_sched_copy_callback (range-aware copy), src/llama-graph.cpp build_moe_ffn (second branch) via a new patch package, tools/lab/flash-next/expert-place.py, moe-copy-ab.sh

## Validation

Logits against a full-resident reference inside the stack's envelope; prefill and decode against --n-cpu-moe + 1336 at equal VRAM; upload bytes per ubatch reduced as predicted; single AllReduce per block preserved.

## Effort & Risk

High: a second MoE branch in the graph, device-ownership rules in the meta backend, and range awareness in the copy callback. Needs a higher-quant file to be meaningful (none downloaded; /mnt/data has ~41 GB free).

## Standards



## Acceptance Criteria



## Notes

Related results already in hand: host-expert baselines with 1336 (MET01), the single-card profile-pinned cache 1337 + 1338 (MET07) which is the same idea without the tensor split, and usage-placed experts inside the split (MET08). Supersedes the 'CPU tail' wording in MET04's title for the over-VRAM case.

## Change Log

- 2026-10-06T16:28:05.477298+00:00 (created-by): Created by agent
