---
id: QFP03
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T15:20:25.602966+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: S
---

# 1294 deterministic TOP_K ties (lowest-index tie-break in HIP radix top-k)

## 2026-10-14 QFP03 implementation audit — remove redundant ordered-radix gather (planning only)

**Disposition:** keep patch `1294_topk_deterministic_ties` **validated** and enabled for its proven deterministic-selection purpose. QFP03 remains **pending only for a default-on ordered-path simplification and exact-pin qualification**, not a second TOP_K implementation. No new HIP result, build, patch application, or speedup is claimed here. Source audit: pinned llama.cpp **b11552** (`23b0202a189c44a54625aadcb37a946dd1d6278d`), current upstream `ggml/src/ggml-cuda/top-k.cu`, BigCherry `1294/patch.py` and owned `bc-topk-ties.cuh` source. Current upstream retains the same radix gather and has no BigCherry deterministic ordered pass.

### Concrete redundant dataflow

`ggml/src/ggml-cuda/top-k.cu::top_k_radix_cuda` chunks rows to a 64 MiB histogram budget. For every chunk it runs `top_k_radix_init`, four radix histogram/select passes, then `top_k_radix_reset_counters` and `top_k_radix_gather`. The gather atomically increments `states[row].greater_count/equal_count` and writes up to `k` output indices in scheduling-dependent order.

Patch 1294 inserts `top_k_radix_gather_ordered<BLOCK_SIZE>` **after** that gather when both `BIGCHERRY_TOPK_DETERMINISTIC` and `BIGCHERRY_TOPK_ORDERED` are on (both default on). The ordered kernel reads **only** `src`, `states[row].prefix`, `states[row].rank`, `ncols`, `k` and `dst`; it neither reads the counters nor the previous `dst` contents. It overwrites every selected index. Therefore, for the ordered arm, the preceding counter-reset and atomic-gather launches and the gather's full row read are **dead work**. This is a source-proven dependency, not a measured latency gain.

The ordered output is **not globally ascending by column** despite the patch's flag description: it writes all `key > prefix` indices in ascending column order, followed by the lowest-index `key == prefix` indices in ascending order. Example `scores=[0,2,0,3], k=3` yields `[1,3,0]`, not `[0,1,3]`. This partitioned order is deterministic and must remain unchanged for sparse-gather consumers and floating-point reduction order. Do not silently replace it with global sorting.

### One in-place patch candidate; preserve all other modes

Extend **existing** `1294/patch.py::TOPK` at its chunk-local gather/reset seam, not a second scheduler, kernel, dispatch table or flag. Factor the existing upstream reset/gather calls under the same ordered decision already used by 1294; the radix histogram/select and 64 MiB chunking stay unchanged:

```cpp
const bool ordered = det_ties && top_k_bc_ordered();
for (each existing radix chunk) {
    radix_init_and_four_histogram_select_passes(); // unchanged
    if (ordered) {
        top_k_radix_gather_ordered<BLOCK_SIZE><<<iter_nrows, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k);
    } else {
        top_k_radix_reset_counters<<<...>>>(states, iter_nrows);
        top_k_radix_gather<BLOCK_SIZE><<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
        if (det_ties) {
            top_k_radix_gather_ties<BLOCK_SIZE><<<iter_nrows, BLOCK_SIZE, 0, stream>>>(
                src, dst, states, ncols, k);
        }
    }
    advance_src_dst_by_iter_nrows(); // unchanged
}
```

Preserve `BIGCHERRY_TOPK_DETERMINISTIC=0` stock atomic-gather behavior, `BIGCHERRY_TOPK_ORDERED=0` tie-only behavior, non-radix bitonic/CUB paths, same-stream ordering, HIP graph capture, per-chunk pointers, and gfx1030 draft fallback. The ordered kernel must not depend on an uninitialized `equal_count`/ `greater_count` after the change. Theoretical opportunity: **two fewer launches and one fewer full-input gather read per radix chunk**, with no change to radix passes, output bytes or host/device transfer count. Actual exposed prefill/decode saving is unknown.

### Cheap discriminator completed; hardware gate not yet met

- **11/11** direct pinned-source/patch assertions: reset writes only counters; stock gather consumes counters; ordered kernel uses prefix/rank but no counters or previous output; ordered launch follows gather; chunking and default flags confirmed.
- Host-only reference model: **6 fixed + 20,000 seeded random cases** (unique values, all ties, positive/negative zeros, infinities, mixed integer ties, `k=1..n`) matched the existing gather-then-ordered output against direct ordered selection. The partitioned-order counterexample passed. This is not a HIP or numerical parity test; it excludes unqualified NaN/special-payload claims.
- Existing historical evidence: 1294 cross-start no-MTP greedy stability at 32K/80K; its **2026-10-03** speed-neutral result predates the **2026-10-07** ordered-output addition. The profile-v6 versus native gains in 1294 README combine many patches and cannot be attributed to ordered TOP_K or this candidate. No isolated b11552 ordered-gather-elision timing exists.

**Stage A (before hardware):** implement only in 1294 after confirming exact b11552 anchor and file-local `top_k_radix_state` availability. Run patch-lint, composition/idempotence and a CPU/C++ source-shaped fixture that compares direct ordered vs current two-pass outputs, including NaNs and signed zeros by exact `top_k_float_to_ordered` key; `k=1`, `k=ncols`, zero/ambiguous cutoff ties, `ncols` around 1024/4096, `nrows` spanning multiple 64 MiB chunks, and all four flag combinations. Assert every `dst` slot is written exactly once on the ordered arm, no stale pool-state read and graph replay stability. Compare `TOP_K` backend-op full index vectors, not merely selected-score multisets.

**Stage B (only if A passes):** on physical gfx1100/gfx1201, with gfx1030 draft as a separately gated control, compare current validated 1294 versus gather-elided 1294 at 10K/80K/160K (where memory fits), ub512/1024, MTP off/on, no-MTP QSA and repeated same-process requests. Collect exact radix-route/flag activation, chunk count, GPU kernel names/launch count, per-stage microseconds, total TOP_K time, index-vector hashes, greedy/logit/KLD/MTP acceptance, peak VRAM, and paired end-to-end prefill/step time. No P2P assumption. Require no missing selection work, identical partitioned index order and numerical/greedy behavior. Use >=4 independent sessions and >=10 paired ABBA rounds/session. **Promote only with CI95-low >=3% exposed E2E improvement, <=1% unaffected-control regression, and full correctness; otherwise keep validated 1294 unchanged and close QFP03.** Do not queue a conflicting active GPU lane.

### Ownership, duplication and external baseline

QFP03 owns **only** deterministic radix TOP_K semantics and this in-place dead-gather elimination; `1294` remains the single implementation. The patch manifest's historical `plan-ids=[RNX02]` links the original QSA extraction, not authority to create another top-k kernel. RNX02 owns D=256 quantized-KV attention/1300, QFP15 owns residual target/draft acceptance nondeterminism, QFP04/1295 consumes ordered indices, QFP25/1334 owns sparse-attention compaction, and PNRO07/1256 is a conflicting alternate TOP_K route. No edits to their kernels or queued experiments.

Upstream llama.cpp [PR #28713](https://github.com/ggml-org/llama.cpp/pull/28713) supplies the bounded row-chunk/dispatch baseline, **not** ordered deterministic radix output. [vLLM issue #54521](https://github.com/vllm-project/vllm/issues/54521) independently distinguishes unstable **set** under ties from unstable **order** without ties in sparse TOP_K; [SGLang's DSA backend](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py) exposes deterministic tie-break selection via FlashInfer. These corroborate the semantic contract, not an AMD-portable speedup or a reason to replace 1294.

## Description

Patch 1294_topk_deterministic_ties (evaluated, in production profile): HIP radix TOP_K picked tied QSA cells in atomic order (ReLU-sum scores give many exact zeros), so long-context greedy output differed between server starts. Lowest-index tie-break makes 32K/80K output identical across starts. Speed neutral. Conflicts with 1256.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Same prompt, two server starts, greedy identical at 32K and 80K; ms/step unchanged.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Upstream fact (GPT online search): QSA HIP's >1K-context cliff was TOP_K falling back to CPU, fixed by upstream radix TOP_K, so TOP_K is on the critical path. Distinct from PNRO06/PNRO07 (TOP_K speed). GPT review in flight: req_af96db92b8e34cfe.

## Change Log

- 2026-10-03T15:20:25.602966+00:00 (created-by): Created by agent
