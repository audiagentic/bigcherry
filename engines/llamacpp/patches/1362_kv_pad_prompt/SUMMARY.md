# 1362_kv_pad_prompt

**Status:** untested  
**Plan item:** QFP53

## What it does

Adds opt-in `BIGCHERRY_KV_PAD=<cells>` graph padding to both dimensions that grow during Qwen4Exp/GLM5-Next prompt
prefill:

- KV rows use `max(n_pad, 256, N)` as the existing `get_n_kv()` padding quantum.
- k-pool rows use `max(64, GGML_PAD(N / indexer_kpool, 64))` pools as their padding quantum.

Unset/zero keeps both upstream paths exactly: KV remains `max(n_pad, 256)` and `kpool_pad` remains
`max(64, GGML_PAD(n_pool + 1, 64))`.

The environment value is parsed once by `llama_bigcherry_kv_pad(kv_size)`, declared in `llama-kv-cache.h` and
defined in `llama-kv-cache.cpp`. `llama-memory-hybrid-idx.cpp` calls that accessor instead of parsing the environment
again. This keeps validation and the per-cache "N exceeds KV size" disable decision identical for KV and pooled-block
sizing.

Valid nonzero values are multiples of 256 and no larger than the active KV cache. `BIGCHERRY_PATCH_TRACE=1` logs
`BIGCHERRY_PATCH_HIT patch=1362_kv_pad_prompt pad=<N>` once when active.

## Qwen4Exp failure fixed

The first version of 1362 stabilized `get_n_kv()` but not the k-pool graph dimension. Qwen4Exp
`llm_graph_input_kpool::can_reuse` also requires `pool_cells->ne[0] == mctx->get_n_kpool()`. Upstream
`kpool_pad(n_pool_real)` advances in only 64-pool buckets, so a full 512-token prompt batch that creates more than
64 pools changes `pool_cells` every batch and prevents graph reuse even while `n_kv` is stable.

With 1362 enabled, `kpool_pad` now derives its bucket from the same cell quantum. It still pads `n_pool + 1`, so the
comment `// The last padded pool is always unused.` remains true. The derived pool quantum can be a non-power-of-two
multiple of 64, so the second rounding uses integer ceil-to-multiple rather than `GGML_PAD` (whose alignment operand
must be a power of two).

## KV padding invariants and consumers

The original 1362 KV-side invariants are unchanged. 256 remains the minimum KV quantum; the constructor's
`GGML_ASSERT(kv_size % n_pad == 0)` and stored `n_pad` are untouched. `get_n_kv()` still clamps the padded end to
`cells.size()`, so a configured N does not need to divide the allocation.

At b11552:

- `llama_kv_cache_context::next`, `get_k/get_v`, KQ masks and relative-position inputs all consume the same padded
  `n_kv`; empty KV cells receive the existing drop mask.
- generic `llama-graph.cpp` attention allocation and `can_reuse_kq_mask` use that same `n_kv`.
- iSWA/hybrid wrappers forward their underlying cache contexts; base/SWA caches remain independently bounded.
- DSA/MSA indexer caches are ordinary KV caches and therefore use the same accessor. DSV4's separate compressed-cache
  and compression-plan padding remain unchanged.
- hybrid indexers use `n_kv` only as the one-past-live sentinel for padded `pool_idxs`; pool layout is built from
  occupied indexer cells, not padded KV rows.
- state save/restore walks occupied cell ranges and does not serialize padding-only rows, so state content/size is
  unchanged.

## b11552 k-pool sizing audit

Every `kpool_pad` consumer at b11552 uses the same new value:

- `kpool_build_sizes()` records the real pool count only; padding is deliberately applied later by the shared helper.
- `get_n_kpool()` returns `kpool_pad(mem, n_pool_real)`; Qwen4Exp and GLM5-Next allocate `pool_cells`,
  `pool_idxs`, `pool_mask` and their downstream score/selection shapes from it.
- `set_input_kpool()` asserts against the same `kpool_pad(mem, n_pool_real)` and fills exactly that width.
- `kpool_build_state()` uses the same padded count minus the guaranteed dead final pool when bounding `n_new_g`.
- Full-cache graph reservation first sets `n_pool_real` to the cache-wide maximum, then reaches graph tensor sizing
  through `get_n_kpool()`, so reserve and runtime graph construction use the same helper.

No separate state/save allocation is based on the padded pool count; the persistent pool keys remain stored in real
indexer KV rows.

### `n_new` / `get_n_kpool_new()`

No extra padding policy is required for the measured full 512-token single-sequence prompt path. `kpool_build_state()`
sets the graph count to at least

`bound = ubatch.n_tokens / kpool + ubatch.n_seqs_unq`.

For consecutive full 512-token batches this bound is constant. In the reported case `512 / kpool > 64`; after the
first full batch the pool-capacity term is already above that bound, so it does not move with `n_pool_real`.
`n_pool_max` also does not bind during ordinary prefill. Therefore `n_new_g` is the same stable bound on consecutive
full batches. Irregular/final batches or cache-capacity edges can legitimately have different shapes and are not the
reuse target of this patch.

## Graph-reuse audit

Qwen4Exp `llm_graph_input_kpool::can_reuse` terms for two consecutive full prompt batches of the same sequence:

- `k_idxs->ne[0]`: equals `ubatch.n_tokens`; stable at 512.
- `pool_cells->ne[0]`: was the moving term; stabilized by the new k-pool quantum.
- `pool_mask->ne[1]`: equals `ubatch.n_tokens`; stable at 512.
- `tail_idxs->ne[1]`: equals `ubatch.n_tokens`; stable at 512.
- `n_kv == idx->get_n_kv()`: stabilized by the original 1362 KV padding until the configured KV bucket boundary.
  The indexer and attention caches mirror size/padding/slots, so their Qwen4Exp equality assertion remains valid.
- `n_new == get_n_kpool_new()`: stable for the full-batch path as described above.
- PLE rows: `llm_graph_input_qwen4exp_ple::can_reuse` checks
  `rows->ne[0] == ple_n_heads * ubatch.n_tokens`; `ple_n_heads` is model metadata and `n_tokens` is 512, so it is
  stable across these batches.

Generic hybrid inputs in `llama-graph.cpp` add no growing prompt-length dimension:

- attention `self_k_idxs` is sized by `n_tokens`;
- KQ mask width is `get_n_kv()` and its other dimensions are `n_tokens / n_stream`, 1, and `n_stream`;
- recurrent `s_copy` is the live recurrent-cell range, `s_copy_main` is `ubatch.n_seqs`, and reuse also checks
  recurrent `head` and `rs_z`. For the same single sequence, recurrent `find_slot()` retains its existing owned tail
  cell, so those values do not advance with prompt length. Multi-sequence/layout changes are outside this reuse case.
- the `_k` and `_iswa` hybrid variants use the same checks; `_iswa` checks base/SWA KQ masks independently.

## Top-k and padded-pool correctness

Increasing the padded pool count does not make padding visible:

- `kpool_layout_update()` creates pools only from occupied indexer cells.
- `set_input_kpool()` fills every padding-only `pool_mask` entry with `-INFINITY`.
- padding-only `pool_idxs` entries use the `n_kv` sentinel; `pool_cells` uses a finite dummy cell only so the gather
  itself is address-valid.
- Qwen4Exp may have top-k return an invisible padded pool when fewer real pools are visible. `build_qsa_sel()` derives
  `live_pool` from the selected score; `-INFINITY` becomes dead, and every dead slot is remapped to its own dump row
  beyond `n_kv` before the final mask view is restricted back to the first `n_kv` rows. Missing tail cells use the
  same sentinel/dead-slot treatment.
- Qwen4Exp `n_sel` and GLM5-Next selection width are derived from `min(n_pool, indexer_top_k / kpool)` plus the tail.
  A larger padded `n_pool` cannot increase `n_top_pool` beyond the model's top-k width; extra ranked slots remain
  dead under the existing masks. GLM5-Next's `sel_mask` likewise marks unavailable selection slots `-INFINITY`.

## Correctness and validation status

The change affects graph allocation/fill width only. It does not create real pools, change occupied KV cells, alter
pool representatives, or expose padding to attention. The larger padding is protected by the existing `-INFINITY`
pool mask and `n_kv` sentinels described above.

The merged KV-only 1362 was reported hardware bit-identical for greedy output, but still had `reused=0` on all 40
full 512-token submits and no prefill gain. This k-pool extension has not yet been hardware-verified here. Required
acceptance remains greedy bit identity plus `reused=1` on consecutive full prompt batches and prefill measurements at
8K/24K/98K. Decode/training behavior and irregular/multi-sequence prompt reuse are not claimed qualified.

## Upstream

Local BigCherry enhancement against llama.cpp b11552 (`23b0202a189c44a54625aadcb37a946dd1d6278d`).
