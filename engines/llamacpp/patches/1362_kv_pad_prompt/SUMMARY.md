# 1362_kv_pad_prompt

**Status:** untested
**Plan item:** QFP53

## What it does

Adds opt-in `BIGCHERRY_KV_PAD=<cells>` handling to `llama_kv_cache::get_n_kv`. The environment value is parsed once.
Unset or zero keeps the upstream `max(n_pad, 256)` behavior. A valid nonzero value changes the graph padding quantum to
`max(n_pad, 256, N)`. Values that are not multiples of 256, or that exceed a cache's allocation, warn once and are
ignored for that cache. `BIGCHERRY_PATCH_TRACE=1` logs
`BIGCHERRY_PATCH_HIT patch=1362_kv_pad_prompt pad=<N>` once when the option is active.

## Why 256 and cache bounds remain valid

256 is already the minimum padding quantum in b11552. Requiring the override to be a multiple of 256 preserves that
granularity while leaving the constructor's `GGML_ASSERT(kv_size % n_pad == 0)` and the original `n_pad` unchanged.
The existing result expression remains
`min(cells.size(), max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur)))`: if N does not divide the cache size,
the last padded bucket can exceed the allocation but `std::min(cells.size(), ...)` clamps the graph range to exactly
the allocated KV size. N therefore does not need to divide `kv_size`.

## b11552 consumer audit

- `llama-kv-cache.cpp`: `llama_kv_cache_context::next` receives the larger `n_kv`; `get_k/get_v` expose larger
  in-bounds views and KQ/relative-position inputs iterate the same range. Empty cells are already masked, and allocated
  KV buffers are zero-initialized.
- `llama-graph.cpp`: generic attention mask allocation and `can_reuse_kq_mask` see the larger width. This is the
  intended graph-reuse effect. Position-bucket inputs also grow with `n_kv`; empty-cell values are irrelevant because
  the attention mask excludes them.
- iSWA and hybrid wrappers (`llama-kv-cache-iswa.cpp`, `llama-memory-hybrid.cpp`,
  `llama-memory-hybrid-iswa.cpp`) only forward their underlying KV contexts. Base/SWA caches independently clamp to
  their own allocations; recurrent state is unchanged.
- DSA/MSA/DSV4 wrappers: DSA main/indexer caches and MSA base/indexer caches are ordinary `llama_kv_cache` instances,
  so their masks/views grow consistently. MiniMax M3's batch `cell_blk` input grows to the base cache's `n_kv` and
  maps empty cells only. DSV4 raw/SWA views can grow; its compressed-cache graph sizing and its separate 256-cell
  compression-plan padding do not use this override.
- Hybrid indexers (`llama-memory-hybrid-idx.cpp`, Qwen4Exp and GLM5-Next) use the larger `n_kv` as the sentinel
  one row past the graph's live KV rows. Pool layout is built from actual occupied `seq_pos_get` cells, not from
  `n_kv`, so padding cannot create a real pool.
- State save/restore does not consume `get_n_kv`: `llama_kv_cache::state_write` walks actual non-empty cell ranges,
  and iSWA/hybrid/indexer wrappers delegate that state. Saved state size/content is unchanged.

### Qwen4Exp QSA

QSA cannot select a padding-only pool as a live KV block. `kpool_layout_update` creates pools only from actual
occupied indexer cells. Padded pool entries receive an `-INFINITY` pool mask. If top-k still returns one because
fewer real pools are visible than the fixed selection width, `build_qsa_sel` derives `live_pool` from the selected
score; an invisible `-INFINITY` entry is dead and is remapped to a per-slot dump row beyond `n_kv` before the final
view is restricted to the first `n_kv` rows. Missing tail cells use the same sentinel/dead-slot rule. The Qwen4Exp
assertion that attention and indexer `n_kv` match remains valid because those caches mirror the same size, padding,
streams and slots.

## Correctness and validation status

The change only exposes additional already-allocated empty KV rows behind the existing masks; no occupied cell,
write index, state record, or visible attention entry changes. Bit-identical greedy output is therefore the required
hardware check, not yet claimed as evidence. QFP53 qualification will measure graph reuse and prefill at 8K/24K/98K
for N=2048/4096/8192, plus Qwen3.8-27B. No training qualification is claimed.

## Upstream

Local BigCherry enhancement against llama.cpp b11552 (`23b0202a189c44a54625aadcb37a946dd1d6278d`).
