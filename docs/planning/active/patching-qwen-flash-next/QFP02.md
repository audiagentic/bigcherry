---
id: QFP02
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T15:20:21.946886+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# 1292 QSA k-pool tail truncation — source-level admission and closure gate

**2026-10-14 BCOP198 audit:** Patch `1292_kpool_tail_truncate` remains validated on historical evidence. QFN01 is the **runtime/patch owner** (already named by 1292's `patch.toml` and SUMMARY); QFP02 owns only the **unresolved cross-operation correctness and current-pin qualification gate**, not another implementation. No new scheduler, allocator, cache or dispatch table.

## Code and data flow

At llama.cpp b11552, `src/llama-memory-hybrid-idx.cpp::llama_memory_hybrid_idx::kpool_layout_update` traverses each sequence's `mem_idx->get_cells(unified ? 0 : s).seq_pos_get(s)`, retaining `sq.{cells,pools,j_next,pos_min,strm}`. Stock rebuilds all cells and pool starts on every stale edit. `mem_idx_stale_set`/`mem_idx_stale_pos` are called by `seq_rm`, `seq_add`, `seq_cp`, `seq_keep`, restore and other cache mutations; edits at/before the first position require a full rebuild.

1292 inserts `bc_tail_cut` before the clean append path. Its algorithm is:
```text
if stale != CLEAN and stale > pos_min and old_cells and new_cells
   and old_pos_min == new_first_position:
    cut = lower_bound(old_cells, (stale, 0))
    if cut > 0:
        old_cells.resize(cut)
        while pools and pools.back + kpool > cut: pools.pop()
        j_next = pools.empty ? 0 : pools.back + kpool
        tail_cut = true
if (stale == CLEAN or tail_cut) and first position unchanged:
    append new_map.upper_bound(last_retained_cell)..end
if cell_counts_differ or (stale != CLEAN and !tail_cut):
    full_rebuild_from_map()
scan_remaining_pools_from(j_next)
```

`llama_memory_hybrid_idx_context::kpool_build_state` then marks pools whose last member is at/after the stale position, plus pools touched by the current ubatch; `rep_gen` deduplicates pooled representative writes for shared cells. `set_input_kpool` consumes those pool/rep indices. **Matching pool starts alone is insufficient:** unchanged representative rows and correctly marked stale/new rows must be proven, including shared-sequence cases.

## Most important performance boundary

`src/models/qwen4exp.cpp` sets `indexer_kpool_by_order=true`. In this rank-contiguous mode, `j_next` after the last complete retained pool leaves at most a partial pool plus the edited tail to rescan. The generic **position-based** branch does not have that bound: when no complete prefix pool remains, `j_next=0` scans the unchanged prefix again. A host sparse-position control with 500 retained cells reproduces that route without changing the result. Therefore 1292 is **not universally O(tail)**, and Qwen4Exp performance cannot be projected to future hybrid-indexer models. This patch changes CPU bookkeeping, not GPU KV/indexer allocation, H2D/D2H transport or graph scheduling.

## Source, history and consolidation

- b11552 and upstream master have identical `src/llama-memory-hybrid-idx.cpp` blob `21ec0b2fe1f358138a3bd3b7813b66654665b2e7`; each of the three 1292 patch anchors matches exactly once. Source checks are not a compiled patch-composition test.
- Historical 0504396 Flash-Next MTP3 ABBA at ~80K: 1291-only 77.1/77.3 ms/step versus 1291+1292 65.5/66.4 and 65.5/65.6 (-14% reported). At 10K -2..3%. Greedy parity at 10K/80K was established **without MTP** using 1294 as a deterministic TOP_K tie control. 1292 README's native-v6 comparison is multi-patch and cannot isolate 1292. No new b11552 GPU timing was measured.
- QFN01 retains production profile and patch state; QFP04/1295 owns selected-cell attention; QFP17/1332 owns QSA chunk masks. Upstream llama.cpp issue #30005 (open, 2026-10-05) addresses mirrored **GPU** indexer-cache row residency, not host k-pool layout. GLM-5.3-Flash PR #27752 (closed unmerged 2026-10-01) proposed another `llama_memory_hybrid_idx` consumer with different row/input semantics; do not port or qualify absent a real merged model. SGLang UnifiedRadixCache's component isolation is a different radix-prefix mechanism and not a GGML pool implementation.

## Cheapest discriminator already run

Source-shaped Python host model: **120,070/120,070** deterministic and boundary cases matched a full rebuild across pool widths 2/4/8/16, both by-order/by-position, duplicate positions, gaps, tail cuts and representative cell/position selection. A separate sparse-position control demonstrates prefix re-scan. These are **host-model** results, not compiled C++, real multi-sequence scheduler state, `rep_gen` parity, GPU, logits, graph replay or inference benchmarks. No change to production patch status follows.

## Bounded next gate (QFN01 implementation owner)

1. **Patch composition:** exact b11552 + production patch stack, offline patch-lint/rebase-check, negative anchor/idempotence and compilation. If anchors diverge, fail closed; do not silently drop 1292.
2. **Host/C++ differential:** add a test fixture at the existing hybrid-indexer test seam (or a minimal disposable C++ fixture) that compares forced full rebuild with 1292 for ordered `(position,cell)` lists, pool starts, `j_next`, representative stream/cell, stale `is_new` and `rep_gen`. Exercise `seq_rm` interior/tail, `seq_add` positive/negative, `seq_cp`, `seq_keep`, state restore, empty/head edits, multiple sequences sharing cells, `-kvu` unified and non-unified, M-RoPE repeated positions, and both pool modes. Include multi-request same-process and multi-ubatch.
3. **GPU correctness:** same prompt/model/pin/quant, 1291+1294 control vs +1292; 10K/80K/160K/220K where VRAM allows, with and without MTP, deterministic greedy and logits/KLD; verify acceptance, actual selected pool work, graph replay and no stale/OOB pooled rows. No P2P assumption; 6900 XT is the draft GPU, XTX/R9700 target ranks.
4. **Performance:** instrument existing CPU profiling for `kpool_layout_update` wall, stale/cut/full-rebuild counts, map-iteration and rescan lengths, GPU idle, end-to-end ms/MTP step. Paired ABBA, independent sessions, same accepted/drafted counts, CI95. If current-pin exposed ceiling <3%, retain existing validated patch without new optimisation. Promote a new configuration only on CI95-low >0, no numerical regression, and >=3% end-to-end improvement versus the best current control; otherwise reject extension and retain stock full-rebuild fallback.

## Terminal disposition

Close QFP02 once current-pin composition and cross-operation pool/representative differential pass, or when evidence proves no exposed work worth additional qualification. Do not duplicate an already queued QFN01 lane. The original patch remains validated on its historical contract independently of this planning gate.

## Change Log

- 2026-10-03T15:20:21.946886+00:00 (created-by): Created by agent
- 2026-10-14T01:17:00Z (BCOP198): Source-level audit, owner split and bounded current-pin gate.
