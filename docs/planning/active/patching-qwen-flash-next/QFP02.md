---
id: QFP02
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T15:20:21.946886+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P3
work: S
---

# 1292 QSA k-pool tail truncation

## 2026-10-12 implementation audit: preserve the validated fast path; bound residual position-mode scan

**Disposition:** Patch `1292_kpool_tail_truncate` is already **validated** and part of the Flash-Next production profile. The old "fold review / contract evidence for promotion" steps below are historical, not an open promotion request. QFP18 supplied promotion and recorded incremental A/B; QFP02 is the technical owner of only a conditional **non-order k-pool tail-rescan** remainder. Priority P3; no new package, no default flip, no hardware queue.

### Measured vs source-derived vs modeled

- **Measured BigCherry, 2026-10-03:** two historical ~80K MTP3 ABBA series, 1291-only ~77.1/77.3 ms per step versus 1291+1292 ~65.5/66.4 and ~65.5/65.6 ms, approximately **-14% step latency**. At 10K the recorded reduction was ~2-3%. Greedy text was identical in the stored 1294-controlled comparisons. These are **not** new measurements or a standalone b11474 hardware requalification. The v6 vs native +29%/+45% decode results include many patches and must not be assigned to 1292.
- **Source:** `engines/llamacpp/patches/1292_kpool_tail_truncate/patch.py` inserts `bc_tail_cut` in `src/llama-memory-hybrid-idx.cpp::llama_memory_hybrid_idx::kpool_layout_update`. It lower-bounds the stale position, keeps cells before the edit, removes pools intersecting the edit, and resets `sq.j_next` to the end of the last surviving pool (or zero). It then uses the native append/re-pool path. The base `mem_idx_stale_set` and `mem_idx_stale_pos` contract is enforced by `seq_rm`, `seq_cp`, `seq_add`, `seq_div`, `state_read`, and `state_drop`; `llama_memory_hybrid_idx_context::apply` and `kpool_build_state` consume the resulting pool identities and representative cell indices. Do not bypass these lifecycle gates.
- **Architecture/route:** `src/models/qwen4exp.cpp::load_arch_hparams` explicitly sets `indexer_kpool_by_order=true`. In that production route, pools are contiguous ranks; resetting to the last complete pool is already the correct constant-tail resumption. The potential residual is the **other**, position-aligned branch of `kpool_layout_update` (e.g. a qualifying non-order hybrid indexer), not the measured Qwen4Exp fast path. This is host map/vector work, not a HIP kernel, PCIe or GPU-P2P optimisation.
- **Source-level opportunity, not observed slowdown:** For a position-mode sequence with no surviving complete pools, `sq.j_next=0` makes each tail edit re-scan the unchanged prefix. With sparse/nonconsecutive positions this can remain O(n_context) despite 1292. It is a redundant CPU scan, **not** a proven wrong pool, cache corruption, or BigCherry performance regression.
- **Cheapest host discriminator run in this audit:** A standalone Python model of the pinned full-rebuild, existing tail-cut and proposed resume-bound rules tested **50,000** seeded unique-position and **20,000** seeded duplicate-position prefix-preserving edit cases, across pool sizes 2/3/4/5/8/16 and both order modes: **zero pool-layout mismatches**. In a synthetic 80,000-cell, every-other-position, kpool=16 tail edit, the existing position-mode rule scanned **79,985** candidate starts; the bounded-resume model scanned **10**, with identical pool lists. These counts are algorithmic fixture iterations, **not measured CPU time, GPU time, or end-to-end gain**. The host model does not cover all real seq-sharing, generation-mark, concurrency or device cases.

### Implementation-ready conditional gate (no speculative patch yet)

1. **Stop/go before implementation.** Reuse existing CPU `perf`/step-timing producers on a real `indexer_kpool_by_order=false` hybrid model with repeated interior/tail edits. Require (a) this route actually executes patch 1292, (b) at least 5% of **non-overlapped step wall time** in `kpool_layout_update` candidate scanning, and (c) an Amdahl best-case end-to-end ceiling >=3%. If any is absent, close the residual without code/hardware work; do not infer a gain from the synthetic 80K fixture. Do not borrow active Radiance n-gram, native attention, QFP15 or QFP53 lanes.
2. **One local extension only if the gate passes:** Within the existing `bc_tail_cut` block, capture `old_j_next` before truncation and, **only when `!indexer_kpool_by_order`**, use the candidate resumption index:
   `last_end = sq.pools.empty() ? 0 : sq.pools.back() + kpool;`
   `boundary = n_cut >= kpool ? n_cut - kpool + 1 : 0;`
   `sq.j_next = max(last_end, min(old_j_next, boundary));`
   A candidate start strictly before `boundary` cannot include a changed cell at or after `n_cut`; the old scan already examined starts before `old_j_next`. Preserve existing `n_cut>0`, unchanged `pos_min`, cell-count fallback, and full-rebuild branches. Never use this bound for by-order mode. Do not create a second pool cache, scheduler, generation table or runtime knob.
3. **Required host/reference tests:** Extend existing patch mechanics infrastructure with a deterministic pure-reference `kpool_layout_update` model and a compiled/source-bound test where feasible. Cover empty/one-cell sequences, stale at first position, pool-boundary +/-1, no surviving pools, long gaps, duplicate positions with distinct cell IDs, exact-tail and interior edits, `seq_cp` sharing, `seq_add` shifts, `seq_div`, state restore, unified and non-unified streams, two sequences interleaved and multiple consecutive edits. Assert identical `sq.cells`, `sq.pools`, `n_pool_real`, representative rows, `n_new`, `pool_cells` and generation marking versus forced full rebuild. Compare the *post-next-append* state, not only one update. Missing activation or a single mismatch rejects the extension.
4. **Only after host parity:** On an eligible isolated gfx1100 or gfx1201 serving lane, keep the same binary and fixed workload; run 10+ paired ABBA rounds across >=4 sessions, same-process repeated requests, MTP acceptance, greedy tokens and full logits/KLD, 8K/80K contexts and pool-boundary cuts. Record CPU scan candidates, `kpool_layout_update` CPU time, end-to-end step time, GPU work and VRAM. CI95 lower bound >=3% end-to-end improvement and <=1% regression on Qwen4Exp by-order / non-MTP controls; otherwise reject and retain validated 1292 unchanged. gfx1030 auxiliary GPU and mixed no-P2P topology are controls only; this CPU-only hypothesis does not justify changing placement.
5. **Ownership and current upstream:** QFP02 owns the tail-cut algorithm; QFN01 owns the Flash-Next operational profile, QFP18 the historical promotion, and QFP03/QFP04/QFP17/QFP53 their separate QSA mask/attention/graph work. Do not modify those active capabilities here. Pinned b11474 and upstream `master` currently have the **same** `src/llama-memory-hybrid-idx.cpp` blob (`21ec0b2f...`); upstream has not absorbed BigCherry's 1292. Upstream llama.cpp PR #30023 consolidates context-state testing, but does not implement this resumption bound. vLLM's KV coordinator has explicit MTP/prefix-tail block-boundary rules, and SGLang issue #23392 describes shared-prefix/cancellation corruption: use as **lifecycle test precedents only**, not ports or AMD speed claims.
6. **Documentation ownership:** Patch `SUMMARY.md` historically says `Plan item: QFN01`; this audit records **QFP02 technical / QFN01 operational** without changing validated patch metadata or the production recipe. Reconcile `patch.toml` `plan-ids` only through the normal patch-catalog owner if a future code change is approved.

Sources: https://github.com/ggml-org/llama.cpp/pull/30023 ; https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_coordinator.py ; https://github.com/sgl-project/sglang/issues/23392 . No C++/HIP build, repository pytest or GPU benchmark was run in this audit.

---


## Description

Patch 1292_kpool_tail_truncate (evaluated, in production profile): truncates the QSA compressed k-pool view to the filled tail instead of the full allocated context, so pooled-key work scales with used context rather than -c.

## Steps

1. Fold GPT review (edge cases near the threshold, empty pools). 2. Contract evidence for promotion.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Greedy identical vs untruncated at 32K/80K (with 1294 for determinism); ms/step ABBA.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Evidence: -14% ms/step at 80K (flashnext ABBA, 2026-10-03). Part of the production candidate measured +26.5% at 80K together with 1291/1294/1297. GPT code review in flight: req_af96db92b8e34cfe.

## Change Log

- 2026-10-03T15:20:21.946886+00:00 (created-by): Created by agent
