---
id: BCOP198
order: 198
plan: patching-bc-optimizations
state: done
created-at: '2026-10-14T01:17:00Z'
created-by: agent
priority: P2
work: S
---

# QFP02: 1292 k-pool tail-layout admission and ownership reconciliation

**Discovery/change:** 1292's three source anchors still match pinned b11552, and upstream master shares the exact `llama-memory-hybrid-idx.cpp` blob. `kpool_layout_update` retains complete prefix pools and appends the edited tail. Qwen4Exp's order-based pooling bounds the re-scan to a partial pool plus tail; generic position-based pooling can rescan the entire unchanged prefix if no complete pool remains. This limits performance generalization but is not a reproduced correctness defect. `kpool_build_state` representative-generation parity is an independent unqualified safety boundary.

**Owner/consolidation:** QFN01 and validated patch 1292 retain production/implementation ownership. QFP02 is the narrow current-pin and cross-operation qualification owner; it must not create another scheduler/cache/allocator/dispatcher or repeat QFN01 hardware work. QFP04/1295 and QFP17/1332 own distinct QSA mechanisms. QFP02 last independently changed 2026-10-03; QFN01 2026-10-04; 1292 patch directory moved mechanically 2026-10-09. No independent 1292/QFP02/QFN01 work, PR, queued experiment or hardware lane was visible within 12 hours. Active Radiance mailbox/R9700 helper and engine patch-prefix/release capabilities were excluded and untouched. Recent BCOP188-197 address different slices.

**External mechanism:** Upstream llama.cpp issue #30005 (open) concerns mirrored GPU indexer-cache residency, not host pool recomputation. GLM-5.3-Flash PR #27752 (closed/unmerged) would reuse the hybrid-indexer with different row/input semantics; no adoption now. SGLang UnifiedRadixCache component isolation is not an equivalent per-sequence pool algorithm.

**Evidence:** Historical 0504396 Qwen4Exp MTP3 ABBA at ~80K: 77.1/77.3 ms/step base versus 65.5/66.4 and 65.5/65.6 with 1292 (reported -14%); no-MTP greedy parity. This run's source checks 3/3 and host-model differential 120,070/120,070 passed, plus sparse-position prefix-rescan control. No compiled C++, HIP, GPU, logits, graph, multi-request or current-pin benchmark.

**Unresolved action / terminal gate:** QFP02 now specifies exact b11552 patch composition, full-rebuild differential for `seq_rm/seq_add/seq_cp/seq_keep`, restore, unified/non-unified multi-slot, shared-cell representative generations and MTP rollback; then matched current-pin ABBA with CPU layout timing, actual selected work, numerical parity and >=3% exposed opportunity. Retain stock rebuild fallback; reject extensions on any changed logits, stale pooled-row read or missing work. Close the qualification item after a validated terminal result; do not change patch 1292's existing validated status without new evidence.
