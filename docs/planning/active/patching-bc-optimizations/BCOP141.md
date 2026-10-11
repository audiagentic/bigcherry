---
id: BCOP141
order: 141
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-11T15:13:22+00:00'
created-by: agent
priority: P2
work: S
---

# MET09: separate MoE ARGSORT, fused tie semantics, and Q8_1 range dedup

## Discovery and disposition

Pinned b11552 Qwen4Exp routes through `ggml_argsort_top_k` (`GGML_OP_ARGSORT`) or eligible fused `topk_moe_cuda`, **not** patch 1294's QSA `GGML_OP_TOP_K`. The fused route already ties by lowest current physical expert ID; a physically reordered GGUF can therefore select different original experts at exact cutoff ties even with deterministic selection. Patch 1281 already contains QFP30's Q8_1 range-dedup scatter for eligible broadcast shapes; the old MET09 assumption of universally disabled dedup is stale. No new router or MMQ implementation is justified without an executed-path trace.

## Ownership, novelty, protected work

MET09 owns first-divergence attribution; MET11 owns B2 post-selection old-to-new ID mapping; MET02/1281 owns range semantics; MET04/1283 and MET08 own EP and placement performance. 1294 owns QSA TOP_K only. No independent MET09/1281/1283/MET11 implementation, benchmark, PR or queue activity was identified in the preceding 12 hours on the inspected branches. This is distinct from recent BCOP140 Vulkan queue work and BCOP139 generic MoE MMVQ geometry. Protected PA49/b11552 patch reconciliation, QFP41/1356, QFP48, QFP49/1360/1361, MTP/1359 and PGC17 transport were not edited. Existing 1343 patch number is occupied; MET11 must not reuse it.

## Bounded next action / terminal gate

One MTP-off, matched original-versus-reordered first-divergence trace: executed fused/ARGSORT route; F32 router scores at cutoff; inverse-mapped original IDs, weights, MMVQ/MMQ/range-dedup path, per-rank partials and logits. Exact cutoff tie with changed old IDs => MET11 B2 mapping validation, not another tie kernel. Identical IDs/weights with numerical drift => arithmetic/dispatch attribution only. No reproduced mismatch => close tie hypothesis; Q8_1 range-dedup already admitted => retire duplicate dedup proposal. No GPU or new BigCherry throughput result in this audit; source-static checks 10/10 after correcting QSA file location, host permutation/F32/dedup fixture passed. See MET09 for technical gates.
