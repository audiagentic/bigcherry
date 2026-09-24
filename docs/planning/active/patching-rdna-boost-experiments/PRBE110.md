---
id: PRBE110
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-23T09:19:09.142442+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Extract RD07 (and a scoped RD05) from rejected 1203 as new untested patches

## Description

TODO -- extraction of already-verified edits. 1203_rd050607_rdna4_wmma_fa_q6k_mmq was rejected (PA39); RD05 and RD07 passed inside that bundle and per GPT lifecycle review req_f34f50a25c6240fe may only return as NEW untested patches with fresh hardware evidence (the 1203 receipt does not carry over). This plan splits 1203's existing, already-anchor-verified edits (source: patches/1203_rd050607_rdna4_wmma_fa_q6k_mmq/patch.py, read in full 2026-09-24) into two new patch packages: a scoped RD05 (WMMA FA tile_Q reuse race fix only, no RD06 head-576 enablement) and RD07 (Q6_K mmq sub-scale fold). RD06 (config table, DKQ gate, WMMA head gating, softcap) is explicitly excluded from both -- it stays parked pending its own remediation per PA41. No GPT design request was needed: the edits, anchors and rationale already exist verbatim in 1203's patch.py at the current pin (b11126, same pin 1203 was authored against), this is a mechanical re-packaging plus one new RD05-only activation marker.

## Steps

1. Create patches/<next-order>_rd05_wmma_fa_tileq_sync/ (RD05-only): MOVE rd05-k00-sync and rd05-kbc-sync verbatim from 1203's fattn-mma-f16.cuh FilePatch. RECREATE rd0506-atomic-include as an RD05-only <atomic> include edit in fattn.cu (drop its RD06 coupling). REPLACE rd0506-activation-markers with a single standalone RD05 marker (unconditional GGML_LOG_WARN at the BEST_FATTN_KERNEL_MMA_F16 dispatch case, no nested RD06 sub-check) -- do not copy 1203's nested marker verbatim. Verified unique b11126 anchors for this package: `if (np > 1) { __syncthreads(); }`; the `kbc += iter_k; kbc -= kbc % iter_k;` block; the fattn.cu include block; `case BEST_FATTN_KERNEL_MMA_F16: ...`. 2. EXCLUDE from RD05: rd0506-config-table, rd06-dkq-gate, rd0506-softcap-read, rd06-wmma-gating, rd0506-softcap-matrix (all RD06-owned, stay parked per PA41). 3. Create patches/<next-order>_rd07_q6k_mmq_scale_fold/: MOVE rd07-hoist-base-scale, rd07-fold-subscale, rd07-sum-line (mmq-vec-dot.cuh, in that exact order -- rd07-fold-subscale's anchor rides on rd07-hoist-base-scale's inserted text, so order is load-bearing), rd07-atomic-include, rd07-jmax-env verbatim from 1203. RECREATE rd07-activation-marker using the CORRECTED no-forced_J Q6_K block already verified in this item's notes (b11126's mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream); has no forced_J parameter -- do not copy 1203's forced_J-bearing text). 4. REWRITE rd07-perf-cases: do not copy 1203's version verbatim -- its replacement also adds RD05/RD06 FlashAttention perf cases which do not belong in an RD07-only package. Rewrite at the verified unique make_test_cases_perf() tail to retain ONLY Q6_K MUL_MAT shapes plus Q8_0/F16/F32 controls; omit the FA loop entirely. 5. EXCLUDE from RD07 (diagnostic-only, do not carry into either package): rd07-include-numeric, rd07-timing-head, rd07-timing-mid0, rd07-timing-mid1, rd07-timing-tail. 6. Preserve RD07 edit ORDER strictly: hoist -> fold -> sum (fold's anchor depends on hoist's inserted text having already been applied). 7. Write patch.toml for each (state=untested, requires=[]). 8. Run patch-lint and patch-rebase-check for both packages standalone and composed with the production patchset. 9. Diff each new package's edit-id list against 1203's full edit-id list to confirm none of the excluded RD06/diagnostic-only ids leaked in.

## Detailed Solution & Technical Design

RD05 package scope: fattn-mma-f16.cuh's flash_attn_ext_f16_process_tile k00-loop sync fix (rd05-k00-sync, anchor `if (np > 1) { __syncthreads(); }` -> guards `k00 + nbatch_combine < DV/2` too) and its kbc-loop sync fix (rd05-kbc-sync, anchor ends `kbc += iter_k; kbc -= kbc % iter_k;`, inserts a __syncthreads() before it) -- these fix a real tile_Q buffer-reuse race for np==1 configs and are independent of RD06's head-576 enablement (RD06 touches a different function, ggml_cuda_get_best_fattn_kernel's WMMA gating, and a different config table). Because the RD06 config-table/DKQ-gate edits are excluded, the WMMA path stays gated at head<=128 exactly as upstream b11126 has it (RD06 is what raises the ceiling to 576) -- so RD05 alone only changes correctness of the existing head<=128 WMMA path, not its enablement envelope. Activation evidence: since RD06's gating is absent, the combined 1203 marker (which nested an RD06 sub-check) must be replaced with a standalone RD05 marker -- unconditional at the same fattn.cu dispatch site (case BEST_FATTN_KERNEL_MMA_F16, immediately before ggml_cuda_flash_attn_ext_mma_f16(ctx, dst)), proving the WMMA-F16 kernel (which contains RD05's fix) was dispatched. RD07 package scope: mmq-vec-dot.cuh's three Q6_K MMQ warp-kernel edits that hoist row base-scales out of the k01/j0 loop and fold them with per-k01 sub-scales into one f32/element (removing an int-multiply from the hot accumulation line), plus RD07's own activation marker at mmq.cu's Q6_K dispatch case, the GGML_CUDA_MMQ_J_MAX env override in mmq.cuh (test/tuning knob for the perf cases), and the Q6_K/FA perf test cases appended to tests/test-backend-ops.cpp's make_test_cases_perf(). All six edits are verbatim copies of 1203's Edit objects (same anchor/guard/text) since the base files at b11126 are unchanged by dropping the RD06/RD0506 edits (each excluded edit is anchored on disjoint text elsewhere in the same files, confirmed by reading 1203's patch.py: rd0506-config-table is a different code block in fattn-mma-f16.cuh than rd05-k00-sync/rd05-kbc-sync; rd0506-softcap-read/rd06-wmma-gating/rd0506-atomic-include are in ggml_cuda_get_best_fattn_kernel, a different function than the dispatch switch RD05's marker sits in).

## Code Samples & Guidance

RD05 package -- patches/<order>_rd05_wmma_fa_tileq_sync/patch.toml:
```
schema = 1
id = "<order>_rd05_wmma_fa_tileq_sync"
order = <order>
state = "untested"
kind = "enhancement"
origin = "external-fork"
backend = "hip"
plan-item = "RD05"
external-source = "stew675-rdna-boosts"
experiment-contracts = ["RD05-WMMA-FA-CORRECTNESS-BARRIERS"]
requires = []
```
patch.py (fattn-mma-f16.cuh FilePatch): copy `_K00_SYNC_OLD`/`_K00_SYNC_NEW` and `_KBC_ANCHOR_OLD`/`_KBC_ANCHOR_NEW` plus Edit(id="rd05-k00-sync", ...) and Edit(id="rd05-kbc-sync", ...) verbatim from patches/1203_rd050607_rdna4_wmma_fa_q6k_mmq/patch.py lines ~175-202 and ~668-696.

fattn.cu FilePatch (new, RD05-only marker):
```python
_FATTN_INCLUDES_OLD = '''#include "common.cuh"\n#include "fattn-common.cuh"\n#include "fattn-mma-f16.cuh"\n#include "fattn-tile.cuh"\n#include "fattn-vec.cuh"\n#include "fattn.cuh"\n'''
_FATTN_INCLUDES_NEW = _FATTN_INCLUDES_OLD + '\n#include <atomic>\n'

_DISPATCH_OLD = '''        case BEST_FATTN_KERNEL_MMA_F16:\n            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);\n            break;'''
_DISPATCH_NEW = '''        case BEST_FATTN_KERNEL_MMA_F16: {\n            // bigcherry: RD05 activation-evidence instrumentation, not part of the ported fork change.\n            if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {\n                static std::atomic_flag bigcherry_rd05_logged = ATOMIC_FLAG_INIT;\n                if (!bigcherry_rd05_logged.test_and_set(std::memory_order_relaxed)) {\n                    GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=<order>_rd05 path=wmma_f16_dispatch contract=RD05\\n");\n                }\n            }\n            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);\n            break;\n        }'''
```
Both `_FATTN_INCLUDES_OLD` and `_DISPATCH_OLD` are the real, unmodified b11126 anchors (verified: identical to 1203's `_FATTN_INCLUDES_OLD`/`_DISPATCH_OLD` at patches/1203_rd050607_rdna4_wmma_fa_q6k_mmq/patch.py lines 227-243 and 256-258, and unaffected by dropping RD06's edits since those anchor on `_WMMA_OLD`, a different span in `ggml_cuda_get_best_fattn_kernel`).

RD07 package -- patches/<order>_rd07_q6k_mmq_scale_fold/patch.toml:
```
schema = 1
id = "<order>_rd07_q6k_mmq_scale_fold"
order = <order>
state = "untested"
kind = "enhancement"
origin = "external-fork"
backend = "hip"
plan-item = "RD07"
external-source = "stew675-rdna-boosts"
experiment-contracts = ["RD07-Q6K-MMQ-PREFILL-FOLD"]
requires = []
```
patch.py: copy verbatim from patches/1203_rd050607_rdna4_wmma_fa_q6k_mmq/patch.py: mmq-vec-dot.cuh FilePatch (Edit ids rd07-hoist-base-scale, rd07-fold-subscale, rd07-sum-line, lines ~298-421 / 751-780, keep this exact order -- rd07-fold-subscale's anchor rides on rd07-hoist-base-scale's inserted text and rd07-sum-line's anchor rides on rd07-fold-subscale's), mmq.cu FilePatch (rd07-atomic-include, rd07-activation-marker, lines ~432-460/789-810 -- rename BIGCHERRY_PATCH_HIT patch= value to the new package id), mmq.cuh FilePatch (rd07-jmax-env, lines ~464-472/816-826), and tests/test-backend-ops.cpp's rd07-perf-cases Edit only (NOT rd0506-softcap-matrix, which is RD05/06-owned) -- lines ~615-650/896-908.

## Files

New: patches/<order>_rd05_wmma_fa_tileq_sync/{patch.toml,patch.py,README.md,SUMMARY.md}; patches/<order>_rd07_q6k_mmq_scale_fold/{patch.toml,patch.py,README.md,SUMMARY.md}. Reference-only (read, not edited by this plan): patches/1203_rd050607_rdna4_wmma_fa_q6k_mmq/patch.py (source of the copied edits); ggml/src/ggml-cuda/fattn-mma-f16.cuh, ggml/src/ggml-cuda/fattn.cu, ggml/src/ggml-cuda/mmq-vec-dot.cuh, ggml/src/ggml-cuda/mmq.cu, ggml/src/ggml-cuda/mmq.cuh, tests/test-backend-ops.cpp (upstream files touched by the new packages, unchanged at b11126 vs when 1203 was authored).

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint` on both new packages; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay <rd05-id> --source bigcherry-tuning` and same for the RD07 id, both standalone and composed with the current production patchset (they must not require or conflict with 1203, which stays rejected/inert). Unit: existing test-backend-ops FLASH_ATTN_EXT and MUL_MAT correctness cases must stay green after applying each package alone. Hardware (on Brutus via python -m bigcherry.patch.validation_campaign, not run here): RD05 -- FA correctness at head<=128 WMMA configs on gfx1201 with BIGCHERRY_PATCH_TRACE=1, confirm the new marker fires and FLASH_ATTN_EXT parity holds under forced np==1; RD07 -- Q6_K prefill throughput at the qwen35-27B mmq shapes (the rd07-perf-cases test_mul_mat entries) on gfx1100/gfx1201, correctness via test-backend-ops MUL_MAT Q6_K, fresh evidence recorded independently of 1203's PA39 receipt per req_f34f50a25c6240fe.

## Effort & Risk

M; low code risk (verbatim copy of already-anchor-verified edits) but real process risk is under-scoping the split (accidentally carrying an RD06 edit, or losing RD07's edit-order dependency chain in mmq-vec-dot.cuh, which would break the anchors). Mitigate by diffing the new packages' edit ids against 1203's full edit-id list to confirm none of rd0506-config-table/rd06-dkq-gate/rd0506-softcap-read/rd06-wmma-gating/rd0506-activation-markers/rd0506-atomic-include/rd0506-softcap-matrix leaked in.

## Standards

No legacy/backward-compat shims (do not keep 1203 alive as a fallback path); new patch ids only, no reuse of 1203's promotion/rejection history as evidence (req_f34f50a25c6240fe); package-only edits per bigcherry-patch-author conventions.

## Acceptance Criteria



## Notes

Follow-on to PA39's terminal rejection of 1203. PA41's RD06 remediation deferral is unchanged.

2026-09-24 relevance at b11126: source patch patches/1203_rd050607_rdna4_wmma_fa_q6k_mmq/patch.toml state="rejected" (confirmed read); its patch.py read in full and is the exact source of the copied Edit objects. No GPT design request needed -- this is a verified mechanical extraction, not new design; anchors are identical to 1203's own (same pin, disjoint code spans from the excluded RD06 edits).

2026-09-24 CORRECTION after direct re-verification against the live b11126 mirror (git -C work/upstream/llama.cpp.git show b11126:ggml/src/ggml-cuda/mmq.cu): 1203's own `_MMQ_SWITCH_OLD` anchor text `mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream, forced_J);` does NOT match current b11126 -- the real line 45 is `mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);` with NO forced_J parameter (grep for "forced_J" across mmq.cu/mmq.cuh at b11126 returns zero hits; this parameter does not exist in the current pin's mul_mat_q_case signature, unlike whatever base 1203 was originally anchored against). This means 1203's rd07-activation-marker edit (and by extension any verbatim copy of it) would fail to apply at b11126 as-is -- step 4 of this plan must use the CORRECTED anchor/replacement below, not a literal copy of 1203's _MMQ_SWITCH_OLD/_MMQ_SWITCH_NEW. All other RD05/RD07 anchors (rd05-k00-sync, rd05-kbc-sync in fattn-mma-f16.cuh; rd0506-atomic-include/rd0506-activation-markers base text in fattn.cu; rd07-hoist-base-scale/rd07-fold-subscale/rd07-sum-line in mmq-vec-dot.cuh, confirmed unique -- a second near-identical `x_sc = (const int *) x_df + MMQ_TILE_NE_K/QI6_K` block exists at mmq-vec-dot.cuh:1077 but its 4th anchor line differs (`(ntx*tile_A::I)` vs `rows_per_warp`), so the anchor is unambiguous; rd07-jmax-env in mmq.cuh) WERE independently re-verified present and byte-identical at b11126 by direct grep/show. Corrected mmq.cu block for step 4:
```python
_MMQ_SWITCH_OLD = '''        case GGML_TYPE_Q6_K:\n            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);\n            break;'''
_MMQ_SWITCH_NEW = '''        case GGML_TYPE_Q6_K: {\n            // bigcherry: RD07 activation-evidence instrumentation, not part of the ported fork change.\n            if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {\n                static std::atomic_flag bigcherry_rd07_logged = ATOMIC_FLAG_INIT;\n                if (!bigcherry_rd07_logged.test_and_set(std::memory_order_relaxed)) {\n                    GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=<order>_rd07 path=q6k_mmq_dispatch contract=RD07\\n");\n                }\n            }\n            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);\n            break;\n        }'''
```
This is the same site/rationale as 1203's marker, just without the (non-existent at b11126) forced_J argument. Whoever implements this package MUST re-grep every anchor against the live mirror before finalizing patch.py, not trust 1203's stored text -- this correction is itself proof that source drift between when 1203 was authored and the current pin is real and must be re-checked per-anchor, not assumed.

2026-09-24 GPT review req_b43762f844fb40b3 applied: this item's earlier forced_J correction (mul_mat_q_case<GGML_TYPE_Q6_K> has no forced_J at b11126, already recorded in prior notes) is confirmed correct by GPT. Corrected the steps, which still said 'copy verbatim': specified the exact per-edit RD05/RD07 extraction map (move/recreate/exclude per edit id), since rd07-perf-cases in particular cannot be copied verbatim -- 1203's version also adds RD05/RD06 FlashAttention perf cases that don't belong in an RD07-only package -- and RD07's edit order (hoist->fold->sum) must be preserved since later edits anchor on earlier edits' inserted text.

## Change Log

- 2026-09-23T09:19:09.142442+00:00 (created-by): Created by agent
- 2026-09-24T04:37:10.302016+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-24T04:41:54.772832+00:00 (updated-by): Updated: section:notes
- 2026-09-24T05:10:12.282151+00:00 (updated-by): Updated: section:steps, section:notes
