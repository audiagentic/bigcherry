# Flash-Next router split-K / cache-local eviction experiment

Archived from `exp/router-splitk-with-cache-evict` (tip `fa6fdca1a`, base historical `a40258234`; last branch-specific commits 2026-10-09/10). This branch predates the main-history rewrite; raw `origin/main...branch` commit counts are **not** a meaningful list of unique work.

## What and why

The branch assembled queue scripts and comparisons for 1357 MoE router split-K with the 1358 cache-local Meta eviction patch, plus patch-ablation, CPU-reference, multi-request decode/draft checks and a 1330 large-prompt QSA mask/ub1024 candidate.

## Outcome and why stopped

**(a): 1357 patch, tests, queuing and promotion evidence already on main; 1358 promoted via #107, and the 1330 prompt-only gate went to main in #77. (b): proposed ub1024/1330 production switch not adopted at 245,760 context (#77). (c): none found after content comparison.** Main retains `engines/llamacpp/patches/1357_moe_router_splitk/`, `tools/lab/flash-next/queue-router-{splitk,with-evict}.sh`, follow-up/ablation scripts, and `releases/evidence/1357_moe_router_splitk-promotion.md`; no patch archive is necessary.

Recorded four-request ABBA for 1357: +3.2% pooled prefill; pooled decode 69.8 → 67.2 tok/s (-3.8%), heavily influenced by changed text/acceptance in one request. Against CPU f32 at 8K, top-1 was 23/24 both on and off. The 1330 ub1024 trial at ~240K showed ~+13% prefill in a preliminary A/B but was **not adopted** in final 245,760-context confirmation. The branch closed because the evaluated pieces were retained or rejected through main PRs; no separate merge of this composite branch was needed.

## Evidence and resumption

- Branch-specific commits `534be549f`, `3968218ab`, `7180db816`, `1890e5172`, `cfd483c3d`, `fa6fdca1a`; compare by content, not ancestry.
- `releases/evidence/1357_moe_router_splitk-promotion.md` (run IDs `rsk1-*`, `rse1-*`, `rsk1-fid`, `om-q36-1357-2-B`), PR #77, PR #107; `docs/planning/active/patching-qwen-flash-next/QFP17.md`.
- Resume from the current 1357, 1358 and 1330 patches and run scripts; revisit changed-text decode and the final context-size gate before enabling any default.
