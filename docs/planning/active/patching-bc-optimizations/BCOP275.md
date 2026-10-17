---
id: BCOP275
order: 275
plan: patching-bc-optimizations
state: done
created-at: '2026-10-17T06:12:00Z'
created-by: agent
priority: P1
work: S
---

# Reconcile published and failed BCOP audits into the existing development branch

**Purpose:** One documentation reconciliation, not another optimisation proposal. Target is the already independently active `feat/men12-radiance-17`; do not create a BCOP-only branch, merge unreviewed bulk addenda, change a scheduled task or touch the concurrently developed MEN12/Radiance capability.

## Publication inventory (checked 2026-10-17)

- **Published but not merged to main:** unique one-commit historical documentation branches BCOP223, 225, 227, 228, 234, 235, 236, 237, 238, 239-published, 241 and 252. A branch existing is not evidence its changes reached `main`. The alternate BCOP233, BCOP239-disposition and BCOP242 branch refs had no unique ahead commit in the checked comparison.
- **Commit objects without matching published BCOP branch refs:** independently fetched BCOP265 `3e2796d4cb38`, BCOP266 `a8a535d097c0`, BCOP267 `8d8f5ba0e09`. These objects are not branch publication and were not cherry-picked.
- **Unpublished/failed:** BCOP268-274 have draft/fixture records but no matching BCOP publication branch; prior writes failed at GitHub blob/ref steps. BCOP253-264 likewise require individual commit/ref verification before anyone calls them published. Do not infer a commit from a local draft file or a created blob.
- **Preserve provenance:** leave old branches for normal cleanup after reconciliation/PR review. No force push, deletion, new PR or protected-`main` modification in this task.

## Trimmed, authoritative decisions published on this one branch

| Owner | Change | Evidence and scope |
| --- | --- | --- |
| `1357_moe_router_splitk/README.md` | Correct obsolete 'Not promoted' heading | 2026-10-10 promotion receipt supports profile-scoped +3.2% prefill versus SGEMM; greedy text differs, pooled decode 69.8->67.2 tokens/s. This is NOT a partial/reduce fusion gain. Commit `ec80d574`. |
| `MET08.md` | Replace obsolete model-copy/retest steps with **pre-write fail-closed** tensor/profile/shard validation and transactional GGUF derivation | Current `expert-place.py` edits copied shards before its completeness check; fused-only and duplicate-name cases can pass. No observed corrupted model. MET11/MET09 retain mapping/divergence ownership. Commit `5997c745`. |
| `QFP29.md` | Require actual IU4 target-verify kernel witness before ROCmI4 W4A4 MTP port/benchmark | Historical four-token target verification is <=5 rows; external fork documents <=8-row W8A8/MMVQ. Fork selector/actual executed trace not yet reverified. Prefill is independent. Commit `2b9d4786`. |
| `QFP48.md` | Retire the stale 1357 split-K partial->reduce adjacent-fusion candidate | Partial workgroups cannot use a block barrier as a global fence; measured 1.93 s was partial GEMM, not proven removable reduction cost. Keep QFP36/1357 as sole implementation owner; conditional reopening requires measured >=3% exposed E2E reduction time and safe graph epoch protocol. Commit `d0a63f6e`. |

**Not imported:** longer BCOP239/241/252 addenda, old 223-238 branch hunks, and speculative 265-274 kernel designs. Do not repeat the already-measured negative RAD06 scalar MoE or neutral 1314 CPU-root fusion as hypothetical free speed. Missing current-pin GPU evidence is a defer/reject gate, not licence to queue duplicate experiments. THA33's historical gfx1201 MMVF crash remains open, and PEF01's exact gfx1100 Q6_K J112 quarantine stays in force; this reconciliation changes neither owner.

**Done criterion:** those four edits plus this ledger are reachable from the *same* live branch; source/bench/patch files were not altered by this cleanup. No new GPU run, HIP build, GGUF copy, mock or numerical benchmark was performed here. Historical performance is explicitly labeled. A merge to protected `main` still needs normal checks and review.
