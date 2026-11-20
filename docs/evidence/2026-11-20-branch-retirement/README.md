# BigCherry branch retirement — archival evidence, 2026-11-20

This is **inert historical preservation**, not a production patch release. [inventory.json](inventory.json) captures every remote branch tip, historical merge-base, divergence, and non-equivalent commit identity. Branches with unmatched changes have exact compressed, binary-safe Git diffs in [historical-patches](historical-patches/) with SHA-256 digests. Original worktree data was not changed.

Each archived patch is made with git diff --binary --full-index MERGE_BASE BRANCH_TIP. To reconstruct, check out the recorded merge-base in an isolated historical worktree and apply the decompressed patch there. Never apply obsolete experiments to current main.

## Failure and success evidence

- **RSC11 P1:** obsolete native sidecar; [Nov 13 reconciliation](../2026-11-13-unpublished-branch-reconciliation/README.md) establishes original-tree equality, tests, and incompatibility with current P1.
- **RHP19:** earlier 240-second timeout and free-text drift progress, superseded by structured bounded progress in PRs #828, #833 and #836. Prior 12/12 offline result is not GPU acceptance.
- **XDP03:** PR #787 closed unmerged; planner and selector instead merged through PR #786.
- **RCS14:** v26 GPU contract checkpoint superseded by live v26 cutover PR #1014.
- **RAD49:** old v1 diagnostic and capture path, never to overwrite v2 serving contracts.
- **ENG06:** scale-broadcast kernel had no gain at 248K; [evidence](../2026-11-19-eng06-bf16i8/README.md) retained.
- **MRC, PRQ06, ENG05/06, Strata, inventory:** successors integrated through PRs #1195–#1247; original patches remain archived as needed.

## Branches older than five days

| Branch | Original tip | Disposition |
|---|---|---|
| archive/rhp19-drift-envelope-20261113 | e76e57daf63f | Patch archived; retire after merge |
| archive/rsc11-s13-p1-engine-native-20261113 | e4f6c4bd79c1 | Patch archived; retire after merge |
| exp/xtx27-stage2-quant-plan | 9381da997a21 | Patch archived; retire after merge |
| feat/rcs14-v26-gpu-contract | fa5fa96f2fcf | Patch archived; retire after merge |
| fix/rad49-tp-tail-kl | ef877101a23c | Patch archived; retire after merge |

## Uncommitted worktree files safely copied

| Branch | Original | Archive | SHA-256 |
|---|---|---|---|
| docs/radiance-maint-20261119 | docs/reference/engines/radiance/README.md | docs/evidence/2026-11-20-branch-retirement/worktree-snapshots/docs-radiance-maint-20261119.README.patch.gz | 11e5f32d9398700ce1efbbabb2c36cf5a298f2d4af633d7652af7903560b48c7 |
| agent/legacy-followups | tools/lab/radiance/fp-ab.sh | docs/evidence/2026-11-20-branch-retirement/worktree-snapshots/agent-legacy-followups.fp-ab.sh.txt | 0bf74aac64ba3f9501d843b8cfabdd7feeea11c7ef9241db9de75d5787f16d04 |

These are copies, not execution-ready tools: fp-ab.sh is archived as inert text. The old Radiance README change remains a patch. The original worktrees have not been modified.

## Holds and retirement gate

Keep open TP3 PR #1190, active Radiance 2.2 bump, and release-please automation branches. Leave local IQ6R qualification, radiance-22-candidate, and planning wave4 untouched. The planning worktree contains 15 tracked changes and 13 untracked files.

Do not delete refs before the archive PR is confirmed merged and its SHA checks pass. Before deleting any remote ref, compare its fresh head to the recorded tip; do not delete moving heads or any open PR. Worktree removal belongs to its owning agent and must never be forced. Planning and Ledger MCP records must not be fabricated.

The untracked lab script snapshot has exactly two machine-local calibration paths replaced by BC_WORK_ROOT to satisfy the repository path policy. The SHA-256 of its original bytes remains in snapshots.json. No other lines were changed.
