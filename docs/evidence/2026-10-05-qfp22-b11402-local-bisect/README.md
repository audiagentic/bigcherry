# QFP22: recovered local b11402 pin-regression bisect (5 October 2026)

This is an **immutable historical Git investigation archive**. It records eight commits that existed only on a local `bisect-b11401` branch; it is not a change to the active llama.cpp pin, an approved source release, or new performance validation.

## Source and capture

- Local branch: `bisect-b11401` in a disposable BigCherry test clone; its tracking ref was `origin/patch-refactor`.
- Base revision: `185ff9fd466861b9ec49bf89e32ee2714ca508e7`. This historical base is **not** an ancestor of the present `main` as inspected on 2026-11-13.
- Local branch tip: `8246eeeeea718e29de5bef63d6e7a536e335dc17`.
- Captured on 2026-11-13 from a clean working tree. The eight commits were authored on 2026-10-05 (local time); no Git history was rewritten.
- `commit-ledger.json` records each complete commit and parent ID, timestamp and subject. `patch-series.tar.gz` preserves the exact `git format-patch` output for all eight commits, in commit order, as a byte-preserving archive (Git line-ending normalization would otherwise change the original patches).
- `records/` contains the four newly created historical release-record JSON files. Each is `stage: pulled`, with no patch audit or promotion. They are **not** added to today's live `releases/` index.

## Purpose and results

Four short-lived source selections were used to compare the b11402 bump against intermediate upstream points. Each pin-selection commit has a corresponding local record commit:

| Pin revision | Pin commit | Record commit |
| --- | --- | --- |
| `b11401` (`a7fb71fab83b...`) | `8bf1af33` | `49b64d3a` |
| `0eb6d9a81` (upstream #29940) | `41e77548` | `f6f5ec00` |
| `2ca15f540` (upstream #29612) | `8f1d0011` | `6da86458` |
| `0bb496dbd` (upstream #29622) | `80fa53d9` | `8246eeee` |

The actual measured findings and uncertainty are already recorded in [QFP22](../../planning/active/patching-qwen-flash-next/QFP22.md) and [BCOP27](../../planning/active/patching-bc-optimizations/BCOP27.md), with a later analysis in [b11552 patch notes](../../../releases/b11552-patches.md). QFP22 distinguishes the larger mixed-batch regression attributed to upstream #29622 from the small dense-27B 32K prefill residual attributed by elimination to #29612; neither inference should be strengthened by this archive.

**Limitations:** the eight commits are test-input bookkeeping, not raw GPU result files or independently validated regression evidence. No local test or GPU run was executed during recovery. The patch-series includes unrelated line-ending normalization in the first pin commit, which can look like thousands of code changes. An ignore-EOL-whitespace comparison shows no substantive changes to `releases/patch-rebase.json`, `releases/pin-transition.json`, `releases/0504396140d1.json`, or `tools/tests/patch/test_1316_node_hash_trace.py` relative to the old base.

## Recovered materialization receipts

Four additional source-materialization receipts were identified under the old clone's ignored work cache. They correspond exactly to the four upstream revisions in this bisect. Because these receipts include content hashes and materialized source-tree identities, they are useful to distinguish the **actual build inputs** from a bare pin name.

- `materialization-receipts.tar.gz` contains the four original `source-metadata.json` files, under their original hashed run-ID subdirectories, with exact bytes retained.
- `materialization-ledger.json` maps those four run IDs to upstream commits, source-tree object IDs and materialization-plan identifiers, and records each original file's SHA-256.
- All four tar members were compared byte-for-byte against the original ignored files on the development host on 13 November 2026.
- These are **source-input receipts**, not benchmark output or a substitute for the QFP22 raw hardware measurements. Compiled copies are not part of this compact tracked bundle.

## Do not merge the experimental branch

The current llama.cpp pin and release ledger have moved on. Replaying the eight commits would move the configured pin backward and overwrite newer metadata. Preserve this compact source/commit record as historical review material only. The production source of truth remains the managed QFP22/BCOP27 notes and today's release metadata.

Each captured payload, including the byte-preserving patch archive, is covered by `SHA256SUMS`. Only after this archive is reviewed and present on `main` should the local-only bisect branch be considered for retirement; no cleanup is authorized by this document.
