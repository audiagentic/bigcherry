---
name: bump-radiance
description: Decide whether to move bigcherry's pinned Radiance revision forward and run the bump -- triage upstream's diff against what bigcherry anchors on, run `pin-bump`, re-port libr3's native rewrite blocks, and the hardware checks after. Use when asked "do we bump", to move the Radiance pin, or when upstream publishes a new Radiance tag.
---

# Bump the pinned Radiance revision

Radiance is the engine bigcherry pins (`engines/radiance/recipes.toml`: `pinned`, `pinned-version`). bigcherry
changes it only through managed patches and the components under `engines/radiance/components/` (libr3 compiles
libr4d for gfx11 through `rewrites.txt` and `native/*.rw` block rules that FAIL CLOSED when upstream changes a
line they anchor on). A bump is therefore: find what upstream changed, see which of OUR anchors it touches,
run the `pin-bump` gates, repair what drifted, then measure on hardware. Never skip the triage: it is what tells
you whether the bump is worth doing at all.

## 1. Is there something to bump to? (read-only, a minute)

    git -C /work/bigcherry/upstream/radiance.git fetch -q origin "+refs/*:refs/*"
    git -C /work/bigcherry/upstream/radiance.git tag -l 'v*' | sort -V | tail -5      # newest tags
    grep -n '^pinned' engines/radiance/recipes.toml                                    # what we are on

`/work/bigcherry/upstream/radiance.git` is the shared bare mirror (CI uses the same layout). Do not clone
anything else and do not touch `vendor/radiance` for triage.

## 2. Triage the diff against OUR anchors (no checkout, no build)

    A=v1.16.0 B=v1.17.0; G=/work/bigcherry/upstream/radiance.git
    git -C $G log --oneline $A..$B | wc -l ; git -C $G diff --stat $A $B | tail -1
    git -C $G diff --name-only $A $B > changed.txt

Then intersect `changed.txt` with what bigcherry depends on:

- **libr3 native blocks and rewrites**: every file name under `engines/radiance/components/libr3/native/*.rw`
  (strip `.rw`) and every `r4d_*.{hip,cpp,h}` named in `rewrites.txt`. A hit means a block may drift.
  The exact test, with no checkout: extract only libr4d and run the rewriter's own check against it:

      S=$(mktemp -d); git -C $G archive $B libr4d | tar -x -C $S
      python3 engines/radiance/components/libr3/r3_rewrite.py --source $S/libr4d \
          --rules engines/radiance/components/libr3/rewrites.txt \
          --native engines/radiance/components/libr3/native --check      # {"failures": {}} when all apply

- **Managed patches**: the files each `engines/radiance/patches/*/patch.py` anchors on (rad0002 hand-off and the
  routing-evidence / manifest patches touch `core/engine.cpp`, `core/runtime/ctx.cpp`, `core/mem/*`). Changed
  there = expect an anchor failure, read the commit subjects (`git log --format='%h %s' $A..$B -- core/mem core/runtime core/engine.cpp`).
- **What the commits are about**: skim `git log --format=%s $A..$B`. A bump that is all another model family
  (e.g. Gemma) and a new drafter with no change to the Qwen3.8 paths we serve is a low-value bump; one that
  touches the loan/KV-tier/vram-budget code or the rows we tune (MXFP4 gemm rows, the 8a8 prefill tile) is
  worth doing and worth measuring.

Write the verdict (bump / not now, and why) before running anything heavy.

## 3. Run the bump (hours; detached; a slice worktree)

The orchestrator is `pin-bump`; its reference is `docs/reference/build/PIN_BUMP.md` ("Per-engine bump gates").

    git worktree add -b bump/radiance-<ver> worktrees/bump/radiance-<ver> origin/main
    # in a detached script (setsid nohup), after sourcing tools/env/bigcherry-env.sh with BIGCHERRY_ENVIRONMENT set:
    nice -n 15 python3 -m bigcherry --engine radiance pin-bump <tag> \
        --source radiance --build stock --platform linux-radiance > <run dir>/pin-bump.log 2>&1

Phases, in order: fetch and rebase patches (the `drift` phase also runs the libr3 native-rewrite check), compile
gate (`components-build`, `component-plugin-abi`, `component-tests`), lane build and CPU smoke, optional
`--accept` (hardware acceptance against stored baselines: needs the `gpu` lease). A failure writes
`artifacts/pin-bump/radiance/resume-<tag>/failure.json` in the worktree: read `failure.code` and
`evidence`, fix, then `--resume`.

### Things that bit (2026-11-05, v1.16.0 -> v1.17.0)

- **`pin-bump` moves `vendor/radiance` in the PRIMARY checkout to the target tag.** That is by design (the vendor
  clone belongs to the primary, `BC_PRIMARY_ROOT` must name a primary checkout) and `vendor/` is gitignored, so no
  tracked file changes, but every tool reading that clone now sees the new source. KEEP IT ON THE TARGET while the
  bump is in flight: a `--resume` checks the tree it finds, and with the clone back on the old tag the drift check
  fails with your own new patterns "not found". If you abandon the bump, `git -C vendor/radiance checkout <old tag>`.
  Builds are not affected (they read content-addressed copies under `/work/bigcherry/sources/`).
- **`--resume` needs `--source`, `--build` and `--platform` again** when the run stopped before the compile phase
  saved them (`LANE_BUILD_INPUT_MISSING`).
- **Native `.rw` blocks chain**: a later block's `find` text is the OUTPUT of the earlier block. When upstream edits
  a line (it added `perf_vocab = 0, perf_iqr = 0` to a declaration), update the first block's find AND replace
  with upstream's new text, and the following block's find to the first block's new replace. Re-run the
  extract-and-check from step 2 until `failures` is empty, then commit the `.rw` change in the bump branch.
- A bump that stops on `NATIVE_REWRITE_DRIFT` is the cheap case; a patch-anchor failure means re-anchoring the
  patch (see `bigcherry-patch-author` and `bigcherry-patch-verify`), which can be real work.

## 4. After the gates pass: measure, do not assume

A green bump says it builds and loads. It does not say it is as fast.

- **Parity A/B on Flash-Next** (launcher defaults, four cards, `temperature: 0`, alternate A B A B under the `gpu`
  lease): compare COLD PREFILL and STEP TIME. Do not compare tokens/s across builds: draft acceptance differs
  per build even with the text pinned (`bigcherry-benchmark` for the harness rules).
- **Kernel rows** touched by upstream (rows, prefill tile): `rad-kbench` replay with `kbench_ab.py` against the
  previous build, per row, minutes each.
- **Quality**: KLD and top-1 against BF16 if numerics could move.
- Then `pin-release` (see `docs/reference/build/RELEASES.md`) and the pin PR through the slice workflow
  (`bigcherry-slice-workflow`); release PRs are merged by the lead once clean.

## Rules that apply throughout

Never edit the primary checkout's tracked files; heavy builds `nice`, one at a time, never beside a serving
engine; no machine names or attribution lines in commits or PRs; managed patches and components only (no
hand-edited vendor source).
