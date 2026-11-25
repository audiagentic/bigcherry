# Pin-bump procedure (llama.cpp pin movement)

The procedure for moving the llama.cpp pin, and the completion rule that says
when it is done. Companion to `bigcherry --engine llamacpp pin-status` (RE48), which is the guard
that names the current state at every step.

## Why the ordering is load-bearing

A bump legitimately spans several hours and several tools. The state in between
is not an error — but it must be *nameable*. The engine that names it
(`pin-status`) distinguishes:

- **consistent** — vendor checkout is at the pinned revision (always
  content-unverified: patch verification is audit's job; dirty counts are
  reported, never judged)
- **uncommitted-transition** — the transition marker exists in the working
  tree but is not committed: the bump is declared, but not yet declared
- **mid-rebase** — a **committed transition marker**
  (`releases/pin-transition-llamacpp.json`) declares the move from the current vendor
  revision to the pinned revision, and the declared base's release record is
  not broken; the move is in flight
- **drift** — vendor checkout is at a revision the pin never declared (a
  release record for that revision is *evidence*, not a transition: without
  a committed marker the state is drift, with sub-reasons stale-marker /
  marker-mismatch / broken-base / no-transition)

Every pin command acts on one engine and needs the global `--engine` option
(`bigcherry --engine <engine> pin-status`). The engine's `engines/<engine>/engine.toml` names its upstream, its
vendor checkout and its recipes file (which holds the pin); each engine has its own transition marker
`releases/pin-transition-<engine>.json`, so one engine's bump never reads as another's drift. `pin-release` and
`release-notes` additionally need a release line for the engine: an `[engine.<name>]` table in
`config/release.toml` and a package in `release-please-config.json` (`engines/<engine>`); an engine without them
is refused with that message. The release stages after a bump (ledger, notes, release-please PR, tag, pin-release
record) are common to every engine and described in [RELEASES.md](RELEASES.md). This document describes the
llama.cpp bump.

The transition that turns "drift" into "mid-rebase" is **committing the pin
move and the transition marker together, before moving the checkout**.
`bigcherry --engine llamacpp repin` writes the marker atomically with the pin line; the commit
is the declaration that a bump is in flight, and every tree that shares the
bigcherry history reads the same state from it. A manual
`git -C vendor/llama.cpp checkout <sha>` that skips the pin move is exactly
the state the guard exists to catch — it is how a stale-trace
incident occurred (evidence captured on b10257 while the pin said b10502):
a release record for the stale revision existed, and "record exists" was
once the only rule — which is why the marker supersedes it.

## Read-only pre-bump triage

Before moving a pin, run `python3 -m bigcherry --engine <engine> pin-triage <target-ref>`.
Add `--json` for a machine-readable report. This only reads the
controller and vendor tree; it may fetch missing refs into the shared bare
upstream mirror. It reports current pin, diff stat, changed rewrite
anchors, changed registered patch targets, core commit subjects, and exact
source rewrite drift with closest target-source lines. Its verdict is
`clean`, `coupling-affected`, `drift-only` or `patches-affected`
with counts and, for Radiance, `coupling_failures` and `upstream_abi`. Non-clean
results exit zero: triage is advice, not a bump gate. Usage and environment
errors exit 2.

### Radiance-specific readiness and acceptance

The Radiance recipe pin, patch selections and build profiles are in
`engines/radiance/recipes.toml`. Follow the current
[bump-radiance skill](../../../.claude/skills/bump-radiance/SKILL.md)
and the live [coupling checklist](../engines/radiance/BUMP_COUPLING.md),
not a frozen patch-count table. This subsection is Radiance-only;
the llama.cpp repin/transition procedure below is independent.

1. **Triage an exact revision.** Run
   `python3 -m bigcherry --engine radiance pin-triage <ref> --json`,
   review changed patch targets and libr3 `native/*.rw` anchors,
   and record upstream ABI and exported layout fingerprint. The verdict
   is advisory; missing ABI/coupling/rewrite compatibility blocks the bump.
   Native rewrite rules whose target source is created by a managed patch
   (the `bc_*_impl.h` files) cannot be judged against pristine upstream, so
   triage lists them as "checked on the patched tree after rebase" instead
   of drift; the `pin-bump` drift phase checks them after the patches apply.
   A missing source that no registered patch creates is still drift.
2. **Prove every composition.** Once the new exact source is checked out,
   run `patch-rebase-check --source NAME` separately for each current
   recipe `[source.*]` (`radiance-native` as the unpatched control, `radiance`,
   `radiance-diag`, `radiance-routing-evidence`, `radiance-vtp44-evidence`), then
   `patch-rebase-check --all` across the non-rejected registry.
   An unselected patch failure is still a catalog disposition. Archive
   upstream-superseded packages **whole and unchanged** with a reason,
   successor and evidence row in `patches/archive/ARCHIVE.md`; do not
   re-port obsolete patches simply to satisfy the check.
3. **Minimise attachment drift.** Run
   `python3 -m bigcherry --engine radiance patch-footprint --check`
   before modifications; after an exact composition change regenerate
   `patch-footprint`, record upstream anchor counts before/after, and
   rerun `--check`. Prefer owned typed hooks and the common
   `libr4d/bc_kv_rows.h` registration inventory for new KV geometries
   over repeated upstream row edits. Kernel constraints must reflect
   actual kernel limits, including rank-local shapes.
4. **Check the installed model roster before GPU.** All managed
   `engines/radiance/configs/NAME.yaml` files are single native
   Radiance YAML + `x-bigcherry` v1: inspect every file with the
   matched engine using `bc-serve explain NAME --native-config
   engines/radiance/configs/NAME.yaml --model-root ${BC_MODEL_ROOT}
   --engine-binary <candidate-binary> --json`. Do not use separate
   layouts/catalogues or managed `--set`. Explicitly inspect upstream
   **retired weight encodings** and per-rank TP ownership against the
   replacement container: IQ8R uses 256-column int8 scales regardless
   of its 128-wide Hadamard rotation. Preflight before device allocation.
5. **Build and test one complete runtime.** Use
   `python3 -m bigcherry --engine radiance pin-bump <ref> --source
   radiance --build stock --platform linux-radiance`
   (resumable with `--resume`). Gates include the component coupling
   and libr3 native rewrite checks, engine and component compilation,
   CTests, plugin ABI + **layout fingerprint**, lane build and CPU smoke.
   The serving contract is **edition 3 only**, generated from the flat
   `components/serving/contract/serving.toml`; run
   `bc_contract_check` to enforce generated-v3 schema/semantic
   fingerprint equality, plus v3 writer/reader/replay tests. Bind sealed
   exact YAML bytes, root, parser/build and fresh measured admission.
   `bc-serve --work-root ${BC_WORK_ROOT}` enables the named-config store:
   a named start reuses the last measured plan of its config (the model is
   identified by path, size and mtime and never read). A reused plan is not
   trusted: resource admission and the live room check run on every start,
   and the engine refuses a plan that does not fit; that is the safety net,
   so a stored plan needs no revalidation of its own.
   A start under a new build or a new work-root prefix carries the stored
   measurement instead of measuring again: it logs that the stored manifest
   is reused under this build, re-seals it, and records the source in
   `plan_carried_from` (`carried-from.json`); no measuring engine runs.
6. **Hardware acceptance before release.** Qualify each shipped
   config from the installed bundle (canary, stop/drain, and a second
   named start with `--work-root`, which is the reuse path), with
   helper-on/off and mixed-rank paths
   wherever the roster requires them. The bump records ONE fresh
   measurement per new build, in a separate work root, and reports how its
   plan differs from the previous build's; plan differences are reported,
   they do not gate the A/B. For model-quality KLD/top-1 use
   a temporary YAML arm with `num-speculative-tokens: 0`; leave
   production draft depth unchanged for throughput tests. Compare
   matched old/new model, KV dtype, prompts and topology with
   interleaved A B B A pairs for prefill, TTFT and decode. Two launches
   per arm is the minimum the baseline tolerance needs (it is derived
   from the spread between the old launches; one launch is refused);
   run a third pair when a result lies within the old arms' spread or
   an arm fails. Long-prefill/long-8 may be bimodal: evaluate spread
   and repeat inconclusive comparisons instead of gating on a single pair.
   No release or production pin change until the exact acceptance
   matrix and installed/relocated runtime pass.

7. **Sync the checkouts after the release.** When the release PR is merged
   and its runtime bundle is published, and the bundle is installed and
   smoke-tested natively:
   - In the primary checkout run `git fetch` then `git pull --ff-only`
     only; never edit it. Confirm `HEAD` equals `origin/main` and
     `python3 -m bigcherry --engine radiance pin-status` reports
     **consistent** (the vendor clone is at the pinned commit; `pin-bump`
     moves it during the bump, and a pull of the new pin must leave the two
     agreeing).
   - Close the bump's gateway sessions, then remove the task worktrees with
     `audiagentic worktree finish` (see [WORKTREES.md](../../guides/WORKTREES.md)).
     Archive evidence and scratch you want to keep first; `finish` refuses
     to discard ignored files without `discard-ignored N`.
   - Record the new pin in the plan and ledger.

### Lessons from the 2.2 bump (RUM09)

- **Read the engine's KV sizing before measuring.** The 2.2 build first
  failed one gated decode metric (long-8 +16.6%). The cause was the GPU KV
  pool: 17.6 GiB a card against 20.97 GiB on the old build, because the
  pool refit fell back to the calibrated minimum whenever independent
  whole-MiB rounding missed the budget by one MiB. Compare the engine log
  line "the cache can grow to N GiB a card" between the old and new build
  first; it is the quickest explanation for a warm (prefix-hit) decode
  difference, which cold and aggregate numbers hide.
- **Tracing is not free to add later.** The old owner refused lab tracing
  and the 2.2 owner refused native-config plus `--lab-experiment`, so no
  per-rank timing could be taken when it was needed. Per-op rank-0 timing
  from the phase-trace tooling is what located the slowdown to paged
  attention. Keep a native-config-compatible phase trace available.
- **Paired-run harness.** Seed a stored manifest index only into the old
  arm (the 2.2 owner rejects the old format). Each arm writes run records
  under its own work root: merge them into the environment's work root
  before `baseline` and `compare`. Take a short per-arm `gpu` lease. The
  arms of one comparison share a work root, so both run the same measured
  plan (the named-config store hands the second arm the first's plan); a
  fresh measurement of a new build goes in a work root of its own. Do not
  reuse a work root across differing configs.
- **Triage at the current pin must be clean.** Run `pin-triage` at the
  pinned commit; a non-clean verdict there is a defect in the check, not
  upstream drift (the patch-owned rule false positives were found this way).
- **`audiagentic worktree start` names the branch by account prefix**, and
  that prefix fails the forbidden-terms check: rename the branch before the
  first commit (`git branch -m`).

The `pin-bump` drift gate's mechanical `[coupling]` result does
not certify semantic protocol compatibility, supported model formats,
KLD or hardware performance. Record those as separate release evidence.

## The procedure

Work on one deployment checkout at a time. Record the first bump and its
acceptance before changing the next deployment.

1. **Move the pin, then commit — in that order, in the same minute.**

   ```text
   bigcherry --engine llamacpp repin <tag>
   git add config/recipes.toml releases/pin-transition-llamacpp.json
   git commit -m "pin: <old> -> <tag> (<sha-prefix>) — rebase in flight"
   ```

   `repin` rewrites the `pinned = "..."` line **and** writes the
   transition marker `releases/pin-transition-llamacpp.json` atomically. Both go in
   the SAME commit: an uncommitted marker is the `uncommitted-transition`
   state (a named, distinguishable state), and `bigcherry --engine llamacpp pull` refuses to
   move the checkout while the marker is uncommitted. The committed marker
   is the declaration that a bump is in flight; from this moment on, the old
   checkout state is *mid-rebase*, not drift, on every tree that shares the
   bigcherry history.

2. **Move the checkout.**

   ```text
   bigcherry --engine llamacpp pull --source bigcherry
   ```

3. **Audit the new upstream tree, then mechanically rebase-check the selected
   patches against it.** `apply` (with or without `--rebase-report`) still
   requires a passing audit on the UNPATCHED tree before it will run —
   `patch-rebase-check` doesn't touch that gate (it's observational and can
   run any time after `pull`), but it needs to come after `audit` in this
   runbook or you'll hit `apply`'s refusal before you ever get to use its
   report.

   ```text
   bigcherry --engine llamacpp audit
   bigcherry --engine llamacpp patch-rebase-check --source bigcherry --json releases/patch-rebase.json
   ```

   This is a mechanical gate, not prose review: `patch-rebase-check` runs in
   an isolated worktree (it never mutates the checkout you just moved) and
   fix everything it reports before touching `apply`.

   Per-patch status is one of `CLEAN` / `CLEAN_NOOP` / `NOT_APPLICABLE_BY_DESIGN`
   (fine, no action) or `FAILED_NEEDS_RECONCILIATION` / `BLOCKED_BY_DEPENDENCY`
   / `QUARANTINED` (needs a human: a moved/renamed anchor, or a patch that
   only worked because an earlier, now-broken patch's edit was silently in
   the tree — the undeclared-dependency case `QUARANTINED` exists to name).
   The JSON report carries structured `reason_code`s and bounded
   reconciliation context (a diff of the failing file across the pin bump
   when the previous revision is known) instead of just an anchor-mismatch
   count. `docs/archive/build/PIN_REBASE_REVIEW_B10502.md` remains the historical narrative
   template for writing up what you fixed and why, once the tool has told
   you what's actually broken.

   You do not have to fix every patch before making progress: once the
   report is clean enough to accept, apply exactly its known-good,
   dependency-closed subset —

   ```text
   bigcherry apply --rebase-report releases/patch-rebase.json --known-good
   ```

   — which fails closed (a stale-report error naming the mismatch) if
   *anything* about the tree has moved since the report was written: the
   upstream revision, the BigCherry tooling revision, any patch's bytes, the
   overlay's bytes, or the patch-application semantics version. A report is
   single-use evidence for the exact state it was computed against, not a
   standing waiver.

   A *full* known-good apply (every selected patch reproved clean) advances
   the release record `audited -> patched` exactly like a plain `apply`
   would. A *partial* known-good apply is only accepted while the record is
   at stage `audited` or `broken` — applying a partial subset onto an
   already-`patched`-or-later tree would leave later-stage evidence
   (generated/built/tested/validated) describing a composition that was
   never actually fully applied, so it's refused unless you pass `--force`
   and accept invalidating that evidence yourself. A partial apply never
   advances the stage past `audited`: it is reconciliation progress, not a
   finished tree. Fix the reconciliation list and re-run
   `patch-rebase-check` (then `apply --known-good` again) until the full
   selection is clean and the record reaches `patched`.

   `patch-rebase-check` and `pin-status` (step 4) answer different
   questions and are never substitutes for each other: `pin-status` is pure
   revision identity (is the checkout at the pinned SHA?); `patch-rebase-check`
   is whether the patches still apply to whatever revision is checked out.

4. **`bigcherry --engine llamacpp pin-status --all-remotes`** — confirm the local tree now
   reads `consistent` (or `mid-rebase` while the checkout move of step 2 is
   outstanding on some tree — the marker is shared history, so every tree
   reads the same transition state until its checkout catches up and the
   marker is cleared). Remote trees report only what the probe can see:
   consistent / mismatch / unreachable — the controller never infers a
   remote mid-rebase from its own records.

5. **Rebuild the campaign surfaces on the new tree** (build dirs, catalog
   generation, descriptors), then `pin-status` again.

   Then run the standing bump-validation matrix on real hardware:
   `python3 tools/lab/bump-validation/run_bump_validation.py --output
   <work-dir>` (build server only). It builds one fresh binary at the new
   pin and launches it under the real production runtime-profiles across
   every real GPU individually (`production-safe-single`,
   `tierB-qwen9b-q6k`) plus the real dual-XTX multi-GPU topology
   (`production-dual-xtx`, `tierL-qwen27b-q8`) -- a real-server-launches-
   and-completes-correctly gate that `patch-rebase-check`/`patch-lint`
   cannot provide, since none of the static gates ever launch a server. A
   failing cell is a bump blocker; investigate before declaring the bump
   complete.

6. **Walk the invalidation list below.** Every entry is a question:
   *is there an artifact of this kind that the next step will consume, that
   was produced at the old revision?* If yes, it is stale now.

7. **Completion rule — the bump is DONE only when:**

   - `bigcherry --engine llamacpp pin-status --complete --all-remotes` reports PASS: the local
     tree is `consistent`, every **required** configured tree is reachable
     and converged at the same pin, and (for `role = "campaign"` trees) the
     tooling revision matches the expected one, AND
   - the transition marker has been cleared (`pin_transition.clear()` / file
     deletion) in the same commit that records completion, AND
   - the invalidation list in step 6 has been walked and the dispositions
     recorded, AND
   - the verbatim `pin-status` output is appended to the release record
     notes (so the evidence of completion is itself evidence).

   `[[trees]]` in `config/recipes.toml` is where the required trees are
   declared; the aggregate additionally reports pin divergence across trees
   (each tree can be individually consistent and still disagree with its
   neighbours — a prior incident had three upstream revisions live at
   once).

   **Release.** Once the hardware evidence is in (step 4b of the bump skill:
   build, smoke, A/B at the new pin), one command does the gate above and
   everything after it:

   ```sh
   python -m bigcherry --engine llamacpp pin-release <tag> --evidence @<file with the hardware result>
   ```

   Phases, each skipped when already done, so the command is re-run after
   any stop: `gate` (`pin-status --complete --all-remotes`), `record` (gate
   output and evidence into `releases/<tag>.json`, transition marker removed,
   one commit), `notes` (`release-notes` plus package-scoped `release-as` in
   `release-please-config.json`, one commit), `main` (the work branch is pushed, then a checked pull request to
   `main` is opened/reused, brought up to `main`, and merge-committed once its required checks pass), `release` (the exact
   bc-llamacpp release-please PR is file-validated, merged and the command waits
   for the tag; a single confirmation comes first: a terminal is asked once, `--yes` confirms without asking, and with
   neither the command stops before tagging with the finished phases recorded), `unpin` (branch `chore/release-as-clear-<version>` removes
   `release-as` through another checked pull request), `sync` (the work
   branch takes current `main` back). `--through <phase>` stops early;
   `--dry-run` reports the concrete PRs and manifest versions without writes.

   The version is carried only by the package-scoped `release-as` in
   `release-please-config.json`. Commit-message `Release-As` footers are not used.

   The release line follows the pin by itself. The `declare` phase of `pin-bump` compares the new pin's release
   major (the llama.cpp build number; Radiance 2.2.0 is 2020) with the one in `.release-please-manifest.json` and,
   when they differ, sets `release-as` `<major>.0.0` in the same commit that moves the pin
   (`tools/bigcherry/release/release_line.py`). Once the release exists (the manifest has reached `release-as`),
   the release-please workflow opens a pull request that removes the key and queues it to merge; the `unpin`
   phase above does the same synchronously for a `pin-release` run. The test
   `tools/tests/release/test_release_line.py::test_this_repository_declares_its_release_line` (and
   `python -m bigcherry.release.release_line check`) fails when the pin's major differs from the manifest's without
   a `release-as`, when the key does not name the pin's major, and when it is older than the manifest.

   Required PR checks include conventional-title and index-line-endings. Release-please uses RELEASE_PLEASE_TOKEN to trigger workflows; PRs created with GITHUB_TOKEN may not trigger required checks. If no checks run, close and reopen the PR as the account. The pin-release main/release/unpin branches must be updated against main, preserve the released package version, wait for required checks and refuse on failures.

   Versions and tags. release-please owns the tags: `bc-llamacpp-<llama build>.<minor>.<patch>` (the llama.cpp engine's release line; the shared platform has its own, `bc-platform-<version>`).
   A pin bump is released as `<build>.0.0` (`bc-11474.0.0`; the workflow also
   adds the readable tag `bc-llamacpp-b11474-r0` to it; later releases at the same build get `-r1`, `-r2`, ...). While the pin stays, further
   BigCherry releases use the same command with `--bump minor` (features) or
   `--bump patch` (fixes), giving `bc-11474.1.0`, `bc-11474.1.1`, ...;
   `--version` states the version. Each release has its own notes file
   (`docs/releases/notes/bc-llamacpp-<version>.md`) and checking its tag out reproduces
   that patch set. There is no separate `supports/<release>` tag any more (the
   last one is `supports/b11402`). A tag push does not start a release: the
   workflow runs on pushes to `main` only.

8. **Record a ledger event for the bump as a whole** (`ag-ledger`
   `record_change_event`, `change-class: release`), even though `pin-bump`
   auto-commits the pin flip and release record themselves — those commits
   are structured evidence, not a narrative record of what a human/agent
   actually had to *decide* during the bump. The ledger event's
   `technical-summary` must include, for every patch `patch-rebase-check`
   reported as `FAILED_NEEDS_RECONCILIATION` / `BLOCKED_BY_DEPENDENCY` /
   `QUARANTINED`:

   - the exact upstream commit(s)/PR(s) that broke it (cite SHAs, not just
     "upstream changed"),
   - whether the patch's underlying fix is still needed at the new pin, or
     superseded (name the specific upstream change that makes it redundant
     if superseded — never guess),
   - what was actually done about it: re-anchored (link the fix), disposed
     as `known_broken` (name the disposition file), or left broken with a
     stated reason,
   - whether it is selected by a production patch-set (`framework` /
     `validated-enhancements`) — a disposition on a non-selected
     experimental patch is routine; a disposition on a selected one is not,
     and must say why that was still the right call.

   This is the durable record of the *investigation*, not just its
   outcome — a future bump hitting the same upstream commit range should be
   able to find this event and reuse the analysis instead of re-deriving
   it. Link the ledger event to the tracking plan item (`plan-item-ids`).
   Preserve — never overwrite — the disposition files and previous ledger
   events from earlier bumps; each bump's investigation is its own record,
   even when the same patch breaks again at a later pin.

## The revision-bound invalidation list

Everything here is keyed, directly or transitively, to the upstream
revision. A pin move invalidates each of these for the old revision; they
are not "wrong", they are *for another pin* — which is exactly why they must
be named, not silently reused.

| artifact | why it is pin-bound | disposition on bump |
| --- | --- | --- |
| kernel-fraction traces (rocprofv3 CSVs) | kernel names are pin-specific; a name-keyed comparison across upstream revisions is not evidence of a behaviour change | retrace at the new pin or run an equivalence check and record the source/build identities |
| tune measurements + journals | per pin AND per build identity (dispatch digest mixes manifest hash + source revision) | new tune; old files are history, not input |
| replay caches (v4) + dispatch DBs | v4 has `rerun_required` for cross-pin entries — the designed path | let the runtime report `rerun_required`; do not hand-convert |
| inventory DBs (`*.sqlite` / `*.json`) | signatures reference revision-specific canonical shapes | regenerate from a fresh record at the new pin |
| campaign build descriptors (`artifacts/<rev>/`) | content-addressed by `upstream_revision` | regenerate; old descriptors stay as historical evidence |
| build directories (`~/bc-build-*`) | compiled from a specific tree | rebuild (ccache makes this cheap) |
| committed evidence bundles (e.g. `docs/evidence/<run-id>/`) | the README names the `source_revision`; every checksum is for that revision | the bundle is *of* that pin — never mutate it; new pin = new bundle |
| `releases/` records + index | monotonic per-revision evidence; the old record stays at its stage | new record for the new revision; old records are never edited |

Patch validation evidence (`evidence/validation.json`, `evidence/promotion.json`, the prior baseline) is **not** on
this list. It carries across the bump: the pin-bump evidence phase and every build apply the same qualification rule as
before the bump, and report the old recorded pin as a note. A patch is rerun only when someone decides its change
warrants it and demotes the patch; a genuine failure (the patch no longer applies, a build or check fails) still stops
the bump as above.

## What `pin-status` does and does not do

It reads. It never moves a checkout, never edits config, never advances a
release stage. If a state needs a fix, the command tells you the state and
the procedure above tells you the fix.

**Policy tiers** (mutually exclusive, decided in `__main__`, not in the
verdict engine):

- *(default)* — diagnostic. Never fails; prints the named state and the
  evidence for it.
- **`--strict`** — pipeline preflight for the gated tree. Fails on
  drift / uncommitted-transition / unavailable / unresolvable-pin;
  allows mid-rebase with a warning (the pipeline's source identity is
  revision-bound, so a declared bump in flight is tolerable). `bigcherry
  build` runs this preflight before any lane starts (fail closed, no
  bypass); campaign-only trees with no git checkout at the clone source
  have nothing to guard and pass vacuously.
- **`--complete`** — the bump-completion gate (step 7). Fails until every
  required tree is reachable, converged, and consistent.

## Compile-only admission after composition (PA44-A)

`bigcherry --engine <name> pin-bump` now runs the bump-time source check as its own resumable phase:

`preflight -> declare -> pull -> drift -> audit -> patch-lint -> coverage -> apply -> compile -> lane-build -> cpu-smoke -> reaudit -> complete`.

The `drift` phase (MEN19) runs right after the pull, on the pristine bumped upstream tree, before the audit, patch rebase and builds: it asks the engine adapter for `source_drift_checks` (llama.cpp has none) and STOPs with the check's code and the failures per file. For radiance it first checks `component-coupling` (source includes and option/env literals,
declared upstream source inventory, plugin ABI and owned wire constants) and STOPs with
`COMPONENT_COUPLING_DRIFT` for any mismatch. It then runs libr3's
`r3_rewrite.py --check` (writes nothing) against the bumped `libr4d` sources and STOPs with `NATIVE_REWRITE_DRIFT`, listing every `rewrites.txt` rule and `native/*.rw` block that no longer matches, so one run shows the whole port instead of the first block the build trips on. A check that cannot run STOPs with `SOURCE_DRIFT_UNCHECKED`.

The `compile` phase runs the engine's own compile checks (the engine adapter's `compile_checks`) against the patched vendor tree, and `audit`/`reaudit` run its `audit_checks`; see [Per-engine bump gates](#per-engine-bump-gates). For llama.cpp the compile check is `bigcherry.build.source_compile_check` with `clang++`; it needs neither ROCm nor a GPU. A successful phase records the exact effective source-tree OID. On `--resume` it is skipped when that OID is unchanged and re-run when the checked tree changed.

A compile failure STOPs with `SOURCE_COMPILE_FAILED`. `failure.evidence.errors[]` names the composed file and line, the production patch/edit that produced that line, the first compiler-message lines, and, when the failing line is upstream text, `owner: upstream` plus the nearest preceding production edit (or the patches that edit that file). Fix the named owning patch and resume the same bump.

If `clang++` is unavailable, pin-bump STOPs with `SOURCE_COMPILE_UNAVAILABLE` and an install hint. There is no automatic skip. `--skip-compile` is the explicit exception and writes that skip into the release record notes so a released bump cannot hide that the gate was omitted.

Every patch or packaging change that can affect composed source still passes the `Patch compile validation` pull-request check. That CI job remains general admission coverage (including its CPU-only `llama-server` compile), but a pin bump no longer relies on a later PR job to discover source/type breakage introduced by the upstream move: the `compile` phase catches it before `reaudit`.

This check is not hardware qualification and does not replace the HIP/ROCm build and real-hardware smoke/identity gates. Those remain mandatory after the bump.
## Patch promotion releases (PA45)

A patch promotion on the current pin does not repin llama.cpp. Run `bigcherry patch-promote <id>... --evidence @file [--profile-only] [--release]`; its `feat(patch):` squash title drives a minor release and a `fix(patch):` title drives a patch release. Promotion evidence is copied under `releases/evidence/` and normally must name two distinct models; `--profile-only` is the explicit exception for profile-scoped work. The promotion PR must merge through normal checks/review before release-please can create the version PR.

## Lane build and CPU smoke: the first runtime gate (MEN15, MEN16)

After `compile` and before `reaudit`, `pin-bump` builds the bumped engine itself and runs the engine's RAM-only smoke
against that build: a tiny model served on the CPU with no GPU visible, three short greedy completions run twice
(non-empty and identical), a graceful shutdown, and a check that the server log never shows a device being opened. The
smoke needs no GPU, so a busy host cannot hold up a bump, and it comes before any GPU smoke.

**`lane-build`** builds the lane `<--source>:<--build>:<--platform>` through the campaign build machinery (the same
planner and runner as `bigcherry build --lane`), with the engine's serve binary as the launcher. `--build` is a recipe
`[build.*]` profile; there is no default, so a run without `--build` (or without `--platform`) is refused up front with
`LANE_BUILD_INPUT_MISSING`, before any phase. The phase records the lane, `build_plan_id`, the build directory and the
launcher's sha256 in `<report dir>/state.json` (`lane_build`) and as a line in the bump's release record notes. A
failing build STOPs with `LANE_BUILD_FAILED` (a lane the recipes cannot plan: `LANE_BUILD_REQUEST_INVALID`).

**`cpu-smoke`** then runs on that build. A failing smoke STOPs with `CPU_SMOKE_FAILED` (every reason in
`failure.evidence.failures`, the result and server log under `<report dir>/cpu-smoke/`); `--resume` reuses the saved
build and does not build again.

- `--cpu-smoke-build <build dir>` is an explicit override: the lane build is skipped, recorded as
  `lane_build.status = "skipped-by-override"` (and `cpu_smoke.build_override`), and that directory is smoked.
- `--skip-cpu-smoke` is the only way past the smoke itself and needs no build; it is saved as
  `cpu_smoke.status = "skipped"` (`lane_build.status = "skipped"`) and printed in the final summary.
- On `--resume`, `--build` and `--cpu-smoke-build` follow the `--platform` rule: omitted, the saved value is reused; a
  different `--build` is refused (`RESUME_BUILD_MISMATCH`), as is a different `--cpu-smoke-build`, or any
  `--cpu-smoke-build` once the lane build passed (`RESUME_CPU_SMOKE_BUILD_MISMATCH`).

The same smoke runs on its own with `bigcherry --engine <engine> smoke-cpu --build <dir>` (see the engine's README or
`docs/reference/build/BUILD.md`).

llama.cpp:

    python -m bigcherry --engine llamacpp pin-bump <ref> --source bigcherry --build control --platform linux-multi

## Hardware acceptance: stored baselines (MEN17)

After the CPU smoke an engine that declares `[[acceptance]]` entries in `engine.toml` can be held to its stored numbers:

    python -m bigcherry --engine <engine> accept --output <new dir> [--only <entry>]

Each entry names a server-comparison config and baseline files (`docs/reference/engines/<engine>/baselines/*.json`: arm,
metric, value, one-sided tolerance taken from the measured noise, direction, and the run id, pin and date it was measured
on). The command runs the configs, prints metric / baseline / measured / delta / pass-fail, writes `acceptance.json` and
exits 1 on any regression. `pin-bump <ref> --accept` does the same inside the bump (after `cpu-smoke`, same phase); it needs
the GPUs and the configs' environment, so it is never the default. With no GPU it STOPs with `ACCEPT_NO_GPU`; a regression
STOPs with `ACCEPT_FAILED` (evidence names the failing rows and `acceptance.json`). A pass is saved in `state.json`
(`accept.status = "passed"`) and noted in the release record. A baseline moves only by a deliberate new measurement.

## Per-engine bump gates

The compile and audit gates are invariants of one engine's source, so each engine's adapter
(`tools/bigcherry/engine_adapters/`) supplies them; pin-bump and `bigcherry audit` only run what the adapter returns and
stop on the first failing check under that check's own failure code.

| | compile checks (`compile` phase) | audit checks (`audit`, `reaudit`, `bigcherry audit`) |
|---|---|---|
| llama.cpp | `ggml-cuda-contracts` (`SOURCE_COMPILE_FAILED`, needs `clang++`) | the ggml HIP/CUDA invariants and overlay sync |
| radiance | the component gate below | overlay sync (nothing to sync when the engine has no overlay directory), `patches.application_state` (no registered patch half-applied, no patch targeting a file the tree lacks), `components.source_present` and `components.abi_header` (each declared component has its source; `abi/rad_abi.h` declares a numeric `RAD_ABI_VERSION`) |

**Owned-unit origin check (RCS23).** Each unit the `bc-iqr` overlay owns is listed in the component's `origins.toml` with its upstream path, the commit it was taken from and `last-integrated`. `pin-triage` reports each as `clean` (the origin path is unchanged between `last-integrated` and the target), `candidate` (it differs: the commits are listed) or `blocked` (a commit missing from the mirror, the path absent at `last-integrated`, a rename or removal, a target that is not a descendant). The bump's drift phase runs the same check on the target tree: `blocked` stops with `OWNED_ORIGIN_BLOCKED`, `candidate` with `OWNED_ORIGIN_CHANGED` (review the commits: a three-way merge for a whole unit, a structural review for the extracted header, re-prove with the unit harness, then set `last-integrated` by hand; the bump never writes it), `clean` passes. An extracted-family header is reported the same way, so an upstream change to the shared 8a8 prefill header stops the bump until it is reviewed.

**radiance component gate.** radiance's BigCherry plugin components (`[components]` in `engine.toml`, e.g. libr3, which
also produces libr3moe and the helpers) are built against the engine's plugin ABI and internal headers, so a bump can break
them without any patch anchor failing. The compile phase runs three checks in order, each STOPping the bump:

1. `components-build` (`COMPONENT_BUILD_FAILED`; needs `cmake`, `ninja`, the ROCm toolchain): configure and build the bumped
   engine in `work/source-compile-check/radiance-build`, then build every component through
   `campaign.runtime_bundle.build_runtime`, the same function `bigcherry build` uses for the radiance lane. The configure
   is the one `bigcherry build` makes for the build platform named with `--platform` (its `${VAR}` compilers resolved from
   the host environment, its options, its GPU targets plus the components'), and the components are built with the same
   compilers; a default system compiler can fail where the platform's GCC 14 builds. `--platform` is required (the recipes
   do not link a source to a platform); a missing, unknown or unresolvable platform STOPs at the start of the compile phase
   with `SOURCE_COMPILE_FAILED`. It is saved in `state.json` (`compile.platform`); a `--resume` with none reuses it and a
   different one is refused (`RESUME_PLATFORM_MISMATCH`).
2. `component-plugin-abi` (`PLUGIN_ABI_MISMATCH`): every `*.so` in the build's `radiance_home/kernels` (the engine's own and
   the components') is loaded in a child process. Its exported `rad_plugin_abi_version()` must equal `RAD_ABI_VERSION`
   from the bumped source's `abi/rad_abi.h`, and its exported `rad_plugin_abi_layout()` must match the layout fingerprint
   from the engine's own `libr4d`. The loader enforces both; `nm` could show symbols but not their values or agreement.
3. `component-tests` (`COMPONENT_TEST_FAILED`; needs `ctest`): each component's `[test] targets` in `component.toml` are
   built and run with `ctest` in the component's build tree.

The radiance bump command (`--build` names the lane's build profile; pin-bump builds the bumped engine and smokes that build):

    python -m bigcherry --engine radiance pin-bump <ref> --source radiance --build stock --platform linux-radiance

These need a radiance checkout and a Linux host with the toolchain; on a host without `cmake`/`ninja`/`ctest` the phase STOPs
with `SOURCE_COMPILE_UNAVAILABLE` (or `--skip-compile`, recorded in the release record).
