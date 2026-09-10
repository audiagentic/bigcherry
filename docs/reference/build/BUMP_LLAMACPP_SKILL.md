# Bump llama.cpp: skill reference

Committed copy of the Claude Code skill at `.claude/skills/bump-llamacpp/SKILL.md`
(that directory is gitignored in this repo, so this is the shared, tracked copy).
See that file locally for the live, invocable version.

# Bump the pinned llama.cpp revision

This is the real, hard-won procedure -- not a survey of options. It was built
from a live b10502->b10680 bump on 2026-08-29/30 that found and fixed real
bugs in every layer of the tooling along the way. Follow it in order. Do not
skip the sources-check step: it is the one step that answers "is anything we
maintain now redundant, merged upstream, or silently broken" -- and it was
never run for over a week of real bumps until this skill's authoring session
went looking for it.

## 0. Orient before touching anything

```
python -m bigcherry pin-status --all-remotes
```

Read the `VERDICT` for local and every remote tree. If it already says
`consistent` at the target you're about to bump to, you're done (or already
mid-flight -- read the `marker` line). If `mid-rebase`, a bump is already in
flight from a previous session; do not start a second one on the same tree.

**Multi-agent hazard**: this is a shared, multi-session repo. Before touching
ANY tree (especially a remote/SSH campaign tree), check for a live
tree-activity lease and check for other obviously-active work, even if git
status looks clean:

```
ssh <tree-host> "ps aux | grep -E 'llama-server|llama-bench|ninja|cmake|tail.*experiment' | grep -v grep"
```

A clean git status is NOT sufficient on its own -- the 2026-08-29 bump found
The build server's configured campaign tree looked clean-enough in git but had two
active experiment log tails from a concurrent session. `tools/bigcherry/core/tree_activity.py`
(HI151) exists specifically to make this checkable in code
(`tree_activity.list_live_leases(work_root, project_root)`), but not every
long-running runner writes a lease yet -- treat an empty lease list as
necessary, not sufficient, until adoption is complete (see HI151's plan item
for which runners are wired).

If a configured tree looks even slightly ambiguous, prefer a **separate,
known-clean checkout** over risking a shared one -- e.g. this project keeps
`~/bigcherry` (a plain, always-idle clone) alongside the real configured
campaign path `$BC_SHARE/bigcherry` on the build server for
exactly this reason.

## 1. Check what's redundant or merged upstream FIRST

This is the step the user is most likely to ask about directly ("did we check
if our patches are now merged/redundant") and the one most likely to be
skipped, because it's slow and its output needs a human, not a script, to
interpret.

```
python -m bigcherry sources status          # offline: what we track, ported/planned/superseded
python -m bigcherry sources check            # ONLINE: has a tracked fork moved/rebased/merged/drifted
```

- `sources status` reads `config/external-sources.toml` -- every patch that
  backports code from an external fork or PR is declared there, with its
  provenance chain and current status (`ported-benched` / `ported-untested`
  / `planned` / `superseded` / `deferred-hardware` / `excluded`).
- `sources check` actually clones each tracked fork's branch and diffs
  against what's recorded. It reports:
  - `rebased`: the fork's branch moved out from under the active snapshot --
    needs a new snapshot + re-audit before further planning against it.
  - `drift: <old> re-committed as <new> (content identical)`: cosmetic, no
    action beyond eventually refreshing the recorded SHA.
  - `FINDING drifted ... CHANGED content`: the fork's current version of a
    tracked commit differs from what we based work on. **If the patch_id in
    that finding is already `ported-*` (i.e. it's a live patch in
    `patches/`), this is the highest-priority thing to review** -- the fork
    may have fixed or changed something we should pull forward.
  - `FINDING drifted ... not found by title`: the commit vanished from the
    fork's history under its recorded title. This does NOT mean "merged
    upstream" -- it means "go look, by hand, at what actually happened"
    (squashed, renamed, reverted, or genuinely dropped). Never assume
    redundancy from this line alone.
- **Known sharp edge**: `locator` in the TOML must be a bare git ref (`git
  clone --branch <locator>` is called verbatim) -- if a source's `locator`
  field has prose baked into it, `sources check` will report `CLONE-FAILED`
  for that source and silently check nothing. This was found and fixed for 4
  of 5 registered sources in this session (commit `99ac0ab`) -- if it
  recurs, fix the TOML's `locator` field (move the prose into that source's
  existing `notes` field, which already carries commentary), don't touch the
  code.
- Some forks are genuinely large (one tracked personal research fork has
  17500+ commits) -- a `blob:none --no-checkout` clone of it can take several
  minutes. Don't wrap this in a short timeout; if it's cut off, that's not a
  real failure, just re-run with more time.
- Triage every real finding into its own plan item (do not fix inline during
  the bump unless trivial) -- see `HI154` for the template this session used
  after the first real run surfaced 19 findings, including one on an
  already-applied production patch (RD05/RD06/RD07 -> `patches/1203_...`).

## 2. Resolve the target and run the orchestrator

```
python -m bigcherry pin-bump <target-tag> [--source bigcherry] [--resume]
```

This is `tools/bigcherry/release/pin_bump.py` (HI153), a single-tree state
machine over the individually-proven verbs below. It:

1. acquires the tree-activity maintenance lock (refuses if a live lease
   exists -- HI151);
2. resolves the ref, requires a clean controller checkout, refuses an
   incompatible uncommitted transition marker;
3. repins + commits `config/recipes.toml` + `releases/pin-transition.json`
   together, in one commit;
4. pulls the vendor checkout to the new tag;
5. audits it -- with ONE narrow, provably-safe auto-repair: if
   `overlay.vendor_sync` is the *only* failed check and every drifted file
   differs from the current overlay *only* by newline normalization, it
   self-heals via the existing overlay-copy path and re-audits. Anything
   else (including a `FileNotFoundError`-shaped stale build artifact, or
   more than one failed check) STOPS;
6. runs `patch-lint`;
7. runs `patch-rebase-check --all` (every non-rejected patch in the
   registry, not just the recipe's subset) and enforces HI152's coverage
   gate: a recipe-selected patch must be clean, no escape hatch ever; a
   non-selected patch may be clean OR carry a matching `known_broken`
   disposition (see below); anything undiscovered or untriaged STOPs;
8. applies the recipe's known-good subset;
9. re-audits with `--strict`.

On any STOP it writes a structured JSON failure envelope (see the exact
schema in `tools/bigcherry/release/pin_bump.py`'s `failure_envelope()`) under
`artifacts/pin-bump/resume-<target>/failure.json` and exits non-zero. Read
`failure.code` and `failure.recommended_actions` -- don't re-derive what
happened from scratch. After fixing the underlying cause, re-run with
`--resume` to continue from the failed phase rather than restarting.

**If a non-recipe patch is genuinely broken against the new revision** (the
class of gap `HI149` found, though that specific instance turned out NOT to
be broken -- verify before assuming):

```
python -m bigcherry patch-disposition set --patch-id <id> \
  --revision <full-target-sha> --digest <implementation_digest-from-the-report> \
  --failure-status FAILED_NEEDS_RECONCILIATION \
  --reason "..." --owner <team-or-plan> --tracking-item <plan-item-id>
```

This is bound to the exact (revision, digest) pair -- it stops applying the
instant either changes. It is never a standing waiver, and it can never
excuse a patch that IS in the build recipe.

**This orchestrator has not yet been fire-tested against a real live bump**
(as of this skill's authoring) -- only its composable pieces are unit-tested.
Treat its first few real invocations with extra scrutiny; fall back to the
manual procedure below if it does something you don't trust.

## 2b. Manual fallback (if you don't trust the orchestrator yet, or need to
work around something it doesn't handle -- e.g. multi-tree sequencing)

See `docs/reference/build/PIN_BUMP.md` for the full authoritative manual
procedure this skill and the orchestrator both implement. Short version:

```
python -m bigcherry repin <tag>
git add config/recipes.toml releases/pin-transition.json
git commit -m "pin: <old> -> <tag> (<sha-prefix>) -- rebase in flight"
python -m bigcherry pull --source bigcherry
python -m bigcherry audit
python -m bigcherry patch-rebase-check --source bigcherry --json releases/patch-rebase.json
python -m bigcherry apply --rebase-report releases/patch-rebase.json --known-good [--force]
python -m bigcherry pin-status --all-remotes
```

`--force` on `apply` is expected and correct, not a red flag, when `audit`
fails purely because the overlay changed since the tree was last patched
(normal after any real edit to `src/` between bumps) -- verify the failure
really is overlay staleness (`overlay.vendor_sync` only) before forcing past
it; never force past anything else.

## 3. Repeat per required tree, one at a time

`config/recipes.toml`'s `[[trees]]` names every tree that must converge.
**Never bump two required trees in the same window.** For each remaining
tree: push the declaring commit, SSH in, confirm liveness (step 0), pull it
to the same commit, then repeat steps 2 (or 2b) locally on that tree.

```
git push origin <branch>
ssh <tree-host> "cd <tree-path> && git pull --ff-only origin <branch>"
```

If a configured tree has substantial uncommitted local state or ambiguous
liveness, don't force it -- use a separate known-clean checkout to validate
the *process*, and leave the real completion gate (`--complete`) legitimately
unmet until the real tree is confirmed safe. Diverged-but-honest beats
forced-but-risky.

## 4. Rebuild campaign surfaces + walk the invalidation list

Once every tree that can be safely bumped is `patched`/audit-PASS, rebuild
build directories, catalogs, and descriptors, then walk the revision-bound
invalidation list in `PIN_BUMP.md` (kernel-fraction traces, tune
measurements, replay caches, inventory DBs, campaign build descriptors,
evidence bundles, release records -- every one keyed to the old revision is
stale now, not wrong, just *for another pin*).

## 4b. MANDATORY: build and run a real smoke test on real hardware

**A bump is not done until this has actually run and passed.** Patch-rebase
and audit passing only proves the SOURCE reconciles; it proves nothing about
whether the result actually builds or runs. This step is what caught 5 real
bugs live during the first bump this skill was authored from -- do not skip
it because the rest of the bump looked clean.

```
python -m bigcherry build --lane <source>:<build>:<platform> \
  --model <path-to-a-real-gguf> --hip-visible-devices 0
```

- Pick the lane matching the tree you just bumped (e.g.
  `bigcherry-native:control:windows-gfx1100` for a local Windows GPU,
  `bigcherry:control:linux-multi` for the build server).
- `--model` triggers the built-in runtime-smoke validation automatically --
  don't treat a plain compile-only `build` (no `--model`) as sufficient.
- Confirm the GPU is actually idle first (step 0's liveness check applies
  here too -- a build+smoke run is real hardware use).
- If it fails, get the REAL underlying error before assuming it's a stale
  environment issue -- run the built binary directly (bypassing the
  campaign harness) with matching flags to isolate whether the failure is
  in the binary itself or in the harness's own subprocess/environment
  handling. Real bugs found this way so far: a stale/shallow build-mirror
  clone never tracking new tags; a platform config missing compiler paths
  entirely; a missing PATH entry for the toolchain's runtime libraries; the
  smoke worker's environment override replacing (not merging with) the
  real process environment, crashing on Windows; and result-parsing that
  assumed stdout was pure JSON from byte 0 when the toolchain itself prints
  a diagnostic line first on some platforms.
- A real pass looks like a JSON result with plausible non-zero throughput
  numbers for every expected row, not just a nonzero exit code.

## 5. Completion gate

```
python -m bigcherry pin-status --complete --all-remotes
```

Must report PASS: every required tree reachable, converged, consistent, and
(for campaign-role trees) at the expected tooling revision. Append the
verbatim output to the release record's notes -- the evidence of completion
is itself evidence. Clear the transition marker in the same commit that
records completion. Only then tag `supports/<release>`.

## 6. Ledger + housekeeping

- Record a change event via the `ag-ledger` MCP tool (`change-class`,
  `files`, `technical-summary`, `user-summary-candidate`, `status:
  "unreleased"`) -- required by this project's CLAUDE.md for substantive
  work, and a bump always is.
- Never `git stash` (shared multi-agent working tree). Never force-push.
  Never edit a concurrent session's untracked/uncommitted files.
- `git status` before every commit and stage explicit paths -- another
  session can be committing in this same working tree concurrently
  (observed live twice during the authoring bump: once a concurrent
  `git commit` swept up this session's staged-but-uncommitted plan docs
  into its own commit message). Nothing is lost when that happens, but
  don't assume you have exclusive use of the index.

## Known real gaps in this process (as of authoring)

- `pin-bump` (step 2) has not been fire-tested end-to-end on a real bump --
  only unit-tested piece by piece.
- Multi-tree sequencing (step 3) is still fully manual; a Phase 2 controller-
  driven version is deliberately deferred until the single-tree orchestrator
  has survived several real bumps (see `HI150`'s design notes for why).
- Not every long-running runner writes a tree-activity lease yet (`HI151`) --
  an empty lease list is necessary but not sufficient evidence a tree is
  idle.
- `sources check` (step 1) has real, un-triaged findings sitting in `HI154`
  as of this writing -- check whether that item is still open before
  assuming the last bump's provenance is clean.
