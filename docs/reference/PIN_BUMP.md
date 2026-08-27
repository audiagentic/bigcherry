# Pin-bump procedure (llama.cpp pin movement)

The procedure for moving the llama.cpp pin, and the completion rule that says
when it is done. Companion to `bigcherry pin-status` (RE48), which is the guard
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
  (`releases/pin-transition.json`) declares the move from the current vendor
  revision to the pinned revision, and the declared base's release record is
  not broken; the move is in flight
- **drift** — vendor checkout is at a revision the pin never declared (a
  release record for that revision is *evidence*, not a transition: without
  a committed marker the state is drift, with sub-reasons stale-marker /
  marker-mismatch / broken-base / no-transition)

The transition that turns "drift" into "mid-rebase" is **committing the pin
move and the transition marker together, before moving the checkout**.
`bigcherry repin` writes the marker atomically with the pin line; the commit
is the declaration that a bump is in flight, and every tree that shares the
bigcherry history reads the same state from it. A manual
`git -C vendor/llama.cpp checkout <sha>` that skips the pin move is exactly
the state the guard exists to catch — it is how the 2026-08-21 S1 stale-trace
incident happened (evidence captured on b10257 while the pin said b10502):
a release record for the stale revision existed, and "record exists" was
once the only rule — which is why the marker supersedes it.

## The procedure

Work on ONE tree at a time. Do not bump H: and J: (Brutus) in the same
window; bump one, record, then bump the other.

1. **Move the pin, then commit — in that order, in the same minute.**

   ```text
   bigcherry repin <tag>
   git add config/recipes.toml releases/pin-transition.json
   git commit -m "pin: <old> -> <tag> (<sha-prefix>) — rebase in flight"
   ```

   `repin` rewrites the `pinned = "..."` line **and** writes the
   transition marker `releases/pin-transition.json` atomically. Both go in
   the SAME commit: an uncommitted marker is the `uncommitted-transition`
   state (a named, distinguishable state), and `bigcherry pull` refuses to
   move the checkout while the marker is uncommitted. The committed marker
   is the declaration that a bump is in flight; from this moment on, the old
   checkout state is *mid-rebase*, not drift, on every tree that shares the
   bigcherry history.

2. **Move the checkout.**

   ```text
   bigcherry pull --recipe bigcherry
   ```

3. **Re-verify the patches against the new revision.**
   This is a mechanical gate, not prose review: run `patch-rebase-check` in
   an isolated worktree (it never mutates the checkout you just moved) and
   fix everything it reports before touching `apply`/`audit`.

   ```text
   bigcherry patch-rebase-check --recipe bigcherry --json releases/patch-rebase.json
   ```

   Per-patch status is one of `CLEAN` / `CLEAN_NOOP` / `NOT_APPLICABLE_BY_DESIGN`
   (fine, no action) or `FAILED_NEEDS_RECONCILIATION` / `BLOCKED_BY_DEPENDENCY`
   / `QUARANTINED` (needs a human: a moved/renamed anchor, or a patch that
   only worked because an earlier, now-broken patch's edit was silently in
   the tree — the undeclared-dependency case `QUARANTINED` exists to name).
   The JSON report carries structured `reason_code`s and bounded
   reconciliation context (a diff of the failing file across the pin bump
   when the previous revision is known) instead of just an anchor-mismatch
   count. `docs/reference/PIN_REBASE_REVIEW_B10502.md` remains the narrative
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
   standing waiver. Applying a partial known-good subset does not advance
   the release record to `patched` — it is reconciliation progress, not a
   finished tree; fix the reconciliation list and re-run
   `patch-rebase-check` until the full selection is clean, then:

   ```text
   bigcherry audit
   ```

   which advances the release record `pulled -> audited` (or `-> broken`).
   `patch-rebase-check` and `pin-status` (step 4) answer different
   questions and are never substitutes for each other: `pin-status` is pure
   revision identity (is the checkout at the pinned SHA?); `patch-rebase-check`
   is whether the patches still apply to whatever revision is checked out.

4. **`bigcherry pin-status --all-remotes`** — confirm the local tree now
   reads `consistent` (or `mid-rebase` while the checkout move of step 2 is
   outstanding on some tree — the marker is shared history, so every tree
   reads the same transition state until its checkout catches up and the
   marker is cleared). Remote trees report only what the probe can see:
   consistent / mismatch / unreachable — the controller never infers a
   remote mid-rebase from its own records.

5. **Rebuild the campaign surfaces on the new tree** (build dirs, catalog
   generation, descriptors), then `pin-status` again.

6. **Walk the invalidation list below.** Every entry is a question:
   *is there an artifact of this kind that the next step will consume, that
   was produced at the old revision?* If yes, it is stale now.

7. **Completion rule — the bump is DONE only when:**

   - `bigcherry pin-status --complete --all-remotes` reports PASS: the local
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
   neighbours — the 2026-08-21 incident had three upstream revisions live at
   once).

   Tag `supports/<release>` only after step 7. Before that the release
   record stage tells the truth (`tuned`, `production`), and the tag must
   not run ahead of it.

## The revision-bound invalidation list

Everything here is keyed, directly or transitively, to the upstream
revision. A pin move invalidates each of these for the old revision; they
are not "wrong", they are *for another pin* — which is exactly why they must
be named, not silently reused.

| artifact | why it is pin-bound | disposition on bump |
| --- | --- | --- |
| kernel-fraction traces (rocprofv3 CSVs) | kernel *names* are pin-specific: the 22dc605→0adcc3bb move renamed the small_k mmvq instantiation (added `halve_iters` template parameter) with zero behaviour change — a name-keyed comparison across pins is meaningless (S1b, h36-campaign bundle README) | retrace at the new pin, or run the S1b-style equivalence check and record it |
| tune measurements + journals | per pin AND per build identity (dispatch digest mixes manifest hash + source revision) | new tune; old files are history, not input |
| replay caches (v4) + dispatch DBs | v4 has `rerun_required` for cross-pin entries — the designed path | let the runtime report `rerun_required`; do not hand-convert |
| inventory DBs (`*.sqlite` / `*.json`) | signatures reference revision-specific canonical shapes | regenerate from a fresh record at the new pin |
| campaign build descriptors (`artifacts/<rev>/`) | content-addressed by `upstream_revision` | regenerate; old descriptors stay as historical evidence |
| build directories (`~/bc-build-*`) | compiled from a specific tree | rebuild (ccache makes this cheap) |
| committed evidence bundles (e.g. `docs/reference/h36-campaign-27b-r9700/`) | the README names the `source_revision`; every checksum is for that revision | the bundle is *of* that pin — never mutate it; new pin = new bundle |
| `releases/` records + index | monotonic per-revision evidence; the old record stays at its stage | new record for the new revision; old records are never edited |

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
