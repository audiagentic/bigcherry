# Branching, review and release standard

How work gets from an idea to a released BigCherry version. It is trunk-based development with short-lived branches,
pull requests gated by automated checks, conventional commits, and release-please for versions: the practice most
continuously delivered projects use, adapted to two things that are particular here: several agents work in the
repository at once, and a change is only proven on the lab hardware.

## The rules

1. **`main` is always releasable.** Everything on it has passed the offline checks, and anything that changes what
   the build does has hardware evidence. Nobody commits to `main` directly; it only moves by merging a pull request.
2. **One slice of work, one branch, one pull request.** A slice is the smallest change that is complete on its own:
   one patch, one re-base, one tooling feature, one plan update. A branch lives hours to a few days. If it needs more,
   it is more than one slice.
3. **Branches start from current `main`** and are named `<type>/<plan item>-<topic>`, lower case, hyphens:
   `feat/qfp31-mtp-deferred-catchup`, `fix/1347-warp-size`, `chore/bpb01-rebase-1337`, `docs/branching-standard`,
   `bump/b11474`. The type is the conventional-commit type of the change.
4. **The pull request title is a conventional commit**, because it becomes the commit on `main` (squash merge) and
   release-please reads it:

   | Title starts with | Use for | Effect on the next version |
   |---|---|---|
   | `feat(patch): ` | a patch promoted to the production set, or a new capability | minor + 1 |
   | `fix(patch): ` / `fix: ` | a fix to a promoted patch or to tooling that ships | patch + 1 |
   | `perf: ` | a speed-up inside an already promoted patch | patch + 1 |
   | `chore: `, `docs: `, `test: `, `refactor: `, `ci: `, `build: ` | plans, experiments, diagnostics not in the build, tooling internals | none |

   A pin bump is the exception to automatic numbering: its notes commit carries `Release-As: <llama build>.0.0`
   (made by `bigcherry pin-release`).
5. **A pull request merges when** its checks are green, its description is filled in, and, if it changes the
   production build, the hardware evidence is in the description. Merge method: squash. The branch is deleted on
   merge.
6. **Releases are cut from `main` by release-please.** Every merged `feat` / `fix` / `perf` updates one open release
   pull request; merging that pull request tags `bc-<llama build>.<minor>.<patch>` and publishes the notes. Several
   promotions merged before the release pull request is merged ship as one release.
7. **No long-lived integration branch.** `patch-refactor` served that role until `bc-11474.0.0`; it is retired once
   the slices in flight on it have merged.

## What a slice goes through

| Step | Who | What |
|---|---|---|
| 1. Branch | author (an agent or a person) | `git switch -c <type>/<item>-<topic> origin/main` |
| 2. Build the change | author | Commits on the branch can be small and informal; only the pull request title has to be conventional. Push early: the offline checks run on every push. |
| 3. Offline checks | CI | Patch mechanics tests, catalog and governance tests, patch-lint, composition of the production set and of every experiment that names a changed patch, compile-only pass when PA44-A lands. |
| 4. Hardware validation | the validating agent, on Brutus | Check out the slice branch in the lab tree, build it, run the smoke test and the A/B the change calls for. The lab tree returns to `main` afterwards. |
| 5. Evidence | validator | Paste the result into the pull request: build id, activation marker, the A/B table, identity statement. For a promotion the same text goes into the patch's README. |
| 6. Review | a second agent or the owner | Reads the diff and the evidence. The author does not approve their own pull request. |
| 7. Merge | validator or owner | Squash merge with the conventional title. |
| 8. Release | `bigcherry pin-release` / release-please | The release pull request is merged when the owner wants the batch out. |

Changes that cannot alter the build (plans, docs, lab scripts, tests) skip steps 4 and 5.

## Working in parallel

- An agent works on its own branch and never pushes to another agent's branch. Hand-over is by pull request or by
  asking for the branch to be continued.
- Two slices that edit the same patch are sequenced, not run in parallel; the second one branches after the first has
  merged.
- `git stash` is not used (shared trees). A local tree is on exactly one branch; a second line of work uses a second
  worktree. Use `bigcherry slice start <branch>`; if a tool has already changed tracked files in the primary
  checkout, use `bigcherry slice start --carry <branch>` to move that diff into the new worktree and leave primary clean.
- The lab tree on Brutus is a shared resource: one slice is checked out at a time, the lab queue's lock orders the
  runs, and nobody pulls while a script is executing there.
- Rebase a slice branch onto `main` before it merges if `main` has moved; do not merge `main` into the slice. History
  on `main` stays linear.

## Branch clean-up

A merged branch is deleted, always, so that the list of branches is the list of work in flight.

- The repository setting "Automatically delete head branches" is on: merging a pull request deletes its branch.
- The agent that merges runs `bigcherry slice finish <branch>`. Finish is resumable: it tolerates an already-removed
  worktree record or remote branch, removes only empty/verified-clean leftovers under `worktrees/`, deletes an
  eligible local/remote branch, prunes metadata, and is a no-op after successful completion. It never uses `--force`.
- The lab tree is returned to `main` and its copy of the slice branch is deleted in the same step.
- A branch that will not be merged is closed on purpose: close its pull request with the reason, then delete the
  branch. Work worth keeping is a plan item, not a parked branch.
- `bigcherry slice prune` (PA46) lists remote branches that are fully merged into `main` and deletes them, and lists
  unmerged branches older than 14 days with their last author for a decision. It never deletes an unmerged branch.
- Exceptions that are not slice branches and are left alone: `main` and release-please's own release branch
  (`release-please--branches--main--components--bc`), which it reuses.

## Protection on `main`

Enforced in the repository settings, so the rules do not depend on memory:

- pull request required, at least one approving review, stale approvals dismissed on new commits;
- required status checks: the offline patch validation and the other path-specific validations that apply;
- linear history (squash merge only), no force pushes, no deletion;
- the release-please bot and the release command are the only automation allowed to merge the release pull request.

## Hotfix

A defect in a released version is fixed the same way, on `fix/<item>-<topic>` from `main`, with the smallest change
that corrects it and the evidence that it does. Merging it raises the patch number at the next release. A released
tag is never moved.

## Why these choices

- *Trunk-based with short-lived branches* keeps integration continuous: the pin bump of 2026-10-08 found three plan
  files edited on both `main` and a week-old integration branch, and one plan id used for two different items.
- *Squash merge with conventional titles* gives one commit per slice on `main`, a changelog that reads as a list of
  slices, and version numbers that follow from what was merged instead of being chosen by hand.
- *Evidence in the pull request* makes "validated" a property of the change that anyone can read, in the place the
  merge decision is made.
- *Protection in settings* is what makes the standard hold when several agents act at once.
