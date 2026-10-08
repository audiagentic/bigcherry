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
7. **No long-lived integration branch.** `patch-refactor` served that role until `bc-11474.0.0`; it was merged into
   `main` and deleted on 2026-10-08. `main` is the only trunk.

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

- Keep the **primary checkout on `main`**. It is updated only by a fast-forward,
  never by `git switch`, `git checkout`, or a forced branch switch. No commits
  are made directly to `main` or to a branch shared with another agent.
- **One agent per worktree**. Run `bigcherry slice start <branch>` from the primary
  or any linked checkout; it fetches `origin` and creates
  `<primary>/worktrees/<branch>` from `origin/main`. Names follow the rules
  above; existing local/remote branches and paths are rejected. All generated
  `worktrees/` paths are gitignored.
- Put edits and commits only in the agent's slice worktree; push its own branch
  and open one PR. Two slices touching the same patch must be sequenced; the
  second begins only after the first merges. Never push to someone else's branch.
- No `git stash`, `git reset`, force switches, or destructive rebase on a shared
  checkout. Preserve unfinished work in its owning slice worktree.
- All slices use the primary checkout's single vendor llama.cpp and shared
  work/build caches. `BC_PRIMARY_ROOT` explicitly overrides the primary root;
  otherwise Git's common directory resolves it. Failure to resolve is an error,
  not permission to create another vendor tree inside the slice.
- A dedicated worktree does not imply exclusive access to the shared vendor
  checkout or build cache; the existing tooling's locks still govern operations
  that mutate them. The lab queue retains control of GPU allocation.

## Lab tree

- Give the lab its own host entry in the untracked `config/environment.local.toml` and name it with `--host`:
  `hostname` (the ssh alias), `repo` (the lab primary checkout) and `cache-root` (the lab work root). On this
  project that is `[host.brutus-lab]` with `repo = /mnt/vault/development/projects/bigcherry/workspaces/main`
  and `cache-root = /mnt/data/bigcherry-work`; `[host.build-server]` describes a different checkout on the
  same machine and must not be used for lab runs. The lab
  creates detached worktrees under `cache-root/worktrees/` and stores queued
  logs under `cache-root/runs/`. Use `--dry-run` to inspect SSH commands.
- The lab primary checkout **must stay on `main` and clean** before and after
  every run. `bigcherry slice lab <branch> --host brutus-lab <script> [args]`
  fetches the branch without checking it out in the primary, runs from a
  detached worktree via the plan-qualification queue, then removes it on exit.
- Queue wrapper scripts that submit their own jobs must not hold an outer GPU
  lock while awaiting an inner queue: the inner queue retains the original GPU
  locks. Ordinary script rows continue to use the existing queue lock.
- No lab session switches branches in the primary tree, force-removes a dirty
  worktree, or pulls while another operation modifies shared source state.
  If cleanup finds untracked edits, it fails visibly rather than discarding them.

## Branch clean-up

- `bigcherry slice status`: one table showing worktree path, branch, PR
  number/state, ahead/behind `origin/main` and dirty status.
- `bigcherry slice start <branch> --carry`: moves the primary checkout's uncommitted tracked changes into the new
  worktree (for tools that write plan or ledger files in the primary). It refuses untracked files and restores
  the primary if the changes do not apply.
- `slice finish` is resumable: if an earlier run stopped part-way (on Windows a shell sitting inside the worktree
  blocks its removal), running it again completes the remaining steps and is a no-op once everything is gone.
- `bigcherry slice finish <branch>`: requires a merged or closed PR via `gh`,
  a clean worktree and a clean primary checkout on `main`. It fast-forwards
  the primary `main` from `origin/main`, removes the clean worktree without
  `--force`, and removes its local and remote branch. Open PRs are refused.
- `bigcherry slice prune`: reports merged remote branches, stale unmerged
  branches and orphaned/merged worktrees. Default is read-only. `--apply`
  deletes eligible merged branches and removes clean orphaned worktrees, never
  using `--force`; dirty worktrees are reported and left intact.
- Merge order stays one PR per slice. `patch-refactor` was retired and deleted on 2026-10-08; the three stale lab-local branches were preserved as `archive/brutus-*` tags and deleted the same day. The owner enables main protection and auto-delete-head-branches in GitHub.

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
