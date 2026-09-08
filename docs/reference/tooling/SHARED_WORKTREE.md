# Shared-worktree source-control protocol

BigCherry is routinely edited by concurrent agents in one checkout. This document operationalizes the existing AGENTS.md and TOOLING.md doctrine. It is a source-control safety protocol, not a scheduler or lock service.

## Required protocol

Before editing or committing:

```bash
git status --short
git diff --name-only
git rev-parse HEAD
```

Treat every pre-existing modified/untracked path as owned by another session unless you know otherwise. If a file you need already has unrelated uncommitted edits, do not overwrite, restore, or fold them into your change; use another owned file/slice or stop that slice.

During work:

- Never use `git stash`, `git reset`, or `git rebase` in the shared checkout.
- Do not use broad staging (`git add -A`, `git add .`) or `git commit -a`.
- Stage explicit owned paths only: `git add -- path/to/a path/to/b`.
- Inspect the staged set before commit: `git diff --cached --name-only` and `git diff --cached`.
- Do not edit another session's uncommitted path merely to make your own tests/status clean.
- Build outputs, `artifacts/`, and `vendor/` are already isolated/ignored; do not add a coordination mechanism for those directories without a demonstrated collision.

Generated tracked files require the same ownership rule. In particular, do not blindly "revert `releases/*.json` noise": if another session may have generated it, leave it unstaged. Restore a generated file only when you can establish that the change was introduced by your own operation and is not part of another session's work.

Immediately before commit, repeat `git status --short`; immediately after commit, verify the commit contains only owned paths:

```bash
git show --stat --oneline HEAD
git show --name-only --format= HEAD
```

Push normally. Never force-update the shared branch to resolve concurrent movement.

## When isolation is needed

For long-running or mutation-heavy validation, prefer an isolated clone/worktree and a separate work/artifact root while sharing only immutable/cache inputs. The canonical source materialization code already uses detached worktrees for generated source trees; do not build a second general coordination engine around it.

## Locking policy

No repository-wide lock/session registry is justified yet. A global lock would serialize unrelated work and duplicate concerns already handled by explicit ownership plus isolated campaign/build outputs. Add lock/session machinery only after a reproducible collision remains possible while every participant follows this protocol; scope the lock to the smallest mutable resource that actually collides.

## Concurrent-session acceptance exercise

This exercise requires two live sessions on the same checkout; it cannot be proven from GitHub-only review.

1. Session A records `git status --short`, creates an uncommitted probe file in a chosen tracked-area test path, and leaves it present. Capture the transcript outside the checkout (for example under a host temporary directory); do not write evidence into the checkout while the cleanliness claim is being tested.
2. Session B records `git status --short` and must identify A's probe as pre-existing/unowned before changing anything.
3. Session B creates a distinct probe, stages only its own path with `git add -- <B-path>`, and records `git diff --cached --name-only`; A's path must not be staged or modified.
4. Session B unstages/removes only B's probe. Session A verifies its probe is byte-identical, then removes only A's probe.
5. After both sessions have released their probes and `git status --porcelain` is clean, copy/assemble the externally captured transcripts as compact evidence under `docs/evidence/<date>-to03-shared-worktree/` with revision and checksums per `docs/evidence/README.md`. Each session may remove only state it created; never use blanket cleanup or delete unknown files. Preserve failed transcripts and record the blocker rather than marking the exercise passed.

Passing this exercise closes the remaining TO03 evidence requirement. Failure is evidence to revisit scoped locking; it is not permission to introduce a generic coordination service preemptively.
