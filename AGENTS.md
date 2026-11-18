# BigCherry agent instructions

Project rules are shared across agents; see the generated workflow below.

<!-- ag:managed:begin -->
_Managed by AUDiaGentic — generated from component configs. Edit the owning component and re-run surface apply; edits here are overwritten._

## BigCherry workflow

Read [the shared project rules](docs/reference/agents/WORKFLOW.md) before making changes.
This is the same project policy for Claude, Codex and other agents.

- Work in an isolated task worktree based on fresh origin/main; never edit the primary checkout.
  Respect container resource leases. Do not stash, reset, rebase, force-push or force-remove worktrees.
- Use Planning MCP for plan/review lifecycle and Ledger MCP for substantive changes.
  Pass the task worktree explicitly; commit generated plan and event files with their work.
  If tools are unavailable, report the missing connection; do not edit authoritative records by hand.
- Keep project-local builds, caches and task worktrees under the configured work root.
  Preserve referenced evidence, installed-kernel build artifacts and other actors' work during cleanup.
- Read docs/reference/tooling/TOOLING.md before adding or moving tooling or lab code.
  Migrate internal callers together; do not introduce new legacy fallbacks. Retained compatibility
  entrypoints follow the tooling disposition and retirement gates, not blanket deletion.
- Validate the actual change, open a PR, wait for required checks and merge confirmation,
  then clean only this task's merged worktree and scratch. Report what was observed.

These files are generated. Edit this project instruction source or the owning AUDiaGentic
component, then regenerate both AGENTS.md and CLAUDE.md; do not hand-edit managed blocks.

## MCP onboarding

Configure MCP access only for explicitly requested harnesses.
Discover servers with `audiagentic mcp list` and targets with
`audiagentic harness list`. Use `audiagentic mcp MCP_ID help` for setup.
Install with `audiagentic mcp install MCP_ID harness ID[,ID...]`.
This sets that MCP's complete native target selection; other MCPs remain unchanged.
Component installation alone does not select native MCP exposure.
For every directory instead of one project, `audiagentic setup harness ID[,ID...]`
registers the worktree-scoped servers at that harness's user scope
(`dry-run` reports only); their tools take a per-call `worktree`.
Report configured separately from verified.

## Task worktrees

One task is one branch in its own git worktree; never edit the primary checkout.
- `audiagentic worktree start SLUG [item ID]` creates branch OWNER/SLUG from the fresh
  remote default branch and prints the worktree path. Pass that path as the `worktree`
  argument of the planning, ledger and gateway tools.
- Publish with `audiagentic worktree ship title "type(scope): subject" body TEXT`. It syncs the
  branch with the default branch, runs the pre-push checks the project names, pushes the exact
  checked commit without force, opens or reuses the pull request and asks GitHub to
  auto-merge, then returns. It never rebases and aborts on a conflict. Run `ship` again
  whenever the default branch moved (a release landed) and the pull request shows BEHIND;
  it reports `merged` when done. `sync`, `push` and `merge [dry-run yes]` are the single steps.
- After `ship` reports merged, `audiagentic worktree finish BRANCH` preserves the work
  (quarantine ref, archive) and removes the worktree and branch. It refuses unmerged work;
  `abandon TIP12` discards it only after a verified bundle. `prune` reports what is eligible.
- Never use `git worktree remove --force`, force-push, or edit the primary checkout.
Details: docs/guides/WORKTREES.md.
<!-- ag:managed:end -->
