# BigCherry task worktrees

The installed AUDiaGentic generator links here from both agent instruction files.
Project policy is maintained in [the shared workflow](../reference/agents/WORKFLOW.md).

Use `audiagentic worktree help` for the installed command syntax and
`audiagentic worktree start SLUG` to create a task. Supply the resulting path to
Planning, Ledger and Gateway MCP calls. Publish using `worktree ship`, wait for
confirmed merge, then use `worktree finish` for that task. Preserve unmerged work
and do not force-remove a worktree. Manual worktrees must remain isolated under
the configured project work root; do not edit the primary checkout.
