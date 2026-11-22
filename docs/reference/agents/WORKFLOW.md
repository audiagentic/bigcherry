# Shared agent workflow

Project policy for every agent. `AGENTS.md` and `CLAUDE.md` expose the same
project-owned instruction contribution, maintained in
[bigcherry-workflow.yaml](../../../.audiagentic/config/project/instructions/bigcherry-workflow.yaml).
Framework component procedures come from the installed AUDiaGentic descriptors;
change their owners rather than patch generated output. Container instructions
apply in addition to this project policy.

## Worktrees and shared resources

Start each task from fresh `origin/main` in its own branch/worktree. The primary
checkout is for fetch/pull only. Follow the installed AUDiaGentic task-worktree
commands when available, and pass the resulting worktree explicitly to MCP calls.
For manually created worktrees, use the configured project work root's
`work/worktrees/` directory. Honor a container-mandated scratch location when
present; otherwise keep task scratch in project-local `work/`.

Never stash, reset or rebase a shared checkout. Do not force-push or force-remove
worktrees. Leave unrelated edits untouched. Acquire the container's measurement
lease before benchmarks/profiling/KLD; do not start heavy builds during another
actor's measurement. Stop only processes you started, by recorded PID.

## Planning, ledger and gateway

Use Planning MCP for multi-session work, requested plans, and plan/review state.
First list existing groups and use the owning group. New group names start with
`build-`, `patching-`, `run-` or `tuning-`. Items use an uppercase topic prefix and
sequence number. Include concrete scope, files, validation and acceptance.
For findings-driven corrections, create the review before revising the item;
close incorporated reviews. Complete items only after implementation, validation,
nonempty acceptance criteria and closure of linked reviews. Keep actionable unfinished
work pending; move explicitly paused but resumable work to `deferred` only after
recording a reason and re-entry gate, then reopen to pending before execution.
Use superseded/deprecated only for displaced or rejected scope; do not delete history.

After substantive implementation, record a Ledger MCP change event with
`change-class`, `files`, `technical-summary`, `user-summary-candidate` and
`status: unreleased`. Include applicable `plan-item-ids`. The current ledger uses
per-event files under `docs/releases/events/`; commit the event with the work.
Do not append to a frozen historical ledger snapshot. Read ledger state before
changing release notes or workflow outputs. Use current tool schemas as the
parameter authority; do not maintain copied API-field catalogs in entry files.

Gateway MCP manages explicitly authorized agent jobs/sessions. Availability of
a gateway does not itself authorize dispatch or override the session's delegation
rules. Bind a job to the intended worktree and inspect its result before reporting
it complete. Use configured execution profiles rather than copying provider/model
settings: `execution-profile-id` is distinct from workflow profile and role ID.
Explicit profile selection precedes an agent definition's profile and its default.
If the profile conflicts with role requirements, refuse the conflict. Do not invent
an `agent-role-id` tool argument; use the exposed tool schema.

A configured MCP connection and a verified tool call are separate facts. When the Planning or Ledger MCP tools are not exposed in an agent session,
check the underlying AUDiaGentic Python API facade before declaring the lifecycle
blocked. Follow [Planning API access](PLANNING_ACCESS.md): it calls the same
authoritative implementation against the task worktree, not handwritten plan or
ledger files. If neither interface works, report the limitation and continue
unaffected work. Do not bypass the authority by editing plan or ledger state.

## Code and migration policy

Before adding, moving or retiring tooling, scripts, lab experiments or shims,
read [TOOLING.md](../tooling/TOOLING.md). Use the existing owner; do not create a
second campaign engine, patch loader or evidence framework. Lab code stays
self-contained and must not be imported by production.

When changing an internal interface, migrate all internal callers and tests in
the same change. Do not add optional arguments or alternate paths solely to keep
old internal behavior reachable. Name and justify any real external consumer
exception. Existing compatibility entrypoints explicitly retained by
[TOOL_DISPOSITION.md](../tooling/TOOL_DISPOSITION.md) remain governed by its
retirement gates; this migration rule is not permission to delete them.

## Files, evidence and cleanup

Keep builds, caches, source mirrors and worktrees in the configured project work
root. Do not write task files into the user profile or loose repository root.
Manual logs go under `artifacts/logs/manual/<task>/`; lab scratch code under
`tools/lab/<topic>/`; generated lab output under `artifacts/lab/<topic>/`.
Plan-owned raw evidence uses the owning run/item ID and the maintained
`bigcherry.core.paths.evidence_dir` API. Compact tracked proof belongs in
`docs/evidence/`; live scope and acceptance belong in `docs/planning/`.

Tracked configuration uses project-relative paths and variable references.
Host-specific paths, addresses and device locators belong in local environment
configuration or local receipts. Do not commit credentials.

Cleanup must preserve referenced results, reproducibility inputs and other
actors' files. An installed custom kernel's source, patch series, config,
generated headers and `Module.symvers` are protected build prerequisites; check
installed module build/source links and boot references before removing them.
An ignored directory or old timestamp alone does not establish disposability.

## Validation and completion

Run checks appropriate to the changed behavior. Distinguish patch applicability,
compilation, hardware activation, numerical correctness and performance evidence.
Do not claim a capability or speedup from clean anchors or offline tests alone.

Publish through a PR and wait for required checks and confirmed merge. Never
bypass branch protection. After merge, use the supported worktree finish/cleanup
flow; remove only this task's safe scratch. Report merge/push state and any
remaining work explicitly. Do not leave a queued auto-merge described as merged.

## Regenerating entry files

The project source is `.audiagentic/config/project/instructions/bigcherry-workflow.yaml`.
Use the installed AUDiaGentic surface generator for both `codex` and `claude`.
Preview changes and inspect the diff: component install/enable state determines
which framework procedures appear. Do not copy a stale managed block to preserve
an outdated procedure. Regeneration must be idempotent and must retain the same
project workflow content in both provider files.

The shared workspace root may have a local `AGENTS.md` pointing to the canonical
checkout for policy discovery. It must instruct sessions to create task worktrees,
not to edit that primary checkout. Its machine-specific path stays outside Git.
