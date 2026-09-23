

## Source control doctrine

Never use `git stash`, `git reset`, `git rebase` — this is a shared, multi-agent working tree and destructuve git commands can silently collide with another session's live edits. If work needs to be set
aside, split it into its own deliberate check-in group or leave it uncommitted.

## Tooling architecture and governance

Before adding, moving, or retiring any `tools/bigcherry` module, script, lab
experiment, or compatibility shim, read [`docs/reference/tooling/TOOLING.md`](docs/reference/tooling/TOOLING.md) —
it is the normative source for domain ownership, the compatibility-shim
policy, lab/non-package boundaries, hygiene checks, and evidence/acceptance
rules produced by the tooling-rationalisation program (TR00–TR18,
`docs/planning/*/rationalisation/`). Do not create a second campaign engine,
patch loader, or evidence framework without an explicit architecture review.

## Ad-hoc output placement

Never write logs, scratch scripts, or one-off run output directly into the
repo root — it is a shared, multi-agent working tree and loose root files are
repo-wide clutter, not scoped to any one session.

- Ad-hoc command output / logs (e.g. `cmd > foo.log`) go under
  `artifacts/logs/manual/`, not the repo root.
- One-off or exploratory shell/python scripts go under `tools/lab/<topic>/`,
  named for what they're for — never loose in the repo root or in `/tmp`.
- Raw/large evidence for a specific plan-item run (benchmark traces, worktree
  copies, measurement jsonl) goes under `artifacts/<run-id>/` where `run-id`
  is the owning plan-item ID (optionally with a short suffix) — create it via
  `bigcherry.core.paths.evidence_dir(run_id)`, never a free-text name. See
  [`docs/reference/tooling/TOOLING.md`](docs/reference/tooling/TOOLING.md)'s
  "Evidence and acceptance boundaries" section — `bigcherry check --quick`
  enforces this (`TR14.ARTIFACT_UNTRACEABLE_RUN`).
- `artifacts/` and `tmp/` are both gitignored; that does not make them a free
  dumping ground — keep output inside a named subdirectory for the run/task
  it belongs to, not loose at their top level.
- Never write output outside the project folder: not the user profile
  (`~`, `%LOCALAPPDATA%`, `~/.cache`), not the system temp dir, not a drive
  root. Tool caches, builds, mirrors and isolated worktrees live under the
  gitignored project-local `work/` (the `ProjectContext.work_root` default;
  override only with `BIGCHERRY_WORK_ROOT`). Agents needing an isolated
  checkout create it under `work/worktrees/`, and keep scratch helpers in
  `work/` or `tools/lab/<topic>/` — never in ad hoc hidden directories.
- Never commit host-specific values (absolute machine paths, user home
  directories, drive roots, host addresses, device PCI locators). Only
  project-relative paths belong in git. Host facts live in the untracked
  `config/environment.local.toml` (template: `config/environment.example.toml`,
  file location overridable with `BIGCHERRY_ENVIRONMENT`, any key with
  `BIGCHERRY_HOST_<KEY>`); tracked config refers to install locations as
  `${VAR}` (e.g. `${HIP_PATH}`, `${ROCM_PATH}`, `${BIGCHERRY_BRUTUS_TREE}`).
  `bigcherry check --quick` enforces this (`TR14.HOST_SPECIFIC_VALUE`,
  `TR14.USER_FOLDER_DEFAULT`).

<!-- ag:managed:begin -->
_Managed by AUDiaGentic — generated from component configs. Edit the owning component and re-run surface apply; edits here are overwritten._

## Agent ledger process

After substantive implementation work, record a change event with the ag-ledger
MCP tool record_change_event — the ledger is the authoritative release record.
Required fields: change-class, files, technical-summary, user-summary-candidate,
status ('unreleased'). Other fields are auto-generated.
- Check release ledger state before changing release notes, changelog fragments, or release workflow files.
- Keep release artifacts and job records synchronized with implementation and review outcomes.
- Do not bypass ledger updates by editing generated release outputs only.

When recording a change event for work associated with one or more plan items,
include "plan-item-ids" in the event dict (e.g. {"plan-item-ids": ["CC07", "CC08"]}).
The planning component will automatically link the ledger event ID to those items.

## Planning process

Use the ag-planning MCP tools to manage plan items in docs/planning/.

## When to use
- User asks to create a plan or work items 
- Tracking multi-step implementation across sessions
- Reviewing or updating the state of outstanding items

## Item lifecycle
1. Create items with plan_create_item — lands in docs/planning/active/<plan>/
2. Revise content with plan_update_item as work progresses; for a findings-driven correction (not routine progress), record a plan_create_review first, then close it once incorporated
3. After review triage, close handled reviews with plan_set_review_state(review_id, 'closed')
4. Mark done with plan_set_state(item_id, 'completed') only when implementation and validation are done
5. Keep unfinished work pending or in terminal discard states (superseded, deprecated); remove stale items with plan_delete_item

## Item ID convention
Combine a short uppercase plan prefix with a sequence number: CC07, LSP01, ML01.
Choose a prefix matching the plan name (CC → code-cleanup, LSP → lsp-mcp-enhancement).

## Required fields for plan_create_item
- plan: plan directory name (e.g. "code-cleanup")
- title: short descriptive title

## State discipline
- Do not leave incorporated reviews in created/considered; close them.
- Do not mark a parent item completed just because reviews were handled.
- If an active item is superseded, transition to 'superseded' or 'deprecated' rather than deleting (preserves history and linked reviews).

## Optional fields (fill out where applicable to provide complete planning context)
- work: S / M / L — blast radius of the change
- skill: basic / intermediate / advanced — cognitive difficulty
- order: integer sort key (default 0)
- description: Body section describing the work
- steps: Implementation steps
- detailed_solution: Detailed Solution & Technical Design (architecture, components, design)
- code_samples: Code Samples & Guidance (including config samples and schemas)
- files: Files to create/update
- validation: How to validate the implementation
  (include comprehensive tests where possible)
- effort_risk: Complexity and risk assessment
- standards: Applicable standards/rules
- notes: Key design principles and additional context

## Execution profile doctrine

Execution profiles bind a provider to a specific model with optional
execution parameters. They are stored in the canonical
.audiagentic/config/agents.yaml document alongside prompts, roles,
agent definitions, and triggers.

## When to use
- A job needs a predefined provider+model configuration
- Execution parameters (temperature, max-tokens) should be
  profile-driven
- Multiple projects need different default model configurations

## Resolution precedence at job launch
1. Explicit `execution-profile-id` in job request
2. Explicit provider-id / model-id in job request
3. Execution profile selected by the Agent Definition
4. Default execution profile (marked `is-default: true`)

## Naming
Use `execution-profile-id` (NOT `profile-id`) in job requests to avoid
collision with `workflow-profile` (lite/standard/strict stage
pipelines).

## Role selection precedence (AS61/RO01)

Today a Role is reached only through `agent_id` -> Agent Definition ->
`role_id` (agent_task_submit). There is no standalone `agent-role-id`
job-request field yet — that wiring belongs to AS78, not this
component. This section states the precedence/conflict rule that
MUST hold once one is introduced, so the rule is fixed before the
field exists rather than improvised after.

## Resolution precedence at job launch (once agent-role-id exists)
1. Explicit `agent-role-id` in job request
2. Role carried by an explicit `agent-id` (Agent Definition)
3. Execution profile selection (`execution-profile-id` / provider-id
   + model-id / default execution profile) is independent of role
   selection — they compose, they do not override each other.
4. `workflow-profile` (lite/standard/strict stage pipelines) and
   `component-profile` (per-process config layer) are orthogonal to
   both and never interact with role/profile precedence.

## Conflicts fail closed
A Role requirement that contradicts an explicitly selected Execution
Profile is an error raised before dispatch — never a silent override
in either direction, and never a permissive fallback. This applies
the moment AS78 exposes a path where the two could disagree; there is
no such path yet, so no conflict-checking code exists today.

## Naming
`agent-role-id` must stay distinguishable from `execution-profile-id`,
`workflow-profile`, and `component-profile` — three distinct
"profile" concepts already exist in this project; a role is a fourth,
separate axis and must not collapse into any of them by accident.

## Source control doctrine

Never use `git stash` — this is a shared, multi-agent working tree and a stash
can silently collide with another session's live edits. If work needs to be set
aside, split it into its own deliberate check-in group or leave it uncommitted.
<!-- ag:managed:end -->
