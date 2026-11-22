# Planning API access from remote agent sessions

This runbook is for BigCherry remote sessions when the `ag-planning`
MCP tools are not projected into the current agent's tool list. It documents a
**second interface to the same authoritative AUDiaGentic implementation**, not
permission to edit `docs/planning/` directly. It also applies to the ledger.

## Find the authority and select the right worktree

- Start with [WORKFLOW.md](WORKFLOW.md) and the project `AGENTS.md`. Use a
  fresh, isolated **BigCherry task worktree**, not the main checkout. Pass that
  worktree to every API call.
- AUDiaGentic is **a separate project**. Its Python API source is
  `src/audiagentic/components/planning/planning_api.py` (delegating to
  `items.py` and `reviews.py`); the MCP server
  `src/audiagentic/components/planning/planning_mcp.py` calls this facade.
  Ledger uses `src/audiagentic/components/ledger/ledger_api.py`.
  Do not copy these modules into BigCherry.
- Prefer the connected `ag-planning` and `ag-ledger` MCP tools when available.
  The AUDiaGentic CLI can **discover** server definitions using
  `audiagentic --project "$BC_WORKTREE" mcp list`; presence in that listing
  does not prove a harness has the MCP connected.
- If MCP tools are absent, call the **same Python APIs** using the AUDiaGentic
  environment. Do not assume missing ChatGPT tool names imply the APIs are
  unavailable.

## Read-only verification

Supply two paths: the separate local AUDiaGentic checkout and the current
BigCherry **task worktree**. Host paths are parameters, not tracked project
configuration. The remote shell tool is stateless between calls and disallows
input redirection/heredocs, so the example uses `python -c` and positional
arguments rather than a here-document or persisted `export`.

```bash
# Replace these two /path/to placeholders before executing this single command.
uv run --project /path/to/AUDiaGentic --no-sync python -c '
import sys
from pathlib import Path
from audiagentic.components.planning import planning_api
from audiagentic.components.ledger import ledger_api

root = Path(sys.argv[1]).resolve(strict=True)
items = planning_api.list_items(root, state="active")
print("Active plan items:", len(items))
if items:
    item = planning_api.get_item(root, items[0]["id"])
    print("Example item:", item["id"], item["state"])
print("Ledger summary type:", type(ledger_api.get_current_summary(root)).__name__)
' /path/to/bigcherry-task-worktree
```

Run the command against your isolated worktree, never the shared main checkout.
This exact `python -c` argument shape was tested through the remote shell tool.

On 2026-11-22, the API import and `get_item(..., "MEN11")` succeeded on
the remote execution host and `list_items(..., state="active")` returned 357 items. These counts
are a point-in-time verification, not a gate. A complete-cohort
`list_items(..., state="completed")` attempt raised `VAL-PLN-035`
(invalid historical planning item ID); do not suppress this integrity failure
or infer completion from a partially readable cohort.

`--no-sync` uses the existing AUDiaGentic Python environment; it **does not
install missing dependencies**. If imports fail, check that environment against
AUDiaGentic's `pyproject.toml` and `uv.lock` and provision it through the
AUDiaGentic project. For example, an incomplete environment may lack
`pydantic` or `platformdirs`; the system `python3` is not a substitute.
Do not change BigCherry's dependencies or commit an AUDiaGentic lockfile
change from a BigCherry task.

## Mutation procedure (Planning API or MCP)

1. Read the existing group, item, linked reviews and evidence in the **task
   worktree**; do not allocate another owner for the same work.
2. For findings-driven changes, call
   `planning_api.create_review(worktree, {"review-of": item_id, "title": ...})`
   **before** revising the plan.
3. Use `planning_api.update_item(worktree, item_id, updates)` for scope,
   validation and acceptance. By default, `notes` are appended; other
   sections are replaced unless included in `append=[...]`. Preserve unique
   acceptance criteria before consolidating or superseding a plan.
4. After resolution, use `planning_api.set_review_state(worktree, review_id,
   "closed")`. Use `planning_api.set_state(worktree, item_id, new_state)` with a valid
   non-active state (`deferred`, `completed`, `superseded`, or `deprecated`) only
   after the correct evidence and lifecycle gates. `deferred` parks paused, resumable
   work in the historical tree; it is **not completed** and must be reopened to
   pending before execution. Do not claim a quality pass for failed/waived gates.
5. For substantive implementation, use the authoritative Ledger API/MCP to
   record the change event with `plan-item-ids`; do not write ledger event
   files manually. Inspect generated plan/review/event diffs, run checks,
   commit from the task worktree and use the standard PR flow.

The Python facade enforces record validation, durability and review-link
rules. Some apparently read-oriented operations may reconcile pending durable
mutations: keep all interactions scoped to the intended **task worktree**.
Do not fall back to manual Markdown/YAML edits if both interfaces are
unavailable. Record the exact failed command and error instead.

## API source of truth

The current API functions and signatures live in AUDiaGentic's
`planning_api.py`, `items.py`, `reviews.py`, and `ledger_api.py`.
Read them before scripting a mutation; do not keep a second argument catalog
or duplicate their validations in BigCherry.
