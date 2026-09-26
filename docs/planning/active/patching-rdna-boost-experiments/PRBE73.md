---
id: PRBE73
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:58:39.838631+00:00'
breadth: ''
skill: ''
created-by: capability-rebaseline-v3
priority: null
---

# Hardware-blocked patch retirement policy (RD21/RD22-class items)

## Description

TODO. No hardware_blocked_since/last_source_recheck/last_upstream_equivalence_check/next_review_date fields exist in the planning schema today (grep of docs/planning and tools/ for these names finds only this item's own text). PRBE16 (gfx1151 defer item, blocked on missing hardware) is the concrete real example this policy would apply to.

TODO, ownership and factual basis corrected per GPT review. Two real defects: (1) ownership of the planning schema was never resolved -- this item's own step 1 already flagged 'confirm which repo owns this' but the plan proceeded to design fields/audit tooling anyway without that answer; must resolve first, no coding agent can proceed otherwise. (2) Step 6's premise is now STALE per this same batch's PRBE72 correction: RD22's real experiment-contract scope (config/experiment-contracts.toml ~488) is `architectures = ["gfx1201"]` with NO integrated/uma traits set -- NOT 'gfx1151+integrated+UMA' as this item's prior draft claimed (that was Scope's own docstring aspiration, not RD22's actual current binding; PRBE72 tracks migrating RD22 to the traits-based form). This item's own audit tooling must use RD22's REAL current scope, not the not-yet-migrated target scope.

## Steps

1. Locate the planning item schema/validator (the ag-planning MCP tool's backing store -- grep the repo for where docs/planning/*/*.md frontmatter fields like 'state', 'work', 'skill' are validated, likely a Python module under tools/ or a separate planning-tool package; if the planning tool itself is not in this repo, this item's schema change may belong to that tool's own repo instead -- confirm and note which repo owns it before writing code).
2. If schema lives in this repo: add four optional frontmatter/body fields to hardware-blocked items: hardware_blocked_since (date), last_source_recheck (date), last_upstream_equivalence_check (date), next_review_date (date, auto-computed as blocked_since + threshold, e.g. 90 days, unless explicitly overridden).
3. Add a validator/report script (tools/bigcherry/... or tools/lab/plan-hardware-blocked-audit/) that scans docs/planning/active/*/*.md for items whose state is pending AND whose description/notes mention hardware-blocked/gate-verified-blocked (PRBE16's own pattern) but lack these fields, flagging them for retrofit.
4. Add a second check: items past next_review_date get listed in a 'stale hardware-blocked items' report requiring an explicit human retire/recheck decision (transition to deprecated/superseded, or a fresh next_review_date with a recorded reason) -- never auto-deleted, never silently carried forward.
5. Apply the new fields to PRBE16 itself (hardware_blocked_since = its creation date 2026-09-09, last_source_recheck = today if this pass re-verifies patch 1208 still matches, next_review_date = +90 days) as the first real audit case.
6. Recheck RD21/RD22-class items specifically for whether current gfx1201 hardware now qualifies RD22 (RD22's Scope per PRBE72 is `integrated=True, uma=True` targeting gfx1151 specifically -- gfx1201/R9700 is a discrete GPU, NOT integrated/UMA, so RD22 remains genuinely hardware-blocked on gfx1151 acquisition; record this explicitly rather than leaving it unstated).

1. Resolve ownership FIRST, before any code: check whether the ag-planning MCP tool (the actual backend behind plan_update_item/plan_create_item) is a package in THIS repo (grep for its source, e.g. `find . -iname '*ag_planning*' -o -iname '*ag-planning*' 2>/dev/null | grep -v node_modules`) or an external service/repo. If external, this item's schema-field-addition scope belongs in that tool's own repo/backlog, not here -- file it there and narrow this item to only the BigCherry-side audit/reporting tooling (which can read planning item text/notes regardless of where the schema lives).
2. If BigCherry does own relevant policy config (even without owning the planning tool's schema): add a typed policy module under tools/bigcherry/ (e.g. tools/bigcherry/planning/hardware_block_policy.py) defining the threshold (configurable, default 90 days, not hardcoded) and the field names this project expects items to carry (hardware_blocked_since, last_source_recheck, last_upstream_equivalence_check, next_review_date) as a documented convention layered on top of whatever the planning tool's `notes`/body text actually supports -- do not invent new frontmatter fields the planning tool itself does not persist.
3. Add a `python -m bigcherry planning-audit` (or equivalent) command/script that reads docs/planning/active/*/*.md via the planning tool's own read interface (not raw regex over markdown alone -- use `plan_get_item`/`plan_list_items` where available so it doesn't drift from however the tool actually stores state), extracts the convention fields from notes/body where present, and reports missing-metadata and past-next_review_date items using `next_review_date` (not free-text keyword inference) as the staleness signal once that field is actually populated by convention.
4. Do not infer 'blocked' state from free-text regex matching alone (the prior draft's `hardware.?blocked|gate-verified-blocked` regex) as the sole detector -- cross-reference against items whose `state` is `pending` AND whose Scope (per PRBE72, once migrated) or explicit hardware note names a device/trait this project's own tracked hardware (Brutus: gfx1100/gfx1201, no gfx1151) does not satisfy.
5. Apply the convention fields to PRBE16 (gfx1151 defer item) as the first real audit case: hardware_blocked_since = 2026-09-09 (creation date), next_review_date = +90 days, recorded via plan_update_item's notes append (not invented frontmatter).
6. Recheck RD21/RD22 using RD22's REAL current scope (config/experiment-contracts.toml ~488: architectures=["gfx1201"], no integrated/uma) rather than the stale 'gfx1151+integrated+UMA' assumption -- RD22 is bound to gfx1201 today, which IS present on this project's hardware (Brutus, R9700), so RD22 is not necessarily hardware-blocked in the same way PRBE16 (a genuine gfx1151-only item) is; this needs its own fresh determination once PRBE72's migration lands, not inherited from the old premise.

## Detailed Solution & Technical Design

This is planning-metadata tooling, not ggml/C++ source. The validator should be a small, testable Python script following the style of existing tools/bigcherry/campaign/ modules (see qualification_matrix.py for the dataclass/fail-closed pattern already used for ExperimentContract). Threshold default (90 days) should be configurable, not hardcoded, so future policy tuning doesn't require code changes.

## Code Samples & Guidance

Validator sketch (tools/lab/plan-hardware-blocked-audit/audit.py):
```python
import datetime, re, pathlib

THRESHOLD_DAYS = 90
PLAN_ROOT = pathlib.Path("docs/planning/active")

def find_hardware_blocked_items():
    for md in PLAN_ROOT.glob("*/*.md"):
        text = md.read_text(encoding="utf-8")
        if re.search(r"hardware.?blocked|gate-verified-blocked", text, re.I):
            yield md, text

def check_staleness(md, text):
    m = re.search(r"hardware_blocked_since:\s*(\d{4}-\d{2}-\d{2})", text)
    if not m:
        return ("missing-metadata", md)
    since = datetime.date.fromisoformat(m.group(1))
    if (datetime.date.today() - since).days > THRESHOLD_DAYS:
        return ("stale-needs-review", md)
    return ("ok", md)
```
(schema fields are added to item frontmatter/notes via the existing ag-planning plan_update_item tool, not by hand-editing markdown, once the field names are confirmed against wherever the planning tool's own schema is defined -- step 1.)

## Files

tools/lab/plan-hardware-blocked-audit/audit.py; docs/planning/active/patching-rdna-boost-experiments/PRBE16.md (retrofit); wherever the planning schema itself lives (repo-external if the ag-planning tool is a separate package -- confirm in step 1).

## Validation

Unit tests for the audit script (mock items with/without metadata, past/future threshold dates); a real run of the audit against docs/planning/active/ producing a report listing every currently-flagged item (PRBE16 at minimum) and whether it has the new fields.

## Effort & Risk

S-M; low code risk, but step 1's repo-ownership question (is the schema in this repo or the planning MCP tool's own repo) must be resolved first or the whole plan is misdirected.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Every hardware-blocked item has review metadata and is flagged after the threshold for explicit retire/recheck; no silent accumulation or automatic deletion.

## Notes

Supersedes: RD93
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd93

2026-09-24 relevance at b11126: TODO, no existing schema fields or validator found (grep confirms absence). RD21/RD22 recheck done as part of this planning pass: gfx1201/R9700 is discrete, not integrated/UMA, so RD22 remains genuinely blocked -- not newly qualified by current hardware. GPT design request: gateway rejected all submissions this session (VAL-AGW-025 / EXT-GPTAUTO-003); plan authored directly -- no GPT request id.

2026-09-24 GPT review req_d55aed71224e43a8 applied: NOT-READY -- schema ownership must be resolved before implementation (may belong to the external ag-planning tool, not this repo); corrected stale RD22 premise (real current scope is gfx1201, not gfx1151+integrated+UMA, per this batch's PRBE72 finding); removed free-text-only blocked-state inference in favor of next_review_date once populated.

## Change Log

- 2026-09-09T10:58:39.838631+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:50.884404+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.458618+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.298240+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:20:26.481445+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032047_repaired-five-more-active-succ_6361
- 2026-09-10T03:20:47.960815+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:35:51.439964+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:47:33.405857+00:00 (updated-by): Updated: section:description, section:steps
- 2026-09-24T04:47:39.152629+00:00 (updated-by): Updated: section:notes
