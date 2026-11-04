---
id: BRL01
order: 0
plan: build-release-ledger
state: pending
created-at: '2026-11-04T23:32:02.160482+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: M
priority: P1
---

# Cut bigcherry over to the per-event ledger (AUDiaGentic LE03): loader in the release notes, CI dependency, shared venv, stale-branch conversion

## Description

AUDiaGentic's ledger redesign (LE03) is on main there: slice 1 (#443, per-event store and loader), slice 2 (#445, plan-item links derived at read time, no plan-file writes), slice 3 (#447, release workflows, conversion tool, migration guide). Each change event is its own file under docs/releases/events/ (canonical sorted JSON, write-once); CURRENT_RELEASE_LEDGER.ndjson becomes a frozen baseline nothing writes. That removes the ledger conflict resolved by hand (ledger-union helper) on almost every PR. Published: 0.96.0 (slices 1 and 2); 0.97.0 adds slice 3 (release PR open when this was written). Facts from AUDiaGentic 2026-11-04: it is installed from PyPI (audiagentic>=0.97, or the vX.Y.Z git tag); audiagentic.components.ledger.current_events is NOT importable alone, it needs the whole package (44 packages, about 150 MB on Python 3.14, no slim wheel scheduled); iter_events(project_root, ref=None), pending_events, released_events, anomalies, event_file_name are the supported public API, stable from 0.97, needing no .audiagentic directory; with ref= it reads blobs through git only. The conversion tool convert_legacy_ledger(base_ref) refuses branches that merged main with a union conflict resolution (our usual shape) in 0.97.0: AUDiaGentic has a fix in review and will name the release (probably 0.97.1 or 0.98.0); HOLD the cutover until that release is published. State of ours: the shared dev venv has audiagentic 0.88.1 (no current_events); tools/bigcherry/release/notes.py parses CURRENT_RELEASE_LEDGER.ndjson itself and runs in GitHub Actions where audiagentic is not installed; config/release.toml names the ledger path and has feat and perf rows (11 rows in the baseline use the old classes feat/perf, one row lacks user-summary-candidate and a timestamp).

## Steps

1. Wait for the AUDiaGentic release that contains the union-merge conversion fix (they will name the version); do not cut over on 0.97.0 alone.
2. OWNER DECISIONS first: (a) upgrade the shared dev venv to that version and have every session restart its ag-ledger MCP server (affects all sessions and bigcherry-2's open work); (b) accept audiagentic as a pinned dependency of the release-notes job in GitHub Actions (recommended: no second loader to maintain), or have notes.py read the event files itself.
3. Release notes: tools/bigcherry/release/notes.py reads events through current_events.iter_events(project_root, ref=...) per engine line (filter on files, change-class, timestamp-utc as today); config/release.toml maps the old classes feat/perf to the feature section (alias, no silent drop); tests with a synthetic repo holding baseline rows and event files.
4. CI: the release workflows install the pinned audiagentic version (pin recorded in one place), cache it, and fail closed on a mismatch with the pin; confirm the release-please run time impact.
5. Convert every stale branch BEFORE it merges or rebases onto the frozen main: convert_legacy_ledger(base_ref=origin/main) per worktree (list the worktrees and PRs that still touch CURRENT_RELEASE_LEDGER.ndjson; tell every session). Retire tools-side helpers that exist only for the conflict (ledger-union script in the scratch area, any docs that tell agents to resolve it).
6. Docs and skills: docs/releases/README.md, .claude/skills/bigcherry-slice-workflow/SKILL.md (agents must COMMIT docs/releases/events/*.json with their work; record_change_event returns event-path and pending-count; get_plan_item_events for reverse links), AGENTS/CLAUDE guidance if it names fragment-path.
7. Fix or explicitly accept the 12 legacy rows (11 feat/perf, 1 without user-summary-candidate and timestamp) if bigcherry ever calls AUDiaGentic's archive_for_release; today it does not.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Release notes for a test range built from the old ndjson and from a mix of baseline rows and event files are identical for the same events; a PR that records an event touches only its event file and merges main with no ledger conflict; the release workflow runs with the pinned package; every converted branch's diff no longer touches CURRENT_RELEASE_LEDGER.ndjson; the ledger MCP tools work in a restarted session on the upgraded venv.

## Effort & Risk



## Standards



## Acceptance Criteria

No PR needs a hand-resolved ledger conflict; release notes read the new layout through the supported loader; CI pins the dependency; no open branch still appends to the frozen baseline; agents' docs and skills describe the event-file flow.

## Notes

Cross-session facts from AUDiaGentic 2026-11-04 (messages from the audiagentic session). Hold until the conversion fix release is published and the owner has decided the shared-venv upgrade and the CI dependency.

## Change Log

- 2026-11-04T23:32:02.160482+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261104_233207_planned-the-move-to-the-confli_2775
- 2026-11-04T23:32:11.627988+00:00 (updated-by): Updated: section:ledger-events
