---
id: RCD08
order: 8
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:52:15.711845+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: M
---

# Implement verified evidence harvest, explicit git commit and canonical reports

## Status

**Production implementation complete for the non-hardware path; Brutus end-to-end acceptance remains pending.**

Implemented:

- `tools/bigcherry/jobs/gitops.py`
- `tools/bigcherry/jobs/harvest.py`
- `tools/bigcherry/jobs/report.py`
- `tools/tests/jobs/test_harvest.py`
- `tools/tests/jobs/test_report.py`
- CLI: `jobs evidence`, `jobs harvest --commit|--stage-only`, `jobs report --format json|markdown`, `jobs review`

The implementation deliberately does **not** copy arbitrary campaign files. `validation_campaign` already writes append-only patch evidence inside each attempt's detached worktree. Harvest identifies only records added relative to that attempt's frozen BigCherry commit, verifies them, and merges them through the existing `bigcherry.patch.evidence` API.

RCD08 remains `pending` only because a complete real Brutus series still has to be harvested/reported and compared with existing qualification output before queue cutover.

## Implemented contract

### Harvest input and identity

For every planned run in one series:

1. latest attempt must have `executor-result.json` with return code 0;
2. attempt worktree evidence must exist at the canonical path resolved by `patch.evidence.evidence_path()`;
3. baseline evidence at the attempt's frozen commit is read with `git show <commit>:<path>`;
4. only record digests added relative to that baseline are candidates;
5. every record digest is recomputed and verified;
6. `patch_id`, focal implementation digest and architecture must match the durable run;
7. scientific identity v2 additionally requires exact validation-adapter digest and resolved `{contract id, contract hash}` bindings to match;
8. the series must contain exactly its predeclared planned session count and one focal patch.

Scientific identity itself now freezes:

- focal/common/promoted implementation + validation digests;
- resolved experiment-contract IDs and hashes;
- `config/recipes.toml` bytes;
- `config/experiment-contracts.toml` bytes;
- model/corpus/file-backed producer input bytes;
- platform environment and stable hardware cohort through RCD04/RCD12.

Attempt start re-resolves that identity from the pinned worktree; drift creates no implicit mutation of an existing series.

### Git transaction

Harvest runs under HI151 `MaintenanceLock` and is fail-closed:

```text
acquire maintenance fence
  -> require empty pre-existing git index
  -> require canonical evidence destination has no tracked/untracked operator changes
  -> verify prior committed harvest for idempotent replay
  -> merge records with patch.evidence.write_record()
  -> git add -- <exact canonical evidence path>
  -> git diff --cached --check
  -> verify staged path set is exactly the harvest path set
  -> optional exact-path commit
  -> record harvest/verified-evidence result + durable event
release fence
```

Never used:

- stash;
- reset of unrelated work;
- `git add -A`;
- implicit path discovery;
- lifecycle promotion.

`--stage-only` is explicitly non-review-ready. `--commit` writes `verified-evidence.json` containing the committed SHA and exact run/session/attempt/record-digest manifest. Repeating the same committed harvest returns `already_harvested` without a second commit.

Unrelated staged changes block harvest. Unrelated unstaged files outside the destination are left untouched. A dirty canonical evidence destination blocks before it is parsed or modified.

## Reporting

`tools/bigcherry/jobs/report.py` reads **only** records named by the committed verified-evidence manifest.

It does not create a second evidence policy. It uses the existing `experiment.contract.bootstrap_session_effect()` implementation, preserving session as the replication unit and the existing >=4-session bootstrap semantics.

Report output contains:

- series/patch/scientific identity/hardware cohort;
- planned and harvested session coverage;
- exact evidence record digests and evidence commit;
- persisted per-contract verdict rollup;
- `(role, metric)` session-bootstrap aggregate effect, CI95 and between-session SD from retained `pair_ratios`;
- record-level eligibility;
- explicit review readiness.

A duplicate `(role, metric)` lane within one session is rejected rather than accidentally double-weighting a session.

Contract verdicts remain authoritative for contract-specific pass/fail interpretation. RCD08 does not add a materiality threshold, alter `improvement_no_regression_v1`, or make reference ladder data gating.

## Offline validation already implemented

The jobs-service CI uses temporary real git repositories to prove:

- exact explicit-path evidence commit;
- idempotent replay/no second commit;
- unrelated staged files are refused and preserved;
- dirty canonical destination is refused before merge;
- contract-hash drift fails closed;
- report requires a committed verified harvest;
- four-session lane aggregation uses the existing session-bootstrap estimator;
- report JSON/Markdown is deterministic for fixed evidence.

The broader jobs suite also covers retry/session identity, scientific-identity drift, torn event tails, hardware cohort binding and executor recovery so failed/retried attempts do not become extra scientific sessions.

## Remaining Brutus acceptance

Before marking RCD08 complete:

1. run one real planned >=4-session series through `bigcherry jobs`;
2. harvest it with `bigcherry jobs harvest <series> --commit` with no manual evidence copying;
3. run JSON and Markdown reports;
4. independently recompute representative target/control intervals from committed `pair_ratios` and compare with the report/legacy qualification tooling;
5. prove harness retries remain linked but do not increase scientific session N;
6. prove a scientific FAIL harvests/reports normally without becoming an operational process failure;
7. confirm no unrelated repository path enters the harvest commit.

## Acceptance criteria

- deterministic/idempotent explicit-path harvest;
- committed evidence is traceable to exact run/session/attempt/record digest;
- validation/contract drift cannot be silently harvested;
- review-ready requires committed verified evidence and planned-session completeness;
- statistical aggregation reuses existing contract estimator/policy;
- no stash/add-A/reset or auto-promotion;
- offline tests green;
- Brutus real-series parity complete.

## Change log

- 2026-09-26: initial design.
- 2026-09-26: specified verified harvest/git transaction and report invariants.
- 2026-09-27: implemented verified record-delta harvest, exact git transaction, scientific identity v2 contract/config freezing, JSON/Markdown reports and temp-git/statistical mocks; hardware acceptance remains.
