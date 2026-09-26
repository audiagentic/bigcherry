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

## Description

Replace manual evidence copying plus `summarize.py`/`noise.py` with an idempotent BigCherry harvest/report path. Harvest consumes only job-service evidence whose run/series/attempt/contract/composition/build/hardware identity verifies, copies explicit known files into canonical package evidence locations, stages only explicit paths, checks the staged diff, commits it, and records the resulting commit SHA. It never stashes, `git add -A`, silently edits unrelated files or marks a result review-ready before evidence is committed.

Reporting remains separate from scheduler success: scientific PASS/FAIL is read from verified evidence; ladder is reference-only; anomalous rounds are surfaced but not automatically replaced.

## Steps

1. Inventory current `tools/lab/plan-qualification/summarize.py` and `noise.py` calculations/fields; extract reusable logic without changing evidence policy.
2. Add harvest manifest/schema and identity verifier.
3. Add canonical destination resolver using patch package/evidence APIs.
4. Implement explicit-copy + explicit-stage git transaction and idempotency.
5. Add session/series/patch report models and JSON/Markdown renderers.
6. Add review-readiness rule requiring planned N valid sessions and recorded evidence commit(s).
7. Add offline temp-git tests for dirty trees, conflicts, duplicate harvest, tampering and report aggregation.
8. Integrate as RCD07 operation later; v1/v1.5 may invoke `bigcherry jobs harvest` manually/after completion.

## Detailed Solution & Technical Design

### Harvest input

```python
@dataclass(frozen=True)
class HarvestItem:
    run_id: str
    series_id: str
    attempt_no: int
    source_path: Path
    source_sha256: str
    destination_relpath: str
    record_digest: str

@dataclass(frozen=True)
class HarvestManifest:
    schema: str
    series_identity_hash: str
    items: tuple[HarvestItem, ...]
```

Before git mutation verify:

- run/series/attempt exists and terminal;
- evidence record validates through existing `patch.evidence`;
- run_id/session/contract/base/focal/common/frozen composition match series intent;
- platform environment + hardware cohort match series;
- source byte hash equals manifest;
- destination is inside allowed patch evidence/report roots and contains no path traversal;
- scientific record is not from an interrupted/invalid attempt.

### Git transaction

```python
def harvest(
    manifest: HarvestManifest,
    *,
    repo: Path,
    commit: bool,
) -> HarvestResult: ...
```

Algorithm:

1. acquire HI151 maintenance lock or a dedicated harvest tree lease compatible with maintenance policy; no campaign runner may be using canonical tree;
2. refuse if any destination has unrelated staged changes;
3. unrelated unstaged files may exist only if they are outside destinations and commit pathspecs; record them, never stash/touch;
4. copy to same-directory temp + fsync + atomic replace;
5. `git add -- <explicit destination ...>`;
6. `git diff --cached --check`;
7. inspect staged name list equals exact intended destinations;
8. commit with deterministic message containing series/run IDs, or in `--no-commit` mode return staged state for explicit operator workflow;
9. record commit SHA in run-store harvest result/event.

If commit fails, leave explicit staged changes and report exact recovery; never reset unrelated user work.

Idempotency:
- if destination bytes and recorded harvest commit already match, return `already-harvested`;
- same destination with different identity/content is a conflict, never overwrite silently.

### Reporting

Library models:

```python
def session_report(record: ValidationRecord) -> SessionReport: ...
def series_report(series: SeriesRecord, sessions: tuple[SessionReport, ...]) -> SeriesReport: ...
def render_report_json(report: SeriesReport) -> dict[str, object]: ...
def render_report_markdown(report: SeriesReport) -> str: ...
```

Series report includes:

- planned/valid/missing session count;
- per-session target/control effects and CI inputs;
- aggregate session-bootstrap result under contract policy;
- control-lane regression bound;
- reference ladder table clearly `reference_only`;
- production-lane result if present;
- noise/round diagnostics and telemetry flags;
- harness failures/retries excluded from scientific sample count but linked;
- evidence commit SHA(s);
- `review_ready` only when planned N is complete, all required evidence validates/committed and no identity drift/block remains.

No materiality threshold is added beyond the contract; small positive wins remain eligible under `improvement_no_regression_v1`.

## Code Samples & Guidance

Git must be invoked as argv with `--` path separator. Never infer harvest destinations from arbitrary source filenames; destination resolver owns allowed mapping.

Reporting functions are pure over verified records so they are testable without Slurm/GPU.

## Files

Planned:

- `tools/bigcherry/jobs/harvest.py`
- `tools/bigcherry/jobs/gitops.py`
- `tools/bigcherry/jobs/report.py`
- `tools/tests/jobs/test_harvest.py`
- `tools/tests/jobs/test_report.py`
- migrate reusable computations from:
  - `tools/lab/plan-qualification/summarize.py`
  - `tools/lab/plan-qualification/noise.py`

## Validation

Offline temp-repo tests:

- harvest exact one session and commit;
- repeat exact harvest -> idempotent/no new commit;
- different bytes same destination -> conflict;
- tampered source hash -> refuse before git mutation;
- unrelated unstaged file preserved byte-for-byte;
- unrelated staged file -> refuse rather than mix commit;
- staged destination set exactly equals manifest;
- `git diff --cached --check` failure aborts commit;
- path traversal/outside evidence root rejected;
- missing planned session => `review_ready=false`;
- four valid sessions aggregate deterministically;
- harness retry attempts do not count as extra scientific sessions;
- scientific FAIL reports normally and is not process failure;
- ladder cannot gate final verdict;
- uncommitted evidence => `review_ready=false`.

Hardware acceptance: complete one real series, harvest without manual copy, compare new report with legacy summarize/noise values for equivalent inputs.

## Effort & Risk

Medium. Git mutation is high consequence; keep pathspecs explicit and fail closed around preexisting staged state.

## Standards

Existing patch evidence validation is authoritative. Owner evidence policy remains improvement CI lower bound >0, control regression CI upper <=1%, >=4 planned sessions unless a contract explicitly differs.

## Acceptance Criteria

- harvest is deterministic/idempotent and cannot include unrelated paths;
- every review-ready record is traceable to evidence commit SHA;
- reports reproduce contract/statistical semantics and mark ladder reference-only;
- no stash/add-A/reset of unrelated work;
- offline git/report tests pass.

## Notes

Do not auto-promote patch lifecycle state from harvest. Promotion/rejection remains a deliberate lifecycle operation after evidence review.

## Change Log

- 2026-09-26T00:52:15.711845+00:00 (created-by): Created by agent
- 2026-09-26 (dev-gpt-agent): Specified verified harvest/git transaction and report readiness/statistical invariants.
