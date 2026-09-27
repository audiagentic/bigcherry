---
id: RCD06
order: 6
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:52:08.757524+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Campaign service seams and prepare/execute evolution

## Status — 2026-09-28

The original M1 proposal predated the implemented JobService isolation model. Two proposed seams are now **superseded for monolithic v1**, while two remain useful future campaign/stage interfaces.

### Superseded v1 mechanisms

**External evidence destination:** managed attempts execute in an exact detached BigCherry worktree. Campaign evidence written repository-relative therefore lands only in that isolated attempt checkout. `jobs harvest` later identifies newly added records, re-verifies record/patch/validation/contract identity, refuses dirty/staged canonical destinations, and merges/stages/commits only the exact canonical evidence path. A second `--evidence-output` path is no longer required for safe v1 isolation.

**Separate frozen-composition CLI:** series creation already freezes focal/common/current validated-enhancement IDs/digests plus recipe/contract content identity; attempt start runs from the exact pinned commit and re-resolves that scientific identity. Any composition or digest mismatch blocks before managed execution. A parallel `--frozen-validated-composition` format would duplicate the series authority.

Do not add either mechanism unless a future execution mode bypasses the pinned-worktree + series-identity + verified-harvest boundary.

### Still useful / not complete

1. **Typed producer preflight protocol.** Existing producer/runtime code has individual fail-closed preflights (for example GPU-count/visibility checks), but no single generic `ProducerPreflightResult{ok,retryable,invalid}` phase contract yet. Add it when producer stages are extracted so parser/host failures cannot be confused with invalid scientific inputs.
2. **Structured campaign event sink.** JobService has durable executor/domain events, but `validation_campaign` does not yet expose stable phase/lane/round callbacks. Add this for richer stage progress/UI, never as result authority.
3. **Prepare/execute split.** `--prepare-only`/`--execute-only` is not implemented. RCD07 now provides operation identity/rehydration primitives; if downtime reduction is still needed, build stage extraction directly on those primitives rather than introducing a parallel `PreparedCampaign` resume database.

## Revised implementation direction

For future stage split:

```text
series scientific freeze
  -> RCD07 OperationSpec prepare/build
  -> producer prebuild typed preflight
  -> correctness/activation
  -> producer premeasure typed preflight
  -> timed performance / reference ladder / optional production
  -> evidence-finalize
  -> verified harvest/report
```

Required preflight result:

```python
@dataclass(frozen=True)
class ProducerPreflightResult:
    status: Literal["ok", "retryable", "invalid"]
    code: str
    detail: str
    evidence: Mapping[str, object]
```

Examples remain producer-owned: MTP full-vocabulary row/source-vs-parser checks; `-sm tensor` peer/layer/topology attestation and contract context caps such as 4096. Generic scheduler/domain code must not special-case producer names.

## Cutover consequence

RCD06 is **not a monolithic-v1 cutover blocker** after the pinned-worktree/scientific-freeze/verified-harvest implementation. Typed producer preflights/events and stage splitting remain post-v1 durability/observability work and must preserve direct CLI compatibility.

## Change log

- 2026-09-26: original external-evidence/frozen-composition/prepare-execute design.
- 2026-09-28: reconciled with implemented JobService isolation; retired duplicate M1 authorities and moved typed preflight/events/stage split to RCD07-era evolution.
