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

# Add campaign service seams: external evidence, frozen composition, typed preflights and prepare/execute

## Description

Refactor `bigcherry.patch.validation_campaign` only at explicit orchestration seams needed by the job service, preserving direct CLI behavior by default. Current campaign execution is monolithic, `_prepare_standard_campaign()` builds/materializes in-process, producer paths own their own build pairs, and `patch.evidence.write_record()` writes to repository-relative evidence unless given a root. This item adds explicit external evidence output, exact frozen validated composition, typed producer preflights/progress, and the v1.5 `prepare-only` / `execute-only` split without yet implementing the generalized RCD01 DAG.

The prepared manifest is cryptographically bound to the attempt commit, semantic campaign inputs, source/composition/build identities and artifacts. `execute-only` must fail closed before measurement if any bound fact changed.

## Steps

1. Add external evidence destination while preserving existing `write_record(record)` callers.
2. Add frozen validated-composition input to scaffold resolution and verify every ID+implementation digest before materialization.
3. Add `ProducerPreflightResult`/`run_preflight()` contract and move known MTP/full-vocab and tensor-topology/4096-ctx checks into producer-owned preflights.
4. Add structured campaign progress event sink; stderr human telemetry remains unchanged.
5. Add `--prepare-only --prepared-output PATH`.
6. Add `--execute-only --prepared PATH`; rehydrate/verify prepared artifacts, then run correctness/activation/performance/ladder/production/evidence.
7. Refactor producer runtime sufficiently to expose prepare vs execute, but do not create arbitrary independently resumable operations yet.
8. Build offline fake campaign fixtures that never invoke HIP binaries but exercise manifest binding/tamper/frozen-drift/preflight classifications.

## Detailed Solution & Technical Design

### External evidence

Current `patch.evidence.write_record(record, *, root=None)` already supports an alternate root. Extend compatibly:

```python
def write_record(
    record: Mapping[str, object],
    *,
    root: Path | None = None,
    output: Path | None = None,
) -> Path:
    ...
```

`root` and `output` are mutually exclusive. `output` writes exactly that path using the existing validation/atomic-write rules. Default remains package-local behavior.

CLI:

```text
--evidence-output /mnt/data/.../evidence/validation.json
```

Job-managed campaigns always pass it. Direct/manual invocations need not.

### Frozen validated composition

Introduce a serializable record:

```python
@dataclass(frozen=True)
class FrozenPatch:
    patch_id: str
    implementation_digest: str

def validated_enhancement_patches(
    *,
    patch_id: str,
    common_patches: tuple[str, ...],
    recipes: Path | None = None,
    frozen: tuple[FrozenPatch, ...] | None = None,
) -> tuple[str, ...]: ...
```

If `frozen is None`, current recipe lookup behavior is unchanged. If supplied:

- do not read current `validated-enhancements` membership to choose the control;
- load each named patch from the current attempt checkout;
- verify exact implementation digest;
- reject focal/common duplication and duplicate frozen IDs;
- return exact frozen order/IDs.

CLI:

```text
--frozen-validated-composition <canonical-json-file>
```

File schema binds `schema`, base revision, focal ID+digest, common IDs+digests, validated IDs+digests and composition hash. Drift exits 77 before build.

### Producer preflight

```python
@dataclass(frozen=True)
class ProducerPreflightResult:
    status: Literal["ok", "retryable", "invalid"]
    code: str
    detail: str
    evidence: Mapping[str, object] = field(default_factory=dict)

class ValidationProducer(Protocol):
    def run_preflight(
        self, ctx: ProducerContext, *, phase: Literal["prebuild", "premeasure"]
    ) -> ProducerPreflightResult: ...
```

Rules:

- `retryable`: harness/parser/temporary host/attestation failure; no scientific result.
- `invalid`: supplied model/corpus/topology/producer inputs cannot satisfy contract; terminal invalid.
- `ok`: persisted as preflight evidence.

Examples:
- MTP expected full-vocab rows: parser/source distinction must be explicit, not inferred after a missing result.
- `-sm tensor`: exact allocated peer-capable device identities, required layer/offload/topology attestation, producer-declared context cap (4096 where contract requires it).
- generic scheduler code never contains producer names or special cases.

### Progress events

```python
class CampaignEventSink(Protocol):
    def emit(self, kind: str, data: Mapping[str, object]) -> None: ...
```

Emit stable events at phase/lane/round transitions and preflight/build/benchmark completion. `NullCampaignEventSink` is default. The event sink cannot affect result semantics.

### PreparedCampaignV1

The prepared manifest is a value object, not a resume database:

```python
@dataclass(frozen=True)
class PreparedCampaign:
    schema: Literal["bigcherry.prepared-campaign.v1"]
    attempt_commit: str
    semantic_request_hash: str
    contract_hash: str
    platform_environment_hash: str
    inventory_hash: str
    base_revision: str
    focal_patch: FrozenPatch
    common_patches: tuple[FrozenPatch, ...]
    validated_patches: tuple[FrozenPatch, ...]
    control_composition_hash: str
    sources: tuple[PreparedArtifact, ...]
    builds: tuple[PreparedBuild, ...]
    producer: PreparedProducer | None
```

Each artifact includes path only as locator plus size/content hash/descriptor identity. Each build includes existing `CompletedBuildEvidence.campaign_identity()` and runtime bundle hash. `semantic_request_hash` excludes output/work paths but includes every behavior-affecting argument.

`prepare-only` may materialize/build/generate everything required by current campaign but must not:
- run scientific correctness/performance/ladder/production lanes;
- write final validation evidence;
- evaluate promotion verdict.

`execute-only`:
1. load manifest;
2. verify attempt commit and semantic request;
3. verify contract/composition/platform/inventory identity;
4. byte/descriptor-verify every required prepared output;
5. run premeasure producer preflight;
6. run existing scientific lanes unchanged;
7. write evidence to external sink.

Any verification mismatch is not silently rebuilt in `execute-only`; return 76/77 according to harness-vs-scientific identity drift. A new prepare/attempt decides reuse.

### Current-code seam

Standard campaign preparation should be extracted from current `_prepare_standard_campaign(args, st)` into:

```python
def prepare_standard_campaign(...) -> PreparedStandardCampaign: ...
def execute_standard_campaign(..., prepared: PreparedStandardCampaign) -> ValidationResult: ...
```

The existing `run()` path composes both so direct behavior stays equivalent.

Producer runtime receives analogous `prepare()`/`run_prepared()` methods around its existing materialize/build pair. Do not serialize live Python objects; serialize only explicit identities/artifact descriptors and rehydrate through verification functions.

## Code Samples & Guidance

Add parser mode validation:

```text
default                    -> legacy monolithic behavior
--prepare-only             -> requires --prepared-output; forbids --execute-only
--execute-only --prepared  -> forbids build-changing implicit fallback
```

Tests monkeypatch actual build/benchmark launch seams but use real manifest hashing, atomic persistence and verifier code.

## Files

Modify:

- `tools/bigcherry/patch/validation_campaign.py`
- `tools/bigcherry/patch/evidence.py`
- `tools/bigcherry/patch/campaign/scaffold.py`
- `tools/bigcherry/patch/campaign/producer.py`

Add:

- `tools/bigcherry/patch/campaign/prepared.py`
- `tools/bigcherry/patch/campaign/events.py`
- `tools/tests/patch/test_validation_campaign_service_seams.py`
- `tools/tests/patch/test_prepared_campaign.py`
- producer-specific preflight tests under `tools/tests/patch/`.

## Validation

Offline required:

- default monolithic invocation argv/behavior remains compatible;
- evidence `output=` writes only specified external path; default still writes canonical package path;
- frozen set is unaffected by later recipe promotion;
- changed frozen implementation digest fails before any build mock is called;
- prepared manifest canonical hash deterministic;
- modify any behavior-affecting field -> execute refuses;
- modify build binary/runtime bundle/source artifact bytes -> execute refuses;
- path move with same verified descriptor/content may rehydrate if locator policy allows;
- execute-only never invokes build mock;
- prepare-only never invokes benchmark/performance mock;
- MTP missing rows parser-bug fixture -> retryable; genuinely invalid source fixture -> invalid;
- tensor topology mismatch -> invalid/preflight, not late scientific FAIL;
- event sink receives phase order but Null sink yields identical result;
- producer and standard fake paths both round-trip prepare->execute.

Hardware gates:

- one representative standard producer and one dual-GPU producer prepare/execute pair produce evidence identical in scientific meaning to monolithic execution;
- content-addressed build directories are actually reused;
- prepared binary attestation remains valid after process boundary.

## Effort & Risk

Large. Biggest risk is accidentally changing scientific campaign behavior while splitting orchestration. Keep old monolithic integration path as composition of the same extracted functions and compare records in tests.

## Standards

Use existing `CompletedBuildEvidence`, source identity, evidence validation and atomic writers; do not invent weaker duplicate build/source identity.

## Acceptance Criteria

- job service can write evidence outside runner checkout;
- frozen composition is exact and enforced before build;
- producer semantic preflights are typed;
- prepare/execute works for standard and producer campaigns without rebuilding in execute;
- fake/tamper tests pass;
- direct monolithic CLI remains supported.

## Notes

This is v1.5. Full stage durability/resume belongs to RCD07/RCD01 activation; do not grow `PreparedCampaign` into a generic workflow engine.

## Change Log

- 2026-09-26T00:52:08.757524+00:00 (created-by): Created by agent
- 2026-09-26 (dev-gpt-agent): Grounded service seams in current `validation_campaign`, `scaffold.validated_enhancement_patches` and `evidence.write_record`.
