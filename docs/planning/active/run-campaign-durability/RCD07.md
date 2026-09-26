---
id: RCD07
order: 7
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:52:12.716988+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Activate RCD01 durable operations, split campaign stages and qualify scheduler isolation

## Description

After v1/v1.5 prove the service path, activate the durable operation/result subset of RCD01 and replace the prepare/execute pair with independently schedulable stages. Preserve RCD01's operation-spec hash vs execution hash distinction, immutable results, artifact byte/descriptor re-verification, and `running != success`. Slurm owns scheduling/resources; BigCherry owns operation identity and rehydration.

Stage concurrency policy remains conservative: timed measurement is host-exclusive (`host_activity:2`) until `scheduler-isolation-v1` proves a narrower policy. Builds may not overlap timed measurement by default.

## Steps

1. Implement durable `CampaignIntent`, `OperationSpec`, `OperationResult`, `ArtifactBinding` using RCD04 durability primitives.
2. Compile validation intent into deterministic operation graph.
3. Persist intent/spec before scheduling; persist `running.json` before launch and immutable `result.json` only after verified outputs.
4. Rehydrate successful operations only when operation spec, execution dependency hashes and output descriptors/bytes verify.
5. Split current campaign into prepare/materialize, build, correctness/activation, timed performance, reference ladder, optional production lane, harvest/report dependencies.
6. Map resource claims to `ResourceRequest`/Slurm licenses/GRES; no campaign-internal GPU `ResourceLock` in external-resource mode.
7. Add kill/tamper/dependency-change/resume tests using FakeExecutor.
8. Run `scheduler-isolation-v1`; only then consider allowing a build during timed measurement or active non-conflicting production inference.

## Detailed Solution & Technical Design

### Durable identities

```python
@dataclass(frozen=True)
class OperationSpec:
    operation_id: str
    kind: str
    command_semantics: Mapping[str, object]
    environment_semantics: Mapping[str, str]
    resources: ResourceRequest
    dependencies: tuple[str, ...]
    declared_outputs: tuple[str, ...]

@dataclass(frozen=True)
class ArtifactBinding:
    artifact_id: str
    descriptor_hash: str
    content_hash: str
    bytes: int

@dataclass(frozen=True)
class OperationResult:
    operation_spec_hash: str
    execution_hash: str
    outputs: tuple[ArtifactBinding, ...]
    state: Literal["succeeded", "failed", "interrupted"]
```

`operation_spec_hash` hashes every behavior-affecting operation field except dependency output values. `execution_hash` hashes operation spec + exact verified dependency bindings. Thus unchanged source build spec reruns if generated inputs change.

Directory:

```text
runs/<run>/attempts/NNN/stages/<operation-id>/
  operation.json
  running.json
  result.json
  stdout.log
  stderr.log
```

`running.json` is a crash marker, not success. A present `running.json` without valid `result.json` rehydrates as interrupted. Terminal result is immutable; retry creates an operation attempt record or new BigCherry attempt according to failure class.

### Graph

Initial deterministic graph:

```text
resolve/freeze
  -> materialize
  -> build
  -> correctness+activation
  -> timed-performance
  -> reference-ladder
  -> production-lane? 
  -> evidence-finalize
  -> harvest
  -> report
```

Where current producers require specialized preparation/builds, the producer supplies typed operation contributions; the scheduler does not know producer names.

It is acceptable to keep correctness+activation combined initially if separating them has no scheduling/recovery benefit. Timed performance must be a distinct resource operation before overlap policy can change.

### Resource policy

Default:

```text
build              build_slot:1 + host_activity:1
correctness        required GPU(s) + host_activity:1
timed-performance  required GPU(s) + host_activity:2
reference-ladder   required GPU(s) + host_activity:2
production-lane    required GPU(s) + host_activity:2
harvest/report     no GPU
```

The two-unit `host_activity` pool prevents any `host_activity:1` operation overlapping timed work.

When campaign graph uses Slurm/external scheduling, existing campaign `ResourceLock` GPU/build claims are disabled via `resource_policy="external"`; cache/artifact mutation locks may remain local.

### Rehydration

A stage may be reused only if:

- exact `operation_spec_hash`;
- exact `execution_hash`;
- all dependency outputs reverify descriptors/content;
- all stage outputs reverify descriptors/content;
- platform/hardware constraints encoded in spec still match;
- result state is succeeded.

Never restore a failed/interrupted stage as success. Never infer success from output path existence.

### `scheduler-isolation-v1`

Non-patch experiment, predeclared before observations:

- same binary/workload A/A;
- condition A host otherwise idle;
- condition B one representative HIP compile during measurement;
- for production coexistence, unloaded vs loaded-idle vs representative active inference on GPUs disjoint from measured allocation;
- >=32 randomized paired blocks per condition per qualified environment;
- paired log-ratio primary statistic;
- pass only if condition-effect 95% CI lies within [-0.10%, +0.10%], variance-ratio one-sided upper <=1.15, and no systematic clock/power/thermal shift.

Do not tune bounds after results. Until PASS, no build overlap with timed measurement and active production inference is not allowed during measurement.

## Code Samples & Guidance

RCD01 dormant document remains historical design source; update its state/notes when activation begins rather than creating a second durability specification.

Planned compiler:

```python
def compile_validation_operations(intent: CampaignIntent) -> CampaignGraph: ...
def operation_spec_hash(spec: OperationSpec) -> str: ...
def execution_hash(spec_hash: str, inputs: tuple[ArtifactBinding, ...]) -> str: ...
def rehydrate_result(root: Path, spec: OperationSpec, inputs: tuple[ArtifactBinding, ...]) -> OperationResult | None: ...
```

FakeExecutor must support injected interruption after `running.json` and before result publication.

## Files

Planned:

- `tools/bigcherry/jobs/operations.py`
- `tools/bigcherry/jobs/graph.py`
- `tools/bigcherry/jobs/rehydrate.py`
- campaign stage extraction under `tools/bigcherry/patch/campaign/`
- `tools/tests/jobs/test_operations.py`
- `tools/tests/jobs/test_rehydrate.py`
- `tools/tests/jobs/test_campaign_graph.py`
- non-patch isolation evidence location defined before running experiment.

## Validation

Offline falsification:

- kill after running marker -> interrupted, never success;
- mutate each OperationSpec field -> spec hash changes;
- same spec but dependency content changes -> execution hash changes and descendant reruns;
- byte-flip an output -> rehydration rejected;
- descriptor tamper -> rejected;
- dependency result missing -> descendant blocked;
- successful unrelated stage remains reusable after sibling failure;
- deterministic graph order independent of dict insertion order;
- executor restart + filesystem reload restores valid successes only;
- external-resource mode creates no duplicate GPU locks.

Hardware:

- stage-split scientific output parity against monolithic/v1.5 campaign;
- kill/restart one real stage and resume safely;
- isolation experiment for every environment where overlap will be enabled.

## Effort & Risk

Large. This is the first generalized durable stage engine; keep it campaign-specific and static. No distributed/cross-host lock reclamation is needed for Brutus.

## Standards

RCD01 operation/execution identity and interruption rules; RCD02 scheduler ownership boundary.

## Acceptance Criteria

- deterministic durable graph with immutable verified results;
- interruption/tamper/dependency falsifiers pass;
- Slurm scheduling and BigCherry operation durability do not duplicate resource ownership;
- stage-split output is scientifically equivalent to prior campaign;
- overlap remains disabled unless isolation evidence passes.

## Notes

Projected throughput improvements are secondary to validity. Do not enable concurrency merely because stage split exists.

## Change Log

- 2026-09-26T00:52:12.716988+00:00 (created-by): Created by agent
- 2026-09-26 (dev-gpt-agent): Connected stage split to RCD01 durable operation protocol and explicit falsification/isolation gates.
