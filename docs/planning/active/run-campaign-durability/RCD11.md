---
id: RCD11
order: 11
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T01:20:22.304212+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Implement LocalExecutor and generic production-coexistence gate

## Description

Implement the non-Slurm `LocalExecutor` for Windows/Linux direct/testing/emergency use and the Brutus production coexistence layer for llama-swap. Production may use any GPU or combination and configuration changes over time; therefore no card class/slot is hard-coded. At each timed execution BigCherry derives a fail-closed `ProductionSnapshot`, compares potential production devices with the actual stable-ID allocation, and chooses exclusive window vs idle-attestation. A runtime watchdog invalidates a sample on production/inventory drift.

`LocalExecutor` is an execution adapter, not a scientific allocator: RCD04/RCD12 have already frozen an exact stable-device cohort in the series. Local execution must map and attest those IDs on the current host and **must never silently select a replacement GPU/cohort**.

The privileged production-window helper is a narrow root-owned systemd service; agents may only start/stop that exact service, never arbitrary `scontrol`/systemctl commands.

## Steps

1. Implement `LocalExecutor` against RCD04 `Executor`; consume immutable `SeriesGpuBinding`/attempt-bound stable IDs from RCD12 and map them to current platform launch selectors. Reject missing/drifted binding; never rerun scientific capability selection inside the executor.
2. Define/parse production GPU claim grammar; add llama-swap config + `/running` + backend/process/GPU observation providers.
3. Implement `ProductionSnapshot` reconciliation with fail-closed ambiguity.
4. Implement pre-dispatch gate after allocation: intersection -> exclusive window; disjoint -> production idle attestation.
5. Implement contamination watchdog over config hash, potential set, observed target usage, inventory hash and host noise.
6. Implement root measurement-window helper/unit, request schema, max-duration cleanup, boot recovery and restricted sudoers.
7. Add `bigcherry host-run --class build|bench -- ...` for supported ad-hoc direct work so it participates in activity/window locks.
8. Test binding/attestation, snapshot/gate/window state machines with fakes; hardware/service tests remain explicit.

## Detailed Solution & Technical Design

### LocalExecutor

```python
class LocalExecutor(Executor):
    def __init__(
        self,
        *,
        inventory: HardwareInventory,
        process_runner: LocalProcessRunner,
        resource_root: Path,
    ): ...
```

Input `ExecutionRequest` carries the scheduler/execution projection of the already-frozen series binding. Before spawn the adapter receives/loads the immutable attempt identity containing:

```text
series_id
hardware_cohort_hash
accepted_inventory_hash
bound stable device IDs
platform_environment_hash
```

LocalExecutor then:

1. loads current platform inventory;
2. requires every bound stable device ID to exist exactly once;
3. verifies arch/model/VRAM and relevant peer/topology fingerprint still satisfy the frozen cohort;
4. rejects accepted-inventory or topology drift that invalidates the series binding;
5. acquires host-local `ResourceLock`s for **those exact stable IDs** plus activity class;
6. maps those IDs to launch-local selector values immediately before spawn;
7. launches without shell and emits normalized status/events.

It does **not** call the RCD12 cohort selector to choose another card. If the frozen device is unavailable, the attempt is blocked/invalid for that series; operator creates/resumes a compatible series according to RCD12 policy.

On Windows, ordinal is a launch-local observation only. RCD12 HIP discovery supplies UUID/LUID stable identity and current ordinal. A Windows ordinal change with stable UUID/LUID is remapped safely; a different stable ID is never substituted. Windows HIP gfx1100 remains a separate platform environment/series from Linux ROCm.

LocalExecutor is not a multi-host background scheduler. If exact bound resources are busy it returns/raises `resource-busy` for caller/service policy rather than inventing another queue daemon.

### Production claim grammar

Per llama-swap model convention:

```text
BIGCHERRY_GPU_CLAIM=uuid:<id>[,uuid:<id>...]
BIGCHERRY_GPU_CLAIM=arch:gfx1100,count=2
BIGCHERRY_GPU_CLAIM=all
```

Parser resolves against accepted RCD12 inventory. Unknown device, impossible count, malformed selector, absent claim for a GPU-using backend, mismatch with command/env/process observation => potential devices `ALL`.

```python
@dataclass(frozen=True)
class ProductionSnapshot:
    config_hash: str
    potential_devices: frozenset[str] | AllDevices
    running_devices: frozenset[str]
    observed_devices: frozenset[str]
    ambiguities: tuple[str, ...]
```

Sources, highest-level intent plus runtime verification:

1. llama-swap config bytes/hash and model `cmd`/`env`;
2. `BIGCHERRY_GPU_CLAIM`;
3. `/running`;
4. backend process environment/argv where observable;
5. AMD process/VRAM attribution mapped to stable IDs.

`potential_devices` represents every device production may load onto without config change, not merely currently-running devices.

### Gate

After Executor has an actual `Allocation` containing stable IDs:

```python
def decide_production_gate(
    allocation: Allocation,
    snapshot: ProductionSnapshot,
) -> GateDecision:
    ...
```

- if snapshot ambiguous/ALL -> exclusive window;
- if allocated IDs intersect potential -> exclusive window;
- otherwise -> idle attestation until isolation qualification allows stronger coexistence.

Idle attestation:

- `/running` captured;
- all active backend `/slots` report `is_processing=false`;
- `/metrics` processing/deferred counts zero when supported;
- production observed idle continuously >=10s;
- target GPUs and host noise pass predeclared guard.

This policy is conservative because production on a disjoint GPU can still perturb CPU/PCIe/power.

### Contamination watchdog

Every ~2s during timed measurement verify:

- llama-swap config hash unchanged;
- fresh potential set remains disjoint for non-window execution;
- no production process/VRAM observation appears on target stable IDs;
- accepted hardware inventory hash unchanged;
- allocation still resolves to the frozen series stable IDs;
- production idle condition remains valid until `scheduler-isolation-v1` permits active disjoint inference.

Violation:

- terminate timed measurement/process group;
- mark result invalid environment contamination;
- do not persist scientific round as valid;
- emit wake;
- retry only under normal harness/environment retry policy, never because measured effect was poor.

### Exclusive window

Request file:

```json
{
  "schema": 1,
  "execution_id": "...",
  "native_id": "...",
  "target_device_ids": ["..."],
  "inventory_hash": "...",
  "production_config_hash": "...",
  "deadline": "..."
}
```

Root helper validates schema, file ownership/mode, inventory/config hash, execution is active and target IDs exist. It never executes caller-provided command text.

State:

```text
enter:
  exclusive flock
  validate request
  block supported host-run starts
  drain llama-swap/backend requests
  stop/unload production
  verify no production process on any target + target quiet
  write active state
  remain foreground

exit (idempotent):
  prevent new managed measurement ownership
  terminate/wait window-owned execution only according to policy
  clear active marker
  start llama-swap
  verify health
```

Systemd:

- `Type=simple`;
- `RuntimeMaxSec=4h`;
- `ExecStop` and `ExecStopPost` both call idempotent cleanup;
- boot recovery before llama-swap ensures stale state closes and production health restored.

Sudoers grants `bigcherry` exactly:

```text
systemctl start bigcherry-measure-window.service
systemctl stop bigcherry-measure-window.service
```

No generic sudo `scontrol`, `systemctl`, shell or helper arguments.

### Ad-hoc work

`bigcherry host-run --class build|bench -- command...` acquires shared host activity/device locks and refuses timed-perturbing work while a window is active. Unsupported raw shell/admin work cannot be made impossible; it is outside the supported Brutus operating contract and watchdog should detect common contamination.

## Code Samples & Guidance

Provider protocols are injectable:

```python
class ProductionInspector(Protocol):
    def snapshot(self, inventory: HardwareInventory) -> ProductionSnapshot: ...

class WindowController(Protocol):
    def enter(self, request: WindowRequest) -> None: ...
    def exit(self) -> None: ...
```

Binding helper must be attestation-only:

```python
def map_bound_devices(
    binding: SeriesGpuBinding,
    current: HardwareInventory,
) -> tuple[AllocatedDevice, ...]:
    """Return current locators for exactly binding.device_ids or fail."""
    ...
```

Offline tests use fake HTTP/process/GPU observations, not real llama-swap.

## Files

Planned:

- `tools/bigcherry/jobs/local.py`
- `tools/bigcherry/jobs/production.py`
- `tools/bigcherry/jobs/window.py`
- `tools/bigcherry/cli/host_run.py`
- `config/systemd/bigcherry-measure-window.service`
- `config/systemd/bigcherry-measure-window-recover.service`
- `config/sudoers/bigcherry-measure-window`
- `tools/tests/jobs/test_local_executor.py`
- `tools/tests/jobs/test_production.py`
- `tools/tests/jobs/test_window.py`

## Validation

Offline:

- frozen stable-ID cohort maps to current locator/ordinal without identity change;
- missing frozen device blocks; no fallback/substitution;
- different same-arch/same-model card is rejected for existing series;
- accepted inventory/topology mismatch blocks until RCD12 policy resolves it;
- Windows ordinal changes while UUID/LUID stable are safely remapped;
- claim UUID list/arch-count/all parsing;
- absent/malformed/contradictory claim -> ALL/exclusive;
- currently idle but potential intersects target -> exclusive;
- disjoint potential -> idle attestation;
- config hash changes during measurement -> contamination;
- observed production process appears on target -> contamination;
- target card BDF moves but stable ID mapping updates -> decision remains identity-correct only after accepted inventory/topology validation;
- FakeWindow enter/exit idempotent;
- timeout invokes cleanup in state-machine simulation;
- helper request rejects unknown device/stale inventory/stale production config.

Hardware/service:

- llama-swap drain/stop/restart/health;
- RuntimeMaxSec auto-close;
- boot recovery from synthetic stale marker;
- sudo user cannot execute unrelated root command;
- production claim matches real backend device use;
- disjoint idle campaign survives watchdog;
- intentional production config drift invalidates sample;
- LocalExecutor Windows HIP selector maps the exact frozen stable ID.

## Effort & Risk

Large. Production protection is safety-critical operationally; ambiguity always chooses exclusive window. Root helper input surface must remain declarative and tightly validated. Hardware substitution is fail-closed because the scientific effect policy admits small wins where card-to-card variance matters.

## Standards

RCD12 stable IDs and frozen `SeriesGpuBinding` are the only device identity. RCD07 isolation evidence controls future relaxation; no hard-coded GPU classes/slots.

## Acceptance Criteria

- production conflict policy works for arbitrary current/future GPU placement;
- production cannot load onto conflicting target while exclusive service is active;
- non-conflict execution is continuously contamination-checked;
- root permission is limited to exact service lifecycle;
- stuck window automatically restores production or emits hard wake on failure;
- LocalExecutor/domain remain Slurm-independent;
- LocalExecutor never reselects or substitutes a GPU outside the series-bound stable cohort.

## Notes

The design intentionally does not move llama-swap under Slurm in v1.

## Change Log

- 2026-09-26T01:20:22.304212+00:00 (created-by): Created by agent
- 2026-09-26T01:39:27.166365+00:00 (updated-by): Updated: section:notes
- 2026-09-26 (dev-gpt-agent): Fully specified generic production snapshot/gate/watchdog, LocalExecutor and privilege boundary.
- 2026-09-26 (dev-gpt-agent): Corrected LocalExecutor boundary: it attests/maps the frozen series cohort and never reruns scientific GPU selection.
