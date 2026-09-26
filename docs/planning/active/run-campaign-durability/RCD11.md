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

The privileged production-window helper is a narrow root-owned systemd service; agents may only start/stop that exact service, never arbitrary `scontrol`/systemctl commands.

## Steps

1. Implement `LocalExecutor` against RCD04 Executor using injected process runner and RCD12 platform discovery.
2. Define/parse production GPU claim grammar; add llama-swap config + `/running` + backend/process/GPU observation providers.
3. Implement `ProductionSnapshot` reconciliation with fail-closed ambiguity.
4. Implement pre-dispatch gate after allocation: intersection -> exclusive window; disjoint -> production idle attestation.
5. Implement contamination watchdog over config hash, potential set, observed target usage, inventory hash and host noise.
6. Implement root measurement-window helper/unit, request schema, max-duration cleanup, boot recovery and restricted sudoers.
7. Add `bigcherry host-run --class build|bench -- ...` for supported ad-hoc direct work so it participates in activity/window locks.
8. Test snapshot/gate/window state machine with fakes; hardware/service tests remain explicit.

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

It:

- resolves `GpuRequirement` through RCD12;
- claims host-local `ResourceLock`s by stable device ID + activity class;
- launches process without shell;
- stores process/native handle metadata durably enough for status/cancel during process lifetime;
- maps allocated stable IDs to platform launch selectors at launch time;
- emits normalized Executor events/status.

On Windows, ordinals are launch-local observations only. RCD12 HIP discovery supplies UUID/LUID stable identity and current ordinal. Windows HIP gfx1100 remains a separate platform environment/series from Linux ROCm.

LocalExecutor is not a multi-host service and provides no queued background scheduler; if resources are unavailable it may return/raise `resource-busy` for caller retry rather than invent a queue daemon.

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

After Executor has an actual `Allocation`:

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
`systemctl start bigcherry-measure-window.service`
and
`systemctl stop bigcherry-measure-window.service`.

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

- claim UUID list/arch-count/all parsing;
- absent/malformed/contradictory claim -> ALL/exclusive;
- currently idle but potential intersects target -> exclusive;
- disjoint potential -> idle attestation;
- config hash changes during measurement -> contamination;
- observed production process appears on target -> contamination;
- target card BDF moves but stable ID mapping updates -> decision still identity-correct after inventory acceptance;
- FakeWindow enter/exit idempotent;
- timeout invokes cleanup in state-machine simulation;
- helper request rejects unknown device/stale inventory/stale production config;
- LocalExecutor cannot allocate nonmatching VRAM/arch/peer requirement;
- Windows ordinal changes while UUID/LUID stable do not change series hardware identity.

Hardware/service:

- llama-swap drain/stop/restart/health;
- RuntimeMaxSec auto-close;
- boot recovery from synthetic stale marker;
- sudo user cannot execute unrelated root command;
- production claim matches real backend device use;
- disjoint idle campaign survives watchdog;
- intentional production config drift invalidates sample.

## Effort & Risk

Large. Production protection is safety-critical operationally; ambiguity always chooses exclusive window. Root helper input surface must remain declarative and tightly validated.

## Standards

RCD12 stable IDs are the only device identity. RCD07 isolation evidence controls future relaxation; no hard-coded GPU classes/slots.

## Acceptance Criteria

- production conflict policy works for arbitrary current/future GPU placement;
- production cannot load onto conflicting target while exclusive service is active;
- non-conflict execution is continuously contamination-checked;
- root permission is limited to exact service lifecycle;
- stuck window automatically restores production or emits hard wake on failure;
- LocalExecutor/domain remain Slurm-independent.

## Notes

The design intentionally does not move llama-swap under Slurm in v1.

## Change Log

- 2026-09-26T01:20:22.304212+00:00 (created-by): Created by agent
- 2026-09-26T01:39:27.166365+00:00 (updated-by): Updated: section:notes
- 2026-09-26 (dev-gpt-agent): Fully specified generic production snapshot/gate/watchdog, LocalExecutor and privilege boundary.
