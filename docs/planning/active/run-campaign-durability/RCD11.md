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

# Implement Local/remote execution and generic production-coexistence gate

## Objective

Support exact-cohort local execution on Linux/Windows and preserve an extension path to remote Windows/non-Slurm hosts without coupling scientific identity to Slurm. Separately, protect timed Brutus measurements from llama-swap production, regardless of which GPUs production currently uses.

## Implementation status — 2026-09-27

Implemented:

- `tools/bigcherry/jobs/local.py`: durable detached LocalExecutor with reconnectable process metadata, terminal sentinel status, cancellation and stable-device allocation reporting.
- LocalExecutor receives the already-frozen stable device IDs; it does not rerun scientific GPU selection.
- local runner path acquires OS-held device locks keyed by those stable IDs.
- `tools/bigcherry/jobs/remote.py`: scheduler-neutral `RemoteExecutor`, restricted JSON request/response transport, exact execution correlation, status/control/allocation/events mapping.
- `tools/bigcherry/jobs/worker.py`: restricted worker endpoint backed by LocalExecutor.
- `registry.py` can configure `local` and `remote` targets; `config/jobs/executors.toml` includes a disabled Windows example.
- fake remote transport test proves the protocol without SSH/Windows.
- normalized `Allocation` now has optional `stable_gpu_ids`, preserved across remote protocol.

Not production-ready yet:

1. remote campaign submission still needs target-local staging/workspace/log path translation; controller-local absolute paths are not portable.
2. Windows HIP discovery/stable UUID/LUID -> current ordinal mapping is not implemented yet.
3. production llama-swap snapshot/claim parsing/window/watchdog is not implemented.
4. LocalExecutor currently records stable IDs from the request but target-local discovery must independently attest them before scientific spawn.
5. root measurement-window service/sudoers/boot recovery remains unimplemented.

Therefore remote Windows is an extension scaffold, not an enabled scientific executor.

## LocalExecutor contract

Local execution consumes an immutable series/attempt binding:

```text
series_id
platform_environment_hash
accepted_inventory_hash
hardware_cohort_hash
selected stable device IDs
```

Before production acceptance it must:

1. discover current local hardware;
2. verify every selected stable ID exists exactly once;
3. verify accepted inventory/cohort/topology;
4. map selected stable IDs to launch-local selectors;
5. acquire OS-held locks for exactly those IDs;
6. spawn detached process without shell;
7. persist reconnectable metadata/result.

No same-arch/same-model replacement is allowed for an existing series.

On Windows, launch ordinal is observation only. Stable identity preference is HIP UUID -> Windows LUID -> explicit weak epoch. A reboot/ordinal reorder with the same stable identity may remap; a different identity requires a new series.

## Remote execution architecture

Target architecture:

```text
central JobService
   |
RemoteExecutor
   |
restricted SSH JSON protocol
   |
BigCherry worker on target
   |
LocalExecutor + target-local inventory/workspace/cache
```

The central controller must send portable operation/attempt identity, not assume `/mnt/data/...` or a central Python path exists on Windows.

Required remote worker operations:

```text
capabilities
submit
correlate
status
cancel
control
allocation
events
logs/artifacts (to add before production remote use)
```

Every remote request is keyed by stable `execution_id`; network uncertainty is reconciled by `correlate`, never blind duplicate submission.

### Remote staging requirement

Before enabling remote scientific runs, add a target-local staging protocol:

- exact BigCherry commit SHA;
- llama.cpp/source identity;
- model/corpus content identity;
- immutable job/series/attempt document;
- target-local work/cache/log paths;
- target hardware inventory hash.

Worker fetches/materializes missing content by approved mechanism, verifies hashes, then constructs target-local `ExecutionRequest`. Central absolute cwd/log paths must never be sent as authoritative portable paths.

## Production coexistence

llama-swap may use any GPU/card combination. At each timed Brutus execution derive a fail-closed snapshot:

```python
@dataclass(frozen=True)
class ProductionSnapshot:
    config_hash: str
    potential_devices: frozenset[str] | AllDevices
    running_devices: frozenset[str]
    observed_devices: frozenset[str]
    ambiguities: tuple[str, ...]
```

Inputs:

- llama-swap config bytes/hash;
- declared stable GPU claims;
- `/running`;
- backend argv/env;
- AMD process/VRAM attribution.

Unknown/contradictory device ownership => potential `ALL`.

Gate after exact stable allocation is known:

```text
allocation intersects production potential -> exclusive production window
allocation disjoint                       -> idle attestation only after isolation gate
ambiguous production claim                -> exclusive window
```

Active disjoint inference remains disabled until scheduler-isolation evidence explicitly permits it.

## Exclusive window

Root-owned helper must accept only a declarative request containing execution/native ID, exact target stable IDs, inventory hash, production config hash and deadline. No caller-provided shell command.

State machine:

```text
CLOSED -> REQUESTED -> DRAINING -> ACTIVE -> CLOSING -> CLOSED
```

Entry blocks supported new production/ad-hoc starts before draining existing work. Exit is idempotent; timeout/crash/boot recovery must restore production or emit hard wake.

Agent privilege is limited to start/stop of the exact window unit. No generic root/systemctl/scontrol permission.

## Supported ad-hoc work

Add `bigcherry host-run --class build|bench -- ...` so supported direct work participates in host/device admission. Raw privileged shell work cannot be perfectly fenced and is an operator-policy violation; watchdogs should detect common contamination.

## Tests required

Offline:

- LocalExecutor survives controller restart and reconnects by execution ID;
- local process death releases OS device locks;
- missing/substituted stable ID blocks;
- stable UUID with changed ordinal remaps safely;
- fake remote submit/correlate/status/cancel/allocation/events;
- network retry does not duplicate stable execution ID;
- remote target path translation never uses controller absolute path;
- malformed worker action/payload fails closed;
- production claim parser UUID/arch-count/all/malformed;
- ambiguous claim -> exclusive;
- config/process/inventory drift contaminates timed run;
- window enter/exit/recovery state machine idempotent.

Hardware/service:

- Windows HIP UUID/LUID discovery and selector mapping;
- actual SSH worker restricted command setup;
- target-local model/source staging;
- llama-swap drain/stop/restart/health;
- production process attribution matches real GPU use;
- window timeout/boot recovery/sudo boundary;
- intentional production drift invalidates sample.

## Acceptance criteria

- LocalExecutor never substitutes outside frozen cohort;
- remote worker can run a full target-local mocked campaign without central filesystem assumptions before Windows hardware is enabled;
- network ambiguity cannot duplicate execution;
- production conflict policy is independent of static card placement;
- production cannot load a conflicting target during exclusive window;
- non-conflict execution is continuously contamination checked;
- root privilege surface is exact and minimal.

## Change log

- 2026-09-26: LocalExecutor/production coexistence design.
- 2026-09-27: LocalExecutor, remote protocol/worker, registry integration and fake remote tests implemented.
- 2026-09-27: explicitly retained remote target-local staging and Windows HIP discovery as blockers before scientific remote use.
