---
id: RCD12
order: 12
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T01:39:24.166357+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Implement dynamic GPU discovery, capability resolution, generated GRES and hardware-cohort identity

## Description

Make discovered hardware state authoritative. Brutus cards may be added, removed, replaced or moved; Windows device ordinals may change. Discover stable device identity + current locators/topology at service start and periodically, persist observed/accepted inventory, drain/wake on material drift, generate Slurm architecture-typed GRES, resolve `GpuRequirement` capabilities, and bind scientific series to a hardware cohort rather than a slot/index.

`environment.local.toml` remains host policy/toolchain/model/optional alias configuration, not GPU inventory truth.

## Steps

1. Define provider-neutral hardware models/canonical hashes/topology.
2. Implement Linux AMD discovery with layered stable-ID sources and sysfs locator reconciliation; capture real command/API fixtures before hard-coding parsers.
3. Implement Windows HIP discovery (UUID/LUID + current ordinal) for LocalExecutor.
4. Persist `observed.json`; compare with `accepted.json`; classify drift.
5. Implement explicit operator acceptance of a discovered inventory/hardware epoch.
6. Render architecture-typed `gres.conf` and node `Gres=`; RCD03 validates via `slurmd -G`.
7. Implement `GpuRequirement` feasibility and post-allocation selection, including peer/subset/exact-device over-allocation.
8. Compute platform environment and hardware cohort identities for series.
9. Add boot discovery service and runtime drift monitor that drains Brutus before new work when accepted inventory no longer matches.
10. Add fixture-based tests for add/remove/replacement/move/weak identity/Windows ordinal changes.

## Detailed Solution & Technical Design

### Models

```python
@dataclass(frozen=True)
class DeviceRecord:
    device_id: str
    identity_source: Literal[
        "amd_uuid", "rsmi_unique_id", "serial", "hip_uuid", "windows_luid", "weak"
    ]
    architecture: str
    model: str
    vram_bytes: int
    pci_bdf: str | None
    render_node: str | None
    numa_node: int | None
    driver_version: str
    launch_ordinal: int | None = None

@dataclass(frozen=True)
class HardwareInventory:
    host: str
    devices: tuple[DeviceRecord, ...]
    peer_access: tuple[tuple[str, str], ...]
    topology_fingerprint: str
    observed_at: str
    hardware_epoch: str | None
```

Separate hashes:

```python
def inventory_material_hash(inv: HardwareInventory) -> str:
    # includes operational locators; detects BDF/render/NUMA movement
    ...

def hardware_cohort_hash(inv: HardwareInventory, selected_ids: tuple[str, ...]) -> str:
    # stable IDs + arch/model/VRAM + relevant topology; excludes ephemeral ordinal/BDF
    ...

def platform_environment_hash(env: ExecutionEnvironment) -> str:
    # OS/build/kernel/runtime/HIP/compiler/driver
    ...
```

BDF/render node/index/ordinal are locator state, never `device_id`.

### Linux identity discovery

Preference per physical GPU:

1. AMD-SMI UUID if available and stable;
2. RSMI unique ID;
3. hardware serial if sufficiently unique/stable;
4. `weak` synthetic identity scoped to accepted `hardware_epoch`.

Do not assume any preferred field exists until probed on Brutus. Capture raw sanitized outputs from the installed AMD-SMI/ROCm tools as test fixtures, then implement exact parser/version handling. Reconcile identity with render node/BDF using sysfs (`/sys/class/drm/renderD*/device`) rather than trusting ordinal ordering.

Architecture/model/VRAM/driver come from AMD runtime tooling/sysfs where authoritative. Topology captures at least peer-access eligibility and NUMA/PCIe facts needed by current multi-GPU contracts.

Weak identity rule: automatic continuation across reboot/hardware remap is forbidden unless accepted snapshot proves the same hardware epoch; replacement/ambiguity forces operator acceptance/new cohort.

### Windows discovery

Use a small HIP probe or binding exposing:

- `hipGetDeviceCount`;
- `hipGetDeviceProperties` architecture/model/VRAM/UUID/PCI when available;
- Windows LUID where available;
- driver/runtime versions;
- `hipDeviceCanAccessPeer` matrix.

Stable preference: HIP UUID -> Windows LUID -> weak epoch. Ordinal is recorded only as current launch selector. LocalExecutor resolves UUID/LUID to current ordinal immediately before launch.

### Inventory state/drift

Paths (configurable for tests):

```text
/var/lib/bigcherry/hardware/observed.json
/var/lib/bigcherry/hardware/accepted.json
```

Boot:

1. discover current;
2. atomically write observed;
3. compare accepted;
4. if no accepted state or material mismatch: keep/drain Slurm node, emit wake, forbid new job submission requiring hardware;
5. operator runs `bigcherry hardware accept --inventory-hash ...` after inspection/tests;
6. generate/apply Slurm GRES and resume only after `jobs doctor`/acceptance.

Runtime monitor re-discovers on udev signal and low-rate timer. Any add/remove/replacement/BDF/render/topology material drift drains the node before new managed work. Existing timed measurement is contamination/invalid if inventory changes while running.

Drift classes:

```text
same                    no action
locator-only            drain/reconcile; scientific cohort may remain same if topology unchanged
topology-change         drain; new hardware cohort
device-add/remove       drain; inventory acceptance required
device-replacement      drain; new hardware cohort
identity-ambiguous      drain; new hardware epoch/operator decision
driver/environment      platform environment change; new series
```

A physical card moved slot with unchanged stable ID but changed topology starts a new cohort. Pure locator change with identical relevant topology does not by itself change cohort, but still requires operational drain/reconcile because Slurm `File=` changed.

### GRES generation

Deterministic:

```python
def render_gres_conf(inventory: HardwareInventory) -> str: ...
def render_node_gres(inventory: HardwareInventory) -> str: ...
```

Example:

```text
Name=gpu Type=gfx1100 File=/dev/dri/renderD128 Flags=amd_gpu_env
Name=gpu Type=gfx1100 File=/dev/dri/renderD129 Flags=amd_gpu_env
Name=gpu Type=gfx1201 File=/dev/dri/renderD130 Flags=amd_gpu_env
```

Node summary groups counts by architecture. Never generate `gfx1100_0`, device ID, serial or BDF as GRES Type.

Do not use `AutoDetect=rsmi` as BigCherry identity authority; it may be a diagnostic cross-check only.

### Capability resolution

```python
@dataclass(frozen=True)
class GpuRequirement:
    architecture: str
    count: int = 1
    min_vram_bytes: int = 0
    homogeneous_model: bool = True
    require_peer_access: bool = False
    exact_device_ids: tuple[str, ...] = ()

@dataclass(frozen=True)
class ResolvedGpuRequest:
    slurm_architecture: str
    slurm_count: int
    reserve_all_of_arch: bool
    eligible_device_ids: tuple[str, ...]
    selected_exact_ids: tuple[str, ...]
```

Pre-submit:
- fail if accepted inventory cannot satisfy requirement;
- normal homogeneous capability request -> `--gres=gpu:<arch>:<count>`;
- if only a subset satisfies VRAM/model/peer/exact requirement, request **all accepted devices of that architecture**, ensuring Slurm excludes others, then select required stable IDs inside allocation.

Default Slurm mode preserves scheduler-generated `ROCR_VISIBLE_DEVICES`. Subset-overallocation is the explicit exception: because every device being narrowed away is already allocated exclusively to the same job, BigCherry may narrow `ROCR_VISIBLE_DEVICES` to selected UUID(s); record original allocation and narrowed selection. Never broaden beyond allocation.

At process start re-attest stable IDs/capabilities/peer matrix against accepted inventory hash. Stale mismatch aborts before scientific work.

### Series identity

```text
platform_environment_hash =
  OS family/version
  kernel or Windows build
  runtime family/version
  HIP runtime
  compiler identity/version
  AMD driver

hardware_cohort_hash =
  sorted selected stable device IDs
  architecture/model/VRAM
  relevant peer/topology fingerprint
```

Series key includes patch/contract + architecture + platform environment + hardware cohort. Therefore:
- same-model replacement card => new series;
- Windows HIP vs Linux ROCm => new series;
- same card + changed relevant topology => new series;
- session may not aggregate across hardware cohorts.

### environment.local role

Allowed:

```toml
[host]
name = "brutus"

[toolchains]
# logical aliases -> paths

[models]
# logical refs/root

[gpu_policy]
allowed_architectures = ["gfx1100", "gfx1201", "gfx1030"]

[device_aliases]
# optional stable-device-id -> human label
```

A configured alias/policy cannot override discovered architecture/VRAM/identity.

## Code Samples & Guidance

Discovery providers take a `CommandRunner`/filesystem root abstraction so raw fixtures can test parsing without installed ROCm.

```python
class HardwareProvider(Protocol):
    def discover(self) -> HardwareInventory: ...

def classify_drift(accepted: HardwareInventory, observed: HardwareInventory) -> DriftReport: ...
def resolve_gpu_requirement(req: GpuRequirement, inventory: HardwareInventory) -> ResolvedGpuRequest: ...
```

Never select candidates by current ordinal except as final platform launch mapping.

## Files

Planned:

- `tools/bigcherry/hardware/__init__.py`
- `tools/bigcherry/hardware/model.py`
- `tools/bigcherry/hardware/inventory.py`
- `tools/bigcherry/hardware/topology.py`
- `tools/bigcherry/hardware/linux_amd.py`
- `tools/bigcherry/hardware/windows_hip.py`
- `tools/bigcherry/hardware/slurm.py`
- `tools/bigcherry/cli/hardware.py`
- `config/systemd/bigcherry-hardware-discovery.service`
- `config/systemd/bigcherry-hardware-watch.service/.timer`
- `tools/tests/hardware/fixtures/`
- `tools/tests/hardware/test_inventory.py`
- `tools/tests/hardware/test_drift.py`
- `tools/tests/hardware/test_gpu_requirement.py`
- `tools/tests/hardware/test_slurm_render.py`
- `tools/tests/hardware/test_windows_hip.py`

## Validation

Already falsified in lab mock:

- capability VRAM/arch resolution;
- peer-pair/exact-device selection;
- subset constraints require all-of-arch reservation;
- Windows/Linux environment hash separation;
- replacement changes cohort;
- locator move changes material inventory; topology change changes cohort.

Required offline production tests:

- fixture device reorder produces identical stable IDs/cohort;
- add/remove/replacement classifications;
- same stable card moved BDF/render -> locator drift; generated GRES changes;
- topology change -> cohort changes;
- weak identity requires explicit epoch;
- mixed same-arch models/VRAM triggers over-allocation;
- exact unknown ID rejected;
- peer requirement with no valid pair rejected;
- generated GRES deterministic/grouped by architecture/no slot type;
- Slurm subset narrowing cannot name an unallocated ID;
- Windows ordinal reorder preserves UUID/LUID identity;
- environment.local alias cannot falsify observed hardware.

Hardware gates:

- determine actual UUID/unique-id/serial availability/stability for installed cards and both relevant tooling versions;
- raw fixture capture/redaction;
- `slurmd -G` validates generated output;
- peer matrix matches real `-sm tensor` preflight;
- hot-drift simulation safely drains/reconciles without physically removing a live card;
- reboot preserves stable IDs where provider claims stability.

## Effort & Risk

Large. Stable hardware identity availability is the key empirical uncertainty; weak epoch fallback is designed so lack of UUID cannot silently mix cards/series.

## Standards

RCD02 capability/identity boundary; RCD03 consumes generated GRES; RCD11 consumes stable IDs for production claims.

## Acceptance Criteria

- runtime hardware truth is discovered, persisted and accepted explicitly;
- material drift blocks/drains new managed work;
- GRES uses architecture types only;
- jobs express capability requirements, not physical indices;
- hardware cohort prevents card replacement/topology mixing;
- Windows and Linux discovery map to the same provider-neutral model;
- all offline fixture/drift/resolver tests pass before hardware acceptance.

## Notes

The lab planning simulator is intentionally a simplified model. Production implementation must use real captured tool/sysfs/HIP fixtures before declaring parser support.

## Change Log

- 2026-09-26T01:39:24.166357+00:00 (created-by): Created by agent
- 2026-09-26 (dev-gpt-agent): Fully specified provider model, drift fencing, generated GRES, capability resolver and cohort identity.
