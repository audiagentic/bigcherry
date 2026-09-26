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

Make discovered hardware state authoritative. Brutus cards may be added, removed, replaced or moved; Windows device ordinals may change. Discover stable device identity + current locators/topology at service start and periodically, persist observed/accepted inventory, drain/wake on material drift, generate Slurm architecture-typed GRES, resolve `GpuRequirement` capabilities, and bind each scientific series to one deterministic hardware cohort before its first attempt. `environment.local.toml` remains host policy/toolchain/model/optional alias configuration, not GPU inventory truth.

A series may never discover its physical card only after Slurm happens to allocate one: card-to-card variance is part of scientific identity. Series creation therefore resolves a deterministic stable-device cohort from the accepted inventory; every session in that series targets the same cohort. If Slurm cannot name that subset directly, BigCherry safely over-allocates the architecture and narrows inside the allocation.

## Steps

1. Define provider-neutral hardware models/canonical hashes/topology.
2. Implement Linux AMD discovery with layered stable-ID sources and sysfs locator reconciliation; capture real command/API fixtures before hard-coding parsers.
3. Implement Windows HIP discovery (UUID/LUID + current ordinal) for LocalExecutor.
4. Persist `observed.json`; compare with `accepted.json`; classify drift.
5. Implement explicit operator acceptance of a discovered inventory/hardware epoch.
6. Render architecture-typed `gres.conf` and node `Gres=`; RCD03 validates via `slurmd -G`.
7. Implement deterministic capability binding at series creation and post-allocation verification/selection, including peer/subset/exact-device over-allocation.
8. Compute platform environment and hardware cohort identities before `series_id` is finalized.
9. Add boot discovery service and runtime drift monitor that drains Brutus before new work when accepted inventory no longer matches.
10. Add fixture-based tests for add/remove/replacement/move/weak identity/Windows ordinal changes and discovery-order independence.

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

@dataclass(frozen=True)
class GpuRequirement:
    architecture: str
    count: int = 1
    min_vram_bytes: int = 0
    homogeneous_model: bool = True
    model: str | None = None
    require_peer_access: bool = False
    exact_device_ids: tuple[str, ...] = ()

@dataclass(frozen=True)
class SeriesGpuBinding:
    requirement: GpuRequirement
    selected_device_ids: tuple[str, ...]
    hardware_cohort_hash: str
    accepted_inventory_hash: str
    reserve_all_of_arch: bool
    slurm_architecture: str
    slurm_count: int
```

Separate hashes:

```python
def inventory_material_hash(inv: HardwareInventory) -> str:
    # includes operational locators; detects BDF/render/NUMA movement
    ...

def hardware_cohort_hash(inv: HardwareInventory, selected_ids: tuple[str, ...]) -> str:
    # stable IDs + arch/model/VRAM + relevant topology; excludes ordinal/BDF
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

Do not assume any preferred field exists until probed on Brutus. Capture raw sanitized outputs from the installed AMD-SMI/ROCm tools as test fixtures, then implement exact parser/version handling. Reconcile identity with render node/BDF using sysfs (`/sys/class/drm/renderD*/device`) rather than ordinal ordering.

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

Paths are configurable for tests; host defaults:

```text
/var/lib/bigcherry/hardware/observed.json
/var/lib/bigcherry/hardware/accepted.json
```

Boot:

1. discover current;
2. atomically write observed;
3. compare accepted;
4. if no accepted state or material mismatch: keep/drain Slurm node, emit wake, forbid new hardware work;
5. operator runs `bigcherry hardware accept --inventory-hash ...` after inspection/tests;
6. generate/apply Slurm GRES and resume only after `jobs doctor`/acceptance.

Runtime monitor re-discovers on udev signal and low-rate timer. Any add/remove/replacement/BDF/render/topology material drift drains the node before new managed work. Existing timed measurement is contamination/invalid if inventory changes while running.

Drift classes:

```text
same                    no action
locator-only            drain/reconcile; cohort may remain same if relevant topology unchanged
topology-change         drain; new hardware cohort
device-add/remove       drain; inventory acceptance required
device-replacement      drain; new hardware cohort
identity-ambiguous      drain; new hardware epoch/operator decision
driver/environment      platform environment change; new series
```

A physical card moved slot with unchanged stable ID but changed relevant topology starts a new cohort. Pure locator change with identical relevant topology does not itself change cohort, but still requires drain/reconcile because Slurm `File=` changed.

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

Node summary groups counts by architecture. Never generate `gfx1100_0`, device ID, serial or BDF as GRES Type. `AutoDetect=rsmi` is diagnostic cross-check only, never BigCherry identity authority.

### Deterministic series hardware binding

Series creation must bind hardware **before** computing the final series identity:

```python
def bind_gpu_requirement(
    req: GpuRequirement,
    inventory: HardwareInventory,
) -> SeriesGpuBinding: ...
```

Rules:

1. canonicalize accepted devices by stable `device_id`, never discovery order/current ordinal;
2. filter by architecture, VRAM and optional explicit model;
3. exact IDs must all exist and satisfy every capability;
4. for `homogeneous_model=True` with no explicit model, form model groups; keep only groups that can satisfy `count`; if more than one model group can satisfy the request, **fail as ambiguous** and require `model=` or exact IDs rather than silently choosing one model;
5. peer requirement chooses the lexicographically canonical valid stable-ID tuple from the eligible peer graph; no valid tuple => fail;
6. otherwise choose the canonical first `count` stable IDs;
7. compute `hardware_cohort_hash` immediately from selected IDs/relevant topology;
8. persist selected IDs + accepted inventory hash in `SeriesGpuBinding` and therefore in series intent;
9. every session/attempt in the series reuses that binding; it may never select a different card merely because Slurm allocated one.

This deliberately trades some utilization for scientific comparability. A new hardware cohort requires a new series.

### Slurm capability request and exact cohort enforcement

If all devices of an architecture satisfy the requirement and the selected cohort equals whatever any architecture/count allocation could return, submit normal:

```text
--gres=gpu:<arch>:<count>
```

Otherwise set `reserve_all_of_arch=True`, request every accepted GPU of that architecture, then select only the series-bound stable IDs inside the exclusive allocation. This covers exact IDs, heterogeneous same-arch cards, VRAM subsets and peer-pair subsets without per-slot GRES types.

Post-allocation:

```python
def verify_series_allocation(
    binding: SeriesGpuBinding,
    allocation: Allocation,
    inventory: HardwareInventory,
) -> SelectedAllocation: ...
```

Requirements:

- accepted inventory hash still matches;
- all selected stable IDs are inside the Slurm allocation;
- current capabilities/topology still match the bound cohort;
- narrowed visibility is a subset of the allocation, never broader.

Default normal Slurm mode preserves scheduler-generated `ROCR_VISIBLE_DEVICES`. Subset-overallocation is the explicit exception: every narrowed-away card is already allocated exclusively to the same job, so BigCherry may narrow `ROCR_VISIBLE_DEVICES` to selected UUID(s), recording original allocation and selected subset.

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

series_id = hash(
  scientific request + contract/composition identity
  + platform_environment_hash
  + hardware_cohort_hash
)
```

Consequences:

- same-model replacement card => new series;
- Windows HIP vs Linux ROCm => new series;
- same card + changed relevant topology => new series;
- discovery reorder/BDF/render ordinal reorder alone does not change series;
- session may not aggregate across hardware cohorts;
- if bound hardware disappears/drifts, existing series is blocked; it is never rebound automatically.

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

Discovery providers take a `CommandRunner`/filesystem root abstraction so raw fixtures test parsing without installed ROCm.

```python
class HardwareProvider(Protocol):
    def discover(self) -> HardwareInventory: ...

def classify_drift(accepted: HardwareInventory, observed: HardwareInventory) -> DriftReport: ...
def bind_gpu_requirement(req: GpuRequirement, inventory: HardwareInventory) -> SeriesGpuBinding: ...
def verify_series_allocation(binding: SeriesGpuBinding, allocation: Allocation, inventory: HardwareInventory) -> SelectedAllocation: ...
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

- discovery/device fixture reorder yields the same bound stable IDs and series hardware cohort;
- mixed same-arch model groups that can both satisfy a homogeneous request fail ambiguous unless `model`/exact IDs supplied;
- same stable IDs + only ordinal/BDF/render reorder retains cohort but changes material inventory/GRES as appropriate;
- add/remove/replacement classifications;
- topology change -> cohort changes;
- weak identity requires explicit epoch;
- mixed same-arch VRAM triggers over-allocation;
- exact unknown ID rejected;
- peer requirement selects canonical deterministic valid pair; no valid pair rejected;
- two separately expanded sessions in one series carry identical `SeriesGpuBinding`;
- allocation missing one bound selected ID fails before campaign spawn;
- hardware drift never silently rebinds an existing series;
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

Large. Stable hardware identity availability is the key empirical uncertainty; weak epoch fallback prevents lack of UUID from silently mixing cards/series. Exact cohort binding can over-allocate identical-architecture cards; correctness/comparability takes precedence until a scientifically justified cross-card cohort policy exists.

## Standards

RCD02 capability/identity boundary; RCD03 consumes generated GRES; RCD04 series creation consumes `SeriesGpuBinding`; RCD11 consumes stable IDs for production claims.

## Acceptance Criteria

- runtime hardware truth is discovered, persisted and accepted explicitly;
- material drift blocks/drains new managed work;
- GRES uses architecture types only;
- jobs express capability requirements, not physical indices;
- final series identity is bound to a deterministic exact hardware cohort before attempts are submitted;
- every session in a series uses the same stable-device cohort;
- hardware cohort prevents card replacement/topology mixing;
- Windows and Linux discovery map to the same provider-neutral model;
- all offline fixture/drift/resolver tests pass before hardware acceptance.

## Notes

The lab planning simulator is intentionally simplified and currently validates the broad capability/cohort rules. Production implementation must additionally prove discovery-order-independent binding and mixed-model ambiguity handling with permanent tests before RCD12 completion.

## Change Log

- 2026-09-26T01:39:24.166357+00:00 (created-by): Created by agent
- 2026-09-26 (dev-gpt-agent): Fully specified provider model, drift fencing, generated GRES, capability resolver and cohort identity.
- 2026-09-26 (dev-gpt-agent): Adversarial follow-up: bind deterministic stable-device cohort at series creation; forbid allocation-time cohort drift/discovery-order selection.
