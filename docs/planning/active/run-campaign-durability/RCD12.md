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

## Objective

Make discovered hardware state authoritative. Cards may be added/removed/replaced/moved; Windows ordinals may reorder. Scientific jobs request capabilities, not physical indices, and each series binds one deterministic exact stable-device cohort before its sessions execute.

## Implementation status — 2026-09-27

Implemented:

- `tools/bigcherry/hardware/model.py`: provider-neutral `DeviceRecord`, `HardwareInventory`, `SeriesGpuBinding` and material hashes.
- `hardware/inventory.py`: accepted-inventory catalog, deterministic capability binding by stable ID, homogeneous-model ambiguity rejection, peer-pair selection and proper-subset all-of-architecture reservation.
- `verify_series_allocation()`: executor allocation must attest stable GPU IDs; accepted inventory/cohort/topology drift fails; safe-overallocation must contain the complete architecture pool.
- `hardware/slurm.py`: deterministic architecture-only explicit `gres.conf` rendering and grouped node `Gres=` rendering. Missing/duplicate render nodes fail closed.
- `hardware/drift.py`: accepted vs observed classification for same/locator-only/topology/add/remove/replacement/environment changes.
- RCD04 series planning already includes `platform_environment_hash`, `hardware_cohort_hash`, accepted inventory hash and exact selected stable IDs before run records are created.
- permanent hardware/cohort/GRES/drift tests were added under `tools/tests/jobs`.

Still incomplete/hardware-gated:

1. Linux AMD discovery provider (`amd-smi`/RSMI/sysfs reconciliation) is not implemented because exact Brutus output/provider stability must be captured first.
2. Windows HIP UUID/LUID provider is not implemented.
3. observed/accepted inventory CLI (`hardware discover/diff/accept`) and boot/watch systemd units remain unimplemented.
4. actual Brutus `gres.conf` generation/application + drain/reconfigure/resume workflow remains untested on hardware.
5. Slurm native allocation -> stable ID attestation is not yet production-complete; native ordinals alone fail `verify_series_allocation`.
6. peer topology must be checked against real HIP/tensor-split behavior.

## Models

```python
@dataclass(frozen=True)
class DeviceRecord:
    device_id: str
    identity_source: str
    architecture: str
    model: str
    vram_bytes: int
    pci_bdf: str | None
    render_node: str | None
    numa_node: int | None
    driver_version: str | None
    launch_ordinal: int | None

@dataclass(frozen=True)
class SeriesGpuBinding:
    requirement: GpuRequirement
    selected_device_ids: tuple[str, ...]
    hardware_cohort_hash: str
    accepted_inventory_hash: str
    reserve_all_of_arch: bool
    scheduler_architecture: str
    scheduler_count: int
```

BDF/render node/ordinal are locators only, never scientific identity.

## Stable identity discovery policy

Linux preference:

1. AMD stable UUID if confirmed stable;
2. RSMI unique ID;
3. hardware serial if confirmed unique/stable;
4. explicit weak `hardware_epoch` fallback.

Windows preference:

1. HIP UUID;
2. Windows LUID;
3. weak epoch.

Do not hard-code a provider parser until raw sanitized outputs from the installed Brutus/Windows runtimes are captured as permanent fixtures. Reconcile stable identity to BDF/render node through sysfs/provider facts, never ordinal ordering.

## Accepted inventory lifecycle

Target state:

```text
<work>/hardware/<executor>/observed.json
<work>/hardware/<executor>/accepted.json
```

Boot/runtime flow:

1. discover current inventory;
2. atomically write observed;
3. compare accepted with `classify_drift()`;
4. material mismatch drains/blocks new hardware work and emits wake;
5. operator inspects/accepts exact observed hash;
6. render GRES/node config from accepted state;
7. run `slurmd -G` and jobs/hardware doctor;
8. reconfigure/restart slurmd as required;
9. resume only after attestation passes.

Drift classes:

```text
same                    no action
locator-only            drain/reconcile; cohort may remain same
topology-change         drain; new cohort
device-add/remove       drain + acceptance
device-replacement      drain + new cohort
environment-change      new platform environment/series
identity-ambiguous      explicit new hardware epoch
```

## Capability binding

`GpuRequirement`:

```python
architecture
count
min_vram_bytes
homogeneous_model
model
require_peer_access
exact_device_ids
```

Binding rules:

1. canonical sort by stable `device_id`;
2. filter architecture/VRAM/optional model;
3. exact IDs must satisfy all constraints;
4. if multiple same-arch model groups can satisfy a homogeneous request and model is unspecified, fail ambiguous;
5. peer work chooses the lexicographically canonical valid stable-ID tuple;
6. otherwise choose canonical first `count` IDs;
7. compute cohort hash from stable IDs + arch/model/VRAM + relevant peer/topology + platform environment;
8. persist selected IDs and accepted inventory hash in the series.

Existing series are never rebound because a different card becomes available.

## Architecture-only Slurm reservation

GRES rendering:

```text
Name=gpu Type=gfx1100 File=/dev/dri/renderD128 Flags=amd_gpu_env
Name=gpu Type=gfx1100 File=/dev/dri/renderD129 Flags=amd_gpu_env
Name=gpu Type=gfx1201 File=/dev/dri/renderD130 Flags=amd_gpu_env
```

No slot/UUID in `Type`.

If the frozen cohort is a proper subset of accepted devices of an architecture, request the full architecture count. Once external allocation is exclusively owned, narrow to the frozen stable IDs. This is inefficient but scientifically deterministic.

`verify_series_allocation()` requires:

- accepted material hash unchanged;
- non-empty attested `Allocation.stable_gpu_ids`;
- every frozen selected ID present;
- capability selection still resolves to exactly the frozen cohort;
- cohort hash unchanged;
- no unexpected extra stable GPU unless safe-overallocation was declared;
- safe-overallocation receives the complete accepted architecture pool.

Native scheduler IDs cannot satisfy this check by themselves.

## GRES renderer

`render_gres_conf()`:

- deterministic independent of discovery order;
- requires explicit render node per GPU;
- rejects duplicate render nodes;
- uses architecture-only `Type`;
- never emits stable IDs/serials/BDF as resource names.

`render_node_gres()` groups accepted device counts by architecture.

## environment.local.toml role

It may configure host name, toolchain/model roots, allowed architectures and optional human aliases. It cannot override observed GPU architecture/VRAM/stable identity.

## Offline tests required

Implemented/permanent:

- discovery-order-independent binding;
- mixed-model ambiguity;
- peer pair canonical selection;
- subset -> all-of-arch reservation;
- stable allocation verification;
- native-only allocation fails attestation;
- missing selected card fails;
- inventory drift fails existing allocation;
- architecture-typed deterministic GRES;
- missing render node fails closed;
- locator-only/topology/replacement drift classification.

Still add with provider implementation:

- AMD-SMI/RSMI/sysfs fixtures across installed runtime versions;
- UUID/unique-ID/serial precedence/fallback;
- weak epoch behavior;
- Windows HIP UUID/LUID + ordinal reorder;
- observed/accepted accept workflow;
- generated node config/drain decisions;
- hot-drift watcher behavior.

## Brutus gates

- determine actual stable ID source for every installed GPU;
- prove identity persists across reboot/toolchain views;
- generate real GRES and pass `slurmd -G`;
- map real Slurm allocation to stable IDs;
- run 100 HIP init/property cycles per single/dual allocation;
- prove peer matrix matches `-sm tensor` preflight;
- test safe node drain/reconcile without physically removing live hardware.

## Acceptance criteria

- discovered/accepted hardware state is explicit and auditable;
- material drift blocks new managed work;
- jobs request capabilities, never physical index;
- final series identity contains one deterministic exact hardware cohort before execution;
- allocation attestation prevents substitution;
- GRES is architecture-only and deterministic;
- Windows/Linux use the same provider-neutral identity model;
- all fixture tests green before hardware acceptance.

## Change log

- 2026-09-26: original dynamic discovery/cohort design.
- 2026-09-27: provider-neutral models, deterministic capability binding, allocation verification, GRES renderer and drift classifier implemented with offline tests.
