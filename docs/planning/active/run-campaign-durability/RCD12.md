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

# Dynamic GPU discovery, generated GRES and hardware-cohort identity

## Objective

Make discovered/accepted hardware authoritative. Jobs request capabilities, not physical indices, and every series binds one deterministic exact stable-device cohort before its sessions execute.

## Implementation status — 2026-09-27

Implemented/offline-tested:

- provider-neutral `DeviceRecord`, `HardwareInventory`, `SeriesGpuBinding` and material/cohort hashes;
- observed/accepted inventory catalog with reviewed-hash acceptance and explicit material-change opt-in;
- `hardware show|diff|record-observed|accept|render-gres` CLI;
- Linux AMD discovery provider `hardware/linux_amd.py` using machine-readable AMD-SMI list/static data plus sysfs NUMA:
  - UUID preferred stable identity;
  - serial fallback;
  - explicit weak `hardware_epoch` fallback only when no stable UUID/serial exists;
  - architecture/model/VRAM/BDF/render/HIP ordinal/driver capture;
  - malformed/ambiguous/missing required data fails closed;
- `python -m bigcherry hardware discover EXECUTOR_ID [--record]` integration;
- deterministic capability binding, homogeneous-model ambiguity rejection, peer-pair selection and proper-subset all-of-architecture reservation;
- exact stable-allocation verification; native scheduler ordinals alone fail;
- deterministic architecture-only GRES/node rendering with explicit render-node validation;
- drift classification for same/locator-only/topology/add/remove/replacement/environment changes;
- runtime visible-token -> stable-ID mapping and allocation-local selected positions;
- permanent tests for inventory acceptance, binding, drift, GRES, runtime mapping and AMD-SMI fixture parsing.

Still hardware/Windows gated:

1. capture sanitized real Brutus AMD-SMI output and prove the parser/field meanings against the installed version;
2. prove chosen UUID/serial identity persists across reboot/toolchain views; otherwise use a reviewed weak hardware epoch;
3. discover/verify real peer topology rather than leaving `peer_access` empty in Linux discovery;
4. generate/apply actual Brutus GRES and pass `slurmd -G`;
5. attest real Slurm allocation visibility back to stable IDs and run HIP/llama/tensor-split acceptance;
6. add boot/runtime drift watcher + automatic node drain/reconcile/resume integration;
7. add Windows HIP UUID/LUID discovery and ordinal-reorder fixtures/acceptance.

## Identity rules

Stable identity preference on Linux:

```text
AMD/HIP UUID -> confirmed hardware serial -> explicit weak hardware_epoch
```

PCI BDF, render node and launch ordinal are observations/locators only. Locator-only movement drains/reconciles scheduler mapping but need not start a new scientific cohort when stable identity and relevant topology are unchanged. Device replacement or relevant topology change starts a new cohort/series.

Weak identity is explicit: `weak:<hardware_epoch>:<bdf>`. It never silently establishes continuity across a new epoch.

## Capability binding

`GpuRequirement` includes architecture, count, minimum VRAM, homogeneous-model rule, optional model, peer requirement and optional exact stable IDs. Binding is canonical by stable ID. Existing series are never rebound because a different compatible card becomes available later.

For a frozen cohort that is a proper subset of all accepted cards of its architecture, Slurm reserves the complete architecture pool. BigCherry verifies that complete stable pool, then narrows to the frozen cohort inside the exclusive allocation. This is intentionally conservative.

## Accepted inventory lifecycle

```text
<work>/hardware/<executor>/observed.json
<work>/hardware/<executor>/accepted.json
```

Operator flow:

```bash
python -m bigcherry hardware discover brutus --record
python -m bigcherry hardware diff brutus
python -m bigcherry hardware accept brutus --expected-hash <reviewed-hash> [--allow-material-change]
python -m bigcherry hardware render-gres brutus
```

Material drift blocks/requires review before new scientific binding. `environment.local.toml` may configure host/toolchain/model policy but cannot override discovered GPU identity/architecture/VRAM.

## Brutus acceptance

Before production cutover:

- record real identity source for every GPU and prove reboot/toolchain stability;
- record real peer matrix and compare to HIP/tensor-split preflight;
- render accepted `gres.conf`/node GRES and pass `slurmd -G`;
- for every supported single/dual allocation prove exact stable-ID visibility, HIP properties, repeated initialization, llama-bench/server smoke and required producer preflights;
- test locator-only and material-drift drain/reconcile without silently rebinding an existing series.

## Acceptance criteria

- observed/accepted hardware is explicit/auditable;
- material drift blocks new managed work until reviewed;
- scientific requests contain capabilities/stable IDs, never slot identity;
- series identity contains exact deterministic physical cohort before execution;
- runtime allocation attestation prevents same-model substitution;
- GRES remains architecture-only;
- Linux/Windows share the provider-neutral model;
- real Brutus and Windows provider acceptance completes the remaining platform-specific gates.

## Change log

- 2026-09-26: dynamic discovery/cohort design.
- 2026-09-27: provider-neutral models, binding, allocation verification, GRES renderer and drift classifier implemented.
- 2026-09-27: observed/accepted hardware CLI and fail-closed AMD-SMI Linux discovery implemented with offline fixtures; real Brutus identity/peer/GRES acceptance and Windows discovery remain pending.
