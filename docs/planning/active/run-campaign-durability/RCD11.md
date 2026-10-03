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

# Local/remote execution and dynamic production coexistence

## Status — 2026-09-28

**Scheduler-neutral local/remote protocol and dynamic production policy are implemented; live llama-swap/root-window integration and Windows remote staging remain real-host/service gates.**

Implemented/offline-tested:

- durable LocalExecutor with reconnectable metadata/result, cancellation and stable-device reporting;
- local frozen-cohort mapping to current accepted launch ordinals plus OS-held per-device locks;
- attempt runner re-loads accepted inventory, rebinds/re-attests the exact frozen cohort immediately before scientific child spawn; drift fails closed;
- restricted JSON RemoteExecutor/worker protocol with stable execution correlation and normalized status/control/allocation/events;
- production pure policy for `all`, exact UUID and conservative architecture/count claims;
- malformed/missing/contradictory production ownership fails closed to all-device potential;
- dynamic conflict after exact target allocation, not static card classes;
- production config/claim/target-use contamination detection;
- crash-safe exclusive-window state model/recovery logic;
- monitored attempt execution with stable-ID attestation before child launch.

## Production decision

For each timed Brutus run derive:

```text
config_hash
potential stable devices (or ALL)
running stable devices
observed process/VRAM stable devices
ambiguities
```

Potential means every device production could select under unchanged configuration. `arch:gfx...,count=N` therefore expands to all accepted candidates unless the production launcher has an independently enforced deterministic reservation.

```text
target intersects potential OR ownership ambiguous -> exclusive window
proven disjoint                                      -> loaded-idle only after isolation qualification
```

Until loaded-idle is qualified, timed gating measurement must quiesce production even on disjoint GPUs because host-level interference remains unproven.

Any config hash/potential/ambiguity change or production process/VRAM use on a target GPU during measurement is environment contamination; discard/retry the sample, never classify it as scientific regression.

## Remaining Brutus service work

1. live adapter for llama-swap config + `/running` + backend argv/env + accepted-stable-ID AMD process/VRAM attribution;
2. root-owned admission/window helper and minimal sudoers/systemd surface with boot/crash recovery;
3. prove conflicting production cannot reload while a window is active;
4. qualify loaded-idle/noise before enabling non-conflicting production during gating measurements.

llama-swap model `env` is the intended declarative place for per-model BigCherry GPU claims; ambiguous or missing claims remain fail-closed. The host adapter must verify configured claims against observed process use rather than trust them blindly.

## Remaining remote/Windows work

- target-local source/model/corpus/artifact staging and exact content verification;
- Windows HIP UUID/LUID discovery and stable-ID -> current launch selector attestation;
- hardware acceptance of process-tree cancellation and remote reconnect/recovery.

Controller absolute paths are never portable scientific authority. Windows evidence remains a separate platform series from Linux ROCm.

## Acceptance criteria

- no local execution substitutes outside the frozen cohort;
- network ambiguity cannot duplicate remote execution;
- production conflict policy is independent of card slot/placement;
- conflicting production cannot load onto target GPUs in an exclusive window;
- loaded-idle is gating-enabled only after isolation evidence passes;
- root privilege surface is exact/minimal and recovery restores safe production state;
- remote scientific work remains disabled until target staging + Windows stable identity are proven.

## Change log

- 2026-09-27: Local/Remote executors and production pure policy implemented.
- 2026-09-28: reconciled runner-time local stable-ID re-attestation as implemented; remaining work narrowed to live production/root service and remote Windows staging/acceptance.
