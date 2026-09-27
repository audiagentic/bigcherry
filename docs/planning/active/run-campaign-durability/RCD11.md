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

## Objective

Support scheduler-neutral exact-cohort execution while protecting timed Brutus measurements from llama-swap regardless of which GPUs production currently uses.

## Implementation status — 2026-09-27

Implemented/offline-tested:

- durable detached `LocalExecutor` with reconnectable metadata/result, cancellation and stable-device allocation reporting;
- local exact-cohort mapping from accepted stable IDs to current launch ordinals and OS-held per-device locks;
- restricted JSON `RemoteExecutor` + worker protocol with submit/correlate/status/cancel/control/allocation/events and stable execution IDs;
- registry support for local/remote targets; Windows example remains disabled;
- `tools/bigcherry/jobs/production.py` pure production policy:
  - declarative `all`, `uuid:<id>...`, and conservative `arch:<gfx>,count=N` claims;
  - malformed/missing/contradictory ownership fails closed to all-device potential;
  - dynamic conflict decision after exact target stable IDs are known;
  - continuous snapshot contamination detection for config/claim changes and target GPU process/VRAM use;
  - exclusive-window state model/recovery tests.

Not production-ready/hardware-service gated:

1. live llama-swap config `/running` + backend argv/env + AMD process/VRAM snapshot adapter;
2. root-owned measurement-window service/sudoers admission barrier and boot recovery;
3. real prevention of conflicting production reload during an active exclusive window;
4. qualification of any non-conflicting loaded-idle production mode before it can contribute gating evidence;
5. target-local source/model/corpus/artifact staging for remote scientific execution;
6. Windows HIP UUID/LUID discovery + stable-ID-to-current-ordinal attestation;
7. LocalExecutor target-local hardware re-attestation immediately before scientific spawn.

Remote Windows therefore remains a protocol extension scaffold, not an enabled scientific executor.

## Production policy

At each timed Brutus execution construct:

```text
config_hash
potential stable devices (or ALL)
running stable devices
observed process/VRAM stable devices
ambiguities
```

Potential means every device production could choose under unchanged configuration, not only what it happens to use now. `arch:gfx...,count=N` expands to every accepted candidate of that architecture unless the production launcher itself has deterministic reserved placement.

Decision:

```text
target intersects potential OR ownership ambiguous -> exclusive window
disjoint                                      -> loaded-idle only after isolation qualification
```

Until loaded-idle is scientifically qualified, timed gating measurement should quiesce production rather than treating disjointness as proof of no host-level interference.

During measurement, config hash/potential-set changes, new ambiguity, or production use of a target GPU invalidates the sample as environment contamination. It is not a scientific regression result.

## Exclusive window target

Root-owned state machine:

```text
CLOSED -> REQUESTED -> DRAINING -> ACTIVE -> CLOSING -> CLOSED
```

Request contains only execution/native ID, exact target stable IDs, accepted inventory hash, production config hash and bounded deadline. No caller-provided root shell command. Entry blocks new supported production/ad-hoc starts before drain; exit/recovery is idempotent and must restore production health or emit a hard wake.

## Remote target requirement

Before enabling remote scientific work, central submission must send portable identity, not controller absolute paths. Target worker must materialize/verify exact BigCherry commit, source identity, model/corpus/input hashes, target-local workspace/cache/log paths and accepted hardware hash before constructing its local `ExecutionRequest`. Network uncertainty reconciles by `execution_id`; never blind duplicate submit.

## Acceptance criteria

- LocalExecutor never substitutes outside the frozen cohort;
- real local spawn independently re-attests the frozen stable devices;
- remote worker can execute a target-local staged campaign without controller filesystem assumptions;
- network ambiguity cannot duplicate execution;
- production conflict policy is independent of static card placement;
- conflicting production cannot load onto target GPUs during a window;
- non-conflicting co-residency is gating-enabled only after isolation qualification;
- production drift contaminates/discards the sample;
- root privilege surface is exact/minimal and crash/boot recovery is proven.

## Change log

- 2026-09-26: LocalExecutor/production-coexistence design.
- 2026-09-27: LocalExecutor, remote protocol/worker and fake remote acceptance implemented.
- 2026-09-27: production claim/snapshot/conflict/contamination/window pure policy implemented and tested; live llama-swap/root-window and remote staging remain hardware/service gates.
