# Execution guide for the staged migration

The initial roadmap and reviews merged in PR #1094 before this detail pass.
The authoritative items below remain pending; writing a detailed plan does not
mean the migration was implemented. Edit their scope/state through Planning MCP.

## Execution order

| Stage | Item | Small deliverable and cleanup boundary |
| --- | --- | --- |
| 1 | [RUM01](../../planning/active/patching-radiance-upstream/RUM01.md) | Exact target, one deployed configuration, ownership and patch dispositions. Approve exclusions before coding. |
| 2A | [RUM02](../../planning/active/patching-radiance-upstream/RUM02.md) | Extract injected implementation into owned files; keep direct typed hooks. Remove replaced raw-string implementations in the same slice. |
| 2B | [RUM03](../../planning/active/patching-radiance-upstream/RUM03.md) | Pin JSON independently, port shared Registry eligibility and rebuild one plugin ABI/SDK generation. Remove duplicated include assumptions. |
| 3 | [RUM04](../../planning/active/patching-radiance-upstream/RUM04.md) | Native YAML plus closed extension intent, one parser lifetime and admitted snapshot. Replace argv bridge and migrated duplicate config paths. |
| 4 | [RUM05](../../planning/active/patching-radiance-upstream/RUM05.md) | Shared immutable placement view and external helper reservations in upstream accounting; reuse current calibrated policy. |
| 5 | [RUM06](../../planning/active/patching-radiance-upstream/RUM06.md) | Engine/group-owned remote execution, exact ticket terminal states and drain-before-free; no new transport algorithm. |
| 6 | [RUM07](../../planning/active/patching-radiance-upstream/RUM07.md) | One KV descriptor source and only supported retained formats; delete redundant row-size/dispatch switches. |
| 7 | [RUM08](../../planning/active/patching-radiance-upstream/RUM08.md) | Reuse upstream replay/controller where equivalent; remove old overrides only with focused proof. |
| 8 | [RUM09](../../planning/active/patching-radiance-upstream/RUM09.md) | Minimum production transition gates, pin update and one installable release bundle. |
| Later | [RUM10](../../planning/active/patching-radiance-upstream/RUM10.md) | Optional multi-model helper/reclaim support after single-model acceptance; does not block the bump. |

RUM02 and RUM03 can proceed independently after RUM01; subsequent dependencies
are recorded in each item. Keep independently compilable changes in small PRs.
Combine slices only when an interface migration cannot compile on its own.

## Reuse instead of parallel implementations

- Upstream owns native option validation, fleet lifecycle and physical allocation.
- Existing strict generated contracts and S8/S10 own helper/admission semantics.
- Existing plan/stored-manifest caches own semantic reuse and exact replay.
- Existing admission ledgers and mailbox epochs own parent/ownership lifetimes.
- RSC12/RSC13 and RCS01/RCS08 retain authoring/inventory/component ownership;
  RHP04/RHP09/RHP13 retain calibration/placement/resource ownership.
- RSP01 and current release/SDK tools retain packaging and install verification.

RSC12 and RHP13 have closed coordination reviews RV4225 and RV4226. Their
implementation states and current-pin acceptance have not been completed or
superseded by this proposal. API examples in the items are explicitly sketches;
actual upstream and local symbol names are called out separately.

## Testing budget

During plan authoring: inspect source, check plan identity/links/sections and run
required repository CI. Do not run engine builds, hardware canaries, KLD sweeps,
long soaks or timing writes just to document the work.

During implementation: reuse existing touched-component tests. Source relocation
needs build identity and focused behavior proof, not another performance campaign.
At final acceptance use one fixed production configuration and helper-off control,
one short quality/resource/lifecycle sample, and three interleaved matched
performance pairs. Use existing thresholds; add a test or repeat only when a code
change, real failure or unresolved noise justifies it. New model families, every
layout/precision mode and optional adaptive/suffix/traversal settings are not part
of the initial qualification unless selected for actual production.

## Memory follow-ups

[BHK01](../../planning/active/build-host-kernel/BHK01.md) already owns kernel
recovery. [BHK02](../../planning/active/build-host-kernel/BHK02.md) prepares verified
active timing readout, [BHK03](../../planning/active/build-host-kernel/BHK03.md)
investigates a supported write path, and
[BHK04](../../planning/active/build-host-kernel/BHK04.md) audits and repeats the RAM
comparison. Writer feasibility does not block a read-only before/after report.
No kernel load, RAM write, service restart or reboot occurs during planning.

The historical STREAM-style harness undercounts Triad traffic; its report now
labels that limitation. Correct and rerun the method in BHK04 rather than silently
claiming the old number was measured under the standard three-array convention.
