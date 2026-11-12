# RSC03 plan-memory hardware check

Recorded 2 November 2026 for the Flash-Next four-GPU layout with an R9700
engine. This is a summary of the local RSC03 plan-memory runs; raw local
process logs and temporary paths are not required for the manifest contract.

| Check | Observed outcome |
| --- | --- |
| Engine-only memory plan, before helper ownership | Exited without loading weights. The resulting memory footprint was different because no helper-owned experts had been assigned |
| Launcher with helper owner map | Produced the memory plan and exited before weights were loaded; repeated plans were otherwise identical apart from live free-VRAM readings |
| Normal startup | Reported the same memory quantities as the first plan, subject to log rounding |
| Checkpoint geometry fix | Resolved `want_slots` from 16 to 72 for the eight-session configuration |
| Planned-versus-live admission | A real plan pair passed. Deliberately reduced expected n-gram, embedding, checkpoint or KV quantities were rejected; insufficient live claimable VRAM was rejected |

The free-VRAM measurement varied in 8 KiB steps over a 16 KiB range across
starts. The plan's statics, KV, experts and stage exactly filled its
claimable limit; a cached plan therefore cannot assume future free VRAM
will be byte-identical. Admission needs to compare the required allocation
against the live claimable memory, not require equality between readings.

A further measurement is needed before treating these checks as evidence
for unrestricted rollout. The strict v1 manifest and memory-plan
requirements are in [the serving-manifest specification](../../design/serving-manifest-v2/spec.md).
