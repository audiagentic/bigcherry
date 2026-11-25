# RSC — serving configuration: native YAML, discovered and measured facts

Goal: one native Radiance YAML per deployment, closed x-bigcherry extensions, typed site/power policy YAML and existing C++ bc-serve owner. Native settings stay native; derived facts and measured data are not authored twice. No executable launch manifest, compatibility reader or second resolver. Current implementation still has the v3 launch envelope: S18 removes it in one coordinated cutover. RUM04's authored YAML migration is historical completed scope, not S18 acceptance.

Governing designs: [native YAML launch](../../../design/radiance-native-yaml-launch.md), [serving/config closure sweep](../../../design/radiance-serve-config-review-20261125.md), [upstream bump sequence](../../../design/radiance-bump-integration.md).

| Item | Owns | Dependencies / boundary |
| --- | --- | --- |
| [RSC01](RSC01.md) | Native option authority, closed BC/site/profile YAML, one explain/resolver | RSC12, PWR08 policy fields |
| [RSC12](RSC12.md) | Named full-file variants, native semantic checks, approved release roster | RSC01, RSP01; no inheritance |
| [RSC05](RSC05.md) | Physical selection, ordered visibility/ranks and provider coverage | RSC01, native matched parser |
| [RSC03](RSC03.md) | Measured resources/cache, fresh host/VRAM/pin checks | RSC05; RSC14 contention proof |
| [RSC06](RSC06.md) | Helper role/transport settings, generation, readiness and drain | RSC03/05, RHP16/20 |
| [RSC02](RSC02.md) | Existing owner, native online/offline entry, status/stop, power lifecycle | RSC01/03/05/06/12, PWR07/08 |
| [RSC13](RSC13.md) | AST callable guard audit | RSC02; independent |
| [RSC14](RSC14.md) | Host contention and early refusal/retry qualification | RSC03 and RSC11's one bounded retry |
| [RSC11](RSC11.md) | Single integrator: S17 residual proof and S18 native launch cutover | All above; MEN17/20/21, RSP01, PWR10, RHP12; WSL08 independently gates detached power |

Completed RSC04/07–10 obligations transferred to RSC11 are historical or residual acceptance, not duplicate implementation. S18 order: A contract/roster/callers, B measured facts/helpers/power, C native entry and caller migration, D full installed supported-config acceptance. Routine releases retain Gate E plus affected-path gates. Capture and run manifests may remain evidence only (WSL01). Live policy remains separately qualified (RHP16/20); calibration and derived kernel/lane settings stay with their existing owners. Every changed default passes the existing RHP12 cohort. No source or hardware qualification is implied by this index.
