---
id: RSC05
order: 15
plan: run-serving-configuration
state: pending
created-at: '2026-11-02T03:40:33.029040+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: M
priority: P1
---

# Serving config and layout resolver consume discovered hardware, not a hand-declared device list

## Description

Today the layout resolver and the serving config take GPUs from the hand-written [host.*.devices] list in config/environment.local.toml (index, arch, vram-mib, locator). Nothing reads the live system: no host RAM, free memory, /dev/shm, swap or PCIe link. A separate discovery path already exists for job placement: hardware.linux_amd.discover_linux_amd -> HardwareInventory (amd-smi plus sysfs). It was broken on amd-smi 26.x and is fixed by PR #518. Two device models now coexist: environment.Device (declared) and hardware.DeviceRecord (discovered). The serving stack should use one: discovered.

## Steps

Reuse implemented C++ discovery and current physical identity mapping; audit actual production callers rather than add another inventory. Select ordered BDFs from native cards and x-bigcherry rank/helper constraints; map to child-local ordinals via existing visibility environment, verify native Fleet and offline selected devices. Site YAML carries exclusions/constraints and per-BDF power overrides only, never handwritten hardware facts. Migrate actual serving power consumers of local environment TOML to one YAML authority (PWR08/RSC01); leave general job placement on its existing discovery, no dual serving inventory. Detect/explain names actual facts and origin.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Card-free tests with fake sysfs and amd-smi fixtures (26.x shape included). On this host, `bigcherry hw detect` lists 4 GPUs: gfx1100 x2 at 24,560 MiB, gfx1201 at 32,624 MiB and gfx1030 at 16,368 MiB, with correct HIP ordinals and Gen4/Gen5 x16 links, plus host RAM and shm. `config explain` for flashnext-2xtx-rdna2-tail resolves the same helpers and budgets as today. The RSC01 parity test still passes.

## Effort & Risk



## Standards



## Acceptance Criteria

No serving or layout code reads the device inventory from environment.local.toml; the remaining overrides are keyed by PCI locator. Removing or adding a GPU changes the resolved layout candidates with no config edit, and an unsatisfiable layout fails closed with a message naming the missing card.

Native online/offline/helper and power selection agree on exact ordered physical BDFs including reversed/mixed TP and duplicate-arch boards, no helper overlap, wrong local ordinal refused; selected physical devices cannot change after admission. All serving selection uses discovery; per-device policy YAML only, no second hardware declaration. Existing no-config-edit discovery and missing-card refusal remain.

## Notes

Prerequisite for RSC03, which derives host-pool, KV-tier and tail budgets from detected RAM and VRAM, and for layout-agnostic discovery (RHP12+). Follows no-hand-picked-resource-constants: hardware facts are discovered, and only calibrated per-card choices stay in config.

PR #518 fixed the Python amd-smi discovery used by job placement, which is still needed for the job tooling. Serve-time detection is C++ so the release package stays single-language (owner, 2026-11-02). Prerequisite for RSC03 (derived budgets) and for layout-agnostic discovery (RHP12+).

2026-11-25 owner direction: no launch manifest, all authored serving configuration YAML, no compatibility shims or legacy backports. docs/design/radiance-native-yaml-launch.md supersedes earlier manifest-launch instructions; completed RUM04/05/06/09/12 evidence is historical, not proof of the new launch path. Existing capture manifests may remain output-only. No new programme or second resolver. RUM05 delivered ordered/disjoint-card mapping and initial room admission; reuse and regression test, do not redo its allocator. RSC05 remains owner of discovery/selection and stale authored inventory cleanup.

## Change Log

- 2026-11-02T03:40:33.029040+00:00 (created-by): Created by agent
- 2026-11-02T03:52:53.457829+00:00 (updated-by): Updated: section:steps, section:notes

## Ledger-events

- chg_20261102_035356_serving-start-up-plan-one-str_4985
- 2026-11-02T03:54:07.803289+00:00 (updated-by): Updated: section:ledger-events

## Reviews

- RV4592
- 2026-11-25T01:37:30.794580+00:00 (updated-by): Updated: section:steps, section:notes, section:acceptance_criteria
