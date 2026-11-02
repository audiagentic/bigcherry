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

1. Serve-time detection in C++ in the serving component (RSC01). GPUs come from /sys/class/kfd topology (gfx target, SIMDs, render minor, location) joined to /sys/bus/pci (BDF, mem_info_vram_total, current and max link). Host facts come from /proc/meminfo (MemTotal, MemAvailable, SwapTotal), /dev/shm (statvfs), CPU count and NUMA. There is no amd-smi dependency at serve time.
2. The discovered facts feed the loader's 'discovered' layer, and layout device selection uses them, keyed by PCI locator, with the HIP ordinal derived from kfd node order.
3. The device inventory in environment.local.toml shrinks to per-card overrides keyed by locator: calibrated power (PWR04), reserve and an exclusion flag. Every environment.Device caller in the Python tooling (51 references) is migrated to `bc-serve --detect --json` output, or to hardware.linux_amd for job placement, in the same change. No dual path.
4. `bc-serve --detect [--json]` prints the inventory; explain shows discovered values with origin 'discovered'.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Card-free tests with fake sysfs and amd-smi fixtures (26.x shape included). On this host, `bigcherry hw detect` lists 4 GPUs: gfx1100 x2 at 24,560 MiB, gfx1201 at 32,624 MiB and gfx1030 at 16,368 MiB, with correct HIP ordinals and Gen4/Gen5 x16 links, plus host RAM and shm. `config explain` for flashnext-2xtx-rdna2-tail resolves the same helpers and budgets as today. The RSC01 parity test still passes.

## Effort & Risk



## Standards



## Acceptance Criteria

No serving or layout code reads the device inventory from environment.local.toml; the remaining overrides are keyed by PCI locator. Removing or adding a GPU changes the resolved layout candidates with no config edit, and an unsatisfiable layout fails closed with a message naming the missing card.

## Notes

Prerequisite for RSC03, which derives host-pool, KV-tier and tail budgets from detected RAM and VRAM, and for layout-agnostic discovery (RHP12+). Follows no-hand-picked-resource-constants: hardware facts are discovered, and only calibrated per-card choices stay in config.

PR #518 fixed the Python amd-smi discovery used by job placement, which is still needed for the job tooling. Serve-time detection is C++ so the release package stays single-language (owner, 2026-11-02). Prerequisite for RSC03 (derived budgets) and for layout-agnostic discovery (RHP12+).

## Change Log

- 2026-11-02T03:40:33.029040+00:00 (created-by): Created by agent
- 2026-11-02T03:52:53.457829+00:00 (updated-by): Updated: section:steps, section:notes

## Ledger-events

- chg_20261102_035356_serving-start-up-plan-one-str_4985
- 2026-11-02T03:54:07.803289+00:00 (updated-by): Updated: section:ledger-events
