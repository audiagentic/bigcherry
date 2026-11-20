---
id: MRC02
order: 2
plan: run-measurement-records
state: completed
created-at: '2026-11-19T00:50:28.025472+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: S
---

# Platform host-facts collector attached to every run record (kernel, ROCm, firmware, clocks, power caps, PCIe, host env digest)

## Description

A platform, engine-agnostic collector of the host facts that change measured performance, written into every record (MRC01). The 2026-11-19 Flash-Next investigation had to collect kernel, clocks and power by hand, and read a navi31 hwmon power1_cap of 0 on kernel 7.0.13 that is a kernel getter defect rather than a cap. Facts: kernel release and build string; ROCm version; amdgpu and firmware versions; per card arch, BDF and VRAM; sclk/mclk current and OD maximum; power cap plus cap_default, min and max, with a flag when a value is unreadable; PCIe link speed and width; host environment file digest; the GPU lease holder.

## Steps

- 1. Collector module (platform-owned, no engine imports) returning one host-facts dict with per-fact `readable` flags; reuse workloads.card_paths() and power_state() by moving them into the collector and leaving thin callers.
- 2. Add cap_default/cap_min/cap_max and an `unreadable` marker so a zero cap is reported as suspect rather than as a cap.
- 3. Add kernel, ROCm, amdgpu/firmware, PCIe link, environment-file digest and lease holder (from `agent-lease status` output or its state file).
- 4. Wire into MRC01 records and run manifests, with sysfs fixtures.
- 5. Radiance producers require the native owner's SHA-sealed start receipt; verify session ownership and reject unsealed Radiance records on writes and reads.
- 6. Defer llama.cpp producer and adapter changes to MRC07 without new wrappers.

## Detailed Solution & Technical Design

workloads.py already reads per-card hwmon power and clocks through card_paths(sys_root) and power_state(sys_root), both parameterised on a sysfs root so they can be tested on fixture trees; the collector generalises that and moves it out of the bench package into core. The collector only reads; it never changes power or takes a lease (same rule as workloads.py).

## Code Samples & Guidance



## Files

- tools/bigcherry/bench/workloads.py (card_paths(), power_state(sys_root), _board_watts: moved/reused)
- tools/bigcherry/core/run_manifest.py (`host` field in the manifest: filled today by callers as a free dict)
- docs/evidence/2026-11-19-flashnext-4gpu-regression/README.md (power1_cap=0 finding)

## Validation

Merged: #1191 (engine-agnostic host-facts collector: kernel, ROCm, amdgpu/firmware, per-card BDF/arch/VRAM, clocks, power cap with bounds and zero_cap_suspect, PCIe links, environment digest, lease holder, per-field readability), #1238 (every measured Radiance producer captures a sealed host-facts start receipt bound to its native owner session; publication and reads refuse missing, unwrapped, mismatched or late start facts; no reconstruction after the run). Tests: tools/tests/core/test_host_facts.py, test_measurement_host_start.py, serving bc_session_host_start_test, tools/tests/jobs managed-streaming; core/bench/lab/experiment suites 789 pass. The work root held no run records, so none needed migration.

## Effort & Risk

Small. Risk: sysfs layouts differ by kernel; keep each fact independently readable and tested.

## Standards



## Acceptance Criteria

Every Radiance run record carries the host facts observed at measurement start (sealed, bound to the native owner session), with each field's readability and reason; a record without them is refused. The collector stays engine-independent. llama.cpp start receipts are tracked in LCM02.

## Notes

Add to the fact list (2026-11-19): DIMM population, configured memory speed and profile (`dmidecode -t memory`), and MemAvailable at start. A random 1-in-160 clang crash rate and an engine host-admission refusal at 74.1 GiB available (47.68 GiB n-gram table) were both explained only by host state that no record carried (BHK01/BHK04 notes). The collector is read-only; unreadable values are flagged, not zero.

## Change Log

- 2026-11-19T00:50:28.025472+00:00 (created-by): Created by agent
- 2026-11-19T00:54:08.497481+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-19T07:38:54.834871+00:00 (updated-by): Updated: section:notes
- 2026-11-20T02:22:07.094431+00:00 (updated-by): Updated: section:validation, section:acceptance_criteria
- 2026-11-20T02:22:11.635463+00:00 (state-transition): State: pending → completed
