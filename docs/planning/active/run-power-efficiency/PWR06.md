---
id: PWR06
order: 0
plan: run-power-efficiency
state: in_progress
created-at: '2026-11-02T01:49:10.007322+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: S
priority: P1
---

# R9700 loaded-idle +4 W: GFXOFF fix for GFX12 compute queues (kernel driver) first, MES firmware second, one variable at a time

## Description

Measured 2026-11-02 (flashnext-4gpu, run r9700-idle): the R9700 (gfx1201, GC 12.0.1) idles at 15.0 W with no GPU process and 19.0 W with 3% phantom GPU activity whenever the radiance engine process exists (loaded idle, after a request, 65 s later), dropping back to 15.0 W when it exits. The helper XTX/6900 XT stay at 4-7 W (RAD20's park workaround). This matches the upstream R9700 HIP idle-activity issue (ROCm #5706/#6298) fixed in MES firmware 0x8b; this host runs MES 0x81 (Ubuntu linux-firmware 20240318 ubuntu2.27, gc_12_0_1_* files dated 2026-04-16). The 15 W unloaded floor is the board: graphics rail power-gated (VDDGFX 29 mV), memory clock at minimum, PCIe L1.1/L1.2 already enabled on the root-port-to-switch link; forcing FCLK to its minimum saves ~1 W. RAD20's earlier MES trial was GC 11.0.0 (XTX) and did not fix the XTX wait-packet poll; this is a different card and firmware.

## Update 2026-11-10 (owner brief: R9700 idle power next investigation)

**Root cause (upstream, found after the 2026-11-02 firmware plan).** The 3% phantom activity and +4 W are not a firmware defect that MES 0x8b cures. The in-tree amdgpu driver disables GFXOFF whenever a user-mode compute queue exists on GFX12 (`amdgpu_amdkfd_set_compute_idle`, the `IP_VERSION_MAJ(GC_HWIP)==12` clause, added by 2affe2bbc997 as a workaround for an OpenCL test failure), and `hipStreamCreate` leaves a queue resident, so the graphics block never powers down while a HIP process holds the card. AMD fixed it in `6a1dbb006056` "drm/amdgpu: re-enable GFXOFF for compute on GFX12" (author date 2026-10-21, committed 2026-10-27, one hunk in `drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd.c` removing the `== 12` clause; no `Cc: stable`). Upstream measurement on gfx1200: VDDGFX 695 mV -> 35 mV, load 2% -> 0%, about -2 W. Upstream report ROCm/TheRock#9245 (open, "fix submitted"): MES 0x8b, 0x93 and kernels 7.3.0-rc3/rc6 (drm-tip 2026-10-06) did NOT remove the 3%; only the driver change does. This corrects FINDINGS ("reported fixed in MES 0x8b").

**Is it in our driver? Not by version arithmetic.** Our kernel is 7.0.13-070013-generic (build 2026-06-19, in-tree amdgpu, srcversion 4F5C8C3B57F8DA15B66C039, no DKMS in the container), four months older than the fix, so the clause is expected present; our measurements match (VDDGFX 29 mV with no process, constant 3% with one). Reading the module itself needs the host (the container cannot see /lib/modules); the definitive check is behavioural: VDDGFX (hwmon in0_input) and `gpu_busy_percent` with only a minimal HIP context holding the card (`runs/r9700-gfxoff/hold`). If a minimal context reproduces 3%/raised VDDGFX the clause is active.

**What Linux needs (host prerequisites, owner-run, not manifest settings).** One variable at a time, kernel first, firmware untouched (MES stays 0x81):
1. Preferred: a kernel release that contains `6a1dbb006056` (first mainline tag unknown at fetch time; check `git tag --contains` when published; drm-amdgpu-next had not yet mirrored it).
2. Until then: the same 7.0.13 kernel with the one-hunk patch to amdgpu (out-of-tree amdgpu module via DKMS against the 7.0.13 headers, or a rebuilt kernel). Rollback = previous module/kernel entry. The amdgpu module cannot be swapped live while any of the four cards is in use: expect a reboot into the test kernel/module.
3. Only if the patched driver shows GPU hangs/queue failures on MES 0x81 (the workaround existed because of a MES firmware problem exposed by an OpenCL test; the commit says it has since been addressed without naming the version): the firmware trial (original steps below: gc_12_0_1 MES >=0x8b, backup, rollback), as a separate step.

**Harness (ready, no GPU setting changes):** `${BC_WORK_ROOT}/runs/r9700-gfxoff/` (`run.sh LABEL [REPS]`; `sample.py` raw 1 Hz CSV with timestamp, phase, watts, busy %, VDDGFX mV, sclk/mclk/fclk and the other cards' watts; `env-snapshot.sh` kernel, amdgpu srcversion and parameters, all firmware versions, link and ASPM state; `summarise.py`). Phases: no process, minimal HIP context + stream, model loaded idle, one request then idle, stopped; three reps; each lease window short and separated. Baseline (kernel 7.0.13, MES 0x81) runs 2026-11-10 as `baseline-7.0.13-mes81`; every candidate reruns the same script under its own label.

**Acceptance (owner brief):** loaded-idle activity back to 0% and about -4 W against 19 W, or an explanation; no model unloading, no cold-start latency; identical output; decode/prefill within the RHP12 gate; XTX helper idle unchanged; failed experiments rolled back; every run records driver/firmware and raw samples.

**Remaining 15 W floor (after the GFXOFF result, not before):** which board rails stay on (fabric/SoC, VRAM, PCIe switch); extra ASPM beyond L1.1/L1.2 on the on-board switch (`lspci -vv` needs root); do not repeat force-low (~1 W) or POWER_SAVING (no effect). A 5 W target is not assumed achievable.

**Programme integration.** Kernel and firmware are host prerequisites, never manifest settings. PWR07 receipts record the R9700's MES/KIQ/SMC firmware, kernel, amdgpu srcversion, measured idle watts, `gpu_busy_percent`, VDDGFX (the GFXOFF-active proxy: tens of mV when powered down) and queue count where observable, as observations, not controls. A runtime optimisation, if later proven, goes through the PWR08 profile and PWR09 controller (capability check, lease, rollback, performance validation); no new global environment flag and no logic in the shell launcher.

## Update 2026-11-11 (GFXOFF fix measured; deep-idle / runtime-PM investigation)

**GFXOFF kernel patch (7.0.13-gfxoff, MES firmware 0x81 unchanged), R9700, 3 reps x 60 s per phase, 1 Hz hwmon:** no process 15.0 W / 29 mV; HIP context resident idle 15.0 W, 0% busy, 29 mV (stock kernel: 20.6 W, 3%, 612 mV); after exit 15.0 W. Model loaded idle 15.1 W, after one request 15.0 W; the request decoded coherently at 95 tok/s with the spec head active. Kernel log clean. Kernel is a host prerequisite, not a manifest setting. Raw data: run folder r9700-gfxoff (gfxoff-7.0.13-mes81, baseline-7.0.13-mes81).

**Runtime PM (read-only, run folder r9700-deepidle):**
- All four GPUs report `Using BACO for runtime pm`, control=auto, autosuspend 5000 ms, runtime_usage 0, no active children, no DRM/KFD fds open, all connectors disconnected/disabled, yet every GPU function stays `active`/D0. Each has only 12-17 s of suspended time since boot (early boot), none since. The R9700 audio function (09:00.1) is runtime-suspended in D3hot, so runtime PM works on this platform; the GPU functions are what stay awake.
- Monitoring does not cause it: no-telemetry, runtime-PM-counters-only and hwmon+rocm-smi observation windows (3 reps x 120 s) all show zero suspended time and constant D0.
- Opening and closing the render node (which re-arms autosuspend on close) does not suspend it either; usage stays 0.
- So the blocker is not a held usage reference. Remaining candidates: the amdgpu runtime_idle callback returning busy, or BACO entry being refused for gfx1201; needs ftrace on `amdgpu_pmops_runtime_idle`/`amdgpu_pmops_runtime_suspend` in a maintenance window (not done).
- Platform: ACPI FADT declares PCIe ASPM unsupported ("using BIOS configuration"); the R9700 and its switch port have ASPM L1 enabled by BIOS, the root port 00:06.0 is ASPM-not-supported and limits the path to Gen4 x4; D3hot and D3cold are PME-capable (D3cold_allowed=1). The owner notes BIOS PCI power saving may be off because it caused crashes.
- No deeper state was forced; nothing was written to the device. A resident model could not be shown to survive a BACO/D3 cycle (VRAM is lost across BACO), so runtime suspend is not a serving mechanism unless that is separately proven.

## Steps

0. (Superseded as the first step by the update above: driver change first.)
1. Check whether noble-updates linux-firmware (candidate ubuntu3.1) carries a newer gc_12_0_1 MES/uni_mes (apt-get download + dpkg-deb -c, dates/sizes).
2. Maintenance window: back up /lib/firmware/amdgpu/gc_12_0_1_* , install only the R9700 MES files (package update or upstream linux-firmware files), update initramfs, reboot.
3. Verify the MES version (rocm-smi --showfwinfo), repeat the r9700-idle states A/C/D/E, and the RHP12 fresh probes (decode, 42K, 8 sessions) to confirm no regression.
4. Keep on success; otherwise restore the backup and reboot.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Loaded-idle R9700 power and gpu_busy_percent back to the no-process level (15 W, 0%); RHP12 probes within the gate; helpers unaffected.

## Effort & Risk



## Standards



## Acceptance Criteria

Loaded-idle penalty removed or the trial documented as ineffective and rolled back.

## Notes

Owner-run (root). Evidence: $BC_WORK_ROOT/runs/r9700-idle/result.txt; GPT R9700 idle-power deep dive 2026-11-02 (request req_main_2126dc6cc0a2). Do not reduce GPU_MAX_HW_QUEUES globally (helpers need separate queues).

## Change Log

- 2026-11-02T01:49:10.007322+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261102_014941_documented-the-r9700-idle-powe_8134
- 2026-11-02T01:49:45.639845+00:00 (updated-by): Updated: section:ledger-events


## Power programme status: 2026-11-10

R9700 GFXOFF kernel fix is measured (3 x 60-second phases): HIP-context idle at 15.0 W, 0% busy and 29 mV on patched kernel versus stock 20.6 W, 3% and 612 mV. This is a kernel prerequisite, not a manifest profile setting. PWR06 stays in progress for deep-idle/BACO investigation; no new R9700 clock or voltage default is justified. PWR08/PWR09 must not encode the kernel patch as a control.

## Cross-card isolation / XTX saving regression — 2026-11-11

GFXOFF remains **measured on the R9700 idle phase only**, at 15.0 W with a resident HIP context on kernel 7.0.13-gfxoff (R9700 MES 0x81 unchanged). It does **not** validate any XTX undervolt, SCLK, DPM, or profile change. The same patched kernel was in use when the XTX -125 mV recheck **failed** to save power in `runs/xtx-undervolt-ab/`. Neither the R9700's no-process/loaded-idle delta nor its firmware version can be used to infer XTX decode behavior. Record the actual XTX MES/SMU version and DPM/clock telemetry next time; do not assume the patch is causal for the XTX result. Cross-reference `docs/reference/power-tuning-evidence-2026-11.md` recovery queue. No new GPU validation and no change to the remaining R9700 long-run acceptance requirements.
