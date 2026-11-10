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
