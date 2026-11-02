---
id: PWR06
order: 0
plan: run-power-efficiency
state: pending
created-at: '2026-11-02T01:49:10.007322+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: S
priority: P3
---

# R9700 loaded-idle +4 W: MES firmware trial (gc_12_0_1 MES 0x81 -> >=0x8b) with backup and rollback

## Description

Measured 2026-11-02 (flashnext-4gpu, run r9700-idle): the R9700 (gfx1201, GC 12.0.1) idles at 15.0 W with no GPU process and 19.0 W with 3% phantom GPU activity whenever the radiance engine process exists (loaded idle, after a request, 65 s later), dropping back to 15.0 W when it exits. The helper XTX/6900 XT stay at 4-7 W (RAD20's park workaround). This matches the upstream R9700 HIP idle-activity issue (ROCm #5706/#6298) fixed in MES firmware 0x8b; this host runs MES 0x81 (Ubuntu linux-firmware 20240318 ubuntu2.27, gc_12_0_1_* files dated 2026-04-16). The 15 W unloaded floor is the board: graphics rail power-gated (VDDGFX 29 mV), memory clock at minimum, PCIe L1.1/L1.2 already enabled on the root-port-to-switch link; forcing FCLK to its minimum saves ~1 W. RAD20's earlier MES trial was GC 11.0.0 (XTX) and did not fix the XTX wait-packet poll; this is a different card and firmware.

## Steps

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

Owner-run (root). Evidence: /work/bigcherry/runs/r9700-idle/result.txt; GPT R9700 idle-power deep dive 2026-11-02 (request req_main_2126dc6cc0a2). Do not reduce GPU_MAX_HW_QUEUES globally (helpers need separate queues).

## Change Log

- 2026-11-02T01:49:10.007322+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261102_014941_documented-the-r9700-idle-powe_8134
- 2026-11-02T01:49:45.639845+00:00 (updated-by): Updated: section:ledger-events
