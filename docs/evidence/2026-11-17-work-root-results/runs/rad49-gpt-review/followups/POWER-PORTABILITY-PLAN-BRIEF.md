# Plan the first implementation slice of BigCherry power management so it is PORTABLE and can never fail a start on hardware it does not support

Owner requirement (2026-11-10): "make sure our power work does not fail on other processors or on an AMD platform - it should work on what we can test or assume will work -
plan it with gpt and implement." Your earlier review (req_main_49149ce2ce08, power review) is the base; this request turns it into an implementation plan for the FIRST slice, which I will then implement.

Current code to build on (verify): tools/bigcherry/hardware/gpu_power.py (amdgpu sysfs OverDrive: vddgfx offset, sclk max, power cap; snapshot/restore), hardware/power_setup.py (udev/group), cli/power_cmds.py
(gpu-power status|apply|restore|grant|revoke), core/environment.py (DevicePower, [host.*.devices.power]), cli/host_check.py (RAPL readability, power writability), tools/lab/radiance/launch-flashnext-best.sh
(--gpu-power, lines ~378-389, best-effort apply/restore), the helper's runtime power_save and cpu affinity (moe-remote worker_cpu.h), the manifest contract (v1 closed).
This host: Intel CPU (intel_pstate, governor powersave, EPP balance_performance, turbo on, RAPL not readable without root), 4 AMD GPUs (gfx1100 x2, gfx1201, gfx1030) on amdgpu, Linux container with host sysfs visible, no root.
Other hosts BigCherry may meet (also the build server and a Windows dev PC, see the slice-workflow skill): AMD Ryzen/EPYC CPUs (amd_pstate active/passive/guided or acpi-cpufreq, amd_energy or powercap/RAPL-like
interfaces, different EPP names), Intel with intel_pstate passive/disabled or acpi-cpufreq, ARM/other CPUs with cpufreq or none, virtual machines/containers with no cpufreq or sysfs, read-only sysfs, missing
privileges, different GPUs (amdgpu without OverDrive/pp_od_clk_voltage, older/newer RDNA/CDNA, APUs, NVIDIA, no GPU), Windows (no sysfs at all).

## Requirements for the first slice (challenge these)
1. A power CAPABILITY PROBE, read-only, that never raises: for the CPU (vendor, scaling driver, governors available/current, EPP available/current, turbo/boost state, RAPL/energy counters readable?, C-state info if present) and for each GPU
   (driver, which OverDrive/power files exist and are readable/writable, current values, hwmon power/energy sensors, supported ranges) returns a structured result with per-item status
   {supported, unsupported(reason), unreadable(reason), not_permitted(reason)} and never a bare exception. Works on Linux without root; returns "unavailable: not Linux/no sysfs" elsewhere.
2. A POWER RECEIPT written into every run's output (and printable by a launcher line): the observed state of every layout device and the host CPU at start and at stop (what is, not what was requested), plus requested/applied/restored when a profile
   is used. Absent data is recorded as unavailable with the reason, never omitted or guessed.
3. FAILURE POLICY: no request -> never touches hardware and never fails a start (receipt only). An EXPLICIT request for a setting on a device where the probe says unsupported/not_permitted -> refuse the start with the reason (no silent
   fallback, no partial apply), checked BEFORE any write. Probe failures on an unrequested path are receipts, not errors.
4. Apply only to the devices of the selected layout (not every configured card), with the pre-change snapshot written durably before the first write, exact readback, rollback of already-written devices on any failure, and restore at stop.
5. Testable without hardware: a fake sysfs root (tmp tree) fixture library for Intel/AMD/no-cpufreq/read-only/missing-OverDrive/no-GPU/permission-denied cases; unit tests for each; plus a read-only test on THIS host. Anything we cannot test (AMD Ryzen, Windows, NVIDIA)
   must be handled by the same capability logic and listed as "assumed, covered by fixtures only".
6. No new compatibility shims, no hand-picked constants (ranges come from the device's own files), no Linux-only imports at module import time (the Windows dev PC must still import the package and get "unavailable").
7. Manifest: stage 1 keeps v1 closed; the receipt is a run artefact keyed by manifest digest; say where it is written and how benchmark results reference it.

## Deliverable (concise, implementation-ready)
(a) Where each piece lives (new modules/functions, file paths), data classes and the JSON receipt schema. (b) The probe's exact sysfs/hwmon/cpufreq paths per vendor/driver and what to do when absent. (c) The failure policy as a decision table.
(d) The fixture-tree test plan with named cases and what each asserts. (e) The launcher integration (what changes in launch-flashnext-best.sh and the benchmark harness, and the order of operations). (f) Risks and what you would NOT do in this slice.
(g) A sequence of small PRs. Read-only; cite file:line for existing code you rely on.
