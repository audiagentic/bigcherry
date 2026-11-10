---
id: PWR04
order: 0
plan: run-power-efficiency
state: pending
created-at: '2026-10-27T10:13:52.881547+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Per-device power settings in environment config, a privileged apply/restore helper and a guarded controller

## Description

Power settings found by PWR02/PWR03 (XTX voltage offset, optional clock limit, R9700 cap/offset, CPU EPP) must be configuration, never code: per device in the environment settings' device inventory (keyed by locator, so each card has its own calibrated value), with tooling that applies them. Each setting records how it was validated (offset, stability soak, output identity, measured W and tok/s) and when.

## Steps

1. Schema: optional power block per [[host.*.devices]] entry (vddgfx_offset_mv, sclk_max_mhz, power_cap_w, profile) and a host cpu block (epp); validated by environment.py; absent = driver defaults.
2. A small privileged helper (one-time root install) that applies only those configured values to the listed devices, reads them back, and restores defaults on stop or lease expiry; unprivileged callers talk to it over a local socket.
3. Start-up host check warns when configured settings are not applied or the RAPL read rule is missing.
4. Later: coarse phase-aware selection (idle/decode/prefill) with dwell time and automatic rollback on regression (decode/prefill > ~1-2%, any GPU error).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Unit tests for the schema and helper command building; on the bench host the configured offset is applied and read back, a kill/stop restores defaults, and a PWR01 run with the setting matches the sweep's numbers.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner rules: settings configurable, not hard coded; hosts by role; derive over configure where possible (calibrate per device x build, cache). Prior art: LACT daemon model, amdgpu-clocks (PWR02 notes).

POWER REVIEW 2026-11-10 (gateway, verified against the code): DELIVERED: bigcherry gpu-power status|apply|restore|grant|revoke, per-card configuration ([host.*.devices.power]), launcher --gpu-power apply/restore, host-check. NOT delivered: named power profiles, lease-fenced ownership, durable crash recovery (the saved state is a per-user JSON file with no lease/manifest identity), strict failure (an apply error restores and then continues serving with the cards as they are), apply to the layout's cards only (the CLI selects every configured card), evidence validation (a non-empty string), the systemd boot-unit installer the ledger mentions (tools/admin/install_bigcherry_gpu_power.py is absent). Design: PWR07 (portable capability probe + receipt + safe-failure policy first), then named profiles resolved against per-card calibration (config/power-profiles.toml: default preserves, efficiency, quiet, max-throughput), launcher-level selection with a receipt keyed by the manifest digest in stage 1 and a versioned manifest extension in stage 2 (the v1 contract is closed), a host controller with lease fencing for ownership (hardest question: lease-fenced, crash-recoverable GPU power transactions; Fable brief in runs/rad49-gpt-review/followups, power review section 9). Depends on the lease item for detached servers.

## Change Log

- 2026-10-27T10:13:52.881547+00:00 (created-by): Created by agent
- 2026-11-10T13:12:52.379299+00:00 (updated-by): Updated: section:notes


## Update 2026-11-10 (GPT power manifest review, req_main_eb983900e21b)

Remaining work split: PWR08 (named profiles in the manifest, v2) and PWR09 (Python transactional controller, launcher adapter, strict failure). The environment-to-hardware path in gpu_power/power_cmds applies every inventory card one at a time with no whole-layout preflight and the launcher continues after a failed apply: both are replaced by PWR09, not extended.
