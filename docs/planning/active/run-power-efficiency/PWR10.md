---
id: PWR10
order: 0
plan: run-power-efficiency
state: pending
created-at: '2026-11-25T00:42:29.487523+00:00'
breadth: ''
skill: intermediate
created-by: lead
work: S
priority: P2
---

# Minimum power cap as a launch power profile (restored at stop), then a clock-cap A/B

## Description

A trial on 2020.11.4 (qwen27b, 3 decode requests and 3 prefills of 2,723 tokens per phase): the lowest writable power caps (272 W on the XTX and 6900 XT, 210 W on the R9700, against stock 303 W and 300 W) cut the XTX mean power from about 293 W to 267 W (peak about 305 W to 276 W) with decode 126.1 to 124.7 tokens a second (-1.1 percent, one run each) and prefill unchanged (2,492 to 2,511); the R9700 peak clipped from 228 to 215 W with no speed change. The cap resets at reboot and nothing applies it at start.

## Steps

1. A launch power profile (the launch files already carry a power_profile field) that writes each card's minimum cap at start and restores the previous cap at stop, only for the cards the launch uses. 2. A repeat of the trial with three runs per arm to bound the decode change. 3. A clock-cap A/B (SCLK maximum at 2700, 2400 and 2100 MHz on the XTX, clock offset on the R9700) for decode, prefill and watts under short leases, restoring stock after each arm. 4. Idle: the cap does not change idle power (8 to 16 W a card); a separate idle study (energy performance preference, PCIe link policy, background load) is host-level and needs root.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Launch and stop leave caps at their previous values; decode and prefill within the noise of stock; power measured with amd-smi.

## Effort & Risk

Small. Risk: the cap shifts other agents' measurements while applied; apply only under the GPU lease.

## Standards



## Acceptance Criteria

Minimum-cap profile selectable per launch with measured watts and speed, caps restored at stop.

## Notes

The undervolt (voltage offset) is not in effect on any card (0 mV) and was withdrawn in November: do not re-add it.

## Change Log

- 2026-11-25T00:42:29.487523+00:00 (created-by): Created by lead
- 2026-11-25T00:42:52.701764+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
