---
id: RHP09
order: 20
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-10-31T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Stage 2-3: topology discovery, default-on calibration and the critical-path cost model

## Description

Owns measurement and cost-model validity for every layout; RCS10 owns default-on start-up integration and the tune cache, on one shared calibration schema and implementation. Discovery: stable logical device identity, architecture, usable expert budget after explicit reserves, eligible backends (engines/radiance/archs.toml), PCIe path, negotiated and measured link bandwidth, contention groups (extend tools/lab/radiance/host-link-bw.hip with concurrent and size-sweep measurements: one card, two XTX together, XTX + engine, tail + XTX, CPU BAR pushes + GPU reads; 4 KiB-21 MiB; p50/p95). Calibration per device, kernel variant and row/route bucket from replayed requests (replay-kernel-time.sh, --replay-sweep), helper stage times (act, gemm, write, queue/wake), separately GPU event time and host-observed completion. Cost model: piecewise, interpretable, layer completion = serial part + max over engine and helpers, built from engine-side timestamps (never cross-clock subtraction). Calibration on by default with an opt-out (owner decision 2026-11-01); it defers GPU work while another holder has the agent-lease gpu lease. Layouts keep declared constraints (roles or role preferences, inclusion/exclusion, memory caps, capability requirements); performance policy (head/tail shares, fixed ratios) becomes derived.

## Steps

1. Write the protocol, experiment or collector against the code as it is (radiance 1.14.0, remote-MoE protocol v19).
2. Land card-free tests first where the item has any logic; measure only with the gpu lease.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Card-free: synthetic PCI graphs, missing device, shared uplink, budget overflow, cache keys. Hardware: link repeatability within 10% idle; calibration picks today's W4H2 + 32-row configuration for XTX decode unless measurement says otherwise; predictor within 10% p50 / 20% p95 of held-out 24-row decode and 2048-row prefill layer completions; last-helper prediction correct on >=80% of well-separated cases.

## Effort & Risk

Depends on: RAD23. Added by the 2026-10-31 GPT review of this plan.

## Standards



## Acceptance Criteria

Cards, links and kernel costs are measured automatically on any supported layout; a known topology is a cache hit (<0.5 s); the predictor meets its held-out targets; no change to serving behaviour until RHP04 stage 3 uses it.

## Notes

Supersedes RPL01 (patching-runtime-placement). Budgets: cache hit <0.5 s, changed link 3-8 s, new GPU/kernel 5-15 s, cold first start 10-30 s cap. Plan source: GPT staged plan 2026-11-01.

## Change Log

- 2026-11-01T22:51:30.189099+00:00 (updated-by): Updated: section:title, order=20, section:description, section:validation, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20261101_225732_planning-a-staged-roadmap-for_4092
- 2026-11-01T22:57:47.166041+00:00 (updated-by): Updated: section:ledger-events
