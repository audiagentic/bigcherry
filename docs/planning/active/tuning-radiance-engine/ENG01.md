---
id: ENG01
order: 0
plan: tuning-radiance-engine
state: pending
created-at: '2026-11-10T13:11:33.523674+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# R9700 engine per-op attribution with the live policy on: prefill (document, cold) and decode, with an unprofiled reference

## Description

With the live policy warmed the R9700 engine sets the pace (engine local 6.2 ms document / 6.8 ms cold per layer call against helpers 4.0-5.5 ms) and the decode step (~25 ms one-stream, ~47 ms six-stream) is ~75% outside the remote-MoE path. No profile separates attention, gated delta-net, engine-resident experts, synchronisation and host gaps for the default TP1 R9700 engine (existing per-op datasets are dual-XTX: runs/qwen27b-prefill-profile, runs/rad49-tp-prefill-prof). This item owns the artifact and the go/no-go for kernel work.

## Steps

1. Use the lab experiment {"engine": {"profile_ops": "on"}} (runs/rad49-tp-prefill-prof/exp.json). The measuring start must pass host admission: the 2026-11-10 attempts were refused at the n-gram table (47.68 GiB RAM table + 23.07 GiB other allocations + 4.59 GiB headroom against 74.61 GiB available, short by ~0.7 GiB while another job held host memory): run with --ngram-placement disk for the prefill profile, or when the host is quiet; note the plan key changes.
2. An --experiment start is the lab profile: pass --live-policy FILE (the derived settings of a production manifest) and check the launcher's live placement line says ON; record it.
3. Separate short probes: cold 27k prefill, 44k document prefill, one-stream decode, six-stream decode; collect rank-0 per-op device time, host-site time and idle wait.
4. Run the matched unprofiled configuration for throughput (per-op mode adds synchronisation and disables pass replay: its totals are diagnostic).
5. Rank operators by share of the engine layer call and decide which, if any, is worth a kernel project.
Not rocprofv3 around the engine (Radiance's AQL backend fails under it).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Per-op tables for the four probes with the policy state, measured-plan key, build id and n-gram placement recorded beside them; an unprofiled reference run; operator totals reconcile to the engine local time within the diagnostic overhead.

## Effort & Risk



## Standards



## Acceptance Criteria

A labelled R9700 engine breakdown for prefill and decode exists, the top operators are ranked with their share, and a written go/no-go for each candidate kernel project is recorded in this item.

## Notes

Source: gateway prior-work audit 2026-11-10; my first attempts: runs/engine-profile-policy-on.

## Change Log

- 2026-11-10T13:11:33.523674+00:00 (created-by): Created by agent
