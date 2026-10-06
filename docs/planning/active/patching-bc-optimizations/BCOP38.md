---
id: BCOP38
order: 38
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-06T11:07:00+11:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# Disposition WHIRL adaptive speculation through PRBE52

## Description

2026-10-06 optimisation-audit disposition for WHIRL v0.1.3 adaptive speculative decoding. PRBE52/1255/1268 remain the authoritative BigCherry front-draft-depth owner; FMTP05 remains ahead-overlap economics only.

## What changed/discovered

- BigCherry 1255 currently uses acceptance-driven fixed climb/drop thresholds; 1268 wires that controller per sequence.
- WHIRL's `pickDrafts()` uses conditional acceptance plus measured cycle time to maximize expected tokens/time, with 3% hysteresis and bounded neighbour probes.
- WHIRL reports workload-dependent optimal depth and an auto policy close to the best fixed depth; this is external gfx1201 evidence, not a BigCherry measurement.
- llama.cpp already supports mixed MTP+n-gram proposers. Existing upstream reports show independent combination is workload-sensitive and can be neutral/slower, so a new BigCherry n-gram implementation is not justified.

## Already acted upon

- 1255 provides the single pure adaptive front-depth controller.
- 1268 provides the single runtime wiring/configuration surface.
- 1317/1318 provide speculative/MTP timing telemetry suitable for a cheap trace-replay discriminator.
- PRBE52 already requires the 1210 greedy-exact prerequisite and upstream #29924/equivalent before mixed-drafter interpretation.
- FMTP05 already owns a separate measured-EV/probe controller for future MTP overlap and must not absorb front-depth ownership.

## Unresolved action

Replay existing 1317/1318 round traces through fixed, current-1255 and WHIRL-style E/T policies. Only if E/T reduces policy regret >=5% should 1255 internals be replaced and hardware-qualified. No new controller, CLI option or speculative scheduler is permitted.

## Terminal disposition

- **Promote:** offline >=5% regret reduction, then gfx1201/gfx1100 hardware shows >=5% median effective-TG improvement over current adaptive, exact greedy IDs, and <=2% held-out regression.
- **Reject/close:** offline <5% advantage or hardware gate fails; retain 1255 and preserve WHIRL as external evidence.
- **Mixed MTP+n-gram:** use upstream implementation as control after #29924. Do not create local proposer arbitration unless a measured residual >=5% remains.

## Provenance

- https://github.com/tsaipifong/whirl-llm
- https://github.com/tsaipifong/whirl-llm/blob/main/src/model/spec.cpp
- https://github.com/tsaipifong/whirl-llm/blob/main/docs/guide/en/speculative-decoding.md
- https://github.com/ggml-org/llama.cpp/issues/24507
- https://github.com/ggml-org/llama.cpp/issues/23184
