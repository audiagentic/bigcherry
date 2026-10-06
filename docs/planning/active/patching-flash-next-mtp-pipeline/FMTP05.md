---
id: FMTP05
order: 0
plan: patching-flash-next-mtp-pipeline
state: pending
created-at: '2026-10-04T00:48:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Control ahead depth from hit probability and overlap economics

## Description

Choose how much future MTP work to execute from measured reuse probability and overlap cost. Reuse 1268 as the sole source of current front depth.

The controller must include bounded **probe rounds**: an always-off warmup cannot estimate bridge-match probability or ahead runtime.

## Steps

1. Maintain per-sequence observations: front/accepted lengths, full-front acceptance, bridge match, front source (fresh/promoted), MTP overlap work, target device interval, target sync wait, overhang, promoted/flushed tokens and serial fresh-draft cost.
2. Separate MTP work type:
   - fresh-front continuation cost;
   - promoted-front forced-replay + continuation cost.
   These can differ materially.
3. Define `p_hit = P(full front) * P(bridge match | full front)`.
4. Estimate wall-clock value, not GPU work. Hidden work is cheap; overhang, target contention and scheduling overhead are costs.
5. Use bounded exploration:
   - initial warmup executes ahead on a small fixed fraction or bounded first-N eligible rounds;
   - when policy is off, periodically run a low-frequency probe after cooldown so changed workload conditions can be detected;
   - probes obey the same hard `mtp_ahead` cap and never alter target semantics.
6. First action set: `{0, min(effective_front_depth, mtp_ahead)}`. Add intermediate depths only from evidence.
7. Apply hysteresis/cooldown; repeated large overhang may disable immediately.
8. Never feed ahead misses into 1268's accepted-token signal.
9. p-min applies normally to newly sampled bridge/tail work; do not weaken it to improve hit rate.
10. Emit bounded aggregate decision/probe telemetry.

## Detailed Solution & Technical Design

Per-attempt wall-clock model after target submit:

```text
T_target      = target device interval remaining after submit
T_mtp         = continuation OR promoted-front replay+continuation
T_overhang    = max(0, T_mtp - T_target)
T_hidden      = min(T_mtp, T_target)
```

A hit avoids the next round's **serial fresh-front draft**. On a promoted child round, MTP replay still occurs, but it is scheduled under target verification, so the relevant saving is movement of that work off the serial pre-target path.

Use measured data rather than assuming `T_ahead == T_fresh`:

```text
EV ~= p_hit * T_serial_front_avoided
      - E[T_overhang]
      - measured_target_contention
      - host/scheduler_overhead
      - probe_cost_when_off
```

Suggested observation:

```cpp
struct bigcherry_mtp_ahead_observation {
    int32_t front_len = 0;
    int32_t accepted = 0;
    bool front_was_promoted = false;
    bool bridge_checked = false;
    bool bridge_match = false;
    int64_t serial_front_us = 0;
    int64_t target_submit_us = 0;
    int64_t target_device_us = 0;
    int64_t target_sync_wait_us = 0;
    int64_t mtp_overlap_us = 0;
    int64_t overhang_us = 0;
    bool was_probe = false;
};
```

### Probe policy

Without probes, an off controller cannot learn `bridge_match` or `T_mtp` and can become permanently stuck off.

Initial conservative policy:
- first 16-32 eligible rounds: probe every 4th round;
- after warmup: normal EV decision;
- while off: one probe after each cooldown window (e.g. 32-64 rounds), with exponential backoff if repeatedly negative;
- any repeated severe overhang disables non-probe attempts immediately.

Constants are qualification hypotheses, not production defaults.

### Interaction with 1268

```cpp
const int front_depth = common_speculative_effective_n_max(spec, seq_id);
const int hard_cap = params.mtp_ahead;
const int ahead_depth = ctrl.choose(front_depth, hard_cap);
```

1268 owns how many current tokens are worth asking the target to verify. FMTP05 owns only how much future/replay work to move under that verification window.

## Files

- pure BigCherry controller helper or `common/speculative.*`
- `tools/server/server-context.cpp` observation wiring
- table tests
- evidence parser if useful

## Validation

Table tests:
- high hit + fully hidden => enable;
- low full acceptance => off;
- low bridge match => off;
- large replay/continuation overhang => off;
- one bad sample does not flap;
- hard cap respected;
- off state eventually probes;
- workload changes from negative to positive and probes allow recovery;
- sequence reset clears controller history.

Hardware:
- ahead=0, fixed ahead, adaptive ahead;
- compare cold/flush rounds and promoted-hit rounds separately;
- varied predictable/unpredictable prompts.

## Acceptance Criteria

- Controller cannot alter target acceptance/emission semantics.
- 1268 remains sole front-depth controller.
- Policy can both disable on negative EV and recover from off when workload changes.
- Probe rate/work is bounded.
- No unbounded queue/state.


## 2026-10-06 ownership note: WHIRL adaptive speculation

WHIRL v0.1.3 independently validates the value of measured cost/acceptance economics, but its front-draft controller belongs to PRBE52/1255/1268, not FMTP05. FMTP05 must continue to consume `common_speculative_effective_n_max()` as an input and adapt only **future/ahead overlap work**. It must not add another front-depth controller or n-gram proposer selector.

The useful transferable pattern is the policy shape already present here: measured wall-clock value, hysteresis and bounded probes rather than acceptance-only heuristics. Keep the two decisions separate because front-depth changes target verification batch size while FMTP05 changes work scheduled under an already-submitted target verification window.

Source: https://github.com/tsaipifong/whirl-llm/blob/main/src/model/spec.cpp


### Execution boundary after WHIRL audit

This ownership note is architectural only. FMTP02-FMTP07 remain paused by the recorded owner decision in FMTP03; BCOP38/PRBE52 work does **not** resume this pipeline.

Do not implement, benchmark or tune FMTP05 merely because PRBE52/1255 is being reconciled. Resume FMTP05 only after an explicit FMTP resume decision and the prerequisite FMTP03 path is again hardware-qualified.

WHIRL front-depth calibration data is not automatically valid FMTP05 training data. FMTP05 needs its own observations of hidden overlap, overhang, bridge/full-front hit probability and target contention. It may consume the effective front depth selected by PRBE52, but must not reuse PRBE52's E/T controller state or infer ahead-work value from front-depth acceptance statistics alone.
