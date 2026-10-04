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

# Control ahead depth from acceptance and overlap economics

## Description

Avoid turning useful overlap into wasted work or join stalls. Add a small pure controller that decides whether/how far to continue ahead using measured full-front acceptance, bridge success, target-verification time and continuation overhang. Reuse 1268 as the source of front-draft depth; do not create a second front-depth controller.

## Steps

1. Maintain per-sequence rolling/EMA observations:
   - front length and accepted length;
   - full-front acceptance;
   - bridge match conditional on full acceptance;
   - ahead requested/generated/promoted/flushed tokens;
   - front draft time;
   - target verify time;
   - ahead time;
   - join overhang;
   - observed next-round fresh-draft cost when a promoted tail was unavailable.
2. Define reusable-tail probability:
   `p_hit = P(full front) * P(bridge match | full front)`.
3. Estimate value in wall-clock units, not raw GPU work. Fully hidden work on the separate MTP card is cheap; overhang and cross-device contention are not.
4. Keep policy disabled for a warmup period and when evidence is insufficient.
5. First controller action set: `{0, effective_front_depth}`. Add intermediate depths only after traces prove partial ahead depth is useful.
6. Use hysteresis/cooldown to prevent rapid enable/disable oscillation.
7. Read effective front depth from 1268 when adaptive front depth is enabled; otherwise use the fixed MTP `n_max`.
8. Keep an independent hard cap `mtp_ahead` from FMTP01. Policy may select less, never more.
9. Treat p-min early stop as natural work bounding; do not weaken p-min to improve hit rate.
10. Emit bounded aggregate telemetry for controller decisions/reasons.

## Detailed Solution & Technical Design

The correct optimization target is the serial critical path. A simple estimate is:

```text
T_hidden      = min(T_ahead, T_verify)
T_overhang    = max(0, T_ahead - T_verify)
T_hit_saving  = min(T_ahead, T_next_fresh_draft)

EV ~= p_hit * T_hit_saving
      - T_overhang
      - measured_contention_penalty
      - scheduling_overhead
```

The exact accounting should be fitted from FMTP01/F03 measurements. Avoid double-counting overlap: if target and ahead run concurrently and ahead finishes before verification, the miss cost is near zero wall-clock even though MTP GPU energy was spent.

Suggested pure controller:

```cpp
struct bigcherry_mtp_ahead_observation {
    int32_t front_len = 0;
    int32_t accepted = 0;
    bool bridge_checked = false;
    bool bridge_match = false;
    int64_t front_us = 0;
    int64_t verify_us = 0;
    int64_t ahead_us = 0;
    int64_t overhang_us = 0;
    int64_t fresh_draft_us = 0;
};

struct bigcherry_mtp_ahead_controller {
    int samples = 0;
    double ema_full = 0.0;
    double ema_bridge = 0.0;
    double ema_verify_us = 0.0;
    double ema_ahead_us = 0.0;
    double ema_fresh_us = 0.0;
    int cooldown = 0;

    int32_t choose(int32_t effective_front, int32_t hard_cap) const;
    void observe(const bigcherry_mtp_ahead_observation & o);
    void reset();
};
```

Keep `choose()` deterministic and independent of server objects. This makes table-driven tests possible and mirrors the testable design of NRO06/1268.

### Interaction with 1268 adaptive front depth

1268's controller changes how many **current** MTP tokens are worth asking the target to verify. FMTP05 changes how much **future** MTP work is worth attempting during that verification. The two decisions are related but not identical.

Use:

```cpp
const int front_depth = common_speculative_effective_n_max(spec, seq_id);
const int ahead_depth = ahead_ctrl.choose(front_depth, params.mtp_ahead);
```

Do not feed ahead misses back as rejected draft tokens to 1268; that would corrupt the acceptance signal.

### Conservative startup

A reasonable first policy:
- warmup >= 32 completed verification rounds;
- enable only when EMA `p_hit` and measured verify/ahead timings imply positive EV with margin;
- once disabled, cooldown 16 rounds before reconsidering;
- if join overhang exceeds a configured fraction of verify time repeatedly, disable immediately.

These values are starting hypotheses, not promotion defaults; hardware evidence owns final constants.

## Code Samples & Guidance

Table-test examples:

```cpp
// high reuse, fully hidden => enabled
observe(full=true, bridge=true, verify_us=600, ahead_us=350, fresh_us=350);

// low reuse => off even if hidden
observe(full=false, bridge_checked=false, verify_us=600, ahead_us=350);

// high reuse but 500 us overhang => reduce/off
observe(full=true, bridge=true, verify_us=300, ahead_us=800, fresh_us=350);
```

## Files

- BigCherry-local pure controller helper or `common/speculative.*`
- `tools/server/server-context.cpp` observation wiring
- unit tests
- optional experiment evidence parser

## Validation

Offline table tests:
- high-hit/fully-hidden => enable;
- low full-accept => disable;
- high full-accept/low bridge-match => disable;
- high-hit/large overhang => disable/reduce;
- hysteresis prevents one-sample flapping;
- reset isolates sequences;
- hard cap always respected.

Hardware:
- compare ahead=0, fixed ahead=N, adaptive ahead;
- code, structured output, prose and deliberately low-acceptance prompts;
- verify adaptive spends little steady-state time with negative measured EV.

## Effort & Risk

M / medium. Semantics risk is low because policy only chooses speculative work amount; performance policy can still overfit without varied workloads.

## Acceptance Criteria

- Controller cannot change target acceptance/emission semantics.
- 1268 remains the sole owner of front-depth adaptation.
- Ahead controller turns off on negative-EV workloads.
- No unbounded work or queue growth.
- Fixed-ahead mode remains available for diagnosis/benchmarking.
