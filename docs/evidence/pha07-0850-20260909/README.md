# PHA07 — ordered speculative trace qualification

Date: 2026-09-09  
Host: Brutus (`build-server`)  
Patch: `0850_ordered_speculative_trace`  
Upstream pin: `b10705` (`2578138397d7b422bb0e160efdd429976c55fb55`)  
Architecture: `gfx1100`, GPU0 (`HIP_VISIBLE_DEVICES=0`, `ROCR_VISIBLE_DEVICES=0`)  
Role: experimental hardware/schema evidence only

## Build

The maintained campaign builder was run in the isolated `/home/audumla/pha07-work`
clone after restoring the missing host-local `b10705` tag:

```text
PYTHONPATH=tools python3 -m bigcherry build \
  --lane llama-native:stock:linux-multi --arch gfx1100 \
  --binary-relative-path bin/llama-server \
  --experiment hi166-ordered-trace-only \
  --run-id pha07-0850-upstream-20260909
```

Result: `build_plan_id=4bd62c7fba032f095c698730f677cf1a`.  The resulting
`llama-server` SHA-256 is
`249618ad89ba98aa2e8c658022f48b9ff5f17e03194b5fcba54139e5e7908ce6`.

An attempted framework-composed lane (`bigcherry:control` plus the same
experiment) was rejected by exact source-materialization admission: existing
0100/0700 configuration evidence is bound to the base composition and does not
cover an additive 0850 overlay. This fail-closed result is retained; no stale
evidence escape hatch was used.

## Maintained server-bench and trace check

The built server was launched with the production-shaped 9B Q6_K settings and
`--spec-type draft-mtp --spec-draft-n-max 5`. The maintained
`bench/run_bench.py --bench-type server-bench --bench-configs tg128` completed
cleanly at 184.22 tokens/s (five repetitions; liveness/trace evidence, not a
performance admission claim).

A direct `/completion` request against the same server returned an ordered
`timings.draft_trace` with 13 verification steps:

```json
{
  "draft_n": 56,
  "draft_n_accepted": 18,
  "draft_trace": [[5,2],[5,1],[5,0],[5,0],[5,4],[5,3],
                   [5,2],[5,1],[5,1],[5,3],[3,0],[2,0],[1,1]]
}
```

The server log records MTP initialization and `18 accepted / 56 generated`.
The raw response digest is
`52a284563e2e0d5052f62e0e95fb71f65fe5c812e6b3018e994b4d14be578e07`.

This proves current-pin application, production-shaped server schema emission,
and a real MTP ordered trace for the experimental composition. It does **not**
promote the patch, prove dual-XTX parity, or retire HI166; those remain gated on
the exact normal composition and the required dual-XTX/native-control/replay
qualification.
