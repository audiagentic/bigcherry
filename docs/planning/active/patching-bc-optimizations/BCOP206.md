---
id: BCOP206
order: 206
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-14T09:13:13Z'
created-by: agent
priority: P3
work: S
---

# QFP53: retire coarse KV graph reuse and route residual prefill gaps to QFP48

**Discovery/change:** `1362_kv_pad_prompt` was rejected after a real b11552 ABBA: 24/40 graph reuses and 7.1 -> 3.0 ms host graph build did not improve Flash-Next prefill; shared input wait rose 3.3 -> 17.0 ms. Dense 27B slowed and lost greedy bit identity at 32K. QFP53 still appeared active and 1362 README still claimed `untested`/advertised an obsolete experiment. Reconcile by moving QFP53 to completed/superseded and correcting README.

**Authoritative owners:** QFP53 is terminal evidence only. QFP48 owns residual per-batch gap attribution; QFP50/1359 owns target queue-ahead; QFP16/27 owns host input lifetime; QFP46 owns graph/event invariants. Do not add a graph cache, scheduler or new KV padding configuration.

**Already acting:** 1359 is validated/default-on and already queues the next batch. 1362 `patch.py`/`patch.toml` and `SUMMARY.md` already say rejected. Upstream #29958 stabilizes hybrid graph shape; merged #30453 repairs Meta max-slot graph allocation and is a prerequisite for any future persistent graph path. No independent QFP53/1362 PR, commit or queued experiment within the prior 12h (last relevant PR #182 closed 2026-10-11). RAD03 remote helper and RAD04 two-half prefill are protected and untouched.

**Unresolved action:** QFP48 may perform one existing-tool warm same-binary per-batch HIP API/kernel join at 8K/24K/98K, separating exposed GPU idle from overlapped host graph work. At 1465 t/s, a +3% gate requires ~10.18 ms genuinely removable per 512-token batch. No new hardware lane was queued.

**Terminal gate:** `NO_EXPOSED_GAIN` if <3% removable E2E wall time; `LIFETIME_UNPROVEN` if input/event generations cannot be fenced; `NUMERIC_DRIFT` for greedy/logits/MTP changes; `PROMOTE_UNDER_OWNER` only after a named existing owner delivers CI95-low >=3% prefill/TTFT and <=1% decode regression with repeated-request/ubatch/cancel/long-context checks. QFP53/1362 stays retired absent material new independent evidence.

**Validation:** 10/10 repository-source assertions, 5/5 host critical-path oracle tests and Python compilation. No repository pytest, patch composition, HIP build or new GPU benchmark. Historical numbers are not new measurements.