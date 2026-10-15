---
id: BCOP215
order: 215
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-14T19:04:43Z'
created-by: agent
priority: P2
work: S
---

# QFP08: retire duplicate draft-prefill pipeline; reconcile 1359 switch

**Discovery / change:** QFP08 still requested target/draft chunk overlap, but validated 1348 already defers draft catch-up and validated 1359 already submits target chunk k+1 before collecting k through two pinned NextN buffers and per-physical-child backend events. QFP08 is moved to completed as `SUPERSEDED_IMPLEMENTED`. Historical 1359 README and completed QFP50 prose still called the flag default-off despite source `getenv()==nullptr || atoi()!=0` (unset=on, =0=off); current-state corrections prevent invalid A/B controls. No new kernel/patch or performance claim.

**Authoritative owners:** 1348 snapshot/pending/flush, QFP50/1359 fenced NextN and prefill pipeline; QFP27/1326 scheduler-input lifetime; FMTP02/03 generation-only ahead; QFP15 draft-seed attribution. QFP08 has no remaining implementation ownership. No new scheduler, allocator, ring, cache, event API, telemetry or hardware queue.

**Existing measured evidence:** b11474 Flash-Next 1348 +5.4/+6.3/+9.7% prefill at 8K/24K/98K versus native hook; 1359 +6.0/+7.4–10.7/+10.0% versus the then-default 1348-only path; +15.7% at 24K versus native hook order. Long >=10ms gaps 81 -> 13, idle 17.4 -> 12.9%; greedy identity and neutral decode in reported runs. Historical, not new b11552 data; controls differ and gains are not additive.

**Novelty/12-hour:** selected feat/rad03-gpu-lanes from independent `e4d95b98fadef068fc6e1058dec3ec61b29e768b` (RAD05, 2026-10-14 18:54 UTC). QFP08 last independent update 2026-10-05; 1348/1359 last 2026-10-11. No same-path independent activity or queued run found in preceding 12h. Excluded active RAD02/03/04/05 helper/mailbox/routing/prefill; recent BCOP205–214 audited other mechanisms.

**External/correctness:** upstream llama.cpp #26827 (open) proposes MTP multi-ubatch KV synchronization; #30459 (merged 2026-10-14, after b11552) adds independent draft n_ubatch. SGLang-JAX EAGLE FULL prefill hidden-state capture is a semantic reference, not an AMD port. Do not conflate either with a new 1359 gain.

**Unresolved action:** None under QFP08. Only QFP50/1359/1348 may investigate a **new measured** residual gap on the exact current pin with full per-child event/host-buffer and target/draft work accounting, multi-request/cancel/shift/graph/MTP correctness. Reject duplicated pipeline work or <3% exposed E2E prefill/TTFT opportunity. Reconcile any future source-default changes in the owning patch and launch recipes, not a new flag.

**Validation:** 10/10 pinned source assertions; 7/7 host-only lifetime/shift/byte/ceiling tests. No repository pytest, HIP build, patch application, GPU test or new benchmark. **Terminal:** `SUPERSEDED_IMPLEMENTED`; later owner-specific findings use `OWNER_RECONCILIATION` or `NO_NEW_EXPOSED_GAIN`.
