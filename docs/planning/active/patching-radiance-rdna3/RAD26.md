---
id: RAD26
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-27T02:15:24.188732+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# moe-remote: large-lane GPU-to-CPU request wake (host callback / graph host node / HSA interrupt) harness

## Description

The large lane yield-spins on its request word during bursts because the engine publishes from GPU code, which cannot wake a futex. Measure candidate wake paths in a two-process harness without model load: hipLaunchHostFunc -> eventfd, hipGraphAddHostNode -> eventfd, HSA interrupt signal, against the current spin. Large requests are few (prefill steps of 256+ rows), so tens of microseconds may be acceptable; per-request small-lane callbacks are rejected (engine stream stall per MoE layer).

## Steps

1. Harness: publish->wake latency p50/p99, engine stream stall, CPU use, AQL packets (barrier check for graph host nodes).
2. If one is cheap: implement for the large lane only; A/B prefill and mixed workloads.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Harness numbers recorded in the RAD20 research record; end-to-end prefill within noise before adoption.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

HSA IPC blocked waits already measured at 100% CPU (FINDINGS); interrupt relay spun before blocking. Do not repeat unchanged.

2026-10-27 harness result (rad26-wake-bench, gfx1201 engine stream; runs rad26-wake-bench-1027*): spin round trip p50 6-8 us at 100% helper CPU; hipLaunchHostFunc -> futex wake p50 25-67 us (p99 49-222 us) at 0.4-0.7% CPU but holds the engine stream ~107 us per callback; on a side stream behind an event (sidefunc) p50 29-70 us, p99 73-221 us, 0.5-1.8% CPU, engine stream cost 4.5 us per callback. Design: in publish_launch (moe_mailbox_ops.hip) the host knows rows; for a large-lane request (rows above the largest small class) record an event after the publish kernel, make a per-device side stream wait on it, and hipLaunchHostFunc on the side stream to call notify(host header); the large lane waits with wait_for on Header::notify instead of yield. Check first: whether these launches are captured into HIP graphs (MTP/multi-stream decode steps of 9-24 rows also use the large lane), callback context lifetime through close_mailbox, and stop/fault wakes. Gate: prefill (pp1024/4096) and 2-6 stream decode A/B within noise, helper CPU down, kill test PASS.

2026-10-27 FIRST HARDWARE RUN FAILED (branch feat/rad26-large-lane-wake 74ba88796, not merged): the first large-lane step (a 4-stream decode step above 8 rows) ran under HIP stream capture; the callback captured on the engine stream never woke the helper (helper served 0 large-lane requests), the engine hung, every later request failed, and the open burst left small-lane waits armed (~96 W). A large lane that sleeps only on the callback is unsafe while the engine captures large steps into graphs. Before retrying: find which steps radiance captures and whether ROCm graph host nodes run and wake reliably on replay (harness: capture publish+hostfunc into a graph, replay N times); or keep a wake path that does not depend on the callback for captured steps. Large lane stays on the yield loop until then.

2026-10-27 graph test (rad26-wake-bench graph mode, run rad26-wake-bench-10272150): a publish + hipLaunchHostFunc captured into a HIP graph and replayed 2000 times woke the futex waiter on every replay in all 3 runs (launch->ack p50 33-50 us, p99 129-144 us). So ROCm graph host nodes work; the engine hang points at how radiance captures/replays (e.g. graph update or clone dropping the node, or capture without replay). Next: count callbacks fired in the engine (log at teardown) and log the helper's wakes, in one short run with a multi-stream step.

## Change Log

- 2026-10-27T02:15:24.188732+00:00 (created-by): Created by agent
- 2026-10-27T08:31:10.859985+00:00 (updated-by): Updated: section:notes
- 2026-10-27T10:21:28.314791+00:00 (updated-by): Updated: section:notes
- 2026-10-27T11:16:32.240768+00:00 (updated-by): Updated: section:notes
