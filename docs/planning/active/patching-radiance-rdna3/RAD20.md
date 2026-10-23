---
id: RAD20
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-23T00:50:15.230044+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
---

# moe-remote helper: replace the lane threads' yield-polling with GPU-side waits and blocking syncs

## Description

The helper's lane threads (engines/radiance/components/moe-remote/r3_moe_worker.cpp) still poll with std::this_thread::yield(): the large lane waits for the engine's request_sequence word (written by the engine GPU), and each small lane waits for its own done_sequence (written by the helper's own GPU). These loops predate R3_MOE_WORKER_COMBINE and run in every configuration; they burn CPU next to the engine's single-threaded host loop, which decode is sensitive to. A GPU-written word has no CPU waker, so the fix is structural: move the wait onto the GPU (a pre-queued wait kernel on the lane's stream, s_sleep, checking helper_fault and a bound, as the engine's mailbox wait kernel does) and let the CPU block in a hipEventSynchronize / hipStreamSynchronize under hipDeviceScheduleBlockingSync, so it sleeps until the device signals completion. From the GPT synchronisation review (2026-10-23): no continuously polling CPU threads, every blocking wait cancellable and bounded, fault paths independent of joins.

## Steps

1. Large lane: pre-queue a device wait kernel for request_sequence >= next (with helper_fault and stop checks, bounded) ahead of the sized lane launch; the CPU blocks on an event recorded after it, then reads rows and launches. Or keep a one-deep pre-queued wait so the sized launch follows immediately.
2. Small lanes: replace the done_sequence yield loop with blocking event syncs on the pre-queued requests (events recorded after each lane request).
3. Set the helper's device flags to blocking sync (hipSetDeviceFlags(hipDeviceScheduleBlockingSync)) and confirm the wake-up latency is acceptable for decode (small lane, 4 rows).
4. Stop: request_stop must also release any pre-queued device waits (store the words they wait on, or have them check helper_fault/stop in mapped memory).

## Detailed Solution & Technical Design

Grounding (2026-10-23): the lanes already wait on the device -- enqueue_lane (moe_remote_gpu.h) queues hipStreamWaitValue32(lane.request_sequence >= seq) on the lane stream, a command-processor wait, before the request's kernels. The CPU loops exist only for bookkeeping:
- small lanes (r3_moe_worker.cpp ~551): yield-poll own lane.done_sequence to re-queue (depth 4) and collect timing samples;
- large lane (~497): yield-poll lane.request_sequence because the launch is sized to the request's rows, unknown until it arrives;
- startup: helper 1 sleep(1)-polls for helper 0's segment file (~327) and for helper_count (~352).

Design:
1. Small lanes: record a per-slot completion event after lane_publish_done (events created hipEventBlockingSync | hipEventDisableTiming, or device flag hipDeviceScheduleBlockingSync). The lane thread blocks in hipEventSynchronize on the oldest outstanding request, collects the sample, re-queues. Wake latency is off the decode critical path because 4 requests stay pre-queued.
2. Large lane: pre-queue hipStreamWaitValue32(request_sequence >= next) followed by an 'arrived' blocking-sync event on the lane stream; block in hipEventSynchronize(arrived); read rows; enqueue the sized work (its own wait passes immediately). Cost: one interrupt wake per prefill step (tens of us vs ms-scale steps). Then hipStreamSynchronize as today (blocking sync) before the combine.
3. Stop: an unsatisfied device wait would block its CPU waiter forever, so request_stop's owner (the main thread, after helper_fault and the combine futex) host-stores every served lane's request_sequence to a value past its pending wait, releasing the device waits; lane threads observe stopping() after wake and exit without launching. The existing small-lane shutdown already releases its pre-queued waits this way.
4. Startup: helper 0 futex-wakes helper_count after storing it; helper 1 futex-waits (timed, checking stopping()). Drop the segment-file sleep(1) loop: the harness starts helper 0 and waits for its ready line before starting helper 1; keep a bounded retry only if a hand launch can race (inotify on the shm directory as the blocking alternative).
5. No CPU thread polls anything; all waits are bounded by stop/fault paths.

## Code Samples & Guidance



## Files



## Validation

Helper CPU usage idle and during decode near zero (measure with top/pidstat). End-to-end A/B in the working layout (helper-ab.json style, 4 pairs): tg128/tg512/tg2048 and pp1024/pp4096 not worse; decode wake latency measured. Component tests pass. Failure tests: helper 1 killed mid-step, engine stopped mid-step, SIGINT with idle lanes -- every process exits within a bounded time with no hang.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Split out of RAD18 (CPU combine). The combine path itself is already blocking (helper 0 combines after its stream sync, futex handoff from helper 1, futex-sleeping pool). The stop word is an atomic futex word since commit 41f3cae42.

## Change Log

- 2026-10-23T00:50:15.230044+00:00 (created-by): Created by agent
- 2026-10-23T00:53:16.066938+00:00 (updated-by): Updated: section:detailed_solution
