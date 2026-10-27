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

EXECUTION PLAN (2026-10-26, owner-supplied, adopted):
Phase 1 - prove replay-safe inference activation (no GPU changes): locate in radiance's request lifecycle / Ctx::run_step a BURST_BEGIN (no active work -> active, before first GPU replay) and BURST_END (all outstanding GPU work and helper results consumed); concurrent sequences share one burst; requests arriving during drain prevent disarm; fires on every inference (host scheduler, not plugin launch code). Wake helpers once per burst via blocking eventfd/futex. Gate: traces for single-stream, multi-stream and MTP; no lost begin/end or requests at shutdown.
Phase 2 - isolated arm/disarm prototype (extend ipc-signal-bench): IDLE (helper CPU blocked, GPU queues empty, ~9-15 W) -> BURST_BEGIN (engine host notifies, helper arms depth-4 BARRIER_AND waits, acks READY) -> ACTIVE (engine GPU publishes, signal releases barrier, pre-queued work runs, CPU blocks on completion events) -> BURST_END (stop publication for the epoch, drain published, release unused barriers, drain queues, ack IDLE). Tests: repeated bursts, short gaps, concurrent requests, stop mid-compute, engine death. Gate: no polling, no idle pending barrier, no stale completions across epochs.
Phase 3 - active-path performance: idle <=15 W per XTX, GFX ~0%, helper CPU ~0%; gate/up GEMV ~65 us; sustained decode not worse than main; first-request activation p50/p99 sub-ms; deterministic bounded drain; power measured after BURST_END with models resident.
Phase 4 - production (only after 1-3): engine burst lifecycle notifications; narrow ROCclr HIP-stream-wait-on-HSA-signal; IPC signal/slot protocol; blocking events; generation-safe arm/drain/disarm; remove legacy waits and lab kernels; 4 rotated A/B pairs, power, KLD, fault injection.
Phase 5 - fallback: engine-owned helper GPU execution (engine owns XTX contexts/weights/queues); compare with interrupt-driven host submission (30-80 us per wake).

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

engines/radiance/components/moe-remote/r3_moe_worker.cpp (lane threads ~497/~551, startup ~327/~352, stop path)
engines/radiance/components/moe-remote/moe_remote_gpu.h (enqueue_lane, events)
engines/radiance/components/moe-remote/moe_lane_common.h (Request, generations)
engines/radiance/components/moe-remote/tests
tools/lab/radiance/helper-ab.json (A/B vs parent-commit build)

## Validation

Helper CPU usage idle and during decode near zero (measure with top/pidstat). End-to-end A/B in the working layout (helper-ab.json style, 4 pairs): tg128/tg512/tg2048 and pp1024/pp4096 not worse; decode wake latency measured. Component tests pass. Failure tests: helper 1 killed mid-step, engine stopped mid-step, SIGINT with idle lanes -- every process exits within a bounded time with no hang.

## Effort & Risk



## Standards



## Acceptance Criteria

No helper thread polls or yields (per-thread CPU near zero at idle and during decode); stop at idle/decode/prefill and helper-kill/engine-kill all exit within a bounded time; A/B vs the parent build: tg and pp not worse, p99 inter-token latency not worse; component tests pass; polling code removed, no fallback flag.

## Notes

Split out of RAD18 (CPU combine). The combine path itself is already blocking (helper 0 combines after its stream sync, futex handoff from helper 1, futex-sleeping pool). The stop word is an atomic futex word since commit 41f3cae42.

Split out of RAD18. GPT review req_bigcherry_200e404f8776 (2026-10-23): HIP blocking events first; HSA IPC signals only if measured event wake latency hurts decode; hipLaunchHostFunc and KFD events rejected. GPT suggested keeping the polling path as a fallback until hardware results pass -- overridden by the owner's no-legacy rule: compare against the previous build instead.


2026-10-23 observations: during the combine A/B each r3_moe_worker held ~102% CPU (yield polling) - two cores busy that
the CPU combine method also needs; fixing this helps the `cpu` combine. GPT review req_bigcherry_97cebda33722 caution:
ROCm documents hipStreamWaitValue32 for hipMallocSignalMemory only; the shared segment is host-registered, so the
existing large-lane device wait (moe_remote_gpu.h:493) and the design's 'arrived' event behind it need explicit
validation (stress, cross-process visibility, stop/abort) or a different wake path. A stranded device wait would
deadlock shutdown at hipStreamSynchronize (r3_moe_worker.cpp:516).

2026-10-26 Option A test plan (owner-supplied, GPT-authored) adopted with review. Phases: A1 cross-process BARRIER_AND correctness (10k iterations, no false release); A2 four-slot ring (re-arm only after the slot's consumer retired: set 1, queue BARRIER_AND, confirm armed, then producer decrements - the earlier '+1/-1 freely reorderable' idea is unsafe); A3 idle power (gate); A4 GEMV interference at 0/1/2/4/8 pending barriers; A5 latency from GPU timestamps; A6 shutdown/kill; A7 narrow ROCclr extension hipExtStreamWaitHsaSignalZero. Results so far: BARRIER_VALUE rejected on gfx1100. A3 FAILS as measured: one pending BARRIER_AND on an XTX = 92-94 W / GFX 100% (baseline ~10 W), same on ROCm 7.2.4 / 7.14 / 10.1 userspace and on kernel 7.0.13 + AMD gfx11 MES timer fix (rad20-stage-a-10260009, -10261057). Main today: helpers idle at 97.8 / 101.5 W, GFX 100% (R9700 engine 20 W). Per the plan's own go/no-go, Option A is not implemented yet; next discriminators: (1) loaded-but-unarmed control: empty live queue, queue that ran work and drained, process holding queues; (2) amdgpu.sched_policy=2 (static HQD, no HWS/MES) with reboot. Wait-kernel commit on feat/rad20-blocking-lanes stays measurement-only (owner: polling/resident waits not approved).

2026-10-26 discriminators: an XTX queue that is empty or has drained idles at 9.0 W / 0.3% (baseline 10.4 W); only a queue holding an UNSATISFIED wait packet (BARRIER_AND) burns ~92 W / GFX 100%. amdgpu.sched_policy=2 (non-HWS) does not help (1 pending barrier 91.1 W; 4 queues crashed; baseline rose to 17-19 W) - reverted. Kernel MES-timer fix does not help. Conclusion: on gfx1100 (MES fw 0x80) any GPU-side wait left pending costs ~80 W per card; event-driven GPU waits are only energy-acceptable while work is flowing. Design direction: arm-while-busy (pre-queue barrier-class waits only while requests flow, drain to empty queues when idle, re-arm from a replay-safe host trigger) or no GPU-side waiting at all (engine-enqueued work). Brutus restored: standard kernel 7.0.13-070013-generic, sched_policy 0, GRUB saved default = that kernel.

2026-10-26 decision (owner + GPT): Option A (pending IPC BARRIER_AND / signal rings / ROCclr barrier extension) REJECTED for this hardware - no further work. Remaining: A' arm-while-busy (~9-10 W drained; cost = first-request activation once per burst), B helper CPU submits per request on notification (relay ~30-80 us per request, viable but expensive for decode), C engine submits helper AQL work directly (feasibility unknown: cross-process queue ownership, GPU access without P2P, executable ownership, doorbell). Next: establish a replay-safe once-per-burst activation for A'; if impossible, investigate C before accepting B. Gates: cold idle->first-token latency measured separately from sustained decode/prefill; reference = production idle ~200 W combined XTX; target ~20 W combined helper idle with processes loaded.

2026-10-26 PHASE 1 GATE PASS (rad20-burst-trace-10261159, PR #420): rad0002 engine_step/engine_idle call sites in Engine::loop (core/engine.cpp: step where the idle backoff resets; idle where the step batch is cleared) + BcMoeRemotePlugin.burst(active). Production layout with MTP: single 1 burst 0.59 s; 4 concurrent requests share 1 burst 1.60 s; 3 back-to-back requests 1 burst 1.43 s (no churn); long prefill 1 burst 1.44 s; 4 begins / 4 ends strictly alternating, ended before shutdown. Side note: radiance's own idle loop is a sleep backoff (5 us .. 1 ms) - engine-side polling, separate follow-up. Next: phase 2 isolated arm/disarm prototype.

GPT review 2026-10-26 (req_bigcherry_ffa2a770369f): recommend A' (burst/step-scoped arming) with bounded drain and generation-safe cancellation; B (per-request CPU submission, 30-80 us) measured fallback; C (engine-direct AQL into helper queues) rejected for RAD20. Highest risk is safe, prompt cancellation of armed waits, not the wake: an armed GPU wait cannot be cancelled by changing a host generation; release must not run stale work or publish false completion (existing: front kernel marks stale generation invalid -> no routes; publish_done skips stale) and must not race outstanding work. State machine IDLE -> ARMING -> ACTIVE -> (step end) QUIESCING -> next begin back to ACTIVE, else cancel unused + drain -> IDLE; grace via timed futex only delays cancellation. Start with depth 1, prove cancellation + ~9 W after drain, then raise depth. Busy-step pending waits still burn power (acceptable while active, not free). Native BARRIER_AND preferred (~2 us interference) but needs a supported HSA-signal producer on the engine GPU with correct ordering; ROCclr blit wait and s_sleep kernels rejected. Slices: 1 instrument run_step/replay/first publish/helper arm timestamps (lead time); 2 per-helper epochs + futex wake + blocking dispatcher; 3 step-scoped native waits + safe cancel/drain; 4 cross-step continuation/grace; 5 A/B vs main and vs B. Measurements: quiescent loaded helper power ~20 W combined, cold idle->first MoE completion, tg128/512/2048, pp1024/4096, back-to-back transitions, replay identical, zero-route steps, fault/shutdown. Go/no-go: if safe cancellation needs complex GPU machinery or drains hurt decode, switch to B.

2026-10-26 owner-supplied final triage plan adopted: P1 firmware provenance + candidate search for GC 11.0.0 MES/MEC/ME/PFP in linux-firmware (account for 59ab535919b4 Aug-2026 update and 9a7c283fa003 Sep-2026 stability rollback; never the reverted revision, never another GPU's firmware) -> gate F1 (NO CANDIDATE => skip swapping); P2 at most one controlled firmware trial with backups/rollback/verification on both XTX; P3 repository-contained AMD reproducer tools/lab/radiance/rad20-mes-idle.{cpp,sh} (role-resolved GPU, INFO_BREAK handled, EMPTY/DRAINED/PENDING/RELEASED, watchdog, machine-readable results, no host changes); P4 AMD issue (gitlab.freedesktop.org/drm/amd) prepared, published only with owner approval; P5 A' feasibility gate (cross-GPU IPC signal -> XTX BARRIER_AND end to end, GEMV interference at 1/2/4/8 pending, payload visibility, slot reuse with ordered generations, safe release of unused waits, active vs idle power); P6 implement A' lifecycle IDLE->ARMING->ACTIVE->DRAINING->IDLE (engine waits for helpers READY before the burst's first publish; no per-layer CPU notification; generation-safe cancel); P7 production acceptance (4 rotated A/B pairs, tg128/512/2048, pp1024/4096, concurrency, MTP, short bursts, long idle, stop/kill, idle <=15 W per XTX, KLD). Known results not to repeat: queue priority low/normal/high all ~95-98 W (GPT's binary needed HSA_STATUS_INFO_BREAK fix), ordinary vs IPC signal both fail. Firmware/AMD report proceed independently of A'.

2026-10-26 P1/P2 firmware: Brutus had Ubuntu linux-firmware 20240318 (ubuntu2.27): MES 0x80, KIQ 0x104, MEC 0xa5a, ME 0x974, PFP 0xa14. Upstream GC 11.0.0 history: d1363e89 (2026-05-07) is the current non-reverted MES set; later updates 59ab5359 (reverted 9a7c283f), cd0c6107 (reverted 0a5565a4), 972f521e (2026-10-16, reverted 4d3ecb24 2026-10-17) excluded. Controlled trial of upstream head (me/mec/mes1/mes_2/pfp differ; imu/rlc/mes.bin identical): loaded MES 0x88, KIQ 0x108, MEC 0xa78, ME 0x9a6, PFP 0xa50 with no driver errors; pending BARRIER_AND still 92.6 W (XTX0) / 91.7 W (XTX1) at GFX 100% vs empty 8.8 / 9.5 W. FAIL -> rolled back (backup fw-backup-rad20-10261236, checksums verified, MES 0x80 confirmed after reboot). Firmware avenue closed. Next: P3 AMD reproducer, then P5/P6 A'.

2026-10-26 AMD issue filed by owner: https://github.com/ROCm/TheRock/issues/9484. Reproducer PR #421 (rad20-mes-idle.{cpp,sh}): both XTX empty 9-10 W, drained 12-13 W, pending BARRIER_AND 95 W / GFX 100%, released 12-12.5 W (runs rad20-mes-idle-1026130105, -1026130959). Owner code-level diagnostic plan adopted: P0 harden reproducer (6 phases incl DESTROYED, 3 reps/GPU, telemetry.csv/phases.json/summary.json); P1 GFXOFF debugfs (amdgpu_gfxoff, amdgpu_gfxoff_status: 0 off,1 exiting,2 on,3 entering) synchronized to phases; P2 UMR read-only CP/GRBM/HQD snapshots S0-S6; P3 MES event log (amdgpu.mes_log_enable=1, amdgpu_mes_event_log) on a diagnostic boot; P4 packet variants (BARRIER_OR, satisfied dep, AGENT vs SYSTEM fences, SINGLE vs MULTI); P5 driver instrumentation (gfx_off_req_count tracing; enable_reg_active_poll 1->0 one-variable lab kernel; verify no-HWS path); P6 mes_dbgext only if compatible; P7 classify (software GFXOFF veto / MES state / CP dependency wait / other PM); P8 A' regardless; P9 acceptance. Stop: no more firmware swaps without a new hypothesis.

2026-10-26 ROOT CAUSE (classification C, CP dependency wait). GFXOFF: amdgpu_gfxoff reads 0 in every phase incl 9 W idle (GFXOFF not engaged; idle comes from clock/power gating); amdgpu_gfxoff_status unsupported on gfx1100 (EINVAL). UMR (built from source, freedesktop tomstdenis/umr ac8a7ce) snapshots on XTX 03:00, one HSA queue + one pending BARRIER_AND, no kernels: CP_CPC_STATUS=0xa0200401 (MEC1_BUSY, TCIU_BUSY, MEC_DATA_CACHE_BUSY, CPG_CPC_BUSY, CPC_BUSY), CP_CPC_BUSY_STAT=0x880 (MEC1_TC_BUSY, MEC1_PIPE1_BUSY), CP_CPC_STALLED_STAT1=0x04000800, GRBM_STATUS=0xa800302c (GUI_ACTIVE, ANY_ACTIVE, CP_BUSY; no SPI/TA/SC/PA/DB/GE), GRBM_STATUS2=0x3000000c (CPF_BUSY, CPC_BUSY), GRBM_STATUS3=0x32004000 (CH_BUSY, GUS_BUSY, GUS_LINK_BUSY, SEDC_BUSY); all 0 when empty and after release; CP_STAT/CP_BUSY_STAT/RLC_STAT/MES_BUSY stay 0. Interpretation: the MEC pipe busy-polls the barrier's dependency signal through its TC/TCIU memory path, keeping GUI_ACTIVE set so DPM runs full clocks (~95 W). Fix belongs in MEC firmware (event/interrupt-driven or sleeping dependency wait); evidence text prepared for TheRock#9484.

2026-10-26 P4 packet variants (XTX 03:00, 30 s): BARRIER_AND system 95.0 W, BARRIER_OR 100.2 W, BARRIER_AND agent scope 101.1 W, BARRIER_AND no fences 98.8 W, all GFX 100% and PASS (held + released); baseline 10.2 W. No GPU-side wait form idles on MEC 0xa5a / MES 0x80: GATE 4 closed. GPU waits are acceptable only while a burst is active -> A'.

2026-10-27: full record of the wait/queue/idle-power attempts, measurements and the assessed alternatives (GPT req_bigcherry_8ac7a7f2625d + architecture note, checked against the code): docs/research/radiance-history/RAD20-helper-waits-2026-10.md. Next: HIP blocking completion events for the small lane; harness for host-callback/graph-host-node -> eventfd wake on the large lane only (small-lane per-request callbacks rejected: engine stream stall per MoE layer).

## Execution gate review (2026-10-23)

- **Blocking P0:** do not enqueue an *additional* `hipStreamWaitValue32` on host-registered cross-process shared memory until the documented signal-memory contract and cross-process system visibility have been proven. The existing engine-to-helper wait is not evidence that the proposed arrived-event path is safe. Treat cancellation, generation wrap, and device progress independently.
- **Slice RAD20-A (offline/hardware test first):** isolate a minimal two-process signal-memory probe for the exact HIP/ROCm device and mapping, including idle, prefill, fault, cancel, rerun after stop, and long-run sequence wrap. If unsupported, large-lane arrival must use a supported CPU wake bridge (bounded host wait/futex on a CPU-published notification) or another proven event path; never make an event downstream of an unbounded device wait.
- **Slice RAD20-B (implementation PR):** replace small-lane done polling with non-timing blocking HIP events for the four outstanding slots; replace large-lane request polling only after the arrival proof; make explicit per-slot generation/retirement so events are not recycled while pending. Startup must not rely on unbounded sleep loops.
- **Stop/fault invariant:** every queued wait must be satisfiable, cancellable or provably skipped **before** stream synchronization or join. Stale/early `request_sequence` must never release work on a different generation. Simulate killing each helper and engine at idle, decode, prefill, and mid-shutdown; bound exit times and verify no stale writes.
- **Promotion:** same-source parent-build A/B with the lab queue, four rotated pairs; per-thread user+system CPU usage (not process aggregate), event wake/requeue p50/p99, pp1024/4096, tg128/512/2048 and fixed-token latency. CPU reduction alone does not justify decode latency regression. No permanent polling fallback; retain old build solely as comparison baseline.

## Change Log

- 2026-10-23T00:50:15.230044+00:00 (created-by): Created by agent
- 2026-10-23T00:53:16.066938+00:00 (updated-by): Updated: section:detailed_solution
- 2026-10-23T01:10:23.534143+00:00 (updated-by): Updated: section:steps, section:notes
- 2026-10-26T00:31:47.503769+00:00 (updated-by): Updated: section:notes
- 2026-10-26T00:44:36.625738+00:00 (updated-by): Updated: section:notes
- 2026-10-26T00:46:23.416745+00:00 (updated-by): Updated: section:notes
- 2026-10-26T00:55:05.001788+00:00 (updated-by): Updated: section:steps
- 2026-10-26T01:08:49.312341+00:00 (updated-by): Updated: section:notes
- 2026-10-26T01:15:03.602469+00:00 (updated-by): Updated: section:notes
- 2026-10-26T01:33:29.267371+00:00 (updated-by): Updated: section:notes
- 2026-10-26T01:43:37.284225+00:00 (updated-by): Updated: section:notes
- 2026-10-26T02:17:21.593552+00:00 (updated-by): Updated: section:notes
- 2026-10-26T02:27:22.858565+00:00 (updated-by): Updated: section:notes
- 2026-10-26T02:32:05.617898+00:00 (updated-by): Updated: section:notes
- 2026-10-27T00:12:46.369115+00:00 (updated-by): Updated: section:notes
