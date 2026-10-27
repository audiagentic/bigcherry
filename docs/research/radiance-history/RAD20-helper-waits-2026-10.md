# RAD20: helper waiting, queues and idle power (2026-10-23 to 10-27)

Layout: radiance engine on the R9700 (gfx1201), one `r3_moe_worker` helper per RX 7900 XTX (gfx1100), the build server,
ROCm 7.2.x, kernel 7.0.13. Plan item: RAD20. Summary on the model page
(`docs/reference/engines/radiance/models/qwen3.8-flash-next.md`, "Helper waiting, queues and idle power").

## Why the helpers drew ~95 W idle

1. **A pending AQL wait.** Any wait left in a queue (`hipStreamWaitValue32` blit kernel or a barrier-AND packet)
   is busy-polled by the MEC; the GPU never reaches its idle states.
2. **More than `num_cp_queues` compute queues** (8 on XTX, KFD topology). MES is then oversubscribed and permanently
   busy (`CP_CPC_STATUS` MES bits busy with 13 queues, idle with 5). Oversubscription also dispatches faster:
   decode launch gap p95 3.6 us against 7.6 us (`rad20-queue-probe.sh`). This is power save's decode cost.

Path traced from source: ROCclr `Device::acquireQueue` gives one HW queue per stream below `GPU_MAX_HW_QUEUES`;
KFD `add_queue_mes`/`remove_queue_mes`; `update_queue` with `queue_percent = 0` deactivates a queue (what ROCr
`AqlQueue::Suspend/Resume` do). Identical on TheRock ROCm 10.1; the stock MES `oversubscription_timer` is not the cause.

## Attempts

| Attempt | Result | Why | Evidence |
|---|---|---|---|
| Burst gating (Option A, PR #422): small-lane waits armed only inside engine bursts, released and drained at park (protocol v17 `Header::burst`, `LaneSignal::park`, Dekker seq_cst handshake) | idle still ~95 W | 13 queues per XTX (cause 2) | `burst-idle.sh` |
| + streams only for the lanes a helper serves (13 -> 5 queues) | idle 6-12 W, gate PASS | | `rad20-burst-idle-*` |
| Stale-publish guard reading `q->sequence` from the GPU | +1 us a request | PCIe read of host memory | replaced by the device-memory publish limit `d_publish_limit` |
| `perf` mode: waits armed always, gated only at park | no gain | the cost is the queue count, not arming | removed |
| Power save toggle `R3_MOE_WORKER_POWER_SAVE` (default 1); off opens all 10 lanes | off equals main (launch gap p95 3.56 vs 3.60 us) | | queue probe |
| Queue warm-up (`worker_queue_warm.h`): placeholder HSA queues toggled active/parked per burst with `AMDKFD_IOC_UPDATE_QUEUE`, so the GPU is oversubscribed only inside bursts | harness: parked 9 W, active 91 W, 140-260 us a queue switch. Live: CPF page fault (permission, read) on both XTX in the first request, MES REMOVE_QUEUE failed, mode-1 reset of both GPUs; the engine then hung 8.7 h | switching queues while other queues run kernels, which the harness never did | run `rad20-burst-profile-10262303`; shelved, `evaluated`, default off |
| Option B: barrier-packet wake instead of stream wait (bc2 harness) | 0.3-0.9 us slower p50; 96-98 W with either kind of wait pending | any pending wait is cause 1; earlier tail was arm-order warm-up (first R9700 stores slow ~15 ms) | PRs #423-#425 |
| GPU wait kernel with `s_sleep` polling | not approved | resident wave keeps the GPU busy and polls host memory over PCIe | owner rule 2026-10-25 |
| Two streams per helper card | hung | libr3 MoE GEMM not safe concurrently on one card | RAD02, reverted 563455f8 |

## Measurements on the final Option A (a2a7630b9 / refactor 8029cb75d)

Flash-Next A/B, 4 rotated pairs, CPU combine on in both arms (`rad20-burst-ab-10270752`):

| | main | power save | delta |
|---|---|---|---|
| pp1024 | 3,630 | 3,588 | -1.1%, separated |
| pp4096 | 3,554 | 3,549 | -0.1% |
| tg128 | 97.5 | 96.7 | -0.9%, separated |
| tg512 | 145.3 | 144.6 | -0.5%, separated |
| tg2048 | 152.4 | 151.7 | -0.5%, separated |

Refactor PR #426 (`rad20-burst-ab-10271007`): pp1024 3,591, tg512 143.4 (one 139.5 outlier), tg2048 151.8 - same as
#422. Idle 6-8 W (`rad20-burst-idle-10270959`).

Concurrency, 2 pairs (`rad20-burst-ab-10270937`): aggregate tok/s 2 streams 135.2 -> 133.7, 4: 172.8 -> 170.6,
6: 200.4 -> 196.1 (6-stream draft acceptance 0.65 vs 0.61, so not equal work); TTFT unchanged.

tg128 deep profile, same build, combine off vs on, 8 prompts (`rad20-burst-profile-10270820`): helper per-request GPU
time p50 133.4 vs 132.6 us, launch gaps identical; prefill +6% on every prompt; decode 101.8 vs 101.0 with per-prompt
swings -37% to +35% that follow draft acceptance. Combine changes the add order, so the text differs: the tg128 dip
is a content effect.

## What still polls

Inside a burst the small-lane completion wait and the large-lane request wait yield-spin (~100% of a core per lane
thread, 6 per helper); between bursts they sleep on the burst futex. The engine does not notice a dead helper mid-request.

## Alternatives assessed (GPT req_bigcherry_8ac7a7f2625d and its 2026-10-27 architecture note, checked against the code)

| Alternative | Verdict | Reason |
|---|---|---|
| HIP blocking completion events (`hipEventBlockingSync`, one per queued slot) for the small lane | **do next** | the CPU only refills a `kQueueDepth`-deep queue, so the wake latency is mostly hidden; removes the in-burst yield loop |
| `hipLaunchHostFunc` / graph host node -> eventfd for **small-lane requests** | **rejected** | small-lane requests are already gated on the GPU (pre-queued stream waits, armed only in bursts); a host callback on the engine stream holds the stream until it returns, adding a host round trip to every MoE layer of every decode step. GPT's "one callback per burst" does not fit either: a burst carries many requests |
| Host callback / host node -> eventfd for **large-lane requests** | **measure** | the only CPU wake path for a GPU-published request; large requests are few (prefill steps of 256+ rows), so tens of us may be noise. Harness first: publish->wake latency and engine stream stall |
| CPU request gate (futex) for requests | rejected | the engine publishes from GPU code, which cannot wake a futex |
| HSA blocked wait / async handler | after HIP events, only if their wake latency hurts | needs an interrupt-backed signal; GPT cites ROCm issues on async-handler CPU use (not verified here) |
| Direct AQL submission (hipfire-style) | not now | does not change either idle-power cause; maintenance cost |
| Queue warm-up inside the new runtime | no | shelved after the GPU fault above |
| Memory ordering note (seq_cst does not order GPU writes) | agreed | the GPU side publishes with system-scope release in `lane_publish_done`; any new notification must follow the publish |

Prior art usable here: llama.cpp server/RPC stop pattern (flag under mutex, notify, join); vLLM's rule that a CPU
event confirms a GPU event was recorded before anyone waits on it (applies to slot reuse). Strata issue #884 (a
`hipGraphLaunch` barrier stalling a queue) is a reason to inspect AQL for any graph host-node path.
