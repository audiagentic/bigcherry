# RHP11 slice C: admission and quiescence

The hook coordinator owns the admission ledger. The managed rad0002 package
supplies the scheduler/rank bridge; upstream files contain only extension
calls, an extension member, and includes. No ticket is derived from a
scheduler step number.

## Admission and completion

The bridge admits before Scheduler::step, checkpoint restoration and
prepare_release. An issue-ahead reservation uses try_admit; a closed or full
gate skips run-ahead without waiting, allowing the current parent to finish.
An empty batch cancels only its unused tail reservation; a rejected cancel
faults the ledger and stops the step loop (RAD_E_STATE). A promoted batch
inherits its admission.

The existing rank barrier publishes the issued ticket. Each rank installs
that value in thread-local storage; Ctx explicitly passes it to
bc_moe_remote_before_pass before every run_pass. The ordered device setter
remains outside tape capture. Trunk and MTP/history/draft passes share the
parent's immutable ticket.

A parent retires in admission order only after all three predicates hold:

* Host logical completion, after finish_block_drafter/finish_serial_drafter.
* Its own terminal event on every actual participating rank, recorded after
  mark_step_end and before issue-ahead. Events are queried, never synchronized.
  Each rank owns one event per ledger ring slot (admission_completion.h),
  created once when the rank's stamp ring is allocated and destroyed at
  shutdown; a slot's event is re-recorded only after its previous parent
  retired. Recording and querying need no device switch.
* Every rank's matching ticket/epoch/slot consumption proof, with equal
  published-request and consumed-request counts. Zero requests need no ACK.

### Where the consumption proof is counted (consumption_ring.h)

Per layer the mailbox kernels add no host-mapped traffic beyond main's; the
proof is counted in rank-local device memory and copied to the host once per
parent step.

* Each rank owns a device-memory ConsumptionStamp ring, ticket_id % capacity
  like the host-mapped one. The ticket setter that before_pass launches on
  every pass arms the slot on the parent's first pass on that rank
  (consumption::arm); later passes of the same parent keep its counts.
* mailbox_publish_kernel checks the slot belongs to its pass's ticket and adds
  the number of helper requests the pass will release to published_count, in
  one device-scope atomic (no returned value) before any is released.
  mailbox_add_finish_kernel checks the same slot from the pass's ticket copy,
  adds the number of lanes the pass published to consumed_count in one atomic
  (a block count), and records the lane's (and a combined step's) consumed sequence with the
  pass's epoch/slot in ConsumedRecords, also device memory.
* mailbox_wait_helpers is main's wait: the acquire of done_sequence == seq and
  the Request's status, nothing else. done_sequence names exactly the request
  this pass published with its own ticket's epoch/slot; the helper's lane
  done_epoch/done_slot are copies of those words, so reading them back per
  layer would prove nothing. lane_publish_done is main's kernel: one
  system-scope release of done_sequence and no other store over the link.
  The lane's done_epoch/done_slot, which the quiescence check compares, are
  written by the helper CPU once per drain, from its last completed Request,
  after the lane's stream is drained and before its closure ACK (release).
  A hardware bisect showed the two per-request GPU stores of those words cost
  most of slice C's decode overhead (+0.8% to +0.3% when removed). Placement
  safety is the ledger's (a parent never spans an epoch flip) and the
  quiescence check's (r3_moe_wait_drained compares done and consumed
  epoch/slot per lane at drain). Hardware traces showed each extra mapped
  store or load here cost about 2 us per helper request on the engine's
  critical path.
* Once a parent's terminal event on a rank is complete, a sealed stamp that
  does not match its ticket/epoch/slot, or whose published and consumed counts
  differ, faults the ledger at once, naming the parent: the event follows the
  seal on the same stream, so nothing can still change the stamp.
* At seal, before the parent's terminal event on each rank,
  mailbox_seal_kernel copies the changed lane/combine records into
  LaneSignal::consumed_* and CombineSignal::result_consumed_* (the quiescence
  check's words) and the parent's slot into the host-mapped stamp, ticket_id
  last with a system-scope release. The terminal event (system-scope release)
  follows on the same stream, so a complete event implies the sealed stamp is
  visible.

The counts are exact per (parent, rank):

* Stream order on a rank is: parent P's passes (trunk, history, MTP/draft),
  P's seal and event, then the run-ahead parent P+1's setter and passes. Every
  publish and finish of P runs between P's setter and P's seal, and counts
  into P's slot: each kernel checks the slot holds its own pass's ticket.
* P+1 runs ahead of P's retirement in its own slot: the ledger never admits
  P + capacity while P is live, so two outstanding parents never share a slot,
  and the setter's reset of a slot happens after its previous owner was
  sealed and retired.
* Each released request adds exactly one to published_count before the
  helper is signalled, and its finish adds exactly one to consumed_count after
  the helper's done word and the add. The seal therefore sees
  published == consumed iff every request P published on that rank was
  consumed; it traps if consumed ever exceeds published. A wrap would need
  2^32 publications by one parent on one rank, far beyond passes x layers x
  helpers.
* Ranks count independently in their own rings, and each rank's lanes are its
  own (a two-rank engine uses its rank slot; several helpers serve only a
  one-rank engine), so the lane records never race across ranks.
* A stale host stamp (an older parent of the same slot, or a rank that never
  sealed) has a different ticket_id and never proves a newer parent; a
  sealed stamp with an unconsumed request keeps its parent pending.

## Capacity derivation at the pinned engine (4a61d72)

There is **no hard scheduler bound** on outstanding GPU parents:

* core/engine.cpp:569 holds exactly one scalar `ahead` batch; lines
  819–826 issue it after the current parent's terminal point.
* core/engine.cpp:813–819 and 905–909 include draft/MTP passes in the
  same parent's lifetime. core/runtime/ctx.h:116–120 explicitly states
  that draft passes report into their parent's same step bank. Pass count
  therefore does not multiply the number of ledger entries.
* core/engine.cpp:1284 returns without waiting when no output rows exist.
  core/sched/scheduler.cpp:2390–2397 explicitly says a no-output prefill
  chunk commits while the GPU still runs. step_finished only records host
  timing; it supplies no GPU backpressure.
* core/runtime/ctx.h:99–124 describes the RouteSlot ring and declares its
  staging array. Its pinned bank count is 32. core/runtime/ctx.cpp:630–631
  indexes it by step; lines 656–660 count overwritten, uncredited banks
  rather than waiting. The ring is a resource ceiling reference, **not**
  evidence that upstream restricts prefill to that many live parents.

rad0002's own core/place/bc_moe_remote.cpp derives the ledger ceiling as
`extent(RouteSlot::staging) + BcMoeRemoteAdmission::kIssueAheadDepth`:
the engine's existing ring plus its single explicit ahead reservation
(33 at this pin). Registration passes this ceiling through the size-checked
plugin before replaying layer maps or lowering rank graphs. There is no
environment override, workload seed, or default capacity. Unconfigured
admission fails closed.

The CPU ledger is a fixed ring of that ceiling, allocated once and indexed by
ticket_id % ceiling like the stamps; serving allocates nothing. Completion
polling (Ledger::poll) visits host-complete parents under one lock and, per
rank, stops at the first pending event: a rank's terminal events are recorded
in admission order on one stream. The ledger high-water is retained
and reported with outstanding count and ceiling on every close and shutdown.
Rank consumption stamps allocate the ceiling once, before graph capture:
their address and ticket-ID modulus never change.
Cancelled, unused tail reservations reclaim their IDs so live IDs cannot
alias the fixed stamp ring. A full issue-ahead reservation is skipped; a full
ordinary admission or allocation failure faults closed without a serving-path
GPU wait. A workload exceeding the ceiling needs a reviewed change derived
from engine resources; no claim is made that every prefill workload fits.

## Controller API

moe_remote_hook.h exports:

* r3_moe_close_admission(snapshot): atomically close the ledger gate and
  snapshot its cutoff, old epoch, outstanding count, high-water and capacity.
  A unique closure ID rejects stale drain proofs.
* r3_moe_wait_drained(snapshot, deadline_monotonic_ns): poll per-parent GPU
  events and consumption proofs, then request helper drain for that closure.
  The deadline uses absolute CLOCK_MONOTONIC nanoseconds.
* r3_moe_reopen_admission(epoch, generation_slot): require successful drain
  and an exact match with the committed placement, then resume helpers and
  admission. Same-epoch reopen retains the same slot.

Only quiescence waits. Shutdown, helper faults and drain timeouts keep the
gate closed and cannot advance the ledger epoch. A scheduler already waiting
at the gate allocates the reopened epoch, rather than a stale pre-wait value.

LaneSignal uses formerly reserved v23 words at offsets 40 and 48 for
drain_request/drain_ack; its size and existing offsets stay fixed. After the
ledger drains, each small helper lane stops refilling, suppresses phantom
completion publication, releases its prequeued waits and drains its stream,
even with power saving disabled. Large lanes ACK after their stream and
CPU combine jobs finish. ACKs are specific to the closure; helpers remain
parked until reopen.

The request reaches the helpers through the burst word, never through reads in
their serving spins: the engine stores each lane's drain_request, then ends a
running burst (or moves an idle word by 2, parity unchanged, so futex sleepers
wake) and holds it even until reopen. radiance's own burst transitions during
the drain are recorded and applied at reopen, after the drain words are
cleared and before the ledger reopens. A lane reads drain_request only where it
has seen the burst end or in wait_burst, so its serving and waiting spins read
exactly what main's do. Hybrid A/B (swapping only the helper binary) showed
per-iteration drain reads in those spins cost about 0.5% decode.

## No-op epoch cycle through PlacementControl

worker_epoch.h extends the slice B Controller with begin_epoch_cycle:
ACTIVE -> QUIESCING -> COMMIT -> ACTIVE. Each cycle allocates a fresh
transaction ID and journal record but retains exactly the same epoch,
generation slot, digest, owner bytes and helper pointer tables. It skips
staging and retirement because no generation is replaced. Normal unchanged
placement proposals remain no-ops; this explicit diagnostic entry point
cannot make a live move.

EpochCycle accepts its period as a typed chrono duration, with zero disabled.
Helper zero owns the controller and publishes the existing v23
PlacementControl through SharedTransport; every helper participates through
its own ACK record. The engine publishes its actual rank mask on first
admission. The helper loop starts a due cycle only during serving and never
overlaps a cycle or starts another before lane drain requests are cleared.

The engine's protocol thread observes QUIESCING, closes admission and waits
for the existing ledger/event/consumption/helper-drain proof against the
controller's monotonic deadline. Only then does it publish exact
QUIESCED ACKs for participating ranks. Each helper independently checks
every served lane's closure-specific drain_ack, including prequeued stream
waits and large-lane CPU combine work, before its QUIESCED ACK.
The controller requires all helper and rank proofs before recommitting the
same placement. All actors ACK COMMITTED while helpers remain parked.
Only the resulting ACTIVE publication permits the engine to reopen the
same epoch/slot and release the lanes. Deadlines are checked even if no
lane drains or a burst ends mid-cycle. Faults, shutdown, stale ACKs and
missing participants cannot reopen admission. There is no hook-side
periodic test driver.

**Interim interface: R3_MOE_WORKER_EPOCH_CYCLE_S (diagnostic).** The cycle
period reaches the helpers through this environment variable, set on every
helper (component.toml; 0 = off, the default). worker_config.h parses it as
whole seconds bounded only by what EpochCycle's std::chrono::nanoseconds
period represents, and r3_moe_worker.cpp hands it to EpochCycle as a chrono
duration. It is a validation diagnostic, not a serving setting.

**Migration (RSC08):** the variable moves to the typed
`experiment.helper.epoch_cycle_s` of the v1 bc_serving manifest, passed to
every helper through the typed helper launch configuration. Zero stays
disabled; the schema/decoder rejects negative, non-finite or unrepresentable
durations; the environment variable is then removed, not kept beside it. The
controller, shared transport, helper polling and engine protocol participant
need no change. Routing through two owner generations and live helper
weight/table migration remain later slices.

## Validation

The ledger test is card-free and covers admission, closed try-admission,
ordered retirement under out-of-order rank signals, capacity/high-water,
blocked scheduler reopen, shutdown, faults, deadlines, dynamic prefill backlog
and stale helper ACKs. The epoch component test exercises 1,000 same-placement
cycles through the real shared PlacementControl/ACK transport with simulated
GPU/lane proofs, including stale closures, absent commit ACKs, idle/mid-cycle
burst end, delayed reopen, unchanged owner bytes and fail-closed timeouts.
The patch suite compiles and runs a fake-device scheduler bridge test,
checks ABI registration, all 28 upstream edits, five created extension files,
fail-closed anchors, idempotence and output blob identities. The engine.h
fixture is the verbatim pinned upstream blob.

Run the component ctests excluding r3_route_compact_test and
r3_ticket_replay_test, which require hardware. Compile the worker/plugin and
syntax-check the managed engine files without running the GPU.

### Hardware evidence

runs/rhp11-c2-ab7 measured this slice (commit 71c6af599) against main on the
four-card Flash-Next layout: nine interleaved arms (three main, three slice C
with cycles off, three with R3_MOE_WORKER_EPOCH_CYCLE_S=2), each a 1-stream
512-token and a 6-stream 2048-token soak with MTP. Median engine decode step
against main: +0.14% at 1 stream with cycles off, +0.34% with cycles on (6
streams: -0.46% off, +0.30% on). 1-stream text was byte-identical in every
arm, and each cycling arm completed 28 placement epoch cycles with no hang or
fault.

### Remaining limitations

* RHP15: an encoder-only (media) pass has no scheduler-parent ticket, and
  before_pass rejects it, so multimodal serving with remote MoE fails closed
  until admission distinguishes encoder passes. Text-only serving is
  unaffected.
* The epoch cycle is a same-placement no-op; moving owners between two
  generations and migrating helper weights/tables are later slices.
* The period is the interim diagnostic environment variable until RSC08.
* The admission ceiling is derived from the pinned engine's RouteSlot ring
  plus its issue-ahead depth; a workload with more outstanding no-output
  prefill parents than that fails closed rather than waiting.

## Slice D part 1: engine-side owner generations (no live commit yet)

Each engine rank owns two stable GPU bitmap allocations, with one byte-addressed
helper bitset per generation derived from the canonical v23 generation map.
The stride is ceil(actual layer_count * expert_count / 8), not a protocol-maximum
layer allocation. The two pointers are immutable for the lifetime of captured
graphs. The publish kernel uses the StepTicket already loaded per pass, indexes
the appropriate pointer and validates the GPU-resident epoch/digest binding.
The route-mark kernel uses the immutable pass ticket copy (not the rank cell)
to select the same generation and to cross-check routes. No mutable global
device active-owner selector is read. The host owns-expert callback remains a
bootstrap engine-weight-residency check; Stage 1 never changes local versus
remote eligibility.

Each rank has one private HIP upload stream, two events and pinned per-slot
staging buffers. The inactive slot's map/digest is validated against
PlacementControl and its complete GenerationHeader before packing. The
binding metadata is copied after the bitmap on that same stream; the event
follows both. Querying readiness is nonblocking and is not a serving-path
sync. The bootstrap active generation is synchronized once during startup.

Upload rejects the active slot and any unretired ledger entry for its target
slot. Host-complete alone does not release a ticket: retirement requires
all ranks' terminal events and exact sealed consumption proof. The event and
staging buffer cannot be reused while an upload is in progress. No independent
in-flight counters or wall-clock retirement rules exist.

The Part-1 bitmap/upload boundary is now wired into the Part-3 coordinator
below for no-move cycles; it still refuses reuse of the active or any
unretired generation slot. This historical section describes the Part-1
mechanism, not the current completeness of the slice.

## Slice D part 2: exact engine Request identity and per-parent consumed ACK

**No helper per-request epoch publication is introduced.** The helper GPU
still release-publishes only `LaneSignal::done_sequence`, as in slice C;
`Request::completion_epoch/slot` are not written by `lane_publish_done`.
The helper CPU copies its last completed Request's epoch/slot into
`LaneSignal::done_epoch/slot` **after** its stream drain and before its
closure-specific drain ACK. An engine wait cannot independently certify
which helper generation actually executed using a second per-request helper
stamp. The actual strict check possible at this boundary is:

* `mailbox_wait_helpers` retains **main's exact** helper
  `done_sequence == sequence` system-scope acquire and the existing
  `Request::status` check. It does **not** decode a ticket or reread
  `Request::sequence`, `placement_epoch` or `generation_slot` on this
  per-request hot path. Helper epoch/slot are independently corroborated at
  drain ACK, not by repeated reads of the engine's own Request words.
* `mailbox_route_mark_kernel` (already on both output-reading paths)
  decodes the immutable device pass ticket and validates the engine-owned
  Request's sequence/epoch/slot through the shared
  `mailbox_completion::matches_request` predicate. The existing
  `launch_maps` queues route-mark **before** `mailbox_add_rows_kernel` on
  normal and combined steps and before `mailbox_shadow_rows_kernel` in
  shadow mode. There is no helper-output reader bypassing that route-mark.
  The original route-mark already read `Request::sequence` (u32), rank,
  rows, hidden, route count/offset and (for noncombined steps) status;
  Part 1 added only the `Request::placement_epoch` (u64),
  `Request::generation_slot` (u32), and `Request::layer` (u32)
  identity/owner checks **inside those existing per-request route-mark
  Request accesses**. It also added the accepted owner_has test per
  routed expert. No new Request read or owner check was added in Part 3.
  Thus a combined chunk cannot be read with a mismatching Request even
  though its final helper done_sequence arrives after the chunkwise add.
* `mailbox_add_finish_kernel` is ordered after route-mark, add/shadow result
  reading, and the normal or combined tail wait on the **same rank stream**.
  It checks the device copy of the pass ticket against the already armed
  device-memory consumption stamp, records the consumed sequence/epoch/slot
  in device memory and increments the existing one-per-pass count. It does
  not reread the mapped Request after those preceding checks.

**Result-consumed ACK is per parent and rank**, not per helper request:
`mailbox_seal_kernel` follows the parent's **last** add/finish, all trunk and
MTP/draft passes, and any combined tail/shadow consumption. It now explicitly
rejects a stamp not belonging to its exact `StepTicket` or whose published
and consumed counts differ. The existing seal publishes
`ConsumptionStamp{ticket_id, placement_epoch, generation_slot,
published_count, consumed_count}` once to the mapped host ring, with
`ticket_id` as the release publication word. At the same boundary it
flushes only changed per-lane and combined-plane consumed identities to
shared memory. The rank's **existing** terminal radiance event is recorded
after this seal, and the CPU ledger accepts the stamp only after that event
completes on every actual rank. There is no second per-step host sync,
event, or per-request system-scope publication.

`r3_moe_all_ranks_consumed(snapshot)` is the nonblocking coordinator query
now used by Part 3 at QUIESCING and RETIRING: it returns **1** only for the exact active closure ID,
cutoff and epoch whose ledger is empty *after* polling the existing rank
events and matching consumption stamps; **0** while any rank/parent remains,
and a **negative** result for stale closure, shutdown or fault. The gate
must already be closed by `r3_moe_close_admission`; no second admission
gate exists. An unused tail reservation cancelled after closure does not
invalidate the snapshot. The query does **not** initiate a helper drain
and alone does not authorize PREPARED/RETIRING or reopen admission.
Part 3 also requires all rank upload-ready events and the existing
helper drain-ACK identity/`r3_moe_wait_drained` proof.

**Hot-path delta (source level, not hardware-measured):** the wait is now
**byte-semantically main's wait**: zero extra system-scope loads or mapped
Request fields, stores or fences per helper completion. The route-mark
already read `Request::sequence`; this slice also reads its epoch
(u64), generation slot (u32) and layer (u32) there for identity/ownership,
without a new bitmap bit test per route. The publish kernel checks the
exact pass-selected device bitmap once per block (the owner binding is
loaded by thread zero, shared with the block), and route-mark reconstructs
the pass ticket only once per block. Its per-route loop performs the main
weight check and atomic mark, not an extra owner bitmap lookup. There is no
second epoch/slot/sequence Request reread in the wait or finish. Ticket
metadata and consumption stamps remain rank-local device memory. Parent
seal adds a constant-time predicate but no new mapped stores, fences or
event beyond slice C. Coordinator upload/ready/retire is entirely
outside serving. Hardware latency remains unmeasured.


## Slice D part 3: true two-generation diagnostic cycles and coordinator

`R3_MOE_WORKER_EPOCH_CYCLE_S` now drives a **real** same-owner-map epoch
transition, rather than recommitting the same epoch/slot. Helper zero
copies the canonical owner map into the inactive generation's separate
shared-memory slot as `STAGING`, with `pending_epoch=active_epoch+1`,
`pending_slot=1-active_slot`, a digest equal to the active map's digest,
and distinct transaction ID. The source and destination bytes, shape,
digest and per-transaction journal identities are checked by
`same_map_cycle`; any moved expert is refused. This does **not** migrate
helper weights or pointer tables (slice E) or implement forced moves
(slice G). The CPU helper driver now traverses the full
PROPOSED/RESERVING/COPYING/PREPARED/QUIESCING/COMMIT/RETIRING/ACTIVE state
machine; exact RESERVED, READY, QUIESCED, COMMITTED and RETIRED ACKs are
required from every expected helper and actual engine rank.

The engine coordinator in `moe_remote_hook.cpp::placement_loop`:

1. **RESERVING.** Every real engine rank validates the actual contiguous
   rank mask, its registration and completed active generation event, the
   exact current transaction and candidate STAGING metadata/digest via
   the shared `service_rank_reservation` predicate, then release-publishes
   its own RESERVED ACK. No synthesized rank ACK is used by the tests.
   Mismatched/corrupt rank metadata reports a current-phase fault, never
   a serving fault before admission closes.
2. **COPYING/PREPARED.** For each actual rank, issues
   `r3_moe_owner_generation_upload(rank,pending_slot,pending_epoch,
   pending_digest)` once, on its rank-private asynchronous stream.
   It publishes a rank READY ACK **only after**
   `r3_moe_owner_generation_ready` confirms its real HIP event complete
   for the exact candidate epoch/digest/slot and the shared canonical
   map still matches. A missing/incomplete event cannot count. Upload
   failures publish a current-phase UPLOAD/DEVICE/CHECKSUM fault code;
   the controller aborts precommit to ACTIVE(N) with a journal fault,
   and a later attempt is a new transaction. Incomplete event readiness
   holds COPYING until its deadline, then fails READY_TIMEOUT; no GPU
   host sync is introduced.
3. **QUIESCING.** Exactly once for this transaction, closes the
   **existing** admission ledger, calls `r3_moe_wait_drained` with the
   phase deadline (waits for all parent/rank terminal events and sealed
   consumption stamps, requests helper drain, verifies every
   `lane_quiesced` identity), and requires
   `r3_moe_all_ranks_consumed(snapshot)==1` before publishing rank
   QUIESCED ACKs. The helper actors also require the matching lane drain
   ACKs before acknowledging. No second gate.
4. **COMMIT.** The existing helper-zero `Controller::commit`
   release-publishes the new active `{epoch,slot,digest}` through the
   shared PlacementControl seqlock; no separate engine writer exists.
   The helper marks the new GenerationHeader ACTIVE; the engine ranks
   acknowledge COMMITTED only after checking this exact new active
   generation and their own recorded upload readiness.
5. **RETIRING.** The engine repeats the exact closed-snapshot
   all-ranks-consumed query, verifies the same drain proof,
   `helper_drained`/lane identities and absence of any unretired
   ticket referencing old slot N; only then does each engine rank
   publish RETIRED. The helper actors must also acknowledge the
   closure. Controller collects **every** RETIRED ACK before it
   marks the old GenerationHeader RETIRED and publishes ACTIVE(N+1).
   The Part-1 ledger guard still forbids uploading the old slot
   **until** the existing `r3_moe_reopen_admission` switches the
   ledger's active slot to N+1. After reopen, that same guard allows
   upload of the now-inactive slot N on the next transaction.
   Before admission reopens the engine also reconfirms each new
   generation upload event/epoch/digest against the active map.
6. **REOPEN barrier.** The existing engine gate clears drain words, restores
   the burst state and waits for every park with one absolute monotonic
   deadline derived from `Deadlines{}.quiesce_ns`. A timeout raises the
   sticky serving fault and prevents admission reopen. Only after
   `ledger.reopen(epoch,slot)` succeeds does it release-publish a
   transaction/epoch/digest/slot-exact REOPENED ACK in the no-longer-collected
   rank-0 actor slot. Helper zero requires that exact ACK; cleared drain
   words alone cannot authorize a new generation upload. The old slot is
   kept intact until the ledger's replacement guard permits reuse.

The diagnostic remains a **no-move** path only; a candidate that changes
owner placement cannot be committed by this cycle. No helper GPU hot-path
stores, fences or extra engine bitmap reads per route were introduced.
The CPU transition does map digest checks and HIP event queries off the
serving path. A precommit diagnostic staging or RESERVED/READY fault
journals an abort and retains ACTIVE(N), with all helpers still serving;
only a closed admission gate or a committed/published bad generation is
terminal. The helper logs phase, transaction and fault code before any
epoch fault-driven exit.

The 1,000-cycle card-free shared-transport tests run each **supported**
shape separately: one helper with two ranks and three helpers with one
rank (including the tail helper). They drive the same production
`service_rank_reservation` participant, fake per-rank upload events,
missing and corrupted ACKs, consumption/retirement ordering, old-slot
guards, exact post-ledger reopen and retryable precommit timeout.
Additional card-free tests cover a live parent through abort/retry,
an invalid staged candidate, stalled parks, and non-byte-aligned
helper-strided bitmap packing. These tests cannot replace hardware
safety or timing measurements; the cycles-off 1-stream delta after
the per-route bitmap removal has not yet been measured.
