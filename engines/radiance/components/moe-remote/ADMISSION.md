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
periodic test driver or new environment variable.

**Single RSC06 integration point:** the marked EpochCycle construction in
r3_moe_worker.cpp currently passes chrono::seconds{0}. After the bc_serving
schema merges, pass the validated typed
`experiment.helper.epoch_cycle_s` value there as a chrono duration on
**every helper**, via the typed helper launch configuration. Preserve zero
as disabled; reject negative, non-finite or unrepresentable durations in
that schema/decoder. No environment bridge is needed. The controller,
shared transport, helper polling and engine protocol participant are
already wired. Routing through two owner generations and live helper
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

Hardware acceptance remains required: an engine run with admission closed
and reopened periodically under 6 streams with MTP, expecting no hang,
identical single-stream text and step time within noise against main.
Enable the typed epoch-cycle period after the RSC06 integration above.
Test power saving both enabled and disabled, and record the ledger high-water
for the matched workload. Card-free checks do not establish this hardware
acceptance; no GPU validation was performed in this revision.
