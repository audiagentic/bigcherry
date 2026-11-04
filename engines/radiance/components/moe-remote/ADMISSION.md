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
The route-mark kernel uses the immutable per-pass ticket (not the rank cell)
to validate each Request before any helper-output read. It does not reselect
the owner pointer or recheck each route's ownership; no mutable global device
active-owner selector is read. The host owns-expert callback remains a
bootstrap engine-weight-residency check; Stage 1 never changes local versus
remote eligibility.

Each rank has one private HIP upload stream, two events and pinned per-slot
staging buffers. The inactive slot's map/digest is validated against
PlacementControl and its complete GenerationHeader before packing. The
binding metadata is copied after the bitmap on that same stream; the event
follows both. Querying readiness is nonblocking and is not a serving-path
sync. The bootstrap active generation is synchronized once during startup,
after its final owner map is sealed.

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
  Main's route-mark already read Request sequence, rank, rows, hidden,
  route count/offset and (on noncombined steps) status. This slice reuses the
  sequence comparison and reads Request placement_epoch (u64) and
  generation_slot (u32) there. The rank/rows/hidden comparisons and the
  later-added layer check were redundant: publish generated those immutable
  engine-owned fields. There is no per-route owner-bit test or new barrier.
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

**Hot-path delta vs origin/main (source-level load/compare/barrier accounting;
not an ISA disassembly or a GPU timing result):**

| Kernel | Remaining source-level additions (and removals) | Synchronization and validation |
| --- | --- | --- |
| `mailbox_publish_kernel` | One thread-0 device ticket read; comparisons of the device stamp ID/epoch/slot, and generation binding epoch/digest; select one of two bitmap pointers; write four u32 ticket words to `published[]`. Per helper Request: stores for placement_epoch (u64) and generation_slot (u32). The device-scope published-count add was already present on main. Relative to main, the stamp validation is **no longer repeated by every thread**. | Uses existing block barrier, with no new barrier/fence. Wrong ticket, stamp or owner binding traps before request release. |
| `mailbox_route_mark_kernel` | Each participating thread reads four u32 ticket words from `published[]` (two ID halves, epoch, slot), compares nonzero ID/epoch and slot ≤ 1, and compares two additional mapped Request fields (epoch u64, slot u32). It reuses main's existing Request sequence comparison, but **removes** main's rank/rows/hidden comparisons and the later-added layer comparison. Existing per-route loop has no extra owner-bit lookup. | **Zero new `__syncthreads`**; previous extra shared-memory ticket exchange and block barrier removed. `matches_request` is the stream-ordered sequence/epoch/slot gate before every normal, combined and shadow helper-output reader. |
| `mailbox_wait_helpers` and combined tail | Zero extra per-request ticket/Request loads, comparisons, stores or fences vs main. | Retains existing system-scope done_sequence acquire, sequence match and status check. Helper generation corroboration occurs at post-drain CPU ACK. |
| `mailbox_add_finish_kernel` | Decodes the same four device ticket words as main; `pass_ticket` newly rejects zero epoch/invalid slot. **Removes** repeated device stamp `owns` loads/comparisons and lane-class validation. Same per-lane consumed record and one per-pass agent-scope consumed-count add as main. | Reuses original `__syncthreads_count`; no extra barrier. Stamp identity validated at publish and again at the parent seal. |
| `mailbox_seal_kernel` | Main read one device stamp and rejected consumed > published. `seal_ready` now also compares stamp ticket ID, epoch and slot to the exact parent and requires published == consumed. | Behind last combined/shadow/normal result-plane reader, before existing rank terminal event. One parent-level host release only; no new per-request system-scope traffic. |

The host admission hot path performs one acquired cached per-rank readiness
mask check, set only after successful sealed bootstrap upload/event/complete
rank initialization. It never recalculates the owner map's digest per parent.
No new GPU kernels or helper per-request system-scope stores were added. The
hardware 0.3% decode-regression target remains **unverified**.

### Round 2: one-time canonical bootstrap seal

`init` seeds slot 0 with all-local owner bytes and a digest of **that initial
map only**. The real workers then publish disjoint ownership and drop the
protected or absent runtime-layer owners. Every helper marks its final map
filtered only **after** the last such drop and layer load. Helper 0, the sole
control publisher before the epoch controller starts, waits for all helpers,
computes the final digest, and publishes GenerationHeader 0's digest and
`ready_helpers` plus PlacementControl's `active_digest` through the existing
controller's coherent odd/even seqlock (seq 0 -> odd 1 -> even 2). It is
explicitly **one-time**: subsequent publish/drop/mark calls reject the seal.
A bad candidate digest is no longer repaired by the engine.

Engine ownership lookups used during layer-map construction recheck the
seqlock before accepting generation header and owner bytes, preventing a
false fault during the single publication. In measuring-start the engine
registers its layer-map callback and can reach GPU kernel declaration
before helper 0 seals. The former eager rejection caused the plan-check
failure and subsequent launcher teardown/worker stop. Rank setup now waits
outside the registration mutex for the final even-sequence publication
within the same startup window as segment attach, then verifies the
complete filtered mask and digest; a malformed seal still fails closed.
The waiter never mutates the shared control. Helper 0 alone seals exactly
once, logging individual failed preconditions and values; the worker
diagnoses stop-before-seal separately. No serving hot-path operation changes.
The rank bootstrap
`r3_moe_mailbox_device` refuses unsealed maps. It builds its bitmap and
binding only from the sealed control epoch/digest; it synchronizes the
private HIP upload event **before** advertising that rank ready. No parent
admission is allowed before every participating rank's ready bit, so initial
GPU routing cannot use the temporary all-local digest. Later generation
uploads follow the existing exact event/ledger guards.

The card-free 1,000-cycle fixtures for the one-helper/two-rank and
three-helper/one-rank shapes now use actual `init`,
`publish_owned`/`drop_bootstrap_owner`, filtered-mask and
`SharedTransport::seal_bootstrap`. An unsealed real `EpochCycle` records
INVALID_MAP without killing serving. `stage_same_map_generation` diagnoses
each distinct failed predicate once, not once per tick.

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

The production engine-rank coordinator in `engine_rank_coordinator.h::EngineRankCoordinator::step`, called once per placement snapshot by `moe_remote_hook.cpp::placement_loop`:

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

The periodic diagnostic itself remains a **no-move** proposal source.
The slice-E participant now accepts changed-owner candidates from the same
Controller transaction without adding any forced or heat-driven proposal
policy. No helper GPU hot-path stores, fences or extra engine bitmap reads
per route were introduced.
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


## Slice D fix round 4: local-only admission, final reopen timeout, shared rank coordinator

**Handoff-declared policy and admission.** The patch's
`bc_moe_remote_wanted_upto` is the single source of truth about whether
remote mailbox kernels can be declared. The host extension resolves that
very same exported function after installing its default row bounds and
caches the conservative answer for all permitted step rows. For
`R3_MOE_REMOTE_MIN_ROWS=1 R3_MOE_REMOTE_MAX_ROWS=0`, the function
says no mailbox can run: a scheduler admission returns success with a
zero ticket without requiring initialized HIP ranks. The zero-ticket
per-pass callback is also a local-only no-op. With remote handoff
possible, `engine_admission_gate.h::admission_bootstrap` requires the
entire actual rank-ready mask before allowing admission; a partial or
uninitialized rank always fails. No independent row-policy parser or
second environment flag is added. The path costs exactly **one cached
atomic acquire of the rank mask** per remote admission, and **zero**
such reads for local-only admission. No mapped Request reads, GPU round
trips or additional kernel operations are introduced.

**Post-commit final ACK.** The helper's
`EpochCycle::awaiting_reopen_` now receives a transaction-bound
absolute deadline at RETIRING -> ACTIVE, derived from
`Deadlines::retire_ns`. A delayed ACK from an older transaction does
not satisfy `reopened_exactly`. Missing the current REOPENED ACK at
the deadline journals a `REOPEN_TIMEOUT` terminal placement fault,
stops future cycles without returning a worker-loop error, and leaves
the helper lanes, committed ACTIVE generation and serving admission intact. A deadline counter overflow faults placement.
The controller does *not* set shared `PlacementControl.fault_code`
for this already-committed, post-ledger diagnostic failure, as doing
so would shut down serving on the healthy committed generation.

**Rank coordinator, production and tests.** `EngineRankCoordinator::step`
is the sole participant producing rank RESERVED, READY, QUIESCED,
COMMITTED, RETIRED and final REOPENED publications. The production
`placement_loop` constructs one coordinator and a
`LiveRankOperations` adapter: its callbacks hold the real rank-setup
mutex when reading HIP bindings, query actual upload events, invoke
`r3_moe_close_admission`/`r3_moe_wait_drained` and the existing
ledger's result-consumed and generation-reuse proofs, and reopen through
`r3_moe_reopen_admission`. The latter no longer publishes REOPENED
itself: the shared coordinator publishes exactly once after its
`reopen` callback succeeds. The fake-device test supplies the same
interface with real host-side ledger closure/proofs, per-lane
`lane_quiesced` checks and fake upload events. The 1,000-cycle loops
for both legal shapes (one helper/two ranks, three helpers/one rank)
use the actual production phase dispatcher; they cannot synthesize
rank ACKs on their own.

**Bootstrap lock order.** Engine rank setup delegates to
`owner_generation::with_bootstrap_rank_lock`. Its seal wait is outside
the engine mutex; the user callback executes the entire verified rank
setup under that mutex. The concurrent startup test calls that same
boundary and publishes an outstanding runtime layer map while the rank
waits. Reversing this order is caught by the test: the layer callback
cannot acquire the setup mutex. The ready bit is still published only
after the fully sealed map and private-stream upload event.

The card-free policy-to-admission test compiles the generated
`rad0002` policy itself and exercises disabled and enabled rows through
the same admission predicate. The epoch tests include missing and
delayed older REOPENED ACKs. Removing the production RESERVED handler,
the REOPENED publisher, or changing the shared bootstrap lock order
makes the corresponding tests fail. No hot-path publish, route-mark,
wait, add/finish, seal, or helper GPU kernel was modified.

## Slice E parts 1–3: expert-weight arena, two generation tables, runtime free-memory gate

The budget in layouts.toml / R3_MOE_WORKER_MIBS is **an expert-weight
budget**, not a whole-card VRAM budget. The helper and the GPU-free
bc::experts::plan_owners / verify_owner_map use the same budgets and
select_owned policy, with no table/lane/staging subtraction. The
weight slot stride is the largest four-plane expert with each plane 256-byte
aligned. Exactly the selected experts are packed into one device arena
at id * stride. The slice-B residency ledger charges the physical strides.
Preflight and allocation failures cannot charge a phantom ACTIVE slot.
A **separate** one-stride incoming staging allocation resides in the card's
remaining VRAM and has one extra ledger ID for a future move. It cannot be
taken by bootstrap.

Before allocating the arena, the worker calls hipMemGetInfo. It took
an initial free-memory sample before preparing enabled lanes (which allocates
real plugin scratch via RadScratchFn and every corresponding GPU buffer).
It compares lane_bytes + two pointer-table banks + frozen runtime
slot map + one incoming staging slot + selected-expert arena against
that initial free memory, and the *not-yet-allocated* subset against the
runtime's **current** free memory. This avoids double-counting already
allocated lane buffers. A shortage or size overflow fails startup closed
with the required bytes, free samples and shortfall; no experts are silently
dropped. HIP allocation errors also abort startup. Both immutable GPU
pointer banks contain separate arrays for each generation and have the same
resident experts at bootstrap. The runtime eligibility slot-map stays frozen.

The real Flash-Next W4 49-layer/512-expert geometry has 2,534,400 bytes
per padded four-plane expert (zero 256-byte padding overhead). The full
expert-weight budgets select 9,320 / 9,320 / 5,931 experts on
the 22,528 / 22,528 / 14,336 MiB helper layout, as on main.
Card-free tests parse every production layout and compare ordered expert
IDs and bytes from the helper selector against the manifest planner,
and exercise the manifest verifier itself on a decoded example. No GPU
run or empirical free-VRAM observation has been made.

The still-unpublished v23 Request is revised on both sides together:
generation_slot moves from byte +204 (64-byte line 3) to byte +36
(line 0), swapping with diagnostic-only row_base. The unfused
lane_select_tables already validates sequence/rank/rank_count/layer/
rows/hidden/route_count and activation strides in **lines 0 and 1**.
The fused lane_front_small validates those same lines and writes
trace[3] in **line 2**. Neither needs line 3 for generation selection
after this revision. placement_epoch remains at +232 in line 3:
the helper CPU uses it at drain, not in the selection kernels.
No new per-request system-scope stores/fences/host round trips or GPU
block barriers were added. The engine publisher uses the shared Request
field names and offset assertions; there is no older layout path.

The production changed-map implementation is described in the part-4
section below. The fake GPU tests prove the CPU slot/protocol contract;
the HIP upload path still needs hardware safety and timing measurements.


## Slice E part 4: live changed-owner staging and retirement

A changed map enters through the **same** helper-zero
`Controller::propose(candidate_digest)` and the existing PROPOSED/RESERVING
protocol. The periodic `epoch_cycle_s` experiment still proposes *only*
same-map generations. There is no forced move or heat policy here, and no
new engine-side transaction path. `placement_candidate_valid` is shared
by the helper and engine-rank participants: it verifies old and new canonical
owner bytes, digest/header/journal identity, valid owner indices and exact
epoch/slot pairing. A current-phase fault aborts before quiescence.

Per helper, `worker_moves::Participant` compares the two canonical owner
maps and tracks retained, gained and lost expert IDs. `worker_slots::Pool`
and the slice-B residency ledger are the only physical-slot authority. A
retained expert keeps its already allocated slot. An incoming expert reserves
an unreferenced slot; the budget and physical slot count still include every
old slot, even one that will leave in this transaction. With no free slot,
the current-phase RESERVED ACK carries `CAPACITY` (sticky across polling)
and the controller aborts forward to unchanged ACTIVE(N). The helper drains
any precommit copy and cancels its reservations after that abort.

During COPYING the GPU runner decodes each gained expert's existing model
planes into one pinned host staging stride. It issues one H2D transfer of
the packed expert planes (up to one stride) on the **private nonblocking stream**, records a **fresh
disabled-timing HIP event per expert**, and queries that event. There is
at most one outstanding copy, matching the single separately budgeted
incoming slot. Only after that event completes are the expert's four
planes marked complete/READY in the residency ledger. If several free
physical IDs exist they are copied sequentially, not concurrently. The
candidate's per-layer pointer arrays are built in the *inactive* generation
bank **after every gained expert is ready**; their device allocations are
reused on later epochs (no unbounded table allocations). Failed uploads,
bindings or event checks cannot publish helper READY.

The engine's runtime expert-index/eligibility map is immutable from
bootstrap and remains **one** GPU array, exactly as in part 3. Retained
experts and newly gained experts use their already-declared engine
`layer_slot` value. The candidate pointer table omits experts leaving
this helper, and includes the fully uploaded arrivals. A lane picks its
pointer table using `Request::generation_slot` at +36, already read by
the part-3 validation kernels; the engine slot mapping and routing kernels
do not change. **Relative to part 3 there are zero additional per-request
reads, stores, fences, host round trips, branches or HIP launches.** All
planning, checksum, HIP event queries and table construction are confined
to control-plane placement polling. The active bank and its physical
weights are not written while ACTIVE(N) can still serve.

COMMIT activates incoming physical slots and marks departing slots
RETIRING **without releasing their ledger charges or weights**. The helper
must observe its lane drain closure, the engine coordinator's closed
all-ranks-consumed proof and all rank/helper RETIRED ACKs. Only after the
engine actually reopens admission for N+1 and publishes an
epoch/digest/slot/**transaction-exact REOPENED** proof does each helper
reclaim its departing physical slots. The next candidate can then reserve
them; a missing reopen proof leaves them held. The pool's device
allocations and the per-generation table buffers remain fixed.

The expanded `r3_moe_epoch_test` runs changed-map transactions through the
actual Pool, residency ledger, Controller, EpochCycle helper participant and
EngineRankCoordinator with fake GPU rank operations and fake independent
copy events. It exercises CAPACITY and rollback before COMMIT, immutable
old-bank service during COPYING, premature READY and reclamation refusal,
and 1,000 alternating move/return cycles on each supported helper/rank
topology (including the smaller tail allocation). Mutation checks
individually break the event-before-READY, bank-before-READY,
active-bank, capacity, sticky-fault and REOPENED gates and require a failing
card-free test. GPU execution and real VRAM/timing validation are not
claimed by those tests.

## Slice E part 5: review fixes (helper migration)

**Owner classification is sealed at bootstrap.** The shared
placement_candidate_valid predicate compares each old/new canonical
owner byte, journal, digests and indices. Engine-local experts remain
engine-local; experts initially owned by a helper can move only to
another helper. The engine never allocates local weights for an expert
it did not load at bootstrap. Engine-rank coordinator, helper epoch
driver and slot migration participant invoke this same predicate
before RESERVING. A single helper has no legal changed-owner move;
the one-helper/two-rank fixture exercises unchanged-map cycles and
both prohibited directions, while two- and three-helper fixtures
exercise real helper-to-helper swaps.

**Committed budget differs from transient physical capacity.** The slot
ledger still charges ACTIVE, RESERVED, READY and RETIRING physical slots
against the expert budget plus exactly one incoming staging stride
until reclaim/cancel. Before reserving, the participant separately
projects charged - departing + arriving expert bytes and requires them
to fit the configured *expert-only* budget, excluding staging. An
arrival at an already full helper without a departure is CAPACITY
and aborts forward; a one-for-one full-budget swap stages in the
extra physical slot. Tests assert both bounds across 1,000 cycles.

**Preallocated full-budget arena.** Runner::allocate now sizes its
weight arena by slot_plan.expert_capacity * stride, not the number
of bootstrap residents. Pool::prepare allocates the entire arena and
incoming slot at startup, including helpers starting with no experts;
the hipMemGetInfo check includes both. The card-free zero-bootstrap
test reserves, copies and activates its first expert without calling
bootstrap. For flashnext-4gpu at 2,534,400 bytes/expert, arenas are
23,620,608,000 bytes (9,320 slots) for each of helpers 0 and 1,
and 15,031,526,400 bytes (5,931 slots) for helper 2. Main already
selected exactly these weight bytes for this layout. Each helper
also needs one separate 2,534,400-byte incoming staging slot. On
underfilled layouts, allocation may need more VRAM than the previous
selected-only arena and can now fail the startup free-memory check
rather than advertise nonexistent future capacity.

**Exact transaction closure is a proposal precondition.** Controller
tracks the previous transaction until a committed transaction has
the engine post-ledger REOPENED ACK and CLEANED ACKs from every helper
after reclamation. For a precommit forward abort, each helper CLEANED
ACK follows its upload synchronization and reservation cancellation,
and every engine rank CLEANED follows completion of any candidate
bitmap upload. These use existing per-actor ACK slots, release
publication, and exact transaction/active-epoch/digest/slot identity.
RankOperations::abort_upload_done checks an already-queued rank
upload event after the candidate was removed; normal READY must not
be reused after abort because the pending map identity no longer
exists. Both Controller::propose and same-map auto-staging refuse
to race an outstanding cleanup. Missing proof holds future placement,
not unrelated ACTIVE serving.

**Bounded cleanup and abort event errors (slice E part 6).** The
controller starts a monotonic absolute cleanup deadline as the transaction
becomes ACTIVE: the existing `Deadlines::retire_ns` for a committed
RETIRING->ACTIVE transition (120 s by default), or
`Deadlines::copy_ns` for precommit abort cleanup (180 s by default).
Helper zero checks exact CLEANED identities on each ACTIVE tick, including
when the synthetic epoch period is zero. Lost, duplicate, stale, and
mismatched ACKs cannot prove cleanup. A missing participant at expiration
halts *placement only*, leaving ACTIVE serving and the unsafe inactive
bank/slot protected. The journal retains the original abort reason in
`fault`, and records the terminal cause and first missing actor in
`cleanup_fault` and `cleanup_actor` (previously reserved journal
words); `control.fault_actor` also identifies that participant.
`control.fault_code` stays zero so admitted parent steps remain valid.
Helpers may retry an incomplete precommit upload drain without publishing
CLEANED or setting `helper_fault`; the controller's deadline handles a
participant that never finishes. A lost REOPENED ACK halts each helper
locally after its reopen deadline; only helper zero owns the controller
and journals REOPEN_TIMEOUT, while other helpers keep old residency
untouched.

An aborted rank bitmap event returning a HIP query/device/identity error
is `RankResult::DeviceError`, not Pending. The production rank participant
publishes a transaction- and ACTIVE-identity-exact CLEANED-phase DEVICE
fault, which helper zero records immediately as a terminal placement-only
condition. The rank coordinator remembers the error so a later success
cannot certify that inactive generation as CLEANED; the old serving
generation stays operational. Only a genuine Pending event retries,
bounded by the abort cleanup deadline.

All checks run on the placement control plane, not in lane processing.
Against part 3 there are zero additional per-request Request-field
reads, system-scope stores/fences, host round trips, barriers or GPU
launches. The existing generation-slot read still selects the
immutable pointer bank. These CPU tests are not GPU timing evidence.
