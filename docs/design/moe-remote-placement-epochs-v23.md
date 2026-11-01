# Remote-MoE live placement: mailbox protocol v23 (Stage 1)

Status: implementation specification; target milestone **RHP11 + RHP06 + RHP04 Stage 1**, gated by RHP12. Protocol **v23 only**; do not read, translate, or dual-write v22 segments. This design moves **helper-owned regular experts among configured helper GPUs during service**. It does not change the engine-local/helper split, engine placement, model weights, router, quantization, helper inventory, or numerical kernels. Every movement is staged from each destination helper's immutable host weight source, and every activation is a forward-only placement epoch. Layout, eligibility, capacity, and source availability are inputs, not device-index/architecture constants. Deterministic mode forbids proposal and commit.

## 1. Current v22 facts and the immutability audit

All paths below are relative to `engines/radiance/components/moe-remote/`. References are **file::symbol**, and describe v22 unless marked **v23 change**.

| Fact / invariant presently relied on | Code evidence and consequence |
| --- | --- |
| `kVersion == 22`; `kMaxEngineRanks == 2`, `kMaxHelpers == 4`, `kMaxSlots == 4`; `slot_count = max(kMaxEngineRanks, helpers)`; `kLanes = kMaxSlots * (4 small classes + 1 large)`; `kLaneSlots == 4`. | `moe_remote_shm.h::kVersion, slots_for, lane_of, layout_for, init, valid`. Actual serving engine world may be 1 even though `Header::rank_count` is set to 2. Never use `Header::rank_count` as the active engine-rank participant count. |
| Ownership is a set of **per-helper bitmaps** at `Header::owner_offset`; `owner_slot_bytes = kMaxLayers * ceil(experts / 8)`, sized using the model's expert count; bit `layer * expert_count + expert` denotes ownership. | `moe_remote_shm.h::layout_for, owner_bits, owner_bit, owns, set_owned, clear_owned`. It is not a canonical one-byte-per-expert owner map and no live generation is present. |
| Each helper independently constructs a deterministic start-up `Owned` candidate list, publishes its bitmap, then removes protected/undeclared experts after runtime layer maps arrive. | `worker_owners.h::select_owned, publish_owned`; `r3_moe_worker.cpp::main` (publish, `clear_owned` and `load_layer` loop). These mutations are initialization-only; there is no post-start rendezvous or ownership epoch. |
| The **engine's** `publish_layer_map` copies the model-global `int32_t[layer][expert]` runtime slot map into `Header::layer_map_offset` (negative means local/protected); it publishes `layer_map_ready[layer]`, then waits for `layer_map_consumed[layer]`. A layer containing any negative slot is made entirely local. | `moe_remote_hook.cpp::publish_layer_map`; `moe_remote_shm.h::layer_slots, layer_slot`; `r3_moe_worker.cpp::main` (runtime-map filtering). A second changed map is only logged, not replaced. The slot map is **not** the helper-owner map; it is an immutable eligibility/input mapping for Stage 1. |
| Engine registration exposes only `layer_map`, `owns_expert`, `burst`. Ownership lookup scans static helper bitmaps; the engine may omit helper-owned weights from its own residency plan at start. | `moe_remote_hook.cpp::EnginePlugin, owner_query, register_mailbox`; `engines/radiance/patches/rad0002_moe_remote_handoff/patch.py::_BC_MOE_REMOTE_H, _BC_MOE_REMOTE_CPP (bc_moe_remote_skip_unit)`. The current ABI has **no step admission, completion or epoch callback**. `burst_transition` is an idle/burst power handshake, not a model-step fence. |
| For each engine rank, `r3_moe_mailbox_device` allocates a single device bitmap `RankState::owner`, then **once** calls `hipMemcpy(..., owner_bits(map,0), slot_count * owner_slot_bytes, H2D)`; `s.dev.owner` remains that allocation. | `moe_remote_hook.cpp::RankState, r3_moe_mailbox_device, close_mailbox`; `moe_remote_hook.h::R3MoeMailboxDevice`. There is no live device ownership upload or double buffer. |
| Publish routes by reading `d->owner`, one bitmap per helper, and checking the *separate* runtime `slot_map`. GPU writes routes, request metadata and `q->sequence`, system-fences, and release-stores `LaneSignal::request_sequence`. | `moe_mailbox_ops.hip::publish_launch, mailbox_publish_kernel, owner_has, step_slot, store_sys_rel`. The launch has no placement token. `route_mark`, waits, and add use request sequence, class and result planes, never ownership epoch. |
| The request ring is `Request[20 * 4]`, fixed 256-byte entries; the route/activation/result **bulk planes are per request slot**, not per ring element. A one-rank/multiple-helper step uses each helper's slot and shares activation slot 0. | `moe_remote_shm.h::Request, layout_for, lane_request, result_index`; `moe_mailbox_ops.hip::mailbox_stage_kernel, mailbox_publish_kernel, mailbox_add_rows_kernel`; `r3_moe_worker.cpp::main` (activation-offset alias). Therefore plane reuse must be fenced independently of ring-entry reuse. |
| Small helper lanes enqueue four waits **ahead of receiving requests**, with a fixed rows class; the large lane enqueues at arrival. Power-save burst closure drains unconsumed waits by temporarily limiting publishes and synchronizing the stream. | `worker_lanes.h::Lanes::run_small, run_large, park_lane, wait_burst`; `moe_remote_gpu.h::Runner::enqueue_lane, set_publish_limit, sync_lane`. `power_save == false` leaves waits armed. A mere host `active_epoch` store would not change the already-enqueued table pointers or guarantee no old request executes. |
| Helper weight storage is a monotonically filled arena; `load_layer` uploads gate/up and down planes, sets `Expert::live`, then builds one runtime pointer-table set. `d_layer_tables_` and `d_expert_slot_` are **single-generation GPU tables**. | `moe_remote_gpu.h::Runner::allocate, load_layer, upload_weight, build_layer_table, build_one_table, owns, enqueue_lane`. `load_layer` returns early for an already-loaded layer; `upload_weight` uses synchronous H2D copies and `arena_take`. No live slot allocator, retire event, generation table, or asynchronous move exists. |
| The helper GPU writes its result plane and chunk sequence with system release; helper 0's CPU combine waits on each helper's chunk sequence, fills the combined output plane, publishes `chunks.done[c]`, and finally `combine.done`. The engine can consume chunks **before** `combine.done`. | `moe_lane_reduce.h::lane_chunk_publish, lane_reduce_tokens, lane_publish_done`; `worker_combine.h::combine_step, CombinePool::combine`; `moe_mailbox_ops.hip::mailbox_add_rows_kernel, mailbox_add_tail_wait_kernel`. **Neither** `chunks.done` **nor** `combine.done` proves the engine has finished reading the combined plane. |
| Noncombined waits check exact lane `done_sequence`; combined adds wait per `chunks.done`, then tail-wait on helper completion. `mailbox_add_finish_kernel` clears device `published` but writes **no host-visible consumed ACK**. | `moe_mailbox_ops.hip::mailbox_wait_helpers, mailbox_add_wait_kernel, mailbox_route_mark_kernel, mailbox_add_rows_kernel, mailbox_add_tail_wait_kernel, mailbox_add_finish_kernel, add_launch`. This is an old-output-plane lifetime hole for a control-plane commit. |
| Helper fault is sticky; the engine watches process identity/PID lifetime. Host waits use `Header::notify` + futex generation; GPU cross-process flags use system-scope release/acquire and explicit `__threadfence_system()`. | `moe_remote_shm.h::load, store, notify, wait_for, raise_fault`; `moe_remote_hook.cpp::register_mailbox, burst_transition`; `moe_mailbox_ops.hip::store_sys_rel, load_sys_acq, mailbox_publish_kernel`; `worker_lanes.h::Lanes::stop_and_join`. Preserve these domains; a CPU-only release does not by itself flush an earlier GPU stream. |
| `worker_heat.h::Heat::tick/observe` and `propose` only calculate a plan. Present `propose` requires strictly cheaper destination scalar cost, has single moves only, and cannot balance equal-cost helpers. | `worker_heat.h::validate, propose, delta_matches`. Stage 1 must add the per-layer peak-load objective, bounded swaps, and validation before ever handing a plan to the protocol. |

### v22 reserved and offset audit

Offsets are byte offsets **within the named struct**, under the current `kMaxHelpers=4` and `kMaxSlots=4` bounds (checked 2026-11-01 with compile-time `offsetof` asserts against v22; keep those asserts in the v23 change):

| Struct | v22 reserved region | Offset / bytes | v23 treatment |
| --- | --- | --- | --- |
| `Request` | `reserved32` | 204 / 4 | `generation_slot` |
| `Request` | `reserved` | 232 / 8 | `placement_epoch` |
| `Request` | tail padding of `alignas(64)` 256-byte request | 240 / 16 | `completion_epoch`, `completion_slot`, `reserved_tail` |
| `Header` | `reserved32[24 - 3*kMaxHelpers - 1]` = 11 words | 2748 / 44 | preserve as zero; new explicit fields **after** `chunks` |
| `LaneSignal` | `reserved[13]` | 12 / 52 | completion and result-consumed stamps |
| `CombineSignal` | `reserved[19]` | 52 / 76 | result-consumed stamp |
| `Header` | unused space within the `alignas(kPage)` first page | `chunks` is at 2944 (320 bytes) and ends at 3264; `sizeof(Header)` is 4096, so 832 bytes remain | append v23 offset/size descriptors |

Evidence: `moe_remote_shm.h::Request, Header, LaneSignal, CombineSignal, CombineChunks, layout_for`. The old owner/slot fields remain distinguishable: `Header::layer_map_offset` is still the **runtime** expert-to-execution-slot map. The v22 `Header::owned_count/owned_bytes/owner_offset/owner_slot_bytes` are tombstoned in v23 (set to zero and checked zero); they are not a second ownership source.

## 2. Wire format and exact layout

No compatibility mode. Keep `kMagic`, change `kVersion = 23`; reject every other version in `valid/valid_self` **before pinning** (`moe_remote_shm.h::valid, valid_self`; `moe_remote_hook.cpp::register_mailbox`; `worker_segment.h::Segment::open`).

Definitions below are normative C++ layout, standard-layout trivially-copyable structs, with fixed-width types; no pointers, vectors, `std::atomic` objects, or process-local handles in shared memory. All `reserved` words are zero on creation and validated when the transaction starts.

```cpp
constexpr uint8_t kOwnerLocal = 0xff;  // engine/local; never migrates in Stage 1
constexpr uint32_t kVersion = 23;
enum class PlacementState : uint32_t {
    ACTIVE=0, PROPOSED=1, RESERVING=2, COPYING=3,
    PREPARED=4, QUIESCING=5, COMMIT=6, RETIRING=7
};
enum class AckPhase : uint32_t {
    NONE=0, RESERVED=1, READY=2, QUIESCED=3, COMMITTED=4, RETIRED=5
};
enum class GenerationState : uint32_t { EMPTY=0, STAGING=1, READY=2, ACTIVE=3, RETIRED=4 };

// Shared coordinator words: only the engine coordinator writes state,
// epoch/slot/digests/deadline/masks and journal_index.
// Participants write their own PlacementAck and increment notify.
struct alignas(64) PlacementControl {
    uint32_t seq;                     // +0: odd writer / even stable snapshot
    uint32_t state;                   // +4 PlacementState
    uint32_t active_slot;             // +8, 0/1
    uint32_t pending_slot;            // +12, 0/1 or UINT32_MAX
    uint64_t active_epoch;            // +16, starts at 1
    uint64_t pending_epoch;           // +24, 0 without transaction
    uint64_t active_digest;           // +32
    uint64_t pending_digest;          // +40
    uint64_t transaction_id;          // +48, monotonically increasing
    uint64_t deadline_ns;             // +56, CLOCK_MONOTONIC absolute
    uint32_t ready_helpers;           // +64; READY ACKs validated by engine
    uint32_t ready_ranks;             // +68
    uint32_t quiesced_helpers;        // +72
    uint32_t quiesced_ranks;          // +76
    uint32_t commit_helpers;          // +80
    uint32_t commit_ranks;            // +84
    uint32_t retired_helpers;         // +88
    uint32_t retired_ranks;           // +92
    uint32_t fault_code;              // +96, FaultCode
    uint32_t fault_actor;             // +100: kind<<16 | index
    uint32_t notify;                  // +104: futex notification generation
    uint32_t expected_helpers;        // +108: (1<<helper_count)-1
    uint32_t expected_ranks;          // +112: (1<<actual_world_size)-1
    uint32_t journal_index;           // +116: current journal record, 0/1
    uint32_t reserved[34];            // +120..255
}; // sizeof == 256

// One header immediately followed by map_bytes *uint8_t* owners.
// The slot is not rewritten while its epoch is ACTIVE or RETIRING.
struct alignas(64) GenerationHeader {
    uint64_t epoch;                   // +0
    uint64_t digest;                  // +8
    uint32_t map_bytes;               // +16, == layer_count*expert_count
    uint32_t layer_count;             // +20
    uint32_t expert_count;            // +24
    uint32_t slot;                    // +28, 0 or 1
    uint32_t ready_helpers;           // +32, immutable when READY
    uint32_t ready_ranks;             // +36, immutable when READY
    uint32_t state;                   // +40 GenerationState
    uint32_t reserved[5];             // +44..63
}; // sizeof == 64

// Per-participant, single-writer ACK, following release-store of phase.
// Indices [0, kMaxHelpers) are helpers; [kMaxHelpers, +kMaxEngineRanks) are engine ranks.
struct alignas(64) PlacementAck {
    uint64_t transaction_id;          // +0
    uint64_t epoch;                   // +8
    uint64_t digest;                  // +16
    uint64_t drained_step;            // +24, last step completely retired locally
    uint32_t phase;                   // +32 AckPhase, release-published LAST
    uint32_t generation_slot;         // +36
    uint32_t fault_code;              // +40
    uint32_t reserved32;              // +44
    uint64_t reserved[2];             // +48..63
}; // sizeof == 64

struct alignas(64) JournalRecord {
    uint64_t transaction_id;          // +0
    uint64_t before_epoch;            // +8
    uint64_t after_epoch;             // +16
    uint64_t before_digest;           // +24
    uint64_t candidate_digest;        // +32
    uint64_t start_ns;                // +40
    uint64_t commit_ns;               // +48, zero until committed
    uint64_t finish_ns;               // +56
    uint32_t state;                   // +64, last durable transition
    uint32_t fault;                   // +68
    uint32_t old_slot;                // +72
    uint32_t new_slot;                // +76
    uint32_t ready_helpers;           // +80
    uint32_t ready_ranks;             // +84
    uint32_t commit_helpers;          // +88
    uint32_t commit_ranks;            // +92
    uint32_t retire_helpers;          // +96
    uint32_t retire_ranks;            // +100
    uint32_t reserved[6];             // +104..127
}; // sizeof == 128
struct alignas(64) PlacementJournal {
    JournalRecord entry[2];           // sizeof == 256
};
```

**Request (unchanged 256-byte stride, intentionally changed semantics):** replace the field at offset 204 with `uint32_t generation_slot`; replace the field at 232 with `uint64_t placement_epoch`; fill existing tail bytes 240..255 with `uint64_t completion_epoch`, `uint32_t completion_slot`, `uint32_t reserved_tail`. The engine zeroes completion fields, writes the request epoch and slot **before publishing its request sequence**, and the helper publishes its completion epoch/slot **before** the done sequence. `Request::completion_sequence` also becomes a GPU completion stamp, matching `LaneSignal::done_sequence` (`moe_remote_shm.h::Request, LaneSignal`; `moe_mailbox_ops.hip::mailbox_publish_kernel`; `moe_lane_reduce.h::lane_publish_done`).

**LaneSignal (still 64 bytes):** replace `reserved[13]`, starting at offset 12, with `uint32_t consumed_sequence` (+12), `uint32_t done_slot` (+16), `uint32_t consumed_slot` (+20), `uint64_t done_epoch` (+24), `uint64_t consumed_epoch` (+32), `uint32_t reserved[6]` (+40). Done is a helper GPU release; consumed is an **engine GPU system-scope release after its last read of that request's result plane**. Waiters require both sequence and epoch/slot; these words cannot be reused to certify another request (`moe_remote_shm.h::LaneSignal`; v23 changes to `moe_lane_reduce.h::lane_publish_done` and `moe_mailbox_ops.hip::mailbox_add_finish_kernel`).

**CombineSignal (still 128 bytes):** replace its `reserved[19]` at +52 with `uint32_t result_consumed_generation` (+52), `uint32_t result_consumed_slot` (+56), `uint32_t reserved_align` (+60), `uint64_t result_consumed_epoch` (+64), `uint32_t reserved[14]` (+72). `CombineChunks::seq/done` remain as v22, **never** lifetime ACKs (`moe_remote_shm.h::CombineSignal, CombineChunks`; `worker_combine.h::combine_step`; `moe_mailbox_ops.hip::mailbox_add_rows_kernel`).

**Header and regions:** keep its existing v22 fields and first-page layout for the separate runtime layer map and lane arrays, but deprecate/zero `owned_count, owner_slot_bytes, owned_bytes, owner_offset`. Append, immediately after `Header::chunks` (offset 3208), the following v23 descriptors, in this exact order:

```cpp
uint64_t placement_offset;           // Header +3264 (directly after chunks)
uint64_t journal_offset;             // +3272
uint64_t ack_offset;                 // +3280
uint64_t generation_offset[2];       // +3288, +3296
uint32_t owner_map_bytes;            // +3304: layer_count*expert_count
uint32_t generation_slot_bytes;      // +3308: page-aligned 64+map bytes
uint32_t placement_control_bytes;    // +3312: sizeof(PlacementControl)
uint32_t placement_ack_count;        // +3316: kMaxHelpers+kMaxEngineRanks (ends +3320, inside the 4096-byte page)
```

The existing `Header::reserved32[11]` remains zero. `sizeof(Header)==4096`, since `Header` is page-aligned. Rebuild `Layout`, `init`, `valid`, `valid_self`, and `segment_shape` as a **breaking** layout update; erase the old mutable `owner_bits/owns/set_owned/clear_owned` API, and replace with `generation_map(base,slot)`, `owner_at(base,slot,layer,expert)`, `placement_control(base)`, and `placement_ack(base,actor)`. No v22 region is accepted or copied (`moe_remote_shm.h::Layout, layout_for, init, valid, valid_self, owner_bits`; `worker_shape.h::segment_shape`).

Change `layout_for` signature to `layout_for(const LayoutShape&, uint32_t layer_count, uint32_t slots, Layout*)` so both generations are sized from the **actual** layer/expert counts; `layer_count` must be in `[1,kMaxLayers]`, `slots==slots_for(helper_count)` at init. Use the existing checked `mul/add/page` helpers, and reject any 32-bit size overflow. Region order:

```text
[0,4096) Header
ring          = 4096
placement     = align_page(ring + sizeof(Request)*kRingSlots)
journal       = placement + 256
acks          = journal + 256                     // 6 entries * 64 = 384 bytes
generation[0] = align_page(acks + (kMaxHelpers+kMaxEngineRanks)*64)
map_bytes     = checked_mul(layer_count, shape.experts)
slot_bytes    = align_page(64 + map_bytes)
generation[1] = generation[0] + slot_bytes
layer_map     = generation[1] + slot_bytes        // existing int32[kMaxLayers][experts] runtime slot map
route/activation/result slots, then segment bytes // existing per-slot order and page alignment
```

For each `generation[g]`, `GenerationHeader` is at its region base and the contiguous canonical map starts at `generation[g]+64`, with `owner[layer*expert_count+expert]`: `0xff` = engine-local or protected, `0..helper_count-1` = one helper. **Exactly one owner byte**, including for experts not placed remotely. A helper with zero active experts remains a valid destination if its capability, source and budget checks pass; no topology-specific cases. The v22 `layer_map` still uses its separate runtime `int32` encoding and is not generation-swapped (`moe_remote_shm.h::layer_slots, layout_for`; `moe_remote_hook.cpp::publish_layer_map`).

For the current constants, `kRingSlots=80`, hence `placement=24576`, `generation[0]=28672`, and the control + journal + ACKs use **896 bytes within one placement page**. A 48-layer, 512-expert map contains 24,576 bytes; each generation slot is `align_page(64+24576)=28672` bytes, so `generation[1]=57344`, `layer_map=86016`, and the following route region starts after the runtime layer-map allocation (not at 86016). These numbers are **examples only**; the runtime layout is shape-derived (`moe_remote_shm.h::layout_for, kRingSlots, kPage`).

Digest: FNV-1a 64 over an explicit little-endian byte serialization of `shape_hash` (u64), `layer_count` (u32), `expert_count` (u32), `helper_count` (u32), then **all** canonical owner bytes in model-global order. Never hash padding, pointers or uninitialized bytes. Each helper checks its own eligibility/source and validates this digest before READY. `GenerationHeader` is written by the engine, marked `READY` with release semantics only after the complete map and masks are present; once READY, contents cannot change until that slot is safely reused.

Add all of these mandatory compile-time assertions in the shared header (compiled by normal C++ and by HIP), plus runtime checks against reconstructed offsets/shape:

```cpp
static_assert(kVersion == 23 && kRingSlots == kLanes*kLaneSlots);
static_assert(sizeof(Header)==4096 && alignof(Header)==4096);
static_assert(sizeof(Request)==256 && alignof(Request)==64);
static_assert(offsetof(Request,generation_slot)==204);
static_assert(offsetof(Request,placement_epoch)==232);
static_assert(offsetof(Request,completion_epoch)==240);
static_assert(offsetof(Request,completion_slot)==248);
static_assert(sizeof(LaneSignal)==64 && offsetof(LaneSignal,done_epoch)==24);
static_assert(offsetof(LaneSignal,consumed_epoch)==32);
static_assert(sizeof(CombineSignal)==128);
static_assert(offsetof(CombineSignal,result_consumed_generation)==52);
static_assert(offsetof(CombineSignal,result_consumed_epoch)==64);
static_assert(sizeof(PlacementControl)==256 && alignof(PlacementControl)==64);
static_assert(offsetof(PlacementControl,active_epoch)==16);
static_assert(offsetof(PlacementControl,ready_helpers)==64);
static_assert(offsetof(PlacementControl,notify)==104);
static_assert(sizeof(GenerationHeader)==64);
static_assert(sizeof(PlacementAck)==64);
static_assert(sizeof(JournalRecord)==128 && sizeof(PlacementJournal)==256);
static_assert(offsetof(Header,placement_offset)==3208);
static_assert(offsetof(Header,owner_map_bytes)==3248);
static_assert(sizeof(PlacementControl)+sizeof(PlacementJournal)+
              (kMaxHelpers+kMaxEngineRanks)*sizeof(PlacementAck)<=kPage);
```

## 3. Ownership, coordinator and epoch state machine

**Coordinator: engine hook** (`moe_remote_hook.cpp::HookState, register_mailbox, r3_moe_mailbox_device, burst_transition`). This process owns the per-rank GPU copies, has the engine registration callback and process-death watcher, and is the only place that can stop admission **before** new forwards are issued. Helper 0 is only a weight owner/optional combine worker (`r3_moe_worker.cpp::main`; `worker_combine.h::combine_step`): making it coordinator would require a new, second engine-admission authority, and a helper-0 failure would strand a commit. Implement a background coordinator thread under `HookState`; serialize one transaction, without holding `g.mu` across waits, HIP uploads or patch callbacks.

**ACK participants:** all configured helpers, even one with zero owned experts; all *actual* engine TP ranks (1 or 2, as established by `r3_moe_mailbox_device(rank,world_size)`). No independent resident engine weight manager participates in Stage 1 because the local/remote split is frozen; each helper's slot manager is represented by its helper ACK. Engine rank ACK means both its host scheduler and its GPU stream dependencies satisfied the phase.

**Publication and ordering:** one coordinator CPU writer updates control snapshot by `seq` odd (release), writes state/epoch/slot/digests/masks, then `seq` even (release), then `__atomic_fetch_add(&control.notify,1,__ATOMIC_SEQ_CST)` + shared `FUTEX_WAKE`. Readers acquire-load even `seq`, read fields, re-acquire `seq`; retry if changed/odd. Each participant writes only its `PlacementAck[index]`: data fields first, `__atomic_store_n(&ack.phase,phase,__ATOMIC_RELEASE)` last, then notifies. Coordinator acquire-loads phase and accepts only **exact** `{transaction_id, pending_epoch, pending_digest, pending_slot, expected_phase, fault_code=0}` before setting its own control masks. An old delayed `READY` bit alone is never proof. CPU single-writer ACK records may be updated only between phase publications, with coordination through `control.seq` / phase change; reject publication with an outstanding previous transaction. For GPU mailbox data use the **existing** system-scope `__hip_atomic_store/load(...,__HIP_MEMORY_SCOPE_SYSTEM)`, `__threadfence_system()`, and host `__atomic acquire/release` patterns; do not substitute device/agent scope for shared host flags (`moe_remote_shm.h::load, store, notify, wait_for`; `moe_mailbox_ops.hip::store_sys_rel, load_sys_acq, mailbox_publish_kernel`; `moe_lane_reduce.h::lane_chunk_publish`).

The coordinator uses a monotonic 64-bit epoch: initialize `ACTIVE(1,slot0)`, never reuse an epoch or decrement it. `pending_epoch=active_epoch+1`; reject `UINT64_MAX` and sequence wrap. Rollback after a successful commit is a **new forward epoch** carrying the saved previous map. A failed *precommit* proposal discards inactive staging and returns to `ACTIVE(N)` without modifying `active_epoch`. Keep a two-entry `PlacementJournal`: current candidate and last completed/committed record. Journal writes are coordinator-only, with the same control publication discipline; records are recovery/diagnostic evidence, not a second active-map pointer.

| State and driver | Shared-memory writes / acceptance criteria | Failure outcome |
| --- | --- | --- |
| **ACTIVE(N)** — engine coordinator | `active_epoch=N`, `active_slot=g`, `active_digest=hash(map[g])`, `pending_epoch=0`, no pending slot; only immutable generation g may route. Controller consumes coherent heat snapshots in a background policy thread, not a dispatch callback (`worker_heat.h::Heat::observe, propose`; `moe_remote_hook.cpp::snapshot_loop`). | Invalid active digest, missing helper or changed immutable runtime slot map: `FAULT_PROTOCOL`, fail closed. |
| **PROPOSED** — engine | Copy active map into inactive `1-g`; apply whole-map candidate; validate one owner per regular expert, **no local/helper transitions**, protected/all-local layers fixed, helper eligibility, source availability, hard VRAM budget, per-layer cap and move count/bytes. Assign new txn ID, epoch N+1, candidate digest and journal record; publish `state=PROPOSED`. No GPU changes yet. | Reject/no-change returns ACTIVE(N); invalid proposal `FAULT_INVALID_MAP`. |
| **RESERVING** — engine requests, **each helper slot manager** acts | Publish RESERVING and reset phase masks; helpers allocate destination slots + scratch, charge active + shadow + retiring + staging against hard budget, and publish `ACK(RESERVED,N+1,digest,slot)`; coordinator checks every exact ACK and records masks. No in-place replacement of N. | `FAULT_CAPACITY` (including full two-way swap without reserve), `FAULT_SOURCE`, or timeout: cancel reservations, stay ACTIVE(N), cooldown. |
| **COPYING** — helpers and engine-rank upload workers | Destination helpers read immutable host container planes, relayout as needed and launch asynchronous H2D uploads to **separate shadow slots**; create per-generation helper GPU pointer tables and ready events. Each engine rank builds/uploads its **derived** per-helper bitmap for generation 1-g into separate device allocation, on a staging stream; record an event. `state=COPYING`. | `FAULT_UPLOAD`, `FAULT_CHECKSUM`, `FAULT_DEVICE`, deadline: sync/cancel staged buffers, stay ACTIVE(N), no leaked charge. |
| **PREPARED(N+1)** — engine after everyone READY | Check helper GPU copy-complete events, table/digest readbacks, and engine rank upload-ready events. Each participant then `ACK(READY,N+1,digest,slot)`; require all expected helper/rank masks. Set `GenerationHeader.state=READY` last and publish PREPARED. Do not expose new bitmap to a serving step yet. | Missing/mismatched ACK or any died participant: abort safely before commit. |
| **QUIESCING** — engine scheduler and all lanes | Set a host admission gate **before** snapshotting in-flight step set; allow every admitted forward (all its MoE layers and MTP draft/verify passes) to complete under N. Require per-request exact completion **and engine result-consumed ACK**, combined-plane ACK, final per-rank GPU stream events, and helper lane drain. Each actor publishes `ACK(QUIESCED,N+1,digest,slot,drained_step)`. | `FAULT_QUIESCE_TIMEOUT`, `FAULT_RESULT_NOT_CONSUMED`, or helper death: fail serving; do not commit or reclaim. |
| **COMMIT(N+1)** — engine only | With all quiescence/READY ACKs still valid, install inactive device generation on **every rank**, switch each helper's launch-generation/table selector, then **release-publish** `active_slot=1-g, active_epoch=N+1, active_digest=pending_digest, state=COMMIT` as one seqlock snapshot. Participants verify and `ACK(COMMITTED,N+1,digest,slot)`. Open admission **only after all** committed ACKs and generation agreement. | Before publication, remain at N; after publication, do **not** overwrite N+1 or revert pointer in place. On missing COMMITTED ACK, keep gate closed, fault service, or recover with new epoch if all old residency still exists. |
| **RETIRING(N)** — helper managers and engine rank managers | No new N step can enter. After every N `done/consumed` stamp, combined-plane ack, helper stream-synchronization and engine GPU event prove no N references, discard N generation pointers/device owner arrays, reclaim only old slots not referenced by N+1, and publish exact `ACK(RETIRED,N+1,digest,slot)`; journal records last commit and retire masks. | `FAULT_RETIRE_TIMEOUT`: retain memory; do not reuse inactive generation. Fault budget/service rather than unsafe free. |
| **ACTIVE(N+1)** — engine | Require all retirement ACKs; clear pending fields and expose g as active. Restore policy evaluation after cooldown. | Stale or duplicate proposals rejected by txn/epoch equality. |

**Timeouts:** `CLOCK_MONOTONIC` deadline per phase, checked through bounded futex waits and GPU-event polling **off the serving stream**. Start with configurable defaults: reserve 60 s, copying 180 s (extend only within an explicit configured byte/rate ceiling), READY 30 s, quiesce 30 s, commit ACK 10 s, retirement 120 s; never unbounded `wait_for(...,nullptr)` on a placement phase. Record phase, participant and elapsed time; a helper death immediately dominates the timeout (`moe_remote_shm.h::wait_for, raise_fault`; `moe_helper_watch.h::HelperWatch`). Proposed `FaultCode : uint32_t`: `NONE=0, INVALID_MAP=1, CAPACITY=2, SOURCE=3, UPLOAD=4, CHECKSUM=5, DEVICE=6, HELPER_DEAD=7, READY_TIMEOUT=8, QUIESCE_TIMEOUT=9, EPOCH_MISMATCH=10, COMMIT_TIMEOUT=11, RETIRE_TIMEOUT=12, RESULT_NOT_CONSUMED=13, SEQUENCE_EXHAUSTED=14, INCOMPATIBLE_LAYOUT=15, RANK_DEAD=16`. Store first fault + actor in control and journal. Recoverable precommit errors report failure/cooldown without setting global `helper_fault`; uncertainty after commit or any possible partial output raises the existing sticky `helper_fault` and stops unsafe serving (`moe_remote_shm.h::raise_fault`).

## 4. Concrete quiesce and commit barrier

1. **Add real model-step admission:** extend the engine patch's `BcMoeRemotePlugin` contract with `begin_step(step_id, phase, rank_world)` and `end_step(step_id, rank, success)`, including the MTP draft/verify forwards under the same parent `step_id`. The first callback must occur **before** graph/stream work for any layer is enqueued; coordinator blocks new admissions using a condition variable and increments a bounded in-flight step ledger. Return an immutable `StepTicket {uint64_t step_id, epoch; uint32_t slot, reserved;}` to **every rank**, with the same epoch and generation. `end_step` fires only after all GPU effects and dependent MTP passes finish, not merely after CPU submission. `burst_transition` remains only for power management; its odd/even burst word cannot substitute for this barrier (`moe_remote_hook.cpp::EnginePlugin, burst_transition`; `engines/radiance/patches/rad0002_moe_remote_handoff/patch.py::_BC_MOE_REMOTE_H, _BC_MOE_REMOTE_CPP::bc_moe_remote_engine_step/idle`). All 48 model MoE layers in the target model, plus optional MTP work, are one admission unit; derive the layer count from `Header::layer_count`, not from literal 48 in code.
2. **Bind that ticket to actual GPU work:** extend `R3MoeMailboxDevice` with `owner_device[2]` and a per-forward immutable ticket pointer/handle; each `publish_launch`/kernel receives that forward's ticket explicitly, not a mutable global `active_slot` sampled separately per layer. The engine patch must give graph replay a distinct ticket lifetime until that replay's last consumer event; an async copy into **one shared ticket buffer** is not safe when another step can be submitted before the first GPU read. The publish kernel checks `ticket.{epoch,slot}` against the prepared generation, routes using `owner_device[ticket.slot]`, writes the ticket into every `Request`, then system-fences and releases `request_sequence`. A rank's complete forward, including MTP, never sees a different bitmap. Reject a request whose slot is not READY/ACTIVE or whose epoch is absent/mismatched (`moe_remote_hook.h::R3MoeMailboxDevice`; `moe_mailbox_ops.hip::publish_launch, mailbox_publish_kernel, owner_has`).
3. **Exact helper completion:** `lane_prepare_activation*`, `lane_front_small`, `lane_route_count` and `lane_select_tables` validate `Request::sequence, placement_epoch, generation_slot` before touching tables or weights. `Runner::enqueue_lane` passes the generation-specific `DeviceLayerTables` and not the old single `d_layer_tables_`. The final `lane_publish_done` writes `q->completion_epoch/slot/sequence`, `LaneSignal::done_epoch/slot`, fences system-wide, and finally system-release-stores `done_sequence`. Both the normal wait and combined tail-wait require `done_sequence==seq`, `done_epoch==step.epoch`, `done_slot==step.slot`, matching request fields, plus `status==0`. Any mismatch faults without treating a stale done word as completion (`moe_remote_gpu.h::Runner::enqueue_lane`; `moe_lane_front.h::lane_front_small, lane_route_count`; `moe_lane_reduce.h::lane_publish_done`; `moe_mailbox_ops.hip::mailbox_wait_helpers`).
4. **Engine result-consumed ACK:** change `mailbox_add_finish_kernel` to take `base, sequence, published, step_ticket` and, **after** `mailbox_add_rows_kernel` and combined tail-wait on the same ordered engine stream, system-fence and release-publish each actually used lane's `consumed_epoch/slot` and `consumed_sequence` (sequence last). The chunked path **also** release-publishes `CombineSignal::result_consumed_epoch/slot` and `result_consumed_generation` (generation last), after all last reads of `result_index(0,1)`, including fallback rows. `mailbox_shadow`/diagnostic consumption gets an equivalent terminal ACK after its own last plane read. The host quiesce requires every issued `(lane,seq,epoch,slot)` has its consumed stamp; the coordinator must not infer that from `combine.done`, `chunks.done`, or `published==0` (`moe_mailbox_ops.hip::add_launch, mailbox_add_rows_kernel, mailbox_add_tail_wait_kernel, mailbox_add_finish_kernel, shadow_launch`; `worker_combine.h::combine_step`).
5. **Small-lane drain even when power-save is off:** introduce an explicit `epoch_quiesce` control notification, separate from `burst`. Once new admission is shut and every old step has the above consumed ACK, each `Lanes::run_small` stops refilling `kQueueDepth`, executes the existing `park_lane` release sequence with the last real published sequence and `Runner::set_publish_limit`, then `Runner::sync_lane`; it reports QUIESCED **only after** no old wait packet or old-table GPU kernel remains. For the power-save-off path, force the same drain rather than taking the existing `if (!power_save) continue` branch. Large-lane threads `sync_lane` after their current request; helper 0 waits for any CPU `CombinePool` job and its chunk publication to finish. During QUIESCING no lane re-arms, even if a burst opens; re-arm on COMMITTED with the new table slot. Preserve the v22 park handshake's SEQ_CST park/burst recheck and `notify` wakeups; add a transaction ID to reject a late park ACK from an older drain (`worker_lanes.h::Lanes::run_small, park_lane, run_large, wait_burst`; `moe_remote_gpu.h::Runner::set_publish_limit, sync_lane`; `worker_combine.h::CombinePool::combine`).
6. **Switch only at the global step boundary:** coordinator checks its rank-ledger empty, the last GPU terminal events and every helper's QUIESCED ACK, then commits one active generation. A participant's COMMITTED ACK is released only after the selector and its device-side table/bitmap are usable **on the stream/queue that will serve new work**. Do not overwrite generation-N pointers/tables, or the inactive owner buffer, until all old GPU readers retired. The first N+1 request records the new epoch; the last N request has already been consumed. A partial commit does not permit N+1 admission (`moe_remote_hook.cpp::r3_moe_mailbox_device, burst_transition`; `moe_remote_gpu.h::Runner::build_layer_table, enqueue_lane`; `moe_mailbox_ops.hip::mailbox_publish_kernel`).
7. **Retirement includes output planes:** hold old expert VRAM, per-generation GPU pointer tables, engine bitmap allocations, **and** mapped output/combine plane reuse rights until completed+consumed ACKs and GPU/CPU stream drains. Mapped route/result planes are shared by slot, so copy completion and `combine.done` alone are insufficient. Free old-only shadow/retiring slots on the owning helper after its RETIRED ACK can be proven, clear their pointers, and poison reclaimed slots in stress mode. Do not free a slot still referenced by the new generation (`moe_remote_shm.h::layout_for, result_index`; `moe_remote_gpu.h::Runner::allocate, build_layer_table`; `worker_combine.h::combine_step`).

**Migration slot manager:** new `worker_residency.h` owns slots with `FREE -> RESERVED -> COPYING -> READY -> ACTIVE -> RETIRING -> FREE`, exact bytes and source for all four executable planes (gate/up code+scale, down code+scale) and stored TP slices. Separate active and shadow generations: `Runner::upload_weight` must be split into immutable source/relayout, bounded pinned bounce buffering and async H2D into a **provided** destination; attach a HIP ready event to each staged expert and do not build READY tables until the event completes. Start with one concurrent upload per destination helper, byte-capped globally, no GPU P2P. Charge active + shadow + old retiring + lane scratch + transfer staging to the helper's hard budget; a full two-way exchange with no spare slot is **DEFERRED_CAPACITY**, not unsafe in-place overwrite. A helper whose initial share was zero must be able to reserve/load into an initially empty arena. These are missing capabilities in `moe_remote_gpu.h::Runner::allocate, upload_weight, load_layer, build_layer_table`; helpers already open the immutable source via `worker_container.h::Container::open, pread_full` and `r3_moe_worker.cpp::main`.

## 5. Ordered implementation slices (each builds/tests before the next)

| Slice | Exact files/functions to change | Card-free / local exit criterion |
| --- | --- | --- |
| **A — ABI/data** | `moe_remote_shm.h::kVersion, Request, LaneSignal, CombineSignal, Header, Layout, layout_for, init, valid, valid_self, lane_request, owner_bits/owns/set_owned/clear_owned`: v23-only layout, accessors, atomic snapshots, digest, offset checks. `worker_shape.h::segment_shape`: pass actual layers. `moe_remote_shm_test.cpp`: new fixtures, offsets, invalid descriptors. | C++ and HIP agree on every offset/stride; corrupt map length/version/epoch fails closed; no unchanged v22 caller compiles. |
| **B — controller no GPU** | New `worker_epoch.h`: pure state machine, ACK validation/journal, deadline/fault table; new `worker_residency.h`: abstract slot ledger/source eligibility; `worker_heat.h::validate, propose`: complete-map max-per-layer weighted-heat metric, bounded equal-cost rebalance and two-way swap; use coherent RHP02 interval deltas and cooldown, limit moves. | Deterministic replay, 0/1/4 helper permutations, stale ACKs, budgets, protected layers, zero-owner helper, hard cost/no-change tests. |
| **C — engine patch/admission** | `engines/radiance/patches/rad0002_moe_remote_handoff/patch.py::_BC_MOE_REMOTE_H/_BC_MOE_REMOTE_CPP` plus its narrowly anchored scheduler/model-step call sites: extend `BcMoeRemotePlugin`, step tickets/admission, forward/MTP completion, all-rank barrier. `moe_remote_hook.cpp::EnginePlugin, HookState, register_mailbox, burst_transition, close_mailbox`: coordinator thread + step ledger + shutdown/fault behavior. | Fake rank/scheduler tests prove blocked admission, N across every layer and MTP, safe shutdown and no old tickets. |
| **D — engine GPU** | `moe_remote_hook.h::R3MoeMailboxDevice`; `moe_remote_hook.cpp::RankState, r3_moe_mailbox_device, close_mailbox, owner_query`: two owner bitmap allocations derived from generation maps, async copies/events, release/retire. `moe_mailbox_ops.hip::publish_launch, mailbox_publish_kernel, mailbox_wait_helpers, mailbox_route_mark_kernel, mailbox_add_rows_kernel, mailbox_add_finish_kernel, add_launch, shadow_launch`: per-step ticket/epoch validation, exact done and consumed ACK, chunked terminal ACK. | Compile + fake-device boundary tests for ticket lifetime, no mixed owner bits, stale completion, and combined-plane consumption. |
| **E — helper GPU/slot migration** | `moe_remote_gpu.h::Runner::allocate, load_layer, upload_weight, build_one_table, build_layer_table, enqueue_lane, sync_lane, set_publish_limit`: explicit VRAM slots, async upload, two immutable generations of runtime pointers, generation-bound enqueue and events. `moe_lane_front.h::lane_front_small, lane_route_count, lane_select_tables`; `moe_lane_reduce.h::lane_publish_done`: checked epoch and completion stamps. `worker_owners.h::select_owned, publish_owned`: startup map/bootstrap only, no independent live planner. | Fake allocator and HIP-independent table model: old slot not freed while referenced; no planes published half-loaded. |
| **F — helper control/drain/combine** | `r3_moe_worker.cpp::main`: construct per-helper epoch actor/slot manager; replace startup bitmap publication with canonical bootstrap agreement; service ACK loop and faults. `worker_lanes.h::Lanes::run_small, run_large, park_lane, wait_burst, stop_and_join`: explicit epoch drain/rearm. `worker_combine.h::combine_step, CombinePool::combine`: request epoch/slot and participant-set verification; no combine span across flip. `worker_segment.h::Segment::open`: require v23. | Fake GPU queue with 4 prequeued waits, disabled power save, stalled combine, late/faulted ACK: zero live old references at RETIRED. |
| **G — integrated gate** | `r3_moe_worker.cpp::main` forced-move mode; `tools/lab/radiance/` existing replay/ABBA harness; RHP12 baseline records and evidence. | >=1,000 transitions; replay and full RHP12 gates pass before any automatic activation. |

Any engine-side edit must be packaged in the existing managed patch (or a new narrowly scoped managed patch) with reproducible pinned-source patch tests; do not modify an installed Radiance source tree. References: `engines/radiance/patches/rad0002_moe_remote_handoff/patch.py`; `moe_remote_hook.cpp::register_mailbox`; `moe_mailbox_ops.hip::publish_launch`.

## 6. Card-free verification and forced-move hook

Add `moe_remote_epoch_test.cpp` beside `moe_remote_shm_test.cpp`, with a fake clock, fake actor/ACK transport, deterministic event/stream simulator, mock `SlotManager` and fake engine step ledger; no HIP context or model weights required. Use named cases:

- `V23Layout_ExactOffsetsAndBounds`: `Request/Header/Lane/Combine/Control/ACK/Journal` static/runtime sizes, alignment, dynamic `layers*experts`, generation regions and page bounds; v22 version rejected; overflow and malformed offsets rejected.
- `V23Map_LocalProtectedAndDisjoint`: every `(layer,expert)` has exactly one canonical byte; negatives from frozen runtime slots stay local; helper index never exceeds `helper_count`; no engine-local↔helper migration; zero-active-expert helper may receive an eligible expert.
- `V23Digest_ExactAndImmutable`: byte-order-stable hash; both generation map digests agree across actors, wrong model/shape/reordered byte rejected; ACTIVE/RETIRING maps and tables never overwritten.
- `V23Ack_OldTxnDelayedDuplicateLost`: reject duplicate/wrong phase/old digest/old epoch/wrong slot/stale txn/late ACK; lost ACK times out; waiter before and after notification has no lost wakeup.
- `V23Epoch_ForwardOnlyAbortRetry`: all eight specified state transitions, precommit abort leaves epoch N, committed rollback produces N+2, simultaneous proposals serialize, 32-bit sequence / 64-bit epoch wrap refused.
- `V23Admission_AllLayersMtpRanks`: 48 MoE layers plus MTP verify/draft are one step ticket, actual rank count not header bound, new-step gate closes before submissions, a queued/captured replay cannot read a recycled ticket, no split epochs.
- `V23Request_SequenceAndEpochMatch`: identical sequence/wrong epoch or slot cannot complete; no stale/missing/duplicate route; helper completion release precedes done; consumed follows last engine read.
- `V23Lane_PrequedDepth4Drain`: 4 pre-enqueued small waits, large-lane in flight, burst reopen, power-save off, stop during park; COMMITTED waits for `sync_lane`, stale old kernel never dereferences new tables.
- `V23Combine_ChunkNotLifetimeFence`: chunk 0 combined/read before last chunk, helper 0 CPU pool still active, fallback row, all helper sequences done but engine consumed ACK absent; retirement blocked until combined result-consumed generation/epoch and lane ACKs.
- `V23Slots_ReserveCopyRetireFaults`: four weight planes, asynchronous last-plane completion, partial write/poison, one-inflight-upload limit, 2-way full exchange, deferred capacity, old-live slot retention, no leaks or unbounded budget growth.
- `V23Fault_BeforeAndAfterCommit`: helper death before prepare, mid-copy, during quiesce, just after commit; mismatched digest, stalled mapped-host update, engine rank death; precommit abort versus postcommit fail-closed/forward recovery.
- `V23Policy_HeatEqualCostAndSwap`: equal-cost layer peak rebalance, zero/stale coverage, repeatable tie-break, protected experts, per-layer caps and budgets, <=8 moves and <=256 MiB/transaction, >=10% estimated benefit, 30-minute residence and failure cooldown; exhaustive small-instance optimum oracle.

**Forced-move mode:** add `R3_MOE_PLACEMENT_FORCE_MOVE=layer:expert:destination_helper` as an explicit **test-only**, default-off control accepted by the coordinator. It requests one legal map delta using the **production** RESERVING→COPYING→PREPARED→QUIESCING→COMMIT→RETIRING codepath; it bypasses only the policy gain and dwell-time thresholds, never eligibility, budget, source, digest, ACK, or step barriers. Reject an unsupported/unavailable target with a fault and no ownership change. Also accept a harness schedule (ordered validated triples + optional inject-at-phase/delay) for >=1,000 transitions; expose active/pending epoch, digest, each participant ACK, slot charges, bytes staged, last-consumed step, pauses and faults as bounded aggregate evidence. `worker_heat.h::propose` never directly mutates a live map; `r3_moe_worker.cpp::main` owns no independent second live policy.

**End-to-end stop/go (RHP12):** first run short mailbox/epoch/replay, forced prequeued-wait and combine tests; then >=1,000 forced-churn transitions including cross-helper swaps, zero-owner destination, MTP and 42K prefill, partial-copy/crash/timeout injection. Compare static vs live in >=4 clean paired ABBA cohorts for **8-session prefill**, **6-stream decode** (model step time from metrics), and **42K probe**. Gate idle-path median step time <=1% regression and p95 <=2%; during migration median <=2% and p95 <=5%, report individual pause spikes; no VRAM budget violation or stale/missing/duplicate contribution. Compare **same-expert/same-kernel** input and contribution bytes for exact equality across equivalent helper architectures; for heterogeneous helpers or changed helper reduction/grouping or CPU combined-fp16 execution, compare reference/KLD/top-1 instead of asserting whole-model bit identity (`moe_remote_gpu.h::Runner::enqueue_lane`; `worker_combine.h::combine_step`; `moe_mailbox_ops.hip::mailbox_add_rows_kernel`). A returned known layout must re-pass its immutable baseline; on regression, the old validated owner map is applied by a **new** forward epoch, never a restart.

## 7. Open questions (must be resolved against the actual engine integration, not guessed)

1. **Precise host scheduler and MTP lifetime:** `EnginePlugin` currently has only `burst`; identify every scheduler / draft / verify enqueue and final-consumer call site in the pinned Radiance engine before anchoring `begin_step/end_step`. Whether multiple forwards and multiple rank streams overlap is not demonstrated by `moe_remote_hook.cpp::EnginePlugin` or `patch.py::_BC_MOE_REMOTE_CPP`. The implementation must test and explicitly bind the ticket per replay/forward, not infer one request equals one full step.
2. **Graph replay and ticket ABI:** how an immutable per-forward `StepTicket` is threaded through the engine's recorded `RadArgs`/graph instances is not established by `moe_mailbox_ops.hip::publish_launch` and `moe_remote_hook.h::R3MoeMailboxDevice`. Choose a caller-owned, event-retired per-forward ticket buffer or safely patched graph node parameters **after** locating that capture/replay owner. Do not introduce a single mutable rank-global ticket.
3. **Engine step-consumed boundary:** confirm the last device/cross-stream consumer of each `moe_mailbox_add` output and of shadow diagnostics in pinned engine scheduling. If `add_launch`'s ordered stream is not the last consumer of the **shared output plane**, move the terminal ACK to the true last stream event; `moe_mailbox_ops.hip::add_launch, shadow_launch` alone cannot certify unrelated streams.
4. **Runtime source and budgets:** validate which immutable model file and mapped segments persist when `R3_MOE_WORKER_SKIP_OWNED` removes engine-local weights, plus actual destination format parity and hard helper budget source. `moe_remote_gpu.h::Runner::upload_weight` currently loads through `Container` and has no async slot manager; do not assume the engine's removed placeholders are a fallback source.
5. **Independent residency actors:** if a later engine patch introduces an independent residency manager that can evict helper weight sources or move the engine/local boundary, it must become an explicit ACK participant. Stage 1 `r3_moe_worker.cpp::main` and `moe_remote_hook.cpp::owner_query` expose no such independent live actor; do not silently add one or expand this milestone into Stage 4.

Acceptance: a single v23 owner map per generation, one engine-controlled transaction path, no helper restarts, exact model-step and result-plane lifetime proofs, strict fault containment, and the RHP12 no-regression evidence.
