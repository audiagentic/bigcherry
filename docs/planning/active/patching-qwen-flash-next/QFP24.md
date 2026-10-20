---
id: QFP24
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-05T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Safe pinned scheduler input ring for asynchronous large prefill host->GPU copies

## 2026-10-13 binding correction (see QFP27)

**Do not implement the private `bc_async_input_lane` / `bc_async_input_slot::done[]` pseudocode below as an approved parallel scheduler.** Upstream #29963's input-event generations are enabled only for layer-pipeline `n_copies>1`, while production Meta tensor split uses `n_copies=1` and its device has `events=false` / `event_new=nullptr`. Upstream therefore does **not** currently solve QFP24's tensor-split source lifetime. The detailed code below is historical design, not a ready patch.

QFP16/1326 remains implementation owner; QFP27 owns the admission/compatibility and all-child fanout proof. QFP24 remains **conditional pinned-backing qualification only**. First measure >=1 ms AND >=3% prefill cost; if qualified, add <=64 MiB pinned backing inside 1326 with every child completed before reuse or retain pageable/synchronous fallback. No independent event-query API, scheduler registry or second ring implementation.

## Description

Patch 1326/QFP16 proved that synchronous split-input handling is expensive and async host->device copies can improve serving. The original pinned-source shortcut improved prefill but was unsafe because graph-input storage may be rewritten before true DMA consumes it. Current 1326 therefore copies into scheduler-owned pageable `std::vector<uint8_t>` staging and caps the path at 4 MiB; safe for decode, but large prefill inputs lose the overlap.

QFP24 extends **the existing 1326 code**, not a second scheduler copy path: large eligible inputs stage into a persistent pinned ring whose slots are reused only after every destination stream has recorded completion.

Pinned llama.cpp `0504396` APIs available to use:

```cpp
ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device);
ggml_backend_buffer_t      ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
void *                     ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer);

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device);
void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend);
void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event);
void ggml_backend_event_synchronize(ggml_backend_event_t event);
```

There is no generic nonblocking `event_query` at this pin. The plan therefore specifies both a minimal safe v1 and the preferred v2 query hook instead of pretending an API already exists.

## Steps

1. QFP27 must first establish that large prefill input handling on the actual tensor-split production pin costs >=1 ms AND >=3% of prefill wall.
2. Retain QFP16/1326's <=4 MiB pageable path as the control; the proposed private ring below is historical, not an approved second scheduler.
3. If Gate 1 passes, extend 1326 in-place with <=64 MiB scheduler-owned pinned backing, not another registry or independent generation system.
4. For Meta mirrored fanout, obtain and join completion from **every** consuming simple backend; Meta has no device event at b11552.
5. Fall back before partial submission when any event/shape/child is unsupported; never reuse pinned bytes until all children finish.
6. Test immediate overwrite, 10,000 generations, 2/3/4 slots, multi-request, graph rebind, cancel, delayed child, MTP/full logits/KLD and peak pinned bytes.
7. Qualify paired control/subject on idle hardware: >=5% median prefill/TTFT gain, positive CI95-low, <=2% decode regression; otherwise close without implementation.

## Detailed Solution & Technical Design

### 1. Replace the current 1326 staging object with owned pinned slots

Current 1326 code is effectively:

```cpp
static thread_local std::unordered_map<const ggml_tensor *, std::vector<uint8_t>> bc_staging;
std::vector<uint8_t> & bc_stage = bc_staging[input_cpy];
bc_stage.resize(ggml_nbytes(input));
memcpy(bc_stage.data(), input->data, ggml_nbytes(input));
ggml_backend_tensor_set_async(split_backend, input_cpy, bc_stage.data(), 0, ggml_nbytes(input));
```

Replace only the large-input branch with a scheduler-owned pool:

```cpp
struct bc_async_input_slot {
    ggml_backend_buffer_t host_buf = nullptr;
    void * ptr = nullptr;
    size_t capacity = 0;
    uint64_t generation = 0;
    bool live = false;

    ggml_backend_event_t done[GGML_BACKEND_META_MAX_DEVICES] = {};
    uint32_t n_done = 0;
};

struct bc_async_input_lane {
    bc_async_input_slot slots[4];
    uint32_t depth = 3;
    uint32_t next = 0;
};

struct bc_async_input_pool {
    std::unordered_map<uint64_t, bc_async_input_lane> lanes;
    size_t reserved = 0;
    size_t cap_bytes = 256ull << 20;
};
```

Put the pool in scheduler lifetime (`ggml_backend_sched` context), not `thread_local`; the scheduler owns tensor-copy buffers and can free all pinned buffers/events in its destructor/reset path.

### 2. Pinned allocation through the GGML backend abstraction

For a non-Meta GPU destination:

```cpp
static bool bc_alloc_pinned_slot(
        ggml_backend_t dst_backend,
        bc_async_input_slot & slot,
        size_t bytes) {
    ggml_backend_dev_t dev = ggml_backend_get_device(dst_backend);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
    if (buft == nullptr) {
        return false;
    }

    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, bytes);
    if (buf == nullptr || !ggml_backend_buffer_is_host(buf)) {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
        return false;
    }

    slot.host_buf = buf;
    slot.ptr = ggml_backend_buffer_get_base(buf);
    slot.capacity = ggml_backend_buffer_get_size(buf);
    return slot.ptr != nullptr && slot.capacity >= bytes;
}
```

For Meta, do not assume its `caps.host_buffer` bit is usable at this pin. Prefer the first simple child's host-buffer type only after verifying all child CUDA/HIP devices accept the same pinned host allocation; otherwise allocate via the CUDA/HIP registry helper used by the child backends. Keep that logic inside a Meta helper, not in generic scheduler code.

### 3. Internal tracked Meta fanout

Current Meta `set_tensor_async` fans a host pointer into each simple destination. Add a private core helper in `ggml-backend-impl.h` / `ggml-backend-meta.cpp`, not public API:

```cpp
struct ggml_backend_meta_async_ticket {
    ggml_backend_event_t events[GGML_BACKEND_META_MAX_DEVICES];
    uint32_t n_events;
};

bool ggml_backend_meta_set_tensor_async_tracked(
        ggml_backend_t backend,
        ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size,
        ggml_backend_meta_async_ticket * ticket);
```

Implementation shape:

```cpp
bool ggml_backend_meta_set_tensor_async_tracked(
        ggml_backend_t backend,
        ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size,
        ggml_backend_meta_async_ticket * ticket) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ticket->n_events = 0;

    const size_t n = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_t child = ggml_backend_meta_simple_backend(backend, i);
        ggml_tensor * child_tensor = bc_meta_simple_tensor_for_set(tensor, i); // factor from current set_tensor_async
        if (child_tensor == nullptr) {
            continue;
        }

        ggml_backend_tensor_set_async(child, child_tensor, data, offset, size);

        ggml_backend_dev_t dev = ggml_backend_get_device(child);
        ggml_backend_event_t ev = ggml_backend_event_new(dev);
        if (ev == nullptr) {
            return false;
        }
        ggml_backend_event_record(ev, child); // same child stream, immediately after H2D copy
        ticket->events[ticket->n_events++] = ev;
    }
    return ticket->n_events > 0;
}
```

`bc_meta_simple_tensor_for_set()` is not an upstream symbol: extract/refactor the exact child-tensor mapping already present in current `ggml_backend_meta_set_tensor_async()` so the tracked and untracked paths cannot drift.

Do not allocate/free events per ubatch in production; the code above shows ordering only. Real implementation pre-creates one event per `(slot, child)` when the slot/lane is initialized and re-records it each generation.

### 4. Slot acquisition v1: safe and simple

Because generic GGML lacks event query at this pin, first prove the design with slot-local synchronization only on reuse:

```cpp
static void bc_slot_wait_and_reuse(bc_async_input_slot & slot) {
    if (!slot.live) {
        return;
    }
    for (uint32_t i = 0; i < slot.n_done; ++i) {
        ggml_backend_event_synchronize(slot.done[i]);
    }
    slot.live = false;
    slot.n_done = 0;
}
```

With depth 3/4 the copy may already be complete when a slot wraps, so this can be nearly free while being unquestionably correct. Measure before widening API surface.

### 5. Preferred v2: nonblocking completion query

If v1 shows useful overlap but slot-wrap synchronization is material, add an optional backend proc-address rather than changing the public device iface/API version:

```cpp
typedef bool (*ggml_backend_event_query_t)(ggml_backend_event_t event);
```

CUDA/HIP registry exports:

```cpp
static bool ggml_backend_cuda_event_query(ggml_backend_event_t event) {
    auto * cuda_event = (ggml_backend_cuda_event *) event->context;
    cudaError_t err = cudaEventQuery(cuda_event->event);
    if (err == cudaSuccess) {
        return true;
    }
    if (err == cudaErrorNotReady) {
        return false;
    }
    GGML_CUDA_CHECK(err);
    return false;
}
```

and returns it from `get_proc_address("ggml_backend_event_query")`. The pool resolves one query function per child device at initialization. Unknown backend -> v1 sync/fallback; never cast a CUDA event from generic code.

Then:

```cpp
static bool bc_slot_ready_nonblocking(const bc_async_input_slot & slot) {
    for (uint32_t i = 0; i < slot.n_done; ++i) {
        if (!slot.query[i](slot.done[i])) {
            return false;
        }
    }
    return true;
}
```

### 6. Scheduler fast path anchored to current 1326 code

Inside `ggml_backend_sched_compute_splits`, retain all existing eligibility checks and current safe fallback. Add the pinned branch only above the 4 MiB cap/fallback:

```cpp
const size_t nbytes = ggml_nbytes(input);
const bool bc_large = nbytes > ((size_t) 4 << 20);

if (bc_async_inputs && bc_large &&
        split_backend->iface.set_tensor_async != nullptr &&
        input->buffer != nullptr &&
        ggml_backend_buffer_is_host(input->buffer) &&
        ggml_backend_buffer_get_usage(input->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_is_contiguous(input) && ggml_is_contiguous(input_cpy) &&
        nbytes == ggml_nbytes(input_cpy)) {

    ggml_backend_synchronize(input_backend); // preserve 1326 producer ordering first

    bc_async_input_slot * slot = bc_pool_try_acquire(sched->bc_async_pool, input_cpy, nbytes);
    if (slot != nullptr) {
        memcpy(slot->ptr, input->data, nbytes);

        ggml_backend_meta_async_ticket ticket = {};
        bool ok = ggml_backend_is_meta(split_backend)
            ? ggml_backend_meta_set_tensor_async_tracked(
                  split_backend, input_cpy, slot->ptr, 0, nbytes, &ticket)
            : bc_simple_set_tensor_async_tracked(
                  split_backend, input_cpy, slot->ptr, nbytes, &ticket);

        if (ok) {
            bc_slot_adopt_ticket(*slot, ticket, ++sched->bc_async_generation);
            continue;
        }
    }
}

// Existing 1326 pageable staging / upstream synchronous fallback follows unchanged.
```

`bc_pool_try_acquire` must try all ring slots and return `nullptr` when none are safely reusable; do not wait indefinitely.

### 7. Lane key and bounded memory

Do not key by source tensor pointer. Key by destination copy buffer identity plus rounded capacity:

```cpp
static uint64_t bc_lane_key(const ggml_tensor * input_cpy, size_t bytes) {
    const uint64_t sz_class = 1ull << (64 - __builtin_clzll(std::max<size_t>(bytes, 1) - 1));
    return bc_hash_combine((uintptr_t) input_cpy->buffer, sz_class);
}
```

Use a portable round-up helper instead of this exact builtin if MSVC support matters. The invariant is stable destination lifetime + bounded size classes.

## Code Samples & Guidance

Implementation sequence:

```text
1. move staging ownership from thread_local into scheduler context
2. add pinned slot allocation/free and v1 wrap synchronization
3. factor current Meta child fanout into reusable helper
4. add per-slot child completion events
5. stress immediate source overwrite for correctness
6. benchmark depth 2/3/4
7. only if wrap sync matters, add optional CUDA/HIP event-query proc
8. fold into 1326; delete any duplicate experimental path
```

Diagnostics:

```text
BIGCHERRY_ASYNC_INPUT name=... bytes=... lane=... slot=2 gen=41 pinned=1 fanout=3 fallback=0 memcpy_us=... enqueue_us=...
BIGCHERRY_ASYNC_INPUT_POOL reserved=... active=... busy_fallbacks=... wrap_wait_us=...
```

## Files

- `patches/1326_sched_async_host_inputs/patch.py` + tests/evidence.
- `ggml/src/ggml-backend.cpp`: scheduler pool + fast path.
- `ggml/src/ggml-backend-meta.cpp`: factor child fanout + tracked events.
- `ggml/src/ggml-backend-impl.h`: private ticket/helper declaration if needed.
- CUDA/HIP registry source only if v2 nonblocking event query is justified.

## Validation

Mechanics:
- overwrite source immediately after scheduler staging;
- depth-2 ring with deliberately delayed child copy;
- 3-rank mirrored fanout where one rank is delayed;
- graph rebuild and repeated requests;
- ring full -> fallback, never stale reuse;
- >=10,000 generations with byte pattern generation tags.

Hardware: 10K/80K/200K fills, ub512/1024 where possible. Compare upstream sync, current 1326 pageable staging, pinned v1, pinned v2 if implemented. Record input-handling wall, CPU memcpy, H2D overlap, prefill t/s, TTFT, copy-engine utilization and pinned high-water.

Decode/MTP <=4 MiB path should remain current 1326 behavior unless the ring is separately proven better.

## Effort & Risk

M. Localized code, high lifetime-corruption risk. Main failure mode is slot reuse before the slowest Meta child copy completes.

## Standards

Scheduler-owned lifetime; bounded pinned memory; stable addresses; events recorded after copy on the same child stream; safe fallback; no assumed pageable-copy semantics; no public API expansion unless required.

## Acceptance Criteria

- Source-overwrite/ring-wrap stress passes >=10,000 generations.
- Large prefill input handling drops >=30% or prefill improves >=3% vs current safe 1326.
- No decode regression >1%.
- Pinned allocation bounded/leak-free.
- Final code consolidated into patch 1326.

## Change Log

- 2026-10-05T00:00:00+00:00 (created-by): Created from QFP16/1326 follow-up.
- 2026-10-05: Added concrete scheduler/Meta/pinned-buffer/event code paths grounded in llama.cpp 0504396 APIs and current 1326 implementation.

## 2026-10-16 BCOP236 clarification

The 1326 `thread_local` pointer-keyed **pageable** stage has a separate potential retained-capacity issue across graph/scheduler lifetimes. QFP16 owns its bounded scheduler-lifetime correction; QFP27 owns the source and Meta lifetime discriminator. Do not treat this as evidence for the **large pinned** ring here or implement the historical independent `bc_async_input_lane`. Pinned work remains blocked until >=1 ms AND >=3% exposed prefill input cost is measured; any subsequent path requires all-child completion and existing hardware gates. Upstream #29963 still does not supply Meta completion for tensor split.
