"""Measured per-candidate workspace, via the pool's own bookkeeping (HI52 part 1).

`measurement.workspace_bytes` today is each candidate's *declared* upper bound,
returned by its own `workspace()` function. It is not a measurement, and
candidates within a family declare the same value, so it has never discriminated
anything: HI45's `low-memory` Pareto profile reports exactly 0.0% workspace
savings on every corpus ever collected, because it is ranking on a constant.

An earlier attempt (2026-08-10, reverted) tried to measure this with a
device-global `hipMemGetInfo` delta around each candidate. All 310 measurements
read exactly 0. The cause is structural, not a bug: every candidate's scratch
comes from `ggml_cuda_pool_alloc(ctx.pool(), ...)`, a caching pool that grows
once to a high-water mark and reuses it, so a device-global snapshot is blind to
per-candidate demand by construction.

The pool knows the answer. This patch has it keep it.

Why the bookkeeping lives in the base struct rather than in each pool
--------------------------------------------------------------------
`ggml_cuda_pool` has two implementations, `ggml_cuda_pool_leg` and
`ggml_cuda_pool_vmm`, and only the VMM one already tracks bytes handed out
(`pool_used`); the legacy one's `pool_size` counts cached-but-free buffers and
never decrements on `free`, so it cannot be reused for this. Instrumenting both
would mean four separate edits inside `ggml-cuda.cu`, including three
independent return paths in `ggml_cuda_pool_leg::alloc`.

None of that is necessary. `pool->alloc` and `pool->free` have **exactly two
call sites in the entire CUDA/HIP backend**, both inside `ggml_cuda_pool_alloc`
(`common.cuh`), and `free` is always passed back the same `actual_size` that
`alloc` returned. Verified by grepping the whole `ggml-cuda/` tree: every other
apparent caller (`mmf.cu` and friends) goes through `ggml_cuda_pool_alloc::alloc`
and therefore through that one line.

So a counter maintained at that choke point cannot be bypassed and cannot drift,
covers both pool implementations identically, needs no virtual dispatch, and
touches one upstream file instead of two.

Which number this is
--------------------
`bc_requested_size`, i.e. what THIS candidate asked for (`size * sizeof(T)` at
its own `alloc()` call), not `actual_size` -- what the pool happened to set
aside to serve that request. The two diverge in exactly the case this metric
exists to catch: on a best-fit reuse hit, `actual_size` can be the leftover
size of a buffer some earlier, unrelated candidate freed back into the pool,
which is the same cross-candidate contamination HI56's cache-clear call works
around from the other direction (evicting the stale buffer instead of
mislabelling its size). Tracking `actual_size` instead of the request would
silently reintroduce that contamination on any reuse hit HI56's clear did not
already prevent.

`note_alloc`/`note_free` must stay paired on the same quantity -- the real
pool operation (`pool->free(ptr, actual_size)`) keeps using `actual_size`
unchanged; only the accounting side-channel reads `bc_requested_size` instead,
set once at `alloc()` time so the destructor's `note_free` sees the exact
value `note_alloc` counted.

This measures the pool high-water mark a candidate reaches, which is its real
scratch demand. It is not the candidate's total VRAM footprint: model weights
and the graph's own tensors are not pool allocations and are not counted.
"""

from bigcherry.patcher import Edit, FilePatch

_LEG_CLEAR_CACHE = """
#ifdef GGML_HIP_WORKSPACE_METRICS
    // bigcherry (HI56): `alloc()`'s best-fit reuse can hand a tiny request the
    // full size of whatever oversized buffer an EARLIER, unrelated candidate
    // freed back into `buffer_pool` -- `note_alloc(*actual_size)` then measures
    // that inherited size as though it were this candidate's own need.
    // Confirmed on real hardware: one dispatch had 9 candidates with 9
    // different declared workspace() values all report the identical
    // pool_peak_bytes, which is only explicable as shared cache history, not
    // 9 independent genuine measurements. Called before warmup at the tuner's
    // two isolated measurement sites (never at the interleaved final/
    // confirmation sites, which share a cache deliberately) so THIS
    // candidate's own warmup allocation is what gets cached, and every sample
    // after it reuses that exact-sized buffer rather than a stranger's.
    void bc_workspace_clear_cache() override { clear_pool(); }
#endif
"""


_MEMBERS = """
#ifdef GGML_HIP_WORKSPACE_METRICS
    // bigcherry (HI52): measured workspace accounting.
    //
    // Maintained by `ggml_cuda_pool_alloc` below, which is the only thing in the
    // backend that calls `alloc`/`free` on a pool -- so these cannot be bypassed,
    // and cannot drift, because `free` is always given back the `actual_size`
    // that `alloc` returned.
    //
    // Deliberately non-virtual and defined here rather than per-implementation:
    // both concrete pools then get identical semantics for free, and the legacy
    // pool (the only one that runs on RDNA, where VMM is unsupported) does not
    // have to grow a counter across three separate return paths.
    size_t bc_workspace_in_use = 0;
    size_t bc_workspace_peak   = 0;

    void bc_workspace_note_alloc(size_t bytes) {
        bc_workspace_in_use += bytes;
        if (bc_workspace_in_use > bc_workspace_peak) {
            bc_workspace_peak = bc_workspace_in_use;
        }
    }
    void bc_workspace_note_free(size_t bytes) {
        // Guarded rather than trusted: an unbalanced free would wrap this
        // unsigned counter to a colossal value and every later reading would be
        // garbage that still looks like a plausible measurement.
        bc_workspace_in_use = bytes <= bc_workspace_in_use
            ? bc_workspace_in_use - bytes : 0;
    }

    size_t bc_workspace_bytes() const { return bc_workspace_in_use; }
    size_t bc_workspace_peak_bytes() const { return bc_workspace_peak; }

    // Rebases the high-water mark to whatever is live right now, so a caller
    // measures its own peak *increment* rather than inheriting memory that was
    // already outstanding for unrelated reasons.
    void bc_workspace_reset_peak() { bc_workspace_peak = bc_workspace_in_use; }

    // HI56: drop every cached-but-freed buffer, so the NEXT alloc() this pool
    // serves is a genuine fresh allocation rather than a best-fit reuse of
    // whatever a PRIOR, unrelated caller left behind. Default no-op: only the
    // legacy pool has a size-mismatched reuse cache to clear (see the
    // override below); a pool with nothing to clear correctly does nothing.
    virtual void bc_workspace_clear_cache() {}
#endif
"""

POOL_METRICS_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/common.cuh",
    description="pool-level measured workspace accounting (HI52 part 1, HI40 requested-size fix)",
    edits=(
        Edit(
            id="pool-workspace-members",
            anchor=r"^    virtual void free\(void \* ptr, size_t size\) = 0;$",
            rationale="the ggml_cuda_pool abstract interface",
            text=_MEMBERS,
            guard=r"bc_workspace_note_alloc",
        ),
        Edit(
            id="pool-alloc-requested-size-member",
            anchor=r"^    size_t actual_size = 0;$",
            rationale="ggml_cuda_pool_alloc<T>'s own fields, alongside actual_size",
            text=(
                "\n#ifdef GGML_HIP_WORKSPACE_METRICS\n"
                "    // bigcherry (HI40): what THIS candidate asked for, not what the\n"
                "    // pool happened to hand back. `actual_size` can be a best-fit\n"
                "    // reuse hit reflecting a stranger's leftover cached buffer -- the\n"
                "    // exact contamination HI56's cache-clear call works around -- so\n"
                "    // accounting must key off the request, not the allocation.\n"
                "    size_t bc_requested_size = 0;\n"
                "#endif\n"
            ),
            guard=r"bc_requested_size",
        ),
        Edit(
            id="pool-workspace-note-free",
            anchor=r"^            pool->free\(ptr, actual_size\);$",
            rationale="ggml_cuda_pool_alloc's destructor, the only free call site",
            text=(
                "\n#ifdef GGML_HIP_WORKSPACE_METRICS\n"
                "            pool->bc_workspace_note_free(this->bc_requested_size); // bigcherry (HI40)\n"
                "#endif\n"),
            guard=r"bc_workspace_note_free\(this->bc_requested_size\)",
        ),
        Edit(
            id="pool-workspace-note-alloc",
            anchor=r"^        ptr = \(T \*\) pool->alloc\(size \* sizeof\(T\), &this->actual_size\);$",
            rationale="ggml_cuda_pool_alloc::alloc, the only alloc call site",
            text=(
                "\n#ifdef GGML_HIP_WORKSPACE_METRICS\n"
                "        this->bc_requested_size = size * sizeof(T); // bigcherry (HI40)\n"
                "        pool->bc_workspace_note_alloc(this->bc_requested_size); // bigcherry (HI52)\n"
                "#endif\n"
            ),
            guard=r"bc_workspace_note_alloc\(this->bc_requested_size\)",
        ),
    ),
)

POOL_LEG_CLEAR_CACHE_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="expose ggml_cuda_pool_leg's cache clear for isolated measurement (HI56)",
    edits=(
        Edit(
            id="pool-leg-clear-cache",
            # `clear_pool()` -- unlike `alloc(...)`, which both concrete pools
            # define -- is a method name unique to ggml_cuda_pool_leg, so this
            # anchor cannot accidentally attach to ggml_cuda_pool_vmm.
            anchor=r"^    void clear_pool\(\) \{$",
            mode="insert_before",
            rationale="ggml_cuda_pool_leg, immediately before clear_pool()",
            text=_LEG_CLEAR_CACHE,
            guard=r"bc_workspace_clear_cache\(\) override",
        ),
    ),
)

PATCHES = (POOL_METRICS_PATCH, POOL_LEG_CLEAR_CACHE_PATCH)
