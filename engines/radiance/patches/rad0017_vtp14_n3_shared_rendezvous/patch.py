"""rad0017 (VTP14 N3): one all-reduce rendezvous shared by every kernel library in the process.

libr4d's all-reduce publishes each rank's buffers through a `Rendezvous g_rv` in the library's own
static storage, and meets its peers at rv_barrier(world). Kernel libraries are dlopened RTLD_LOCAL,
so in a world whose ranks run different libraries -- libr3 (BigCherry's gfx1100 build of the libr4d
source) on the RX 7900 XTX ranks, libr4d on the R9700 rank -- each library counts only its own ranks,
nobody reaches `world`, and declare fails after the 60 s barrier timeout (reproduced on cards, TP2 and
wider; VTP14.md).

- core: rad_bc_shared_slot(key, size, align, init, out, why, len), exported by the engine binary
  (ENABLE_EXPORTS, whole-archive rad_core): one zeroed, aligned slot per key for the process,
  constructed once by the first caller and never destroyed; a key asked for with another size or
  alignment is refused with the reason. bc_shared_slot_users(key) names the libraries that resolved
  it.
- libr4d (and libr3, which is generated from this source): g_rv becomes the shared slot, keyed on the
  rendezvous layout (size, alignment, rank bound, mutex/cv/link sizes), when the host exports the
  service. A host without it (r4d_selftest, kbench, the test binaries) cannot form a mixed world and
  keeps the library-local rendezvous. When the service exists but refuses the slot, the rendezvous is
  poisoned: every barrier returns RAD_E_STATE at once with the reason printed -- never a silent
  fallback to the local copy, which would bring the deadlock back.
- rv_barrier poisons the rendezvous on timeout and wakes every waiter, so the other ranks fail now
  instead of each waiting out its own 60 s.
- engine (with --rank-devices): after declare, the receipt names the libraries sharing the slot.

Source-pinned to radiance v1.17.0 (19d4e8fe). libr3's rewrite anchors in r4d_ar_entry.hip are left
intact (the <condition_variable>/<cstdio> pair, ArInst's host/gate_ctr, align16, the flag words).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from bigcherry.patcher import Edit, FilePatch

SRC = Path(__file__).resolve().parent / "src"

# The build identity hashes this file's bytes only (patch_implementation_digest), so the src/ files it
# creates are pinned here: a src/ change must update these digests, which changes this file and the build.
SRC_SHA256 = {
    "bc_shared_slot.cpp": "663c5618be297b2f8b3e9e67f187940e4aaa56cdb2b92e5846aa859d9d832bec",
    "bc_shared_slot.h": "5c68f1b7bc09fa31d846022c6f468e9219197f146f4e7a9b60bd62094d521506",
    "bc_shared_slot_test.cpp": "e820233bf87f51ea8b038e07cb81612be4b226280e1d8424865d74fb62480492",
}


def _src_text(name: str) -> str:
    data = (SRC / name).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if SRC_SHA256.get(name) != got:
        raise RuntimeError(f"{name}: sha256 {got} is not the pinned {SRC_SHA256.get(name)}; update SRC_SHA256")
    return data.decode("utf-8")

RV_KEY_PREFIX = "libr4d.ar.rendezvous/v2"


def _edit(id: str, before: str, after: str, *, guard: str, max_span: int | None = None,
          rationale: str = "VTP14 N3: one all-reduce rendezvous across kernel libraries") -> Edit:
    return Edit(
        id=id,
        anchor=re.escape(before),
        mode="replace",
        text=after,
        guard=re.escape(guard),
        expect_matches=1,
        max_span_lines=max_span or max(6, before.count("\n") + 1),
        rationale=rationale,
    )


def _created(path: str, source: str, guard: str) -> FilePatch:
    return FilePatch(
        path=path,
        create=True,
        description="VTP14 N3 " + source,
        language="none",
        edits=(Edit(
            id="create-" + source.replace(".", "-").replace("_", "-"),
            anchor=r"\A",
            mode="insert_after",
            text=_src_text(source),
            guard=re.escape(guard),
            max_span_lines=1,
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any collision.",
        ),),
    )


AR_ENTRY = FilePatch(path="libr4d/r4d_ar_entry.hip", language="none", description="VTP14 N3 shared rendezvous", edits=(
    _edit(
        "n3-include",
        "#include \"r4d_ar_links.h\"\n",
        "#include \"r4d_ar_links.h\"\n"
        "\n"
        "#include <dlfcn.h> /* BC_VTP14_N3_INCLUDE: the engine's shared-slot service */\n"
        "#include <new>\n",
        guard="BC_VTP14_N3_INCLUDE",
        max_span=2,
    ),
    _edit(
        "n3-rv-owner",
        r"""    int  failed = 0;""",
        r"""    int  failed = 0;
    int  owner = 0;  /* BC_VTP14_N3_OWNER: engine lifetime within this group */""",
        guard="BC_VTP14_N3_OWNER",
    ),
    _edit(
        "n3-shared-rv",
        r"""Rendezvous& rv_for(long long group) {
    std::lock_guard<std::mutex> lk(g_rv_mu);
    std::unique_ptr<Rendezvous>& r = g_rvs[group];
    if (!r) r.reset(new Rendezvous());
    return *r;
}

""",
        r"""Rendezvous& rv_for(long long group) {
    using SlotFn = int (*)(const char*, size_t, size_t, void (*)(void*), void**, char*, size_t);
    using OwnerFn = int (*)();
    static const auto slot = reinterpret_cast<SlotFn>(dlsym(RTLD_DEFAULT, "rad_bc_shared_slot"));
    static const auto current_owner =
        reinterpret_cast<OwnerFn>(dlsym(RTLD_DEFAULT, "rad_bc_rendezvous_owner"));
    // BC_VTP14_N3_SHARED_RV: standalone tools use their local per-group rendezvous.
    if (slot && current_owner) {
        char key[96], why[320] = {};
        std::snprintf(key, sizeof key, "libr4d.ar.rendezvous/v2/group=%lld", group);
        void* storage = nullptr;
        if (slot(key, sizeof(Rendezvous), alignof(Rendezvous),
                 [](void* p) { new (p) Rendezvous(); }, &storage, why, sizeof why) == RAD_OK && storage) {
            auto& rv = *static_cast<Rendezvous*>(storage);
            const int owner = current_owner();
            std::lock_guard<std::mutex> lk(rv.m);
            if (rv.owner != owner) {
                // A slot base can be claimed again after all prior ranks have joined.
                rv.owner = owner;
                rv.failed = 0;
                rv.arrived = 0;
                rv.epoch = 0;
                std::memset(rv.scratch, 0, sizeof rv.scratch);
                std::memset(rv.flags, 0, sizeof rv.flags);
                std::memset(rv.seq, 0, sizeof rv.seq);
                std::memset(rv.dev, 0, sizeof rv.dev);
            }
            return rv;
        }
        std::fprintf(stderr, "[r4d] shared all-reduce group %lld: %s\n", group, why);
    }
    std::lock_guard<std::mutex> lk(g_rv_mu);
    std::unique_ptr<Rendezvous>& r = g_rvs[group];
    if (!r) {
        r.reset(new Rendezvous());
        if (slot || current_owner) r->failed = 1;
    }
    return *r;
}

""",
        guard="BC_VTP14_N3_SHARED_RV",
    ),
    _edit(
        "n3-barrier-poison",
        r"""int rv_barrier(Rendezvous& rv, int world) {
    std::unique_lock<std::mutex> lk(rv.m);
    const int e = rv.epoch;
    if (++rv.arrived == world) {
        rv.arrived = 0;
        rv.epoch++;
        rv.cv.notify_all();
        return RAD_OK;
    }
    const bool ok = rv.cv.wait_for(lk, std::chrono::seconds(60),
                                   [&] { return rv.epoch != e; });
    /* A rank that gave up takes its arrival back: a peer that failed before arriving would
     * otherwise leave it counted, and the group's next engine (its id is reused) would be released
     * one rank early. */
    if (!ok) --rv.arrived;
    return ok ? RAD_OK : RAD_E_STATE;
}

""",
        r"""int rv_barrier(Rendezvous& rv, int world) {
    std::unique_lock<std::mutex> lk(rv.m);
    if (rv.failed) return RAD_E_STATE;  /* BC_VTP14_N3_POISON */
    const int e = rv.epoch;
    if (++rv.arrived == world) {
        rv.arrived = 0;
        rv.epoch++;
        rv.cv.notify_all();
        return RAD_OK;
    }
    const bool ok = rv.cv.wait_for(lk, std::chrono::seconds(60),
                                   [&] { return rv.epoch != e || rv.failed; });
    /* A rank that gave up takes its arrival back: a peer that failed before arriving would
     * otherwise leave it counted, and the group's next engine (its id is reused) would be released
     * one rank early. */
    if (rv.failed) return RAD_E_STATE;
    if (!ok) {
        --rv.arrived;
        if (rv.owner != 0) {
            std::fprintf(stderr, "[r4d] all-reduce group timed out at %d of %d ranks\n",
                         rv.arrived + 1, world);
            rv.failed = 1;
            rv.cv.notify_all();
        }
    }
    return ok ? RAD_OK : RAD_E_STATE;
}

""",
        guard="BC_VTP14_N3_POISON",
    ),
    _edit(
        "n3-rank-buffer-cleanup",
        r"""/* Build the world. `max_numel` is the largest message any launch on this instance carries.""",
        r"""void free_rank_buffers(ArInst* in) {  /* BC_VTP14_N3_RANK_FREE */
    /* Free only this rank's own allocations; peer pointers belong to the peer. */
    if (in->host) {
        r4d_ar_host_free(in->my_scratch);
        r4d_ar_host_free(in->my_flags);
    } else {
        r4d_ar_ipc_free(in->my_scratch);
        r4d_ar_ipc_free(in->my_flags);
    }
    r4d_ar_ipc_free(in->my_seq);
    r4d_ar_ipc_free(in->locpack);
    r4d_ar_ipc_free(in->my_stage);
}

void free_failed_rank(void* instance) {
    auto* in = static_cast<ArInst*>(instance);
    int previous = -1;
    const bool restore = hipGetDevice(&previous) == hipSuccess && previous != in->allocation_device;
    if (hipSetDevice(in->allocation_device) != hipSuccess)
        std::fprintf(stderr, "[r4d] could not bind device %d for failed all-reduce cleanup\n",
                     in->allocation_device);
    free_rank_buffers(in);
    if (restore) (void)hipSetDevice(previous);
    delete in;
}

using DeferRankCleanup = int (*)(void*, void*, void (*)(void*));
using ForgetRankCleanup = void (*)(void*, void*);

struct RankCleanupBridge {
    DeferRankCleanup defer = reinterpret_cast<DeferRankCleanup>(
        dlsym(RTLD_DEFAULT, "rad_bc_defer_rank_cleanup"));
    ForgetRankCleanup forget = reinterpret_cast<ForgetRankCleanup>(
        dlsym(RTLD_DEFAULT, "rad_bc_forget_rank_cleanup"));
};

const RankCleanupBridge& rank_cleanup_bridge() {
    static const RankCleanupBridge bridge;
    return bridge;
}

/* Build the world. `max_numel` is the largest message any launch on this instance carries.""",
        guard="BC_VTP14_N3_RANK_FREE",
        rationale="Free rank-local all-reduce allocations when initialization fails before publication.",
    ),
    _edit(
        "n3-own-buffers-before-publication",
        r"""    char handle[R4D_AR_HANDLE_BYTES];
    int  hlen = 0;
    long sc = 0, fl = 0, sq = 0, lp = 0;
    int  dev = 0;

    /* Fine-grained: the peer writes the scratch and this rank reads it bypassing L2, so a
     * coarse-grained allocation could leave the peer's bytes in a cache this rank never
     * invalidates, and the launches run without an acquire fence. r4d_ar_ipc_alloc throws when it
     * cannot allocate fine-grained memory; there is no coarse fallback.
     *
     * Without peer access the scratch and flags move to pinned host memory and nothing else
     * changes: kernels touch other ranks only through those two, so no card addresses another's
     * VRAM. A message then crosses two links instead of one; only rows requiring device.p2p == 0
     * take this path (r4d_rows.cpp). Sequence counters stay in VRAM, since only their own rank
     * reads them, and coarse-grained: only this rank's kernels touch them, with device- and
     * agent-scope atomics, which then complete in L2 rather than past it -- the first returning
     * atomic of every launch (r4d_ar_seq_take) is on the push's path. */
    R4D_TRY("ar_ipc_alloc",
        if (hipGetDevice(&dev) != hipSuccess) return RAD_E_DEVICE;
        if (host) {
            sc = r4d_ar_host_alloc((long)nslots * in->slot_bytes);
            fl = r4d_ar_host_alloc((flag_words + 2L * maxb) * 4 + kWakeBytes);
        } else {
            sc = r4d_ar_ipc_alloc((long)nslots * in->slot_bytes, 1, handle, &hlen);
            fl = r4d_ar_ipc_alloc((flag_words + 2L * maxb) * 4 + kWakeBytes, 1, handle, &hlen);
        }
        sq = r4d_ar_ipc_alloc(kSeqWords(maxb) * 4, 0, handle, &hlen));

    /* A host mailbox is read over the link past every cache, so the 2-rank kernels' push check
     * copies what it reads into this VRAM mirror for the reduce, crossing the link once
     * (r4d_ar_seq.h). Coarse-grained: only this rank's kernels touch it, each block its own words.
     * The wide family has no push check and reads its slots once. Host-memory slots free the same
     * VRAM over P2P, so the mirror keeps the footprint where the P2P instance has it. */
    long stg = 0;
    if (host && world == 2) {
        R4D_TRY("ar_ipc_alloc(stage)",
            stg = r4d_ar_ipc_alloc((long)nslots * in->slot_bytes, 0, handle, &hlen));
    }

    /* This rank's own packed copy, for the rotated 6-bit wire only: the reduce must fold exactly
     * the bytes it sent. The 6-bit payload is 3/8 of the bf16 message and the f32 scales 1/32 of
     * it, laid out from half the slot; a row padded to whole 512-element chunks
     * (ar_rmsnorm_quant_fp8) adds up to 7 groups' scales a row, so the scales get a sixteenth
     * (enough for any row of 512 or more) plus a page. ar_rmsnorm_quant checks the fit. */
    if (in->dtype != 2) {
        R4D_TRY("ar_ipc_alloc(locpack)",
            lp = r4d_ar_ipc_alloc(in->slot_bytes / 2 + in->slot_bytes / 16 + 4096, 0,
                                  handle, &hlen));
    }

    in->my_scratch = sc;
    in->my_flags   = fl;
    in->my_seq     = sq;
    in->locpack    = lp;
    in->my_stage   = stg;
    in->gate_ctr   = sq + 2L * maxb * 4;   /* the word after the sequence counters (r4d_ar_seq.h) */

    Rendezvous& rv = rv_for(group);
    {
        std::lock_guard<std::mutex> lk(rv.m);
        rv.scratch[rank] = sc;
        rv.flags[rank]   = fl;
        rv.seq[rank]     = sq;
        rv.dev[rank]     = dev;
    }
""",
        r"""    char handle[R4D_AR_HANDLE_BYTES];
    int  hlen = 0;
    int  dev = 0;

    /* Fine-grained: the peer writes the scratch and this rank reads it bypassing L2, so a
     * coarse-grained allocation could leave the peer's bytes in a cache this rank never
     * invalidates, and the launches run without an acquire fence. r4d_ar_ipc_alloc throws when it
     * cannot allocate fine-grained memory; there is no coarse fallback.
     *
     * Without peer access the scratch and flags move to pinned host memory and nothing else
     * changes: kernels touch other ranks only through those two, so no card addresses another's
     * VRAM. A message then crosses two links instead of one; only rows requiring device.p2p == 0
     * take this path (r4d_rows.cpp). Sequence counters stay in VRAM, since only their own rank
     * reads them, and coarse-grained: only this rank's kernels touch them, with device- and
     * agent-scope atomics, which then complete in L2 rather than past it -- the first returning
     * atomic of every launch (r4d_ar_seq_take) is on the push's path. */
    // BC_VTP14_N3_UNPUBLISHED: clean up any acquired buffers if allocation or setup fails.
    std::unique_ptr<ArInst, decltype(&free_rank_buffers)> unpublished(in, &free_rank_buffers);
    R4D_TRY("ar_ipc_alloc",
        if (hipGetDevice(&dev) != hipSuccess) return RAD_E_DEVICE;
        if (host) {
            in->my_scratch = r4d_ar_host_alloc((long)nslots * in->slot_bytes);
            in->my_flags = r4d_ar_host_alloc((flag_words + 2L * maxb) * 4 + kWakeBytes);
        } else {
            in->my_scratch = r4d_ar_ipc_alloc((long)nslots * in->slot_bytes, 1, handle, &hlen);
            in->my_flags = r4d_ar_ipc_alloc((flag_words + 2L * maxb) * 4 + kWakeBytes, 1, handle, &hlen);
        }
        in->my_seq = r4d_ar_ipc_alloc(kSeqWords(maxb) * 4, 0, handle, &hlen));

    /* A host mailbox is read over the link past every cache, so the 2-rank kernels' push check
     * copies what it reads into this VRAM mirror for the reduce, crossing the link once
     * (r4d_ar_seq.h). Coarse-grained: only this rank's kernels touch it, each block its own words.
     * The wide family has no push check and reads its slots once. Host-memory slots free the same
     * VRAM over P2P, so the mirror keeps the footprint where the P2P instance has it. */
    if (host && world == 2) {
        R4D_TRY("ar_ipc_alloc(stage)",
            in->my_stage = r4d_ar_ipc_alloc((long)nslots * in->slot_bytes, 0, handle, &hlen));
    }

    /* This rank's own packed copy, for the rotated 6-bit wire only: the reduce must fold exactly
     * the bytes it sent. The 6-bit payload is 3/8 of the bf16 message and the f32 scales 1/32 of
     * it, laid out from half the slot; a row padded to whole 512-element chunks
     * (ar_rmsnorm_quant_fp8) adds up to 7 groups' scales a row, so the scales get a sixteenth
     * (enough for any row of 512 or more) plus a page. ar_rmsnorm_quant checks the fit. */
    if (in->dtype != 2) {
        R4D_TRY("ar_ipc_alloc(locpack)",
            in->locpack = r4d_ar_ipc_alloc(in->slot_bytes / 2 + in->slot_bytes / 16 + 4096, 0,
                                  handle, &hlen));
    }

    in->gate_ctr   = in->my_seq + 2L * maxb * 4;   /* the word after the sequence counters (r4d_ar_seq.h) */

    Rendezvous& rv = rv_for(group);
    {
        std::lock_guard<std::mutex> lk(rv.m);
        if (rv.failed) return RAD_E_STATE;
        if (rv.owner != 0) {
            const auto& bridge = rank_cleanup_bridge();
            const int rc = bridge.defer && bridge.forget
                ? bridge.defer(&rv, in, free_failed_rank) : RAD_E_STATE;
            if (rc != RAD_OK) {
                rv.failed = 1;
                rv.cv.notify_all();
                return rc;
            }
            in->allocation_device = dev;
            in->pending_rv = &rv;
        }
        rv.scratch[rank] = in->my_scratch;
        rv.flags[rank]   = in->my_flags;
        rv.seq[rank]     = in->my_seq;
        rv.dev[rank]     = dev;
        // Published peer pointers require group-level unwind on failure.
        unpublished.release();
    }
""",
        guard="BC_VTP14_N3_UNPUBLISHED",
        rationale="Free rank-local all-reduce allocations when initialization fails before publication.",
    ),
    _edit(
        "n3-release-rank-buffers",
        r"""    /* Free only this rank's own allocations; peer pointers belong to the peer. */
    if (in->host) {
        r4d_ar_host_free(in->my_scratch);
        r4d_ar_host_free(in->my_flags);
    } else {
        r4d_ar_ipc_free(in->my_scratch);
        r4d_ar_ipc_free(in->my_flags);
    }
    r4d_ar_ipc_free(in->my_seq);
    r4d_ar_ipc_free(in->locpack);
    r4d_ar_ipc_free(in->my_stage);
    delete in;""",
        r"""    free_rank_buffers(in);  /* BC_VTP14_N3_RELEASE */
    delete in;""",
        guard="BC_VTP14_N3_RELEASE",
        rationale="Free rank-local all-reduce allocations when initialization fails before publication.",
    ),
    _edit(
        "n3-own-published-rank",
        r"""    long my_scratch = 0, my_flags = 0, my_seq = 0, locpack = 0;""",
        r"""    long my_scratch = 0, my_flags = 0, my_seq = 0, locpack = 0;
    Rendezvous* pending_rv = nullptr;  /* BC_VTP14_N3_RANK_PENDING */
    int allocation_device = -1;""",
        guard="BC_VTP14_N3_RANK_PENDING",
    ),
    _edit(
        "n3-defer-failed-rank",
        r"""    int rc = build(in, k.group, k.rank, k.world, k.numel, k.dtype, k.host);
    if (rc != RAD_OK) { delete in; return rc; }
    {
        std::lock_guard<std::mutex> lk(g_cache_mu);
        g_cache.push_back(CacheSlot{ k, in, 1 });
    }
    *out = in;""",
        r"""    // BC_VTP14_N3_FAILED_RANK: once published, failed instances stay live until the owner retires them.
    int rc = build(in, k.group, k.rank, k.world, k.numel, k.dtype, k.host);
    if (rc == RAD_OK) {
        try {
            std::lock_guard<std::mutex> lk(g_cache_mu);
            g_cache.push_back(CacheSlot{ k, in, 1 });
        } catch (const std::bad_alloc&) {
            rc = RAD_E_NOMEM;
        }
    }
    Rendezvous* rv = in->pending_rv;
    if (rc != RAD_OK) {
        if (rv) {
            std::lock_guard<std::mutex> lk(rv->m);
            rv->failed = 1;
            rv->cv.notify_all();
        } else {
            delete in;
        }
        return rc;
    }
    if (rv) {
        rank_cleanup_bridge().forget(rv, in);
        in->pending_rv = nullptr;
    }
    *out = in;""",
        guard="BC_VTP14_N3_FAILED_RANK",
    ),
))

ENGINE = FilePatch(path="core/engine_bringup.cpp", language="none", description="VTP14 N3 receipt", edits=(
    _edit(
        "n3-engine-include",
        "#include \"plugin/bc_rank_devices.h\" /* BC_VTP14_N1_ENGINE_INCLUDE */\n",
        "#include \"plugin/bc_rank_devices.h\" /* BC_VTP14_N1_ENGINE_INCLUDE */\n"
        "#include \"plugin/bc_shared_slot.h\" /* BC_VTP14_N3_ENGINE_INCLUDE */\n",
        guard="BC_VTP14_N3_ENGINE_INCLUDE",
        max_span=2,
    ),
    _edit(
        "n3-engine-receipt",
        "        if (bc_rank_collectives_agree(colls, &why) != RAD_OK) {\n"
        "            RAD_ERR(\"--rank-devices: the ranks' collectives differ: %s\", why.c_str());\n"
        "            return RAD_E_STATE;\n"
        "        }\n",
        "        if (bc_rank_collectives_agree(colls, &why) != RAD_OK) {\n"
        "            RAD_ERR(\"--rank-devices: the ranks' collectives differ: %s\", why.c_str());\n"
        "            return RAD_E_STATE;\n"
        "        }\n"
        "        /* BC_VTP14_N3_ENGINE_RECEIPT: which libraries met at the shared all-reduce rendezvous. */\n"
        "        {\n"
        "            std::string users;\n"
        "            const std::string k = \"" + RV_KEY_PREFIX + "/group=\" + std::to_string(slot_base_);\n"
        "            for (const std::string& u : bc_shared_slot_users(k))\n"
        "                    users += (users.empty() ? \"\" : \", \") + u;\n"
        "            RAD_INFO(\"rank receipt: all-reduce rendezvous shared by %s\",\n"
        "                     users.empty() ? \"no library (no all-reduce was initialised)\" : users.c_str());\n"
        "        }\n",
        guard="BC_VTP14_N3_ENGINE_RECEIPT",
    ),
))

AR_IPC = FilePatch(path="libr4d/r4d_ar_oneshot_2rank_exact.hip", language="none",
                   description="Release the all-reduce IPC buffer after initialization failure", edits=(
    _edit(
        "n3-ipc-init-cleanup",
        r"""  HIPOK(hipMemset(p, 0, size));
  // The zeroes must land before the pointer reaches a peer, or its first push or flag store could
  // race the queued memset.
  HIPOK(hipStreamSynchronize(nullptr));
  // The handle is for an out-of-process peer.""",
        r"""  try {  // BC_VTP14_N3_IPC_INIT: ownership passes to ArInst only on return.
    HIPOK(hipMemset(p, 0, size));
    // The zeroes must land before the pointer reaches a peer, or its first push or flag store could
    // race the queued memset.
    HIPOK(hipStreamSynchronize(nullptr));
  } catch (...) {
    (void)hipFree(p);
    throw;
  }
  // The handle is for an out-of-process peer.""",
        guard="BC_VTP14_N3_IPC_INIT",
        rationale="Free rank-local all-reduce allocations when initialization fails before publication.",
    ),
))

ENGINE_RETIRE = FilePatch(path="core/engine.cpp", language="none",
                          description="Release failed rendezvous allocations after the owning engine drains",
                          edits=(
    _edit(
        "n3-retire-rank-include",
        '#include "build/startup.h"\n',
        '#include "build/startup.h"\n'
        '#include "plugin/bc_shared_slot.h" /* BC_VTP14_N3_RETIRE_INCLUDE */\n',
        guard="BC_VTP14_N3_RETIRE_INCLUDE",
    ),
    _edit(
        "n3-retire-rank",
        "    for (auto& r : ranks_) r->ctx.shutdown();\n"
        "    if (cards_claimed_) {",
        "    for (auto& r : ranks_) r->ctx.shutdown();\n"
        "    // BC_VTP14_N3_RETIRE: rank workers have joined, and the library and devices are still live.\n"
        "    bc_shared_slot_retire_owner(owner_);\n"
        "    if (cards_claimed_) {",
        guard="BC_VTP14_N3_RETIRE:",
    ),
))

PATCHES = (
    AR_ENTRY,
    AR_IPC,
    ENGINE,
    ENGINE_RETIRE,
    _created("core/plugin/bc_shared_slot.h", "bc_shared_slot.h", "BC_VTP14_N3: one process-wide named slot"),
    _created("core/plugin/bc_shared_slot.cpp", "bc_shared_slot.cpp", "BC_VTP14_N3: see bc_shared_slot.h"),
    _created("tests/bc_shared_slot_test.cpp", "bc_shared_slot_test.cpp", "BC_VTP14_N3 (BigCherry managed patch rad0017)"),
)
