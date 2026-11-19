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
    "bc_shared_slot.cpp": "de4ea6a2e6001a34e00cec18b96a7951daa88eaad740031e68ce81c4aed7e3e4",
    "bc_shared_slot.h": "e1ca434a4ec14bddac2c91a5f924a5557c9a39220c9714506846fff85c4c9a46",
    "bc_shared_slot_test.cpp": "e820233bf87f51ea8b038e07cb81612be4b226280e1d8424865d74fb62480492",
}


def _src_text(name: str) -> str:
    data = (SRC / name).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if SRC_SHA256.get(name) != got:
        raise RuntimeError(f"{name}: sha256 {got} is not the pinned {SRC_SHA256.get(name)}; update SRC_SHA256")
    return data.decode("utf-8")

RV_KEY_PREFIX = "libr4d.ar.rendezvous/v2"


def _edit(id: str, before: str, after: str, *, guard: str, max_span: int | None = None) -> Edit:
    return Edit(
        id=id,
        anchor=re.escape(before),
        mode="replace",
        text=after,
        guard=re.escape(guard),
        expect_matches=1,
        max_span_lines=max_span or max(6, before.count("\n") + 1),
        rationale="VTP14 N3: one all-reduce rendezvous across kernel libraries",
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

PATCHES = (
    AR_ENTRY,
    ENGINE,
    _created("core/plugin/bc_shared_slot.h", "bc_shared_slot.h", "BC_VTP14_N3: one process-wide named slot"),
    _created("core/plugin/bc_shared_slot.cpp", "bc_shared_slot.cpp", "BC_VTP14_N3: see bc_shared_slot.h"),
    _created("tests/bc_shared_slot_test.cpp", "bc_shared_slot_test.cpp", "BC_VTP14_N3 (BigCherry managed patch rad0017)"),
)
