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

import re
from pathlib import Path

from bigcherry.patcher import Edit, FilePatch

SRC = Path(__file__).resolve().parent / "src"

RV_KEY_PREFIX = "libr4d.ar.rendezvous/v1"


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
            text=(SRC / source).read_text(encoding="utf-8"),
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
        "n3-shared-rv",
        "Rendezvous g_rv;\n",
        "/* BC_VTP14_N3_SHARED_RV (BigCherry rad0017): ranks of one world may run different kernel\n"
        " * libraries (libr3 on gfx1100, libr4d on gfx1201), each dlopened RTLD_LOCAL with its own statics,\n"
        " * so the rendezvous lives in a process-wide slot the engine exports (rad_bc_shared_slot) and every\n"
        " * library built from this source resolves the same one. A host without the service (selftest,\n"
        " * kbench, test binaries) runs one library and keeps a local rendezvous. A host with it that\n"
        " * refuses the slot gets a poisoned rendezvous: every barrier fails at once, with the reason. */\n"
        "typedef int (*BcSharedSlotFn)(const char*, size_t, size_t, void (*)(void*), void**, char*, size_t);\n"
        "\n"
        "void rv_construct(void* p) { new (p) Rendezvous(); }\n"
        "\n"
        "Rendezvous& rv_slot() {\n"
        "    static Rendezvous* const slot = [] {\n"
        "        static Rendezvous local;\n"
        "        static Rendezvous poisoned;\n"
        "        const auto fn = reinterpret_cast<BcSharedSlotFn>(dlsym(RTLD_DEFAULT, \"rad_bc_shared_slot\"));\n"
        "        if (!fn) return &local;\n"
        "        char key[192];\n"
        "        std::snprintf(key, sizeof key, \"" + RV_KEY_PREFIX + "/size=%zu/align=%zu/ranks=%d/mutex=%zu/cv=%zu/link=%zu\",\n"
        "                      sizeof(Rendezvous), alignof(Rendezvous), (int)R4D_MAX_RANKS, sizeof(std::mutex),\n"
        "                      sizeof(std::condition_variable), sizeof(R4DArLink));\n"
        "        void* p = nullptr;\n"
        "        char why[320] = { 0 };\n"
        "        const int rc = fn(key, sizeof(Rendezvous), alignof(Rendezvous), rv_construct, &p, why, sizeof why);\n"
        "        if (rc != RAD_OK || !p) {\n"
        "            std::fprintf(stderr, \"[r4d] all-reduce rendezvous: the engine's shared slot refused %s: %s. \"\n"
        "                                 \"Every all-reduce init on this library fails rather than wait for \"\n"
        "                                 \"peers it cannot meet.\\n\", key, why[0] ? why : \"no reason given\");\n"
        "            poisoned.failed = 1;\n"
        "            return &poisoned;\n"
        "        }\n"
        "        return static_cast<Rendezvous*>(p);\n"
        "    }();\n"
        "    return *slot;\n"
        "}\n"
        "\n"
        "#define g_rv (rv_slot())\n",
        guard="BC_VTP14_N3_SHARED_RV",
        max_span=2,
    ),
    _edit(
        "n3-barrier-poison",
        "    std::unique_lock<std::mutex> lk(g_rv.m);\n"
        "    const int e = g_rv.epoch;\n"
        "    if (++g_rv.arrived == world) {\n"
        "        g_rv.arrived = 0;\n"
        "        g_rv.epoch++;\n"
        "        g_rv.cv.notify_all();\n"
        "        return RAD_OK;\n"
        "    }\n"
        "    const bool ok = g_rv.cv.wait_for(lk, std::chrono::seconds(60),\n"
        "                                     [&] { return g_rv.epoch != e; });\n"
        "    return ok ? RAD_OK : RAD_E_STATE;\n",
        "    std::unique_lock<std::mutex> lk(g_rv.m);\n"
        "    /* BC_VTP14_N3_POISON: a rendezvous that timed out or could not be shared fails every later\n"
        "     * barrier at once, and a timeout wakes every waiter, so no rank waits out its own 60 s. */\n"
        "    if (g_rv.failed) return RAD_E_STATE;\n"
        "    const int e = g_rv.epoch;\n"
        "    if (++g_rv.arrived == world) {\n"
        "        g_rv.arrived = 0;\n"
        "        g_rv.epoch++;\n"
        "        g_rv.cv.notify_all();\n"
        "        return RAD_OK;\n"
        "    }\n"
        "    const bool ok = g_rv.cv.wait_for(lk, std::chrono::seconds(60),\n"
        "                                     [&] { return g_rv.epoch != e || g_rv.failed; });\n"
        "    if (ok && g_rv.epoch != e) return RAD_OK;\n"
        "    if (!g_rv.failed)\n"
        "        std::fprintf(stderr, \"[r4d] all-reduce rendezvous: %d of %d rank(s) arrived within 60 s; \"\n"
        "                             \"the world's ranks do not meet here (a rank declares a different graph, or \"\n"
        "                             \"runs a library that does not share this rendezvous)\\n\", g_rv.arrived, world);\n"
        "    g_rv.failed = 1;\n"
        "    g_rv.cv.notify_all();\n"
        "    return RAD_E_STATE;\n",
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
        "            for (const std::string& k : bc_shared_slot_keys(\"" + RV_KEY_PREFIX + "/\"))\n"
        "                for (const std::string& u : bc_shared_slot_users(k))\n"
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
