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
    "bc_ar_rendezvous_impl.h": "154d3b33222b6be90db9ab4d30ba7c12d5f87d07ee152f62571ce2963e571f8d",
    "bc_ar_lifecycle_impl.h": "f91c125cb86f643833728b3e96eeb8e4869b3469828004716f682d2188387c1b",
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


# The two owned headers preserve the frozen C++ typed definitions. They live
# at two stable attachment sites in the all-reduce unit; failed ranks remain
# published until the owning engine retires their allocations.
AR_ENTRY = FilePatch(
    path="libr4d/r4d_ar_entry.hip", language="none",
    description="Two typed rad0017 rendezvous/lifecycle implementation hooks",
    edits=(
        Edit(
            id="n3-rendezvous-hook",
            anchor=(r"(?s)\#include \"r4d_ar_links\.h\"\n.*?"
                    r"(?=/\* One resolved instance's view of the world:)"),
            mode="replace", text='#include "bc_ar_rendezvous_impl.h"\n\n\n',
            guard=re.escape('#include "bc_ar_rendezvous_impl.h"'),
            expect_matches=1, max_span_lines=240,
            rationale="long-anchor: replace one bounded rendezvous section with the typed owned definition",
        ),
        Edit(
            id="n3-lifecycle-hook",
            anchor=(r"(?s)struct ArInst \{.*?"
                    r"(?=\}  /\* namespace \*/)"),
            mode="replace", text='#include "bc_ar_lifecycle_impl.h"\n\n\n',
            guard=re.escape('#include "bc_ar_lifecycle_impl.h"'),
            expect_matches=1, max_span_lines=650,
            rationale="long-anchor: replace one bounded rank lifecycle section with its typed owned implementation",
        ),
    ),
)

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
    _created("libr4d/bc_ar_rendezvous_impl.h", "bc_ar_rendezvous_impl.h", "BC_VTP14_N3_OWNER"),
    _created("libr4d/bc_ar_lifecycle_impl.h", "bc_ar_lifecycle_impl.h", "BC_VTP14_N3_RANK_FREE"),
    AR_IPC,
    ENGINE,
    ENGINE_RETIRE,
    _created("core/plugin/bc_shared_slot.h", "bc_shared_slot.h", "BC_VTP14_N3: one process-wide named slot"),
    _created("core/plugin/bc_shared_slot.cpp", "bc_shared_slot.cpp", "BC_VTP14_N3: see bc_shared_slot.h"),
    _created("tests/bc_shared_slot_test.cpp", "bc_shared_slot_test.cpp", "BC_VTP14_N3 (BigCherry managed patch rad0017)"),
)
