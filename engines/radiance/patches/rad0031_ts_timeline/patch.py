"""rad0031 (VTP36): a diagnostic timeline for the exact two-shot all-reduce (three, four and eight ranks).

rad0023 splits the exact one-shot into push, peer wait and fold, but since rad0021 and rad0024 every prefill and
multi-stream decode all-reduce of a three-rank host-staged world is the link-weighted two-shot, which records nothing.
The two-shot kernel already carries upstream's per-launch timeline (`tl_all`, a device buffer the eight-rank selftest
passes): a stamp after the sequence take, an (arrive, release) pair per handshake and one after the last assemble.
This patch gives every launch of the kernel a ring to stamp into while `R4D_AR_TIMELINE=DIR` is set, as rad0023 does
for the one-shot.

The ring lives in mapped pinned host memory, one per all-reduce instance (keyed by its sequence counters). A launch's
entry is picked on the device from the block's handshake sequence number, so recorded and replayed passes record too.
An entry holds one record for each of the first four blocks: four header words (sequence number; the message's
16-byte words; blocks, pieces, rank and world packed; this rank's shard in words) and upstream's R4D_AR_TL_WORDS
stamps. At exit each ring is written to `DIR/ar-ts-r<rank>-ws<world>-p<pid>-i<n>.bin` (int64 words, -1 unset).

Unset or empty, the launcher passes a null ring and the kernel takes no stamp: the same work and the same bits.
Nothing about a model, world or card is built in. libr3 compiles this unit, so a mixed world's ranks all record.

Source-pinned to radiance v1.17.0 (19d4e8fe), after rad0021 (the three-rank weighted wrapper).
"""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

F = "libr4d/r4d_ar_twoshot_Nrank_exact.hip"


def _edit(id: str, before: str, after: str, *, guard: str, max_span: int | None = None) -> Edit:
    return Edit(
        id=id,
        anchor=re.escape(before),
        mode="replace",
        text=after,
        guard=re.escape(guard),
        expect_matches=1,
        max_span_lines=max_span or max(6, before.count("\n") + 1),
        rationale=("long-anchor: the anchor is one whole launch macro; " if (max_span or 0) > 6 else "")
        + "VTP36: two-shot all-reduce phase timeline (diagnostic)",
    )


TWOSHOT = FilePatch(path=F, language="none", description="VTP36 two-shot all-reduce timeline", edits=(
    _edit(
        "tstl-head",
        '#include "r4d_chain.h"\n'
        "#include <stdexcept>\n"
        "#include <string>\n",
        '#include "r4d_chain.h"\n'
        "#include <stdexcept>\n"
        "#include <string>\n"
        "#include <cstdio>   /* BC_VTP36_TSTL_HEAD (BigCherry rad0031): the two-shot's diagnostic timeline ring */\n"
        "#include <cstdlib>\n"
        "#include <cstring>\n"
        "#include <mutex>\n"
        "#include <vector>\n"
        "#include <unistd.h>\n"
        "\n"
        "// The ring (rad0031): R4D_TSTL_RING entries, each R4D_TSTL_BLOCKS block records of four header words (seq;\n"
        "// the message's 16-byte words; blocks | pieces << 8 | rank << 16 | world << 24; this rank's shard words)\n"
        "// and the kernel's R4D_AR_TL_WORDS stamps.\n"
        "#define R4D_TSTL_RING   8192\n"
        "#define R4D_TSTL_BLOCKS 4\n"
        "#define R4D_TSTL_BWORDS (4 + R4D_AR_TL_WORDS)\n",
        guard="BC_VTP36_TSTL_HEAD",
    ),
    _edit(
        "tstl-kparam",
        "    int slot_stride16, int rank, int drain, int acq, int pub,\n"
        "    long long* gate, unsigned int* gate_ctr, int pieces_arg, long long* tl_all) {\n"
        "  if (CHAIN) r4d_chain_wait();\n"
        "  constexpr int NPEERS = WS - 1;\n"
        "  const int pieces = PIECED ? pieces_arg : 1;\n",
        "    int slot_stride16, int rank, int drain, int acq, int pub,\n"
        "    long long* gate, unsigned int* gate_ctr, int pieces_arg, long long* tl_all,\n"
        "    long long* tlring /* BC_VTP36_TSTL_KPARAM: null unless R4D_AR_TIMELINE */) {\n"
        "  if (CHAIN) r4d_chain_wait();\n"
        "  constexpr int NPEERS = WS - 1;\n"
        "  const int pieces = PIECED ? pieces_arg : 1;\n",
        guard="BC_VTP36_TSTL_KPARAM",
    ),
    _edit(
        "tstl-kstart",
        "  long long* const tl = tl_all && (FORM != R4D_AR8_FORM_SPLIT || 4 * pieces + 2 <= R4D_AR_TL_WORDS)\n"
        "                            ? tl_all + (size_t)b * R4D_AR_TL_WORDS : nullptr;\n"
        "  r4d_ar_tl_stamp(tl, 0);\n",
        "  long long* tl = tl_all && (FORM != R4D_AR8_FORM_SPLIT || 4 * pieces + 2 <= R4D_AR_TL_WORDS)\n"
        "                      ? tl_all + (size_t)b * R4D_AR_TL_WORDS : nullptr;\n"
        "  // BC_VTP36_TSTL_KSTART: this block's record of the launch's ring entry, by its device sequence number\n"
        "  // (replay-safe). A launch given upstream's own timeline keeps that one.\n"
        "  if (!tl && tlring && b < R4D_TSTL_BLOCKS &&\n"
        "      (FORM != R4D_AR8_FORM_SPLIT || 4 * pieces + 2 <= R4D_AR_TL_WORDS)) {\n"
        "    long long* e = tlring + ((size_t)(s_seq % R4D_TSTL_RING) * R4D_TSTL_BLOCKS + (size_t)b) * R4D_TSTL_BWORDS;\n"
        "    if (tid == 0) {\n"
        "      e[0] = (long long)s_seq;\n"
        "      e[1] = (long long)sh.off[WS];\n"
        "      e[2] = (long long)nb | ((long long)pieces << 8) | ((long long)rank << 16) | ((long long)WS << 24);\n"
        "      e[3] = (long long)(sh.off[rank + 1] - sh.off[rank]);\n"
        "    }\n"
        "    tl = e + 4;\n"
        "  }\n"
        "  r4d_ar_tl_stamp(tl, 0);\n",
        guard="BC_VTP36_TSTL_KSTART",
    ),
    _edit(
        "tstl-kmacro",
        "    unsigned int* gate_ctr, int pieces_arg, long long* tl_all\n"
        "#define R4D_ARTS_ARGS in4, peers, my_base, my_flags, out4, seq_ctrs, sh, slot_stride16, rank,    \\\n"
        "    drain, acq, pub, gate, gate_ctr, pieces_arg, tl_all\n",
        "    unsigned int* gate_ctr, int pieces_arg, long long* tl_all, long long* tlring\n"
        "/* BC_VTP36_TSTL_KMACRO: the ring rides behind upstream's timeline */\n"
        "#define R4D_ARTS_ARGS in4, peers, my_base, my_flags, out4, seq_ctrs, sh, slot_stride16, rank,    \\\n"
        "    drain, acq, pub, gate, gate_ctr, pieces_arg, tl_all, tlring\n",
        guard="BC_VTP36_TSTL_KMACRO",
    ),
    _edit(
        "tstl-host",
        "// --- host side ---\n"
        "template <int WS>\n"
        "static void ar_twoshot_launch(const long* peer_scratch, const long* peer_flags, long my_scratch,\n",
        "// --- host side ---\n"
        "\n"
        "// BC_VTP36_TSTL_HOST: one timeline ring per all-reduce instance (keyed by its sequence counters), in mapped\n"
        "// pinned host memory, written to R4D_AR_TIMELINE at exit. Null when the variable is unset or empty.\n"
        "namespace {\n"
        "struct TstlRing { long seq; int rank, ws; long long* host; };\n"
        "std::mutex g_tstl_mu;\n"
        "std::vector<TstlRing>* g_tstl = nullptr;\n"
        "const char* tstl_dir() {\n"
        '  static const char* const d = std::getenv("R4D_AR_TIMELINE");\n'
        "  return (d && *d) ? d : nullptr;\n"
        "}\n"
        "void tstl_dump() {\n"
        "  std::lock_guard<std::mutex> lk(g_tstl_mu);\n"
        "  if (!g_tstl || !tstl_dir()) return;\n"
        "  for (size_t i = 0; i < g_tstl->size(); ++i) {\n"
        "    const TstlRing& r = (*g_tstl)[i];\n"
        "    char path[4096];\n"
        '    std::snprintf(path, sizeof path, "%s/ar-ts-r%d-ws%d-p%d-i%zu.bin", tstl_dir(), r.rank, r.ws,\n'
        "                  (int)getpid(), i);\n"
        '    if (FILE* f = std::fopen(path, "wb")) {\n'
        "      std::fwrite(r.host, sizeof(long long), (size_t)R4D_TSTL_RING * R4D_TSTL_BLOCKS * R4D_TSTL_BWORDS, f);\n"
        "      std::fclose(f);\n"
        "    }\n"
        "  }\n"
        "}\n"
        "long long* tstl_ring(long seq_ctrs, int rank, int ws) {\n"
        "  if (!tstl_dir()) return nullptr;\n"
        "  std::lock_guard<std::mutex> lk(g_tstl_mu);\n"
        "  if (!g_tstl) { g_tstl = new std::vector<TstlRing>(); std::atexit(tstl_dump); }\n"
        "  for (const TstlRing& r : *g_tstl)\n"
        "    if (r.seq == seq_ctrs) {\n"
        "      void* d = nullptr;\n"
        "      return hipHostGetDevicePointer(&d, r.host, 0) == hipSuccess ? (long long*)d : nullptr;\n"
        "    }\n"
        "  const size_t bytes = (size_t)R4D_TSTL_RING * R4D_TSTL_BLOCKS * R4D_TSTL_BWORDS * sizeof(long long);\n"
        "  void* h = nullptr;\n"
        "  if (hipHostMalloc(&h, bytes, hipHostMallocMapped) != hipSuccess || !h) return nullptr;\n"
        "  std::memset(h, 0xff, bytes);\n"
        "  g_tstl->push_back({ seq_ctrs, rank, ws, (long long*)h });\n"
        '  std::fprintf(stderr, "[r4d] two-shot all-reduce timeline: rank %d of %d, instance %zu -> %s\\n", rank, ws,\n'
        "               g_tstl->size() - 1, tstl_dir());\n"
        "  void* d = nullptr;\n"
        "  return hipHostGetDevicePointer(&d, h, 0) == hipSuccess ? (long long*)d : nullptr;\n"
        "}\n"
        "}  // namespace\n"
        "\n"
        "template <int WS>\n"
        "static void ar_twoshot_launch(const long* peer_scratch, const long* peer_flags, long my_scratch,\n",
        guard="BC_VTP36_TSTL_HOST",
    ),
    _edit(
        "tstl-launch",
        "  hipStream_t st = (hipStream_t)stream_i;\n"
        "#define R4D_AR_TS_LAUNCH_F(T, LN, P, F)                                               \\\n"
        "  (chain ? r4dchain_ar_twoshot_wide_exact<T, LN, WS, P, F>                            \\\n"
        "         : r4d_ar_twoshot_wide_exact_kernel<T, LN, WS, P, F>)<<<nb, nt, 0, st>>>(     \\\n"
        "      (const int4*)inp, peers, (const int4*)my_scratch, (unsigned int*)my_flags,      \\\n"
        "      (int4*)out, (unsigned int*)seq_ctrs, sh, (int)slot_stride16,                    \\\n"
        "      (int)rank, (int)drain, (int)acq, (int)pub, (long long*)gate,                    \\\n"
        "      (unsigned int*)gate_ctr, np, (long long*)timeline)\n",
        "  hipStream_t st = (hipStream_t)stream_i;\n"
        "  long long* const tlring = timeline ? nullptr : tstl_ring(seq_ctrs, (int)rank, WS);  /* BC_VTP36_TSTL_LAUNCH */\n"
        "#define R4D_AR_TS_LAUNCH_F(T, LN, P, F)                                               \\\n"
        "  (chain ? r4dchain_ar_twoshot_wide_exact<T, LN, WS, P, F>                            \\\n"
        "         : r4d_ar_twoshot_wide_exact_kernel<T, LN, WS, P, F>)<<<nb, nt, 0, st>>>(     \\\n"
        "      (const int4*)inp, peers, (const int4*)my_scratch, (unsigned int*)my_flags,      \\\n"
        "      (int4*)out, (unsigned int*)seq_ctrs, sh, (int)slot_stride16,                    \\\n"
        "      (int)rank, (int)drain, (int)acq, (int)pub, (long long*)gate,                    \\\n"
        "      (unsigned int*)gate_ctr, np, (long long*)timeline, tlring)\n",
        guard="BC_VTP36_TSTL_LAUNCH",
        max_span=10,
    ),
))

PATCHES = (TWOSHOT,)
