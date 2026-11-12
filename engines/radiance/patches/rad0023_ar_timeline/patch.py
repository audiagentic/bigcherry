"""rad0023 (VTP31): a diagnostic timeline for the exact one-shot all-reduce at three, four and eight ranks.

Where does a decode all-reduce's time go on a host-staged world: the push of this rank's message to its peers'
slots, the wait for the peers' flags, or the read and fold of their messages? `--profile-ops` gives each call's
total only. With `R4D_AR_TIMELINE=DIR` set, the wide one-shot (`r4d_ar_oneshot_wide_exact_kernel`) stamps the device
wall clock (100 MHz) four times in each of its first eight blocks: on entry, when its push has drained (the
handshake's first stamp), when every peer's flag has been seen (its second), and after the fold. The stamps go to a
ring in pinned host memory indexed by the launch's sequence number, which the kernel already takes on the device, so
recorded and replayed passes keep working. At exit each ring is written to `DIR/ar-tl-r<rank>-ws<world>-p<pid>-i<n>.bin`
(int64 words; per entry: seq, words, blocks, rank, then four stamps a block, -1 unset).

Unset, the launcher passes a null ring and the kernel takes no stamp: the same work, the same bits. Nothing about a
model, world or card is built in. libr3 is generated from this source, so a mixed world's ranks both record.

Source-pinned to radiance v1.17.0 (19d4e8fe).
"""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

F = "libr4d/r4d_ar_oneshot_Nrank_exact.hip"


def _edit(id: str, before: str, after: str, *, guard: str, max_span: int | None = None) -> Edit:
    return Edit(
        id=id,
        anchor=re.escape(before),
        mode="replace",
        text=after,
        guard=re.escape(guard),
        expect_matches=1,
        max_span_lines=max_span or max(6, before.count("\n") + 1),
        rationale="VTP31: all-reduce phase timeline (diagnostic)",
    )


ONESHOT = FilePatch(path=F, language="none", description="VTP31 one-shot all-reduce timeline", edits=(
    _edit(
        "artl-head",
        '#include "r4d_ar_hc.h"      // the gather and hc write words, shared with the two-rank fused forms\n'
        "#include <stdexcept>\n"
        "#include <string>\n",
        '#include "r4d_ar_hc.h"      // the gather and hc write words, shared with the two-rank fused forms\n'
        "#include <stdexcept>\n"
        "#include <string>\n"
        "#include <cstdio>   /* BC_VTP31_ARTL_HEAD (BigCherry rad0023): the diagnostic timeline */\n"
        "#include <cstdlib>\n"
        "#include <mutex>\n"
        "#include <vector>\n"
        "#include <unistd.h>\n"
        "\n"
        "// The timeline ring (rad0023): R4D_ARTL_RING entries of R4D_ARTL_WORDS int64 words, a header (seq, words,\n"
        "// blocks, rank) and four stamps for each of the first R4D_ARTL_BLOCKS blocks.\n"
        "#define R4D_ARTL_RING   16384\n"
        "#define R4D_ARTL_BLOCKS 8\n"
        "#define R4D_ARTL_WORDS  (4 + 4 * R4D_ARTL_BLOCKS)\n",
        guard="BC_VTP31_ARTL_HEAD",
        max_span=4,
    ),
    _edit(
        "artl-kparam",
        "    unsigned int* __restrict__ seq_ctrs, int n16, int slot_stride16, int rank,\n"
        "    int drain, int acq, int pub, long long* gate, unsigned int* gate_ctr) {\n"
        "  constexpr int NPEERS = WS - 1;\n"
        "  // The reduce holds WS source words live per output word, so the 2x unroll spills at ws=8.\n",
        "    unsigned int* __restrict__ seq_ctrs, int n16, int slot_stride16, int rank,\n"
        "    int drain, int acq, int pub, long long* gate, unsigned int* gate_ctr,\n"
        "    long long* tlring /* BC_VTP31_ARTL_KPARAM: null unless R4D_AR_TIMELINE */) {\n"
        "  constexpr int NPEERS = WS - 1;\n"
        "  // The reduce holds WS source words live per output word, so the 2x unroll spills at ws=8.\n",
        guard="BC_VTP31_ARTL_KPARAM",
    ),
    _edit(
        "artl-kstart",
        "  // Push, 4x unrolled; loaded words are reused across peers so the input is read once.\n",
        "  // BC_VTP31_ARTL_KSTART: the timeline entry for this launch, by its device sequence number (replay-safe).\n"
        "  long long* tl = nullptr;\n"
        "  if (tlring && b < R4D_ARTL_BLOCKS) {\n"
        "    long long* e = tlring + (size_t)(s % R4D_ARTL_RING) * R4D_ARTL_WORDS;\n"
        "    if (b == 0 && tid == 0) { e[0] = (long long)s; e[1] = n16; e[2] = nb; e[3] = rank; }\n"
        "    tl = e + 4 + 4 * b;\n"
        "    if (tid == 0) tl[0] = (long long)wall_clock64();\n"
        "  }\n"
        "\n"
        "  // Push, 4x unrolled; loaded words are reused across peers so the input is read once.\n",
        guard="BC_VTP31_ARTL_KSTART",
        max_span=2,
    ),
    _edit(
        "artl-khand",
        "  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);\n"
        "  // Every peer's push for this launch has landed and my own retired behind its flag store, so\n",
        "  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub,\n"
        "                       tl ? tl + 1 : nullptr);  /* BC_VTP31_ARTL_KHAND: pushed, then peers seen */\n"
        "  // Every peer's push for this launch has landed and my own retired behind its flag store, so\n",
        guard="BC_VTP31_ARTL_KHAND",
        max_span=3,
    ),
    _edit(
        "artl-kend",
        "    for (; k < end; k += nt) out4[k] = ar_vsum<T, LANES, WS>(src, k);\n"
        "  }\n"
        "}\n",
        "    for (; k < end; k += nt) out4[k] = ar_vsum<T, LANES, WS>(src, k);\n"
        "  }\n"
        "  if (tl) {  /* BC_VTP31_ARTL_KEND: the fold done (tl is uniform over the block) */\n"
        "    __syncthreads();\n"
        "    if (tid == 0) tl[3] = (long long)wall_clock64();\n"
        "  }\n"
        "}\n",
        guard="BC_VTP31_ARTL_KEND",
        max_span=4,
    ),
    _edit(
        "artl-host",
        "// --- host side ---\n"
        "template <int WS>\n"
        "static void ar_wide_launch(",
        "// --- host side ---\n"
        "\n"
        "// BC_VTP31_ARTL_HOST: one timeline ring per all-reduce instance (keyed by its sequence counters), in mapped\n"
        "// pinned host memory, written to R4D_AR_TIMELINE at exit. Null when the variable is unset or empty.\n"
        "namespace {\n"
        "struct ArtlRing { long seq; int rank, ws; long long* host; };\n"
        "std::mutex g_artl_mu;\n"
        "std::vector<ArtlRing>* g_artl = nullptr;\n"
        "const char* artl_dir() {\n"
        "  static const char* const d = std::getenv(\"R4D_AR_TIMELINE\");\n"
        "  return (d && *d) ? d : nullptr;\n"
        "}\n"
        "void artl_dump() {\n"
        "  std::lock_guard<std::mutex> lk(g_artl_mu);\n"
        "  if (!g_artl || !artl_dir()) return;\n"
        "  for (size_t i = 0; i < g_artl->size(); ++i) {\n"
        "    const ArtlRing& r = (*g_artl)[i];\n"
        "    char path[4096];\n"
        "    std::snprintf(path, sizeof path, \"%s/ar-tl-r%d-ws%d-p%d-i%zu.bin\", artl_dir(), r.rank, r.ws,\n"
        "                  (int)getpid(), i);\n"
        "    if (FILE* f = std::fopen(path, \"wb\")) {\n"
        "      std::fwrite(r.host, sizeof(long long), (size_t)R4D_ARTL_RING * R4D_ARTL_WORDS, f);\n"
        "      std::fclose(f);\n"
        "    }\n"
        "  }\n"
        "}\n"
        "long long* artl_ring(long seq_ctrs, int rank, int ws) {\n"
        "  if (!artl_dir()) return nullptr;\n"
        "  std::lock_guard<std::mutex> lk(g_artl_mu);\n"
        "  if (!g_artl) { g_artl = new std::vector<ArtlRing>(); std::atexit(artl_dump); }\n"
        "  for (const ArtlRing& r : *g_artl)\n"
        "    if (r.seq == seq_ctrs) {\n"
        "      void* d = nullptr;\n"
        "      return hipHostGetDevicePointer(&d, r.host, 0) == hipSuccess ? (long long*)d : nullptr;\n"
        "    }\n"
        "  const size_t bytes = (size_t)R4D_ARTL_RING * R4D_ARTL_WORDS * sizeof(long long);\n"
        "  void* h = nullptr;\n"
        "  if (hipHostMalloc(&h, bytes, hipHostMallocMapped) != hipSuccess || !h) return nullptr;\n"
        "  std::memset(h, 0xff, bytes);\n"
        "  g_artl->push_back({ seq_ctrs, rank, ws, (long long*)h });\n"
        "  std::fprintf(stderr, \"[r4d] all-reduce timeline: rank %d of %d, instance %zu -> %s\\n\", rank, ws,\n"
        "               g_artl->size() - 1, artl_dir());\n"
        "  void* d = nullptr;\n"
        "  return hipHostGetDevicePointer(&d, h, 0) == hipSuccess ? (long long*)d : nullptr;\n"
        "}\n"
        "}  // namespace\n"
        "\n"
        "template <int WS>\n"
        "static void ar_wide_launch(",
        guard="BC_VTP31_ARTL_HOST",
        max_span=3,
    ),
    _edit(
        "artl-launch",
        "  hipStream_t st = (hipStream_t)stream_i;\n"
        "#define R4D_AR_WIDE_LAUNCH(T, LN)                                                     \\\n"
        "  r4d_ar_oneshot_wide_exact_kernel<T, LN, WS><<<nb, nt, 0, st>>>(                     \\\n"
        "      (const int4*)inp, peers, (const int4*)my_scratch, (unsigned int*)my_flags,      \\\n"
        "      (int4*)out, (unsigned int*)seq_ctrs, n16, (int)slot_stride16, (int)rank,        \\\n"
        "      (int)drain, (int)acq, (int)pub, (long long*)gate, (unsigned int*)gate_ctr)\n",
        "  hipStream_t st = (hipStream_t)stream_i;\n"
        "  long long* const tlring = artl_ring(seq_ctrs, (int)rank, WS);  /* BC_VTP31_ARTL_LAUNCH */\n"
        "#define R4D_AR_WIDE_LAUNCH(T, LN)                                                     \\\n"
        "  r4d_ar_oneshot_wide_exact_kernel<T, LN, WS><<<nb, nt, 0, st>>>(                     \\\n"
        "      (const int4*)inp, peers, (const int4*)my_scratch, (unsigned int*)my_flags,      \\\n"
        "      (int4*)out, (unsigned int*)seq_ctrs, n16, (int)slot_stride16, (int)rank,        \\\n"
        "      (int)drain, (int)acq, (int)pub, (long long*)gate, (unsigned int*)gate_ctr, tlring)\n",
        guard="BC_VTP31_ARTL_LAUNCH",
    ),
))

PATCHES = (ONESHOT,)
