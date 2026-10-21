#!/usr/bin/env python3
"""RAD17: patches a scratch copy of libr4d's r4d_hc_bf16.hip to dump the real W_down operand (raw E4M3 codes +
inline f32 scales, the on-disk row stride dld -- row_bytes(K,128), NOT K, confirmed against a live container
2026-10-21: 10560 vs the naive 10240) to a file during a live hc_read call, gated by BC_HC_WDUMP_ROWS=<n rows>.

Never edits the pinned radiance tree: copy the pinned source elsewhere first, patch that copy, recompile just
r4d_hc_bf16.hip.o and relink libr4d.so (see RAD15/RAD17 notes for the exact compile/relink commands used), then
point RADIANCE_HOME at a radiance_home with the patched libr4d.so swapped in.

usage: python3 hc-wdump-libr4d-patch.py <path to the scratch copy's r4d_hc_bf16.hip>
"""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = '''    const long HN  = (long)hc * n;
    const long ntt = (ncol + 15) / 16;                        /* plane n-tiles */
    uint4* const          plane = (uint4*)wsp;'''
new = '''    const long HN  = (long)hc * n;
    if (const char* wr = std::getenv("BC_HC_WDUMP_ROWS")) {  // RAD17 ground-truth W_down dump, throwaway debug build
        const long nrows = std::atol(wr);
        const long dump_bytes = nrows * dld;
        std::vector<unsigned char> host(dump_bytes);
        hipMemcpy(host.data(), down, dump_bytes, hipMemcpyDeviceToHost);
        FILE* f = std::fopen("/mnt/data/bigcherry-work/wdump.bin", "wb");
        if (f) { std::fwrite(host.data(), 1, dump_bytes, f); std::fclose(f); }
        std::fprintf(stderr, "libr4d: BC_HC_WDUMP_ROWS dumped %ld rows, dld=%ld, f8=%d\\n", nrows, dld, (int)f8);
    }
    const long ntt = (ncol + 15) / 16;                        /* plane n-tiles */
    uint4* const          plane = (uint4*)wsp;'''
assert old in s, "old anchor not found"
s = s.replace(old, new, 1)
if '#include <vector>' not in s:
    s = s.replace('#include <cstdlib>', '#include <cstdlib>\n#include <vector>', 1)
if '#include <cstdio>' not in s:
    s = s.replace('#include <cstdlib>', '#include <cstdlib>\n#include <cstdio>', 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("PATCHED")
