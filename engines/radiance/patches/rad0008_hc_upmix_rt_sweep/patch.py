"""Radiance 1.9.0: hc_up_mix_wide_kernel PF2 RT sweep, default off (RAD17/RAD18).

Sibling investigation to RAD17's FP8 down-product, targeting the shared BF16 up-mix kernel
instead. Per GPT dev-agent review: a diagnostic hipBLASLt GEMM + separate HIP epilogue
(tools/lab/radiance/hc-upmix-epilogue-diag.hip) measured 395us combined vs the fused kernel's
261us -- the separate epilogue is bandwidth-bound, round-tripping the full fp32 up-product through
global memory that the fused kernel never materializes. GPT's own decision tree therefore rejected
the GEMM-replacement angle and recommended instead sweeping hc_up_mix_wide_kernel's RT (rows per
workgroup, which sets MW waves-per-block) to see whether occupancy/register pressure -- not grid
fill, which is already ample at this shape (n/kWideCH = 2560/16 = 160 column tiles, so even the
smallest legal RT gives thousands of workgroups on a 64-CU part) -- is where the remaining headroom
is.

A first read of the kernel (vendor/radiance/libr4d/r4d_hc_bf16.hip) ruled out GPT's first proposed
lever (sweeping CH, the output tile width): CH is not a software parameter at all -- it is the
gfx1201 WMMA instruction's native 16-wide N dimension, hard-wired into frag()'s `c*16` fragment
offset and the epilogue's `lane & 15` column index. A true CH sweep would need to restructure the
WMMA fragment mapping itself (multiple 16-wide tiles per stream), a materially bigger change than
scoped here. RT, by contrast, is a genuine template parameter already used this way elsewhere in
this same file (the FP8 down-product's own RT fix was this session's single largest win).

This patch adds a runtime-selectable RT for the PF2 up-mix launch (hc_pf2_launch's call site,
~line 3167 as of the 1.9.0 pin), choosing among three legal RT values at MW=1/2/3 (RT must be a
multiple of 16*kWaves=128 for the kernel's own static_asserts to hold): RT=128 (half the current
production value), RT=256 (== kPf2UpRT, today's unconditional default), and RT=384 (one and a half).
The kernel body (hc_up_mix_wide_kernel itself) is completely unchanged -- RT is already one of its
template parameters, so this only changes which instantiation gets called and with what grid.

RT=512 (the originally-planned "double") was tried first and found to be a real, hardware-enforced
dead end, not merely untested: hc_up_mix_wide_kernel's LDS use is sA[RT*kTAP] + sB[NT*kTAP] shorts
(kTAP=68, NT=64) = RT*136 + 8704 bytes, and at RT=512 that is 78336 bytes -- over gfx1201's 65536-byte
workgroup LDS limit, confirmed by a real compile failure ("local memory (78336) exceeds limit
(65536)") when actually compiling that instantiation standalone. RT=384 is the largest legal value
(60928 bytes), so it replaces RT=512 as the "larger" sweep point.

RAD_HC_UPMIX_RT unset, or set to anything other than "128"/"256"/"384", falls back to the exact
original call (RT=kPf2UpRT), so default behaviour is bit-identical to pre-patch. No new kernel body,
no new scratch allocation, no numerics change at any RT value -- this patch can only change which
grid/instantiation a single existing kernel template is launched with.

Benchmark: r4d_selftest --perfhc at RAD_HC_UPMIX_RT=128/256/384, comparing kernel latency and
(where obtainable) occupancy/register metrics, at the production PF2 shape (M=2048, hc=4, n=2560,
lr=320). Not yet benchmarked on real hardware as of this patch revision -- see SUMMARY.md.
"""

from bigcherry.patcher import Edit, FilePatch

PATCH = FilePatch(
    path="libr4d/r4d_hc_bf16.hip",
    description="Sweep hc_up_mix_wide_kernel's PF2 RT via RAD_HC_UPMIX_RT, default off (RAD17/RAD18)",
    edits=(
        Edit(
            id="hc-pf2-upmix-rt-sweep",
            anchor=(
                r"    hipLaunchKernelGGL\(\(hc_up_mix_wide_kernel<4, false, kPf2UpRT>\),\n"
                r"                       dim3\(\(unsigned\)\(n / kWideCH\), \(unsigned\)\(\(rows \+ kPf2UpRT - 1\) / kPf2UpRT\)\),\n"
                r"                       dim3\(kThreads\), 0, \(hipStream_t\)s, \(const void\*\)ubf, lr,\n"
                r"                       \(const unsigned short\*\)dbuf, h, w, \(const float\*\)scale, x, rows, n, lr, hrow,\n"
                r"                       xrow, wadd\);\n"
                r"    if \(hipGetLastError\(\) != hipSuccess\) return RAD_E_DEVICE;\n"
            ),
            mode="replace",
            text=(
                "    /* RAD18 (GPT dev-agent review req_bigcherry_0641281ad540): sweep RT (rows a workgroup,\n"
                "     * selects MW waves-per-block) to test whether occupancy/register pressure -- not grid\n"
                "     * fill, already ample at this shape -- is where this kernel's remaining headroom is.\n"
                "     * Default off: RAD_HC_UPMIX_RT unset or not one of the three legal values below falls\n"
                "     * back to the exact original call, bit-identical to pre-patch. The kernel template\n"
                "     * itself (hc_up_mix_wide_kernel) is completely unchanged; only the instantiation and\n"
                "     * grid dimension picked here differ. */\n"
                "    static const int rad18_upmix_rt = [] {\n"
                "        const char* v = std::getenv(\"RAD_HC_UPMIX_RT\");\n"
                "        if (!v) return kPf2UpRT;\n"
                "        const int rt = atoi(v);\n"
                "        return (rt == 128 || rt == 256 || rt == 384) ? rt : kPf2UpRT;\n"
                "    }();\n"
                "    switch (rad18_upmix_rt) {\n"
                "    case 128:\n"
                "        hipLaunchKernelGGL((hc_up_mix_wide_kernel<4, false, 128>),\n"
                "                           dim3((unsigned)(n / kWideCH), (unsigned)((rows + 128 - 1) / 128)),\n"
                "                           dim3(kThreads), 0, (hipStream_t)s, (const void*)ubf, lr,\n"
                "                           (const unsigned short*)dbuf, h, w, (const float*)scale, x, rows, n,\n"
                "                           lr, hrow, xrow, wadd);\n"
                "        break;\n"
                "    case 384:\n"
                "        hipLaunchKernelGGL((hc_up_mix_wide_kernel<4, false, 384>),\n"
                "                           dim3((unsigned)(n / kWideCH), (unsigned)((rows + 384 - 1) / 384)),\n"
                "                           dim3(kThreads), 0, (hipStream_t)s, (const void*)ubf, lr,\n"
                "                           (const unsigned short*)dbuf, h, w, (const float*)scale, x, rows, n,\n"
                "                           lr, hrow, xrow, wadd);\n"
                "        break;\n"
                "    default:\n"
                "        hipLaunchKernelGGL((hc_up_mix_wide_kernel<4, false, kPf2UpRT>),\n"
                "                           dim3((unsigned)(n / kWideCH), (unsigned)((rows + kPf2UpRT - 1) / kPf2UpRT)),\n"
                "                           dim3(kThreads), 0, (hipStream_t)s, (const void*)ubf, lr,\n"
                "                           (const unsigned short*)dbuf, h, w, (const float*)scale, x, rows, n,\n"
                "                           lr, hrow, xrow, wadd);\n"
                "        break;\n"
                "    }\n"
                "    if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;\n"
            ),
            guard=r"rad18_upmix_rt",
            expect_matches=1,
            rationale="Unique PF2 up-mix launch call site inside hc_pf2_launch; code only, since the engine "
                       "strips comments and string literals before matching anchors and guards. The only place that "
                       "selects hc_up_mix_wide_kernel's RT template parameter and grid for this form",
        ),
    ),
)
