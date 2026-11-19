"""Select the PF2 wide up-mix tile. PF3 retains its recording path and fixed tile."""

import re

from bigcherry.patcher import Edit, FilePatch


_WIDE_LAUNCH = """        (void)R4D_EMIT(s, (hc_up_mix_wide_kernel<4, false, kPf2UpRT>), gu, dim3(kThreads), 0,
                       (const void*)ubf, lr, (const unsigned short*)dbuf, h, w,
                       (const float*)scale, x, rows, n, lr, hrow, xrow, wadd);"""


def _launch(rt: int) -> str:
    return f"""            (void)R4D_EMIT(s, (hc_up_mix_wide_kernel<4, false, {rt}>), grid, dim3(kThreads), 0,
                           (const void*)ubf, lr, (const unsigned short*)dbuf, h, w,
                           (const float*)scale, x, rows, n, lr, hrow, xrow, wadd);"""


PATCH = FilePatch(
    path="libr4d/r4d_hc_bf16.hip",
    description="Choose the PF2 wide up-mix tile while retaining upstream dispatch recording",
    edits=(
        Edit(
            id="hc-pf2-upmix-rt-sweep",
            anchor=re.escape("    else\n" + _WIDE_LAUNCH),
            mode="replace",
            text="""    else {
        static const int rad18_upmix_rt = [] {
            const char* v = std::getenv("RAD_HC_UPMIX_RT");
            const int rt = v ? std::atoi(v) : 128;
            return (rt == 256 || rt == 384) ? rt : 128;
        }();
        const dim3 grid((unsigned)(n / kWideCH),
                        (unsigned)((rows + rad18_upmix_rt - 1) / rad18_upmix_rt));
        if (rad18_upmix_rt == 128)
""" + _launch(128) + """
        else if (rad18_upmix_rt == 384)
""" + _launch(384) + """
        else
""" + _launch(256) + """
    }""",
            guard=r"rad18_upmix_rt",
            expect_matches=1,
            rationale="Only the wide PF2 branch of hc_pf2_launch takes this kernel and argument list.",
        ),
    ),
)
