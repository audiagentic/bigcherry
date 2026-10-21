// bc_hc_prefill.hip -- hc_read for prefill-sized steps on gfx1201 (RAD15). Stage 1: declines every call, so the row hands
// each call to libr4d's PF2/wide kernels; this proves the plugin's selection, schema and layout plumbing before the
// kernel is replaced (same two-stage approach as bc_hc_read.hip's decode row and bc_moe_gemv.hip).
//
// Target: the down-product step of libr4d's PF2 form, which a measured prefill map (RAD15) shows at ~41-44 TOPS (about
// 22% of bf16 WMMA peak) and which scales worse than linearly from 2048 to 4096 rows for reasons not yet root-caused
// (a K-split theory was tested and refuted by direct measurement; see RAD15's change log). An FP8 WMMA redesign of the
// down and up products, with blockwise-quantised bf16 activations against the stored E4M3 mixing matrices, is the next
// stage once this row's selection is proven.
#include "rad_abi.h"

extern "C" int bc_hc_read_prefill(const RadArgs* a, RadStream s) {
    (void)a;
    (void)s;
    return RAD_E_UNSUPPORTED;
}
