// HI82 eligibility-gate evidence for RD50 (1221_rd50_gdn_chunked_recurrence)
// and RD21 (1208_rd21, same GGML_CUDA_CC_IS_RDNA3_5 gate).
//
// Compiles the REAL macro text (verbatim, byte-for-byte copy of lines
// 50-90 of vendor/llama.cpp/ggml/src/ggml-cuda/common.cuh at the pin this
// session validated against) with the actual vendored HIP clang, and
// evaluates GGML_CUDA_CC_IS_RDNA3_5(cc) for cc computed the same way
// ggml_cuda.cu's device-probe path computes it (info.devices[id].cc =
// GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100, confirmed via
// ggml_cuda_parse_id's gfxNNNN hex-parse producing 0xNNNN for the
// hundreds+tens+ones triplet) -- i.e. cc = OFFSET_AMD + 0xNNNN for a
// gfxNNNN part.
#include <cstdio>
#include "macros.h"

static int cc_for_gfx(unsigned gfx_hex) {
    return GGML_CUDA_CC_OFFSET_AMD + (int)gfx_hex;
}

int main() {
    struct { const char *name; unsigned gfx_hex; } archs[] = {
        {"gfx1100", 0x1100}, // local RX 7900 GRE -- this session's validation GPU
        {"gfx1201", 0x1201}, // Brutus R9700 (device 2)
        {"gfx1151", 0x1151}, // AMD AI Max 395 / Strix Halo -- RD50's actual target
    };
    for (auto &a : archs) {
        int cc = cc_for_gfx(a.gfx_hex);
        printf("%s cc=%d is_rdna3_5=%d\n", a.name, cc, GGML_CUDA_CC_IS_RDNA3_5(cc) ? 1 : 0);
    }
    return 0;
}
