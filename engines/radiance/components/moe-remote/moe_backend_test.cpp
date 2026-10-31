// Offline test of the expert-backend contract (moe_backend.h): sizing, capability probe and refusal, for Flash-Next's W4 experts and
// Qwen3.6-35B-A3B's FP8 block-scaled experts. No GPU, no container.
#include "moe_backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "r3_moe_backend_test: FAILED %s\n", what); std::exit(1); }
}

using namespace r3::moe_backend;

// A card type's backend is data (engines/radiance/archs.toml -> R3_MOE_WORKER_FORMATS), parsed, never looked up by architecture.
static Backend backend(const char* name, const char* formats) {
    Backend b{};
    std::strncpy(b.name, name, sizeof(b.name) - 1);
    check(parse_formats(formats, &b.formats), "formats parse");
    return b;
}

int main() {
    // Flash-Next: hidden 2560, 512 experts, top-k 10, expert width 640 (gate+up fused to 1280 rows), read from the container 2026-10-19.
    const MoeShape flash{49, 512, 2560, 10, 640, 1, 64};
    const MoeWeightDesc w4{MOE_FMT_W4NL64, MOE_ACT_FP8_E4M3_GROUP, MOE_SCALE_E4M3, 0, 64, 0};
    const MoeExpertDesc fe = expert_for(flash, w4);
    check(fe.gu_rows == 1280 && fe.gu_cols == 2560 && fe.dn_rows == 2560 && fe.dn_cols == 640, "expert geometry from shape");
    // codes rows*cols/2, scales rows*(cols/64) one byte each: gate/up 1638400 + 51200, down 819200 + 25600
    check(expert_bytes(fe) == 1638400 + 51200 + 819200 + 25600, "W4 expert bytes match the helper's relayout geometry");

    // Qwen3.6-35B-A3B FP8 (41 layers with the MTP layer): hidden 2048, 256 experts, top-k 8, expert width 512, 128x128 bf16 scales.
    const MoeShape qwen{41, 256, 2048, 8, 512, 1, 64};
    const MoeWeightDesc fp8{MOE_FMT_FP8_E4M3_BLOCK, MOE_ACT_BF16, MOE_SCALE_BF16, 128, 128, 0};
    const MoeExpertDesc qe = expert_for(qwen, fp8);
    // gate/up 1024x2048: 2097152 codes + (8*16)*2 scale bytes; down 2048x512: 1048576 codes + (16*4)*2
    check(expert_bytes(qe) == 2097152 + 256 + 1048576 + 128, "FP8 block-scaled expert bytes");

    // scale tiles round up at the edge: 130 rows x 130 cols needs 2x2 tiles
    check(matrix_bytes(fp8, 130, 130) == 130 * 130 + 4 * 2, "FP8 scale tiles round up at the edge");
    // other scale dtypes change only the scale bytes
    MoeWeightDesc f32 = fp8; f32.scale_dtype = MOE_SCALE_F32;
    check(matrix_bytes(f32, 128, 128) == 128 * 128 + 4, "f32 scale is 4 bytes per tile");

    // geometry that does not fit the format is refused, not rounded
    check(matrix_bytes(w4, 1024, 100) == MOE_BAD_GEOMETRY, "W4 needs the column count to be a multiple of 64");
    check(matrix_bytes(fp8, 128, 0) == MOE_BAD_GEOMETRY, "an empty matrix is refused");
    MoeWeightDesc noscale = fp8; noscale.scale_dtype = MOE_SCALE_NONE;
    check(matrix_bytes(noscale, 128, 128) == MOE_BAD_GEOMETRY, "FP8 block-scaled needs a scale dtype");
    check(matrix_bytes(MoeWeightDesc{99, 0, 0, 0, 0, 0}, 8, 8) == MOE_UNSUPPORTED_FORMAT, "an unknown format is refused");
    check(matrix_bytes(fp8, UINT64_MAX / 2, 4) == MOE_OVERFLOW, "an oversized matrix is an overflow, not a wrapped size");

    // formats are a list of names; a backend is whatever the list says
    uint32_t mask = 0;
    check(parse_formats("w4nl64", &mask) && mask == (1u << MOE_FMT_W4NL64), "one format");
    check(parse_formats("w4nl64,fp8_e4m3_block", &mask) && mask == ((1u << MOE_FMT_W4NL64) | (1u << MOE_FMT_FP8_E4M3_BLOCK)), "two formats");
    mask = 7;
    check(!parse_formats("", &mask) && !parse_formats(nullptr, &mask) && !parse_formats("w4nl64,", &mask) && !parse_formats("w4", &mask) &&
          !parse_formats("w4nl64,w4nl64", &mask) && !parse_formats("fp8_e4m3_block;w4nl64", &mask) && mask == 7,
          "empty, unknown, truncated, repeated and malformed lists are refused and leave the mask alone");
    check(std::strcmp(format_name(MOE_FMT_W4NL64), "w4nl64") == 0 && std::strcmp(format_name(MOE_FMT_FP8_E4M3_BLOCK), "fp8_e4m3_block") == 0, "format names");

    // probe: a backend that runs W4 and FP8, one that runs W4 only; the answer comes before any expert is touched
    const Backend both = backend("card-a", "w4nl64,fp8_e4m3_block");
    check(probe(both, flash, w4) == MOE_OK, "a W4 backend accepts Flash-Next W4");
    check(probe(both, qwen, fp8) == MOE_OK, "a W4+FP8 backend accepts FP8 block-scaled (radiance's fp8a8 grouped GEMM)");
    MoeWeightDesc nolayout{99, 0, 0, 0, 0, 0};
    check(probe(both, qwen, nolayout) == MOE_UNSUPPORTED_FORMAT, "an unknown format is refused");
    MoeShape bad = flash; bad.top_k = flash.experts + 1;
    check(probe(both, bad, w4) == MOE_BAD_GEOMETRY, "top-k above the expert count is refused");
    bad = flash; bad.hidden = 0;
    check(probe(both, bad, w4) == MOE_BAD_GEOMETRY, "a zero dimension is refused");
    const Backend w4_only = backend("card-b", "w4nl64");
    check(probe(w4_only, flash, w4) == MOE_OK, "a W4-only backend accepts Flash-Next W4");
    check(probe(w4_only, qwen, fp8) == MOE_UNSUPPORTED_FORMAT, "a W4-only backend refuses the Qwen3.6 FP8 experts");
    check(std::strcmp(w4_only.name, "card-b") == 0, "the backend keeps its card's name");

    std::puts("r3_moe_backend_test: ok");
    return 0;
}
