// libbcgfx12: BigCherry's gfx1201 (R9700) kernel rows for radiance (RAD15). Listed ahead of libr4d in --kernels, its rows win
// where their constraints hold (radiance's plugin hierarchy: the first plugin with a matching row supplies it) and libr4d
// serves everything else. The op schemas are libr4d's (this plugin declares none), and each row reuses libr4d's own
// weight layout hooks through its exported symbols, so the stored bytes are exactly libr4d's: no relayout, no format of our
// own. A row's launch may hand a call back to libr4d's kernel.
//
// hc_read (hyper-connections read, Qwen3.8-Flash-Next): decode-sized steps only (M <= 16), E4M3 mixing matrices.
// moe_gemm_q (fp8a8 grouped expert GEMM, the 35B): decode-sized steps only (M <= 16), the packed-f16 GEMV.
#include "rad_abi.h"

#include <dlfcn.h>
#include <link.h>
#include <cstdio>
#include <cstring>
#include <string>

extern "C" int bc_hc_read_decode(const RadArgs* a, RadStream s);        // bc_hc_read.hip
extern "C" int bc_hc_read_prefill(const RadArgs* a, RadStream s);       // bc_hc_prefill.hip
extern "C" int bc_moe_gemm_fp8_decode(const RadArgs* a, RadStream s);   // bc_moe_gemv.hip

namespace {

/* libr4d's exported entry points, resolved once from the copy the engine already loaded: found among the loaded objects by
 * file name (the engine may load plugins through a resolved path, and libr4d carries no soname), else beside this plugin.
 * Called only after every plugin is loaded, so libr4d is present. */
struct R4d {
    RadLaunchFn hc_read = nullptr;
    RadScratchFn hc_read_scratch = nullptr;
    RadLayoutFn layout_hc_e4m3 = nullptr;
    RadRelayoutFn relayout_hc_e4m3 = nullptr;
    RadUnrelayoutFn unrelayout_hc_e4m3 = nullptr;
    RadShapeFn shape_hc_read = nullptr;
    RadLaunchFn moe_gemm_q = nullptr;
    RadScratchFn moe_gemm_q_scratch = nullptr;
    RadLayoutFn layout_moe_fp8 = nullptr;
    RadShapeFn shape_moe_gemm_q = nullptr;
    bool ok = false;
    bool moe_ok = false;
};

int find_loaded_r4d(struct dl_phdr_info* info, size_t, void* out) {
    const char* n = info->dlpi_name;
    if (!n || !*n) return 0;
    const char* b = std::strrchr(n, '/');
    if (std::strcmp(b ? b + 1 : n, "libr4d.so")) return 0;
    *static_cast<std::string*>(out) = n;
    return 1;
}

const R4d& r4d() {
    static const R4d r = [] {
        R4d x;
        std::string path;
        dl_iterate_phdr(find_loaded_r4d, &path);
        if (path.empty()) {
            Dl_info di{};
            if (dladdr((const void*)&find_loaded_r4d, &di) && di.dli_fname) {
                path = di.dli_fname;
                const size_t s = path.rfind('/');
                path = s == std::string::npos ? std::string() : path.substr(0, s + 1);
            }
            path += "libr4d.so";
        }
        void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_NOLOAD);
        if (!h) h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) {
            std::fprintf(stderr, "libbcgfx12: libr4d.so not loaded and not found beside the plugin: %s\n", dlerror());
            return x;
        }
        x.hc_read = (RadLaunchFn)dlsym(h, "r4d_rad_hc_read");
        x.hc_read_scratch = (RadScratchFn)dlsym(h, "r4d_rad_hc_read_scratch");
        x.layout_hc_e4m3 = (RadLayoutFn)dlsym(h, "r4d_rad_layout_hc_e4m3");
        x.relayout_hc_e4m3 = (RadRelayoutFn)dlsym(h, "r4d_rad_relayout_hc_e4m3");
        x.unrelayout_hc_e4m3 = (RadUnrelayoutFn)dlsym(h, "r4d_rad_unrelayout_hc_e4m3");
        x.shape_hc_read = (RadShapeFn)dlsym(h, "r4d_shape_hc_read");
        x.ok = x.hc_read && x.hc_read_scratch && x.layout_hc_e4m3 && x.relayout_hc_e4m3 && x.shape_hc_read;
        if (!x.ok) std::fprintf(stderr, "libbcgfx12: libr4d does not export the hc_read entry points this plugin needs\n");
        x.moe_gemm_q = (RadLaunchFn)dlsym(h, "r4d_rad_moe_gemm_q");
        x.moe_gemm_q_scratch = (RadScratchFn)dlsym(h, "r4d_rad_moe_gemm_q_scratch");
        x.layout_moe_fp8 = (RadLayoutFn)dlsym(h, "r4d_rad_layout_moe_fp8");
        x.shape_moe_gemm_q = (RadShapeFn)dlsym(h, "r4d_shape_moe_gemm_q");
        x.moe_ok = x.moe_gemm_q && x.moe_gemm_q_scratch && x.layout_moe_fp8 && x.shape_moe_gemm_q;
        if (!x.moe_ok) std::fprintf(stderr, "libbcgfx12: libr4d does not export the moe_gemm_q entry points this plugin needs\n");
        return x;
    }();
    return r;
}

int hc_read_launch(const RadArgs* a, RadStream s) {
    const int rc = bc_hc_read_decode(a, s);
    if (rc != RAD_E_UNSUPPORTED) return rc;
    return r4d().ok ? r4d().hc_read(a, s) : RAD_E_UNSUPPORTED;   // a geometry the decode kernel declines: libr4d's
}
int hc_read_prefill_launch(const RadArgs* a, RadStream s) {
    const int rc = bc_hc_read_prefill(a, s);
    if (rc != RAD_E_UNSUPPORTED) return rc;
    return r4d().ok ? r4d().hc_read(a, s) : RAD_E_UNSUPPORTED;   // a geometry the prefill kernel declines: libr4d's
}
int64_t hc_read_scratch(const RadArgs* a) {
    return r4d().ok ? r4d().hc_read_scratch(a) : -1;            // our kernel needs no more scratch than libr4d's
}
int layout_hc(const RadParam* p, int n_p, int op, const RadEncoding* e, const int* sel, const RadTensor* pl, int n_pl,
              RadLayout* out) {
    return r4d().ok ? r4d().layout_hc_e4m3(p, n_p, op, e, sel, pl, n_pl, out) : RAD_E_UNSUPPORTED;
}
int relayout_hc(const RadParam* p, int n_p, int op, const RadEncoding* e, const int* sel, const RadTensor* pl, int n_pl,
                void* dst, int64_t dst_bytes) {
    return r4d().ok ? r4d().relayout_hc_e4m3(p, n_p, op, e, sel, pl, n_pl, dst, dst_bytes) : RAD_E_UNSUPPORTED;
}
int unrelayout_hc(const RadParam* p, int n_p, int op, const RadEncoding* e, const int* sel, const void* src,
                  int64_t src_bytes, const RadTensor* pl, int n_pl) {
    return r4d().ok && r4d().unrelayout_hc_e4m3 ? r4d().unrelayout_hc_e4m3(p, n_p, op, e, sel, src, src_bytes, pl, n_pl)
                                                : RAD_E_UNSUPPORTED;
}
int shape_hc(const RadParam* p, int n_p, int op, RadOpdDesc* out) {
    return r4d().ok ? r4d().shape_hc_read(p, n_p, op, out) : RAD_E_UNSUPPORTED;
}

int moe_gemm_q_launch(const RadArgs* a, RadStream s) {
    const int rc = bc_moe_gemm_fp8_decode(a, s);
    if (rc != RAD_E_UNSUPPORTED) return rc;
    return r4d().moe_ok ? r4d().moe_gemm_q(a, s) : RAD_E_UNSUPPORTED;
}
int64_t moe_gemm_q_scratch(const RadArgs* a) {
    return r4d().moe_ok ? r4d().moe_gemm_q_scratch(a) : -1;   // libr4d's, for the calls handed back to it
}
int layout_moe_fp8(const RadParam* p, int n_p, int op, const RadEncoding* e, const int* sel, const RadTensor* pl,
                   int n_pl, RadLayout* out) {
    return r4d().moe_ok ? r4d().layout_moe_fp8(p, n_p, op, e, sel, pl, n_pl, out) : RAD_E_UNSUPPORTED;
}
int shape_moe_gemm_q(const RadParam* p, int n_p, int op, RadOpdDesc* out) {
    return r4d().moe_ok ? r4d().shape_moe_gemm_q(p, n_p, op, out) : RAD_E_UNSUPPORTED;
}

/* libr4d's moe_gemm_fp8a8 constraints, narrowed to decode-sized steps. */
const RadConstraint kMoeGemmQDecode[] = {
    RAD_CGE("M", 1), RAD_CLE("M", 16), RAD_CDIV("N", 64), RAD_CDIV("K", 128), RAD_CGE("n_expert", 1),
    RAD_CLE("n_expert", 65536), RAD_CGE("top_k", 1), RAD_CEQ("group", 128), RAD_CIN("dtype", "fp8a8"),
};

/* libr4d's hc_read_e4m3 constraints, narrowed to decode-sized steps. */
const RadConstraint kHcReadDecode[] = {
    RAD_CGE("M", 1), RAD_CLE("M", 16), RAD_CDIV("n", 512), RAD_CLE("n", 7552), RAD_CGE("hc", 2), RAD_CLE("hc", 4),
    RAD_CDIV("lowrank", 64), RAD_CLE("lowrank", 512), RAD_CLE("group", 128), RAD_CLE("inject", 1),
    RAD_CIN("dtype", "bf16"), RAD_CIN("mix", "e4m3"),
};
const RadConstraint kHcReadPrefill[] = {
    RAD_CGE("M", 17), RAD_CDIV("n", 512), RAD_CLE("n", 7552), RAD_CGE("hc", 2), RAD_CLE("hc", 4),
    RAD_CDIV("lowrank", 64), RAD_CLE("lowrank", 512), RAD_CLE("group", 128), RAD_CLE("inject", 1),
    RAD_CIN("dtype", "bf16"), RAD_CIN("mix", "e4m3"),
};

const RadKernelInfo kRows[] = {
    { "bc_hc_read_e4m3_decode", "hc_read", "norm",
      "hc_read at decode-sized steps (RAD15): libr4d's computation and stored form, with the mixing matrices streamed once "
      "for all rows of the step by more workgroups than libr4d's per-row grid",
      "M <= 16; otherwise as libr4d's hc_read_e4m3",
      "bf16 stream, E4M3 mixing matrices with f32 scales, f32 gain, bf16 out, E4M3 + f32 scale for the twin",
      RAD_DOMAIN_DEVICE, 60,
      kHcReadDecode, (int)(sizeof(kHcReadDecode) / sizeof(kHcReadDecode[0])),
      nullptr, 0, nullptr,
      nullptr, nullptr, hc_read_launch, hc_read_scratch, layout_hc, relayout_hc, shape_hc, unrelayout_hc,
      nullptr },
    { "bc_moe_gemm_fp8a8_decode", "moe_gemm_q", "moe",
      "moe_gemm_q at decode-sized steps (RAD15): one block a (routed row pair, 64 outputs), each wave an output's dot "
      "product on packed f16, instead of libr4d's fp8 WMMA tile masked down to a run's one or two rows",
      "M <= 16, K <= 2560, token-order activation (gate/up); otherwise as libr4d's moe_gemm_fp8a8",
      "E4M3 activation with an f32 scale per 128 of K, E4M3 weight with a bf16 scale per 128x128 tile, bf16 out",
      RAD_DOMAIN_DEVICE, 60,
      kMoeGemmQDecode, (int)(sizeof(kMoeGemmQDecode) / sizeof(kMoeGemmQDecode[0])),
      nullptr, 0, nullptr,
      nullptr, nullptr, moe_gemm_q_launch, moe_gemm_q_scratch, layout_moe_fp8, nullptr, shape_moe_gemm_q, nullptr,
      nullptr },
    { "bc_hc_read_e4m3_prefill", "hc_read", "norm",
      "hc_read at prefill-sized steps (RAD15 stage 1): stub that delegates every call to libr4d while the plugin's "
      "selection and layout plumbing for M > 16 is proven; a WMMA kernel replaces it next",
      "M >= 17; otherwise as libr4d's hc_read_e4m3",
      "bf16 stream, E4M3 mixing matrices with f32 scales, f32 gain, bf16 out, E4M3 + f32 scale for the twin",
      RAD_DOMAIN_DEVICE, 60,
      kHcReadPrefill, (int)(sizeof(kHcReadPrefill) / sizeof(kHcReadPrefill[0])),
      nullptr, 0, nullptr,
      nullptr, nullptr, hc_read_prefill_launch, hc_read_scratch, layout_hc, relayout_hc, shape_hc, unrelayout_hc,
      nullptr },
};

}  // namespace

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = {
        RAD_PLUGIN_KERNEL, "libbcgfx12", "0.0.1", "BigCherry gfx1201 kernel rows ahead of libr4d (RAD15)", "gfx1201",
    };
    return &info;
}
extern "C" int rad_kernel_schema_count(void) { return 0; }
extern "C" const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }
extern "C" int rad_kernel_count(void) { return (int)(sizeof(kRows) / sizeof(kRows[0])); }
extern "C" const RadKernelInfo* rad_kernel_at(int i) { return i >= 0 && i < rad_kernel_count() ? &kRows[i] : nullptr; }
extern "C" int rad_kernel_concurrent(int) { return 0; }
