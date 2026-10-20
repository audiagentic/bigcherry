// libbcgfx12: BigCherry's gfx1201 (R9700) kernel rows for radiance (RAD15). Listed ahead of libr4d in --kernels, its rows win
// where their constraints hold (radiance's plugin hierarchy: the first plugin with a matching row supplies it) and libr4d
// serves everything else. The op schemas are libr4d's (this plugin declares none), and each row reuses libr4d's own
// weight layout hooks through its exported symbols, so the stored bytes are exactly libr4d's: no relayout, no format of our
// own. A row's launch may hand a call back to libr4d's kernel.
//
// hc_read (hyper-connections read, Qwen3.8-Flash-Next): decode-sized steps only (M <= 16), E4M3 mixing matrices.
#include "rad_abi.h"

#include <dlfcn.h>
#include <cstdio>
#include <string>

extern "C" int bc_hc_read_decode(const RadArgs* a, RadStream s);   // bc_hc_read.hip

namespace {

/* libr4d's exported entry points, resolved once from the copy the engine already loaded (it sits beside this plugin in
 * the kernels directory). Called only after every plugin is loaded, so libr4d is present. */
struct R4d {
    RadLaunchFn hc_read = nullptr;
    RadScratchFn hc_read_scratch = nullptr;
    RadLayoutFn layout_hc_e4m3 = nullptr;
    RadRelayoutFn relayout_hc_e4m3 = nullptr;
    RadUnrelayoutFn unrelayout_hc_e4m3 = nullptr;
    RadShapeFn shape_hc_read = nullptr;
    bool ok = false;
};

const R4d& r4d() {
    static const R4d r = [] {
        R4d x;
        Dl_info di{};
        std::string dir;
        if (dladdr((const void*)&r4d, &di) && di.dli_fname) {
            dir = di.dli_fname;
            const size_t s = dir.rfind('/');
            dir = s == std::string::npos ? std::string() : dir.substr(0, s + 1);
        }
        void* h = dlopen((dir + "libr4d.so").c_str(), RTLD_NOW | RTLD_NOLOAD);
        if (!h) h = dlopen((dir + "libr4d.so").c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) {
            std::fprintf(stderr, "libbcgfx12: libr4d.so not found beside the plugin: %s\n", dlerror());
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
        return x;
    }();
    return r;
}

int hc_read_launch(const RadArgs* a, RadStream s) {
    const int rc = bc_hc_read_decode(a, s);
    if (rc != RAD_E_UNSUPPORTED) return rc;
    return r4d().ok ? r4d().hc_read(a, s) : RAD_E_UNSUPPORTED;   // a geometry the decode kernel declines: libr4d's
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

/* libr4d's hc_read_e4m3 constraints, narrowed to decode-sized steps. */
const RadConstraint kHcReadDecode[] = {
    RAD_CGE("M", 1), RAD_CLE("M", 16), RAD_CDIV("n", 512), RAD_CLE("n", 7552), RAD_CGE("hc", 2), RAD_CLE("hc", 4),
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
