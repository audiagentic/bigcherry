// libbciqr: BigCherry's owned IQR kernel rows, listed ahead of libr3 and libr4d in --kernels (RCS23). A row here wins
// where its constraints hold (radiance's plugin hierarchy: the first plugin with a matching row supplies it) and the base
// library serves everything else. The op schemas are the base's (this plugin declares none), and each row reuses the base's
// own weight layout, relayout and shape hooks through its exported symbols, so the stored bytes are exactly the base's: no
// relayout, no format of our own. A launch the owned kernel declines is handed back to the base's.
//
// One plugin, both architectures: the device code is built for gfx1100 and gfx1201 and the runtime picks the code object of
// the card the launch runs on. The base library whose hooks are used is libr4d.so, else libr3.so (the engine loads them;
// the hooks are host code of r4d_iqr_layout.cpp and r4d_shapes.cpp, the same in both); a launch the owned kernel declines goes
// to the launch of the base that matches the card (libr4d for RDNA4, libr3 for RDNA3).
//
// embed_lookup_iqr (bc_iqr_embed.hip): the embedding gather over an IQR table.
#include "rad_abi.h"
#include "rad_abi_layout.h"

#include <hip/hip_runtime_api.h>

#include <dlfcn.h>
#include <link.h>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

extern "C" int bc_rad_embed_lookup_iqr(const RadArgs* a, RadStream s);   // bc_iqr_embed.hip

/* The dispatch log r4d_emit.h's launch helper reads (libr4d's r4d_record_dispatches fills its own copy; the engine never calls
 * it, only libr4d's selftest does, to see which form ran). This plugin keeps the log off. */
__attribute__((visibility("hidden"))) const void** r4d_dispatch_log = nullptr;
__attribute__((visibility("hidden"))) int           r4d_dispatch_cap = 0;
__attribute__((visibility("hidden"))) int           r4d_dispatch_n   = 0;

namespace {

/* One base library's exported entry points. */
struct Lib {
    RadLaunchFn embed_lookup_iqr = nullptr;
    RadLayoutFn layout_iqr_embed = nullptr;
    RadRelayoutFn relayout_iqr_embed = nullptr;
    RadShapeFn shape_embed_lookup = nullptr;
    bool hooks() const { return layout_iqr_embed && relayout_iqr_embed && shape_embed_lookup; }
};

struct Found {
    std::string r4d, r3;
};

int find_loaded(struct dl_phdr_info* info, size_t, void* out) {
    const char* n = info->dlpi_name;
    if (!n || !*n) return 0;
    const char* b = std::strrchr(n, '/');
    const char* base = b ? b + 1 : n;
    Found* f = static_cast<Found*>(out);
    if (!std::strcmp(base, "libr4d.so")) f->r4d = n;
    else if (!std::strcmp(base, "libr3.so")) f->r3 = n;
    return 0;
}

/* The entry points of the copy of `file` the engine already loaded (`loaded` is its path, from the loaded objects), else of the
 * one beside this plugin. Empty (all null) when there is none. */
Lib open_lib(const std::string& loaded, const char* file) {
    Lib x;
    std::string path = loaded;
    if (path.empty()) {
        Dl_info di{};
        if (dladdr((const void*)&find_loaded, &di) && di.dli_fname) {
            path = di.dli_fname;
            const size_t s = path.rfind('/');
            path = s == std::string::npos ? std::string() : path.substr(0, s + 1);
        }
        path += file;
    }
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_NOLOAD);
    if (!h && loaded.empty()) h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);   // not loaded by the engine: only a test setup gets here
    if (!h) return x;
    x.embed_lookup_iqr = (RadLaunchFn)dlsym(h, "r4d_rad_embed_lookup_iqr");
    x.layout_iqr_embed = (RadLayoutFn)dlsym(h, "r4d_rad_layout_iqr_embed");
    x.relayout_iqr_embed = (RadRelayoutFn)dlsym(h, "r4d_rad_relayout_iqr_embed");
    x.shape_embed_lookup = (RadShapeFn)dlsym(h, "r4d_shape_embed_lookup");
    return x;
}

struct Bases {
    Lib r4d, r3;
    bool ok = false;
};

/* The base libraries. A failed resolution is not kept: the next call tries again (the hooks are called only after every plugin
 * is loaded, so it normally succeeds the first time). */
const Bases& bases() {
    static std::mutex m;
    static Bases cached;
    std::lock_guard<std::mutex> lock(m);
    if (cached.ok) return cached;
    Found f;
    dl_iterate_phdr(find_loaded, &f);
    Bases b;
    b.r4d = open_lib(f.r4d, "libr4d.so");
    b.r3 = open_lib(f.r3, "libr3.so");
    b.ok = b.r4d.hooks() || b.r3.hooks();
    if (!b.ok) {
        static bool told = false;
        if (!told) {
            told = true;
            std::fprintf(stderr, "libbciqr: no base kernel library (libr4d.so or libr3.so) with the embed_lookup_iqr hooks is loaded\n");
        }
        static Bases failed;
        failed = b;
        return failed;
    }
    cached = b;
    return cached;
}

/* The layout, relayout and shape hooks are host code of r4d_iqr_layout.cpp and r4d_shapes.cpp, the same in both libraries. */
const Lib& hook_lib() {
    const Bases& b = bases();
    return b.r4d.hooks() ? b.r4d : b.r3;
}

/* The base's launch for the card this call runs on: libr4d's for an RDNA4 card, libr3's for an RDNA3 one (the other library's
 * device code would not run there). */
RadLaunchFn base_launch() {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return nullptr;
    static std::mutex m;
    static std::map<int, bool> rdna4;
    bool is4;
    {
        std::lock_guard<std::mutex> lock(m);
        auto it = rdna4.find(dev);
        if (it == rdna4.end()) {
            hipDeviceProp_t p;
            if (hipGetDeviceProperties(&p, dev) != hipSuccess) return nullptr;
            it = rdna4.emplace(dev, std::strncmp(p.gcnArchName, "gfx12", 5) == 0).first;
        }
        is4 = it->second;
    }
    const Bases& b = bases();
    return is4 ? b.r4d.embed_lookup_iqr : b.r3.embed_lookup_iqr;
}

int embed_launch(const RadArgs* a, RadStream s) {
    const int rc = bc_rad_embed_lookup_iqr(a, s);
    if (rc != RAD_E_UNSUPPORTED) return rc;
    const RadLaunchFn fallback = base_launch();   // a form the owned kernel declines
    return fallback ? fallback(a, s) : RAD_E_UNSUPPORTED;
}
int layout_embed(const RadParam* p, int n_p, int op, const RadEncoding* e, const int* sel, const RadTensor* pl, int n_pl,
                 RadLayout* out) {
    /* Without the base's hook the row cannot say what it reads: refuse (RAD_E_DTYPE is the loader's "not my encoding"), never
     * accept, or it would claim every embedding table of the op. */
    const Lib& l = hook_lib();
    return l.hooks() ? l.layout_iqr_embed(p, n_p, op, e, sel, pl, n_pl, out) : RAD_E_DTYPE;
}
int relayout_embed(const RadParam* p, int n_p, int op, const RadEncoding* e, const int* sel, const RadTensor* pl, int n_pl,
                   void* dst, int64_t dst_bytes) {
    const Lib& l = hook_lib();
    return l.hooks() ? l.relayout_iqr_embed(p, n_p, op, e, sel, pl, n_pl, dst, dst_bytes) : RAD_E_UNSUPPORTED;
}
int shape_embed(const RadParam* p, int n_p, int op, RadOpdDesc* out) {
    const Lib& l = hook_lib();
    return l.hooks() ? l.shape_embed_lookup(p, n_p, op, out) : RAD_E_UNSUPPORTED;
}

/* The base's embed_lookup_iqr constraints. */
const RadConstraint kEmbedBf16[] = { RAD_CIN("dtype", "bf16") };

const RadKernelInfo kRows[] = {
    { "bc_embed_lookup_iqr", "embed_lookup", "vocab",
      "embedding gather over an IQR table kept at its codes' width (RCS23 S3a): libr4d's computation and stored form, the "
      "128-wide Hadamard of a rotated table by wave shuffles where the card has them (gfx1100)",
      "n_embd a multiple of 128; a negative or out-of-range id writes a zero row; otherwise as libr4d's embed_lookup_iqr",
      "i32 tokens, an IQR table (r4d_iqr.h), bf16 output; each value the form's int8 times its row's scale, the rotation "
      "undone in f32 and rounded once, bit for bit libr4d's",
      RAD_DOMAIN_DEVICE, 60,
      kEmbedBf16, (int)(sizeof(kEmbedBf16) / sizeof(kEmbedBf16[0])),
      nullptr, 0, nullptr,
      nullptr, nullptr, embed_launch, nullptr, layout_embed, relayout_embed, shape_embed, nullptr,
      nullptr },
};

}  // namespace

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT  /* the loader refuses a plugin whose struct layout differs from the engine's */
extern "C" const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = {
        RAD_PLUGIN_KERNEL, "libbciqr", "0.0.1", "BigCherry owned IQR kernel rows ahead of libr3 and libr4d (RCS23)",
        "gfx1100 gfx1201",
    };
    return &info;
}
extern "C" int rad_kernel_schema_count(void) { return 0; }
extern "C" const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }
extern "C" int rad_kernel_count(void) { return (int)(sizeof(kRows) / sizeof(kRows[0])); }
extern "C" const RadKernelInfo* rad_kernel_at(int i) { return i >= 0 && i < rad_kernel_count() ? &kRows[i] : nullptr; }
extern "C" int rad_kernel_concurrent(int) { return 0; }
