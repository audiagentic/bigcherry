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
// logits_gemm_vocab_iqr (bc_iqr_vocab.hip): the vocabulary-weighted IQR head.
// gemm_iqr_nt_m64, gemm_iqr_gated, gemm_iqr_tiled (bc_iqr_gemm*.hip): the IQR GEMM family, the fused single-band prefill tile included.
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
extern "C" int bc_rad_logits_gemm_vocab(const RadArgs* a, RadStream s);   // bc_iqr_vocab.hip
extern "C" int64_t bc_rad_logits_gemm_vocab_scratch(const RadArgs* a);    // bc_iqr_vocab.hip
extern "C" int bc_rad_gemm_iqr_m64(const RadArgs* a, RadStream s);        // bc_iqr_gemm*.hip
extern "C" int64_t bc_rad_gemm_iqr_m64_scratch(const RadArgs* a);
extern "C" int bc_rad_gemm_iqr_gated(const RadArgs* a, RadStream s);
extern "C" int64_t bc_rad_gemm_iqr_gated_scratch(const RadArgs* a);
extern "C" int bc_rad_gemm_iqr_tiled(const RadArgs* a, RadStream s);
extern "C" int64_t bc_rad_gemm_iqr_tiled_scratch(const RadArgs* a);

/* The dispatch log r4d_emit.h's launch helper reads (libr4d's r4d_record_dispatches fills its own copy; the engine never calls
 * it, only libr4d's selftest does, to see which form ran). This plugin keeps the log off. */
__attribute__((visibility("hidden"))) const void** r4d_dispatch_log = nullptr;
__attribute__((visibility("hidden"))) int           r4d_dispatch_cap = 0;
__attribute__((visibility("hidden"))) int           r4d_dispatch_n   = 0;

/* What the owned GEMM source asks of the base's fp8a8 unit (bc_iqr_gemm.hip declares them): the split-K counter block, and the narrow
 * tile's geometry rule (host functions, the same text in libr3 and libr4d; the base of the card the call runs on answers). */
bool bc_card_is_gfx11();
unsigned int* bc_8a8_split_ctr(RadStream s, const void* out);
uint32_t bc_8a8_narrow_mw(uint32_t n, uint32_t m);
uint32_t bc_8a8_narrow_ksplit(uint32_t N, uint32_t K, uint32_t M, uint32_t* ck, uint32_t target);
namespace {
unsigned int* bc_8a8_split_ctr_impl(RadStream s, const void* out);
uint32_t narrow_mw_impl(uint32_t n, uint32_t m);
uint32_t narrow_ksplit_impl(uint32_t N, uint32_t K, uint32_t M, uint32_t* ck, uint32_t target);
}
namespace {
int card_family();
}
bool bc_card_is_gfx11() { return card_family() == 11; }
unsigned int* bc_8a8_split_ctr(RadStream s, const void* out) { return bc_8a8_split_ctr_impl(s, out); }
uint32_t bc_8a8_narrow_mw(uint32_t n, uint32_t m) { return narrow_mw_impl(n, m); }
uint32_t bc_8a8_narrow_ksplit(uint32_t N, uint32_t K, uint32_t M, uint32_t* ck, uint32_t target) { return narrow_ksplit_impl(N, K, M, ck, target); }

namespace {

/* One base library's exported entry points. */
struct Lib {
    RadLaunchFn embed_lookup_iqr = nullptr;
    RadLayoutFn layout_iqr_embed = nullptr;
    RadRelayoutFn relayout_iqr_embed = nullptr;
    RadShapeFn shape_embed_lookup = nullptr;
    RadLaunchFn logits_gemm_vocab = nullptr;
    RadLayoutFn layout_iqr_vocab = nullptr;
    RadRelayoutFn relayout_iqr_vocab = nullptr;
    RadUnrelayoutFn unrelayout_iqr_vocab = nullptr;
    RadLaunchFn gemm_iqr_m64 = nullptr, gemm_iqr_gated = nullptr, gemm_iqr_tiled = nullptr;
    RadLayoutFn layout_iqr = nullptr;
    RadRelayoutFn relayout_iqr = nullptr;
    RadUnrelayoutFn unrelayout_iqr = nullptr;
    RadFuseFn fuse_gated_iqr = nullptr;
    unsigned int* (*split_ctr)(void*, const void*) = nullptr;   // r4d_8a8_split_ctr, C++ linkage
    uint32_t (*narrow_mw)(uint32_t, uint32_t) = nullptr;        // r4d_8a8_narrow_mw
    uint32_t (*narrow_ksplit)(uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t) = nullptr;   // r4d_8a8_narrow_ksplit
    bool gemm_hooks() const { return layout_iqr && relayout_iqr && unrelayout_iqr; }
    bool hooks() const { return layout_iqr_embed && relayout_iqr_embed && shape_embed_lookup; }
    bool vocab_hooks() const { return layout_iqr_vocab && relayout_iqr_vocab && unrelayout_iqr_vocab; }
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
    x.logits_gemm_vocab = (RadLaunchFn)dlsym(h, "r4d_rad_logits_gemm_vocab");
    x.layout_iqr_vocab = (RadLayoutFn)dlsym(h, "r4d_rad_layout_iqr_vocab");
    x.relayout_iqr_vocab = (RadRelayoutFn)dlsym(h, "r4d_rad_relayout_iqr_vocab");
    x.unrelayout_iqr_vocab = (RadUnrelayoutFn)dlsym(h, "r4d_rad_unrelayout_iqr_vocab");
    x.gemm_iqr_m64 = (RadLaunchFn)dlsym(h, "r4d_rad_gemm_iqr_m64");
    x.gemm_iqr_gated = (RadLaunchFn)dlsym(h, "r4d_rad_gemm_iqr_gated");
    x.gemm_iqr_tiled = (RadLaunchFn)dlsym(h, "r4d_rad_gemm_iqr_tiled");
    x.layout_iqr = (RadLayoutFn)dlsym(h, "r4d_rad_layout_iqr");
    x.relayout_iqr = (RadRelayoutFn)dlsym(h, "r4d_rad_relayout_iqr");
    x.unrelayout_iqr = (RadUnrelayoutFn)dlsym(h, "r4d_rad_unrelayout_iqr");
    x.fuse_gated_iqr = (RadFuseFn)dlsym(h, "r4d_fuse_gemm_nt_q_gated_iqr");
    x.split_ctr = (unsigned int* (*)(void*, const void*))dlsym(h, "_Z17r4d_8a8_split_ctrPvPKv");
    x.narrow_mw = (uint32_t (*)(uint32_t, uint32_t))dlsym(h, "_Z17r4d_8a8_narrow_mwjj");
    x.narrow_ksplit = (uint32_t (*)(uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t))dlsym(h, "_Z21r4d_8a8_narrow_ksplitjjjPjj");
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
const Lib& vocab_hook_lib() {
    const Bases& b = bases();
    return b.r4d.vocab_hooks() ? b.r4d : b.r3;
}
/* The same for the GEMM family's hooks: a base that has the embedding hooks but lacks these is passed over. */
const Lib& gemm_hook_lib() {
    const Bases& b = bases();
    return b.r4d.gemm_hooks() ? b.r4d : b.r3;
}

/* The base's launch for the card this call runs on: libr4d's for an RDNA4 card, libr3's for an RDNA3 one (the other library's
 * device code would not run there). `vocab` picks the head's launch instead of the embedding gather's. */
/* The architecture of the card this call runs on: "gfx12*" is RDNA4, "gfx11*" RDNA3; cached per device. */
int card_family() {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return 0;
    static std::mutex m;
    static std::map<int, int> family;
    std::lock_guard<std::mutex> lock(m);
    auto it = family.find(dev);
    if (it == family.end()) {
        hipDeviceProp_t p;
        if (hipGetDeviceProperties(&p, dev) != hipSuccess) return 0;
        it = family.emplace(dev, std::strncmp(p.gcnArchName, "gfx12", 5) == 0 ? 12 : std::strncmp(p.gcnArchName, "gfx11", 5) == 0 ? 11 : 0).first;
    }
    return it->second;
}

/* The base of the card the call runs on: libr4d for RDNA4, libr3 for RDNA3 (the other library's device code would not run there);
 * null for any other card. */
const Lib* base_for_card() {
    const int f = card_family();
    const Bases& b = bases();
    return f == 12 ? &b.r4d : f == 11 ? &b.r3 : nullptr;
}
RadLaunchFn base_launch(bool vocab = false) {
    const Lib* l = base_for_card();
    return !l ? nullptr : vocab ? l->logits_gemm_vocab : l->embed_lookup_iqr;
}

int embed_launch(const RadArgs* a, RadStream s) {
    const int rc = bc_rad_embed_lookup_iqr(a, s);
    if (rc != RAD_E_UNSUPPORTED) return rc;
    const RadLaunchFn fallback = base_launch();   // a form the owned kernel declines
    return fallback ? fallback(a, s) : RAD_E_UNSUPPORTED;
}
/* The split-K counter block of the base on this card: the base keys it on the output buffer's lifetime, so the overlay asks for the
 * same one instead of keeping a second pool. Called by the owned GEMM source (bc_8a8_split_ctr). */
unsigned int* bc_8a8_split_ctr_impl(RadStream s, const void* out) {
    const Lib* l = base_for_card();
    return l->split_ctr(s, out);
}

uint32_t narrow_mw_impl(uint32_t n, uint32_t m) {
    const Lib* l = base_for_card();
    return l->narrow_mw(n, m);   // gemm_helpers() was true before the launch
}
uint32_t narrow_ksplit_impl(uint32_t N, uint32_t K, uint32_t M, uint32_t* ck, uint32_t target) {
    const Lib* l = base_for_card();
    return l->narrow_ksplit(N, K, M, ck, target);
}

/* The helpers the owned GEMM source asks of the base's fp8a8 unit (split-K counters, the narrow tile's geometry) must exist for the card
 * before a launch is accepted: without them the launch declines (no GPU work done), it never guesses a geometry. */
bool gemm_helpers() {
    const Lib* l = base_for_card();
    return l && l->split_ctr && l->narrow_mw && l->narrow_ksplit;
}
/* No handing back to the base: the owned source is the base's own code, so a form it declines the base declines too (libr3 does not
 * build the unit at all). */
template <int (*Owned)(const RadArgs*, RadStream)>
int gemm_launch(const RadArgs* a, RadStream s) {
    return gemm_helpers() ? Owned(a, s) : RAD_E_UNSUPPORTED;
}
template <int64_t (*Owned)(const RadArgs*)>
int64_t gemm_scratch(const RadArgs* a) {
    return gemm_helpers() ? Owned(a) : -1;   // the scratch size asks the helpers too
}
int layout_gemm(const RadParam* p, int n_p, int op, const RadEncoding* e, const int* sel, const RadTensor* pl, int n_pl,
                RadLayout* out) {
    const Lib& l = gemm_hook_lib();
    const Bases& b = bases();
    const bool helpers = (b.r4d.split_ctr && b.r4d.narrow_mw && b.r4d.narrow_ksplit) || (b.r3.split_ctr && b.r3.narrow_mw && b.r3.narrow_ksplit);
    return l.gemm_hooks() && helpers ? l.layout_iqr(p, n_p, op, e, sel, pl, n_pl, out) : RAD_E_DTYPE;   // refuse, never accept, without the base's hook
}
int relayout_gemm(const RadParam* p, int n_p, int op, const RadEncoding* e, const int* sel, const RadTensor* pl, int n_pl,
                  void* dst, int64_t dst_bytes) {
    const Lib& l = gemm_hook_lib();
    return l.gemm_hooks() ? l.relayout_iqr(p, n_p, op, e, sel, pl, n_pl, dst, dst_bytes) : RAD_E_UNSUPPORTED;
}
int unrelayout_gemm(const RadParam* p, int n_p, int op, const RadEncoding* e, const int* sel, const void* src, int64_t src_bytes,
                    const RadTensor* pl, int n_pl) {
    const Lib& l = gemm_hook_lib();
    return l.gemm_hooks() ? l.unrelayout_iqr(p, n_p, op, e, sel, src, src_bytes, pl, n_pl) : RAD_E_UNSUPPORTED;
}
/* The base's fusion claim for the gated row (a chain of two rows: the GEMM, then the rotating quantiser). */
int fuse_gated(const RadParam* p, int n_p, uint64_t present, int step, RadFuseStep* out) {
    const Lib& l = gemm_hook_lib();
    return l.fuse_gated_iqr ? l.fuse_gated_iqr(p, n_p, present, step, out) : RAD_E_UNSUPPORTED;
}

int vocab_launch(const RadArgs* a, RadStream s) {
    const int rc = bc_rad_logits_gemm_vocab(a, s);
    if (rc != RAD_E_UNSUPPORTED) return rc;
    const RadLaunchFn fallback = base_launch(true);
    return fallback ? fallback(a, s) : RAD_E_UNSUPPORTED;
}
int layout_vocab(const RadParam* p, int n_p, int op, const RadEncoding* e, const int* sel, const RadTensor* pl, int n_pl,
                 RadLayout* out) {
    const Lib& l = vocab_hook_lib();
    return l.vocab_hooks() ? l.layout_iqr_vocab(p, n_p, op, e, sel, pl, n_pl, out) : RAD_E_DTYPE;   // refuse, never accept, without the base's hook
}
int relayout_vocab(const RadParam* p, int n_p, int op, const RadEncoding* e, const int* sel, const RadTensor* pl, int n_pl,
                   void* dst, int64_t dst_bytes) {
    const Lib& l = vocab_hook_lib();
    return l.vocab_hooks() ? l.relayout_iqr_vocab(p, n_p, op, e, sel, pl, n_pl, dst, dst_bytes) : RAD_E_UNSUPPORTED;
}
int unrelayout_vocab(const RadParam* p, int n_p, int op, const RadEncoding* e, const int* sel, const void* src, int64_t src_bytes,
                     const RadTensor* pl, int n_pl) {
    const Lib& l = vocab_hook_lib();
    return l.vocab_hooks() ? l.unrelayout_iqr_vocab(p, n_p, op, e, sel, src, src_bytes, pl, n_pl) : RAD_E_UNSUPPORTED;
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
/* The base's gemm_iqr_* constraints (R4D_FP8_BLOCK is 128). */
const RadConstraint kGemmM64[] = { RAD_CLE("M", 64), RAD_CDIV("K", 256), RAD_CDIV("N", 16), RAD_CEQ("group", 128),
                                   RAD_CIN("dtype", "i8a8 i8a8h") };
const RadConstraint kGemmGated[] = { RAD_CLE("M", 16), RAD_CDIV("K", 256), RAD_CDIV("N", 256), RAD_CEQ("group", 128),
                                     RAD_CIN("dtype", "i8a8h"), RAD_CIN("act", "silu") };
const RadConstraint kGemmTiled[] = { RAD_CDIV("K", 256), RAD_CDIV("N", 128), RAD_CEQ("group", 128), RAD_CIN("dtype", "i8a8 i8a8h") };
/* The base's logits_gemm_vocab_iqr constraints. */
const RadConstraint kVocab[] = { RAD_CDIV("n_embd", 256), RAD_CIN("dtype", "bf16") };

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
    { "bc_logits_gemm_vocab_iqr", "logits_gemm_vocab", "vocab",
      "the vocabulary-weighted IQR head in one launch over every class (RCS23 S3b): libr4d's computation and stored form, the "
      "gfx11 accumulator-layout and lookahead forms of libr3 under __gfx1100__",
      "any M (64 tokens a pass); n_embd a multiple of 256; each class a multiple of 256 rows; otherwise as libr4d's "
      "logits_gemm_vocab_iqr",
      "bf16 x, rotated by fwht128 and quantised to 16 bits a (token, 128) in the call's scratch; classes of tcq_code=mad codes "
      "under one f32 scale a row; F32 LOGITS OUT, int32 block sums folded in fp32, bit for bit libr4d's",
      RAD_DOMAIN_DEVICE, 70,
      kVocab, (int)(sizeof(kVocab) / sizeof(kVocab[0])),
      nullptr, 0, nullptr,
      nullptr, nullptr, vocab_launch, bc_rad_logits_gemm_vocab_scratch, layout_vocab, relayout_vocab, nullptr, unrelayout_vocab,
      nullptr },
    /* The base rows' relative order is kept (m64 60, tiled 40, the gated fold on its own op): +30 each. */
    { "bc_gemm_iqr_nt_m64", "gemm_nt_q", "gemm",
      "gemm_i8a8_nt_m16 over an IQR weight kept at its codes' width, the 16- and 64-token tiles (RCS23 S3c): libr4d's computation "
      "and stored form, libr3's gfx11 forms under __gfx1100__",
      "M <= 64; K a multiple of 256; N a multiple of 16, and of a trellis's run; otherwise as libr4d's gemm_iqr_nt_m64",
      "as libr4d's gemm_iqr_nt_m64, bit for bit",
      RAD_DOMAIN_DEVICE, 90, kGemmM64, (int)(sizeof(kGemmM64) / sizeof(kGemmM64[0])), nullptr, 0, nullptr,
      nullptr, nullptr, gemm_launch<bc_rad_gemm_iqr_m64>, gemm_scratch<bc_rad_gemm_iqr_m64_scratch>, layout_gemm, relayout_gemm,
      nullptr, unrelayout_gemm, nullptr },
    { "bc_gemm_iqr_gated", "gemm_nt_q_gated", "gemm",
      "gemm_iqr_nt_m64's GEMV and 16-token tile over a gate_up weight with the gated rotate-and-quantise pass in the epilogue "
      "(RCS23 S3c)",
      "M <= 16; K a multiple of 256; N a multiple of 256; act silu; otherwise as libr4d's gemm_iqr_gated",
      "as libr4d's gemm_iqr_gated, byte-identical to gemm_iqr_nt_m64 followed by gated_had_quant_i8g",
      RAD_DOMAIN_DEVICE, 80, kGemmGated, (int)(sizeof(kGemmGated) / sizeof(kGemmGated[0])), nullptr, 0, nullptr,
      nullptr, nullptr, gemm_launch<bc_rad_gemm_iqr_gated>, gemm_scratch<bc_rad_gemm_iqr_gated_scratch>, layout_gemm,
      relayout_gemm, nullptr, unrelayout_gemm, fuse_gated },
    { "bc_gemm_iqr_tiled", "gemm_nt_q", "gemm",
      "the prefill GEMM over an IQR weight (RCS23 S3c): the single-band fused tile up to 128 rows, the decoded-plane route above",
      "any M; K a multiple of 256 and N of 128; otherwise as libr4d's gemm_iqr_tiled",
      "as libr4d's gemm_iqr_tiled, through the int8 prefill tile's arithmetic and output bytes",
      RAD_DOMAIN_DEVICE, 70, kGemmTiled, (int)(sizeof(kGemmTiled) / sizeof(kGemmTiled[0])), nullptr, 0, nullptr,
      nullptr, nullptr, gemm_launch<bc_rad_gemm_iqr_tiled>, gemm_scratch<bc_rad_gemm_iqr_tiled_scratch>, layout_gemm,
      relayout_gemm, nullptr, unrelayout_gemm, nullptr },
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
