#include "r3_ngram_api.h"
#include "rad_abi.h"
#include "rad_plugin.h"

#include <dlfcn.h>
#include <link.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

extern "C" int r3_ngram_enabled(void);

namespace {

static const RadConstraint kNgramConstraints[] = {
    RAD_CEQ("n_embd", 160),
    RAD_CEQ("n_vocab", 320001536),
};

struct Helper {
    void* h = nullptr;
    decltype(&r3_ngram_helper_abi) abi = nullptr;
    decltype(&r3_ngram_helper_capacity) capacity = nullptr;
    decltype(&r3_ngram_helper_prepare) prepare = nullptr;
    decltype(&r3_ngram_helper_load) load = nullptr;
    decltype(&r3_ngram_helper_gather) gather = nullptr;
    decltype(&r3_ngram_helper_release) release = nullptr;
};

struct Shard {
    int dev = -1;
    int64_t first = 0;
    int64_t rows = 0;
    std::vector<int32_t> ids;
    std::vector<int> pos;
    std::vector<uint16_t> out;
    int count = 0;
};

struct State {
    Helper hp;
    int n = 0;
    int64_t vocab = 0;
    int max_m = 0;
    int refs = 0;
    std::vector<Shard> shards;
    const void* loaded_table = nullptr;

    void* avx_h = nullptr;
    RadLaunchFn avx_launch = nullptr;

    std::vector<int32_t> miss_ids;
    std::vector<int> miss_pos;
    std::vector<uint16_t> miss_out;

    bool last_valid = false;
    const void* last_table = nullptr;
    int last_m = 0;
    int last_voff = 0;
    uint32_t last_scale_bits = 0;
    std::vector<int32_t> last_ids;
    std::vector<uint16_t> last_out;
};

std::mutex g_mu;
std::unique_ptr<State> g;

long long pget(const RadParam* p, int n, const char* key, long long def) {
    for (int i = 0; i < n; ++i)
        if (!std::strcmp(p[i].key, key) && (p[i].kind == RAD_P_INT || p[i].kind == RAD_P_RANGE))
            return p[i].kind == RAD_P_RANGE ? p[i].ihi : p[i].ival;
    return def;
}

std::vector<int> parse_devices() {
    std::vector<int> out;
    const char* e = std::getenv("R3_NGRAM_DEV");
    if (!e || !*e) return out;
    const char* p = e;
    while (*p) {
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p || v < 0 || v > 255) return {};
        out.push_back((int)v);
        p = end;
        if (!*p) break;
        if (*p != ',') return {};
        ++p;
    }
    return out;
}

std::string sibling_helper() {
    Dl_info di{};
    if (!dladdr((const void*)&r3_ngram_enabled, &di) || !di.dli_fname) return "libr3_ngram_helper.so";
    std::string p(di.dli_fname);
    const size_t s = p.rfind('/');
    return (s == std::string::npos ? std::string() : p.substr(0, s + 1)) + "../helpers/libr3_ngram_helper.so";
}

bool open_helper(Helper* h) {
    const char* e = std::getenv("R3_NGRAM_HELPER_SO");
    const std::string path = e && *e ? e : sibling_helper();
    h->h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h->h) {
        std::fprintf(stderr, "libr3: R3_NGRAM_DEV helper dlopen failed: %s\n", dlerror());
        return false;
    }
#define R3H(n) h->n = reinterpret_cast<decltype(h->n)>(dlsym(h->h, "r3_ngram_helper_" #n))
    R3H(abi); R3H(capacity); R3H(prepare); R3H(load); R3H(gather); R3H(release);
#undef R3H
    if (!h->abi || h->abi() != R3_NGRAM_HELPER_ABI || !h->capacity || !h->prepare ||
        !h->load || !h->gather || !h->release) {
        std::fprintf(stderr, "libr3: R3_NGRAM_DEV helper ABI mismatch\n");
        return false;
    }
    return true;
}

struct AvxPath { std::string path; };
int avx_cb(dl_phdr_info* i, size_t, void* opaque) {
    if (i && i->dlpi_name && std::strstr(i->dlpi_name, "libavx.so")) {
        static_cast<AvxPath*>(opaque)->path = i->dlpi_name;
        return 1;
    }
    return 0;
}

bool open_avx(State* s) {
    if (s->avx_launch) return true;
    AvxPath found;
    dl_iterate_phdr(avx_cb, &found);
    const char* e = std::getenv("R3_NGRAM_AVX");
    const std::string path = !found.path.empty() ? found.path : (e && *e ? e : "libavx.so");
    s->avx_h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!s->avx_h) return false;
    using CountFn = int (*)(void);
    using AtFn = const RadKernelInfo* (*)(int);
    auto count = reinterpret_cast<CountFn>(dlsym(s->avx_h, "rad_kernel_count"));
    auto at = reinterpret_cast<AtFn>(dlsym(s->avx_h, "rad_kernel_at"));
    if (!count || !at) return false;
    for (int i = 0; i < count(); ++i) {
        const RadKernelInfo* k = at(i);
        if (k && k->op && !std::strcmp(k->op, "embed_lookup_q") &&
            k->domain == RAD_DOMAIN_HOST && k->launch) {
            s->avx_launch = k->launch;
            return true;
        }
    }
    return false;
}

float scale_of(const RadArgs* a) {
    if (a->n_t <= 2 || !a->t[2].data) return 1.0f;
    if (a->t[2].dtype == RAD_BF16) return rad_bf16_to_f32(*(const uint16_t*)a->t[2].data);
    if (a->t[2].dtype == RAD_F32) return *(const float*)a->t[2].data;
    return 0.0f;
}

uint32_t fbits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

int fallback(State* s, const RadArgs* a, int miss, int n, uint16_t* dst) {
    if (!miss) return RAD_OK;
    if (!open_avx(s)) {
        std::fprintf(stderr, "libr3: R3_NGRAM_DEV has %d miss row(s), but libavx embed_lookup_q is not loaded\n", miss);
        return RAD_E_NOKERNEL;
    }
    if (a->n_t < 4 || a->n_t > 5 || a->n_p > 16) return RAD_E_SHAPE;
    RadTensor t[5]{};
    for (int i = 0; i < a->n_t; ++i) t[i] = a->t[i];
    t[0].data = s->miss_ids.data();
    t[0].rank = 1;
    t[0].shape[0] = miss;
    rad_tensor_pack(&t[0]);
    t[3].data = s->miss_out.data();
    t[3].rank = 2;
    t[3].shape[0] = miss;
    t[3].shape[1] = n;
    rad_tensor_pack(&t[3]);
    if (a->n_t > 4) t[4].data = nullptr;  // helper hits need no disk read-ahead

    RadParam p[16];
    std::memcpy(p, a->p, (size_t)a->n_p * sizeof(RadParam));
    for (int i = 0; i < a->n_p; ++i)
        if (!std::strcmp(p[i].key, "M")) {
            p[i].kind = RAD_P_INT;
            p[i].ival = miss;
            p[i].ihi = 0;
        }
    RadArgs aa = *a;
    aa.t = t;
    aa.p = p;
    aa.instance = nullptr;
    const int rc = s->avx_launch(&aa, nullptr);
    if (rc != RAD_OK) return rc;
    for (int i = 0; i < miss; ++i)
        std::memcpy(dst + (size_t)s->miss_pos[i] * n,
                    s->miss_out.data() + (size_t)i * n, (size_t)n * sizeof(uint16_t));
    return RAD_OK;
}

int init_ngram(const RadParam* p, int n_p, int, int, void** out) {
    std::lock_guard<std::mutex> lk(g_mu);
    const std::vector<int> devs = parse_devices();
    if (devs.empty()) return RAD_E_INVAL;
    const int n = (int)pget(p, n_p, "n_embd", 0);
    const int64_t vocab = pget(p, n_p, "n_vocab", 0);
    int max_m = (int)pget(p, n_p, "M", 1);
    if (const char* e = std::getenv("R3_NGRAM_MAX_M")) max_m = std::max(max_m, std::atoi(e));
    else max_m = std::max(max_m, 32768);
    if (n <= 0 || vocab <= 0 || max_m <= 0) return RAD_E_SHAPE;

    if (g) {
        if (g->n != n || g->vocab != vocab || g->max_m < max_m) return RAD_E_STATE;
        ++g->refs;
        if (out) *out = g.get();
        return RAD_OK;
    }

    std::unique_ptr<State> s(new State());
    s->n = n; s->vocab = vocab; s->max_m = max_m; s->refs = 1;
    if (!open_helper(&s->hp)) return RAD_E_DEVICE;

    uint64_t reserve = 2048ull << 20;
    if (const char* e = std::getenv("R3_NGRAM_RESERVE_MIB"))
        reserve = (uint64_t)std::strtoull(e, nullptr, 10) << 20;
    uint64_t limit = UINT64_MAX;
    if (const char* e = std::getenv("R3_NGRAM_MIB"))
        limit = (uint64_t)std::strtoull(e, nullptr, 10) << 20;

    int64_t first = 0;
    for (int dev : devs) {
        if (first >= vocab || limit == 0) break;
        uint64_t usable = 0;
        int rc = s->hp.capacity(dev, reserve, &usable);
        if (rc != RAD_OK) return rc;
        usable = std::min(usable, limit);
        const uint64_t call_bytes = (uint64_t)max_m * (sizeof(int32_t) + (uint64_t)n * sizeof(uint16_t)) + (1ull << 20);
        if (usable <= call_bytes) continue;
        const int64_t rows = std::min<int64_t>(vocab - first, (int64_t)((usable - call_bytes) / (uint64_t)n));
        if (rows <= 0) continue;
        rc = s->hp.prepare(dev, first, rows, n, max_m);
        if (rc != RAD_OK) return rc;
        Shard sh;
        sh.dev = dev; sh.first = first; sh.rows = rows;
        sh.ids.resize((size_t)max_m);
        sh.pos.resize((size_t)max_m);
        sh.out.resize((size_t)max_m * n);
        s->shards.push_back(std::move(sh));
        first += rows;
        const uint64_t used = (uint64_t)rows * n;
        limit = limit == UINT64_MAX ? UINT64_MAX : (limit > used ? limit - used : 0);
    }
    if (s->shards.empty()) return RAD_E_NOMEM;

    s->miss_ids.resize((size_t)max_m);
    s->miss_pos.resize((size_t)max_m);
    s->miss_out.resize((size_t)max_m * n);
    s->last_ids.resize((size_t)max_m);
    s->last_out.resize((size_t)max_m * n);
    const double frac = (double)first / (double)vocab;
    std::fprintf(stderr, "libr3: R3_NGRAM_DEV rows=%lld/%lld (%.1f%%) n=%d devices=%zu\n",
                 (long long)first, (long long)vocab, 100.0 * frac, n, s->shards.size());
    g = std::move(s);
    if (out) *out = g.get();
    return RAD_OK;
}

void fini_ngram(void*) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g || --g->refs > 0) return;
    for (const Shard& sh : g->shards) g->hp.release(sh.dev);
    if (g->avx_h) dlclose(g->avx_h);
    if (g->hp.h) dlclose(g->hp.h);
    g.reset();
}

int launch_ngram(const RadArgs* a, RadStream) {
    if (!a || a->n_t < 4) return RAD_E_INVAL;
    State* s = static_cast<State*>(a->instance);
    if (!s || s != g.get()) return RAD_E_STATE;
    const long long lm = pget(a->p, a->n_p, "M", 0);
    const long long ln = pget(a->p, a->n_p, "n_embd", 0);
    const long long lv = pget(a->p, a->n_p, "n_vocab", 0);
    const int voff = (int)pget(a->p, a->n_p, "vocab_offset", 0);
    if (lm <= 0 || lm > s->max_m || ln != s->n || lv != s->vocab) return RAD_E_SHAPE;
    const int m = (int)lm, n = s->n;
    if (!a->t[0].data || !a->t[1].data || !a->t[3].data ||
        a->t[0].dtype != RAD_I32 || a->t[1].dtype != RAD_F8E4M3 ||
        a->t[3].dtype != RAD_BF16 || !rad_tensor_is_contiguous(&a->t[0]) ||
        !rad_tensor_is_contiguous(&a->t[1]) || !rad_tensor_is_contiguous(&a->t[3]))
        return RAD_E_DTYPE;
    const float scale = scale_of(a);
    if (scale == 0.0f && a->n_t > 2 && a->t[2].data) return RAD_E_DTYPE;
    const int32_t* ids = static_cast<const int32_t*>(a->t[0].data);
    const uint8_t* table = static_cast<const uint8_t*>(a->t[1].data);
    uint16_t* dst = static_cast<uint16_t*>(a->t[3].data);

    std::lock_guard<std::mutex> lk(g_mu);
    const uint32_t sbits = fbits(scale);
    if (s->last_valid && s->last_table == table && s->last_m == m && s->last_voff == voff &&
        s->last_scale_bits == sbits &&
        !std::memcmp(s->last_ids.data(), ids, (size_t)m * sizeof(int32_t))) {
        std::memcpy(dst, s->last_out.data(), (size_t)m * n * sizeof(uint16_t));
        return RAD_OK;
    }

    if (s->loaded_table != table) {
        for (const Shard& sh : s->shards) {
            const int rc = s->hp.load(sh.dev, table + (size_t)sh.first * n, sh.rows * (int64_t)n);
            if (rc != RAD_OK) return rc;
        }
        s->loaded_table = table;
        s->last_valid = false;
    }

    for (Shard& sh : s->shards) sh.count = 0;
    int miss = 0;
    for (int i = 0; i < m; ++i) {
        const int64_t row = (int64_t)ids[i] - voff;
        Shard* hit = nullptr;
        for (Shard& sh : s->shards)
            if (row >= sh.first && row < sh.first + sh.rows) { hit = &sh; break; }
        if (hit) {
            hit->ids[hit->count] = ids[i];
            hit->pos[hit->count] = i;
            ++hit->count;
        } else {
            s->miss_ids[miss] = ids[i];
            s->miss_pos[miss] = i;
            ++miss;
        }
    }

    for (Shard& sh : s->shards) {
        if (!sh.count) continue;
        const int rc = s->hp.gather(sh.dev, sh.ids.data(), sh.count, voff, scale, sh.out.data());
        if (rc != RAD_OK) return rc;
        for (int i = 0; i < sh.count; ++i)
            std::memcpy(dst + (size_t)sh.pos[i] * n, sh.out.data() + (size_t)i * n,
                        (size_t)n * sizeof(uint16_t));
    }
    const int rc = fallback(s, a, miss, n, dst);
    if (rc != RAD_OK) return rc;

    s->last_table = table; s->last_m = m; s->last_voff = voff; s->last_scale_bits = sbits;
    std::memcpy(s->last_ids.data(), ids, (size_t)m * sizeof(int32_t));
    std::memcpy(s->last_out.data(), dst, (size_t)m * n * sizeof(uint16_t));
    s->last_valid = true;
    return RAD_OK;
}

}  // namespace

extern "C" int r3_ngram_enabled(void) {
    const char* e = std::getenv("R3_NGRAM_DEV");
    return e && *e;
}

extern "C" const RadKernelInfo* r3_ngram_row(void) {
    static const RadKernelInfo k = [] {
        RadKernelInfo x{};
        x.name = "r3_ngram_dev";
        x.op = "embed_lookup_q";
        x.family = "vocab";
        x.computes = "E4M3 n-gram gather on non-rank HIP devices with libavx fallback";
        x.shape = "any";
        x.dtypes = "i32,fp8_e4m3,bf16";
        x.domain = RAD_DOMAIN_HOST;
        x.priority = 100;
        x.constraints = kNgramConstraints;
        x.n_constraints = (int)(sizeof(kNgramConstraints) / sizeof(kNgramConstraints[0]));
        x.init = init_ngram;
        x.fini = fini_ngram;
        x.launch = launch_ngram;
        return x;
    }();
    return &k;
}
