#include "r3_ngram_api.h"
#include "rad_abi.h"
#include "rad_plugin.h"

#include <dlfcn.h>
#include <link.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
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

    /* The rows no device holds, [covered, vocab), as a locked copy in host memory: libavx gathers from it as
     * memory (its file path, O_DIRECT reads of the container, is for a table left on disk). R3_NGRAM_RAM=0
     * leaves them in the container instead. */
    /* where a call's time goes, reported once at shutdown */
    uint64_t calls = 0, repeats = 0, dev_rows = 0, host_rows = 0;
    double dev_us = 0, host_us = 0, load_us = 0;

    int64_t covered = 0;
    uint8_t* ram = nullptr;
    size_t ram_bytes = 0;

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

/* The table's bytes, read from the container with ordinary sequential reads when the pointer is a file mapping:
 * faulting a cold mapping in page by page is tens of MB/s on this host, a read of the file hundreds. `sink` is
 * given consecutive pieces. A pointer that is not a file mapping is read as memory. */
struct Backing { int fd = -1; long long off = 0; };

Backing backing_of(const uint8_t* p) {
    Backing b;
    std::FILE* f = std::fopen("/proc/self/maps", "re");
    if (!f) return b;
    char line[4352];
    while (std::fgets(line, sizeof line, f)) {
        unsigned long long lo = 0, hi = 0, off = 0, inode = 0;
        char perms[8];
        unsigned dmaj = 0, dmin = 0;
        int at = 0;
        if (std::sscanf(line, "%llx-%llx %7s %llx %x:%x %llu %n", &lo, &hi, perms, &off, &dmaj, &dmin, &inode, &at) < 7)
            continue;
        if ((uintptr_t)p < lo || (uintptr_t)p >= hi) continue;
        std::string name(line + at);
        while (!name.empty() && (name.back() == '\n' || name.back() == ' ')) name.pop_back();
        if (inode != 0 && !name.empty() && name[0] == '/') {
            b.fd = ::open(name.c_str(), O_RDONLY | O_CLOEXEC);
            b.off = (long long)off + (long long)((uintptr_t)p - lo);
        }
        break;
    }
    std::fclose(f);
    return b;
}

template <class Sink>
int read_table(const uint8_t* src, size_t bytes, Sink sink) {
    const size_t step = 256ull << 20;
    Backing b = backing_of(src);
    if (b.fd < 0) {
        for (size_t o = 0; o < bytes; o += step) {
            const int rc = sink(o, src + o, std::min(step, bytes - o));
            if (rc != RAD_OK) return rc;
        }
        return RAD_OK;
    }
    posix_fadvise(b.fd, b.off, (off_t)bytes, POSIX_FADV_SEQUENTIAL);
    std::vector<uint8_t> buf(std::min(step, bytes));
    int rc = RAD_OK;
    for (size_t o = 0; o < bytes && rc == RAD_OK;) {
        const size_t len = std::min(step, bytes - o);
        size_t got = 0;
        while (got < len) {
            const ssize_t r = ::pread(b.fd, buf.data() + got, len - got, (off_t)(b.off + (long long)(o + got)));
            if (r <= 0) { rc = RAD_E_IO; break; }
            got += (size_t)r;
        }
        if (rc != RAD_OK) break;
        rc = sink(o, buf.data(), len);
        /* read once: these pages are not wanted in the page cache afterwards */
        posix_fadvise(b.fd, (off_t)(b.off + (long long)o), (off_t)len, POSIX_FADV_DONTNEED);
        o += len;
    }
    ::close(b.fd);
    return rc;
}

bool ram_wanted() {
    const char* e = std::getenv("R3_NGRAM_RAM");
    return !(e && *e == '0');
}

void ram_copy(State* s, const uint8_t* table) {
    if (s->ram) { munmap(s->ram, s->ram_bytes); s->ram = nullptr; s->ram_bytes = 0; }
    if (!ram_wanted() || s->covered >= s->vocab) return;
    const size_t bytes = (size_t)(s->vocab - s->covered) * (size_t)s->n;
    void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) {
        std::fprintf(stderr, "libr3: R3_NGRAM_DEV could not map %.2f GiB for the rows off the device(s): %s; "
                     "they are read from the container\n", (double)bytes / (double)(1ull << 30), std::strerror(errno));
        return;
    }
    const uint8_t* src = table + (size_t)s->covered * (size_t)s->n;
    const int rrc = read_table(src, bytes, [&](size_t o, const uint8_t* piece, size_t len) {
        std::memcpy((uint8_t*)m + o, piece, len);
        return (int)RAD_OK;
    });
    if (rrc != RAD_OK) {
        std::fprintf(stderr, "libr3: R3_NGRAM_DEV could not read the rows off the device(s) from the container; "
                     "they are read from it a row at a time\n");
        munmap(m, bytes);
        return;
    }
    const bool locked = mlock(m, bytes) == 0;
    s->ram = (uint8_t*)m;
    s->ram_bytes = bytes;
    std::fprintf(stderr, "libr3: R3_NGRAM_DEV rows %lld..%lld (%.2f GiB) copied into host memory%s\n",
                 (long long)s->covered, (long long)s->vocab, (double)bytes / (double)(1ull << 30),
                 locked ? ", locked" : " (not locked: RLIMIT_MEMLOCK)");
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
    if (s->ram) {
        /* libavx indexes the copy from its first row */
        for (int i = 0; i < miss; ++i) s->miss_ids[i] -= (int32_t)s->covered;
        t[1].data = s->ram;
    }
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
    s->covered = first;

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
    std::fprintf(stderr, "libr3: R3_NGRAM_DEV %llu gathers (+%llu repeats served from the last one): device %llu rows "
                 "%.1f us a gather, host copy %llu rows %.1f us a gather; one-time load %.1f s\n",
                 (unsigned long long)g->calls, (unsigned long long)g->repeats, (unsigned long long)g->dev_rows,
                 g->calls ? g->dev_us / (double)g->calls : 0.0, (unsigned long long)g->host_rows,
                 g->calls ? g->host_us / (double)g->calls : 0.0, g->load_us / 1e6);
    for (const Shard& sh : g->shards) g->hp.release(sh.dev);
    if (g->ram) munmap(g->ram, g->ram_bytes);
    if (g->avx_h) dlclose(g->avx_h);
    if (g->hp.h) dlclose(g->hp.h);
    g.reset();
}

int launch_ngram(const RadArgs* a, RadStream) {
    if (!a || a->n_t < 4) return RAD_E_INVAL;
    State* s = static_cast<State*>(a->instance);
    if (!s || s != g.get()) return RAD_E_STATE;
    const long long ln = pget(a->p, a->n_p, "n_embd", 0);
    const long long lv = pget(a->p, a->n_p, "n_vocab", 0);
    const int voff = (int)pget(a->p, a->n_p, "vocab_offset", 0);
    if (ln != s->n || lv != s->vocab) return RAD_E_SHAPE;
    const int n = s->n;
    /* The rows of THIS call are the output tensor's, as libavx takes them: the M parameter is a declared range and
     * its upper bound is the largest chunk, not what a decode step asks for. */
    long long out_elems = a->t[3].rank > 0 ? 1 : 0;
    for (int d = 0; d < a->t[3].rank; ++d) out_elems *= a->t[3].shape[d];
    const long long lm = out_elems / n;
    if (lm <= 0 || lm > s->max_m || lm * n != out_elems) return RAD_E_SHAPE;
    const int m = (int)lm;
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
        ++s->repeats;
        return RAD_OK;
    }

    using clk = std::chrono::steady_clock;
    auto us_since = [](clk::time_point t0) {
        return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
    };
    if (s->loaded_table != table) {
        const auto t0 = clk::now();
        for (const Shard& sh : s->shards) {
            const int dev = sh.dev;
            const int rc = read_table(table + (size_t)sh.first * n, (size_t)sh.rows * (size_t)n,
                [&](size_t o, const uint8_t* piece, size_t len) {
                    return s->hp.load(dev, (int64_t)o, piece, (int64_t)len);
                });
            if (rc != RAD_OK) return rc;
        }
        ram_copy(s, table);
        s->loaded_table = table;
        s->last_valid = false;
        s->load_us += us_since(t0);
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

    const auto t_dev = clk::now();
    for (Shard& sh : s->shards) {
        if (!sh.count) continue;
        s->dev_rows += (uint64_t)sh.count;
        const int rc = s->hp.gather(sh.dev, sh.ids.data(), sh.count, voff, scale, sh.out.data());
        if (rc != RAD_OK) return rc;
        for (int i = 0; i < sh.count; ++i)
            std::memcpy(dst + (size_t)sh.pos[i] * n, sh.out.data() + (size_t)i * n,
                        (size_t)n * sizeof(uint16_t));
    }
    s->dev_us += us_since(t_dev);
    const auto t_host = clk::now();
    const int rc = fallback(s, a, miss, n, dst);
    if (rc != RAD_OK) return rc;
    s->host_us += us_since(t_host);
    s->host_rows += (uint64_t)miss;
    ++s->calls;

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
        x.computes = "E4M3 n-gram gather on non-rank HIP devices; the rows they do not hold from a host copy through libavx";
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
