#include "moe_remote_shm.h"
#include "rad_format.h"
#include "rad_encoding.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
using namespace r3::moe_shm;
volatile sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

float bf16(uint16_t v) {
    uint32_t u = (uint32_t)v << 16; float f; std::memcpy(&f, &u, 4); return f;
}
uint16_t to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}
float f16(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    if (!e) return std::ldexp((s ? -1.f : 1.f) * (float)m, -24);
    if (e == 31) return m ? NAN : (s ? -INFINITY : INFINITY);
    return std::ldexp((s ? -1.f : 1.f) * (1024.f + (float)m), (int)e - 25);
}
float f8e4m3(uint8_t b) {
    const float sign = (b & 0x80) ? -1.f : 1.f;
    const int e = (b >> 3) & 15, m = b & 7;
    if (e == 0) return sign * std::ldexp((float)m, -9);
    if (e == 15 && m == 7) return NAN; /* OCP E4M3FN's sole NaN payload. */
    return sign * std::ldexp(1.f + (float)m / 8.f, e - 7);
}

uint64_t ubits(const uint8_t* row, int bits, int64_t col) {
    const uint64_t bit = (uint64_t)col * (uint64_t)bits;
    const uint64_t byte = bit >> 3, shift = bit & 7;
    uint64_t x = row[byte];
    if (shift + (uint64_t)bits > 8) x |= (uint64_t)row[byte + 1] << 8;
    return (x >> shift) & ((1ull << bits) - 1);
}
int64_t sbits(uint64_t x, int bits) {
    const uint64_t sign = 1ull << (bits - 1);
    return (int64_t)((x ^ sign) - sign);
}

/* The helper deliberately consumes only Radiance's installed container ABI (rad_format.h and
 * rad_encoding.h): no core/private reader or kernel layout is linked into this reference path. */
struct Container {
    int fd = -1; size_t bytes = 0; const uint8_t* base = nullptr; const RadFileHeader* h = nullptr;
    const RadFileEntry* dir = nullptr; const RadFilePlane* planes = nullptr; const RadEncoding* encs = nullptr;
    const RadFileProfile* profile = nullptr;
    uint32_t layers = 0;
    std::vector<std::array<const RadFileEntry*, 2>> expert; /* [layer*512+expert][gate_up,down]. */

    ~Container() { if (base) munmap((void*)base, bytes); if (fd >= 0) close(fd); }
    bool range(uint64_t off, uint64_t n) const { return off <= bytes && n <= bytes - off; }
    bool open(const char* path) {
        fd = ::open(path, O_RDONLY | O_CLOEXEC); if (fd < 0) return false;
        struct stat st{}; if (fstat(fd, &st) || st.st_size < (off_t)sizeof(RadFileHeader)) return false;
        bytes = (size_t)st.st_size; base = (const uint8_t*)mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (base == MAP_FAILED) { base = nullptr; return false; }
        h = (const RadFileHeader*)base;
        if (h->magic != RAD_MAGIC || h->version != RAD_FORMAT_VER || h->file_bytes > bytes ||
            !range(h->dir_off, h->dir_count * sizeof(RadFileEntry)) ||
            !range(h->plane_off, h->plane_count * sizeof(RadFilePlane)) ||
            !range(h->enc_off, h->enc_count * sizeof(RadEncoding)) ||
            (h->prof_count && !range(h->prof_off, h->prof_count * sizeof(RadFileProfile)))) return false;
        dir = (const RadFileEntry*)(base + h->dir_off); planes = (const RadFilePlane*)(base + h->plane_off);
        encs = (const RadEncoding*)(base + h->enc_off);
        profile = h->prof_count ? (const RadFileProfile*)(base + h->prof_off) : nullptr;
        int32_t max_layer = -1;
        /* Only expert entries count: the container numbers other tensors past the MoE layers (Flash-Next: 139). */
        for (uint64_t i = 0; i < h->dir_count; ++i)
            if (dir[i].layer >= 0 && dir[i].expert >= 0) max_layer = std::max(max_layer, dir[i].layer);
        layers = (uint32_t)(max_layer + 1); expert.resize((size_t)layers * kExperts);
        for (uint64_t i = 0; i < h->dir_count; ++i) {
            const auto& e = dir[i];
            if (e.layer < 0 || e.expert < 0 || e.expert >= (int)kExperts || e.slot < 0 || e.slot >= 2) continue;
            if ((uint32_t)e.layer >= layers || e.enc >= h->enc_count || e.plane_first + e.n_planes > h->plane_count) return false;
            for (uint32_t p = 0; p < e.n_planes; ++p) if (!range(planes[e.plane_first+p].offset, planes[e.plane_first+p].bytes)) return false;
            expert[(size_t)e.layer * kExperts + (uint32_t)e.expert][e.slot] = &e;
        }
        return layers != 0;
    }
    const RadFileEntry* weight(uint32_t layer, uint32_t exp, uint32_t slot) const {
        if (layer >= layers || exp >= kExperts || slot >= 2) return nullptr;
        return expert[(size_t)layer * kExperts + exp][slot];
    }
    const RadEncoding* encoding(const RadFileEntry* e) const { return e ? &encs[e->enc] : nullptr; }
    const RadFilePlane* plane(const RadFileEntry* e, int p) const { return &planes[e->plane_first + p]; }
};

float raw_value(uint32_t dtype, const uint8_t* row, int64_t col) {
    const int bits = rad_dtype_bits(dtype);
    if (dtype == RAD_F32) { float v; std::memcpy(&v, row + col*4, 4); return v; }
    if (dtype == RAD_F16) { uint16_t v; std::memcpy(&v, row + col*2, 2); return f16(v); }
    if (dtype == RAD_BF16) { uint16_t v; std::memcpy(&v, row + col*2, 2); return bf16(v); }
    if (dtype == RAD_F8E4M3) return f8e4m3(row[col]);
    if (dtype == RAD_I8) return (float)((const int8_t*)row)[col];
    if (dtype == RAD_U8) return (float)row[col];
    if (dtype == RAD_I16) { int16_t v; std::memcpy(&v,row+col*2,2); return (float)v; }
    if (dtype == RAD_U16) { uint16_t v; std::memcpy(&v,row+col*2,2); return (float)v; }
    if (dtype == RAD_I32) { int32_t v; std::memcpy(&v,row+col*4,4); return (float)v; }
    if (bits > 0 && bits < 8) {
        const uint64_t u = ubits(row, bits, col);
        const bool si = dtype==RAD_I2 || dtype==RAD_I3 || dtype==RAD_I4 || dtype==RAD_I5 || dtype==RAD_I6 || dtype==RAD_I7;
        return (float)(si ? sbits(u,bits) : (int64_t)u);
    }
    return NAN;
}

float plane_value(const Container& f, const RadFileEntry* e, int pi, int64_t row, int64_t col) {
    const RadEncoding* enc = f.encoding(e); const RadEncPlane& ep = enc->plane[pi]; const RadFilePlane* fp = f.plane(e, pi);
    int64_t rows=1, cols=1; rad_enc_view(e->rank, e->shape, &rows, &cols);
    int64_t pr=0, pc=0; rad_enc_plane_dims(&ep, rows, cols, &pr, &pc);
    int64_t rr=row, cc=col;
    if (ep.kind == RAD_PLANE_TABLE) { rr=row; cc=col; }
    else { rr = ep.block[0] > 0 ? row / ep.block[0] : 0; cc = ep.block[1] > 0 ? col / ep.block[1] : 0; }
    if (rr < 0 || rr >= pr || cc < 0 || cc >= pc) return NAN;
    const int64_t rb = rad_enc_row_bytes(ep.dtype, pc);
    return raw_value(ep.dtype, f.base + fp->offset + rr*rb, cc);
}

bool decode_row(const Container& f, const RadFileEntry* e, int64_t row, std::vector<float>& out) {
    if (!e || row < 0 || e->rank == 0) return false;
    int64_t nr=1, nc=1; rad_enc_view(e->rank, e->shape, &nr, &nc); if (row >= nr || nc <= 0) return false;
    const RadEncoding* enc = f.encoding(e); if (!enc || !rad_enc_valid(enc)) return false;
    out.assign((size_t)nc, 0.f);
    const int codes = rad_enc_find(enc, "codes"); if (codes < 0) return false;
    const int table = rad_enc_find(enc, "table"); const int signs = rad_enc_find(enc, "signs"); const int zero = rad_enc_find(enc, "zero");
    for (int64_t c = 0; c < nc; ++c) {
        float q = plane_value(f,e,codes,row,c);
        if (!std::isfinite(q)) return false;
        if (table >= 0) q = plane_value(f,e,table,0,(int64_t)q);
        if (signs >= 0 && plane_value(f,e,signs,row,c) != 0.f) q = -q;
        if (zero >= 0) q -= plane_value(f,e,zero,row,c);
        float v=q;
        for (int lv=0;;++lv) { int p=rad_enc_find_level(enc,"scale",lv); if(p<0) break; v *= plane_value(f,e,p,row,c); }
        float mn=1.f; bool have_min=false;
        for (int lv=0;;++lv) { int p=rad_enc_find_level(enc,"min",lv); if(p<0) break; mn *= plane_value(f,e,p,row,c); have_min=true; }
        if (have_min) v -= mn;
        out[(size_t)c]=v;
    }
    const int64_t fw = rad_enc_fwht(enc);
    if (fw < 0 || (fw && nc % fw)) return false;
    if (fw) for (int64_t b=0;b<nc;b+=fw) for(int64_t d=1;d<fw;d<<=1)
        for(int64_t i=0;i<fw;i+=2*d) for(int64_t j=0;j<d;++j) {
            float a=out[(size_t)(b+i+j)], z=out[(size_t)(b+i+j+d)];
            out[(size_t)(b+i+j)]=a+z; out[(size_t)(b+i+j+d)]=a-z;
        }
    return true;
}

float dot_bf16(const uint16_t* x, const std::vector<float>& w) {
    float s=0.f; for(size_t i=0;i<w.size();++i) s += bf16(x[i])*w[i]; return s;
}
float silu(float x) { return x / (1.f + std::exp(-x)); }

int compute_host(const Container& f, void* map, Request* q) {
    if (q->rank_count != kMaxRanks || q->rank >= kMaxRanks || q->layer >= f.layers || q->rows > kMaxRows ||
        q->hidden != kHidden || q->route_count > kMaxRoutes || q->activation_stride != kHidden*2) return -EINVAL;
    const Route* routes = ptr<Route>(map,q->route_offset);
    const uint16_t* act = ptr<uint16_t>(map,q->activation_offset);
    uint16_t* dst = ptr<uint16_t>(map,q->result_offset[q->rank]);
    std::vector<float> out((size_t)q->rows*kHidden,0.f);
    std::array<std::vector<uint32_t>, kExperts> by_exp;
    for(uint32_t i=0;i<q->route_count;++i) {
        if(routes[i].token<0 || (uint32_t)routes[i].token>=q->rows || routes[i].expert<0 || routes[i].expert>=(int)kExperts) return -EINVAL;
        by_exp[(uint32_t)routes[i].expert].push_back(i);
    }
    std::vector<float> wg, wu, wd;
    for(uint32_t e=0;e<kExperts;++e) {
        auto& ri=by_exp[e]; if(ri.empty()) continue;
        const RadFileEntry* gu=f.weight(q->layer,e,0); const RadFileEntry* dn=f.weight(q->layer,e,1); if(!gu||!dn) return -ENOENT;
        int64_t gur=1,guc=1,dnr=1,dnc=1; rad_enc_view(gu->rank,gu->shape,&gur,&guc); rad_enc_view(dn->rank,dn->shape,&dnr,&dnc);
        if(guc!=kHidden || dnr!=kHidden || gur!=2*dnc || dnc!=640) return -EINVAL;
        const int parity=(int)(e&1u); const bool wide=((int)q->rank==parity); const int64_t lo=wide?0:384, hi=wide?384:640, width=hi-lo;
        std::vector<float> mid(ri.size()*(size_t)width);
        for(int64_t j=0;j<width;++j) {
            if(!decode_row(f,gu,lo+j,wg) || !decode_row(f,gu,dnc+lo+j,wu)) return -ENOTSUP;
            for(size_t a=0;a<ri.size();++a) {
                const Route& r=routes[ri[a]]; const uint16_t* x=act+(size_t)r.token*kHidden;
                mid[a*(size_t)width+(size_t)j]=silu(dot_bf16(x,wg))*dot_bf16(x,wu);
            }
        }
        for(int64_t h=0;h<kHidden;++h) {
            if(!decode_row(f,dn,h,wd)) return -ENOTSUP;
            for(size_t a=0;a<ri.size();++a) {
                float s=0.f; for(int64_t j=0;j<width;++j) s += mid[a*(size_t)width+(size_t)j]*wd[(size_t)(lo+j)];
                const Route& r=routes[ri[a]]; out[(size_t)r.token*kHidden+(size_t)h] += s*bf16(r.weight_bf16);
            }
        }
    }
    for(size_t i=0;i<out.size();++i) dst[i]=to_bf16(out[i]);
    return 0;
}

const char* shm_name() { const char* e=std::getenv("R3_MOE_HELPER_SHM"); return e&&*e?e:"/r3-moe-helper"; }
}

#include "moe_remote_gpu.h"

namespace {

bool env_on(const char* name) {
    const char* e = std::getenv(name);
    return e && *e && std::strcmp(e, "0") != 0;
}

uint64_t helper_budget_bytes() {
    const char* e = std::getenv("R3_MOE_HELPER_MIB");
    if (!e || !*e) return 28672ull * 1024ull * 1024ull;
    char* end = nullptr;
    errno = 0;
    const unsigned long long mib = std::strtoull(e, &end, 10);
    if (errno || !end || *end || mib == 0 ||
        mib > std::numeric_limits<uint64_t>::max() / (1024ull * 1024ull)) return 0;
    return (uint64_t)mib * 1024ull * 1024ull;
}

int64_t canonical_expert_bytes(const Container& f, uint32_t l, uint32_t e) {
    int64_t n = 0;
    for (uint32_t slot = 0; slot < 2; ++slot) {
        const RadFileEntry* w = f.weight(l, e, slot);
        if (!w) return -1;
        for (uint32_t p = 0; p < w->n_planes; ++p) {
            const uint64_t b = f.plane(w, (int)p)->bytes;
            if (b > (uint64_t)std::numeric_limits<int64_t>::max() - (uint64_t)n) return -1;
            n += (int64_t)b;
        }
    }
    return n;
}

struct Owned {
    std::vector<std::pair<uint32_t,uint32_t>> expert;
    uint64_t bytes = 0;
};

bool select_owned(const Container& f, r3_moe_gpu::Runner* gpu, uint64_t budget, Owned* out) {
    if (!out || !budget || !f.profile || !f.h->prof_count) {
        std::fprintf(stderr, "r3_moe_helper: container has no usable expert profile or helper budget\n");
        return false;
    }
    std::vector<const RadFileProfile*> order;
    order.reserve((size_t)f.h->prof_count);
    for (uint64_t i = 0; i < f.h->prof_count; ++i) {
        const RadFileProfile& p = f.profile[i];
        if (p.layer < 0 || p.expert < 0 || (uint32_t)p.layer >= f.layers ||
            p.expert >= (int32_t)kExperts) continue;
        order.push_back(&p);
    }
    std::stable_sort(order.begin(), order.end(), [](const RadFileProfile* a, const RadFileProfile* b) {
        if (a->share != b->share) return a->share < b->share;
        if (a->layer != b->layer) return a->layer < b->layer;
        return a->expert < b->expert;
    });
    std::vector<uint8_t> seen((size_t)f.layers * kExperts, 0);
    for (const RadFileProfile* p : order) {
        const uint32_t l = (uint32_t)p->layer, e = (uint32_t)p->expert;
        uint8_t& was = seen[(size_t)l * kExperts + e];
        if (was) continue;
        was = 1;
        const int64_t bytes = gpu ? gpu->expert_bytes(l, e) : canonical_expert_bytes(f, l, e);
        if (bytes <= 0) {
            std::fprintf(stderr, "r3_moe_helper: expert (%u,%u) cannot be laid out as w4nl64a8h\n", l, e);
            return false;
        }
        if ((uint64_t)bytes > budget - out->bytes) continue;
        out->expert.emplace_back(l, e);
        out->bytes += (uint64_t)bytes;
    }
    if (out->expert.empty()) {
        std::fprintf(stderr, "r3_moe_helper: R3_MOE_HELPER_MIB fits no profiled expert\n");
        return false;
    }
    return true;
}

void publish_owned(void* map, const Owned& owned) {
    Header* h = static_cast<Header*>(map);
    h->owned_bytes = owned.bytes;
    for (auto [l, e] : owned.expert) {
        if (!set_owned(map, l, e))
            std::fprintf(stderr, "r3_moe_helper: duplicate/invalid ownership (%u,%u)\n", l, e);
    }
}

bool parse_u32(const char* s, uint32_t lo, uint32_t hi, uint32_t* out) {
    if (!s || !*s || !out) return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long v = std::strtoul(s, &end, 10);
    if (errno || !end || *end || v < lo || v > hi) return false;
    *out = (uint32_t)v;
    return true;
}

int compare_mode(const Container& model, r3_moe_gpu::Runner& gpu, const Owned& owned,
                 uint32_t layer, uint32_t rows) {
    if (layer >= model.layers || rows == 0 || rows > kMaxRows) return 2;
    std::vector<uint32_t> avail;
    for (auto [l, e] : owned.expert) if (l == layer) avail.push_back(e);
    if (avail.empty()) {
        std::fprintf(stderr, "r3_moe_helper: --compare layer %u has no owned experts\n", layer);
        return 1;
    }
    const uint32_t k = std::min<uint32_t>(kTopK, (uint32_t)avail.size());
    const Layout lay = layout();
    std::vector<uint8_t> mem((size_t)lay.bytes);
    init(mem.data(), model.layers);
    publish_owned(mem.data(), owned);
    Header* h = reinterpret_cast<Header*>(mem.data());
    Request* q = request(mem.data(), 1);
    q->status = 0; q->layer = layer; q->rank = 0; q->rank_count = kMaxRanks;
    q->rows = rows; q->hidden = kHidden; q->route_count = rows * k;
    q->route_offset = h->route_offset[0]; q->activation_offset = h->activation_offset[0];
    q->activation_stride = kHidden * sizeof(uint16_t);
    for (uint32_t r = 0; r < kMaxRanks; ++r) {
        q->result_offset[r] = h->result_offset[r];
        q->result_stride[r] = kHidden * sizeof(uint16_t);
    }

    uint64_t rng = 0x8f3f73b5cf1c9ad1ull;
    auto rnd = [&]() -> uint32_t {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        return (uint32_t)(rng >> 16);
    };
    uint16_t* act = ptr<uint16_t>(mem.data(), q->activation_offset);
    for (size_t i = 0; i < (size_t)rows * kHidden; ++i) {
        const float x = ((int32_t)(rnd() & 0xffffu) - 32768) * (1.0f / 32768.0f);
        act[i] = to_bf16(x);
    }
    Route* rr = ptr<Route>(mem.data(), q->route_offset);
    uint32_t at = 0;
    for (uint32_t t = 0; t < rows; ++t) {
        const float w = 1.0f / (float)k;
        for (uint32_t j = 0; j < k; ++j) {
            const uint32_t ix = (uint32_t)((t * 17ull + j * 29ull + rnd()) % avail.size());
            rr[at++] = Route{(int32_t)t, (int32_t)avail[ix], to_bf16(w), 0};
        }
    }

    const int grc = gpu.compute(mem.data(), q);
    if (grc != 0) {
        std::fprintf(stderr, "r3_moe_helper: --compare GPU path failed rc=%d\n", grc);
        return 1;
    }
    std::array<std::vector<uint16_t>, kMaxRanks> got;
    for (uint32_t r = 0; r < kMaxRanks; ++r) {
        const uint16_t* p = ptr<uint16_t>(mem.data(), q->result_offset[r]);
        got[r].assign(p, p + (size_t)rows * kHidden);
    }

    bool bad = false;
    for (uint32_t r = 0; r < kMaxRanks; ++r) {
        q->rank = r;
        const int hrc = compute_host(model, mem.data(), q);
        if (hrc != 0) {
            std::fprintf(stderr, "r3_moe_helper: --compare host path rank %u failed rc=%d\n", r, hrc);
            return 1;
        }
        const uint16_t* want = ptr<uint16_t>(mem.data(), q->result_offset[r]);
        long double num = 0.0L, den = 0.0L;
        double max_abs = 0.0;
        for (size_t i = 0; i < got[r].size(); ++i) {
            const double a = (double)bf16(got[r][i]);
            const double b = (double)bf16(want[i]);
            const double d = a - b;
            num += (long double)d * d;
            den += (long double)b * b;
            max_abs = std::max(max_abs, std::abs(d));
        }
        const double rel = std::sqrt((double)(num / std::max<long double>(den, 1.0e-30L)));
        std::printf("rank %u rel_l2=%.8g max_abs=%.8g\n", r, rel, max_abs);
        bad = bad || !std::isfinite(rel) || rel > 2.0e-3;
    }
    return bad ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    bool compare = argc > 1 && std::strcmp(argv[1], "--compare") == 0;
    const char* model_path = nullptr;
    uint32_t compare_layer = 0, compare_rows = 0;
    if (compare) {
        if (argc != 7 || std::strcmp(argv[3], "--layer") != 0 ||
            std::strcmp(argv[5], "--rows") != 0 ||
            !parse_u32(argv[4], 0, kMaxLayers - 1, &compare_layer) ||
            !parse_u32(argv[6], 1, kMaxRows, &compare_rows)) {
            std::fprintf(stderr,
                "usage: r3_moe_helper --compare MODEL.rad --layer L --rows M\n");
            return 2;
        }
        model_path = argv[2];
    } else {
        if (argc != 2) {
            std::fprintf(stderr, "usage: r3_moe_helper MODEL.rad\n");
            return 2;
        }
        model_path = argv[1];
    }

    Container model;
    if (!model.open(model_path) || model.layers > kMaxLayers) {
        std::fprintf(stderr, "r3_moe_helper: cannot open/validate %s\n", model_path);
        return 1;
    }

    const bool cpu = env_on("R3_MOE_HELPER_CPU") && !compare;
    std::unique_ptr<r3_moe_gpu::Runner> gpu;
    if (!cpu) {
        gpu = std::make_unique<r3_moe_gpu::Runner>(model);
        if (!gpu->open()) return 1;
    }

    const uint64_t budget = helper_budget_bytes();
    Owned owned;
    if (!select_owned(model, gpu.get(), budget, &owned)) return 1;
    if (gpu && !gpu->upload(owned.expert)) return 1;
    std::fprintf(stderr, "r3_moe_helper: owns %zu experts, %.3f GiB\n",
                 owned.expert.size(), (double)owned.bytes / (1024.0 * 1024.0 * 1024.0));

    if (compare)
        return compare_mode(model, *gpu, owned, compare_layer, compare_rows);

    const Layout lay = layout();
    const char* name = shm_name();
    int fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) {
        std::fprintf(stderr, "r3_moe_helper: shm_open %s: %s\n", name, std::strerror(errno));
        return 1;
    }
    if (ftruncate(fd, (off_t)lay.bytes) != 0) {
        std::fprintf(stderr, "r3_moe_helper: ftruncate: %s\n", std::strerror(errno));
        shm_unlink(name); close(fd); return 1;
    }
    void* map = mmap(nullptr, lay.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        std::fprintf(stderr, "r3_moe_helper: mmap: %s\n", std::strerror(errno));
        shm_unlink(name); close(fd); return 1;
    }
    init(map, model.layers);
    publish_owned(map, owned);
    auto* h = static_cast<Header*>(map);
    std::signal(SIGINT, on_signal); std::signal(SIGTERM, on_signal);
    std::fprintf(stderr, "r3_moe_helper: ready shm=%s layers=%u bytes=%llu (%s)\n",
                 name, model.layers, (unsigned long long)lay.bytes,
                 cpu ? "host fp32 reference" : gpu->arch().c_str());

    uint32_t seq = 1;
    while (!g_stop) {
        Request* q = request(map, seq);
        while (!g_stop && load(&q->sequence) != seq) {
            const uint32_t seen = load(&h->publish_sequence);
            if (load(&q->sequence) != seq) (void)futex_wait(&h->publish_sequence, seen);
        }
        if (g_stop) break;
        q->status = cpu ? compute_host(model, map, q) : gpu->compute(map, q);
        store(&q->completion_sequence, seq);
        (void)futex_wake(&q->completion_sequence);
        if (++seq == 0) break;
    }

    munmap(map, lay.bytes);
    close(fd);
    shm_unlink(name);
    return 0;
}
