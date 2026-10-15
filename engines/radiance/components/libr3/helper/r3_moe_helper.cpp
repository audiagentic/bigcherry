#include "moe_remote_shm.h"
#include "rad_format.h"
#include "rad_encoding.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <limits>
#include <string>
#include <thread>
#include <utility>
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
    int fd = -1;
    size_t bytes = 0;
    RadFileHeader hdr{};
    const RadFileHeader* h = &hdr;
    std::vector<RadFileEntry> dir_store;
    std::vector<RadFilePlane> plane_store;
    std::vector<RadEncoding> enc_store;
    std::vector<RadFileProfile> profile_store;
    const RadFileEntry* dir = nullptr;
    const RadFilePlane* planes = nullptr;
    const RadEncoding* encs = nullptr;
    const RadFileProfile* profile = nullptr;
    uint32_t layers = 0;
    std::vector<std::array<const RadFileEntry*, 2>> expert;

    ~Container() { if (fd >= 0) close(fd); }
    bool range(uint64_t off, uint64_t n) const { return off <= bytes && n <= bytes - off; }
    bool pread_full(uint64_t off, void* dst, size_t n) const {
        if (!range(off, n)) return false;
        auto* p = static_cast<uint8_t*>(dst);
        while (n) {
            const ssize_t got = ::pread(fd, p, n, (off_t)off);
            if (got < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            if (got == 0) return false;
            p += (size_t)got;
            off += (uint64_t)got;
            n -= (size_t)got;
        }
        return true;
    }
    bool open(const char* path) {
        fd = ::open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) return false;
        struct stat st{};
        if (fstat(fd, &st) || st.st_size < (off_t)sizeof(RadFileHeader)) return false;
        bytes = (size_t)st.st_size;
        if (!pread_full(0, &hdr, sizeof(hdr))) return false;
        if (hdr.magic != RAD_MAGIC || hdr.version != RAD_FORMAT_VER || hdr.file_bytes > bytes ||
            !range(hdr.dir_off, hdr.dir_count * sizeof(RadFileEntry)) ||
            !range(hdr.plane_off, hdr.plane_count * sizeof(RadFilePlane)) ||
            !range(hdr.enc_off, hdr.enc_count * sizeof(RadEncoding)) ||
            (hdr.prof_count && !range(hdr.prof_off, hdr.prof_count * sizeof(RadFileProfile)))) return false;
        dir_store.resize((size_t)hdr.dir_count);
        plane_store.resize((size_t)hdr.plane_count);
        enc_store.resize((size_t)hdr.enc_count);
        profile_store.resize((size_t)hdr.prof_count);
        if ((hdr.dir_count && !pread_full(hdr.dir_off, dir_store.data(), dir_store.size() * sizeof(RadFileEntry))) ||
            (hdr.plane_count && !pread_full(hdr.plane_off, plane_store.data(), plane_store.size() * sizeof(RadFilePlane))) ||
            (hdr.enc_count && !pread_full(hdr.enc_off, enc_store.data(), enc_store.size() * sizeof(RadEncoding))) ||
            (hdr.prof_count && !pread_full(hdr.prof_off, profile_store.data(), profile_store.size() * sizeof(RadFileProfile))))
            return false;
        dir = dir_store.data();
        planes = plane_store.data();
        encs = enc_store.data();
        profile = profile_store.empty() ? nullptr : profile_store.data();
        int32_t max_layer = -1;
        for (uint64_t i = 0; i < hdr.dir_count; ++i)
            if (dir[i].layer >= 0 && dir[i].expert >= 0) max_layer = std::max(max_layer, dir[i].layer);
        layers = (uint32_t)(max_layer + 1);
        expert.resize((size_t)layers * kExperts);
        for (uint64_t i = 0; i < hdr.dir_count; ++i) {
            const auto& e = dir[i];
            if (e.layer < 0 || e.expert < 0 || e.expert >= (int)kExperts || e.slot < 0 || e.slot >= 2) continue;
            if ((uint32_t)e.layer >= layers || e.enc >= hdr.enc_count || e.plane_first + e.n_planes > hdr.plane_count) return false;
            for (uint32_t p = 0; p < e.n_planes; ++p)
                if (!range(planes[e.plane_first + p].offset, planes[e.plane_first + p].bytes)) return false;
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
    struct CacheRow { uint64_t key = ~0ull; std::vector<uint8_t> bytes; };
    thread_local std::array<CacheRow, 16> cache;
    const uint64_t key = fp->offset + (uint64_t)rr * (uint64_t)rb;
    CacheRow& cr = cache[(size_t)((key >> 6) & (cache.size() - 1))];
    if (cr.key != key || cr.bytes.size() != (size_t)rb) {
        cr.bytes.resize((size_t)rb);
        if (!f.pread_full(key, cr.bytes.data(), cr.bytes.size())) return NAN;
        cr.key = key;
    }
    return raw_value(ep.dtype, cr.bytes.data(), cc);
}

bool decode_row(const Container& f, const RadFileEntry* e, int64_t row, std::vector<float>& out,
                bool undo_fwht = true) {
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
    if (undo_fwht && fw) for (int64_t b=0;b<nc;b+=fw) for(int64_t d=1;d<fw;d<<=1)
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


/* Host emulation of Radiance's w4nl64a8h path. The container stores Hadamard-domain expert
 * weights (including the transform's residual scale); unlike decode_row's fp32 reference this
 * path deliberately does not invert that transform. */
void fwht128(float* v) {
    for (int d = 1; d < 128; d <<= 1)
        for (int i = 0; i < 128; i += 2 * d)
            for (int j = 0; j < d; ++j) {
                const float a = v[i + j], b = v[i + j + d];
                v[i + j] = a + b;
                v[i + j + d] = a - b;
            }
}

float e4m3_round(float x) {
    if (!std::isfinite(x)) return x;
    const bool neg = std::signbit(x);
    const float a = std::min(std::abs(x), 448.0f);
    uint8_t best = 0;
    float best_d = std::numeric_limits<float>::infinity();
    for (uint32_t c = 0; c <= 0x7eu; ++c) {
        const float v = f8e4m3((uint8_t)c);
        const float d = std::abs(v - a);
        if (d < best_d || (d == best_d && ((c & 1u) == 0u) && (best & 1u))) {
            best = (uint8_t)c;
            best_d = d;
        }
    }
    return f8e4m3((uint8_t)(best | (neg ? 0x80u : 0u)));
}

bool quant_had_engine(float* v, size_t n) {
    if (!v || n == 0 || n % 128 != 0) return false;
    for (size_t b = 0; b < n; b += 128) {
        fwht128(v + b);
        float m = 0.0f;
        for (size_t j = 0; j < 128; ++j) m = std::max(m, std::abs(v[b + j]));
        const float sc = m > 0.0f ? m * (1.0f / 448.0f) : 1.0f;
        const float inv = 1.0f / sc;
        for (size_t j = 0; j < 128; ++j) {
            const float q = std::max(-448.0f, std::min(448.0f, v[b + j] * inv));
            v[b + j] = e4m3_round(q) * sc;
        }
    }
    return true;
}

float dot_f32(const float* x, const std::vector<float>& w) {
    float s = 0.0f;
    for (size_t i = 0; i < w.size(); ++i) s += x[i] * w[i];
    return s;
}

int compute_host_engine(const Container& f, void* map, Request* q,
                        std::vector<uint16_t>* route_down = nullptr) {
    if (q->rank_count != kMaxRanks || q->rank >= kMaxRanks || q->layer >= f.layers ||
        q->rows == 0 || q->rows > kMaxRows || q->hidden != kHidden ||
        q->route_count > kMaxRoutes || q->activation_stride != kHidden * 2)
        return -EINVAL;

    const Route* routes = ptr<Route>(map, q->route_offset);
    const uint16_t* act = ptr<uint16_t>(map, q->activation_offset);
    uint16_t* dst = ptr<uint16_t>(map, q->result_offset[q->rank]);

    std::vector<float> aq((size_t)q->rows * kHidden);
    for (size_t i = 0; i < aq.size(); ++i) aq[i] = bf16(act[i]);
    for (uint32_t t = 0; t < q->rows; ++t)
        if (!quant_had_engine(aq.data() + (size_t)t * kHidden, kHidden)) return -EINVAL;

    std::array<std::vector<uint32_t>, kExperts> by_exp;
    std::vector<int32_t> slot_route((size_t)q->rows * kTopK, -1);
    for (uint32_t i = 0; i < q->route_count; ++i) {
        const Route& r = routes[i];
        if (r.token < 0 || (uint32_t)r.token >= q->rows || r.expert < 0 ||
            r.expert >= (int32_t)kExperts || r.slot >= kTopK)
            return -EINVAL;
        const size_t fslot = (size_t)r.token * kTopK + r.slot;
        if (slot_route[fslot] >= 0) return -EINVAL;
        slot_route[fslot] = (int32_t)i;
        by_exp[(uint32_t)r.expert].push_back(i);
    }

    std::vector<uint16_t> down((size_t)q->route_count * kHidden, 0);
    std::vector<float> wg, wu, wd;
    for (uint32_t e = 0; e < kExperts; ++e) {
        const auto& ri = by_exp[e];
        if (ri.empty()) continue;
        const RadFileEntry* gu = f.weight(q->layer, e, 0);
        const RadFileEntry* dn = f.weight(q->layer, e, 1);
        if (!gu || !dn) return -ENOENT;
        int64_t gur = 1, guc = 1, dnr = 1, dnc = 1;
        rad_enc_view(gu->rank, gu->shape, &gur, &guc);
        rad_enc_view(dn->rank, dn->shape, &dnr, &dnc);
        if (guc != kHidden || dnr != kHidden || gur != 2 * dnc || dnc != 640 ||
            rad_enc_fwht(f.encoding(gu)) != 128 || rad_enc_fwht(f.encoding(dn)) != 128)
            return -EINVAL;

        const int32_t es = layer_slot(map, q->layer, e);
        if (es < 0) return -ENOTSUP;
        const bool wide = q->rank == ((uint32_t)es & 1u);
        const int64_t lo = wide ? 0 : 384;
        const int64_t width = wide ? 384 : 256;
        std::vector<float> mid(ri.size() * 384u, 0.0f);

        for (int64_t j = 0; j < width; ++j) {
            if (!decode_row(f, gu, lo + j, wg, false) ||
                !decode_row(f, gu, dnc + lo + j, wu, false))
                return -ENOTSUP;
            for (size_t a = 0; a < ri.size(); ++a) {
                const Route& r = routes[ri[a]];
                const float* x = aq.data() + (size_t)r.token * kHidden;
                const float gate = bf16(to_bf16(dot_f32(x, wg)));
                const float up = bf16(to_bf16(dot_f32(x, wu)));
                mid[a * 384u + (size_t)j] =
                    bf16(to_bf16((gate / (1.0f + ::expf(-gate))) * up));
            }
        }
        for (size_t a = 0; a < ri.size(); ++a)
            if (!quant_had_engine(mid.data() + a * 384u, 384u)) return -EINVAL;

        for (int64_t h = 0; h < kHidden; ++h) {
            if (!decode_row(f, dn, h, wd, false)) return -ENOTSUP;
            for (size_t a = 0; a < ri.size(); ++a) {
                float sum = 0.0f;
                const float* m = mid.data() + a * 384u;
                for (int64_t j = 0; j < width; ++j) sum += m[j] * wd[(size_t)(lo + j)];
                down[(size_t)ri[a] * kHidden + (size_t)h] = to_bf16(sum);
            }
        }
    }

    if (route_down) *route_down = down;

    /* r4d_moe gather: ascending original top-k slot order, f32 accumulator, one BF16 narrow. */
    for (uint32_t t = 0; t < q->rows; ++t) {
        for (uint32_t h = 0; h < kHidden; ++h) {
            float acc = 0.0f;
            for (uint32_t slot = 0; slot < kTopK; ++slot) {
                const int32_t ri = slot_route[(size_t)t * kTopK + slot];
                if (ri < 0) continue;
                acc += bf16(routes[(uint32_t)ri].weight_bf16) *
                       bf16(down[(size_t)(uint32_t)ri * kHidden + h]);
            }
            dst[(size_t)t * kHidden + h] = to_bf16(acc);
        }
    }
    return 0;
}

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
        const int32_t es=layer_slot(map,q->layer,e); if(es<0) return -ENOTSUP;
        const int parity=(int)((uint32_t)es&1u); const bool wide=((int)q->rank==parity); const int64_t lo=wide?0:384, hi=wide?384:640, width=hi-lo;
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

uint32_t env_rows(const char* e, uint32_t fallback) {
    if (!e || !*e) return fallback;
    char* end = nullptr;
    errno = 0;
    const unsigned long v = std::strtoul(e, &end, 10);
    if (errno || !end || *end || v == 0) return fallback;
    return (uint32_t)std::min<unsigned long>(v, kMaxRows);
}
/* The largest step handed off (the engine reads the same variable) and the small, pre-queued lane size. */
uint32_t helper_max_rows() { return env_rows(std::getenv("R3_MOE_HELPER_MAX_ROWS"), 64); }
uint32_t helper_small_rows() { return std::min(env_rows(std::getenv("R3_MOE_HELPER_SMALL_ROWS"), 64), helper_max_rows()); }

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

bool select_owned(const Container& f, r3_moe_gpu::Runner* gpu, uint64_t budget, uint32_t helper,
                  uint32_t helpers, Owned* out) {
    if (!out || !budget) {
        std::fprintf(stderr, "r3_moe_helper: no helper budget\n");
        return false;
    }
    std::vector<const RadFileProfile*> order;
    /* A container converted without calibration has no profile: take experts expert-major (unsorted)
     * so the budget spreads over all layers. */
    std::vector<RadFileProfile> flat;
    /* R3_MOE_HELPER_HEAT: route counts a profiling run wrote (R3_MOE_HELPER_HISTOGRAM). Coldest first, so the
     * engine keeps the experts most tokens route to and a decode step hands off as little as possible. */
    if (const char* hp = std::getenv("R3_MOE_HELPER_HEAT"); hp && *hp) {
        std::vector<uint32_t> heat((size_t)kMaxLayers * kExperts, 0);
        FILE* hf = std::fopen(hp, "rb");
        const size_t got = hf ? std::fread(heat.data(), sizeof(uint32_t), heat.size(), hf) : 0;
        if (hf) std::fclose(hf);
        if (got != heat.size()) {
            std::fprintf(stderr, "r3_moe_helper: R3_MOE_HELPER_HEAT %s: want %zu counts, read %zu\n",
                         hp, heat.size(), got);
            return false;
        }
        for (uint32_t l = 0; l < f.layers; ++l)
            for (uint32_t e = 0; e < kExperts; ++e)
                if (f.weight(l, e, 0))
                    flat.push_back(RadFileProfile{(int32_t)l, (int32_t)e, (float)heat[(size_t)l * kExperts + e], 0.0f});
        std::stable_sort(flat.begin(), flat.end(), [](const RadFileProfile& a, const RadFileProfile& b) {
            return a.share < b.share;
        });
        std::fprintf(stderr, "r3_moe_helper: owning the coldest experts of %s first\n", hp);
        for (const RadFileProfile& p : flat) order.push_back(&p);
    } else if (!f.profile || !f.h->prof_count) {
        std::fprintf(stderr, "r3_moe_helper: container has no expert profile; owning experts evenly across layers\n");
        /* Highest ids first: the engine's initial placement keeps low ids resident on the XTX and
         * leaves the high ids in its host pool (measured: every host-pool route was expert >= 302). */
        for (uint32_t e = kExperts; e-- > 0;)
            for (uint32_t l = 0; l < f.layers; ++l)
                if (f.weight(l, e, 0)) flat.push_back(RadFileProfile{(int32_t)l, (int32_t)e, 0.0f, 0.0f});
        for (const RadFileProfile& p : flat) order.push_back(&p);
    }
    for (uint64_t i = 0; flat.empty() && f.profile && i < f.h->prof_count; ++i) {
        const RadFileProfile& p = f.profile[i];
        if (p.layer < 0 || p.expert < 0 || (uint32_t)p.layer >= f.layers ||
            p.expert >= (int32_t)kExperts) continue;
        order.push_back(&p);
    }
    if (flat.empty()) std::stable_sort(order.begin(), order.end(), [](const RadFileProfile* a, const RadFileProfile* b) {
        if (a->share != b->share) return a->share < b->share;
        if (a->layer != b->layer) return a->layer < b->layer;
        return a->expert < b->expert;
    });
    std::vector<uint8_t> seen((size_t)f.layers * kExperts, 0);
    uint64_t position = 0;
    for (const RadFileProfile* p : order) {
        const uint32_t l = (uint32_t)p->layer, e = (uint32_t)p->expert;
        uint8_t& was = seen[(size_t)l * kExperts + e];
        if (was) continue;
        was = 1;
        /* Helper h of n takes every n-th candidate: disjoint sets, each spread over the same order. */
        if (position++ % helpers != helper) continue;
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

void publish_owned(void* map, const Owned& owned, uint32_t helper) {
    Header* h = static_cast<Header*>(map);
    __atomic_fetch_add(&h->owned_bytes, owned.bytes, __ATOMIC_ACQ_REL);
    for (auto [l, e] : owned.expert) {
        if (!set_owned(map, l, e, helper))
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
                 uint32_t layer, uint32_t rows, uint32_t lane_max_rows, uint32_t lane_large_rows,
                 bool synthetic_protect) {
    if (layer >= model.layers || rows == 0 || rows > lane_large_rows) {
        std::fprintf(stderr, "r3_moe_helper: --compare rows %u above R3_MOE_HELPER_MAX_ROWS %u\n",
                     rows, lane_large_rows);
        return 2;
    }
    std::array<int32_t, kExperts> runtime_slot{};
    runtime_slot.fill(-1);
    static constexpr uint32_t kSyntheticProtected[8] = {1, 17, 63, 127, 191, 255, 319, 511};
    uint32_t next_slot = 0;
    for (uint32_t e = 0; e < kExperts; ++e) {
        if (!model.weight(layer, e, 0) || !model.weight(layer, e, 1)) continue;
        bool protect = false;
        if (synthetic_protect)
            for (uint32_t p : kSyntheticProtected) protect = protect || e == p;
        if (!protect) runtime_slot[e] = (int32_t)next_slot++;
    }

    std::vector<uint32_t> remote_expert, local_expert;
    std::vector<uint8_t> owned_here(kExperts, 0);
    for (auto [l, e] : owned.expert) {
        if (l != layer || runtime_slot[e] < 0) continue;
        remote_expert.push_back(e);
        owned_here[e] = 1;
    }
    for (uint32_t e = 0; e < kExperts; ++e)
        if (!owned_here[e] && model.weight(layer, e, 0) && model.weight(layer, e, 1))
            local_expert.push_back(e);
    if (remote_expert.size() < 2 || local_expert.empty()) {
        std::fprintf(stderr,
                     "r3_moe_helper: --compare layer %u needs >=2 owned and >=1 local experts\n",
                     layer);
        return 1;
    }
    if (synthetic_protect) {
        bool shifted_parity = false;
        for (uint32_t e : remote_expert)
            shifted_parity = shifted_parity ||
                (((uint32_t)runtime_slot[e] & 1u) != (e & 1u));
        if (!shifted_parity) {
            std::fprintf(stderr,
                         "r3_moe_helper: --compare synthetic protection did not shift an owned TP class\n");
            return 1;
        }
    }

    const uint32_t remote_per_token =
        std::min<uint32_t>(4, (uint32_t)remote_expert.size());
    const uint32_t total = rows * kTopK;
    std::vector<int32_t> full_expert(total, -1);
    std::vector<uint16_t> full_weight(total, 0);
    std::vector<uint8_t> is_remote(total, 0);
    std::vector<uint16_t> activation((size_t)rows * kHidden);

    uint64_t rng = 0x8f3f73b5cf1c9ad1ull ^ ((uint64_t)layer << 32) ^ rows;
    auto rnd = [&]() -> uint32_t {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        return (uint32_t)(rng >> 16);
    };
    for (size_t i = 0; i < activation.size(); ++i) {
        const float x = ((int32_t)(rnd() & 0xffffu) - 32768) * (1.0f / 32768.0f);
        activation[i] = to_bf16(x);
    }

    std::vector<Route> published;
    published.reserve((size_t)rows * remote_per_token);
    for (uint32_t t = 0; t < rows; ++t) {
        uint32_t ri = 0, li = 0;
        for (uint32_t slot = 0; slot < kTopK; ++slot) {
            const uint32_t flat = t * kTopK + slot;
            const bool remote = ((slot + 3 * t) % kTopK) < remote_per_token;
            const float w = (float)(slot + 1) / 55.0f;
            full_weight[flat] = to_bf16(w);
            if (remote) {
                const uint32_t e = remote_expert[(t * 5u + ri) % remote_expert.size()];
                full_expert[flat] = (int32_t)e;
                is_remote[flat] = 1;
                published.push_back(Route{(int32_t)t, (int32_t)e, full_weight[flat],
                                          (uint16_t)slot});
                ++ri;
            } else {
                const uint32_t e = local_expert[(t * 7u + li) % local_expert.size()];
                full_expert[flat] = (int32_t)e;
                ++li;
            }
        }
    }
    const uint32_t nr = (uint32_t)published.size();

    const Layout lay = layout();
    void* map = mmap(nullptr, lay.bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (map == MAP_FAILED) {
        std::fprintf(stderr, "r3_moe_helper: --compare mmap: %s\n", std::strerror(errno));
        return 1;
    }
    init(map, model.layers);
    publish_owned(map, owned, 0);
    Header* h = static_cast<Header*>(map);
    h->helper_count = 1;
    std::memcpy(layer_slots(map, layer), runtime_slot.data(), sizeof(runtime_slot));
    store(&h->layer_map_ready[layer], 1);
    if (!gpu.load_layer(layer, owned.expert, runtime_slot.data())) {
        std::fprintf(stderr, "r3_moe_helper: --compare layer %u runtime table build failed\n", layer);
        munmap(map, lay.bytes);
        return 1;
    }
    store(&h->layer_map_consumed[layer], 1);
    h->small_rows = lane_max_rows;
    if (!gpu.prepare_lanes(lane_max_rows, lane_large_rows)) {
        munmap(map, lay.bytes);
        return 1;
    }
    if (synthetic_protect)
        std::printf("synthetic_protect layer %u removed=8 regular=%u\n", layer, next_slot);
    bool pinned = false;
    void* dev_map = nullptr;
    if (!r3_moe_gpu::hip_ok(
            hipHostRegister(map, lay.bytes, hipHostRegisterPortable | hipHostRegisterMapped),
            "--compare hipHostRegister")) {
        munmap(map, lay.bytes);
        return 1;
    }
    pinned = true;
    if (!r3_moe_gpu::hip_ok(hipHostGetDevicePointer(&dev_map, map, 0),
                            "--compare hipHostGetDevicePointer")) {
        (void)hipHostUnregister(map);
        munmap(map, lay.bytes);
        return 1;
    }

    bool bad = false;
    constexpr uint32_t sequence = 1;
    for (uint32_t rank = 0; rank < kMaxRanks; ++rank) {
        const uint32_t lane = lane_of(rank, rows > lane_max_rows ? 1u : 0u);
        Request* q = lane_request(map, lane, sequence);
        q->completion_sequence = 0;
        q->status = 0;
        q->layer = layer;
        q->rank = rank;
        q->rank_count = kMaxRanks;
        q->rows = rows;
        q->hidden = kHidden;
        q->route_count = nr;
        q->row_base = 0;
        q->route_offset = h->route_offset[rank];
        q->activation_offset = h->activation_offset[rank];
        q->activation_q_offset = h->activation_offset[rank] + kActivationBf16Bytes;
        q->activation_s_offset =
            h->activation_offset[rank] + kActivationBf16Bytes + kActivationQBytes;
        q->activation_stride = kHidden * sizeof(uint16_t);
        q->activation_q_stride = kHidden;
        q->activation_s_stride = (kHidden / kActGroup) * sizeof(float);
        for (uint32_t r = 0; r < kMaxRanks; ++r) {
            q->result_offset[r] = h->result_offset[rank * kMaxRanks + r];
            q->result_stride[r] = kHidden * sizeof(uint16_t);
        }
        q->sequence = sequence;
        std::memcpy(ptr<Route>(map, q->route_offset), published.data(),
                    published.size() * sizeof(Route));
        std::memcpy(ptr<uint16_t>(map, q->activation_offset), activation.data(),
                    activation.size() * sizeof(uint16_t));
        if (!gpu.stage_compare_activation(map, lane, activation.data(), rows)) {
            std::fprintf(stderr,
                         "r3_moe_helper: --compare rank %u could not stage exact q/s\n", rank);
            bad = true;
            break;
        }

        /* Queue the production lane before publishing the sequence, as the service worker does. */
        const int erc = gpu.enqueue_lane(map, dev_map, lane, sequence,
                                         lane < kMaxRanks ? lane_max_rows : rows, false);
        if (erc != 0) {
            std::fprintf(stderr, "r3_moe_helper: --compare lane rank %u enqueue failed rc=%d\n",
                         rank, erc);
            bad = true;
            break;
        }
        store(&h->lane[lane].request_sequence, sequence);
        if (gpu.sync_lane(lane)) gpu.gemv_debug_report(lane);
        if (!gpu.sync_lane(lane) || load(&h->lane[lane].done_sequence) != sequence || q->status != 0) {
            std::fprintf(stderr,
                         "r3_moe_helper: --compare lane rank %u completion failed status=%d done=%u\n",
                         rank, q->status, load(&h->lane[lane].done_sequence));
            bad = true;
            break;
        }
        /* The helper's per-token rows, taken before the host reference rewrites the same slot. */
        const uint16_t* res = ptr<uint16_t>(map, h->result_offset[rank * kMaxRanks + rank]);
        const std::vector<uint16_t> got(res, res + (size_t)rows * kHidden);

        std::vector<uint16_t> want_route;
        const int hrc = compute_host_engine(model, map, q, &want_route);
        if (hrc != 0 || want_route.size() != (size_t)nr * kHidden) {
            std::fprintf(stderr,
                         "r3_moe_helper: --compare local-path host rank %u failed rc=%d\n",
                         rank, hrc);
            bad = true;
            break;
        }
        /* Host reference per token: the same slot-order f32 sum over this request's routes. */
        std::vector<int32_t> route_of((size_t)rows * kTopK, -1);
        for (uint32_t j = 0; j < nr; ++j)
            route_of[(size_t)published[j].token * kTopK + published[j].slot] = (int32_t)j;
        long double num = 0.0L, den = 0.0L;
        double max_abs = 0.0;
        uint32_t tokens = 0;
        for (uint32_t t = 0; t < rows; ++t) {
            bool any = false;
            for (uint32_t slot = 0; slot < kTopK; ++slot) any = any || route_of[(size_t)t * kTopK + slot] >= 0;
            if (!any) continue;
            ++tokens;
            for (uint32_t col = 0; col < kHidden; ++col) {
                float acc = 0.0f;
                for (uint32_t slot = 0; slot < kTopK; ++slot) {
                    const int32_t j = route_of[(size_t)t * kTopK + slot];
                    if (j < 0) continue;
                    acc += bf16(published[(uint32_t)j].weight_bf16) *
                           bf16(want_route[(size_t)(uint32_t)j * kHidden + col]);
                }
                const double b = bf16(to_bf16(acc));
                const double a = bf16(got[(size_t)t * kHidden + col]);
                const double d = a - b;
                num += (long double)d * d;
                den += (long double)b * b;
                max_abs = std::max(max_abs, std::abs(d));
            }
        }
        const double rel =
            std::sqrt((double)(num / std::max<long double>(den, 1.0e-30L)));
        std::printf("engine_join layer %u rank %u lane %u routes=%u tokens=%u rel_l2=%.8g max_abs=%.8g\n",
                    layer, rank, lane, nr, tokens, rel, max_abs);
        bad = bad || tokens == 0 || !std::isfinite(rel) || rel >= 5.0e-3;
    }

    if (pinned) (void)hipHostUnregister(map);
    munmap(map, lay.bytes);
    return bad ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    /* EVERY LANE STREAM ON ITS OWN HARDWARE QUEUE. A small lane pre-queues "wait for the next request" packets;
     * a stream sharing a hardware queue with one sits behind that wait until a small request arrives. With
     * every step large (KL mode's 64-token steps over an 8-row small lane) none ever does, and the large lane's
     * work never runs. HIP maps streams onto GPU_MAX_HW_QUEUES queues (4 by default) and the helper opens five
     * streams (four lanes and its own), so it asks for more before HIP initialises; an explicit setting wins. */
    (void)::setenv("GPU_MAX_HW_QUEUES", "8", 0);
    /* The helper's grouped MoE GEMM (prefill) on gfx11's native bf16 WMMA path: R9700 engine + two XTX helpers,
     * Flash-Next, prefill 2,518 -> 2,665 tok/s and helper gate/up -25%, KLD 0.064 (noise floor). Read by libr3
     * when the plugin is loaded, so set before it; an explicit setting wins. */
    (void)::setenv("R3_MOE_NATIVE", "1", 0);
    bool compare =argc > 1 && std::strcmp(argv[1], "--compare") == 0;
    bool synthetic_protect = compare && argc > 2 &&
        std::strcmp(argv[argc - 1], "--synthetic-protect") == 0;
    const int carg = argc - (synthetic_protect ? 1 : 0);
    const char* model_path = nullptr;
    uint32_t compare_layer = 0, compare_rows = 0;
    bool compare_one_layer = false;
    if (compare) {
        if (carg == 5 && std::strcmp(argv[3], "--rows") == 0) {
            if (!parse_u32(argv[4], 1, kMaxRows, &compare_rows)) {
                std::fprintf(stderr, "usage: r3_moe_helper --compare MODEL.rad --rows M\n");
                return 2;
            }
        } else if (carg == 7 && std::strcmp(argv[3], "--layer") == 0 &&
                   std::strcmp(argv[5], "--rows") == 0 &&
                   parse_u32(argv[4], 0, kMaxLayers - 1, &compare_layer) &&
                   parse_u32(argv[6], 1, kMaxRows, &compare_rows)) {
            compare_one_layer = true;
        } else {
            std::fprintf(stderr,
                "usage: r3_moe_helper --compare MODEL.rad [--layer L] --rows M [--synthetic-protect]\n");
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

    const char* cpu_env = std::getenv("R3_MOE_HELPER_CPU");
    const bool cpu = cpu_env && *cpu_env && std::strcmp(cpu_env, "0") != 0 && !compare;
    std::unique_ptr<r3_moe_gpu::Runner> gpu;
    if (!cpu) {
        gpu = std::make_unique<r3_moe_gpu::Runner>(model);
        if (const char* e = std::getenv("R3_MOE_HELPER_ENGINE_RANKS"); e && *e) {
            const int n = std::atoi(e);
            if (n < 1 || n > (int)kMaxRanks) {
                std::fprintf(stderr, "r3_moe_helper: R3_MOE_HELPER_ENGINE_RANKS=%s, want 1..%u\n", e, kMaxRanks);
                return 2;
            }
            gpu->engine_ranks_ = (uint32_t)n;
        }
        if (!gpu->open()) return 1;
    }

    /* R3_MOE_HELPER_COUNT=2 with R3_MOE_HELPER_INDEX=0/1: two helpers (the XTX cards) behind a one-rank engine
     * (the R9700) share one segment; helper 0 creates it, helper 1 attaches, each owns half the candidates and
     * serves its own rank slot. */
    uint32_t helper_index = 0, helper_count = 1;
    if (const char* e = std::getenv("R3_MOE_HELPER_COUNT"); e && *e) helper_count = (uint32_t)std::atoi(e);
    if (const char* e = std::getenv("R3_MOE_HELPER_INDEX"); e && *e) helper_index = (uint32_t)std::atoi(e);
    if (helper_count < 1 || helper_count > kMaxRanks || helper_index >= helper_count ||
        (helper_count > 1 && (!gpu || gpu->engine_ranks_ != 1 || compare))) {
        std::fprintf(stderr, "r3_moe_helper: R3_MOE_HELPER_INDEX %u of R3_MOE_HELPER_COUNT %u: want index < count <= %u, "
                     "and two helpers only on GPU behind a one-rank engine (R3_MOE_HELPER_ENGINE_RANKS=1), not --compare\n",
                     helper_index, helper_count, kMaxRanks);
        return 2;
    }
    const uint64_t budget = helper_budget_bytes();
    Owned owned;
    if (!select_owned(model, gpu.get(), budget, helper_index, helper_count, &owned)) return 1;
    if (gpu && !gpu->allocate(owned.expert)) return 1;
    const uint32_t lane_max_rows = helper_small_rows();
    const uint32_t lane_large_rows = helper_max_rows();
    long rss_pages = 0;
    if (FILE* sf = std::fopen("/proc/self/statm", "r")) {
        long total_pages = 0;
        if (std::fscanf(sf, "%ld %ld", &total_pages, &rss_pages) != 2) rss_pages = 0;
        std::fclose(sf);
    }
    const double rss_mib = rss_pages > 0 ? (double)rss_pages * (double)sysconf(_SC_PAGESIZE) / (1024.0 * 1024.0) : -1.0;
    std::fprintf(stderr, "r3_moe_helper: selected %zu expert candidates, %.3f GiB; host RSS %.1f MiB\n",
                 owned.expert.size(), (double)owned.bytes / (1024.0 * 1024.0 * 1024.0), rss_mib);

    if (compare) {
        if (compare_one_layer)
            return compare_mode(model, *gpu, owned, compare_layer, compare_rows, lane_max_rows,
                                lane_large_rows, synthetic_protect);
        int rc = 0;
        for (uint32_t layer : {0u, 20u, 34u, 47u}) {
            if (layer >= model.layers) {
                std::fprintf(stderr, "r3_moe_helper: --compare layer %u unavailable (layers=%u)\n",
                             layer, model.layers);
                rc = 1;
                continue;
            }
            std::printf("layer %u\n", layer);
            rc |= compare_mode(model, *gpu, owned, layer, compare_rows, lane_max_rows,
                               lane_large_rows, synthetic_protect);
        }
        return rc;
    }

    const Layout lay = layout();
    const char* name = shm_name();
    const bool creator = helper_index == 0;
    /* Only the creator removes the segment. */
    auto drop_name = [&] { if (creator) shm_unlink(name); };
    int fd = -1;
    if (creator) {
        fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    } else {
        for (int i = 0; i < 1800 && fd < 0 && !g_stop; ++i) {
            fd = shm_open(name, O_RDWR | O_CLOEXEC, 0);
            if (fd < 0) ::sleep(1);
        }
    }
    if (fd < 0) {
        std::fprintf(stderr, "r3_moe_helper: shm_open %s: %s\n", name, std::strerror(errno));
        return 1;
    }
    if (creator && ftruncate(fd, (off_t)lay.bytes) != 0) {
        std::fprintf(stderr, "r3_moe_helper: ftruncate: %s\n", std::strerror(errno));
        drop_name(); close(fd); return 1;
    }
    void* map = mmap(nullptr, lay.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        std::fprintf(stderr, "r3_moe_helper: mmap: %s\n", std::strerror(errno));
        drop_name(); close(fd); return 1;
    }
    auto* h = static_cast<Header*>(map);
    if (creator) {
        init(map, model.layers);
        h->small_rows = lane_max_rows;
        /* Every helper reads the same step activations: stage them once, in slot 0. */
        for (uint32_t r = 1; r < helper_count; ++r) h->activation_offset[r] = h->activation_offset[0];
        store(&h->helper_count, helper_count);   /* last: an attaching helper waits for it */
    } else {
        for (int i = 0; i < 1800 && !g_stop && load(&h->helper_count) != helper_count; ++i) ::sleep(1);
        if (load(&h->helper_count) != helper_count || !valid(h, lay.bytes) || h->small_rows != lane_max_rows) {
            std::fprintf(stderr, "r3_moe_helper: segment %s is not helper 0's %u-helper segment (small rows %u, ours %u)\n",
                         name, helper_count, h->small_rows, lane_max_rows);
            munmap(map, lay.bytes); close(fd); return 1;
        }
    }
    publish_owned(map, owned, helper_index);
    void* dev_map = nullptr;
    if (!cpu) {
        if (!r3_moe_gpu::hip_ok(hipHostRegister(map, lay.bytes, hipHostRegisterPortable | hipHostRegisterMapped),
                                "hipHostRegister mailbox") ||
            !r3_moe_gpu::hip_ok(hipHostGetDevicePointer(&dev_map, map, 0), "hipHostGetDevicePointer mailbox")) {
            munmap(map, lay.bytes); close(fd); drop_name(); return 1;
        }
    }

    std::fprintf(stderr, "r3_moe_helper: waiting for engine runtime expert maps\n");
    /* The first map waits as long as the engine takes to start. After that the engine declares its
     * layers back to back, so a layer whose map has not come within 10 s is one this engine never
     * declares (the MTP layer in a run without speculation, e.g. KL mode): drop its ownership. */
    bool any_map = false;
    for (uint32_t layer = 0; layer < model.layers; ++layer) {
        while (!load(&h->layer_map_ready[layer])) {
            if (!any_map) {
                (void)futex_wait(&h->layer_map_ready[layer], 0);
                continue;
            }
            struct timespec ts{10, 0};
            (void)::syscall(SYS_futex, &h->layer_map_ready[layer], FUTEX_WAIT, 0, &ts, nullptr, 0);
            if (!load(&h->layer_map_ready[layer])) break;
        }
        if (!load(&h->layer_map_ready[layer])) {
            std::fprintf(stderr, "r3_moe_helper: layer %u has no runtime map (not declared by the engine); not owned\n",
                         layer);
            for (auto [l, e] : owned.expert)
                if (l == layer) (void)clear_owned(map, l, e, helper_index);
            continue;
        }
        any_map = true;
        const int32_t* slots = layer_slots(map, layer);
        if (!slots) {
            munmap(map, lay.bytes); close(fd); drop_name(); return 1;
        }
        for (auto [l, e] : owned.expert)
            if (l == layer && slots[e] < 0) (void)clear_owned(map, l, e, helper_index);
        if (gpu && !gpu->load_layer(layer, owned.expert, slots)) {
            std::fprintf(stderr, "r3_moe_helper: layer %u runtime table build failed\n", layer);
            store(&h->layer_map_consumed[layer], 1);
            (void)futex_wake(&h->layer_map_consumed[layer]);
            munmap(map, lay.bytes); close(fd); drop_name(); return 1;
        }
        store(&h->layer_map_consumed[layer], 1);
        (void)futex_wake(&h->layer_map_consumed[layer]);
    }
    Owned runtime_owned;
    for (auto [l, e] : owned.expert) {
        if (!load(&h->layer_map_ready[l]) || layer_slot(map, l, e) < 0) continue;
        runtime_owned.expert.emplace_back(l, e);
        const int64_t eb = gpu ? gpu->expert_bytes(l, e) : canonical_expert_bytes(model, l, e);
        if (eb > 0) runtime_owned.bytes += (uint64_t)eb;
    }
    owned = std::move(runtime_owned);
    h->owned_bytes = owned.bytes;
    if (gpu && !gpu->prepare_lanes(lane_max_rows, lane_large_rows)) {
        (void)hipHostUnregister(map);
        munmap(map, lay.bytes); close(fd); drop_name(); return 1;
    }
    if (gpu) (void)posix_fadvise(model.fd, 0, 0, POSIX_FADV_DONTNEED);
    std::fprintf(stderr, "r3_moe_helper: runtime map owns %zu experts, %.3f GiB\n",
                 owned.expert.size(), (double)owned.bytes / (1024.0 * 1024.0 * 1024.0));

    /* No SA_RESTART (std::signal sets it): the futex wait must return EINTR so the loop sees g_stop. */
    struct sigaction sa{};
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr); sigaction(SIGTERM, &sa, nullptr);
    std::fprintf(stderr,
                 "r3_moe_helper: ready shm=%s layers=%u bytes=%llu (%s, small lane rows <= %u, large <= %u)\n",
                 name, model.layers, (unsigned long long)lay.bytes,
                 cpu ? "host fp32 reference" : gpu->arch().c_str(), lane_max_rows, lane_large_rows);

    uint64_t stat_req = 0, stat_rows = 0, stat_routes = 0;
    double stat_us = 0.0;
    if (cpu) {
        uint32_t seq = 1;
        while (!g_stop) {
            Request* q = request(map, seq);
            while (!g_stop && load(&q->sequence) != seq) {
                const uint32_t seen = load(&h->publish_sequence);
                if (load(&q->sequence) != seq) (void)futex_wait(&h->publish_sequence, seen);
            }
            if (g_stop) break;
            const auto t0 = std::chrono::steady_clock::now();
            q->status = compute_host(model, map, q);
            stat_us += std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count();
            ++stat_req; stat_rows += q->rows; stat_routes += q->route_count;
            store(&q->completion_sequence, seq);
            (void)futex_wake(&q->completion_sequence);
            if (++seq == 0) break;
        }
    } else {
        constexpr uint32_t kQueueDepth = 4;
        std::array<uint64_t, kLanes> lane_req{}, lane_rows{}, lane_routes{}, lane_samples{};
        std::array<double, kLanes> lane_sample_us{};
        std::array<std::thread, kLanes> workers;
        for (uint32_t lane = 0; lane < kLanes; ++lane) {
            /* Two helpers: each serves only its own rank slot's two lanes. */
            if (helper_count > 1 && lane % kMaxRanks != helper_index) continue;
            workers[lane] = std::thread([&, lane] {
                const bool large = lane >= kMaxRanks;
                uint32_t next = 1;
                uint32_t done_seen = 0;
                if (large) {
                    /* Sized on arrival: wait for the engine's request word, read its rows, launch that size. */
                    while (!g_stop) {
                        if ((int32_t)(load(&h->lane[lane].request_sequence) - next) < 0) {
                            std::this_thread::yield();
                            continue;
                        }
                        const Request* q = lane_request(map, lane, next);
                        const uint32_t rows = load(&q->rows);
                        /* Every large request is timed by stage: they are few, and the engine cannot send
                         * the next one until it has merged this one, so waiting here costs nothing. */
                        const int rc = gpu->enqueue_lane(map, dev_map, lane, next, rows, true);
                        if (rc != 0) {
                            std::fprintf(stderr, "r3_moe_helper: large lane %u seq %u rows %u failed rc=%d\n",
                                         lane, next, rows, rc);
                            g_stop = 1;
                            break;
                        }
                        if (!gpu->sync_lane(lane) ||
                            !gpu->collect_lane_timing(lane, next, &lane_sample_us[lane], &lane_samples[lane])) {
                            g_stop = 1;
                            break;
                        }
                        if (++next == 0) break;
                    }
                } else {
                    auto queue_one = [&]() -> bool {
                        if (next == 0) return false;
                        const bool sample = (next & 63u) == 0;
                        const int rc = gpu->enqueue_lane(map, dev_map, lane, next, lane_max_rows, sample);
                        if (rc != 0) {
                            std::fprintf(stderr,
                                         "r3_moe_helper: lane %u enqueue seq %u failed rc=%d\n",
                                         lane, next, rc);
                            return false;
                        }
                        ++next;
                        return next != 0;
                    };

                    for (uint32_t i = 0; i < kQueueDepth && !g_stop; ++i) {
                        if (!queue_one()) {
                            g_stop = 1;
                            break;
                        }
                    }
                    while (!g_stop) {
                        const uint32_t done = load(&h->lane[lane].done_sequence);
                        if ((int32_t)(done - done_seen) <= 0) {
                            std::this_thread::yield();
                            continue;
                        }
                        if (!gpu->collect_lane_timing(lane, done,
                                                      &lane_sample_us[lane],
                                                      &lane_samples[lane])) {
                            g_stop = 1;
                            break;
                        }
                        done_seen = done;
                        while (!g_stop &&
                               (uint32_t)(next - done_seen - 1u) < kQueueDepth) {
                            if (!queue_one()) {
                                g_stop = 1;
                                break;
                            }
                        }
                    }
                    const uint32_t final_done = load(&h->lane[lane].done_sequence);
                    (void)gpu->collect_lane_timing(lane, final_done,
                                                   &lane_sample_us[lane],
                                                   &lane_samples[lane]);
                    /* Release the pre-queued waits so the stream drains. */
                    store(&h->lane[lane].request_sequence, next - 1u);
                }
                if (!gpu->sync_lane(lane)) {
                    g_stop = 1;
                    return;
                }
                uint64_t stats[3]{};
                if (!gpu->lane_stats(lane, stats)) {
                    g_stop = 1;
                    return;
                }
                lane_req[lane] = stats[0];
                lane_rows[lane] = stats[1];
                lane_routes[lane] = stats[2];
            });
        }
        while (!g_stop) ::sleep(1);
        for (auto& w : workers) if (w.joinable()) w.join();
        /* The lanes are gone: an engine still waiting on them must fail rather than hang. */
        store(&h->helper_fault, 1u);
        uint64_t timing_samples = 0;
        double sampled_us = 0.0;
        for (uint32_t lane = 0; lane < kLanes; ++lane) {
            stat_req += lane_req[lane];
            stat_rows += lane_rows[lane];
            stat_routes += lane_routes[lane];
            timing_samples += lane_samples[lane];
            sampled_us += lane_sample_us[lane];
            gpu->stage_report(lane, lane_samples[lane]);
        }
        std::fprintf(stderr, "r3_moe_helper: large-lane requests %llu (%llu rows)\n",
                     (unsigned long long)(lane_req[kMaxRanks] + lane_req[kMaxRanks + 1]),
                     (unsigned long long)(lane_rows[kMaxRanks] + lane_rows[kMaxRanks + 1]));
        if (timing_samples && stat_req)
            stat_us = (sampled_us / (double)timing_samples) * (double)stat_req;
        std::fprintf(stderr,
                     "r3_moe_helper: GPU timing samples=%llu sampled=%.1f ms avg=%.1f us\n",
                     (unsigned long long)timing_samples, sampled_us / 1000.0,
                     timing_samples ? sampled_us / (double)timing_samples : 0.0);

    }

    std::fprintf(stderr, "r3_moe_helper: served %llu requests, %llu rows, %llu routes, compute %.1f ms (%.1f us a request)\n",
                 (unsigned long long)stat_req, (unsigned long long)stat_rows, (unsigned long long)stat_routes,
                 stat_us / 1000.0, stat_req ? stat_us / (double)stat_req : 0.0);
    if (!cpu) (void)hipHostUnregister(map);
    munmap(map, lay.bytes);
    close(fd);
    drop_name();
    return 0;
}
