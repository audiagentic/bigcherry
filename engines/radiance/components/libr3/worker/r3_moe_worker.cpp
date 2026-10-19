#include "moe_backend.h"
#include "moe_remote_shm.h"
#include "rad_format.h"
#include "rad_encoding.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <memory>
#include <set>
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
    uint32_t experts = 0;      /* routed experts per layer: the highest expert id in the directory + 1 */
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
        int32_t max_layer = -1, max_expert = -1;
        for (uint64_t i = 0; i < hdr.dir_count; ++i)
            if (dir[i].layer >= 0 && dir[i].expert >= 0) {
                max_layer = std::max(max_layer, dir[i].layer);
                max_expert = std::max(max_expert, dir[i].expert);
            }
        layers = (uint32_t)(max_layer + 1);
        experts = (uint32_t)(max_expert + 1);
        if (!layers || !experts || (uint64_t)layers * experts > (1ull << 24)) return false;   /* a directory this large is not a model */
        expert.resize((size_t)layers * experts);
        for (uint64_t i = 0; i < hdr.dir_count; ++i) {
            const auto& e = dir[i];
            if (e.layer < 0 || e.expert < 0 || e.expert >= (int)experts || e.slot < 0 || e.slot >= 2) continue;
            if ((uint32_t)e.layer >= layers || e.enc >= hdr.enc_count || e.plane_first + e.n_planes > hdr.plane_count) return false;
            for (uint32_t p = 0; p < e.n_planes; ++p)
                if (!range(planes[e.plane_first + p].offset, planes[e.plane_first + p].bytes)) return false;
            expert[(size_t)e.layer * experts + (uint32_t)e.expert][e.slot] = &e;
        }
        return layers != 0;
    }
    const RadFileEntry* weight(uint32_t layer, uint32_t exp, uint32_t slot) const {
        if (layer >= layers || exp >= experts || slot >= 2) return nullptr;
        return expert[(size_t)layer * experts + exp][slot];
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

    std::vector<std::vector<uint32_t>> by_exp(f.experts);
    std::vector<int32_t> slot_route((size_t)q->rows * kTopK, -1);
    for (uint32_t i = 0; i < q->route_count; ++i) {
        const Route& r = routes[i];
        if (r.token < 0 || (uint32_t)r.token >= q->rows || r.expert < 0 ||
            r.expert >= (int32_t)f.experts || r.slot >= kTopK)
            return -EINVAL;
        const size_t fslot = (size_t)r.token * kTopK + r.slot;
        if (slot_route[fslot] >= 0) return -EINVAL;
        slot_route[fslot] = (int32_t)i;
        by_exp[(uint32_t)r.expert].push_back(i);
    }

    std::vector<uint16_t> down((size_t)q->route_count * kHidden, 0);
    std::vector<float> wg, wu, wd;
    for (uint32_t e = 0; e < f.experts; ++e) {
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
    std::vector<std::vector<uint32_t>> by_exp(f.experts);
    for(uint32_t i=0;i<q->route_count;++i) {
        if(routes[i].token<0 || (uint32_t)routes[i].token>=q->rows || routes[i].expert<0 || routes[i].expert>=(int)f.experts) return -EINVAL;
        by_exp[(uint32_t)routes[i].expert].push_back(i);
    }
    std::vector<float> wg, wu, wd;
    for(uint32_t e=0;e<f.experts;++e) {
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

const char* shm_name() { const char* e=std::getenv("R3_MOE_WORKER_SHM"); return e&&*e?e:"/r3-moe-helper"; }
}

#include "moe_remote_gpu.h"

namespace {


uint64_t helper_budget_bytes() {
    const char* e = std::getenv("R3_MOE_WORKER_MIB");
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
uint32_t helper_max_rows() { return env_rows(std::getenv("R3_MOE_WORKER_MAX_ROWS"), 64); }
uint32_t helper_small_rows() { return std::min(env_rows(std::getenv("R3_MOE_WORKER_SMALL_ROWS"), 64), helper_max_rows()); }

/* The pre-queued small-lane sizes (RAD12): R3_MOE_WORKER_SMALL_LANES="4,8,16,24" (up to kSmallClasses, strictly ascending,
 * each within the large-lane capacity), else the single size R3_MOE_WORKER_SMALL_ROWS gives (the production behaviour). */
struct SmallClasses {
    std::array<uint32_t, r3::moe_shm::kSmallClasses> rows{};
    uint32_t count = 0;
};
SmallClasses helper_small_classes() {
    SmallClasses sc;
    const char* list = std::getenv("R3_MOE_WORKER_SMALL_LANES");
    if (!list || !*list) {
        sc.rows[0] = helper_small_rows();
        sc.count = 1;
        return sc;
    }
    const uint32_t cap = helper_max_rows();
    const char* p = list;
    while (*p && sc.count < r3::moe_shm::kSmallClasses) {
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p || v <= 0 || (uint32_t)v > cap || (sc.count && (uint32_t)v <= sc.rows[sc.count - 1])) {
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_SMALL_LANES=%s must be up to %u ascending sizes within %u rows\n",
                         list, (unsigned)r3::moe_shm::kSmallClasses, cap);
            return SmallClasses{};
        }
        sc.rows[sc.count++] = (uint32_t)v;
        p = end;
        if (*p == ',') ++p;
        else break;
    }
    if (*p) {
        std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_SMALL_LANES=%s has more than %u sizes or trailing text\n", list,
                     (unsigned)r3::moe_shm::kSmallClasses);
        return SmallClasses{};
    }
    return sc;
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

/* What the container says an expert is (RCS09/RAD11): its weight format, scale type and tile, read from the encoding, never assumed.
 * W4NL64 = 4-bit codes through a lookup table with scales per 64 columns; FP8_E4M3_BLOCK = E4M3 codes with a scale per tile.
 * Anything else is MOE_FMT_NONE and is refused by name before any expert is laid out. */
uint32_t scale_dtype_of(uint32_t dtype) {
    switch (dtype) {
        case RAD_F8E4M3: return MOE_SCALE_E4M3;
        case RAD_F16: return MOE_SCALE_F16;
        case RAD_BF16: return MOE_SCALE_BF16;
        case RAD_F32: return MOE_SCALE_F32;
        default: return MOE_SCALE_NONE;
    }
}

bool describe_expert(const Container& f, uint32_t l, uint32_t e, MoeExpertDesc* out) {
    const RadFileEntry* gu = f.weight(l, e, 0);
    const RadFileEntry* dn = f.weight(l, e, 1);
    if (!gu || !dn || !out) return false;
    const RadEncoding* enc = f.encoding(gu);
    if (!enc || !rad_enc_valid(enc)) return false;
    MoeExpertDesc d{};
    const int codes = rad_enc_find(enc, "codes");
    const int table = rad_enc_find(enc, "table");
    const int scale0 = rad_enc_find_level(enc, "scale", 0);
    const int scale1 = rad_enc_find_level(enc, "scale", 1);
    if (codes >= 0 && scale0 >= 0) {
        const RadEncPlane& cp = enc->plane[codes];
        const RadEncPlane& sp = enc->plane[scale0];
        if (rad_dtype_bits(cp.dtype) == 4 && table >= 0) {
            /* the two scale levels relayout to one E4M3 byte per 64 columns (the helper checks this geometry at upload) */
            (void)scale1;
            d.weight = MoeWeightDesc{MOE_FMT_W4NL64, MOE_ACT_FP8_E4M3_GROUP, MOE_SCALE_E4M3, 0, (uint32_t)sp.block[1], 0};
        } else if (cp.dtype == RAD_F8E4M3 && table < 0) {
            d.weight = MoeWeightDesc{MOE_FMT_FP8_E4M3_BLOCK, MOE_ACT_BF16, scale_dtype_of(sp.dtype),
                                     (uint32_t)sp.block[0], (uint32_t)sp.block[1], 0};
        }
    }
    int64_t gr = 1, gc = 1, dr = 1, dc = 1;
    rad_enc_view(gu->rank, gu->shape, &gr, &gc);
    rad_enc_view(dn->rank, dn->shape, &dr, &dc);
    d.gu_rows = (uint64_t)gr; d.gu_cols = (uint64_t)gc;
    d.dn_rows = (uint64_t)dr; d.dn_cols = (uint64_t)dc;
    *out = d;
    return true;
}

/* The shape the container implies (RCS09): layers, routed experts, hidden, expert intermediate width. top_k is the engine's. */
bool container_shape(const Container& f, MoeShape* s) {
    uint32_t experts = 0;
    for (uint64_t i = 0; i < f.hdr.dir_count; ++i)
        if (f.dir[i].layer >= 0 && f.dir[i].expert >= 0) experts = std::max(experts, (uint32_t)f.dir[i].expert + 1);
    MoeExpertDesc d{};
    if (!experts || !describe_expert(f, 0, 0, &d)) return false;
    *s = MoeShape{f.layers, experts, (uint32_t)d.dn_rows, 0, (uint32_t)d.dn_cols, 0, 0};
    return true;
}

const char* format_name(uint32_t fmt) {
    return fmt == MOE_FMT_W4NL64 ? "w4nl64" : fmt == MOE_FMT_FP8_E4M3_BLOCK ? "fp8_e4m3_block" : "unrecognised";
}

/* Whether the gfx1100 backend can run this container's experts, and why not when it cannot. */
bool backend_accepts(const Container& f, const char* arch, bool verbose) {
    MoeShape shape{};
    MoeExpertDesc d{};
    if (!container_shape(f, &shape) || !describe_expert(f, 0, 0, &d)) {
        std::fprintf(stderr, "r3_moe_worker: the container's first expert could not be described\n");
        return false;
    }
    shape.top_k = 1;      // the container does not carry the engine's top-k; the check here is the format and geometry
    MoeDeviceCaps caps{};
    std::strncpy(caps.arch, arch, sizeof(caps.arch) - 1);
    caps.wave_size = 32;
    caps.has_wmma_f16 = std::strncmp(arch, "gfx11", 5) == 0;
    const int rc = r3::moe_backend::probe(r3::moe_backend::gfx1100_backend(), shape, d.weight, caps);
    if (verbose || rc != MOE_OK)
        std::fprintf(stderr,
                     "r3_moe_worker: container: %u layers, %u experts, hidden %u, expert width %u, weight format %s "
                     "(scale dtype %u, tile %ux%u), gfx1100 backend: %s\n",
                     shape.layers, shape.experts, shape.hidden, shape.intermediate, format_name(d.weight.format),
                     d.weight.scale_dtype, d.weight.block_n, d.weight.block_k,
                     rc == MOE_OK ? "supported" : rc == MOE_UNSUPPORTED_FORMAT ? "weight format not supported"
                     : rc == MOE_UNSUPPORTED_ARCH ? "architecture not supported" : "geometry not supported");
    return rc == MOE_OK;
}

/* Bytes one expert takes on the device. With a GPU this is the backend contract's size for the container's own description of the
 * expert (moe_backend.h), the same number for every weight format; without one it is the container's stored size. */
int64_t laid_out_bytes(const Container& f, r3_moe_gpu::Runner* gpu, uint32_t l, uint32_t e) {
    if (!gpu) return canonical_expert_bytes(f, l, e);
    MoeExpertDesc d{};
    if (!describe_expert(f, l, e, &d)) return -1;
    return r3::moe_backend::expert_bytes(d);
}

/* The shared-memory shape and layout for this container (RCS09-03): hidden, experts and expert width are the container's own; the
 * row capacity, top-k and activation group are the protocol's until the engine negotiates them. A shape other than the one the
 * execution paths take is refused here, before any segment is created, with the reason. */
bool segment_shape(const Container& f, r3::moe_shm::LayoutShape* shape, r3::moe_shm::Layout* lay) {
    MoeShape cs{};
    if (!container_shape(f, &cs)) {
        std::fprintf(stderr, "r3_moe_worker: the container's shape could not be read\n");
        return false;
    }
    *shape = r3::moe_shm::LayoutShape{cs.hidden, r3::moe_shm::kTopK, cs.experts, r3::moe_shm::kMaxRows,
                                      r3::moe_shm::kActGroup, cs.intermediate};
    if (*shape != r3::moe_shm::legacy_shape()) {
        std::fprintf(stderr,
                     "r3_moe_worker: this model (hidden %u, %u experts, expert width %u) is not executable yet: the lane kernels "
                     "still take hidden %u, %u experts and width %u (RCS09-04 to 07)\n",
                     shape->hidden, shape->experts, shape->intermediate, r3::moe_shm::kHidden, r3::moe_shm::kExperts,
                     r3::moe_shm::kLegacyIntermediate);
        return false;
    }
    if (!r3::moe_shm::layout_for(*shape, lay)) {
        std::fprintf(stderr, "r3_moe_worker: the shared-memory layout for this shape is invalid\n");
        return false;
    }
    return true;
}

struct Owned {
    std::vector<std::pair<uint32_t,uint32_t>> expert;
    uint64_t bytes = 0;
};

bool select_owned(const Container& f, r3_moe_gpu::Runner* gpu, uint64_t budget, uint32_t helper,
                  uint32_t helpers, Owned* out) {
    if (!out || !budget) {
        std::fprintf(stderr, "r3_moe_worker: no helper budget\n");
        return false;
    }
    if (!backend_accepts(f, gpu ? gpu->arch().c_str() : "gfx1100", false)) return false;
    if (gpu) {   // the contract's size must be the size the device layout really takes, or no budget arithmetic can be trusted
        const int64_t want = laid_out_bytes(f, gpu, 0, 0), got = gpu->expert_bytes(0, 0);
        if (want != got) {
            std::fprintf(stderr, "r3_moe_worker: expert size disagrees with the device layout (contract %lld, layout %lld)\n",
                         (long long)want, (long long)got);
            return false;
        }
    }
    std::vector<const RadFileProfile*> order;
    /* A container converted without calibration has no profile: take experts expert-major (unsorted)
     * so the budget spreads over all layers. */
    std::vector<RadFileProfile> flat;
    /* R3_MOE_WORKER_HEAT: route counts a profiling run wrote (R3_MOE_WORKER_HISTOGRAM). Coldest first, so the
     * engine keeps the experts most tokens route to and a decode step hands off as little as possible. */
    if (const char* hp = std::getenv("R3_MOE_WORKER_HEAT"); hp && *hp) {
        std::vector<uint32_t> heat((size_t)kMaxLayers * f.experts, 0);
        FILE* hf = std::fopen(hp, "rb");
        const size_t got = hf ? std::fread(heat.data(), sizeof(uint32_t), heat.size(), hf) : 0;
        if (hf) std::fclose(hf);
        if (got != heat.size()) {
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_HEAT %s: want %zu counts, read %zu\n",
                         hp, heat.size(), got);
            return false;
        }
        for (uint32_t l = 0; l < f.layers; ++l)
            for (uint32_t e = 0; e < f.experts; ++e)
                if (f.weight(l, e, 0))
                    flat.push_back(RadFileProfile{(int32_t)l, (int32_t)e, (float)heat[(size_t)l * f.experts + e], 0.0f});
        std::stable_sort(flat.begin(), flat.end(), [](const RadFileProfile& a, const RadFileProfile& b) {
            return a.share < b.share;
        });
        std::fprintf(stderr, "r3_moe_worker: owning the coldest experts of %s first\n", hp);
        for (const RadFileProfile& p : flat) order.push_back(&p);
    } else if (!f.profile || !f.h->prof_count) {
        std::fprintf(stderr, "r3_moe_worker: container has no expert profile; owning experts evenly across layers\n");
        /* Highest ids first: the engine's initial placement keeps low ids resident on the XTX and
         * leaves the high ids in its host pool (measured: every host-pool route was expert >= 302). */
        for (uint32_t e = f.experts; e-- > 0;)
            for (uint32_t l = 0; l < f.layers; ++l)
                if (f.weight(l, e, 0)) flat.push_back(RadFileProfile{(int32_t)l, (int32_t)e, 0.0f, 0.0f});
        for (const RadFileProfile& p : flat) order.push_back(&p);
    }
    for (uint64_t i = 0; flat.empty() && f.profile && i < f.h->prof_count; ++i) {
        const RadFileProfile& p = f.profile[i];
        if (p.layer < 0 || p.expert < 0 || (uint32_t)p.layer >= f.layers ||
            p.expert >= (int32_t)f.experts) continue;
        order.push_back(&p);
    }
    if (flat.empty()) std::stable_sort(order.begin(), order.end(), [](const RadFileProfile* a, const RadFileProfile* b) {
        if (a->share != b->share) return a->share < b->share;
        if (a->layer != b->layer) return a->layer < b->layer;
        return a->expert < b->expert;
    });
    /* R3_MOE_WORKER_BALANCE (RAD06, needs R3_MOE_WORKER_HEAT and two helpers): the same jointly owned set the
     * alternation below would pick, split per layer by predicted cost instead. A layer finishes when both helpers
     * have, so each layer wants max(helper0, helper1) small; alternating down one global ranking balances the number
     * of experts, not the work. Each helper runs this on its own, so everything here is a deterministic function of the
     * heat file and the budget (cost ties break on expert id). Cost of an expert, from its share of its layer's routes:
     *   lam = prefill_tokens * top_k * share      cost = (1 - exp(-lam)) * c0 + c1 * lam + (1 - exp(-lam_d)) * d0
     * (a fixed weight-tile part when the expert is active plus a part a routed row, and the same for a decode step).
     * The heat file holds max + 1 - count (so the hottest sort first); the counts are max - value.
     * R3_MOE_WORKER_BALANCE_SPEED="s0,s1" scales a helper's cost by 1/s (a slower card takes less). */
    if (const char* bv = std::getenv("R3_MOE_WORKER_BALANCE"); bv && *bv && std::strcmp(bv, "0") != 0) {
        if (helpers != 2 || flat.empty()) {
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_BALANCE needs R3_MOE_WORKER_HEAT and two helpers\n");
            return false;
        }
        struct Pick { uint32_t l, e; int64_t bytes; double cost; };
        float maxv = 0.0f;
        for (const RadFileProfile& hp : flat) maxv = std::max(maxv, hp.share);
        std::vector<double> layer_sum(f.layers, 0.0);
        for (const RadFileProfile& hp : flat) layer_sum[(uint32_t)hp.layer] += (double)(maxv - hp.share);
        const double tokens = 2048.0, decode = 11.0, c0 = 15.0, c1 = 0.09, d0 = 20.0;
        auto cost_of = [&](const RadFileProfile& hp) {
            const double share = layer_sum[(uint32_t)hp.layer] > 0.0 ? (double)(maxv - hp.share) / layer_sum[(uint32_t)hp.layer] : 0.0;
            const double lam = tokens * (double)kTopK * share, lam_d = decode * (double)kTopK * share;
            return (1.0 - std::exp(-lam)) * c0 + c1 * lam + (1.0 - std::exp(-lam_d)) * d0;
        };
        double speed[2] = {1.0, 1.0};
        if (const char* sv = std::getenv("R3_MOE_WORKER_BALANCE_SPEED"); sv && *sv) {
            double a = 0.0, b = 0.0;
            if (std::sscanf(sv, "%lf,%lf", &a, &b) == 2 && a > 0.0 && b > 0.0) { speed[0] = a; speed[1] = b; }
        }
        /* The jointly owned set: exactly the alternation's picks (both helpers' budgets filled in turn). */
        std::vector<Pick> joint;
        {
            std::vector<uint8_t> was_seen((size_t)f.layers * f.experts, 0);
            uint64_t held[2] = {0, 0}, pos = 0;
            for (const RadFileProfile* p : order) {
                const uint32_t l = (uint32_t)p->layer, e = (uint32_t)p->expert;
                uint8_t& w = was_seen[(size_t)l * f.experts + e];
                if (w) continue;
                w = 1;
                const uint32_t who = (uint32_t)(pos++ % 2);
                const int64_t bytes = laid_out_bytes(f, gpu, l, e);
                if (bytes <= 0) {
                    std::fprintf(stderr, "r3_moe_worker: expert (%u,%u) cannot be sized for the device\n", l, e);
                    return false;
                }
                if ((uint64_t)bytes > budget - held[who]) continue;
                held[who] += (uint64_t)bytes;
                joint.push_back(Pick{l, e, bytes, cost_of(*p)});
            }
        }
        /* Per layer, heaviest first onto the less loaded helper; a helper whose budget cannot take the expert passes it on.
         * The decision is made in layer / cost order, but each helper's list is emitted in the ranking order the
         * alternation would have used (hottest first): the order is also the order the weights are allocated in. */
        std::vector<uint32_t> idx(joint.size());
        for (uint32_t i = 0; i < idx.size(); ++i) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
            const Pick& x = joint[a];
            const Pick& y = joint[b];
            if (x.l != y.l) return x.l < y.l;
            if (x.cost != y.cost) return x.cost > y.cost;
            return x.e < y.e;
        });
        std::vector<int8_t> assigned(joint.size(), -1);
        uint64_t held[2] = {0, 0};
        double load[2] = {0.0, 0.0};
        uint32_t cur_layer = 0xffffffffu;
        for (uint32_t i : idx) {
            const Pick& pk = joint[i];
            if (pk.l != cur_layer) { cur_layer = pk.l; load[0] = load[1] = 0.0; }
            uint32_t who = load[0] + pk.cost / speed[0] <= load[1] + pk.cost / speed[1] ? 0u : 1u;
            if ((uint64_t)pk.bytes > budget - held[who]) who ^= 1u;
            if ((uint64_t)pk.bytes > budget - held[who]) continue;
            held[who] += (uint64_t)pk.bytes;
            load[who] += pk.cost / speed[who];
            assigned[i] = (int8_t)who;
        }
        for (uint32_t i = 0; i < joint.size(); ++i) {
            if (assigned[i] != (int8_t)helper) continue;
            out->expert.emplace_back(joint[i].l, joint[i].e);
            out->bytes += (uint64_t)joint[i].bytes;
        }
        std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_BALANCE: per-layer cost split of %zu jointly owned experts\n",
                     joint.size());
        if (out->expert.empty()) {
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_MIB fits no profiled expert\n");
            return false;
        }
        return true;
    }
    std::vector<uint8_t> seen((size_t)f.layers * f.experts, 0);
    uint64_t position = 0;
    for (const RadFileProfile* p : order) {
        const uint32_t l = (uint32_t)p->layer, e = (uint32_t)p->expert;
        uint8_t& was = seen[(size_t)l * f.experts + e];
        if (was) continue;
        was = 1;
        /* Helper h of n takes every n-th candidate: disjoint sets, each spread over the same order. */
        if (position++ % helpers != helper) continue;
        const int64_t bytes = laid_out_bytes(f, gpu, l, e);
        if (bytes <= 0) {
            std::fprintf(stderr, "r3_moe_worker: expert (%u,%u) cannot be sized for the device\n", l, e);
            return false;
        }
        if ((uint64_t)bytes > budget - out->bytes) continue;
        out->expert.emplace_back(l, e);
        out->bytes += (uint64_t)bytes;
    }
    if (out->expert.empty()) {
        std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_MIB fits no profiled expert\n");
        return false;
    }
    return true;
}

void publish_owned(void* map, const Owned& owned, uint32_t helper) {
    Header* h = static_cast<Header*>(map);
    __atomic_fetch_add(&h->owned_bytes, owned.bytes, __ATOMIC_ACQ_REL);
    for (auto [l, e] : owned.expert) {
        if (!set_owned(map, l, e, helper))
            std::fprintf(stderr, "r3_moe_worker: duplicate/invalid ownership (%u,%u)\n", l, e);
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

/* REQUEST CAPTURE / REPLAY (RAD06). R3_MOE_WORKER_CAPTURE=<dir> writes real large-lane requests (every
 * R3_MOE_WORKER_CAPTURE_EVERY-th, at most R3_MOE_WORKER_CAPTURE_MAX) as <dir>/req-<lane>-<seq>.bin:
 * CaptureHeader, the layer's runtime slot map (kExperts int32), the routes, the bf16 activation rows.
 * `--replay MODEL FILE` runs --compare's machinery on that request instead of synthetic routing. */
struct CaptureHeader {
    uint32_t magic, version, layer, rows, route_count, hidden;
};
constexpr uint32_t kCaptureMagic = 0x52335243u;  // "CR3R"

void capture_request(void* map, const Request* q, uint32_t lane, uint32_t seq) {
    static const char* dir = std::getenv("R3_MOE_WORKER_CAPTURE");
    if (!dir || !*dir) return;
    static const uint32_t every = [] {
        const char* v = std::getenv("R3_MOE_WORKER_CAPTURE_EVERY");
        return v && *v ? std::max(1, std::atoi(v)) : 37;
    }();
    static const uint32_t max_n = [] {
        const char* v = std::getenv("R3_MOE_WORKER_CAPTURE_MAX");
        return v && *v ? std::max(1, std::atoi(v)) : 16;
    }();
    static std::atomic<uint32_t> written{0};
    if (seq % every != 0 || written.load() >= max_n) return;
    CaptureHeader hd{kCaptureMagic, 2, load(&q->layer), load(&q->rows), load(&q->route_count), kHidden};
    if (hd.layer >= kMaxLayers || hd.rows == 0 || hd.rows > kMaxRows) return;
    const std::string path = std::string(dir) + "/req-" + std::to_string(lane) + "-" + std::to_string(seq) + ".bin";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fwrite(&hd, sizeof(hd), 1, f);
    std::fwrite(layer_slots(map, hd.layer), sizeof(int32_t), kExperts, f);
    std::fwrite(ptr<Route>(map, q->route_offset), sizeof(Route), hd.route_count, f);
    /* The engine publishes the quantised activation (E4M3 plus a scale a group), not bf16: the bf16 plane of the map
     * is empty on the large lane. Version 1 captures stored that empty plane and cannot be replayed. */
    std::fwrite(ptr<uint8_t>(map, q->activation_q_offset), 1, (size_t)hd.rows * kHidden, f);
    std::fwrite(ptr<float>(map, q->activation_s_offset), sizeof(float), (size_t)hd.rows * (kHidden / kActGroup), f);
    std::fclose(f);
    written.fetch_add(1);
}

/* R3_MOE_WORKER_TRACE=N: print the stamps of every Nth finished request ("TRACE,..." on stderr). The engine writes its
 * wait stamps right after it sees the helper's done word, so a request is read once the lane has moved on to the next
 * one. Engine stamps (0-2) and helper stamps (3-4) come from different devices' clocks and are only ever subtracted
 * within their own group; wall_clock64 runs at hipDeviceAttributeWallClockRate (100 MHz on these cards). */
void trace_dump(void* map, uint32_t lane, uint32_t seq) {
    static const uint32_t every = [] {
        const char* v = std::getenv("R3_MOE_WORKER_TRACE");
        return v && *v ? (uint32_t)std::max(0, std::atoi(v)) : 0u;
    }();
    if (!every || seq == 0 || seq % every != 0) return;
    const Request* q = lane_request(map, lane, seq);
    if (!q || load(&q->sequence) != seq) return;
    uint64_t t[5];
    for (int i = 0; i < 5; ++i) t[i] = q->trace[i];
    if (!t[0] || !t[1] || !t[2] || !t[3] || !t[4]) return;
    const double us = 1.0 / 100.0;   /* a tick is 10 ns */
    /* wall_ns: this host's clock when the line is printed, so a benchmark can bucket lines by the round that produced them. */
    const unsigned long long wall_ns = (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::fprintf(stderr, "TRACE,lane,%u,seq,%u,layer,%u,rows,%u,routes,%u,local_us,%.2f,wait_us,%.2f,helper_gpu_us,%.2f,enter,%llu,wall_ns,%llu\n",
                 lane, seq, q->layer, q->rows, q->route_count,
                 (double)(int64_t)(t[1] - t[0]) * us, (double)(int64_t)(t[2] - t[1]) * us,
                 (double)(int64_t)(t[4] - t[3]) * us, (unsigned long long)t[1], wall_ns);
}

struct ReplayInput {
    CaptureHeader hd{};
    std::array<int32_t, kExperts> slots{};
    std::vector<Route> routes;
    std::vector<uint8_t> actq;
    std::vector<float> acts;
};
const ReplayInput* g_replay = nullptr;
/* --replay-sweep with R3_MOE_SWEEP_FAST=1: no CPU reference; each variant's output is compared bit for bit with the
 * first variant's of the same capture. */
bool g_sweep_fast = false;

bool load_replay(const char* path, ReplayInput* out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    bool ok = std::fread(&out->hd, sizeof(out->hd), 1, f) == 1 && out->hd.magic == kCaptureMagic && out->hd.version == 2 &&
              out->hd.hidden == kHidden && out->hd.layer < kMaxLayers && out->hd.rows > 0 && out->hd.rows <= kMaxRows;
    if (ok) ok = std::fread(out->slots.data(), sizeof(int32_t), kExperts, f) == kExperts;
    if (ok) {
        out->routes.resize(out->hd.route_count);
        out->actq.resize((size_t)out->hd.rows * kHidden);
        out->acts.resize((size_t)out->hd.rows * (kHidden / kActGroup));
        ok = std::fread(out->routes.data(), sizeof(Route), out->routes.size(), f) == out->routes.size() &&
             std::fread(out->actq.data(), 1, out->actq.size(), f) == out->actq.size() &&
             std::fread(out->acts.data(), sizeof(float), out->acts.size(), f) == out->acts.size();
    }
    std::fclose(f);
    return ok;
}

int compare_mode(const Container& model, r3_moe_gpu::Runner& gpu, const Owned& owned,
                 uint32_t layer, uint32_t rows, const SmallClasses& classes, uint32_t lane_large_rows,
                 bool synthetic_protect) {
    if (g_replay) {  // a captured real request replaces the synthetic layer, rows, routes and activations
        layer = g_replay->hd.layer;
        rows = g_replay->hd.rows;
    }
    if (layer >= model.layers || rows == 0 || rows > lane_large_rows) {
        std::fprintf(stderr, "r3_moe_worker: --compare rows %u above R3_MOE_WORKER_MAX_ROWS %u\n",
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
    if (g_replay) runtime_slot = g_replay->slots;

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
                     "r3_moe_worker: --compare layer %u needs >=2 owned and >=1 local experts\n",
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
                         "r3_moe_worker: --compare synthetic protection did not shift an owned TP class\n");
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
    if (g_replay) {
        published = g_replay->routes;
        /* The bf16 plane (the host reference's input) from the captured quantised planes; the lane itself reads the
         * captured q and s as they are. */
        for (size_t i = 0; i < activation.size(); ++i) {
            const uint32_t c = g_replay->actq[i];
            const int e = (int)((c >> 3) & 0xfu), m = (int)(c & 7u);
            const float mag = e ? std::ldexp(1.0f + m / 8.0f, e - 7) : std::ldexp(m / 8.0f, -6);
            activation[i] = to_bf16(((c & 0x80u) ? -mag : mag) * g_replay->acts[i / kActGroup]);
        }
    }
    const uint32_t nr = (uint32_t)published.size();

    r3::moe_shm::LayoutShape shape{};
    Layout lay{};
    if (!segment_shape(model, &shape, &lay)) return 1;
    void* map = mmap(nullptr, lay.bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (map == MAP_FAILED) {
        std::fprintf(stderr, "r3_moe_worker: --compare mmap: %s\n", std::strerror(errno));
        return 1;
    }
    if (!init(map, model.layers, shape)) return 1;
    publish_owned(map, owned, 0);
    Header* h = static_cast<Header*>(map);
    h->helper_count = 1;
    std::memcpy(layer_slots(map, layer), runtime_slot.data(), sizeof(runtime_slot));
    store(&h->layer_map_ready[layer], 1);
    if (!gpu.load_layer(layer, owned.expert, runtime_slot.data())) {
        std::fprintf(stderr, "r3_moe_worker: --compare layer %u runtime table build failed\n", layer);
        munmap(map, lay.bytes);
        return 1;
    }
    store(&h->layer_map_consumed[layer], 1);
    h->small_class_count = classes.count;
    for (uint32_t c = 0; c < classes.count; ++c) h->small_class_rows[c] = classes.rows[c];
    if (!gpu.prepare_lanes(classes.rows, classes.count, lane_large_rows)) {
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
        const uint32_t lane = lane_of(rank, step_class(h, rows));
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
        if (g_replay) {
            std::memcpy(ptr<uint8_t>(map, q->activation_q_offset), g_replay->actq.data(), g_replay->actq.size());
            std::memcpy(ptr<float>(map, q->activation_s_offset), g_replay->acts.data(),
                        g_replay->acts.size() * sizeof(float));
        } else if (!gpu.stage_compare_activation(map, lane, activation.data(), rows)) {
            std::fprintf(stderr,
                         "r3_moe_worker: --compare rank %u could not stage exact q/s\n", rank);
            bad = true;
            break;
        }

        /* Queue the production lane before publishing the sequence, as the service worker does. */
        const int erc = gpu.enqueue_lane(map, dev_map, lane, sequence,
                                         lane_is_small(lane) ? h->small_class_rows[lane_class(lane)] : rows, false);
        if (erc != 0) {
            std::fprintf(stderr, "r3_moe_worker: --compare lane rank %u enqueue failed rc=%d\n",
                         rank, erc);
            bad = true;
            break;
        }
        store(&h->lane[lane].request_sequence, sequence);
        if (gpu.sync_lane(lane)) gpu.gemv_debug_report(lane);
        if (!gpu.sync_lane(lane) || load(&h->lane[lane].done_sequence) != sequence || q->status != 0) {
            std::fprintf(stderr,
                         "r3_moe_worker: --compare lane rank %u completion failed status=%d done=%u\n",
                         rank, q->status, load(&h->lane[lane].done_sequence));
            bad = true;
            break;
        }
        /* The helper's per-token rows, taken before the host reference rewrites the same slot. */
        const uint16_t* res = ptr<uint16_t>(map, h->result_offset[rank * kMaxRanks + rank]);
        const std::vector<uint16_t> got(res, res + (size_t)rows * kHidden);
        if (g_sweep_fast) {
            static std::map<std::pair<const ReplayInput*, uint32_t>, std::vector<uint16_t>> base;
            double sum_abs = 0.0;
            size_t nonzero = 0;
            for (uint16_t v : got) { const double x = std::abs((double)bf16(v)); sum_abs += x; nonzero += v != 0; }
            std::printf("sweep_sum layer %u rank %u sum_abs=%.6g nonzero=%zu of %zu\n", layer, rank, sum_abs, nonzero, got.size());
            const auto key = std::make_pair(g_replay, rank);
            const auto it = base.find(key);
            if (it == base.end()) {
                base[key] = got;
                std::printf("sweep_cmp layer %u rank %u baseline mismatches=0\n", layer, rank);
            } else {
                size_t mism = 0;
                for (size_t i = 0; i < got.size(); ++i) mism += got[i] != it->second[i];
                std::printf("sweep_cmp layer %u rank %u mismatches=%zu of %zu\n", layer, rank, mism, got.size());
            }
            continue;
        }

        std::vector<uint16_t> want_route;
        const int hrc = compute_host_engine(model, map, q, &want_route);
        if (hrc != 0 || want_route.size() != (size_t)nr * kHidden) {
            std::fprintf(stderr,
                         "r3_moe_worker: --compare local-path host rank %u failed rc=%d\n",
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

/* --route-selftest: the one-block routing kernels at several expert counts against a CPU model (RCS09-05). No container, no shared
 * memory, no model: synthetic routes and a slot table, run on the first visible device. The unfused count/prefix/scatter path is
 * compared as sets per expert slot (its scatter order is atomic); the fused small-lane front end must match the deterministic
 * order exactly. Exit status is the number of failing cases. */
int route_selftest() {
    using namespace r3::moe_shm;
    using namespace r3_moe_gpu;
    uint32_t rng = 12345u;
    auto next = [&rng] { rng = rng * 1664525u + 1013904223u; return rng >> 8; };
    auto hip = [](hipError_t e, const char* what) {
        if (e != hipSuccess) std::fprintf(stderr, "route-selftest: %s: %s\n", what, hipGetErrorString(e));
        return e == hipSuccess;
    };
    int failures = 0;
    const uint32_t kMaxRowsFused = 8;
    for (uint32_t experts : {128u, 256u, 384u, 512u, 640u, 1000u}) {
        uint32_t blockn = 256;
        while (blockn < experts) blockn <<= 1;
        for (uint32_t rows : {1u, 8u}) {
            for (int permuted = 0; permuted < 2; ++permuted) {
                const uint32_t nr = rows * kTopK;
                std::vector<Route> routes(nr);
                std::vector<uint16_t> slot(experts);
                for (uint32_t e = 0; e < experts; ++e) slot[e] = permuted ? (uint16_t)((e * 7u + 3u) % experts) : (uint16_t)e;
                for (uint32_t t = 0; t < rows; ++t)
                    for (uint32_t k = 0; k < kTopK; ++k)
                        routes[t * kTopK + k] = Route{(int32_t)t, (int32_t)(next() % experts), (uint16_t)(0x3f80u + k), (uint16_t)k};
                std::vector<uint32_t> counts(experts, 0), excl(experts + 1, 0);
                std::vector<std::vector<int32_t>> want(experts);
                for (uint32_t i = 0; i < nr; ++i) {
                    const uint32_t es = slot[(uint32_t)routes[i].expert];
                    ++counts[es];
                    want[es].push_back((int32_t)((uint32_t)routes[i].token * kTopK + routes[i].slot));
                }
                for (uint32_t e = 0; e < experts; ++e) excl[e + 1] = excl[e] + counts[e];

                Request q{};
                q.sequence = 1; q.rank = 0; q.rank_count = 1; q.layer = 0; q.rows = rows; q.hidden = kHidden; q.route_count = nr;
                q.activation_stride = kHidden * (uint32_t)sizeof(uint16_t);
                q.activation_q_stride = kHidden;
                q.activation_s_stride = (kHidden / kActGroup) * (uint32_t)sizeof(float);
                DeviceLayerTables tl{};
                const size_t half = experts / 2;
                std::vector<void*> tables(8, nullptr);
                void *d_route = nullptr, *d_q = nullptr, *d_slot = nullptr, *d_tl = nullptr, *d_counts = nullptr, *d_off = nullptr;
                void *d_cursor = nullptr, *d_flat = nullptr, *d_ew = nullptr, *d_sorted = nullptr, *d_inv = nullptr, *d_stats = nullptr;
                void *d_srcq = nullptr, *d_srcs = nullptr, *d_dstq = nullptr, *d_dsts = nullptr;
                const size_t q_per_row = kHidden / 16, s_per_row = (kHidden / kActGroup) * 4 / 16;
                bool good = hip(hipMalloc(&d_route, nr * sizeof(Route)), "alloc routes") &&
                            hip(hipMalloc(&d_q, sizeof(Request)), "alloc request") &&
                            hip(hipMalloc(&d_slot, experts * sizeof(uint16_t)), "alloc slots") &&
                            hip(hipMalloc(&d_tl, sizeof(DeviceLayerTables)), "alloc tables") &&
                            hip(hipMalloc(&d_counts, experts * 4), "alloc counts") &&
                            hip(hipMalloc(&d_off, (experts + 1) * 4), "alloc offsets") &&
                            hip(hipMalloc(&d_cursor, experts * 4), "alloc cursor") &&
                            hip(hipMalloc(&d_flat, nr * 4), "alloc flat") && hip(hipMalloc(&d_ew, nr * 2), "alloc weights") &&
                            hip(hipMalloc(&d_sorted, nr * 4), "alloc sorted") && hip(hipMalloc(&d_inv, nr * 4), "alloc inverse") &&
                            hip(hipMalloc(&d_stats, 3 * 8), "alloc stats") &&
                            hip(hipMalloc(&d_srcq, kMaxRowsFused * q_per_row * 16), "alloc src q") &&
                            hip(hipMalloc(&d_srcs, kMaxRowsFused * s_per_row * 16), "alloc src s") &&
                            hip(hipMalloc(&d_dstq, kMaxRowsFused * q_per_row * 16), "alloc dst q") &&
                            hip(hipMalloc(&d_dsts, kMaxRowsFused * s_per_row * 16), "alloc dst s");
                for (auto& t : tables) good = good && hip(hipMalloc(&t, (half + 1) * sizeof(void*)), "alloc table");
                if (good) {
                    for (auto& t : tables) good = good && hip(hipMemset(t, 0, (half + 1) * sizeof(void*)), "zero table");
                    for (int c = 0; c < 2; ++c) {
                        tl.gu_code[0][c] = tables[c * 4 + 0]; tl.gu_scale[0][c] = tables[c * 4 + 1];
                        tl.dn_code[0][c] = tables[c * 4 + 2]; tl.dn_scale[0][c] = tables[c * 4 + 3];
                    }
                    tl.expert_slot = static_cast<const uint16_t*>(d_slot);
                    good = good && hip(hipMemcpy(d_route, routes.data(), nr * sizeof(Route), hipMemcpyHostToDevice), "upload routes") &&
                           hip(hipMemcpy(d_q, &q, sizeof q, hipMemcpyHostToDevice), "upload request") &&
                           hip(hipMemcpy(d_slot, slot.data(), experts * 2, hipMemcpyHostToDevice), "upload slots") &&
                           hip(hipMemcpy(d_tl, &tl, sizeof tl, hipMemcpyHostToDevice), "upload tables") &&
                           hip(hipMemset(d_counts, 0, experts * 4), "zero counts") && hip(hipMemset(d_stats, 0, 24), "zero stats") &&
                           hip(hipMemset(d_srcq, 0, kMaxRowsFused * q_per_row * 16), "zero src q") &&
                           hip(hipMemset(d_srcs, 0, kMaxRowsFused * s_per_row * 16), "zero src s");
                }
                const auto* rq = static_cast<const Request*>(d_q);
                const auto* rt = static_cast<const Route*>(d_route);
                const auto* lt = static_cast<const DeviceLayerTables*>(d_tl);
                const uint32_t grid = (nr + 255) / 256;
                auto run_prefix = [&](auto n_const) {
                    constexpr uint32_t N = decltype(n_const)::value;
                    hipLaunchKernelGGL((lane_route_prefix<N>), dim3(1), dim3(N), 0, 0, static_cast<const uint32_t*>(d_counts),
                                       static_cast<int32_t*>(d_off), static_cast<uint32_t*>(d_cursor), experts);
                };
                auto run_front = [&](auto n_const) {
                    constexpr uint32_t N = decltype(n_const)::value;
                    hipLaunchKernelGGL((lane_front_small<N>), dim3(1), dim3(N), 0, 0, rt, const_cast<Request*>(rq), lt,
                                       static_cast<const uint4*>(d_srcq), static_cast<const uint4*>(d_srcs),
                                       static_cast<uint4*>(d_dstq), static_cast<uint4*>(d_dsts), static_cast<uint64_t*>(d_stats),
                                       0u, 0u, 1u, kMaxRowsFused, 1u, nr, experts, static_cast<int32_t*>(d_off),
                                       static_cast<int32_t*>(d_sorted), static_cast<int32_t*>(d_inv), static_cast<uint16_t*>(d_ew),
                                       static_cast<void**>(tables[0]), static_cast<void**>(tables[1]), static_cast<void**>(tables[2]),
                                       static_cast<void**>(tables[3]), static_cast<void**>(tables[4]), static_cast<void**>(tables[5]),
                                       static_cast<void**>(tables[6]), static_cast<void**>(tables[7]));
                };
                std::vector<int32_t> off(experts + 1), sorted(nr), inv(nr);
                std::vector<uint16_t> ew(nr);
                std::vector<uint32_t> got_counts(experts);
                std::string why;
                if (!good) why = "device setup";
                if (good) {   // the unfused path
                    hipLaunchKernelGGL(lane_route_count, dim3(grid), dim3(256), 0, 0, rt, const_cast<Request*>(rq), lt, 0u, 1u,
                                       rows, 1u, experts, static_cast<uint32_t*>(d_counts), static_cast<int32_t*>(d_flat),
                                       static_cast<uint16_t*>(d_ew));
                    switch (blockn) {
                        case 256: run_prefix(std::integral_constant<uint32_t, 256>{}); break;
                        case 512: run_prefix(std::integral_constant<uint32_t, 512>{}); break;
                        case 1024: run_prefix(std::integral_constant<uint32_t, 1024>{}); break;
                        default: why = "no block size";
                    }
                    hipLaunchKernelGGL(lane_route_scatter, dim3(grid), dim3(256), 0, 0, rt, rq, lt, 0u, 1u, rows, 1u, experts,
                                       static_cast<const int32_t*>(d_flat), static_cast<uint32_t*>(d_cursor),
                                       static_cast<int32_t*>(d_sorted), static_cast<int32_t*>(nullptr));
                    good = hip(hipDeviceSynchronize(), "unfused kernels") &&
                           hip(hipMemcpy(got_counts.data(), d_counts, experts * 4, hipMemcpyDeviceToHost), "read counts") &&
                           hip(hipMemcpy(off.data(), d_off, (experts + 1) * 4, hipMemcpyDeviceToHost), "read offsets") &&
                           hip(hipMemcpy(sorted.data(), d_sorted, nr * 4, hipMemcpyDeviceToHost), "read sorted") &&
                           hip(hipMemcpy(ew.data(), d_ew, nr * 2, hipMemcpyDeviceToHost), "read weights");
                    if (!good && why.empty()) why = "unfused run";
                    if (why.empty()) {
                        if (got_counts != counts) why = "counts";
                        for (uint32_t e = 0; why.empty() && e <= experts; ++e)
                            if ((uint32_t)off[e] != excl[e]) why = "offsets";
                        for (uint32_t e = 0; why.empty() && e < experts; ++e) {
                            std::vector<int32_t> g(sorted.begin() + off[e], sorted.begin() + off[e + 1]), w = want[e];
                            std::sort(g.begin(), g.end());
                            std::sort(w.begin(), w.end());
                            if (g != w) why = "sorted routes";
                        }
                        for (uint32_t i = 0; why.empty() && i < nr; ++i)
                            if (ew[(uint32_t)routes[i].token * kTopK + routes[i].slot] != routes[i].weight_bf16) why = "weights";
                    }
                }
                if (good && why.empty() && rows <= kMaxRowsFused) {   // the fused front end: deterministic order
                    std::fill(off.begin(), off.end(), -1);
                    switch (blockn) {
                        case 256: run_front(std::integral_constant<uint32_t, 256>{}); break;
                        case 512: run_front(std::integral_constant<uint32_t, 512>{}); break;
                        case 1024: run_front(std::integral_constant<uint32_t, 1024>{}); break;
                        default: why = "no block size";
                    }
                    int32_t status = 0;
                    good = hip(hipDeviceSynchronize(), "fused kernel") &&
                           hip(hipMemcpy(off.data(), d_off, (experts + 1) * 4, hipMemcpyDeviceToHost), "read fused offsets") &&
                           hip(hipMemcpy(sorted.data(), d_sorted, nr * 4, hipMemcpyDeviceToHost), "read fused sorted") &&
                           hip(hipMemcpy(inv.data(), d_inv, nr * 4, hipMemcpyDeviceToHost), "read fused inverse") &&
                           hip(hipMemcpy(&status, static_cast<uint8_t*>(d_q) + offsetof(Request, status), 4, hipMemcpyDeviceToHost),
                               "read status");
                    if (!good && why.empty()) why = "fused run";
                    if (why.empty() && status != 0) why = "fused status";
                    for (uint32_t e = 0; why.empty() && e <= experts; ++e)
                        if ((uint32_t)off[e] != excl[e]) why = "fused offsets";
                    for (uint32_t e = 0; why.empty() && e < experts; ++e)
                        for (uint32_t j = 0; why.empty() && j < counts[e]; ++j) {
                            if (sorted[excl[e] + j] != want[e][j]) why = "fused order";
                            else if (inv[(uint32_t)want[e][j]] != (int32_t)(excl[e] + j)) why = "fused inverse";
                        }
                }
                for (void* p : {d_route, d_q, d_slot, d_tl, d_counts, d_off, d_cursor, d_flat, d_ew, d_sorted, d_inv, d_stats, d_srcq, d_srcs,
                                d_dstq, d_dsts})
                    if (p) (void)hipFree(p);
                for (void* p : tables)
                    if (p) (void)hipFree(p);
                const bool pass = good && why.empty();
                std::printf("route-selftest experts %u block %u rows %u %s: %s%s%s\n", experts, blockn, rows,
                            permuted ? "permuted" : "identity", pass ? "PASS" : "FAIL", pass ? "" : " ", pass ? "" : why.c_str());
                if (!pass) ++failures;
            }
        }
    }
    std::printf("route-selftest: %d failing case(s)\n", failures);
    return failures;
}

int main(int argc, char** argv) {
    /* EVERY LANE STREAM ON ITS OWN HARDWARE QUEUE. A small lane pre-queues "wait for the next request" packets;
     * a stream sharing a hardware queue with one sits behind that wait until a small request arrives. With
     * every step large (KL mode's 64-token steps over an 8-row small lane) none ever does, and the large lane's
     * work never runs. HIP maps streams onto GPU_MAX_HW_QUEUES queues (4 by default) and the helper opens one
     * stream for every lane in the protocol (kLanes: each small-lane class and the large lane, for each rank)
     * plus its own, so the count is derived from the protocol, with two spare, before HIP initialises (it grew
     * from five streams to eleven with RAD12's lane classes, which hung every request at the old fixed 8);
     * an explicit setting wins. */
    (void)::setenv("GPU_MAX_HW_QUEUES", std::to_string(r3::moe_shm::kLanes + 3).c_str(), 0);
    /* The helper's grouped MoE GEMM (prefill) on gfx11's native bf16 WMMA path: R9700 engine + two XTX helpers,
     * Flash-Next, prefill 2,518 -> 2,665 tok/s and helper gate/up -25%, KLD 0.064 (noise floor). Read by libr3
     * when the plugin is loaded, so set before it; an explicit setting wins. */
    (void)::setenv("R3_MOE_NATIVE", "1", 0);
    /* The same MoE GEMM with E4M3 widened to f16 for the f16 WMMA: bit-identical output (replay rel_l2 0 on four
     * captured requests, KLD unchanged), grouped GEMM 1.7-3.2x faster, prefill 2,130 -> 3,150 tok/s with decode
     * unchanged (RAD06 2026-10-18). Read by libr3 at launch; 0 turns it off. */
    (void)::setenv("R3_MOE_FP16", "1", 0);
    /* And the activation tile widened to half once at staging instead of by every wave that reads it: bit-identical,
     * a further 7-8% off the grouped GEMM on four real captured requests (RAD06 2026-10-18). 0 turns it off. */
    (void)::setenv("R3_MOE_AHALF", "1", 0);
    /* The activation markers component.toml names: one line a flag that is on, printed here and only when on, so a
     * capture's evidence check finds it in the arm that sets the flag and not in the control. FP16 and AHALF are read by
     * libr3's launcher at each launch; VECIO and FUSEROUTE by the lane at each request (default on unless "0"). */
    {
        auto is_on = [](const char* name, bool dflt) {
            const char* v = std::getenv(name);
            return v && *v ? std::strcmp(v, "0") != 0 : dflt;
        };
        const bool native = is_on("R3_MOE_NATIVE", false);
        if (native && is_on("R3_MOE_FP16", false))
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_FP16 on - f16 WMMA MoE GEMM\n");
        if (native && is_on("R3_MOE_FP16", false) && is_on("R3_MOE_AHALF", false))
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_AHALF on - activation tile widened to half at staging\n");
        if (is_on("R3_MOE_WORKER_VECIO", true))
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_VECIO on - 16-byte lane I/O\n");
        if (is_on("R3_MOE_WORKER_FUSEROUTE", true))
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_FUSEROUTE on - fused small-lane front end\n");
    }
    /* --replay-sweep MODEL VARIANTS FILE...: one process, one model load and one expert allocation for every capture,
     * then each capture is replayed once per variant. VARIANTS is ';'-separated, each "NAME=value,NAME=value" (empty
     * = the environment as launched); the variant's variables are set (and the previous variant's unset) before the
     * replay, so anything libr3 reads at launch (R3_MOE_FP16, R3_MOE_PROBE, R3_MOE_PITCH16) can be compared with
     * no further model load. Each run prints "SWEEP run i capture c layer L variant [..]" and its engine_join line. */
    if (argc >= 2 && std::strcmp(argv[1], "--route-selftest") == 0) return route_selftest();
    /* --describe MODEL: print what the container's experts are and whether the gfx1100 backend can run them, then exit
     * (0 supported, 3 refused). No GPU, no allocation, no shared memory. */
    if (argc >= 3 && std::strcmp(argv[1], "--describe") == 0) {
        Container model;
        if (!model.open(argv[2])) {
            std::fprintf(stderr, "r3_moe_worker: --describe %s: not a container\n", argv[2]);
            return 2;
        }
        return backend_accepts(model, "gfx1100", true) ? 0 : 3;
    }
    static std::vector<ReplayInput> sweep_replays;
    static std::vector<std::string> sweep_variants;
    if (argc >= 5 && std::strcmp(argv[1], "--replay-sweep") == 0) {
        for (const char* q = argv[3];;) {
            const char* e = std::strchr(q, ';');
            sweep_variants.emplace_back(q, e ? (size_t)(e - q) : std::strlen(q));
            if (!e) break;
            q = e + 1;
        }
        sweep_replays.resize((size_t)(argc - 4));
        for (int i = 4; i < argc; ++i)
            if (!load_replay(argv[i], &sweep_replays[(size_t)(i - 4)])) {
                std::fprintf(stderr, "r3_moe_worker: --replay-sweep %s: not a capture file\n", argv[i]);
                return 2;
            }
        { const char* f = std::getenv("R3_MOE_SWEEP_FAST"); g_sweep_fast = f && *f && std::strcmp(f, "0") != 0; }
        static std::string sweep_layer, sweep_rows;
        g_replay = &sweep_replays[0];
        sweep_layer = std::to_string(sweep_replays[0].hd.layer);
        sweep_rows = std::to_string(sweep_replays[0].hd.rows);
        static const char* sweep_argv[8];
        const char* rebuilt[] = {argv[0], "--compare", argv[2], "--layer", sweep_layer.c_str(), "--rows",
                                 sweep_rows.c_str(), nullptr};
        std::copy(std::begin(rebuilt), std::end(rebuilt), sweep_argv);
        argv = const_cast<char**>(sweep_argv);
        argc = 7;
    }
    /* --replay MODEL FILE: --compare on a captured real request (layer and rows come from the file). */
    static ReplayInput replay;
    static std::string replay_layer, replay_rows;
    static const char* replay_argv[8];
    if (sweep_replays.empty() && argc == 4 && std::strcmp(argv[1], "--replay") == 0) {
        if (!load_replay(argv[3], &replay)) {
            std::fprintf(stderr, "r3_moe_worker: --replay %s: not a capture file\n", argv[3]);
            return 2;
        }
        g_replay = &replay;
        replay_layer = std::to_string(replay.hd.layer);
        replay_rows = std::to_string(replay.hd.rows);
        const char* rebuilt[] = {argv[0], "--compare", argv[2], "--layer", replay_layer.c_str(), "--rows",
                                 replay_rows.c_str(), nullptr};
        std::copy(std::begin(rebuilt), std::end(rebuilt), replay_argv);
        argv = const_cast<char**>(replay_argv);
        argc = 7;
        std::fprintf(stderr, "r3_moe_worker: replay layer %u rows %u routes %u\n", replay.hd.layer, replay.hd.rows,
                     replay.hd.route_count);
    }
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
                std::fprintf(stderr, "usage: r3_moe_worker --compare MODEL.rad --rows M\n");
                return 2;
            }
        } else if (carg == 7 && std::strcmp(argv[3], "--layer") == 0 &&
                   std::strcmp(argv[5], "--rows") == 0 &&
                   parse_u32(argv[4], 0, kMaxLayers - 1, &compare_layer) &&
                   parse_u32(argv[6], 1, kMaxRows, &compare_rows)) {
            compare_one_layer = true;
        } else {
            std::fprintf(stderr,
                "usage: r3_moe_worker --compare MODEL.rad [--layer L] --rows M [--synthetic-protect]\n");
            return 2;
        }
        model_path = argv[2];
    } else {
        if (argc != 2) {
            std::fprintf(stderr, "usage: r3_moe_worker MODEL.rad\n");
            return 2;
        }
        model_path = argv[1];
    }

    Container model;
    if (!model.open(model_path) || model.layers > kMaxLayers) {
        std::fprintf(stderr, "r3_moe_worker: cannot open/validate %s\n", model_path);
        return 1;
    }

    const char* cpu_env = std::getenv("R3_MOE_WORKER_CPU");
    const bool cpu = cpu_env && *cpu_env && std::strcmp(cpu_env, "0") != 0 && !compare;
    std::unique_ptr<r3_moe_gpu::Runner> gpu;
    if (!cpu) {
        gpu = std::make_unique<r3_moe_gpu::Runner>(model);
        if (const char* e = std::getenv("R3_MOE_WORKER_ENGINE_RANKS"); e && *e) {
            const int n = std::atoi(e);
            if (n < 1 || n > (int)kMaxRanks) {
                std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_ENGINE_RANKS=%s, want 1..%u\n", e, kMaxRanks);
                return 2;
            }
            gpu->engine_ranks_ = (uint32_t)n;
        }
        if (!gpu->open()) return 1;
    }

    /* R3_MOE_WORKER_COUNT=2 with R3_MOE_WORKER_INDEX=0/1: two helpers (the XTX cards) behind a one-rank engine
     * (the R9700) share one segment; helper 0 creates it, helper 1 attaches, each owns half the candidates and
     * serves its own rank slot. */
    uint32_t helper_index = 0, helper_count = 1;
    if (const char* e = std::getenv("R3_MOE_WORKER_COUNT"); e && *e) helper_count = (uint32_t)std::atoi(e);
    if (const char* e = std::getenv("R3_MOE_WORKER_INDEX"); e && *e) helper_index = (uint32_t)std::atoi(e);
    if (helper_count < 1 || helper_count > kMaxRanks || helper_index >= helper_count ||
        (helper_count > 1 && (!gpu || gpu->engine_ranks_ != 1 || compare))) {
        std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_INDEX %u of R3_MOE_WORKER_COUNT %u: want index < count <= %u, "
                     "and two helpers only on GPU behind a one-rank engine (R3_MOE_WORKER_ENGINE_RANKS=1), not --compare\n",
                     helper_index, helper_count, kMaxRanks);
        return 2;
    }
    const uint64_t budget = helper_budget_bytes();
    Owned owned;
    if (g_replay) {
        /* A replay owns exactly the experts the captured request routes to: the capturing helper's ownership
         * (one of two, by heat) is not this process's, and an unowned route would never complete. */
        std::set<std::pair<uint32_t, uint32_t>> seen;
        auto own_routes = [&](const ReplayInput& ri) {
            for (const Route& r : ri.routes)
                if (r.expert >= 0 && r.expert < (int32_t)model.experts &&
                    seen.emplace(ri.hd.layer, (uint32_t)r.expert).second)
                    owned.expert.emplace_back(ri.hd.layer, (uint32_t)r.expert);
        };
        if (sweep_replays.empty()) own_routes(*g_replay);
        else for (const ReplayInput& ri : sweep_replays) own_routes(ri);
    } else if (!select_owned(model, gpu.get(), budget, helper_index, helper_count, &owned)) {
        return 1;
    }
    if (gpu && !gpu->allocate(owned.expert)) return 1;
    const SmallClasses lane_classes = helper_small_classes();
    if (!lane_classes.count) return 1;
    const uint32_t lane_large_rows = helper_max_rows();
    std::string lane_list;                       /* "4,8,16,24", for the log */
    for (uint32_t c = 0; c < lane_classes.count; ++c)
        lane_list += (c ? "," : "") + std::to_string(lane_classes.rows[c]);
    if (const char* sl = std::getenv("R3_MOE_WORKER_SMALL_LANES"); sl && *sl)
        std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_SMALL_LANES on - small lanes %s\n", lane_list.c_str());
    long rss_pages = 0;
    if (FILE* sf = std::fopen("/proc/self/statm", "r")) {
        long total_pages = 0;
        if (std::fscanf(sf, "%ld %ld", &total_pages, &rss_pages) != 2) rss_pages = 0;
        std::fclose(sf);
    }
    const double rss_mib = rss_pages > 0 ? (double)rss_pages * (double)sysconf(_SC_PAGESIZE) / (1024.0 * 1024.0) : -1.0;
    std::fprintf(stderr, "r3_moe_worker: selected %zu expert candidates, %.3f GiB; host RSS %.1f MiB\n",
                 owned.expert.size(), (double)owned.bytes / (1024.0 * 1024.0 * 1024.0), rss_mib);

    if (compare && !sweep_replays.empty()) {
        std::set<std::string> touched;
        int rc = 0, run = 0;
        for (size_t c = 0; c < sweep_replays.size(); ++c) {
            g_replay = &sweep_replays[c];
            for (const std::string& v : sweep_variants) {
                for (const std::string& k : touched) (void)::unsetenv(k.c_str());
                touched.clear();
                for (size_t pos = 0; pos < v.size();) {
                    const size_t comma = v.find(',', pos);
                    const std::string kv = v.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                    const size_t eq = kv.find('=');
                    if (eq != std::string::npos && eq > 0) {
                        (void)::setenv(kv.substr(0, eq).c_str(), kv.c_str() + eq + 1, 1);
                        touched.insert(kv.substr(0, eq));
                    }
                    if (comma == std::string::npos) break;
                    pos = comma + 1;
                }
                std::printf("SWEEP run %d capture %zu layer %u variant [%s]\n", run++, c, g_replay->hd.layer, v.c_str());
                std::fflush(stdout);
                rc |= compare_mode(model, *gpu, owned, g_replay->hd.layer, g_replay->hd.rows, lane_classes,
                                   lane_large_rows, false);
                std::fflush(stdout);
            }
        }
        return rc;
    }
    if (compare) {
        if (compare_one_layer)
            return compare_mode(model, *gpu, owned, compare_layer, compare_rows, lane_classes,
                                lane_large_rows, synthetic_protect);
        int rc = 0;
        for (uint32_t layer : {0u, 20u, 34u, 47u}) {
            if (layer >= model.layers) {
                std::fprintf(stderr, "r3_moe_worker: --compare layer %u unavailable (layers=%u)\n",
                             layer, model.layers);
                rc = 1;
                continue;
            }
            std::printf("layer %u\n", layer);
            rc |= compare_mode(model, *gpu, owned, layer, compare_rows, lane_classes,
                               lane_large_rows, synthetic_protect);
        }
        return rc;
    }

    r3::moe_shm::LayoutShape shape{};
    Layout lay{};
    if (!segment_shape(model, &shape, &lay)) return 1;
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
        std::fprintf(stderr, "r3_moe_worker: shm_open %s: %s\n", name, std::strerror(errno));
        return 1;
    }
    if (creator && ftruncate(fd, (off_t)lay.bytes) != 0) {
        std::fprintf(stderr, "r3_moe_worker: ftruncate: %s\n", std::strerror(errno));
        drop_name(); close(fd); return 1;
    }
    void* map = mmap(nullptr, lay.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        std::fprintf(stderr, "r3_moe_worker: mmap: %s\n", std::strerror(errno));
        drop_name(); close(fd); return 1;
    }
    auto* h = static_cast<Header*>(map);
    if (creator) {
        if (!init(map, model.layers, shape)) { munmap(map, lay.bytes); close(fd); drop_name(); return 1; }
        h->small_class_count = lane_classes.count;
        for (uint32_t c = 0; c < lane_classes.count; ++c) h->small_class_rows[c] = lane_classes.rows[c];
        /* Every helper reads the same step activations: stage them once, in slot 0. */
        for (uint32_t r = 1; r < helper_count; ++r) h->activation_offset[r] = h->activation_offset[0];
        store(&h->helper_count, helper_count);   /* last: an attaching helper waits for it */
    } else {
        for (int i = 0; i < 1800 && !g_stop && load(&h->helper_count) != helper_count; ++i) ::sleep(1);
        bool same_lanes = load(&h->helper_count) == helper_count && valid(h, lay.bytes, shape) && h->small_class_count == lane_classes.count;
        for (uint32_t c = 0; same_lanes && c < lane_classes.count; ++c)
            same_lanes = h->small_class_rows[c] == lane_classes.rows[c];
        if (!same_lanes) {
            std::fprintf(stderr, "r3_moe_worker: segment %s is not helper 0's %u-helper segment (small lanes %u, ours %s)\n",
                         name, helper_count, (unsigned)h->small_class_count, lane_list.c_str());
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

    std::fprintf(stderr, "r3_moe_worker: waiting for engine runtime expert maps\n");
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
            std::fprintf(stderr, "r3_moe_worker: layer %u has no runtime map (not declared by the engine); not owned\n",
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
            std::fprintf(stderr, "r3_moe_worker: layer %u runtime table build failed\n", layer);
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
        const int64_t eb = laid_out_bytes(model, gpu.get(), l, e);
        if (eb > 0) runtime_owned.bytes += (uint64_t)eb;
    }
    owned = std::move(runtime_owned);
    h->owned_bytes = owned.bytes;
    if (gpu && !gpu->prepare_lanes(lane_classes.rows, lane_classes.count, lane_large_rows)) {
        (void)hipHostUnregister(map);
        munmap(map, lay.bytes); close(fd); drop_name(); return 1;
    }
    if (gpu) (void)posix_fadvise(model.fd, 0, 0, POSIX_FADV_DONTNEED);
    std::fprintf(stderr, "r3_moe_worker: runtime map owns %zu experts, %.3f GiB\n",
                 owned.expert.size(), (double)owned.bytes / (1024.0 * 1024.0 * 1024.0));

    /* No SA_RESTART (std::signal sets it): the futex wait must return EINTR so the loop sees g_stop. */
    struct sigaction sa{};
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr); sigaction(SIGTERM, &sa, nullptr);
    std::fprintf(stderr,
                 "r3_moe_worker: ready shm=%s layers=%u bytes=%llu (%s, small lane rows <= %s, large <= %u)\n",
                 name, model.layers, (unsigned long long)lay.bytes,
                 cpu ? "host fp32 reference" : gpu->arch().c_str(), lane_list.c_str(), lane_large_rows);

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
            /* Two helpers: each serves only its own rank slot's lanes. */
            if (helper_count > 1 && lane % kMaxRanks != helper_index) continue;
            if (lane_is_small(lane) && lane_class(lane) >= lane_classes.count) continue;   /* a disabled small class */
            workers[lane] = std::thread([&, lane] {
                const bool large = !lane_is_small(lane);
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
                        capture_request(map, q, lane, next);
                        trace_dump(map, lane, next - 1u);
                        /* Every large request is timed by stage: they are few, and the engine cannot send
                         * the next one until it has merged this one, so waiting here costs nothing. */
                        const int rc = gpu->enqueue_lane(map, dev_map, lane, next, rows, true);
                        if (rc != 0) {
                            std::fprintf(stderr, "r3_moe_worker: large lane %u seq %u rows %u failed rc=%d\n",
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
                        const int rc = gpu->enqueue_lane(map, dev_map, lane, next, lane_classes.rows[lane_class(lane)], sample);
                        if (rc != 0) {
                            std::fprintf(stderr,
                                         "r3_moe_worker: lane %u enqueue seq %u failed rc=%d\n",
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
                        trace_dump(map, lane, done - 1u);
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
        for (uint32_t c = 0; c < lane_classes.count; ++c) {
            uint64_t req = 0, rows = 0;
            for (uint32_t r = 0; r < kMaxRanks; ++r) {
                req += lane_req[lane_of(r, c)];
                rows += lane_rows[lane_of(r, c)];
            }
            std::fprintf(stderr, "r3_moe_worker: small lane class %u (rows <= %u): requests %llu (%llu rows)\n", c,
                         lane_classes.rows[c], (unsigned long long)req, (unsigned long long)rows);
        }
        std::fprintf(stderr, "r3_moe_worker: large-lane requests %llu (%llu rows)\n",
                     (unsigned long long)(lane_req[lane_of(0, kSmallClasses)] + lane_req[lane_of(1, kSmallClasses)]),
                     (unsigned long long)(lane_rows[lane_of(0, kSmallClasses)] + lane_rows[lane_of(1, kSmallClasses)]));
        if (timing_samples && stat_req)
            stat_us = (sampled_us / (double)timing_samples) * (double)stat_req;
        std::fprintf(stderr,
                     "r3_moe_worker: GPU timing samples=%llu sampled=%.1f ms avg=%.1f us\n",
                     (unsigned long long)timing_samples, sampled_us / 1000.0,
                     timing_samples ? sampled_us / (double)timing_samples : 0.0);

    }

    std::fprintf(stderr, "r3_moe_worker: served %llu requests, %llu rows, %llu routes, compute %.1f ms (%.1f us a request)\n",
                 (unsigned long long)stat_req, (unsigned long long)stat_rows, (unsigned long long)stat_routes,
                 stat_us / 1000.0, stat_req ? stat_us / (double)stat_req : 0.0);
    if (!cpu) (void)hipHostUnregister(map);
    munmap(map, lay.bytes);
    close(fd);
    drop_name();
    return 0;
}
