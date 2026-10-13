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
            !range(h->enc_off, h->enc_count * sizeof(RadEncoding))) return false;
        dir = (const RadFileEntry*)(base + h->dir_off); planes = (const RadFilePlane*)(base + h->plane_off);
        encs = (const RadEncoding*)(base + h->enc_off);
        int32_t max_layer = -1;
        for (uint64_t i = 0; i < h->dir_count; ++i) if (dir[i].layer >= 0) max_layer = std::max(max_layer, dir[i].layer);
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

int compute(const Container& f, void* map, Request* q) {
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

int main(int argc,char** argv) {
    if(argc!=2) { std::fprintf(stderr,"usage: r3_moe_helper MODEL.rad\n"); return 2; }
    Container model; if(!model.open(argv[1])) { std::fprintf(stderr,"r3_moe_helper: cannot open/validate %s\n",argv[1]); return 1; }
    const Layout l=layout(); const char* name=shm_name();
    int fd=shm_open(name,O_CREAT|O_EXCL|O_RDWR|O_CLOEXEC,0600);
    if(fd<0) { std::fprintf(stderr,"r3_moe_helper: shm_open %s: %s\n",name,std::strerror(errno)); return 1; }
    if(ftruncate(fd,(off_t)l.bytes)!=0) { std::fprintf(stderr,"r3_moe_helper: ftruncate: %s\n",std::strerror(errno)); shm_unlink(name); return 1; }
    void* map=mmap(nullptr,l.bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(map==MAP_FAILED) { std::fprintf(stderr,"r3_moe_helper: mmap: %s\n",std::strerror(errno)); shm_unlink(name); return 1; }
    init(map,model.layers); auto* h=(Header*)map;
    std::signal(SIGINT,on_signal); std::signal(SIGTERM,on_signal);
    std::fprintf(stderr,"r3_moe_helper: ready shm=%s layers=%u bytes=%llu (host fp32 reference)\n",name,model.layers,(unsigned long long)l.bytes);
    uint32_t seq=1;
    while(!g_stop) {
        Request* q=request(map,seq);
        while(!g_stop && load(&q->sequence)!=seq) {
            uint32_t seen=load(&h->publish_sequence); if(load(&q->sequence)!=seq) futex_wait(&h->publish_sequence,seen);
        }
        if(g_stop) break;
        q->status=compute(model,map,q); store(&q->completion_sequence,seq); futex_wake(&q->completion_sequence);
        if(++seq==0) break;
    }
    munmap(map,l.bytes); close(fd); shm_unlink(name); return 0;
}
