#pragma once

// Host reference: row decode, quantisation and the CPU expert computation, shm name. Included only by r3_moe_worker.cpp, at the position the code had in the single file (RCS03, pure move).

namespace {
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

/* The plain activation quantiser of FP8 block-scaled weights (radiance's quant_act_fp8): per row, per 128 columns, E4M3 with an f32 scale
 * of amax / 448 and no rotation (the Hadamard twin exists for four-bit weights only). Returns the dequantised values. */
bool quant_plain_engine(float* v, size_t n) {
    if (!v || n == 0 || n % 128 != 0) return false;
    for (size_t b = 0; b < n; b += 128) {
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

int compute_host_engine(const Container& f, void* map, Request* q,
                        std::vector<uint16_t>* route_down = nullptr) {
    const uint32_t hid = static_cast<const Header*>(map)->hidden, topk = static_cast<const Header*>(map)->top_k;
    const int64_t hdr_intermediate = static_cast<const Header*>(map)->intermediate;
    if (q->rank_count != kMaxEngineRanks || q->rank >= kMaxSlots || q->layer >= f.layers ||
        q->rows == 0 || q->rows > kMaxRows || q->hidden != hid ||
        q->route_count > kMaxRoutes || q->activation_stride != hid * 2)
        return -EINVAL;

    const Route* routes = ptr<Route>(map, q->route_offset);
    const uint16_t* act = ptr<uint16_t>(map, q->activation_offset);
    uint16_t* dst = ptr<uint16_t>(map, q->result_offset[q->rank]);

    std::vector<float> aq((size_t)q->rows * hid);
    for (size_t i = 0; i < aq.size(); ++i) aq[i] = bf16(act[i]);
    for (uint32_t t = 0; t < q->rows; ++t)
        if (!quant_had_engine(aq.data() + (size_t)t * hid, hid)) return -EINVAL;

    std::vector<std::vector<uint32_t>> by_exp(f.experts);
    std::vector<int32_t> slot_route((size_t)q->rows * topk, -1);
    for (uint32_t i = 0; i < q->route_count; ++i) {
        const Route& r = routes[i];
        if (r.token < 0 || (uint32_t)r.token >= q->rows || r.expert < 0 ||
            r.expert >= (int32_t)f.experts || r.slot >= topk)
            return -EINVAL;
        const size_t fslot = (size_t)r.token * topk + r.slot;
        if (slot_route[fslot] >= 0) return -EINVAL;
        slot_route[fslot] = (int32_t)i;
        by_exp[(uint32_t)r.expert].push_back(i);
    }

    std::vector<uint16_t> down((size_t)q->route_count * hid, 0);
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
        if (guc != hid || dnr != hid || gur != 2 * dnc || dnc != hdr_intermediate ||
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
                const float* x = aq.data() + (size_t)r.token * hid;
                const float gate = bf16(to_bf16(dot_f32(x, wg)));
                const float up = bf16(to_bf16(dot_f32(x, wu)));
                mid[a * 384u + (size_t)j] =
                    bf16(to_bf16((gate / (1.0f + ::expf(-gate))) * up));
            }
        }
        for (size_t a = 0; a < ri.size(); ++a)
            if (!quant_had_engine(mid.data() + a * 384u, 384u)) return -EINVAL;

        for (int64_t h = 0; h < hid; ++h) {
            if (!decode_row(f, dn, h, wd, false)) return -ENOTSUP;
            for (size_t a = 0; a < ri.size(); ++a) {
                float sum = 0.0f;
                const float* m = mid.data() + a * 384u;
                for (int64_t j = 0; j < width; ++j) sum += m[j] * wd[(size_t)(lo + j)];
                down[(size_t)ri[a] * hid + (size_t)h] = to_bf16(sum);
            }
        }
    }

    if (route_down) *route_down = down;

    /* r4d_moe gather: ascending original top-k slot order, f32 accumulator, one BF16 narrow. */
    for (uint32_t t = 0; t < q->rows; ++t) {
        for (uint32_t h = 0; h < hid; ++h) {
            float acc = 0.0f;
            for (uint32_t slot = 0; slot < topk; ++slot) {
                const int32_t ri = slot_route[(size_t)t * topk + slot];
                if (ri < 0) continue;
                acc += bf16(routes[(uint32_t)ri].weight_bf16) *
                       bf16(down[(size_t)(uint32_t)ri * hid + h]);
            }
            dst[(size_t)t * hid + h] = to_bf16(acc);
        }
    }
    return 0;
}

int compute_host_engine_fp8(const Container& f, void* map, Request* q,
                        std::vector<uint16_t>* route_down = nullptr) {
    const uint32_t hid = static_cast<const Header*>(map)->hidden, topk = static_cast<const Header*>(map)->top_k;
    const int64_t hdr_intermediate = static_cast<const Header*>(map)->intermediate;
    if (q->rank_count != kMaxEngineRanks || q->rank >= kMaxSlots || q->layer >= f.layers ||
        q->rows == 0 || q->rows > kMaxRows || q->hidden != hid ||
        q->route_count > kMaxRoutes || q->activation_stride != hid * 2)
        return -EINVAL;

    const Route* routes = ptr<Route>(map, q->route_offset);
    const uint16_t* act = ptr<uint16_t>(map, q->activation_offset);
    uint16_t* dst = ptr<uint16_t>(map, q->result_offset[q->rank]);

    std::vector<float> aq((size_t)q->rows * hid);
    for (size_t i = 0; i < aq.size(); ++i) aq[i] = bf16(act[i]);
    for (uint32_t t = 0; t < q->rows; ++t)
        if (!quant_plain_engine(aq.data() + (size_t)t * hid, hid)) return -EINVAL;

    std::vector<std::vector<uint32_t>> by_exp(f.experts);
    std::vector<int32_t> slot_route((size_t)q->rows * topk, -1);
    for (uint32_t i = 0; i < q->route_count; ++i) {
        const Route& r = routes[i];
        if (r.token < 0 || (uint32_t)r.token >= q->rows || r.expert < 0 ||
            r.expert >= (int32_t)f.experts || r.slot >= topk)
            return -EINVAL;
        const size_t fslot = (size_t)r.token * topk + r.slot;
        if (slot_route[fslot] >= 0) return -EINVAL;
        slot_route[fslot] = (int32_t)i;
        by_exp[(uint32_t)r.expert].push_back(i);
    }

    std::vector<uint16_t> down((size_t)q->route_count * hid, 0);
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
        if (guc != hid || dnr != hid || gur != 2 * dnc || dnc != hdr_intermediate)
            return -EINVAL;

        const int32_t es = layer_slot(map, q->layer, e);
        if (es < 0) {
            std::fprintf(stderr, "r3_moe_worker: fp8 reference: layer %u expert %u has no runtime slot but is routed\n", q->layer, e);
            return -ENOTSUP;
        }
        (void)es;
        const int64_t lo = 0;                 // FP8 experts are served full width, whole experts
        const int64_t width = dnc;
        const size_t mw = (size_t)dnc;
        std::vector<float> mid(ri.size() * mw, 0.0f);

        for (int64_t j = 0; j < width; ++j) {
            if (!decode_row(f, gu, lo + j, wg, false) ||
                !decode_row(f, gu, dnc + lo + j, wu, false)) {
                std::fprintf(stderr, "r3_moe_worker: fp8 reference: gate/up row %lld of expert %u (layer %u) cannot be decoded\n", (long long)j, e, q->layer);
                return -ENOTSUP;
            }
            for (size_t a = 0; a < ri.size(); ++a) {
                const Route& r = routes[ri[a]];
                const float* x = aq.data() + (size_t)r.token * hid;
                const float gate = bf16(to_bf16(dot_f32(x, wg)));
                const float up = bf16(to_bf16(dot_f32(x, wu)));
                mid[a * mw + (size_t)j] =
                    bf16(to_bf16((gate / (1.0f + ::expf(-gate))) * up));
            }
        }
        for (size_t a = 0; a < ri.size(); ++a)
            if (!quant_plain_engine(mid.data() + a * mw, mw)) return -EINVAL;

        for (int64_t h = 0; h < hid; ++h) {
            if (!decode_row(f, dn, h, wd, false)) {
                std::fprintf(stderr, "r3_moe_worker: fp8 reference: down row %lld of expert %u (layer %u) cannot be decoded\n", (long long)h, e, q->layer);
                return -ENOTSUP;
            }
            for (size_t a = 0; a < ri.size(); ++a) {
                float sum = 0.0f;
                const float* m = mid.data() + a * mw;
                for (int64_t j = 0; j < width; ++j) sum += m[j] * wd[(size_t)(lo + j)];
                down[(size_t)ri[a] * hid + (size_t)h] = to_bf16(sum);
            }
        }
    }

    if (route_down) *route_down = down;

    /* r4d_moe gather: ascending original top-k slot order, f32 accumulator, one BF16 narrow. */
    for (uint32_t t = 0; t < q->rows; ++t) {
        for (uint32_t h = 0; h < hid; ++h) {
            float acc = 0.0f;
            for (uint32_t slot = 0; slot < topk; ++slot) {
                const int32_t ri = slot_route[(size_t)t * topk + slot];
                if (ri < 0) continue;
                acc += bf16(routes[(uint32_t)ri].weight_bf16) *
                       bf16(down[(size_t)(uint32_t)ri * hid + h]);
            }
            dst[(size_t)t * hid + h] = to_bf16(acc);
        }
    }
    return 0;
}

int compute_host(const Container& f, void* map, Request* q) {
    const uint32_t hid = static_cast<const Header*>(map)->hidden, topk = static_cast<const Header*>(map)->top_k;
    const int64_t hdr_intermediate = static_cast<const Header*>(map)->intermediate;
    if (q->rank_count != kMaxEngineRanks || q->rank >= kMaxSlots || q->layer >= f.layers || q->rows > kMaxRows ||
        q->hidden != hid || q->route_count > kMaxRoutes || q->activation_stride != hid*2) return -EINVAL;
    const Route* routes = ptr<Route>(map,q->route_offset);
    const uint16_t* act = ptr<uint16_t>(map,q->activation_offset);
    uint16_t* dst = ptr<uint16_t>(map,q->result_offset[q->rank]);
    std::vector<float> out((size_t)q->rows*hid,0.f);
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
        if(guc!=hid || dnr!=hid || gur!=2*dnc || dnc!=hdr_intermediate) return -EINVAL;
        const int32_t es=layer_slot(map,q->layer,e); if(es<0) return -ENOTSUP;
        const int parity=(int)((uint32_t)es&1u); const bool wide=((int)q->rank==parity); const int64_t lo=wide?0:384, hi=wide?384:640, width=hi-lo;
        std::vector<float> mid(ri.size()*(size_t)width);
        for(int64_t j=0;j<width;++j) {
            if(!decode_row(f,gu,lo+j,wg) || !decode_row(f,gu,dnc+lo+j,wu)) return -ENOTSUP;
            for(size_t a=0;a<ri.size();++a) {
                const Route& r=routes[ri[a]]; const uint16_t* x=act+(size_t)r.token*hid;
                mid[a*(size_t)width+(size_t)j]=silu(dot_bf16(x,wg))*dot_bf16(x,wu);
            }
        }
        for(int64_t h=0;h<hid;++h) {
            if(!decode_row(f,dn,h,wd)) return -ENOTSUP;
            for(size_t a=0;a<ri.size();++a) {
                float s=0.f; for(int64_t j=0;j<width;++j) s += mid[a*(size_t)width+(size_t)j]*wd[(size_t)(lo+j)];
                const Route& r=routes[ri[a]]; out[(size_t)r.token*hid+(size_t)h] += s*bf16(r.weight_bf16);
            }
        }
    }
    for(size_t i=0;i<out.size();++i) dst[i]=to_bf16(out[i]);
    return 0;
}

const char* shm_name() { const char* e=std::getenv("R3_MOE_WORKER_SHM"); return e&&*e?e:"/r3-moe-helper"; }
} // namespace
