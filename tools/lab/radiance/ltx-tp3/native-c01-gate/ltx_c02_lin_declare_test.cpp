/* Diagnostic actual LTX2 Lin::declare exercise against Radiance 2.2 / ABI44.
 * Test includes the patched architecture implementation: no alternative Lin,
 * plugin launch, real GPU, or claim of TP3 service.
 * Only the weight/activation declaration and the real core's op registry run.
 */
#include "build/rad_build.h"
#include "build/startup.h"
#include "format/radfile.h"
#include "format/share.h"
#include "device/device.h"
#include "runtime/ctx.h"
#include "ltx_native_aql_roster.h"
#include "ltx2.cpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <thread>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using rad::Builder;
using rad::RadFile;
using rad::Registry;

static void need(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}
static RadBuildCtx context(int rank, int world = 3, int64_t max_tok = 4) {
    RadBuildCtx c{};
    c.rank = rank;
    c.world_size = world;
    c.max_tok = max_tok;
    c.max_seqs = 1;
    c.max_ctx = 4096;
    c.scope = "";
    return c;
}
static RadModelMeta metadata() {
    RadModelMeta m{};
    m.arch_id = "ltx2";
    m.name = "C02 actual Lin declaration receipt";
    return m;
}

static void check_parts(const rad::WeightInfo& w, const ltx2::C01Slice& slice) {
    need(w.decl.shard == RAD_SHARD_ROW, "expected row shard");
    if (slice.banks == 1) {
        need(w.n_part_spans == 0 && w.shard_lo == slice.offset &&
             w.shard_hi == slice.offset + slice.width,
             "one-bank Q weight lost native row span");
        return;
    }
    need(w.n_part_spans == slice.banks, "actual Lin lost bank count");
    need(w.decl.n_row_parts == slice.banks, "actual Lin lost row parts");
    for(int k=0; k<slice.banks; ++k) {
        need(w.part_lo[k]==slice.offset &&
             w.part_hi[k]==slice.offset+slice.width &&
             w.part_total[k]==slice.source_width &&
             w.decl.row_parts[k]==slice.width,
             "actual Lin bank span disagrees with model ownership");
    }
}
static void check_col(const rad::WeightInfo& w, const ltx2::C01Slice& slice) {
    need(w.decl.shard == RAD_SHARD_COL, "expected column shard");
    need(w.shard_lo == slice.offset && w.shard_hi == slice.offset+slice.width,
         "actual Lin column span disagrees with model ownership");
}

constexpr const char* kBiasedOutputs[] = {
    "attn1.out", "audio_attn1.out", "attn2.out",
    "audio_attn2.out", "a2v.out", "v2a.out", "audio_ff_down"
};
enum class ProbeCase {
    Normal, WrongGroup, PrematureBias, InvalidColumn,
    DeferredBias, InvalidDeferred, DuplicateDeferredBias, Reduced, GlobalRms
};
static void run(const RadFile& model, Registry& reg, const std::string& base,
                int banks, int nheads, int rank, int start_heads, int owned_heads,
                ProbeCase variant = ProbeCase::Normal) {
    const bool rows = banks > 0;
    const std::string weight = base + ".weight";
    const RadFileEntry* e = model.find(weight);
    need(e && e->rank == 2, "required model projection unavailable");
    const RadEncoding& enc = model.encoding(*e);
    need(rad::arch::linear_is_iq8r(enc), "real source is not IQ8R");
    const int64_t group = rad::arch::linear_w8h_group(enc);
    const int64_t hd = (rows ? e->shape[0]/banks : e->shape[1])/nheads;
    const int64_t full = hd * nheads, offset = hd*start_heads, width = hd*owned_heads;
    need(!rows || e->shape[0] == banks*full, "source banks are not equally sized");
    const bool deferred = variant == ProbeCase::DeferredBias || variant == ProbeCase::InvalidDeferred || variant == ProbeCase::DuplicateDeferredBias || variant == ProbeCase::Reduced;
    const bool denied = variant == ProbeCase::WrongGroup || variant == ProbeCase::PrematureBias || variant == ProbeCase::InvalidColumn || variant == ProbeCase::InvalidDeferred || variant == ProbeCase::DuplicateDeferredBias;
    const auto slice = rows
        ? ltx2::C01Slice::rows(full,offset,width,hd,banks)
        : deferred ? ltx2::C01Slice::columns_deferred(e->shape[1],offset,width)
                   : ltx2::C01Slice::columns(e->shape[1],offset,width);
    need(denied || slice.valid(rows?banks*width:e->shape[0],
                                   rows?e->shape[1]:width,group),
         "test generated invalid positive C01 geometry");

    Builder b(reg,metadata(),context(rank));
    b.set_sources(rad::rad_container_sources(model));
    ltx2::Names nm;
    ltx2::Geom g{};
    g.w_i8h = true;
    g.w_i8_group = variant == ProbeCase::WrongGroup ? group/2 : group;
    g.w_bf16 = false;
    g.world = 3;
    g.wire_exact = 1;
    g.max_tok = 4;
    g.dtype = "bf16";
    ltx2::Act input{};
    input.n = rows ? e->shape[1] : width;
    input.a.x = ltx2::decl_b(&b,nm.f("probe.x"),RAD_BF16,{4,input.n});
    input.a.q8 = ltx2::decl_b(&b,nm.f("probe.codes"),RAD_I8,{4,input.n});
    input.a.s8 = ltx2::decl_b(&b,nm.f("probe.scales"),RAD_F32,{4,input.n/group});
    input.a.g8 = group;
    input.a.q8_fed = true;
    input.a.q8_rot = true;
    const int64_t out_dim = rows ? banks*width : e->shape[0];
    const rad_buf y = ltx2::decl_b(&b,nm.f("probe.y"),RAD_BF16,{4,out_dim});
    need(input.a.x && input.a.q8 && input.a.s8 && y,"cannot declare activation buffers");

    ltx2::Lin linear{};
    int rc;
    if (banks == 3) {
        rc = linear.declare(&b,nm,g,base.c_str(),
             {"checkpoint.to_q.weight","checkpoint.to_k.weight","checkpoint.to_v.weight"},
             {"checkpoint.to_q.bias","checkpoint.to_k.bias","checkpoint.to_v.bias"},
             {full,full,full},out_dim,e->shape[1],RAD_ACCESS_PER_TOKEN,0,input,y,4,false,&slice);
    } else if(banks == 2) {
        rc = linear.declare(&b,nm,g,base.c_str(),
             {"checkpoint.to_k.weight","checkpoint.to_v.weight"},
             {"checkpoint.to_k.bias","checkpoint.to_v.bias"},
             {full,full},out_dim,e->shape[1],RAD_ACCESS_PER_TOKEN,0,input,y,4,false,&slice);
    } else if (variant == ProbeCase::DuplicateDeferredBias) {
        rc = linear.declare(&b,nm,g,base.c_str(),
             {"checkpoint.ff_down.weight"},
             {"checkpoint.to_out.bias","checkpoint.to_out.bias"},{},
             out_dim,width,RAD_ACCESS_PER_TOKEN,0,input,y,4,false,&slice);
    } else if(variant == ProbeCase::PrematureBias || deferred) {
        const std::string bias_name = base + ".bias";
        need(model.find(bias_name) != nullptr, "source bias missing");
        // These aliases are checkpoint-side convert names; the native RAD2
        // builder resolves the declared 'base.weight' and 'base.bias' directly.
        // Mapping a declared native name to itself would form a map cycle.
        rc = linear.declare(&b,nm,g,base.c_str(),
             {"checkpoint.to_out.weight"},{"checkpoint.to_out.bias"},{},
             out_dim,width,RAD_ACCESS_PER_TOKEN,0,input,y,4,false,&slice);
    } else {
        rc = linear.declare(&b,nm,g,base.c_str(),
             {"checkpoint.ff_down.weight"},{},{},
             out_dim,width,RAD_ACCESS_PER_TOKEN,0,input,y,4,false,&slice);
    }
    if (denied) {
        need(rc==RAD_E_UNSUPPORTED, "unsupported C01 projection did not fail closed");
        need(b.program().weights.size()==1,"refused slice still allocated model weights");
        return;
    }
    if(rc!=RAD_OK) {
        std::fprintf(stderr,"Lin::declare rc=%d, rank=%d, shape=%lldx%lld, l.i8=%d, op=%u, w=%u, bias=%u, defer=%d, src=%s\n",
                     rc,rank,(long long)out_dim,(long long)input.n,
                     (int)linear.l.i8,(unsigned)linear.op,
                     (unsigned)linear.l.w,(unsigned)linear.bias,
                     (int)linear.deferred_bias,weight.c_str());
        for(const auto& msg:b.errors()) std::fprintf(stderr,"  %s\n",msg.c_str());
    }
    need(rc==RAD_OK,"real LTX Lin::declare refused the shard");
    need(linear.deferred_bias == deferred, "output bias-defer contract lost");
    if (variant == ProbeCase::Reduced) {
        need(!linear.bias && !linear.op_b && !linear.post_reduce,
             "C04 partial projection applied bias before collective");
        ltx2::Geom wrong = g;
        wrong.world = 1;
        need(linear.finalize_deferred_output(&b, nm, wrong, base.c_str(),
                                            RAD_ACCESS_PER_TOKEN, 0, y, 4) ==
                 RAD_E_UNSUPPORTED,
             "C04 admitted a non-TP3 collective");
        wrong = g;
        wrong.wire_exact = 0;
        need(linear.finalize_deferred_output(&b, nm, wrong, base.c_str(),
                                            RAD_ACCESS_PER_TOKEN, 0, y, 4) ==
                 RAD_E_UNSUPPORTED,
             "C04 silently ignored the configured lossy wire");
        need(!linear.bias && !linear.op_b && !linear.post_reduce,
             "rejected C04 geometry mutated the builder");
        need(linear.finalize_deferred_output(&b, nm, g, base.c_str(),
                                            RAD_ACCESS_PER_TOKEN, 0, y, 4) == RAD_OK,
             "C04 exact reduction and post-reduce bias declaration failed");
        need(linear.bias && linear.op_b && linear.post_reduce,
             "C04 omitted reduce or full-width BF16 bias");
        need(b.program().weight(linear.bias)->decl.shard == RAD_SHARD_NONE,
             "C04 BF16 bias must be full-width after sum");
        const auto weights_before = b.program().weights.size();
        need(linear.finalize_deferred_output(&b, nm, g, base.c_str(),
                                            RAD_ACCESS_PER_TOKEN, 0, y, 4) ==
                 RAD_E_UNSUPPORTED,
             "C04 accepted a second bias finalization");
        need(b.program().weights.size() == weights_before,
             "duplicate C04 reduction allocated a second bias");
        for (const rad_op handle : {linear.post_reduce, linear.op_b})
            need(b.program().op(handle) != nullptr,
                 "C04 collective or bias missing from native graph");
    }
    if (variant == ProbeCase::GlobalRms) {
        need(rows && (banks == 2 || banks == 3),
             "C05 global Q/K norm needs a real fused QKV/KV projection");
        // Native RAD2 supplies BOTH full-width gains. A KV-only probe uses
        // a local-Q activation stand-in, not another projection.
        const std::string attn = base.substr(0, base.rfind('.'));
        const std::string qname = attn + ".q_norm.weight";
        const std::string kname = attn + ".k_norm.weight";
        for (const auto& name : {qname, kname}) {
            const RadFileEntry* gain_source = model.find(name);
            need(gain_source && gain_source->rank == 1 &&
                 gain_source->shape[0] == full,
                 "C05 source-backed global Q/K gain has wrong shape");
        }
        const rad_weight wq = ltx2::decl_w(
            &b, nm.f("%s", qname.c_str()), RAD_BF16, {full},
            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, ltx2::grp_layer(0));
        const rad_weight wk = ltx2::decl_w(
            &b, nm.f("%s", kname.c_str()), RAD_BF16, {full},
            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, ltx2::grp_layer(0));
        need(wq && wk, "C05 full Q/K gain declaration failed");
        const rad_buf scratch = ltx2::decl_b(
            &b, nm.f("c05.global.scratch"), RAD_BF16, {4,full});
        const rad_buf query = banks == 3 ? y : ltx2::decl_b(
            &b, nm.f("c05.query.local"), RAD_BF16, {4,width});
        need(scratch && query, "C05 missing Q/K buffers");
        ltx2::Tp3GlobalRms nq{}, nk{}, negative{};
        ltx2::Geom wrong = g;
        wrong.world = 1;
        need(negative.declare(&b, wrong, query, scratch, wq, full,
                              offset, width, 0, hd, 4, 1e-6) ==
             RAD_E_UNSUPPORTED, "C05 allowed non-TP3 world");
        wrong = g;
        wrong.wire_exact = 0;
        need(negative.declare(&b, wrong, query, scratch, wq, full,
                              offset, width, 0, hd, 4, 1e-6) ==
             RAD_E_UNSUPPORTED, "C05 silently accepted lossy wire");
        need(negative.declare(&b, g, query, scratch, wq, full,
                              offset+1, width, 0, hd, 4, 1e-6) ==
             RAD_E_UNSUPPORTED, "C05 allowed fractional head span");
        need(!negative.clear && !negative.scratch,
             "C05 rejected geometry left builder state behind");
        need(nq.declare(&b, g, query, scratch, wq, full,
                        offset, width, 0, hd, 4, 1e-6) == RAD_OK,
             "C05 source-backed Q all-head normalization failed");
        need(nk.declare(&b, g, y, scratch, wk, full,
                        offset, width, banks == 3 ? width : 0,
                        hd, 4, 1e-6) == RAD_OK,
             "C05 source-backed K all-head normalization failed");
        need(nq.scratch == nk.scratch && nq.scratch == scratch &&
             nq.full == full && nk.full == full && nq.local == width &&
             nk.local == width && nq.lo == offset && nk.lo == offset,
             "C05 duplicated scratch or lost rank geometry");
        for (const rad_weight h : {wq,wk})
            need(b.program().weight(h)->decl.shard == RAD_SHARD_NONE,
                 "C05 gain must cover all heads");
        for (const rad_op h : {nq.clear,nq.insert,nq.reduce,nq.norm,nq.extract,
                               nk.clear,nk.insert,nk.reduce,nk.norm,nk.extract}) {
            need(h && b.program().op(h) != nullptr,
                 "C05 omitted an existing Radiance operation");
            for (const auto& band : b.program().op(h)->bands)
                need(static_cast<bool>(band.dom[RAD_DOMAIN_HOST]),
                     "C05 native host kernel unavailable");
        }
        need(nq.declare(&b, g, query, scratch, wq, full,
                        offset, width, 0, hd, 4, 1e-6) ==
             RAD_E_UNSUPPORTED, "C05 allowed repeated declaration");
    }
    if (deferred && variant != ProbeCase::Reduced)
        need(!linear.bias && !linear.op_b,
             "partial output declared bias before collective");
    need(linear.l.i8 && linear.l.i8_rot && linear.op,
         "actual Lin did not select rotated IQ8R GEMM");
    need(b.errors().empty(),"native builder errors in actual Lin");
    for (const auto& band : b.program().op(linear.op)->bands)
        need(static_cast<bool>(band.dom[RAD_DOMAIN_HOST]),
             "real native host GEMM kernel unresolved");
    if (linear.op_b)
        for (const auto& band : b.program().op(linear.op_b)->bands)
            need(static_cast<bool>(band.dom[RAD_DOMAIN_HOST]),
                 "real native host bias-add kernel unresolved");
    if (variant == ProbeCase::Reduced)
        for (const auto& band : b.program().op(linear.post_reduce)->bands)
            need(static_cast<bool>(band.dom[RAD_DOMAIN_HOST]),
                 "real native host exact all-reduce kernel unresolved");
    b.set_instances(false);
    need(b.finish()==RAD_OK, "native graph finish failed");
    need(b.errors().empty(),"native graph finish returned builder errors");
    need(linear.l.w && linear.l.ws, "actual Lin omitted code/scale weights");
    const auto* cw=b.program().weight(linear.l.w);
    const auto* sw=b.program().weight(linear.l.ws);
    need(cw && sw && cw->enc_known && sw->enc_known && sw->n_sel==2,
         "real Lin IQ8R code/scale model views unavailable");
    if(rows) {
        need(linear.bias && linear.op_b,"fused Lin omitted bias or table_add");
        check_parts(*cw,slice);
        check_parts(*sw,slice);
        check_parts(*b.program().weight(linear.bias),slice);
    } else {
        if (variant != ProbeCase::Reduced)
            need(!linear.bias && !linear.op_b,
                 "partial or biasless Lin unexpectedly added bias");
        check_col(*cw,slice);
        check_col(*sw,slice);
    }
}


// A *test-only* issued C06 lane. The production Act/Lin/attn_core and
// Radiance's selected native relayout, Ctx and exact collective do all work.
struct C06Issued {
    ltx2::Attn attn{};
    ltx2::Act x{}, ao{}, kv_in{};
    rad_buf qkv=0, gate=0, output=0;
    std::array<int32_t,2> cu{0,1};
    std::vector<float> pos;
    std::vector<std::vector<uint8_t>> weights;
    // Optional diagnostic snapshots. Captured at the actual issue point,
    // before the arena planner may reuse the intermediate's storage.
    struct Capture {
        std::string stage, extension;
        std::vector<uint8_t> data;
    };
    std::vector<Capture> stages;
    std::vector<void*> owned_device;
    void* device_cu=nullptr;
    void* device_pos=nullptr;
    std::unique_ptr<rad::Ctx> ctx;
    int rank=0;
    bool audio=false, gpu=false;
    int64_t width=0, tokens=1, video_rows=1;
    // LTX03 issued feed-forward lane (the production Lin/Act/unary do all the work).
    struct Ffn {
        ltx2::Lin up{}, down{};
        ltx2::Act ff{};
        rad_op act=0;
        rad_buf upbuf=0;
        bool reduce=true;   // false: deliberately omit the exact reduction (negative test)
        // chain: modulate the replicated hidden state, run the FFN, gate the delta back into it
        bool chain=false;
        rad_op mod=0, gate=0;
        rad_buf res=0, tabs=0, result=0;   // hidden state, [shift|scale|gate] table, compared output
        // block: self-attention then FFN in one graph (tabs = two [shift|scale|gate] triples)
        bool pipe=false;   // FFN in two row pieces, each piece's reduction on the second lane
        bool block=false;
        rad_op mod1=0, gate1=0;
        rad_buf attn_out=0;
        bool has_attn=false;
    } ffn;
    ~C06Issued() {
        ctx.reset();
        if (!gpu) return;
        (void)rad_dev_set(rank);
        for(void* p:owned_device) rad_dev_free(p,RAD_MEM_DEVICE);
    }
};

thread_local C06Issued* c06_active=nullptr;
static uint16_t c06_sample(int64_t col) {
    // An exactly representable BF16 input, *not* a TP1 activation capture.
    const float x=float((col*7+11)%61-30)/32.f;
    uint32_t u;
    std::memcpy(&u,&x,sizeof u);
    u+=0x7fffU+((u>>16)&1U);
    return uint16_t(u>>16);
}
static bool c06_snapshot(RadCtx* c, C06Issued& r, const char* stage,
                         rad_buf buf, size_t bytes, const char* extension) {
    if (const char* enabled=std::getenv("BC_LTX_C06_CAPTURE_STAGES");
        !enabled || std::strcmp(enabled,"1")!=0) return true;
    if (!bytes) { rad_step_fail(c,"C06 invalid stage capture size"); return false; }
    const void* from=r.ctx->buf_ptr(buf);
    if (!from) { rad_step_fail(c,"C06 stage capture buffer missing"); return false; }
    std::vector<uint8_t> data(bytes);
    if (r.gpu) {
        if (rad_memcpy_async(data.data(),from,bytes,r.ctx->stream())!=RAD_OK ||
            rad_stream_sync(r.ctx->stream())!=RAD_OK) {
            rad_step_fail(c,"C06 native stage capture failed");
            return false;
        }
    } else std::memcpy(data.data(),from,bytes);
    r.stages.push_back({stage,extension,std::move(data)});
    return true;
}
static void c06_issued_step(RadCtx* c,const RadBatch*) {
    if (!c06_active) { rad_step_fail(c,"C06 rank execution owner missing"); return; }
    C06Issued& r=*c06_active;
    uint16_t* input=(uint16_t*)r.ctx->buf_ptr(r.x.a.x);
    if (!input) { rad_step_fail(c,"C06 source activation missing"); return; }
    if (!r.gpu)
        for (int64_t j=0;j<r.width*r.tokens;++j) input[j]=c06_sample(j);
    r.x.step(c,r.tokens);
    if (r.width<=0 || r.x.a.cg()<=0 || r.width%r.x.a.cg()!=0) {
        rad_step_fail(c,"C06 quantized input shape invalid"); return;
    }
    if (!c06_snapshot(c,r,"input-quant-codes",r.x.a.cq(),
                      (size_t)(r.width*r.tokens),"i8") ||
        !c06_snapshot(c,r,"input-quant-scales",r.x.a.cs(),
                      (size_t)(r.tokens*r.width/r.x.a.cg())*sizeof(float),"f32")) return;
    r.attn.qkv.step(c,r.x,r.qkv,r.tokens);
    if (!c06_snapshot(c,r,"qkv-projection",r.qkv,
                      (size_t)(r.tokens*3*r.attn.inner)*sizeof(uint16_t),"bf16")) return;
    r.attn.gate.step(c,r.x,r.gate,r.tokens);
    if (!c06_snapshot(c,r,"gate-projection",r.gate,
                      (size_t)(r.tokens*r.width/r.attn.hd)*sizeof(uint16_t),"bf16")) return;
    const float* positions=(r.gpu ? (const float*)r.device_pos : r.pos.data())+
                            (r.audio?3*r.video_rows:0);
    const RadOperand pos=ltx2::praw2(positions,RAD_F32,r.audio?1:3,r.tokens);
    ltx2::attn_core(c,r.attn,true,r.qkv,0,r.gate,r.ao,r.tokens,r.tokens,
                    ltx2::praw(r.gpu ? (const int32_t*)r.device_cu : r.cu.data(),
                               RAD_I32,2),RAD_NONE,pos,pos);
    if (!c06_snapshot(c,r,"qkv-after-norm-rope",r.qkv,
                      (size_t)(r.tokens*3*r.attn.inner)*sizeof(uint16_t),"bf16")) return;
    if (!c06_snapshot(c,r,"attention-after-gate",r.ao.a.x,
                      (size_t)(r.tokens*r.attn.inner)*sizeof(uint16_t),"bf16")) return;
    r.ao.step(c,r.tokens);
    r.attn.out.step(c,r.ao,r.output,r.tokens);
}

static std::vector<uint8_t> c06_gather(const RadFile& model,
                                        const RadFileEntry& e,int selection,
                                        const rad::PlaneShare& share) {
    const auto& plane=model.plane(e,selection);
    const uint8_t* src=model.data(plane);
    need(src!=nullptr,"C06 real source plane is not mapped");
    std::vector<uint8_t> data((size_t)share.dense_bytes());
    size_t at=0;
    for (const auto& rect:share.rects)
        for (int64_t row=0;row<rect.rows;++row) {
            const uint8_t* from=src+(rect.row0+row)*share.row_bytes+rect.byte0;
            std::memcpy(data.data()+at,from,(size_t)rect.bytes);
            at+=(size_t)rect.bytes;
        }
    need(at==data.size(),"C06 native weight share gather incomplete");
    return data;
}

// Audit the frozen runtime loader's *own* share geometry for each
// checkpoint plane. The optional issued lane keeps the *same* selected
// planes and kernel relayout output resident for the native Ctx.
static void verify_c06_load_contract(const RadFile& model, rad::Program& program,
                                      int rank,C06Issued* issued=nullptr,int world=3) {
    if (issued) {
        issued->weights.resize(program.weights.size());
        // Reserve before any device allocation: registering a successful
        // native allocation must not itself allocate or throw.
        if (issued->gpu) issued->owned_device.reserve(program.weights.size()+3);
    }
    int64_t selected_bytes = 0, stored_bytes = 0;
    int weights = 0, relayouts = 0;
    for (size_t i=1; i<program.weights.size(); ++i) {
        rad::WeightInfo& w = program.weights[i];
        const auto* e = model.find(w.source);
        need(e != nullptr && w.enc_known && w.n_sel > 0,
             "C06 declared weight lacks real checkpoint source/plane");
        rad::PlaneShare shares[RAD_ENC_MAX_PLANES];
        std::string why;
        const int rc = rad::weight_share(w,model.encoding(*e),e->rank,e->shape,
                                          world,rank,shares,&why);
        if (rc != RAD_OK)
            std::fprintf(stderr,"C06 weight %s rank %d: %s\n",
                         w.name.c_str(),rank,why.c_str());
        need(rc == RAD_OK, "C06 native source plane split refused");
        int64_t share_bytes = 0;
        for (int k=0; k<w.n_sel; ++k) {
            const rad::PlaneShare& sh=shares[k];
            const auto& plane=model.plane(*e,w.sel[k]);
            need(sh.row_bytes > 0 && sh.dense_bytes() > 0 &&
                 plane.bytes >= (uint64_t)sh.row_bytes &&
                 plane.bytes % (uint64_t)sh.row_bytes == 0,
                 "C06 source plane has inconsistent native file extent");
            int64_t covered = 0;
            for (const auto& r:sh.rects) {
                need(r.row0 >= 0 && r.rows > 0 &&
                     r.rows <= int64_t(plane.bytes/sh.row_bytes)-r.row0 &&
                     r.row0 < int64_t(plane.bytes/sh.row_bytes) &&
                     r.byte0 >= 0 && r.bytes > 0 &&
                     r.byte0 <= sh.row_bytes-r.bytes,
                     "C06 native source plane rectangle extends past checkpoint");
                covered += r.rows * r.bytes;
            }
            need(covered == sh.dense_bytes(),
                 "C06 native weight share contains gaps or duplicates");
            share_bytes += covered;
        }
        if (w.identity) {
            need(w.n_sel == 1 && share_bytes == w.stored_bytes,
                 "C06 identity layout is not a direct native file share");
            if (issued)
                issued->weights[i]=c06_gather(model,*e,w.sel[0],shares[0]);
        } else {
            // The production RelayoutPool dispatches through exactly these
            // frozen Builder-selected hook coordinates. A plain memcpy
            // from the RAD container would skip required kernel layout.
            need(!w.widen && !w.narrow &&
                 w.lay_op > 0 && (size_t)w.lay_op < program.ops.size(),
                 "C06 nonidentity weight lacks a native layout owner");
            const auto& oi = program.ops[(size_t)w.lay_op];
            need(w.lay_band >= 0 && (size_t)w.lay_band < oi.bands.size() &&
                 w.lay_dom >= 0 && w.lay_dom < RAD_N_DOMAINS,
                 "C06 nonidentity weight has invalid native layout band");
            const auto& resolved = oi.bands[(size_t)w.lay_band].dom[w.lay_dom];
            need(resolved && resolved.row && resolved.row->info &&
                 resolved.row->info->relayout,
                 "C06 nonidentity weight is missing production relayout hook");
            // Exercise the exact kernel-selected production layout hook
            // against the real checkpoint's selected native plane shares.
            // This is a small host reference fixture, not a parallel
            // streaming loader or a second IQ8R layout implementation.
            std::array<RadTensor,RAD_ENC_MAX_PLANES> source_planes{};
            rad::weight_planes(w,source_planes.data());
            std::vector<std::vector<uint8_t>> data((size_t)w.n_sel);
            for (int k=0;k<w.n_sel;++k) {
                data[(size_t)k]=c06_gather(model,*e,w.sel[k],shares[k]);
                source_planes[(size_t)k].data=data[(size_t)k].data();
            }
            const auto* ki=resolved.row->info;
            constexpr size_t guard=64;
            std::vector<uint8_t> stored((size_t)w.stored_bytes+guard,0xa5);
            const int layout_rc=ki->relayout(
                resolved.geom.params(),resolved.geom.n_params(),w.lay_operand,
                &w.enc,w.sel,source_planes.data(),w.n_sel,
                stored.data(),w.stored_bytes);
            if (layout_rc!=RAD_OK)
                std::fprintf(stderr,"C06 rank %d native %s relayout failed rc=%d\n",
                             rank,ki->name,layout_rc);
            need(layout_rc==RAD_OK,
                 "C06 native kernel relayout refused real checkpoint plane");
            bool wrote_payload = false;
            for(size_t j=0;j<(size_t)w.stored_bytes;++j)
                wrote_payload |= stored[j] != 0xa5;
            need(wrote_payload,"C06 native relayout returned success without writing payload");
            for(size_t j=(size_t)w.stored_bytes;j<stored.size();++j)
                need(stored[j]==0xa5,"C06 native relayout overwrote output guard");
            if (issued) {
                stored.resize((size_t)w.stored_bytes);
                issued->weights[i]=std::move(stored);
            }
            ++relayouts; // Cannot be directly memcpy'ed: use engine loader.
            // Record the exact kernel layout that blocks naive checkpoint
            // pointer binding in a future issue test.
            if (rank == 0)
                std::printf("C06 engine relayout required: %s -> %s (%s, source=%lld stored=%lld)\n",
                            w.source.c_str(),w.name.c_str(),w.layout_tag.c_str(),
                            (long long)share_bytes,(long long)w.stored_bytes);
        }
        if (issued) {
            need(issued->weights[i].size()==(size_t)w.stored_bytes,
                 "C06 resident model weight size mismatch");
            if (issued->gpu) {
                need(rad_dev_set(rank)==RAD_OK,"C06 native rank weight device unavailable");
                void* gpu_ptr=rad_dev_alloc(w.stored_bytes,RAD_MEM_DEVICE);
                need(gpu_ptr!=nullptr,"C06 native GPU checkpoint weight allocation failed");
                issued->owned_device.push_back(gpu_ptr);
                w.ptr=gpu_ptr;
                w.site=rad::Site::Device;
            } else {
                w.ptr=issued->weights[i].data();
                w.site=rad::Site::Host;
            }
        }
        selected_bytes += share_bytes;
        stored_bytes += w.stored_bytes;
        ++weights;
    }
    need(weights > 0 && selected_bytes > 0 && stored_bytes > 0,
         "C06 source-backed weight inventory empty");
    std::printf("C06 load contract: rank=%d weights=%d native relayouts=%d "
                "selected=%lld stored=%lld bytes\n",
                rank,weights,relayouts,
                (long long)selected_bytes,(long long)stored_bytes);
}

static void run_self_attn(const RadFile& model, Registry& reg,
                          int64_t block, bool audio, int rank, int first,
                          int owned, int total_heads,
                          std::unique_ptr<Builder>* instance = nullptr,
                          bool inspect_weights = false, bool aql = false,
                          C06Issued* capture = nullptr, int64_t tokens = 4,
                          bool text_cross = false, bool modal_cross = false) {
    const std::string kind = modal_cross ? (audio ? "v2a" : "a2v") :
                             text_cross ? (audio ? "audio_attn2" : "attn2") :
                             (audio ? "audio_attn1" : "attn1");
    const std::string prefix = "dit.blk." + std::to_string(block) + "." + kind;
    const auto* fused = model.find(prefix + (text_cross ? ".q.weight" : ".qkv.weight"));
    const auto* kv_source = text_cross ? model.find(prefix + ".kv.weight") : nullptr;
    const auto* output = model.find(prefix + ".out.weight");
    const auto* gate = model.find(prefix + ".gate.weight");
    const auto* gnq = model.find(prefix + ".q_norm.weight");
    const auto* gnk = model.find(prefix + ".k_norm.weight");
    need(fused && (!text_cross || kv_source) && output && gate && gnq && gnk,
         "C06 model missing a self-attention source");
    const int64_t full = text_cross ? fused->shape[0] : fused->shape[1];
    const int64_t qdim = fused->shape[1];
    const int64_t kvdim = text_cross ? kv_source->shape[1] : full;
    const int64_t odim = output->shape[0];
    need(fused->rank == 2 && fused->shape[0] == (text_cross ? full : 3 * full) &&
         (!text_cross || (kv_source->rank == 2 && kv_source->shape[0] == 2 * full)) &&
         full % total_heads == 0 &&
         output->rank == 2 && output->shape[1] == full &&
         gate->rank == 2 && gate->shape[0] == total_heads &&
         gate->shape[1] == qdim &&
         gnq->rank == 1 && gnq->shape[0] == full &&
         gnk->rank == 1 && gnk->shape[0] == full,
         "C06 source self-attention geometry mismatch");
    const int64_t hd = full / total_heads, local = owned * hd, lo = first * hd;
    const auto& enc = model.encoding(*fused);
    need(rad::arch::linear_is_iq8r(enc) &&
         rad::arch::linear_w8h_group(enc) == rad::arch::RAD_I8_GROUP_WIDE,
         "C06 IQ8R group encoding changed");
    const ltx2::C01Slice slice = ltx2::C01Slice::rows(full, lo, local, hd, text_cross ? 1 : 3);
    const ltx2::C01Slice kv_slice = ltx2::C01Slice::rows(full, lo, local, hd, 2);
    need(slice.valid((text_cross ? 1 : 3) * local, full, rad::arch::RAD_I8_GROUP_WIDE),
         "C06 invalid fused QKV slice");
    if (aql) need(rad_dev_set(rank) == RAD_OK, "C06 cannot select rank's AQL device");
    auto owned_builder = std::make_unique<Builder>(reg, metadata(), context(rank));
    Builder& b = *owned_builder;
    b.set_sources(rad::rad_container_sources(model));
    ltx2::Model m{};
    ltx2::Geom g{};
    g.w_i8h = true;
    g.w_i8_group = rad::arch::RAD_I8_GROUP_WIDE;
    g.world = 3;
    g.wire_exact = 1;
    g.max_tok = tokens;
    g.dtype = "bf16";
    ltx2::Act x{}, ao{}, kv_in{};
    need(x.declare(&b, m.nm, g, "c06.input", tokens, qdim) == RAD_OK &&
         ao.declare(&b, m.nm, g, "c06.attention", tokens, local) == RAD_OK &&
         (!text_cross || kv_in.declare(&b, m.nm, g, "c06.keys", tokens, kvdim) == RAD_OK),
         "C06 real activation/quantization declaration failed");
    need(x.q.op && ao.q.op && x.a.q8_fed && ao.a.q8_fed,
         "C06 failed to declare the exact native IQ8R activation quantizers");
    const rad_buf qkv = ltx2::decl_b(&b, m.nm.f("c06.qkv"), RAD_BF16, {tokens, (text_cross ? 1 : 3) * local});
    const rad_buf bkv = text_cross ? ltx2::decl_b(&b, m.nm.f("c06.kv"), RAD_BF16, {tokens, 2 * local}) : 0;
    const rad_buf logits = ltx2::decl_b(&b, m.nm.f("c06.gate"), RAD_BF16, {tokens, total_heads});
    const rad_buf out = ltx2::decl_b(&b, m.nm.f("c06.out"), RAD_BF16, {tokens, odim});
    const rad_buf scratch = ltx2::decl_b(&b, m.nm.f("c06.norm_scratch"), RAD_BF16,
                                         {tokens, full});
    need(x.a.x && x.a.q8 && x.a.s8 && ao.a.x && ao.a.q8 && ao.a.s8 &&
         qkv && logits && out && scratch && (!text_cross || (bkv && kv_in.q.op)),
         "C06 activation buffers missing");
    const ltx2::Rope rope = modal_cross ? ltx2::Rope{1, full, "20"} :
                            text_cross ? ltx2::Rope{} :
                            audio ? ltx2::Rope{1, full, "20"} :
                                   ltx2::Rope{3, full, "20 2048 2048"};
    const std::string original = "transformer.transformer_blocks." +
                                 std::to_string(block) + "." +
                                 (modal_cross ? (audio ? "video_to_audio_attn" :
                                                "audio_to_video_attn") : kind);
    ltx2::Attn a{};
    const auto before = b.program().weights.size();
    for(int bad=0; bad<2; ++bad) {
        ltx2::Geom wrong = g;
        if(bad==0) wrong.world=1;
        else wrong.wire_exact=0;
        need(ltx2::attn_declare(&b, m, wrong, a, prefix.c_str(), original.c_str(),
                               total_heads, hd, qdim, kvdim, odim, !text_cross, x, text_cross ? &kv_in : nullptr,
                               qkv, bkv, logits, ao, out, tokens, text_cross ? 2 : tokens, RAD_ACCESS_PER_TOKEN,
                               0, rope, 1e-6, &slice, scratch) == RAD_E_UNSUPPORTED,
             "C06 invalid native geometry admitted");
    }
    ltx2::Act unquantized = x;
    unquantized.q.op = 0;
    need(ltx2::attn_declare(&b, m, g, a, prefix.c_str(), original.c_str(),
                           total_heads, hd, qdim, kvdim, odim, !text_cross,
                           unquantized, text_cross ? &kv_in : nullptr, qkv, bkv, logits, ao, out,
                           tokens, text_cross ? 2 : tokens, RAD_ACCESS_PER_TOKEN, 0, rope,
                           1e-6, &slice, scratch) == RAD_E_UNSUPPORTED,
         "C06 accepted missing input IQ8R quantizer");
    unquantized = ao;
    unquantized.q.op = 0;
    need(ltx2::attn_declare(&b, m, g, a, prefix.c_str(), original.c_str(),
                           total_heads, hd, qdim, kvdim, odim, !text_cross,
                           x, text_cross ? &kv_in : nullptr, qkv, bkv, logits, unquantized, out,
                           tokens, text_cross ? 2 : tokens, RAD_ACCESS_PER_TOKEN, 0, rope,
                           1e-6, &slice, scratch) == RAD_E_UNSUPPORTED,
         "C06 accepted missing output IQ8R quantizer");
    if (text_cross) {
        ltx2::Act no_kv = kv_in;
        no_kv.q.op = 0;
        need(ltx2::attn_declare(&b, m, g, a, prefix.c_str(), original.c_str(),
                               total_heads, hd, qdim, kvdim, odim, false, x, &no_kv,
                               qkv, bkv, logits, ao, out, tokens, 2,
                               RAD_ACCESS_PER_TOKEN, 0, rope, 1e-6, &slice, scratch)
                 == RAD_E_UNSUPPORTED,
             "LTX03 accepted unquantized independent KV");
    }
    need(b.program().weights.size() == before && !a.tp3,
         "C06 rejected geometry allocated weights");
    const int rc=ltx2::attn_declare(
        &b, m, g, a, prefix.c_str(), original.c_str(),
        total_heads, hd, qdim, kvdim, odim, !text_cross, x, text_cross ? &kv_in : nullptr,
        qkv, bkv, logits, ao, out, tokens, text_cross ? 2 : tokens, RAD_ACCESS_PER_TOKEN,
        0, rope, 1e-6, &slice, scratch);
    if(rc != RAD_OK) {
        std::fprintf(stderr,"C06 failed: block=%lld audio=%d rank=%d owned=%d rc=%d\n",
                     (long long)block,(int)audio,rank,owned,rc);
        for(const auto& e:b.errors()) std::fprintf(stderr,"  %s\n",e.c_str());
    }
    need(rc == RAD_OK, "C06 real Attn::declare refused TP3");
    need(a.tp3 && a.head0 == first && a.H == owned &&
         a.inner == local && a.hd == hd &&
         a.tp_q.scratch == scratch && a.tp_k.scratch == scratch,
         "C06 lost local head or scratch ownership");
    need(a.gate.N == total_heads && a.gate.K == qdim && a.gate.bias,
         "C06 replicated head gate missing");
    need((text_cross || (a.op_rq && a.op_rk && a.op_v)) &&
         (!modal_cross || (a.op_rq && a.op_rk)) &&
         a.op_attn && a.op_hg && (text_cross ? a.kv.op : !a.kv.op) &&
         a.qkv.op && a.out.op && a.out.post_reduce && a.out.op_b &&
         a.tp_q.reduce && a.tp_k.reduce && !a.op_qn && !a.op_kn,
         "C06 native self-attention graph incomplete");
    for(const rad_op h:{a.op_rq,a.op_rk}) {
        if (text_cross && !modal_cross) {
            need(!h, "text cross-attention unexpectedly rotates Q/K"); continue;
        }
        const auto* op=b.program().op(h);
        long long head0=-1, heads=0, dim=0;
        need(op && op->base.get_i("head0",&head0) && head0==first &&
             op->base.get_i("n_head",&heads) && heads==owned &&
             op->base.get_i("dim",&dim) && dim==full,
             "C06 native RoPE lost global first-head origin");
    }
    for(const rad_op h:{a.tp_q.reduce,a.tp_k.reduce,a.out.post_reduce}) {
        const auto* op=b.program().op(h);
        long long world=0,exact=0;
        need(op && op->base.get_i("world_size",&world) && world==g.world &&
             op->base.get_i("exact",&exact) && exact==1,
             "C06 rank requires BigCherry's existing exact TP3 collective");
    }
    check_parts(*b.program().weight(a.qkv.l.w), slice);
    if (text_cross) check_parts(*b.program().weight(a.kv.l.w), kv_slice);
    check_parts(*b.program().weight(a.qkv.l.ws), slice);
    if (text_cross) check_parts(*b.program().weight(a.kv.l.ws), kv_slice);
    check_parts(*b.program().weight(a.qkv.bias), slice);
    if (text_cross) check_parts(*b.program().weight(a.kv.bias), kv_slice);
    const auto col=ltx2::C01Slice::columns_deferred(full,lo,local);
    check_col(*b.program().weight(a.out.l.w),col);
    check_col(*b.program().weight(a.out.l.ws),col);
    for(auto wt : {a.w_qn, a.w_kn, a.out.bias, a.gate.w, a.gate.bias})
        need(b.program().weight(wt) &&
             b.program().weight(wt)->decl.shard == RAD_SHARD_NONE,
             "C06 full checkpoint gain, gate or bias missing");
    for(const rad_op handle : {a.qkv.op, a.qkv.op_b, a.gate.op,
                              a.out.op, a.out.post_reduce, a.out.op_b, x.q.op, ao.q.op,
                              a.tp_q.clear, a.tp_q.insert, a.tp_q.reduce,
                              a.tp_q.norm, a.tp_q.extract,
                              a.tp_k.clear, a.tp_k.insert, a.tp_k.reduce,
                              a.tp_k.norm, a.tp_k.extract,
                              a.op_rq, a.op_rk, a.op_attn, a.op_hg, a.op_v, a.kv.op, a.kv.op_b}) {
        const auto* info = b.program().op(handle);
        if (!handle) continue;
        need(info != nullptr, "C06 native operation missing");
        for(const auto& band : info->bands)
            need(bool(band.dom[aql ? RAD_DOMAIN_DEVICE : RAD_DOMAIN_HOST]),
                 aql ? "C06 native AQL operation has no device kernel" :
                       "C06 op unresolved on host reference kernel");
    }
    need(b.errors().empty(), "C06 Builder diagnostics");
    if (instance) {
        if (capture) {
            capture->attn=a;
            capture->x=x;
            capture->ao=ao;
            capture->qkv=qkv;
            capture->gate=logits;
            capture->output=out;
            capture->width=full;
            capture->audio=audio;
            capture->rank=rank;
            capture->tokens=tokens<=4?1:tokens;
            capture->cu[1]=(int32_t)capture->tokens;
            RadVideoGeom geom{};
            geom.n_text=1;
            geom.lat_frames=1;
            geom.lat_h=1;
            geom.lat_w=audio?1:capture->tokens;
            geom.n_audio=audio?capture->tokens:1;
            capture->video_rows=geom.lat_w;
            capture->pos.resize((size_t)(3*geom.lat_w+geom.n_audio));
            need(ltx2::positions(&geom,capture->pos.data(),
                                 geom.n_text+geom.lat_w+geom.n_audio)==
                     geom.n_text+geom.lat_w+geom.n_audio,
                 "C06 native multirow rotary coordinates invalid");
        }
        // Retain exactly the Builder just verified; native all-reduce
        // instance initialization must rendezvous across all three ranks.
        *instance = std::move(owned_builder);
        return;
    }
    b.set_instances(false);
    need(b.finish() == RAD_OK && b.errors().empty(),
         "C06 native full attention graph finish failed");
    if (inspect_weights) verify_c06_load_contract(model,b.program(),rank);
}



// ---------------------------------------------------------------------------
// LTX03 issued FFN lane. One checkpoint FFN (video or audio) is built by the
// production Lin::declare/Act/unary and *issued* on the host reference kernels:
// TP3 (row-sharded up, column-sharded bias-free down partial, exact three-rank
// reduction, the audio bias applied once after it) and a TP1 reference of the
// very same checkpoint tensors and kernels (full width, no reduction).
static void ltx03_ffn_issued_step(RadCtx* c, const RadBatch*) {
    if (!c06_active) { rad_step_fail(c,"LTX03 FFN rank execution owner missing"); return; }
    C06Issued& r=*c06_active;
    uint16_t* input=(uint16_t*)r.ctx->buf_ptr(r.x.a.x);
    if (!input) { rad_step_fail(c,"LTX03 FFN source activation missing"); return; }
    if (!r.gpu)   // on a device the sample was uploaded before the step
        for (int64_t j=0;j<r.width*r.tokens;++j) input[j]=c06_sample(j);
    const int64_t D=r.width, n=r.tokens;
    const RadOperand cu = ltx2::praw(r.gpu ? (const int32_t*)r.device_cu : r.cu.data(),RAD_I32,2);
    const int64_t tab_cols = r.ffn.block ? 6*D : 3*D;   // block: attention triple, then FFN triple
    const int64_t ffn_tab0 = r.ffn.block ? 3*D : 0;
    if (r.ffn.block) {
        uint16_t* hidden=(uint16_t*)r.ctx->buf_ptr(r.ffn.res);
        uint16_t* table=(uint16_t*)r.ctx->buf_ptr(r.ffn.tabs);
        if (!hidden || !table) { rad_step_fail(c,"LTX03 block buffers missing"); return; }
        if (!r.gpu) {
            for (int64_t j=0;j<D*n;++j) hidden[j]=c06_sample(j+17);
            for (int64_t j=0;j<tab_cols;++j) table[j]=c06_sample(j+101);
        }
        RAD_ISSUE_N(c,r.ffn.mod1,n,RAD_NONE,RAD_NONE,
                    ltx2::tab(r.ffn.tabs,0,1,tab_cols,0,2*D),RAD_NONE,cu,
                    ltx2::brows(r.ffn.res,n),ltx2::brows(r.x.a.x,n));
        r.x.step(c,n);
        r.attn.qkv.step(c,r.x,r.qkv,n);
        r.attn.gate.step(c,r.x,r.gate,n);
        const float* positions=(r.gpu ? (const float*)r.device_pos : r.pos.data())+
                                (r.audio?3*r.video_rows:0);
        const RadOperand pos=ltx2::praw2(positions,RAD_F32,r.audio?1:3,n);
        ltx2::attn_core(c,r.attn,true,r.qkv,0,r.gate,r.ao,n,n,cu,RAD_NONE,pos,pos);
        r.ao.step(c,n);
        r.attn.out.step(c,r.ao,r.ffn.attn_out,n);
        RAD_ISSUE_N(c,r.ffn.gate1,n,ltx2::brows(r.ffn.attn_out,n),
                    ltx2::tab(r.ffn.tabs,0,1,tab_cols,2*D,D),cu,ltx2::brows(r.ffn.res,n));
    }
    if (r.ffn.chain) {
        uint16_t* hidden=(uint16_t*)r.ctx->buf_ptr(r.ffn.res);
        uint16_t* table=(uint16_t*)r.ctx->buf_ptr(r.ffn.tabs);
        if (!hidden || !table) { rad_step_fail(c,"LTX03 FFN chain buffers missing"); return; }
        if (!r.gpu && !r.ffn.block) {
            for (int64_t j=0;j<D*n;++j) hidden[j]=c06_sample(j+17);
            for (int64_t j=0;j<tab_cols;++j) table[j]=c06_sample(j+101);
        }
        // norm(hidden) (1 + scale) + shift into the FFN input plane
        RAD_ISSUE_N(c,r.ffn.mod,n,RAD_NONE,RAD_NONE,
                    ltx2::tab(r.ffn.tabs,0,1,tab_cols,ffn_tab0,2*D),RAD_NONE,cu,
                    ltx2::brows(r.ffn.res,n),ltx2::brows(r.x.a.x,n));
    }
    r.x.step(c,r.tokens);
    if (r.ffn.pipe && r.ffn.reduce) {
        // Two pieces of rows (a multiple of 64, the first the larger half): the second piece's
        // projections run on lane 0 while the first piece's exact reduction runs on lane 1,
        // as the Qwen drivers' step_pipelined does for a prefill step.
        const int64_t R = std::min<int64_t>(r.tokens, (r.tokens + 1) / 2 + 63) / 64 * 64;
        for (int piece = 0; piece < 2; ++piece) {
            const int64_t r0 = piece ? R : 0, m = piece ? r.tokens - R : R;
            if (m <= 0) continue;
            r.ffn.up.step(c,r.x,r.ffn.upbuf,m,r0,r0);
            RAD_ISSUE_N(c,r.ffn.act,m,rad::arch::brow_slice(r.ffn.upbuf,r0,m,r.ffn.up.N),
                        rad::arch::brow_slice(r.ffn.ff.a.x,r0,m,r.ffn.ff.n));
            r.ffn.ff.step(c,m,r0);
            r.ffn.down.l.step(c,r.ffn.ff.a,r.output,r0+m,r0,m,r0);
            rad_lane(c,1);
            rad_lane_join(c,0,1);
            r.ffn.down.reduce_rows(c,r.output,r0,m);
            rad_lane(c,0);
        }
        rad_lane_join(c,1,0);
    } else {
    r.ffn.up.step(c,r.x,r.ffn.upbuf,r.tokens);
    RAD_ISSUE_N(c,r.ffn.act,r.tokens,ltx2::brows(r.ffn.upbuf,r.tokens),
                ltx2::brows(r.ffn.ff.a.x,r.tokens));
    r.ffn.ff.step(c,r.tokens);
    if (r.ffn.reduce)
        r.ffn.down.step(c,r.ffn.ff,r.output,r.tokens);
    else   // omitted exact reduction: the bias-free partial GEMM only
        r.ffn.down.l.step(c,r.ffn.ff.a,r.output,r.tokens,0,r.tokens,0);
    }
    if (r.ffn.chain)   // hidden += gate * delta
        RAD_ISSUE_N(c,r.ffn.gate,n,ltx2::brows(r.output,n),
                    ltx2::tab(r.ffn.tabs,0,1,tab_cols,ffn_tab0+2*D,D),cu,ltx2::brows(r.ffn.res,n));
}

// world>1: this rank's `owned` of `heads` FFN units, deferred bias-free down;
// world==1: the whole FFN at full width (the TP1 reference).
static void build_ffn_lane(const RadFile& model, Registry& reg, int64_t block, bool audio,
                           int rank, int first, int owned, int heads, int world,
                           int64_t tokens, std::unique_ptr<Builder>* out, C06Issued& r,
                           bool aql = false, bool chain = false, bool with_attn = false) {
    const std::string prefix = "dit.blk." + std::to_string(block) +
                               (audio ? ".audio_ff" : ".ff");
    const auto* wu = model.find(prefix + "_up.weight");
    const auto* wd = model.find(prefix + "_down.weight");
    need(wu && wd && wu->rank == 2 && wd->rank == 2 &&
         wu->shape[0] == wd->shape[1] && wu->shape[1] == wd->shape[0],
         "LTX03 issued FFN source dimensions disagree");
    if (aql) need(rad_dev_set(rank) == RAD_OK, "LTX03 cannot select rank's AQL device");
    r.gpu = aql;
    const int64_t dim = wu->shape[1], width = wu->shape[0];
    const bool tp = world > 1;
    need(width % heads == 0 && (!tp || (first >= 0 && owned > 0 && first + owned <= heads)),
         "LTX03 issued FFN ownership not derivable from checkpoint");
    const int64_t unit = width / heads;
    const int64_t offset = tp ? first * unit : 0, local = tp ? owned * unit : width;
    const int64_t group = rad::arch::linear_w8h_group(model.encoding(*wu));
    const auto us = ltx2::C01Slice::rows(width, offset, local, unit, 1);
    const auto ds = ltx2::C01Slice::columns_deferred(width, offset, local);
    need(!tp || (us.valid(local, dim, group) && ds.valid(dim, local, group)),
         "LTX03 issued FFN ownership violates model K-scale group");
    auto owned_builder = std::make_unique<Builder>(reg, metadata(), context(rank, world, tokens));
    Builder& b = *owned_builder;
    b.set_sources(rad::rad_container_sources(model));
    ltx2::Model mm{};
    ltx2::Names& nm = mm.nm;
    ltx2::Geom g{};
    g.w_i8h = true; g.w_i8_group = group; g.world = world; g.wire_exact = 1;
    g.max_tok = tokens; g.dtype = "bf16";
    need(r.x.declare(&b,nm,g,"ltx03.ffn.input",tokens,dim) == RAD_OK &&
         r.ffn.ff.declare(&b,nm,g,"ltx03.ffn.mid",tokens,local) == RAD_OK &&
         r.x.q.op && r.ffn.ff.q.op, "LTX03 issued FFN Act declaration failed");
    r.ffn.upbuf = ltx2::decl_b(&b,nm.f("ltx03.ff.up"),RAD_BF16,{tokens,local});
    r.output = ltx2::decl_b(&b,nm.f("ltx03.delta"),RAD_BF16,{tokens,dim});
    need(r.ffn.upbuf && r.output, "LTX03 issued FFN buffers missing");
    r.ffn.chain = chain;
    r.ffn.result = r.output;
    if (chain) {
        r.ffn.res = ltx2::decl_b(&b,nm.f("ltx03.hidden"),RAD_BF16,{tokens,dim});
        r.ffn.tabs = ltx2::decl_b(&b,nm.f("ltx03.mod_table"),RAD_BF16,{1,(with_attn?6:3)*dim});
        need(r.ffn.res && r.ffn.tabs, "LTX03 chain buffers missing");
        if (with_attn) {
            // Self-attention first, in issue order: modulated norm, attention with its
            // exact reductions, gated residual. Then the FFN chain below.
            r.ffn.block = true;
            r.ffn.mod1 = ltx2::mod_op(&b,tokens,dim,1e-6,false,"rms",r.ffn.tabs,r.ffn.tabs,
                                      r.ffn.res,0,r.x.a.x);
            need(r.ffn.mod1, "LTX03 native block modulation unavailable");
            const std::string kind = audio ? "audio_attn1" : "attn1";
            const std::string ap = "dit.blk." + std::to_string(block) + "." + kind;
            const auto* fused = model.find(ap + ".qkv.weight");
            const auto* aout = model.find(ap + ".out.weight");
            need(fused && aout && fused->rank == 2 && fused->shape[1] == dim &&
                 aout->shape[0] == dim && fused->shape[0] % 3 == 0,
                 "LTX03 block attention source geometry mismatch");
            const int64_t full = fused->shape[0] / 3;
            need(full % heads == 0 && aout->shape[1] == full, "LTX03 block attention heads");
            const int64_t hd = full / heads;
            const int64_t alocal = tp ? owned * hd : full;
            const ltx2::C01Slice aslice = ltx2::C01Slice::rows(full, tp ? first * hd : 0, alocal, hd, 3);
            need(r.ao.declare(&b,nm,g,"ltx03.attn.out_act",tokens,alocal) == RAD_OK && r.ao.q.op,
                 "LTX03 block attention Act declaration failed");
            r.qkv = ltx2::decl_b(&b,nm.f("ltx03.qkv"),RAD_BF16,{tokens,3*alocal});
            r.gate = ltx2::decl_b(&b,nm.f("ltx03.gate"),RAD_BF16,{tokens,heads});
            r.ffn.attn_out = ltx2::decl_b(&b,nm.f("ltx03.attn_out"),RAD_BF16,{tokens,dim});
            const rad_buf scratch = ltx2::decl_b(&b,nm.f("ltx03.norm_scratch"),RAD_BF16,{tokens,full});
            need(r.qkv && r.gate && r.ffn.attn_out && scratch, "LTX03 block attention buffers");
            const ltx2::Rope rope = audio ? ltx2::Rope{1, full, "20"} :
                                            ltx2::Rope{3, full, "20 2048 2048"};
            const std::string original = "transformer.transformer_blocks." +
                                         std::to_string(block) + "." + kind;
            const int arc = ltx2::attn_declare(&b, mm, g, r.attn, ap.c_str(), original.c_str(),
                heads, hd, dim, full, dim, true, r.x, nullptr, r.qkv, 0, r.gate, r.ao,
                r.ffn.attn_out, tokens, tokens, RAD_ACCESS_PER_TOKEN, 0, rope, 1e-6,
                tp ? &aslice : nullptr, tp ? scratch : 0);
            need(arc == RAD_OK, "LTX03 block Attn::declare refused");
            need(!tp || (r.attn.tp3 && r.attn.out.post_reduce), "LTX03 block lost the TP3 reduction");
            r.ffn.gate1 = ltx2::gate_op(&b,tokens,dim,r.ffn.tabs,r.ffn.attn_out,r.ffn.res);
            need(r.ffn.gate1, "LTX03 block attention residual unavailable");
            r.ffn.has_attn = true;
            // positions as the C06 lane builds them
            r.cu[1] = (int32_t)tokens;
            RadVideoGeom geom{};
            geom.n_text=1; geom.lat_frames=1; geom.lat_h=1;
            geom.lat_w=audio?1:tokens;
            geom.n_audio=audio?tokens:1;
            r.video_rows=geom.lat_w;
            r.pos.resize((size_t)(3*geom.lat_w+geom.n_audio));
            need(ltx2::positions(&geom,r.pos.data(),geom.n_text+geom.lat_w+geom.n_audio)==
                     geom.n_text+geom.lat_w+geom.n_audio,
                 "LTX03 block rotary coordinates invalid");
        }
        r.ffn.mod = ltx2::mod_op(&b,tokens,dim,1e-6,false,"rms",r.ffn.tabs,r.ffn.tabs,
                                 r.ffn.res,0,r.x.a.x);
        need(r.ffn.mod, "LTX03 native modulation unavailable");
        r.ffn.result = r.ffn.res;
        r.cu[1] = (int32_t)tokens;
    }
    const ltx2::C01Slice* up_slice = tp ? &us : nullptr;
    const ltx2::C01Slice* down_slice = tp ? &ds : nullptr;
    int rc = audio
        ? r.ffn.up.declare(&b,nm,g,(prefix + "_up").c_str(),
                           {"checkpoint.audio_ff.net.0.proj.weight"},
                           {"checkpoint.audio_ff.net.0.proj.bias"}, {},
                           local,dim,RAD_ACCESS_PER_TOKEN,0,r.x,r.ffn.upbuf,tokens,false,up_slice)
        : r.ffn.up.declare(&b,nm,g,(prefix + "_up").c_str(),
                           {"checkpoint.ff.net.0.proj.weight"}, {}, {},
                           local,dim,RAD_ACCESS_PER_TOKEN,0,r.x,r.ffn.upbuf,tokens,false,up_slice);
    need(rc == RAD_OK && r.ffn.up.op, "LTX03 issued FFN up declaration refused");
    r.ffn.act = ltx2::unary(&b,"gelu_tanh",tokens,local,"bf16",r.ffn.upbuf,r.ffn.ff.a.x);
    need(r.ffn.act, "LTX03 issued FFN activation missing");
    rc = audio
        ? r.ffn.down.declare(&b,nm,g,(prefix + "_down").c_str(),
                             {"checkpoint.audio_ff.net.2.weight"},
                             {"checkpoint.audio_ff.net.2.bias"}, {},
                             dim,local,RAD_ACCESS_PER_TOKEN,0,r.ffn.ff,r.output,tokens,false,down_slice)
        : r.ffn.down.declare(&b,nm,g,(prefix + "_down").c_str(),
                             {"checkpoint.ff.net.2.weight"}, {}, {},
                             dim,local,RAD_ACCESS_PER_TOKEN,0,r.ffn.ff,r.output,tokens,false,down_slice);
    need(rc == RAD_OK && r.ffn.down.op, "LTX03 issued FFN down declaration refused");
    r.ffn.pipe = tp && aql && std::getenv("LTX03_PIPE") != nullptr;
    if (tp) {
        // Piped: the reduction is declared at overlap 1 inside the second lane's window, over
        // an output buffer the lane touches (the declared lives no longer bound its bytes).
        if (r.ffn.pipe)
            need(rad_buf_concurrent(&b,r.output) == RAD_OK && rad_lane1_window_begin(&b) == RAD_OK,
                 "LTX03 second-lane window refused");
        need(r.ffn.down.finalize_deferred_output(&b,nm,g,(prefix + "_down").c_str(),
                                                 RAD_ACCESS_PER_TOKEN,0,r.output,tokens,
                                                 r.ffn.pipe ? 1 : 0) == RAD_OK,
             "LTX03 issued FFN exact reduction/final bias refused");
        if (r.ffn.pipe)
            need(rad_lane1_window_end(&b) == RAD_OK, "LTX03 second-lane window close refused");
    }
    if (chain) {
        // Declared after the down projection and its reduction, in issue order: the arena
        // planner derives each buffer's lifetime from the declared operation order.
        r.ffn.gate = ltx2::gate_op(&b,tokens,dim,r.ffn.tabs,r.output,r.ffn.res);
        need(r.ffn.gate, "LTX03 native residual unavailable");
    }
    need(b.errors().empty(), "LTX03 issued FFN Builder diagnostics");
    r.width = dim; r.tokens = tokens; r.rank = rank; r.audio = audio;
    *out = std::move(owned_builder);
}

// Issue one FFN on `world` ranks (1: reference, 3: tensor parallel) on the host
// kernels; returns the per-rank BF16 [tokens, dim] output.
static std::vector<std::vector<uint16_t>> issue_ffn(
        const RadFile& model, Registry& reg, int64_t block, bool audio,
        const std::array<int,3>& split, int heads, int world, int64_t tokens,
        bool reduce, std::array<int,3>* ar_calls = nullptr, bool aql = false,
        bool chain = false, bool with_attn = false, int reps = 0, double* step_ms = nullptr) {
    std::array<std::unique_ptr<Builder>,3> builders;
    std::array<C06Issued,3> lanes;
    struct Retire {
        Registry& registry; std::array<C06Issued,3>& ranks;
        ~Retire() { for (auto& r:ranks) r.ctx.reset(); registry.fini_instances(); }
    } retire{reg,lanes};
    int first = 0;
    for (int rank=0; rank<world; ++rank) {
        lanes[rank].ffn.reduce = reduce;
        build_ffn_lane(model,reg,block,audio,rank,first,world>1?split[rank]:heads,
                       heads,world,tokens,&builders[rank],lanes[rank],aql,chain,with_attn);
        first += world>1 ? split[rank] : 0;
    }
    std::array<int,3> results{RAD_E_STATE,RAD_E_STATE,RAD_E_STATE};
    std::array<std::string,3> failures;
    std::vector<std::thread> workers;
    for (int rank=0; rank<world; ++rank)
        workers.emplace_back([&,rank] {
            try {
                if (aql && rad_dev_set(rank) != RAD_OK)
                    throw std::runtime_error("LTX03 AQL rank device bind failed");
                results[rank] = builders[rank]->finish();
            }
            catch (const std::exception& e) { failures[rank] = e.what(); }
        });
    for (auto& w : workers) w.join();
    for (int rank=0; rank<world; ++rank) {
        for (const auto& m : builders[rank]->errors())
            std::fprintf(stderr,"LTX03 FFN rank%d: %s\n",rank,m.c_str());
        need(failures[rank].empty() && results[rank]==RAD_OK &&
             builders[rank]->errors().empty() &&
             builders[rank]->program().device_holes(!aql).empty(),
             "LTX03 issued FFN native rank instance refused");
    }
    std::vector<std::vector<uint16_t>> outputs((size_t)world);
    for (int rank=0; rank<world; ++rank) {
        C06Issued& r = lanes[rank];
        rad::Program& prog = builders[rank]->program();
        if (aql) need(rad_dev_set(rank)==RAD_OK,"LTX03 AQL rank allocation device missing");
        verify_c06_load_contract(model,prog,rank,&r,world);
        r.ctx = std::make_unique<rad::Ctx>();
        rad::CtxDesc d{};
        d.program=&prog; d.rank=rank; d.world_size=world; d.device=aql?rank:0;
        d.arch_step=ltx03_ffn_issued_step; d.record_passes=false; d.registry=&reg;
        need(r.ctx->init(d)==RAD_OK,"LTX03 issued FFN RadCtx init failed");
        if (aql) {
            for (size_t i=1;i<prog.weights.size();++i) {
                const auto& w=prog.weights[i];
                need(rad_memcpy_async(w.ptr,r.weights[i].data(),w.stored_bytes,
                                      r.ctx->stream())==RAD_OK,
                     "LTX03 native checkpoint weight upload failed");
            }
            std::vector<uint16_t> sample((size_t)(r.width*r.tokens));
            for (int64_t j=0;j<r.width*r.tokens;++j) sample[(size_t)j]=c06_sample(j);
            void* input=r.ctx->buf_ptr(r.x.a.x);
            need(input && rad_memcpy_async(input,sample.data(),sample.size()*2,
                                           r.ctx->stream())==RAD_OK &&
                 rad_stream_sync(r.ctx->stream())==RAD_OK,
                 "LTX03 native activation upload failed");
            r.device_cu = rad_dev_alloc(sizeof(r.cu),RAD_MEM_DEVICE);
            need(r.device_cu,"LTX03 native cu-seqlens allocation refused");
            r.owned_device.push_back(r.device_cu);
            need(rad_memcpy_async(r.device_cu,r.cu.data(),sizeof(r.cu),r.ctx->stream())==RAD_OK,
                 "LTX03 native cu-seqlens upload failed");
            if (with_attn) {
                r.device_pos=rad_dev_alloc(r.pos.size()*sizeof(float),RAD_MEM_DEVICE);
                need(r.device_pos,"LTX03 native position allocation refused");
                r.owned_device.push_back(r.device_pos);
                need(rad_memcpy_async(r.device_pos,r.pos.data(),r.pos.size()*sizeof(float),
                                      r.ctx->stream())==RAD_OK,"LTX03 native position upload failed");
            }
            if (chain) {
                std::vector<uint16_t> hid((size_t)(r.width*r.tokens)),
                    tab((size_t)((with_attn?6:3)*r.width));
                for (size_t j=0;j<hid.size();++j) hid[j]=c06_sample((int64_t)j+17);
                for (size_t j=0;j<tab.size();++j) tab[j]=c06_sample((int64_t)j+101);
                void* hp=r.ctx->buf_ptr(r.ffn.res); void* tp=r.ctx->buf_ptr(r.ffn.tabs);
                need(hp && tp && rad_memcpy_async(hp,hid.data(),hid.size()*2,r.ctx->stream())==RAD_OK &&
                     rad_memcpy_async(tp,tab.data(),tab.size()*2,r.ctx->stream())==RAD_OK &&
                     rad_stream_sync(r.ctx->stream())==RAD_OK,
                     "LTX03 native chain upload failed");
            }
        }
    }
    std::array<int,3> status{RAD_E_STATE,RAD_E_STATE,RAD_E_STATE};
    std::array<std::vector<double>,3> times;
    std::vector<std::thread> runners;
    for (int rank=0; rank<world; ++rank)
        runners.emplace_back([&,rank] {
            C06Issued& r = lanes[rank];
            c06_active = &r;
            const int rc = r.ctx->bind_thread();
            RadBatch batch{};
            batch.phase = RAD_PHASE_PREFILL; batch.n_tok = r.tokens; batch.n_seq = 1;
            status[rank] = rc==RAD_OK ? r.ctx->run_step(&batch) : rc;
            if (status[rank]==RAD_OK && aql) status[rank] = rad_stream_sync(r.ctx->stream());
            // Timing: further warm steps of the same graph; each iteration is bounded by the
            // slowest rank (the collectives make all ranks leave together).
            for (int i=0; i<reps && status[rank]==RAD_OK; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                status[rank] = r.ctx->run_step(&batch);
                if (status[rank]==RAD_OK && aql) status[rank] = rad_stream_sync(r.ctx->stream());
                times[rank].push_back(std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-t0).count());
            }
            c06_active = nullptr;
        });
    for (auto& t : runners) t.join();
    for (int rank=0; rank<world; ++rank) {
        C06Issued& r = lanes[rank];
        if (status[rank]!=RAD_OK)
            std::fprintf(stderr,"LTX03 FFN rank %d error %d: %s\n",rank,status[rank],r.ctx->step_error());
        need(status[rank]==RAD_OK,"LTX03 FFN issue refused");
        if (ar_calls) (*ar_calls)[rank] = (int)r.ctx->ar_calls();
        const void* ptr = r.ctx->buf_ptr(r.ffn.result);
        need(ptr!=nullptr,"LTX03 FFN output arena buffer missing");
        outputs[(size_t)rank].resize((size_t)(r.width*r.tokens));
        if (aql) {
            need(rad_dev_set(rank)==RAD_OK &&
                 rad_memcpy_async(outputs[(size_t)rank].data(),ptr,
                                  outputs[(size_t)rank].size()*2,r.ctx->stream())==RAD_OK &&
                 rad_stream_sync(r.ctx->stream())==RAD_OK,
                 "LTX03 native output download failed");
        } else
            std::memcpy(outputs[(size_t)rank].data(),ptr,outputs[(size_t)rank].size()*2);
    }
    if (step_ms && reps > 0) {
        std::vector<double> per_iter((size_t)reps);
        for (int i=0;i<reps;++i) {
            double slowest = 0;
            for (int rank=0; rank<world; ++rank) slowest = std::max(slowest,times[rank][(size_t)i]);
            per_iter[(size_t)i] = slowest;
        }
        std::sort(per_iter.begin(),per_iter.end());
        *step_ms = per_iter[per_iter.size()/2];
        std::printf("  per-rank median step:");
        for (int rank=0; rank<world; ++rank) {
            auto t = times[rank]; std::sort(t.begin(),t.end());
            std::printf(" r%d=%.3f", rank, t[t.size()/2]);
        }
        std::printf(" ms\n");
    }
    for (auto& r : lanes) r.ctx.reset();
    reg.fini_instances();
    return outputs;
}

static double bf16_to_f64(uint16_t v) {
    const uint32_t u=(uint32_t)v<<16; float x; std::memcpy(&x,&u,sizeof x); return x;
}

// LTX03: checkpoint-derived FFN and replicated residual/modulation declarations.
// The DiT production loop is unchanged; this does not issue a TP3 graph.
static void run_dit_ffn(const RadFile& model, Registry& reg, int64_t block,
                        bool audio, int rank, int first, int owned, int heads) {
    const std::string prefix = "dit.blk." + std::to_string(block) +
                               (audio ? ".audio_ff" : ".ff");
    const auto* wu = model.find(prefix + "_up.weight");
    const auto* wd = model.find(prefix + "_down.weight");
    need(wu && wd && wu->rank == 2 && wd->rank == 2 &&
         wu->shape[0] == wd->shape[1] && wu->shape[1] == wd->shape[0],
         "LTX03 FFN upstream/downstream source dimensions disagree");
    const int64_t dim = wu->shape[1], width = wu->shape[0];
    need(dim > 0 && width > 0 && width % heads == 0 &&
         first >= 0 && owned > 0 && first + owned <= heads,
         "LTX03 FFN rank ownership not derivable from checkpoint");
    const int64_t unit = width / heads, offset = first * unit, local = owned * unit;
    const auto& eu = model.encoding(*wu);
    const auto& ed = model.encoding(*wd);
    need(rad::arch::linear_is_iq8r(eu) && rad::arch::linear_is_iq8r(ed),
         "LTX03 FFN model encoding is not IQ8R");
    const int64_t group = rad::arch::linear_w8h_group(eu);
    need(group == rad::arch::linear_w8h_group(ed),
         "LTX03 FFN source group disagreement");
    const auto us = ltx2::C01Slice::rows(width, offset, local, unit, 1);
    const auto ds = ltx2::C01Slice::columns_deferred(width, offset, local);
    need(us.valid(local, dim, group) && ds.valid(dim, local, group),
         "LTX03 FFN ownership violates model K-scale group");

    const auto* ub = model.find(prefix + "_up.bias");
    const auto* db = model.find(prefix + "_down.bias");
    need((ub != nullptr) == audio && (db != nullptr) == audio &&
         (!audio || (ub->rank == 1 && ub->shape[0] == width &&
                     db->rank == 1 && db->shape[0] == dim)),
         "LTX03 FFN bias geometry is not model-consistent");
    Builder b(reg, metadata(), context(rank));
    b.set_sources(rad::rad_container_sources(model));
    ltx2::Names nm;
    ltx2::Geom g{};
    g.w_i8h = true;
    g.w_i8_group = group;
    g.world = 3;
    g.wire_exact = 1;
    g.max_tok = 4;
    g.dtype = "bf16";
    ltx2::Act x{}, ff{};
    need(x.declare(&b,nm,g,"ltx03.ffn.input",4,dim) == RAD_OK &&
         ff.declare(&b,nm,g,"ltx03.ffn.mid",4,local) == RAD_OK,
         "LTX03 FFN Act declaration failed");
    const rad_buf res = ltx2::decl_b(&b,nm.f("ltx03.residual"),RAD_BF16,{4,dim});
    const rad_buf delta = ltx2::decl_b(&b,nm.f("ltx03.delta"),RAD_BF16,{4,dim});
    const rad_buf tables = ltx2::decl_b(&b,nm.f("ltx03.mod"),RAD_BF16,{4,2*dim});
    const rad_buf gates = ltx2::decl_b(&b,nm.f("ltx03.gates"),RAD_BF16,{4,dim});
    const rad_buf upbuf = ltx2::decl_b(&b,nm.f("ltx03.ff.up"),RAD_BF16,{4,local});
    need(res && delta && tables && gates && upbuf,
         "LTX03 FFN replicated BF16 buffers missing");
    const rad_op mod = ltx2::mod_op(&b,4,dim,1e-6,false,"rms",
                                    tables,tables,res,0,x.a.x);
    const rad_op mod_gated = ltx2::mod_op(&b,4,dim,1e-6,true,"rms",
                                          tables,tables,res,delta,x.a.x);
    const rad_op residual = ltx2::gate_op(&b,4,dim,gates,delta,res);
    need(mod && mod_gated && residual, "LTX03 native modulation/residual unavailable");
    ltx2::Lin up{}, down{};
    int rc = audio
        ? up.declare(&b,nm,g,(prefix + "_up").c_str(),
                     {"checkpoint.audio_ff.net.0.proj.weight"},
                     {"checkpoint.audio_ff.net.0.proj.bias"}, {},
                     local,dim,RAD_ACCESS_PER_TOKEN,0,x,upbuf,4,false,&us)
        : up.declare(&b,nm,g,(prefix + "_up").c_str(),
                     {"checkpoint.ff.net.0.proj.weight"}, {}, {},
                     local,dim,RAD_ACCESS_PER_TOKEN,0,x,upbuf,4,false,&us);
    need(rc==RAD_OK && up.op, "LTX03 FFN up row shard refused");
    const rad_op act = ltx2::unary(&b,"gelu_tanh",4,local,"bf16",upbuf,ff.a.x);
    need(act, "LTX03 native FFN activation missing");
    // Reject fractional IQ8R scale-group ownership before allocating weights.
    const auto bad_slice = ltx2::C01Slice::columns_deferred(
        width, offset + group / 2, local);
    const auto before_bad = b.program().weights.size();
    ltx2::Lin invalid{};
    int bad_rc = audio
        ? invalid.declare(&b,nm,g,(prefix + "_down").c_str(),
                          {"checkpoint.audio_ff.net.2.weight"},
                          {"checkpoint.audio_ff.net.2.bias"}, {},
                          dim,local,RAD_ACCESS_PER_TOKEN,0,ff,delta,4,false,&bad_slice)
        : invalid.declare(&b,nm,g,(prefix + "_down").c_str(),
                          {"checkpoint.ff.net.2.weight"}, {}, {},
                          dim,local,RAD_ACCESS_PER_TOKEN,0,ff,delta,4,false,&bad_slice);
    need(bad_rc == RAD_E_UNSUPPORTED &&
         b.program().weights.size() == before_bad,
         "LTX03 FFN illegal scale-group cut reached native weight declaration");
    rc = audio
        ? down.declare(&b,nm,g,(prefix + "_down").c_str(),
                       {"checkpoint.audio_ff.net.2.weight"},
                       {"checkpoint.audio_ff.net.2.bias"}, {},
                       dim,local,RAD_ACCESS_PER_TOKEN,0,ff,delta,4,false,&ds)
        : down.declare(&b,nm,g,(prefix + "_down").c_str(),
                       {"checkpoint.ff.net.2.weight"}, {}, {},
                       dim,local,RAD_ACCESS_PER_TOKEN,0,ff,delta,4,false,&ds);
    need(rc==RAD_OK && down.op && !down.op_b && !down.bias,
         "LTX03 FFN down was not a bias-free partial GEMM");
    ltx2::Geom wrong = g;
    wrong.world = 1;
    need(down.finalize_deferred_output(&b,nm,wrong,(prefix + "_down").c_str(),
                                       RAD_ACCESS_PER_TOKEN,0,delta,4) == RAD_E_UNSUPPORTED,
         "LTX03 FFN down accepted an incompatible reduction world");
    need(down.finalize_deferred_output(&b,nm,g,(prefix + "_down").c_str(),
                                       RAD_ACCESS_PER_TOKEN,0,delta,4) == RAD_OK,
         "LTX03 FFN down exact reduction/final bias refused");
    need(down.post_reduce && down.deferred_bias &&
         (audio ? (down.bias && down.op_b) : (!down.bias && !down.op_b)),
         "LTX03 FFN biasless/biased output finalization mismatch");
    need(b.program().weight(up.l.w) && b.program().weight(up.l.ws) &&
         b.program().weight(down.l.w) && b.program().weight(down.l.ws),
         "LTX03 FFN source-backed native weight handles missing");
    check_parts(*b.program().weight(up.l.w),us);
    check_parts(*b.program().weight(up.l.ws),us);
    check_col(*b.program().weight(down.l.w),ds);
    check_col(*b.program().weight(down.l.ws),ds);
    if (audio) {
        check_parts(*b.program().weight(up.bias),us);
        need(b.program().weight(down.bias)->decl.shard == RAD_SHARD_NONE,
             "LTX03 FFN audio output bias must be replicated after reduce");
    }
    const rad_op ops[] = {mod,mod_gated,residual,up.op,act,ff.q.op,
                          down.op,down.post_reduce,down.op_b};
    for (rad_op h : ops) {
        if (!h) continue;
        const auto* op=b.program().op(h);
        need(op!=nullptr, "LTX03 FFN native graph lost operation");
        for (const auto& band : op->bands)
            need(bool(band.dom[RAD_DOMAIN_HOST]),
                 "LTX03 FFN operation not available on reference host");
    }
    long long world=0, exact=0;
    const auto* red=b.program().op(down.post_reduce);
    need(red->base.get_i("world_size",&world) && world == g.world &&
         red->base.get_i("exact",&exact) && exact == 1,
         "LTX03 FFN row-parallel reduce not exact TP3");
    need(down.finalize_deferred_output(&b,nm,g,(prefix + "_down").c_str(),
                                       RAD_ACCESS_PER_TOKEN,0,delta,4) == RAD_E_UNSUPPORTED,
         "LTX03 FFN double reduction/bias was accepted");
    b.set_instances(false);
    need(b.finish()==RAD_OK && b.errors().empty(),
         "LTX03 FFN/modulation native Builder finish failed");
}

int main(int argc, char** argv) {
    if(argc!=3 && argc!=4 && argc!=5) {
        std::fprintf(stderr,"usage: ltx_c02_lin_declare_test MODEL.rad PLUGIN_KERNEL_DIR\n");
        return 2;
    }
    try {
        const bool aql = argc >= 4 &&
            (std::string(argv[3]) == "--c06-instances-aql" ||
             std::string(argv[3]) == "--c06-issued-aql" ||
             std::string(argv[3]) == "--c06-issued-aql-16" ||
             std::string(argv[3]) == "--c06-issued-aql-64" ||
             std::string(argv[3]) == "--ltx03-ffn-issued-aql" ||
             std::string(argv[3]) == "--ltx03-block-time-aql" ||
             std::string(argv[3]) == "--ltx03-wire-error-aql");
        if (aql) ltx_test::require_aql_backend();
        RadFile model;
        need(model.open(argv[1])==RAD_OK,"cannot open LTX source model");
        const auto* row=model.find("dit.blk.0.attn1.qkv.weight");
        need(row && row->shape[0]/3==row->shape[1],
             "source fused video QKV is no longer square by bank");
        Registry reg;
        if (aql) ltx_test::configure_aql_registry(reg);
        const int registry_status=reg.load_dir(
            argv[2],aql ? std::vector<std::string>{"libr3","libr4d","libref"} :
                          std::vector<std::string>{"libref"});
        need(registry_status==RAD_OK,"2.2 reference kernel registry cannot load");
        const char* block_text = model.get_s("transformer.num_layers", "");
        const char* head_text = model.get_s("transformer.num_attention_heads", "");
        need(*block_text && *head_text, "model does not declare DiT geometry");
        const int64_t nblocks = std::stoll(block_text);
        const int nheads = std::stoi(head_text);
        need(nblocks > 0 && nheads > 0, "invalid DiT geometry");

        if (argc == 5 && aql && std::string(argv[3]) == "--ltx03-wire-error-aql") {
            // Reports (does not gate) the error of an issued block at world argv[4] against the
            // TP1 issue (the exact wire; the lossy WHT6 wire measured 5-7 times this error for a block
            // that is only level with one card, so the adapter stays exact-only).
            const int w = std::stoi(argv[4]);
            need(w == 2 || w == 3, "LTX03 wire error arm is for two or three cards");
            const int64_t tokens = std::getenv("LTX03_TOKENS") ? std::stoll(std::getenv("LTX03_TOKENS")) : 512;
            const std::array<int,3> split = w == 2 ? std::array<int,3>{16,16,0} : std::array<int,3>{12,12,8};
            for (int variant : {0,1,2})
                for (int64_t block : {int64_t(0), nblocks - 1})
                    for (bool audio : {false,true}) {
                        if (audio && variant == 2 && w == 2) continue;   // the audio attention at two cards is refused
                        const bool chain = variant >= 1, with_attn = variant == 2;
                        const auto ref = issue_ffn(model,reg,block,audio,{0,0,0},nheads,1,tokens,true,nullptr,true,chain,with_attn);
                        const auto out = issue_ffn(model,reg,block,audio,split,nheads,w,tokens,true,nullptr,true,chain,with_attn);
                        double ref_sq = 0, err_sq = 0, max_abs = 0;
                        for (size_t j=0;j<ref[0].size();++j) {
                            const double r = bf16_to_f64(ref[0][j]);
                            const double e = bf16_to_f64(out[0][j]) - r;
                            ref_sq += r*r; err_sq += e*e; max_abs = std::max(max_abs,std::abs(e));
                        }
                        bool replicas = true;
                        for (int rank=1; rank<w; ++rank) replicas = replicas && out[0]==out[(size_t)rank];
                        std::printf("LTX03 wire=%s world=%d %s block=%lld %s tokens=%lld: rel_rms_vs_tp1=%g max_abs=%g replicas_equal=%d\n",
                                    "exact",w,
                                    with_attn?"attention+FFN":chain?"mod+FFN+residual":"FFN",(long long)block,
                                    audio?"audio":"video",(long long)tokens,std::sqrt(err_sq/ref_sq),max_abs,(int)replicas);
                        std::fflush(stdout);
                    }
            return 0;
        }
        if (argc == 5 && aql && std::string(argv[3]) == "--ltx03-block-time-aql") {
            // Warm step time of one issued block on the cards: TP1 (one card) against the
            // three-card splits, at the requested number of tokens. argv[4] = tokens.
            const int64_t tokens = std::stoll(argv[4]);
            need(tokens > 0, "LTX03 timing needs a positive token count");
            const int reps = 20;
            // LTX03_TIME_QUICK: the video lanes at one split only, for A/Bs of a pinned axis
            // (RADIANCE_TUNE) where every arm must hold the GPU lease briefly.
            const bool quick = std::getenv("LTX03_TIME_QUICK") != nullptr;
            const bool two_cards_only = std::getenv("LTX03_TIME_TP2") != nullptr;   // skip the three-card arms
            for (int variant : {0,2})
                for (bool audio : {false,true}) {
                    if (quick && audio) continue;
                    double ms1 = 0;
                    (void)issue_ffn(model,reg,0,audio,{0,0,0},nheads,1,tokens,true,nullptr,true,
                                    variant>=1,variant==2,reps,&ms1);
                    std::printf("LTX03 time %s %s tokens=%lld TP1(one card): %.3f ms/step\n",
                                variant==2?"attention+FFN block":"FFN",audio?"audio":"video",
                                (long long)tokens,ms1);
                    const std::vector<std::array<int,3>> splits = two_cards_only ?
                        std::vector<std::array<int,3>>{} : quick ?
                        std::vector<std::array<int,3>>{{12,12,8}} :
                        std::vector<std::array<int,3>>{{8,8,16},{10,10,12},{12,12,8},{14,14,4}};
                    for (const std::array<int,3>& split : splits) {
                        double ms3 = 0;
                        (void)issue_ffn(model,reg,0,audio,split,nheads,3,tokens,true,nullptr,true,
                                        variant>=1,variant==2,reps,&ms3);
                        std::printf("LTX03 time %s %s tokens=%lld TP3 %d,%d,%d: %.3f ms/step (%.2fx vs TP1)\n",
                                    variant==2?"attention+FFN block":"FFN",audio?"audio":"video",
                                    (long long)tokens,split[0],split[1],split[2],ms3,ms1/ms3);
                        std::fflush(stdout);
                    }
                    if (quick && !two_cards_only) continue;
                    // the two gfx1100 cards alone: world 2, the R9700 not used
                    double ms2 = 0;
                    (void)issue_ffn(model,reg,0,audio,{16,16,0},nheads,2,tokens,true,nullptr,true,
                                    variant>=1,variant==2,reps,&ms2);
                    std::printf("LTX03 time %s %s tokens=%lld TP2 (two gfx1100) 16,16: %.3f ms/step (%.2fx vs TP1)\n",
                                variant==2?"attention+FFN block":"FFN",audio?"audio":"video",
                                (long long)tokens,ms2,ms1/ms2);
                    std::fflush(stdout);
                }
            return 0;
        }
        if (argc == 4 && (std::string(argv[3]) == "--ltx03-ffn-issued-host" || aql &&
                          std::string(argv[3]) == "--ltx03-ffn-issued-aql")) {
            // The issued TP3 FFN against a TP1 issue of the same checkpoint and kernels.
            // LTX03_TOKENS: a row count other than the host test's four (the pipelined arms).
            const int64_t tokens = std::getenv("LTX03_TOKENS") ? std::stoll(std::getenv("LTX03_TOKENS")) : 4;
            int checked = 0;
            for (int variant : {0,1,2})   // 0: FFN, 1: modulation+FFN+residual, 2: attention+FFN block
            for (int64_t block : {int64_t(0), nblocks - 1})
                for (bool audio : {false,true}) {
                    const bool chain = variant >= 1, with_attn = variant == 2;
                    const auto ref = issue_ffn(model,reg,block,audio,{0,0,0},nheads,1,tokens,true,nullptr,aql,chain,with_attn);
                    need(ref.size()==1 && !ref[0].empty(),"LTX03 TP1 FFN reference missing");
                    double ref_sq = 0;
                    for (uint16_t v : ref[0]) {
                        need(std::isfinite(bf16_to_f64(v)),"LTX03 TP1 FFN output not finite");
                        ref_sq += bf16_to_f64(v)*bf16_to_f64(v);
                    }
                    need(ref_sq > 0,"LTX03 TP1 FFN output all zero");
                    for (const std::array<int,3>& split :
                         {std::array<int,3>{8,8,16}, std::array<int,3>{12,12,8}}) {
                        std::array<int,3> calls{};
                        const auto out = issue_ffn(model,reg,block,audio,split,nheads,3,tokens,true,&calls,aql,chain,with_attn);
                        for (int rank=0; rank<3; ++rank)
                            need(calls[rank]==(with_attn ? 4 : 1) + (aql && std::getenv("LTX03_PIPE") && tokens>64 ? 1 : 0),
                                 "LTX03 wrong native reduction count: attention issues Q, K and output, the FFN one");
                        for (int rank=1; rank<3; ++rank) {
                            if (out[0]!=out[(size_t)rank]) {
                                size_t bad=0, first_bad=out[0].size();
                                for (size_t j=0;j<out[0].size();++j)
                                    if (out[0][j]!=out[(size_t)rank][j]) { ++bad; first_bad=std::min(first_bad,j); }
                                std::fprintf(stderr,"LTX03 rank %d differs from rank 0 in %zu/%zu values, first at %zu\n",
                                             rank,bad,out[0].size(),first_bad);
                            }
                            need(out[0]==out[(size_t)rank],
                                 "LTX03 exact three-rank FFN output replicas disagree");
                        }
                        need(out[0].size()==ref[0].size(),"LTX03 TP3/TP1 FFN output width differs");
                        double err_sq = 0, max_abs = 0;
                        for (size_t j=0;j<ref[0].size();++j) {
                            const double e = bf16_to_f64(out[0][j]) - bf16_to_f64(ref[0][j]);
                            err_sq += e*e; max_abs = std::max(max_abs,std::abs(e));
                        }
                        const double rel = std::sqrt(err_sq/ref_sq);
                        std::printf("LTX03 %s issued %s block=%lld %s split=%d,%d,%d: rel_rms_vs_tp1=%g max_abs=%g\n",
                                    with_attn?"attention+FFN block":chain?"mod+FFN+residual":"FFN",aql?"AQL":"host",(long long)block,audio?"audio":"video",split[0],split[1],split[2],rel,max_abs);
                        std::fflush(stdout);
                        // BF16 has 8 significant bits: its unit roundoff 2^-8 bounds the relative
                        // error one reduction of partial sums can cause, so a chain of `calls`
                        // reductions differs from the unsplit issue by at most their sum (the
                        // measured values above sit well inside it; a wrong split or a doubled
                        // or missing reduction is an O(1) error).
                        need(rel <= std::ldexp(1.0,-8) * (double)calls[0],
                             "LTX03 TP3 FFN disagrees with the TP1 issue beyond BF16 rounding");
                        ++checked;
                    }
                }
            // The test must notice an omitted reduction: without it the ranks hold
            // different partial sums.
            std::array<int,3> calls{};
            const auto bad = issue_ffn(model,reg,0,false,{8,8,16},nheads,3,tokens,false,&calls,aql);
            need(calls[0]==0 && calls[1]==0 && calls[2]==0,
                 "LTX03 negative lane still issued a reduction");
            need(bad[0]!=bad[1] || bad[1]!=bad[2],
                 "LTX03 omitted-reduction replicas unexpectedly agree: the test cannot detect it");
            std::printf("LTX03 issued %s FFN: %d TP3/TP1 comparisons over blocks 0 and %lld; omitted reduction detected\n",
                        aql?"AQL":"host",checked,(long long)nblocks-1);
            return 0;
        }
        if (argc == 4 && std::string(argv[3]) == "--ltx03-ffn-modulation") {
            int64_t tested = 0;
            for (int64_t block = 0; block < nblocks; ++block)
                for (bool audio : {false,true})
                    for (const std::array<int,3>& split :
                         {std::array<int,3>{8,8,16}, std::array<int,3>{12,12,8}}) {
                        int first = 0;
                        for (int rank=0;rank<3;++rank) {
                            run_dit_ffn(model,reg,block,audio,rank,first,
                                        split[rank],nheads);
                            first += split[rank];
                            ++tested;
                        }
                        need(first == nheads, "LTX03 FFN head map lost owners");
                    }
            need(tested == nblocks * 2 * 2 * 3,
                 "LTX03 FFN source-backed declaration count mismatch");
            std::printf("LTX03 %lld FFN/modulation/residual declarations across %lld blocks\n",
                        (long long)tested,(long long)nblocks);
            return 0;
        }
        if (argc == 4) {
            if (std::string(argv[3]) == "--ltx03-text-cross" ||
                std::string(argv[3]) == "--ltx03-modal-cross") {
                const bool modal = std::string(argv[3]) == "--ltx03-modal-cross";
                int64_t count = 0;
                for (int64_t block = 0; block < nblocks; ++block)
                    for (bool audio : {false, true})
                        for (const std::array<int,3> split :
                             {std::array<int,3>{8,8,16}, std::array<int,3>{12,12,8}}) {
                            int first = 0;
                            for (int rank = 0; rank < 3; ++rank) {
                                run_self_attn(model, reg, block, audio, rank,
                                              first, split[rank], nheads,
                                              nullptr, false, false, nullptr, 4, true, modal);
                                first += split[rank];
                                ++count;
                            }
                            need(first == nheads, "LTX03 text-cross head ownership gap");
                        }
                need(count == nblocks * 12, "LTX03 text-cross coverage count");
                return 0;
            }
            if (std::string(argv[3]) == "--ltx03-all-blocks") {
                // LTX03: source-derived full-depth geometry, with the same
                // native three-rank declaration and no second model path.
                int64_t declared = 0;
                for (int64_t block = 0; block < nblocks; ++block)
                    for (bool audio : {false, true})
                        for (const std::array<int,3>& split :
                             {std::array<int,3>{8,8,16}, std::array<int,3>{12,12,8}}) {
                            int first = 0;
                            for (int rank = 0; rank < 3; ++rank) {
                                run_self_attn(model, reg, block, audio, rank, first,
                                              split[rank], nheads);
                                first += split[rank];
                                ++declared;
                            }
                            need(first == nheads, "LTX03 all-block head coverage gap");
                        }
                need(declared == nblocks * 2 * 2 * 3,
                     "LTX03 all-block native declaration count drift");
                std::printf("LTX03 %lld native self-attention declarations across %lld blocks\n",
                            (long long)declared, (long long)nblocks);
                return 0;
            }
            if (std::string(argv[3]) == "--c06-weights") {
                // Same builder and exact native loader geometry as C06;
                // no new model reader, memory mover or implementation.
                int tested=0;
                for (bool audio : {false,true})
                    for (const std::array<int,3>& split :
                         {std::array<int,3>{8,8,16},std::array<int,3>{12,12,8}}) {
                        int first=0;
                        for (int rank=0;rank<3;++rank) {
                            run_self_attn(model,reg,0,audio,rank,first,
                                          split[rank],nheads,nullptr,true);
                            first+=split[rank];
                            ++tested;
                        }
                        need(first==nheads,"C06 weight inventory ownership gap");
                    }
                need(tested==12,"C06 weight contract matrix drift");
                return 0;
            }
            const int64_t issue_rows =
                (std::string(argv[3])=="--c06-issued-host-16" ||
                 std::string(argv[3])=="--c06-issued-aql-16") ? 16 :
                (std::string(argv[3])=="--c06-issued-host-64" ||
                 std::string(argv[3])=="--c06-issued-aql-64") ? 64 : 1;
            const bool issue = std::string(argv[3])=="--c06-issued-host" ||
                               std::string(argv[3])=="--c06-issued-aql" ||
                               std::string(argv[3])=="--c06-issued-host-16" ||
                               std::string(argv[3])=="--c06-issued-host-64" ||
                               std::string(argv[3])=="--c06-issued-aql-16" ||
                               std::string(argv[3])=="--c06-issued-aql-64";
            if (std::string(argv[3]) == "--c06-instances" || aql || issue) {
                // Both issued modes reuse the same declared native C06 graph
                // and exact collective instances as the existing proofs.
                std::array<std::vector<uint16_t>,2> split_reference;
                int groups = 0;
                size_t initialized = 0;
                for (bool audio : {false, true})
                    for (const std::array<int,3>& split :
                         {std::array<int,3>{8,8,16}, std::array<int,3>{12,12,8}}) {
                        std::array<std::unique_ptr<Builder>,3> builders;
                        std::array<C06Issued,3> issued;
                        // Keep every Program's frozen geometry alive until
                        // native kernel fini, including when an assertion
                        // throws during rank validation.
                        struct Retire {
                            Registry& registry;
                            std::array<C06Issued,3>& ranks;
                            ~Retire() {
                                // Contexts own streams and can hold native instance
                                // pointers until shutdown; retire them first even on
                                // an exception or failed rank issue.
                                for (auto& rank:ranks) rank.ctx.reset();
                                registry.fini_instances();
                            }
                        } retire{reg,issued};
                        int first = 0;
                        for (int rank=0; rank<3; ++rank) {
                            run_self_attn(model,reg,0,audio,rank,
                                          first,split[rank],nheads,&builders[rank],
                                          false,aql,issue?&issued[rank]:nullptr,
                                          issue?std::max<int64_t>(4,issue_rows):4);
                            if(issue) issued[rank].gpu=aql;
                            first += split[rank];
                        }
                        need(first == nheads, "C06 instance head ownership gap");
                        const size_t instances_before = reg.n_instances();
                        std::array<int,3> results{RAD_E_STATE,RAD_E_STATE,RAD_E_STATE};
                        std::array<std::string,3> failures;
                        std::array<std::thread,3> workers;
                        for (int rank=0; rank<3; ++rank)
                            workers[rank] = std::thread([&,rank] {
                                try {
                                    if (aql && rad_dev_set(rank) != RAD_OK)
                                        throw std::runtime_error("C06 AQL rank device bind failed");
                                    results[rank] = builders[rank]->finish();
                                } catch (const std::exception& e) {
                                    failures[rank] = e.what();
                                }
                            });
                        for (auto& worker : workers) worker.join();
                        for (int rank=0; rank<3; ++rank) {
                            if (!failures[rank].empty())
                                std::fprintf(stderr, "C06 rank%d: %s\n",
                                             rank,failures[rank].c_str());
                            if (results[rank] != RAD_OK)
                                for (const auto& message : builders[rank]->errors())
                                    std::fprintf(stderr, "C06 rank%d: %s\n",rank,message.c_str());
                            need(results[rank] == RAD_OK &&
                                 builders[rank]->errors().empty(),
                                 "C06 native rank instance rendezvous refused");
                            need(builders[rank]->program().device_holes(!aql).empty(),
                                 aql ? "C06 native AQL graph contains device kernel holes" :
                                       "C06 instantiated host kernel coverage incomplete");
                        }
                        // The declared Q, K and output reductions each require
                        // an instance on each rank. No second collective is made.
                        need(reg.n_instances() >= instances_before + 3 * builders.size(),
                             "C06 did not initialize three collectives on every rank");
                        initialized += reg.n_instances() - instances_before;
                        if (issue) {
                            for (int rank=0;rank<3;++rank) {
                                C06Issued& r=issued[rank];
                                if(aql) need(rad_dev_set(rank)==RAD_OK,
                                             "C06 AQL rank allocation device missing");
                                rad::Program& prog=builders[rank]->program();
                                verify_c06_load_contract(model,prog,rank,&r);
                                r.ctx=std::make_unique<rad::Ctx>();
                                rad::CtxDesc d{};
                                d.program=&prog;
                                d.rank=rank;
                                d.world_size=3;
                                d.device=aql?rank:0;
                                d.arch_step=c06_issued_step;
                                d.record_passes=false;
                                d.registry=&reg;
                                need(r.ctx->init(d)==RAD_OK,"C06 real model RadCtx init failed");
                                if (aql) {
                                    for(size_t i=1;i<prog.weights.size();++i) {
                                        const auto& w=prog.weights[i];
                                        need(rad_memcpy_async(w.ptr,r.weights[i].data(),
                                                               w.stored_bytes,r.ctx->stream())==RAD_OK,
                                             "C06 native checkpoint weight upload failed");
                                    }
                                    r.device_cu=rad_dev_alloc(sizeof(r.cu),RAD_MEM_DEVICE);
                                    need(r.device_cu,
                                         "C06 native AQL cu-seqlens allocation refused");
                                    r.owned_device.push_back(r.device_cu);
                                    r.device_pos=rad_dev_alloc(r.pos.size()*sizeof(float),RAD_MEM_DEVICE);
                                    need(r.device_pos,
                                         "C06 native AQL position allocation refused");
                                    r.owned_device.push_back(r.device_pos);
                                    need(rad_memcpy_async(r.device_cu,r.cu.data(),
                                                          sizeof(r.cu),r.ctx->stream())==RAD_OK &&
                                         rad_memcpy_async(r.device_pos,r.pos.data(),
                                                          r.pos.size()*sizeof(float),r.ctx->stream())==RAD_OK,
                                         "C06 native AQL coordinate transfer failed");
                                    std::vector<uint16_t> sample((size_t)(r.width*r.tokens));
                                    for(int64_t j=0;j<r.width*r.tokens;++j)
                                        sample[(size_t)j]=c06_sample(j);
                                    void* input=r.ctx->buf_ptr(r.x.a.x);
                                    need(input && rad_memcpy_async(input,sample.data(),
                                                r.width*r.tokens*2,r.ctx->stream())==RAD_OK &&
                                         rad_stream_sync(r.ctx->stream())==RAD_OK,
                                         "C06 native AQL activation transfer failed");
                                }
                            }
                            std::array<int,3> status{RAD_E_STATE,RAD_E_STATE,RAD_E_STATE};
                            std::array<std::thread,3> runners;
                            for(int rank=0;rank<3;++rank)
                                runners[rank]=std::thread([&,rank] {
                                    C06Issued& r=issued[rank];
                                    c06_active=&r;
                                    const int rc=r.ctx->bind_thread();
                                    RadBatch batch{};
                                    batch.phase=RAD_PHASE_PREFILL;
                                    batch.n_tok=r.tokens;
                                    batch.n_seq=1;
                                    status[rank]=rc==RAD_OK?r.ctx->run_step(&batch):rc;
                                    if(status[rank]==RAD_OK && aql)
                                        status[rank]=rad_stream_sync(r.ctx->stream());
                                    c06_active=nullptr;
                                });
                            for(auto& runner:runners) runner.join();
                            for(int rank=0;rank<3;++rank) {
                                C06Issued& r=issued[rank];
                                if(status[rank]!=RAD_OK)
                                    std::fprintf(stderr,"C06 issued rank %d error %d: %s\n",
                                                 rank,status[rank],r.ctx->step_error());
                                need(status[rank]==RAD_OK,
                                     "C06 checkpoint-backed self-attention issue refused");
                                need(r.ctx->ar_calls()==3,
                                     "C06 must issue exactly three native reductions per rank");
                            }
                            const int64_t full=issued[0].width;
                            std::array<std::vector<uint16_t>,3> results;
                            for(int rank=0;rank<3;++rank) {
                                C06Issued& r=issued[rank];
                                need(rad_dev_set(aql?rank:0)==RAD_OK,
                                     "C06 output rank device bind refused");
                                void* ptr=r.ctx->buf_ptr(r.output);
                                need(ptr!=nullptr,"C06 output arena buffer missing");
                                results[rank].resize((size_t)(full*r.tokens));
                                if(aql) {
                                    need(rad_memcpy_async(results[rank].data(),ptr,
                                                         full*r.tokens*2,r.ctx->stream())==RAD_OK &&
                                         rad_stream_sync(r.ctx->stream())==RAD_OK,
                                         "C06 native AQL attention output download failed");
                                } else {
                                    std::memcpy(results[rank].data(),ptr,(size_t)(full*r.tokens)*2);
                                }
                            }
                            const auto& control=results[0];
                            size_t nonzero=0;
                            for(size_t j=0;j<control.size();++j) {
                                const uint32_t b=(uint32_t)control[(size_t)j]<<16;
                                float x;
                                std::memcpy(&x,&b,sizeof x);
                                need(std::isfinite(x),"C06 checkpoint attention output is not finite");
                                nonzero+=control[(size_t)j]!=0;
                            }
                            need(nonzero>0,"C06 checkpoint attention output unexpectedly all zero");
                            for(int rank=1;rank<3;++rank)
                                need(control==results[rank],
                                     "C06 exact three-rank BF16 output replicas disagree");
                            auto& prior=split_reference[audio?1:0];
                            if(prior.empty()) prior=control;
                            else if(!aql) {
                                need(prior.size()==control.size(),
                                     "C06 TP3 splits changed checkpoint output width");
                                int64_t unequal=0;
                                double max_abs=0, sum_sq=0, max_rel=0;
                                for(size_t j=0;j<control.size();++j) {
                                    const auto f32=[](uint16_t v) {
                                        const uint32_t u=(uint32_t)v<<16;
                                        float x; std::memcpy(&x,&u,sizeof x);
                                        return x;
                                    };
                                    const double v=f32(control[j]), base=f32(prior[j]);
                                    const double err=std::abs(v-base);
                                    unequal+=control[j]!=prior[j];
                                    max_abs=std::max(max_abs,err);
                                    sum_sq+=err*err;
                                    max_rel=std::max(max_rel,err/std::max(1e-3,std::abs(base)));
                                }
                                std::printf("C06 split comparison %s: unequal=%lld/%zu max_abs=%g rms=%g max_rel=%g\n",
                                    audio?"audio":"video",(long long)unequal,control.size(),
                                    max_abs,std::sqrt(sum_sq/control.size()),max_rel);
                                std::fflush(stdout);
                                // Different BF16 partial sums can round differently
                                // when head ownership changes. This is measured, not
                                // accepted as TP1 parity or silenced by a guessed cap.
                            }
                            std::printf("C06 issued %s: %s heads=%d,%d,%d exact BF16 output replicas %lld values; no TP1 capture\n",
                                        aql?"AQL":"host",audio?"audio":"video",
                                        split[0],split[1],split[2],(long long)control.size());
                            // Optional diagnostic capture of the *validated* rank-0
                            // BF16 output, for an external TP1/backend comparison.
                            // The caller explicitly owns the destination directory.
                            if(const char* root=std::getenv("BC_LTX_C06_CAPTURE_DIR");
                               root && *root) {
                                const char* supplied=std::getenv("BC_LTX_C06_CAPTURE_TAG");
                                const std::string tag=supplied && *supplied ? supplied :
                                                          (aql?"aql":"host");
                                need(!tag.empty() && std::all_of(tag.begin(),tag.end(),
                                     [](unsigned char c) {
                                         return (c>='a' && c<='z') ||
                                                (c>='A' && c<='Z') ||
                                                (c>='0' && c<='9') || c=='_' || c=='-';
                                     }),"C06 capture tag must be a simple label");
                                const auto path=std::filesystem::path(root)/
                                    (tag+"-"+(audio?"audio":"video")+"-"+
                                     std::to_string(split[0])+"-"+
                                     std::to_string(split[1])+"-"+
                                     std::to_string(split[2])+".bf16");
                                std::ofstream f(path,std::ios::binary|std::ios::trunc);
                                need(f.is_open(),"C06 output capture directory inaccessible");
                                f.write((const char*)control.data(),
                                        (std::streamsize)(control.size()*sizeof(uint16_t)));
                                need(f.good(),"C06 output capture write failed");
                                for (int rank=0;rank<3;++rank)
                                    for (const auto& captured:issued[rank].stages) {
                                        const auto path=std::filesystem::path(root)/
                                            (tag+"-"+(audio?"audio":"video")+"-"+
                                             std::to_string(split[0])+"-"+
                                             std::to_string(split[1])+"-"+
                                             std::to_string(split[2])+"-r"+
                                             std::to_string(rank)+"-"+
                                             captured.stage+"."+captured.extension);
                                        std::ofstream stream(path,
                                                             std::ios::binary|std::ios::trunc);
                                        need(stream.is_open(),
                                             "C06 stage capture directory inaccessible");
                                        stream.write((const char*)captured.data.data(),
                                            (std::streamsize)captured.data.size());
                                        need(stream.good(),"C06 stage capture write failed");
                                    }
                            }
                            for(auto& r:issued) r.ctx.reset();
                        }
                        // No kernel instance may keep pointers into a
                        // destroyed Builder. Release all domains while the
                        // three rank Programs still own their geometry.
                        reg.fini_instances();
                        need(reg.n_instances() == 0,
                             "C06 native collective instances were not retired");
                        ++groups;
                    }
                need(groups == 4, "C06 instance coverage drift");
                if (issue)
                    std::printf("LTX C06: %d three-rank issued %s checkpoint attention graphs; %zu retired native instances\n",
                                groups,aql?"AQL":"host",initialized);
                else
                    std::printf("LTX C06: %d three-rank %s graphs; %zu initialized native instances; no model execution claimed\n",
                                groups,aql?"AQL":"host",initialized);
                return 0;
            }
            if (std::string(argv[3]) == "--c06") {
                int tested=0;
                for(int64_t block=0; block<nblocks; ++block)
                    for(bool audio:{false,true})
                        for(const std::array<int,3>& split :
                            {std::array<int,3>{8,8,16},
                             std::array<int,3>{12,12,8}}) {
                            int first=0;
                            for(int rank=0; rank<3; ++rank) {
                                run_self_attn(model,reg,block,audio,rank,
                                              first,split[rank],nheads);
                                first+=split[rank];
                                ++tested;
                            }
                            need(first==nheads,"C06 head ownership gap");
                        }
                need(tested==nblocks*2*2*3,
                     "C06 self-attention graph matrix drift");
                std::printf("LTX C06 source-backed self-attention: %d positive native graphs, two modalities, two splits\n", tested);
                return 0;
            }
            if (std::string(argv[3]) == "--c05") {
                const std::pair<const char*, int> qkv_kv[] = {
                    {"attn1.qkv",3}, {"audio_attn1.qkv",3},
                    {"attn2.kv",2}, {"audio_attn2.kv",2},
                    {"a2v.kv",2}, {"v2a.kv",2}
                };
                int proven = 0;
                for (int64_t block = 0; block < nblocks; ++block) {
                    const std::string prefix =
                        "dit.blk." + std::to_string(block) + ".";
                    for (const std::array<int,3>& split :
                         {std::array<int,3>{8,8,16},
                          std::array<int,3>{12,12,8}}) {
                        int offset = 0;
                        for (int rank = 0; rank < 3; ++rank) {
                            for (const auto& kind : qkv_kv) {
                                run(model, reg, prefix+kind.first, kind.second,
                                    nheads, rank, offset, split[rank],
                                    ProbeCase::GlobalRms);
                                ++proven;
                            }
                            offset += split[rank];
                        }
                        need(offset == nheads,
                             "C05 ownership does not cover all heads");
                    }
                }
                need(proven == nblocks * 2 * 3 *
                    static_cast<int>(sizeof qkv_kv/sizeof qkv_kv[0]),
                    "C05 graph census changed");
                std::printf("LTX2 C05 global Q/K RMS: %d fused projections, %d native graph ops, geometry refusals passed\n",
                            proven, proven * 10);
                return 0;
            }
            need(std::string(argv[3]) == "--c04", "unexpected native test mode");
            int proven = 0;
            for (int64_t block = 0; block < nblocks; ++block) {
                const std::string prefix = "dit.blk." + std::to_string(block) + ".";
                for (const std::array<int,3>& split :
                         {std::array<int,3>{8,8,16}, {12,12,8}}) {
                    int offset = 0;
                    for (int rank = 0; rank < 3; ++rank) {
                        for (const char* kind : kBiasedOutputs) {
                            run(model, reg, prefix+kind,
                                0, nheads, rank, offset, split[rank], ProbeCase::Reduced);
                            ++proven;
                        }
                        offset += split[rank];
                    }
                    need(offset == nheads, "C04 rank split lost model heads");
                }
            }
            run(model, reg, "dit.blk.0.audio_attn1.out", 0, nheads,
                0, 0, 10, ProbeCase::InvalidDeferred);
            need(proven == nblocks * 2 * 3 *
                    static_cast<int>(sizeof kBiasedOutputs/sizeof kBiasedOutputs[0]),
                 "C04 source-backed graph coverage changed");
            std::printf("LTX2 C04 native exact reductions + post-bias: %d, invalid-K=1, duplicate=refused\n", proven);
            return 0;
        }
        const std::pair<const char*,int> projections[] = {
            {"attn1.qkv",3}, {"audio_attn1.qkv",3},
            {"attn2.kv",2}, {"audio_attn2.kv",2},
            {"a2v.kv",2}, {"v2a.kv",2},
            {"ff_down",0}
        };
        int count=0;
        for (int64_t block = 0; block < nblocks; ++block) {
            const std::string prefix = "dit.blk." + std::to_string(block) + ".";
            for (const std::array<int,3>& split :
                    {std::array<int,3>{8,8,16}, {12,12,8}, {10,10,12}}) {
                need(split[0]+split[1]+split[2]==nheads,
                     "split no longer covers model heads");
                int offset=0;
                for (int rank=0;rank<3;++rank) {
                    for (const auto& p : projections) {
                        run(model,reg,prefix+p.first,p.second,nheads,
                            rank,offset,split[rank]);
                        ++count;
                    }
                    offset+=split[rank];
                }
            }
        }
        // Every biased attention/cross-modal and audio FFN-down output:
        // model source geometry decides whether the IQ8R K cut is legal.
        int deferred_count = 0, blocked_count = 0;
        for (int64_t block = 0; block < nblocks; ++block) {
            const std::string prefix = "dit.blk." + std::to_string(block) + ".";
            for (const std::array<int,3>& split :
                     {std::array<int,3>{8,8,16}, {12,12,8}, {10,10,12}}) {
                int offset = 0;
                for (int rank = 0; rank < 3; ++rank) {
                    for (const char* suffix : kBiasedOutputs) {
                        const std::string base = prefix + suffix;
                        const auto* e = model.find(base + ".weight");
                        need(e && e->rank == 2, "missing biased output weight");
                        need(e->shape[1] % nheads == 0,
                             "biased output K cannot be owned by whole heads");
                        const auto* be = model.find(base + ".bias");
                        need(be && be->rank == 1 && be->shape[0] == e->shape[0],
                             "missing or incompatible post-collective bias");
                        const auto& enc = model.encoding(*e);
                        need(rad::arch::linear_is_iq8r(enc),
                             "output encoding changed");
                        const int64_t hd = e->shape[1] / nheads;
                        const int64_t wg = rad::arch::linear_w8h_group(enc);
                        const int64_t local = hd * split[rank];
                        const auto slice = ltx2::C01Slice::columns_deferred(
                                e->shape[1], hd * offset, local);
                        const bool valid = slice.valid(e->shape[0], local, wg);
                        // Every model shape is checked. Only one block
                        // needs to trigger each native negative preflight.
                        if (valid || block == 0)
                            run(model, reg, base, 0, nheads, rank, offset,
                                split[rank], valid ? ProbeCase::DeferredBias : ProbeCase::InvalidDeferred);
                        deferred_count += valid;
                        blocked_count += !valid;
                    }
                    offset += split[rank];
                }
                need(offset == nheads, "biased rank slices do not cover heads");
            }
        }
        // 10/10/12 cuts four audio K projections on the first two
        // ranks per block; rank 2 begins on a scale-group boundary.
        const int rank_combinations = 3 * 3;
        const int invalid_per_block = 4 * 2;
        need(blocked_count == nblocks * invalid_per_block &&
             deferred_count == nblocks *
                 ((int)(sizeof kBiasedOutputs / sizeof kBiasedOutputs[0]) * rank_combinations -
                  invalid_per_block),
             "source model's biased-output partition tally changed");
        run(model,reg,"dit.blk.0.attn1.qkv",3,nheads,0,0,8,ProbeCase::WrongGroup);
        run(model,reg,"dit.blk.0.audio_ff_down",0,nheads,0,0,8,ProbeCase::PrematureBias);
        run(model,reg,"dit.blk.0.audio_ff_down",0,nheads,0,0,8,ProbeCase::Normal);
        run(model,reg,"dit.blk.0.audio_ff_down",0,nheads,0,0,8,ProbeCase::DeferredBias);
        run(model,reg,"dit.blk.0.audio_attn1.out",0,nheads,0,0,10,ProbeCase::InvalidColumn);
        run(model,reg,"dit.blk.0.ff_down",0,nheads,0,0,8,ProbeCase::DuplicateDeferredBias);
        std::printf("LTX2 native Lin declarations: fused/biasless=%d, deferred-bias=%d, rejected-K=%d, negative=4\n",
                    count, deferred_count, blocked_count);
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"LTX2 C02 native Lin failure: %s\n",e.what());
        return 1;
    }
}
