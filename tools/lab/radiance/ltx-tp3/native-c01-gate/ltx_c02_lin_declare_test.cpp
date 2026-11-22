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
#include "ltx_native_aql_roster.h"
#include "ltx2.cpp"

#include <array>
#include <memory>
#include <thread>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

using rad::Builder;
using rad::RadFile;
using rad::Registry;

static void need(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}
static RadBuildCtx context(int rank) {
    RadBuildCtx c{};
    c.rank = rank;
    c.world_size = 3;
    c.max_tok = 4;
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
    DeferredBias, InvalidDeferred, MissingDeferredBias, Reduced, GlobalRms
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
    const bool deferred = variant == ProbeCase::DeferredBias || variant == ProbeCase::InvalidDeferred || variant == ProbeCase::MissingDeferredBias || variant == ProbeCase::Reduced;
    const bool denied = variant == ProbeCase::WrongGroup || variant == ProbeCase::PrematureBias || variant == ProbeCase::InvalidColumn || variant == ProbeCase::InvalidDeferred || variant == ProbeCase::MissingDeferredBias;
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
    } else if (variant == ProbeCase::MissingDeferredBias) {
        rc = linear.declare(&b,nm,g,base.c_str(),
             {"checkpoint.ff_down.weight"},{},{},
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
        wrong.world = 2;
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
        wrong.world = 2;
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


// Audit the frozen runtime loader's *own* share geometry for each
// checkpoint plane. This is not a replacement loader or a model issue.
static void verify_c06_load_contract(const RadFile& model, const rad::Program& program,
                                      int rank) {
    int64_t selected_bytes = 0, stored_bytes = 0;
    int weights = 0, relayouts = 0;
    for (size_t i=1; i<program.weights.size(); ++i) {
        const rad::WeightInfo& w = program.weights[i];
        const auto* e = model.find(w.source);
        need(e != nullptr && w.enc_known && w.n_sel > 0,
             "C06 declared weight lacks real checkpoint source/plane");
        rad::PlaneShare shares[RAD_ENC_MAX_PLANES];
        std::string why;
        const int rc = rad::weight_share(w,model.encoding(*e),e->rank,e->shape,
                                          3,rank,shares,&why);
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
        } else {
            ++relayouts; // Cannot be directly memcpy'ed: use engine loader.
            // Record the exact kernel layout that blocks naive checkpoint
            // pointer binding in a future issue test.
            if (rank == 0)
                std::printf("C06 engine relayout required: %s -> %s (%s, source=%lld stored=%lld)\n",
                            w.source.c_str(),w.name.c_str(),w.layout_tag.c_str(),
                            (long long)share_bytes,(long long)w.stored_bytes);
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
                          bool inspect_weights = false, bool aql = false) {
    const std::string kind = audio ? "audio_attn1" : "attn1";
    const std::string prefix = "dit.blk." + std::to_string(block) + "." + kind;
    const auto* fused = model.find(prefix + ".qkv.weight");
    const auto* output = model.find(prefix + ".out.weight");
    const auto* gate = model.find(prefix + ".gate.weight");
    const auto* gnq = model.find(prefix + ".q_norm.weight");
    const auto* gnk = model.find(prefix + ".k_norm.weight");
    need(fused && output && gate && gnq && gnk,
         "C06 model missing a self-attention source");
    const int64_t full = fused->shape[1];
    need(fused->rank == 2 && fused->shape[0] == 3 * full &&
         full % total_heads == 0 &&
         output->rank == 2 && output->shape[0] == full &&
         output->shape[1] == full &&
         gate->rank == 2 && gate->shape[0] == total_heads &&
         gate->shape[1] == full &&
         gnq->rank == 1 && gnq->shape[0] == full &&
         gnk->rank == 1 && gnk->shape[0] == full,
         "C06 source self-attention geometry mismatch");
    const int64_t hd = full / total_heads, local = owned * hd, lo = first * hd;
    const auto& enc = model.encoding(*fused);
    need(rad::arch::linear_is_iq8r(enc) &&
         rad::arch::linear_w8h_group(enc) == rad::arch::RAD_I8_GROUP_WIDE,
         "C06 IQ8R group encoding changed");
    const ltx2::C01Slice slice = ltx2::C01Slice::rows(full, lo, local, hd, 3);
    need(slice.valid(3 * local, full, rad::arch::RAD_I8_GROUP_WIDE),
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
    g.max_tok = 4;
    g.dtype = "bf16";
    ltx2::Act x{}, ao{};
    need(x.declare(&b, m.nm, g, "c06.input", 4, full) == RAD_OK &&
         ao.declare(&b, m.nm, g, "c06.attention", 4, local) == RAD_OK,
         "C06 real activation/quantization declaration failed");
    need(x.q.op && ao.q.op && x.a.q8_fed && ao.a.q8_fed,
         "C06 failed to declare the exact native IQ8R activation quantizers");
    const rad_buf qkv = ltx2::decl_b(&b, m.nm.f("c06.qkv"), RAD_BF16, {4, 3 * local});
    const rad_buf logits = ltx2::decl_b(&b, m.nm.f("c06.gate"), RAD_BF16, {4, total_heads});
    const rad_buf out = ltx2::decl_b(&b, m.nm.f("c06.out"), RAD_BF16, {4, full});
    const rad_buf scratch = ltx2::decl_b(&b, m.nm.f("c06.norm_scratch"), RAD_BF16,
                                         {4, full});
    need(x.a.x && x.a.q8 && x.a.s8 && ao.a.x && ao.a.q8 && ao.a.s8 &&
         qkv && logits && out && scratch, "C06 activation buffers missing");
    const ltx2::Rope rope = audio ? ltx2::Rope{1, full, "20"} :
                                   ltx2::Rope{3, full, "20 2048 2048"};
    const std::string original = "transformer.transformer_blocks." +
                                 std::to_string(block) + "." + kind;
    ltx2::Attn a{};
    const auto before = b.program().weights.size();
    for(int bad=0; bad<2; ++bad) {
        ltx2::Geom wrong = g;
        if(bad==0) wrong.world=2;
        else wrong.wire_exact=0;
        need(ltx2::attn_declare(&b, m, wrong, a, prefix.c_str(), original.c_str(),
                               total_heads, hd, full, full, full, true, x, nullptr,
                               qkv, 0, logits, ao, out, 4, 4, RAD_ACCESS_PER_TOKEN,
                               0, rope, 1e-6, &slice, scratch) == RAD_E_UNSUPPORTED,
             "C06 invalid native geometry admitted");
    }
    ltx2::Act unquantized = x;
    unquantized.q.op = 0;
    need(ltx2::attn_declare(&b, m, g, a, prefix.c_str(), original.c_str(),
                           total_heads, hd, full, full, full, true,
                           unquantized, nullptr, qkv, 0, logits, ao, out,
                           4, 4, RAD_ACCESS_PER_TOKEN, 0, rope,
                           1e-6, &slice, scratch) == RAD_E_UNSUPPORTED,
         "C06 accepted missing input IQ8R quantizer");
    unquantized = ao;
    unquantized.q.op = 0;
    need(ltx2::attn_declare(&b, m, g, a, prefix.c_str(), original.c_str(),
                           total_heads, hd, full, full, full, true,
                           x, nullptr, qkv, 0, logits, unquantized, out,
                           4, 4, RAD_ACCESS_PER_TOKEN, 0, rope,
                           1e-6, &slice, scratch) == RAD_E_UNSUPPORTED,
         "C06 accepted missing output IQ8R quantizer");
    need(b.program().weights.size() == before && !a.tp3,
         "C06 rejected geometry allocated weights");
    const int rc=ltx2::attn_declare(
        &b, m, g, a, prefix.c_str(), original.c_str(),
        total_heads, hd, full, full, full, true, x, nullptr,
        qkv, 0, logits, ao, out, 4, 4, RAD_ACCESS_PER_TOKEN,
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
    need(a.gate.N == total_heads && a.gate.K == full && a.gate.bias,
         "C06 replicated head gate missing");
    need(a.op_rq && a.op_rk && a.op_attn && a.op_hg && a.op_v &&
         a.qkv.op && a.out.op && a.out.post_reduce && a.out.op_b &&
         a.tp_q.reduce && a.tp_k.reduce && !a.op_qn && !a.op_kn,
         "C06 native self-attention graph incomplete");
    for(const rad_op h:{a.op_rq,a.op_rk}) {
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
        need(op && op->base.get_i("world_size",&world) && world==3 &&
             op->base.get_i("exact",&exact) && exact==1,
             "C06 rank requires BigCherry's existing exact TP3 collective");
    }
    check_parts(*b.program().weight(a.qkv.l.w), slice);
    check_parts(*b.program().weight(a.qkv.l.ws), slice);
    check_parts(*b.program().weight(a.qkv.bias), slice);
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
                              a.op_rq, a.op_rk, a.op_attn, a.op_hg, a.op_v}) {
        const auto* info = b.program().op(handle);
        need(info != nullptr, "C06 native operation missing");
        for(const auto& band : info->bands)
            need(bool(band.dom[aql ? RAD_DOMAIN_DEVICE : RAD_DOMAIN_HOST]),
                 aql ? "C06 native AQL operation has no device kernel" :
                       "C06 op unresolved on host reference kernel");
    }
    need(b.errors().empty(), "C06 Builder diagnostics");
    if (instance) {
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

int main(int argc, char** argv) {
    if(argc!=3 && argc!=4) {
        std::fprintf(stderr,"usage: ltx_c02_lin_declare_test MODEL.rad PLUGIN_KERNEL_DIR\n");
        return 2;
    }
    try {
        const bool aql = argc == 4 && std::string(argv[3]) == "--c06-instances-aql";
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

        if (argc == 4) {
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
            if (std::string(argv[3]) == "--c06-instances" || aql) {
                // This is native INSTANCING only, not issued C06,
                // checkpoint weight residency, or video parity. Reuse the
                // same model-backed self-attention declarations as --c06.
                int groups = 0;
                size_t initialized = 0;
                for (bool audio : {false, true})
                    for (const std::array<int,3>& split :
                         {std::array<int,3>{8,8,16}, std::array<int,3>{12,12,8}}) {
                        std::array<std::unique_ptr<Builder>,3> builders;
                        // Keep every Program's frozen geometry alive until
                        // native kernel fini, including when an assertion
                        // throws during rank validation.
                        struct Retire {
                            Registry& registry;
                            ~Retire() { registry.fini_instances(); }
                        } retire{reg};
                        int first = 0;
                        for (int rank=0; rank<3; ++rank) {
                            run_self_attn(model,reg,0,audio,rank,
                                          first,split[rank],nheads,&builders[rank],
                                          false,aql);
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
                        // No kernel instance may keep pointers into a
                        // destroyed Builder. Release all domains while the
                        // three rank Programs still own their geometry.
                        reg.fini_instances();
                        need(reg.n_instances() == 0,
                             "C06 native collective instances were not retired");
                        ++groups;
                    }
                need(groups == 4, "C06 instance coverage drift");
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
        run(model,reg,"dit.blk.0.ff_down",0,nheads,0,0,8,ProbeCase::MissingDeferredBias);
        std::printf("LTX2 native Lin declarations: fused/biasless=%d, deferred-bias=%d, rejected-K=%d, negative=4\n",
                    count, deferred_count, blocked_count);
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"LTX2 C02 native Lin failure: %s\n",e.what());
        return 1;
    }
}
