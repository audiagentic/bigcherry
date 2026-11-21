/* Diagnostic actual LTX2 Lin::declare exercise against Radiance 2.2 / ABI44.
 * Test includes the patched architecture implementation: no alternative Lin,
 * plugin launch, real GPU, or claim of TP3 service.
 * Only the weight/activation declaration and the real core's op registry run.
 */
#include "build/rad_build.h"
#include "build/startup.h"
#include "format/radfile.h"
#include "ltx2.cpp"

#include <array>
#include <cstdio>
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

static void run(const RadFile& model, Registry& reg, const std::string& base,
                int banks, int nheads, int rank, int start_heads, int owned_heads,
                int refusal = 0) {
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
    const auto slice = rows
        ? ltx2::C01Slice::rows(full,offset,width,hd,banks)
        : ltx2::C01Slice::columns(e->shape[1],offset,width);
    need(refusal || slice.valid(rows?banks*width:e->shape[0],
                                   rows?e->shape[1]:width,group),
         "test generated invalid positive C01 geometry");

    Builder b(reg,metadata(),context(rank));
    b.set_sources(rad::rad_container_sources(model));
    ltx2::Names nm;
    ltx2::Geom g{};
    g.w_i8h = true;
    g.w_i8_group = refusal == 1 ? group/2 : group;
    g.w_bf16 = false;
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
    } else if(refusal == 2) {
        rc = linear.declare(&b,nm,g,base.c_str(),
             {"checkpoint.audio_ff_down.weight"},{"checkpoint.audio_ff_down.bias"},{},
             out_dim,width,RAD_ACCESS_PER_TOKEN,0,input,y,4,false,&slice);
    } else {
        rc = linear.declare(&b,nm,g,base.c_str(),
             {"checkpoint.ff_down.weight"},{},{},
             out_dim,width,RAD_ACCESS_PER_TOKEN,0,input,y,4,false,&slice);
    }
    if (refusal) {
        need(rc==RAD_E_UNSUPPORTED, "unsupported C01 projection did not fail closed");
        need(b.program().weights.size()==1,"refused slice still allocated model weights");
        return;
    }
    if(rc!=RAD_OK) {
        std::fprintf(stderr,"Lin::declare rc=%d, rank=%d, shape=%lldx%lld\n",
                     rc,rank,(long long)out_dim,(long long)input.n);
        for(const auto& msg:b.errors()) std::fprintf(stderr,"  %s\n",msg.c_str());
    }
    need(rc==RAD_OK,"real LTX Lin::declare refused the shard");
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
        need(!linear.bias && !linear.op_b,"biasless Lin unexpectedly added bias");
        check_col(*cw,slice);
        check_col(*sw,slice);
    }
}

int main(int argc, char** argv) {
    if(argc!=3) {
        std::fprintf(stderr,"usage: ltx_c02_lin_declare_test MODEL.rad PLUGIN_KERNEL_DIR\n");
        return 2;
    }
    try {
        RadFile model;
        need(model.open(argv[1])==RAD_OK,"cannot open LTX source model");
        const auto* row=model.find("dit.blk.0.attn1.qkv.weight");
        need(row && row->shape[0]/3==row->shape[1],
             "source fused video QKV is no longer square by bank");
        Registry reg;
        const int registry_status=reg.load_dir(argv[2],{"libref"});
        need(registry_status==RAD_OK,"2.2 reference kernel registry cannot load");
        const char* block_text = model.get_s("transformer.num_layers", "");
        const char* head_text = model.get_s("transformer.num_attention_heads", "");
        need(*block_text && *head_text, "model does not declare DiT geometry");
        const int64_t nblocks = std::stoll(block_text);
        const int nheads = std::stoi(head_text);
        need(nblocks > 0 && nheads > 0, "invalid DiT geometry");

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
        run(model,reg,"dit.blk.0.attn1.qkv",3,nheads,0,0,8,1);
        run(model,reg,"dit.blk.0.audio_ff_down",0,nheads,0,0,8,2);
        run(model,reg,"dit.blk.0.audio_attn1.out",0,nheads,0,0,10,3);
        std::printf("LTX2 C02 real Lin::declare model-weight and op receipts: %d, fail-closed=3\n",count);
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"LTX2 C02 native Lin failure: %s\n",e.what());
        return 1;
    }
}
