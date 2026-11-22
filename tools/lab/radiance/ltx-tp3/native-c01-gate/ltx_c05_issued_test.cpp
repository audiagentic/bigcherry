/* LTX02: issue the *existing* patched C05 RadCtx path with real Q/K
 * checkpoint gains on Radiance's native multi-rank host backend.
 * No second normalization or collective implementation is installed.
 * This is an executed oracle; native AQL/GPU parity remains a separate gate.
 */
#include "build/rad_build.h"
#include "build/startup.h"
#include "device/device.h"
#include "format/radfile.h"
#include "runtime/ctx.h"
#include "ltx_native_aql_roster.h"
#include "ltx2.cpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
void require(bool yes, const char* why) {
    if (!yes) throw std::runtime_error(why);
}
uint16_t bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof bits);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return uint16_t(bits >> 16);
}
float f32(uint16_t b) {
    uint32_t bits = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}
uint16_t sample(bool is_key, int row, int64_t global_col) {
    // Exactly representable BF16 inputs: interpretation does not rely on
    // a second quantizer or an invented model activation.
    const int64_t seed = global_col * 7 + row * 11 + (is_key ? 17 : 0);
    return bf16(float(seed % 61 - 30) / 32.0f);
}
struct Rank {
    std::unique_ptr<rad::Builder> builder;
    std::unique_ptr<rad::Ctx> ctx;
    ltx2::Tp3GlobalRms q, k;
    rad_buf bq = 0, bk = 0;
    rad_weight wq = 0, wk = 0;
    int64_t first = 0, owned = 0, full = 0, rows = 0;
    int device = 0;
    int status = RAD_E_STATE;
    std::string error;
    std::vector<void*> owned_device_weights;
    ~Rank() {
        (void)rad_dev_set(device);
        ctx.reset();  // no weight may be freed while RadCtx owns a stream
        for (void* p : owned_device_weights) rad_dev_free(p,RAD_MEM_DEVICE);
    }
};
thread_local Rank* active_rank = nullptr;
void step(RadCtx* c, const RadBatch*) {
    if (!active_rank) {
        rad_step_fail(c, "LTX02 missing rank execution owner");
        return;
    }
    active_rank->q.step(c, active_rank->bq, active_rank->rows);
    active_rank->k.step(c, active_rank->bk, active_rank->rows);
}
RadModelMeta metadata() {
    RadModelMeta m{};
    m.arch_id = "ltx2";
    m.name = "LTX02 issued Q/K native oracle";
    return m;
}
RadBuildCtx context(int rank, int64_t max_rows) {
    RadBuildCtx c{};
    c.rank = rank;
    c.world_size = 3;
    c.max_tok = max_rows;
    c.max_seqs = 1;
    c.max_ctx = 4096;
    c.scope = "";
    return c;
}
void verify_weight(rad::Program& p, rad_weight h, const rad::RadFile& model,
                   const std::string& name, int64_t full, Rank& rank, bool gpu) {
    const auto* e = model.find(name);
    require(e && e->rank == 1 && e->shape[0] == full,
            "LTX02 checkpoint Q/K gain source is missing or changed");
    const auto* w = p.weight(h);
    require(w && w->decl.shard == RAD_SHARD_NONE &&
            w->identity && w->stored_dtype == RAD_BF16 &&
            w->stored_bytes == rad_dtype_bytes(RAD_BF16, full),
            "LTX02 full-width BF16 gain requires native identity layout");
    const auto& plane = model.plane(*e, 0);
    require((int64_t)plane.bytes == w->stored_bytes,
            "LTX02 gain plane bytes mismatch");
    // Both backends consume the very same checkpoint BF16 plane; no
    // model parser, weight format transform, or alternate graph is added.
    if (gpu) {
        void* dev=rad_dev_alloc(w->stored_bytes,RAD_MEM_DEVICE);
        require(dev,"LTX02 device gain allocation failed");
        rank.owned_device_weights.push_back(dev);
        p.weight(h)->ptr=dev;
        p.weight(h)->site=rad::Site::Device;
    } else {
        p.weight(h)->ptr=const_cast<uint8_t*>(model.data(plane));
        p.weight(h)->site=rad::Site::Host;
    }
}
void run_case(const rad::RadFile& model, rad::Registry& reg,
              bool audio, int64_t rows, const std::array<int,3>& split,
              int64_t nheads, bool gpu) {
    std::fprintf(stderr,"LTX02 stage: build %s rows=%lld (%s)\n",
                 audio?"audio":"video",(long long)rows,
                 gpu?"AQL":"host");
    std::fflush(stderr);
    const std::string prefix = audio ? "dit.blk.0.audio_attn1" :
                                       "dit.blk.0.attn1";
    const auto* fused = model.find(prefix + ".qkv.weight");
    require(fused && fused->rank == 2 && fused->shape[0] == 3*fused->shape[1],
            "LTX02 actual QKV checkpoint geometry missing");
    const int64_t full = fused->shape[1];
    require(nheads > 0 && full % nheads == 0 &&
            split[0] + split[1] + split[2] == nheads,
            "LTX02 model-derived full head ownership invalid");
    const int64_t dim = full / nheads;
    const std::string qname = prefix + ".q_norm.weight";
    const std::string kname = prefix + ".k_norm.weight";
    const auto* eq = model.find(qname);
    const auto* ek = model.find(kname);
    require(eq && ek && eq->rank == 1 && ek->rank == 1 &&
            eq->shape[0] == full && ek->shape[0] == full,
            "LTX02 Q/K checkpoint gains absent");
    const auto* gain_q = reinterpret_cast<const uint16_t*>(
        model.data(model.plane(*eq, 0)));
    const auto* gain_k = reinterpret_cast<const uint16_t*>(
        model.data(model.plane(*ek, 0)));
    // Upstream LTX uses 1e-6 only when the checkpoint metadata has
    // no explicit teps/rms_norm_eps (ltx2.cpp's model builder).
    double eps = 1e-6;
    double from_model = 0.0;
    if (model.get_f("rms_norm_eps", &from_model)) eps = from_model;
    else if (const char* raw=model.get_s("rms_norm_eps")) eps=std::stod(raw);
    require(std::isfinite(eps) && eps > 0.0,
            "LTX02 invalid checkpoint RMS epsilon");
    std::array<std::unique_ptr<Rank>,3> ranks;
    int64_t first = 0;
    for (int rank=0;rank<3;++rank) {
        auto r = std::make_unique<Rank>();
        r->first = first*dim;
        r->owned = split[rank]*dim;
        r->full = full;
        r->rows = rows;
        r->device = gpu ? rank : 0;
        require(rad_dev_set(r->device)==RAD_OK,"LTX02 cannot bind native rank device");
        auto b = std::make_unique<rad::Builder>(
            reg, metadata(), context(rank,rows));
        b->set_sources(rad::rad_container_sources(model));
        ltx2::Names nm;
        ltx2::Geom geom{};
        geom.world = 3;
        geom.wire_exact = 1;
        geom.wire_min_bytes = 0;
        geom.dtype = "bf16";
        // Inputs are both populated by the test *before* the step, rather
        // than produced in graph declaration order. Keep them live across
        // the complete pass so the native arena planner cannot alias them.
        const auto mkbuf = [&](const char* name, int64_t n, int kind) {
            return ltx2::decl_b(b.get(),nm.f("%s",name),
                                RAD_BF16,{rows,n},kind);
        };
        r->bq = mkbuf("ltx02.query",r->owned,RAD_BUF_PERSIST);
        r->bk = mkbuf("ltx02.key",r->owned,RAD_BUF_PERSIST);
        const rad_buf scratch=mkbuf("ltx02.shared",full,RAD_BUF_TRANSIENT);
        require(r->bq && r->bk && scratch, "LTX02 native buffers undeclared");
        const rad_weight wq=ltx2::decl_w(
            b.get(),nm.f("%s",qname.c_str()),RAD_BF16,{full},
            RAD_ACCESS_PER_TOKEN,RAD_SHARD_NONE,ltx2::grp_layer(0));
        const rad_weight wk=ltx2::decl_w(
            b.get(),nm.f("%s",kname.c_str()),RAD_BF16,{full},
            RAD_ACCESS_PER_TOKEN,RAD_SHARD_NONE,ltx2::grp_layer(0));
        require(wq && wk, "LTX02 model gains not declared");
        require(r->q.declare(b.get(),geom,r->bq,scratch,wq,full,
                            r->first,r->owned,0,dim,rows,eps)==RAD_OK,
                "LTX02 existing Q all-head graph refused");
        require(r->k.declare(b.get(),geom,r->bk,scratch,wk,full,
                            r->first,r->owned,0,dim,rows,eps)==RAD_OK,
                "LTX02 existing K all-head graph refused");
        // Keep declaration source/native handles; **finish** must be run
        // concurrently on the three ranks. The actual Radiance AR
        // init() rendezvouses with its peers during graph finish.
        r->wq=wq;
        r->wk=wk;
        r->builder=std::move(b);
        ranks[rank]=std::move(r);
        first+=split[rank];
    }
    std::array<std::thread,3> builders;
    for(int rank=0;rank<3;++rank)
        builders[rank]=std::thread([&,rank] {
            Rank& r=*ranks[rank];
            try {
                r.status=rad_dev_set(r.device);
                if (r.status==RAD_OK)
                    r.status=r.builder->finish();
                if (r.status==RAD_OK && !r.builder->errors().empty())
                    r.status=RAD_E_STATE;
            } catch (const std::exception& e) {
                r.status=RAD_E_STATE;
                r.error=e.what();
            }
            if (gpu) {
                std::fprintf(stderr,"LTX02 stage: rank%d builder finish rc=%d\n",
                             rank,r.status);
                std::fflush(stderr);
            }
        });
    for(auto& thread:builders)thread.join();
    for(int rank=0;rank<3;++rank) {
        Rank& r=*ranks[rank];
        if (r.status!=RAD_OK) {
            std::fprintf(stderr,"LTX02 rank%d finish refused rc=%d %s\n",
                         rank,r.status,r.error.c_str());
            for(const auto& line:r.builder->errors())
                std::fprintf(stderr,"  %s\n",line.c_str());
        }
        require(r.status==RAD_OK,"LTX02 native builder rank refused");
        require(r.builder->program().device_holes(!gpu).empty(),
                gpu ? "LTX02 native AQL GPU kernel coverage incomplete" :
                      "LTX02 native host registry has unresolved operations");
        require(rad_dev_set(r.device)==RAD_OK,"LTX02 native device rebind refused");
        verify_weight(r.builder->program(),r.wq,model,qname,full,r,gpu);
        verify_weight(r.builder->program(),r.wk,model,kname,full,r,gpu);
        r.ctx = std::make_unique<rad::Ctx>();
        rad::CtxDesc d{};
        d.program=&r.builder->program();
        d.rank=rank;
        d.world_size=3;
        d.device=r.device;
        d.arch_step=step;
        d.record_passes=false;
        d.registry=&reg;
        require(r.ctx->init(d)==RAD_OK,"LTX02 native RadCtx init failed");
        if(gpu) {
            // Stage the two real model planes through the Ctx-owned AQL
            // compute stream. No new loader, quantization or transport.
            for (const auto [h,name] :
                 {std::pair{r.wq,qname},std::pair{r.wk,kname}}) {
                const auto* e=model.find(name);
                const auto& plane=model.plane(*e,0);
                const auto* w=r.builder->program().weight(h);
                require(rad_memcpy_async(w->ptr,model.data(plane),
                                         w->stored_bytes,r.ctx->stream())==RAD_OK,
                        "LTX02 native checkpoint gain transfer failed");
            }
            require(rad_stream_sync(r.ctx->stream())==RAD_OK,
                    "LTX02 native checkpoint gain transfer did not complete");
        }
    }
    std::array<std::vector<uint16_t>,3> outq,outk;
    for (int rank=0;rank<3;++rank) {
        Rank& r=*ranks[rank];
        const int64_t n=rows*r.owned;
        outq[rank].resize(size_t(n));
        outk[rank].resize(size_t(n));
        auto* q=static_cast<uint16_t*>(r.ctx->buf_ptr(r.bq));
        auto* k=static_cast<uint16_t*>(r.ctx->buf_ptr(r.bk));
        require(q && k && q!=k,
                "LTX02 Q/K fixture inputs unexpectedly share native arena storage");
        std::vector<uint16_t> hostq(static_cast<size_t>(n));
        std::vector<uint16_t> hostk(static_cast<size_t>(n));
        for (int64_t row=0;row<rows;++row)
            for (int64_t col=0;col<r.owned;++col) {
                const int64_t g=r.first+col;
                hostq[size_t(row*r.owned+col)]=sample(false,int(row),g);
                hostk[size_t(row*r.owned+col)]=sample(true,int(row),g);
            }
        require(rad_dev_set(r.device)==RAD_OK,"LTX02 bind input device refused");
        require(rad_memcpy_async(q,hostq.data(),n*2,r.ctx->stream())==RAD_OK &&
                rad_memcpy_async(k,hostk.data(),n*2,r.ctx->stream())==RAD_OK &&
                rad_stream_sync(r.ctx->stream())==RAD_OK,
                "LTX02 native Q/K input transfer failed");
    }
    std::array<std::thread,3> threads;
    for (int rank=0;rank<3;++rank)
        threads[rank]=std::thread([&,rank] {
            Rank& r=*ranks[rank];
            active_rank=&r;
            const int bound=r.ctx->bind_thread();
            RadBatch batch{};
            batch.phase=RAD_PHASE_PREFILL;
            batch.n_tok=rows;
            batch.n_seq=1;
            r.status=bound==RAD_OK ? r.ctx->run_step(&batch) : bound;
            if (r.status==RAD_OK && gpu)
                r.status=rad_stream_sync(r.ctx->stream());
            active_rank=nullptr;
        });
    for(auto& t:threads)t.join();
    for(int rank=0;rank<3;++rank) {
        Rank& r=*ranks[rank];
        if (r.status!=RAD_OK)
            std::fprintf(stderr,"rank%d native issue error %d: %s\n",
                         rank,r.status,r.ctx->step_error());
        require(r.status==RAD_OK,"LTX02 native Q/K RadCtx issue failed");
        require(r.ctx->ar_calls()==2,"LTX02 must issue two real collectives per rank");
        require(rad_dev_set(r.device)==RAD_OK,"LTX02 output device bind refused");
        const int64_t nb=rows*r.owned*int64_t(sizeof(uint16_t));
        require(rad_memcpy_async(outq[rank].data(),r.ctx->buf_ptr(r.bq),
                                 nb,r.ctx->stream())==RAD_OK &&
                rad_memcpy_async(outk[rank].data(),r.ctx->buf_ptr(r.bk),
                                 nb,r.ctx->stream())==RAD_OK &&
                rad_stream_sync(r.ctx->stream())==RAD_OK,
                "LTX02 native Q/K result transfer failed");
    }
    int64_t compared=0, mismatch=0;
    for(int64_t row=0;row<rows;++row) {
        for (bool key:{false,true}) {
            float ss=0;
            for(int64_t i=0;i<full;++i) {
                const float v=f32(sample(key,int(row),i));
                ss+=v*v;
            }
            const float scale=1.f/std::sqrt(ss/float(full)+float(eps));
            for(int rank=0;rank<3;++rank) {
                const Rank& r=*ranks[rank];
                for(int64_t col=0;col<r.owned;++col) {
                    const int64_t index=row*r.owned+col;
                    const int64_t global=r.first+col;
                    const uint16_t expected=bf16(
                        f32(sample(key,int(row),global))*scale*
                        f32((key?gain_k:gain_q)[global]));
                    const uint16_t got=(key?outk[rank]:outq[rank])[size_t(index)];
                    if(got!=expected) {
                        if(mismatch<8)
                            std::fprintf(stderr,
                                "LTX02 mismatch %s rank%d row%lld col%lld got=%04x want=%04x\n",
                                key?"K":"Q",rank,(long long)row,(long long)global,
                                got,expected);
                        ++mismatch;
                    }
                    ++compared;
                }
            }
        }
    }
    std::printf("LTX02 native issued C05 %s rows=%lld: %lld/%lld exact BF16, all three ranks two collectives each\n",
        audio?"audio":"video",(long long)rows,
        (long long)(compared-mismatch),(long long)compared);
    require(mismatch==0,"LTX02 issued C05 differs from full-width BF16 oracle");
}
}
int main(int argc,char** argv) {
    if(argc!=3 && argc!=4) {
        std::fprintf(stderr,
            "usage: ltx_c05_issued_host_test MODEL.rad KERNEL_DIR [--aql]\n");
        return 2;
    }
    try {
        const bool gpu=argc==4;
        require(!gpu || std::strcmp(argv[3],"--aql")==0,
                "LTX02 only accepts --aql in native GPU mode");
        require(gpu != rad::device_is_host(),
                "LTX02 backend mismatches explicit native host/AQL mode");
        if (gpu) ltx_test::require_aql_backend();
        rad::RadFile model;
        require(model.open(argv[1])==RAD_OK,"LTX02 checkpoint cannot open");
        const char* heads=model.get_s("transformer.num_attention_heads","");
        require(heads && *heads,"LTX02 checkpoint head metadata missing");
        const int64_t nheads=std::stoll(heads);
        require(nheads>0,"LTX02 missing positive model head count");
        rad::Registry reg;
        if (gpu) ltx_test::configure_aql_registry(reg);
        std::fprintf(stderr,"LTX02 stage: native registry load begin (%s)\n",
                     gpu?"AQL":"host");
        std::fflush(stderr);
        require(reg.load_dir(argv[2],gpu ?
                std::vector<std::string>{"libr3","libr4d","libref"} :
                std::vector<std::string>{"libref"})==RAD_OK,
                "LTX02 native model kernel registry failed");
        std::fprintf(stderr,"LTX02 stage: native registry loaded\n");
        std::fflush(stderr);
        for(bool audio:{false,true})
            for(int64_t rows:{1,4,16,64})
                for(const std::array<int,3> split:
                    {std::array<int,3>{8,8,16},std::array<int,3>{12,12,8}})
                    run_case(model,reg,audio,rows,split,nheads,gpu);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"LTX02 native issued C05: %s\n",e.what());
        return 1;
    }
}
