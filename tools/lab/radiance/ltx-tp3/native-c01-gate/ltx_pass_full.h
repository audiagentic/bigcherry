// Lab-only native DiT load, context and receipt helpers for uneven placement.
// No second attention/FFN engine and no production routing change.
#pragma once

struct PassDevice {
    int device=0;
    std::unique_ptr<Registry> reg;
    ltx2::Model model{};
    std::unique_ptr<Builder> builder;
    std::vector<void*> weights;
    struct Pass {
        int device=0, variant=0;
        int64_t range_begin=0, range_end=0;
        ltx2::Model* model=nullptr;
        std::unique_ptr<rad::Ctx> ctx;
        std::vector<void*> extra;
        RadVideoGenBatch gen{};
        ~Pass() {
            ctx.reset();
            (void)rad_dev_set(device);
            for (void* p:extra) rad_dev_free(p,RAD_MEM_DEVICE);
        }
    };
    std::vector<std::unique_ptr<Pass>> passes;
    ~PassDevice() {
        passes.clear();
        (void)rad_dev_set(device);
        for(void* p:weights) rad_dev_free(p,RAD_MEM_DEVICE);
        builder.reset();
        reg.reset();
    }
};

thread_local PassDevice::Pass* pass_active=nullptr;
static void pass_dit_step(RadCtx* c, const RadBatch*) {
    const auto* p=pass_active;
    if(!p) {
        rad_step_fail(c,"LTX pass placement thread is unbound");
        return;
    }
    const int branch=p->variant==1 ? 1:0;
    const bool stg=p->variant==2, iso=p->variant==3;
    if(p->range_end>0)
        ltx2::pass_dit_eval_range(c,&p->gen,*p->model,0,branch,stg,iso,
                            p->range_begin,p->range_end);
    else
        ltx2::dit_eval(c,&p->gen,*p->model,0,branch,stg,iso);
}

// The same selected plane gather and native relayout contract exercised by
// verify_c06_load_contract; keep the engine's kernel hook as the only layout.
static std::vector<uint8_t> pass_native_weight(const RadFile& f,
                                                 rad::Program& p,
                                                 const rad::WeightInfo& w) {
    const auto* e=f.find(w.source);
    need(e && w.enc_known && w.n_sel>0,"full pass missing source weight");
    rad::PlaneShare shares[RAD_ENC_MAX_PLANES];
    std::string why;
    need(rad::weight_share(w,f.encoding(*e),e->rank,e->shape,1,0,shares,&why)==RAD_OK,
         "full pass native plane share refused: "+why);
    if(w.identity) {
        need(w.n_sel==1,"full pass identity has multiple selected planes");
        auto result=c06_gather(f,*e,w.sel[0],shares[0]);
        need(result.size()==(size_t)w.stored_bytes,"native identity size mismatch");
        return result;
    }
    need(!w.widen && !w.narrow &&
         w.lay_op>0 && size_t(w.lay_op)<p.ops.size(),
         "full pass cannot use native relayout");
    const auto& oi=p.ops[(size_t)w.lay_op];
    need(w.lay_band>=0 && size_t(w.lay_band)<oi.bands.size() &&
         w.lay_dom>=0 && w.lay_dom<RAD_N_DOMAINS,
         "full pass native relayout band out of range");
    const auto& resolved=oi.bands[(size_t)w.lay_band].dom[w.lay_dom];
    need(resolved && resolved.row && resolved.row->info &&
         resolved.row->info->relayout,"full pass native relayout hook missing");
    std::array<RadTensor,RAD_ENC_MAX_PLANES> source{};
    rad::weight_planes(w,source.data());
    std::vector<std::vector<uint8_t>> src((size_t)w.n_sel);
    for(int k=0;k<w.n_sel;++k) {
        src[(size_t)k]=c06_gather(f,*e,w.sel[k],shares[k]);
        source[(size_t)k].data=src[(size_t)k].data();
    }
    std::vector<uint8_t> out((size_t)w.stored_bytes);
    const auto* ki=resolved.row->info;
    need(ki->relayout(resolved.geom.params(),resolved.geom.n_params(),
                      w.lay_operand,&w.enc,w.sel,source.data(),w.n_sel,
                      out.data(),w.stored_bytes)==RAD_OK,
         "native full pass kernel weight relayout refused");
    return out;
}

static void pass_load_weights(const RadFile& f, PassDevice& d) {
    need(rad_dev_set(d.device)==RAD_OK,"full pass device bind failed");
    rad::Program& prog=d.builder->program();
    d.weights.reserve(prog.weights.size());
    RadStream stream=nullptr;
    need(rad_stream_create(&stream,0)==RAD_OK,"full pass copy stream refused");
    int64_t bytes=0;
    for(size_t i=1;i<prog.weights.size();++i) {
        auto& w=prog.weights[i];
        auto data=pass_native_weight(f,prog,w);
        void* ptr=rad_dev_alloc(w.stored_bytes,RAD_MEM_DEVICE);
        need(ptr!=nullptr,"full pass native weight allocation refused");
        d.weights.push_back(ptr);
        w.ptr=ptr;
        w.site=rad::Site::Device;
        need(rad_memcpy_async(ptr,data.data(),w.stored_bytes,stream)==RAD_OK &&
             rad_stream_sync(stream)==RAD_OK,"full pass native weight copy refused");
        bytes+=w.stored_bytes;
    }
    rad_stream_destroy(stream);
    std::printf("PASS_FULL_LOAD device=%d weights=%zu stored_mib=%.2f\n",
        d.device,prog.weights.size()-1,bytes/1048576.);
    std::fflush(stdout);
}

static void pass_copy(PassDevice::Pass& p, rad_buf b, const void* src, size_t bytes) {
    need(b && src,"full pass initializer omitted buffer");
    void* dst=p.ctx->buf_ptr(b);
    need(dst && rad_memcpy_async(dst,src,(int64_t)bytes,p.ctx->stream())==RAD_OK,
         "full pass activation upload refused");
}

static void pass_seed_bf16(PassDevice::Pass& p, rad_buf b, size_t n, int64_t salt) {
    std::vector<uint16_t> tmp(n);
    for(size_t i=0;i<n;++i) tmp[i]=c06_sample((int64_t)i+salt);
    pass_copy(p,b,tmp.data(),n*sizeof(uint16_t));
    need(rad_stream_sync(p.ctx->stream())==RAD_OK,"full pass BF16 seed sync failed");
}

static void pass_seed_f32(PassDevice::Pass& p, rad_buf b, size_t n, int64_t salt) {
    std::vector<float> tmp(n);
    for(size_t i=0;i<n;++i) tmp[i]=float(((int64_t)i+salt)*7%61-30)/128.f;
    pass_copy(p,b,tmp.data(),n*sizeof(float));
    need(rad_stream_sync(p.ctx->stream())==RAD_OK,"full pass F32 seed sync failed");
}

static void* pass_external(PassDevice::Pass& p, const void* src, size_t bytes) {
    need(rad_dev_set(p.device)==RAD_OK,"full pass external allocator device bind failed");
    void* dst=rad_dev_alloc((int64_t)bytes,RAD_MEM_DEVICE);
    need(dst!=nullptr,"full pass device external allocation refused");
    p.extra.push_back(dst);
    need(rad_memcpy_async(dst,src,(int64_t)bytes,p.ctx->stream())==RAD_OK &&
         rad_stream_sync(p.ctx->stream())==RAD_OK,
         "full pass external device upload refused");
    return dst;
}

static std::unique_ptr<PassDevice::Pass> pass_init(
        PassDevice& d,int variant,const RadVideoGeom& geom) {
    need(rad_dev_set(d.device)==RAD_OK,"full pass init device bind failed");
    auto p=std::make_unique<PassDevice::Pass>();
    p->device=d.device; p->variant=variant; p->model=&d.model;
    p->ctx=std::make_unique<rad::Ctx>();
    rad::CtxDesc desc{};
    desc.program=&d.builder->program();
    desc.rank=0; desc.world_size=1; desc.device=d.device;
    desc.slot=variant;
    desc.arch_step=pass_dit_step; desc.registry=d.reg.get();
    desc.record_passes=false;
    need(p->ctx->init(desc)==RAD_OK,"full pass native context init failed");
    p->gen.geom=geom; p->gen.lane=0; p->gen.has_image=0;
    const auto& m=d.model;
    const size_t S=(size_t)m.maxS,A=(size_t)m.maxA,R=ltx2::kTextRows;
    const size_t D=(size_t)m.D,Da=(size_t)m.Da,L=(size_t)m.nb;
    const int32_t cu[3]={0,ltx2::kTextRows,2*ltx2::kTextRows};
    pass_copy(*p,m.b_rows,cu,sizeof cu);
    pass_seed_bf16(*p,m.cv.b_text,2*R*D,701);
    pass_seed_bf16(*p,m.ca.b_text,2*R*Da,703);
    pass_seed_f32(*p,m.b_vstate,2*S*ltx2::kC,17);
    pass_seed_f32(*p,m.b_astate,2*A*ltx2::kAudioC,19);
    for(const auto& item : std::initializer_list<std::tuple<rad_buf,size_t,int64_t>>{
         {m.b_mv,2*L*9*D,31},{m.b_ma,L*9*Da,37},
         {m.b_mp,L*2*D,41},{m.b_mpa,L*2*Da,43},
         {m.b_mcv,L*4*D,47},{m.b_mcvg,L*D,53},
         {m.b_mca,L*4*Da,59},{m.b_mcag,L*Da,61},
         {m.b_fv,4*D,67},{m.b_fa,2*Da,71}}) {
        pass_seed_bf16(*p,std::get<0>(item),std::get<1>(item),std::get<2>(item));
    }
    std::vector<float> pos(3*S+A);
    need(ltx2::positions(&geom,pos.data(),(int64_t)(R+A+S))==(int64_t)(R+A+S),
         "full pass native position geometry refused");
    p->gen.pos=(const float*)pass_external(*p,pos.data(),pos.size()*sizeof(float));
    const int32_t lat_cu[4]={0,(int32_t)S,0,(int32_t)A};
    p->gen.lat_cu=(const int32_t*)pass_external(*p,lat_cu,sizeof lat_cu);
    need(rad_stream_sync(p->ctx->stream())==RAD_OK,"full pass initial seed sync failed");
    return p;
}

struct PassReceipt { std::vector<uint16_t> video,audio; double ms=0; };
static PassReceipt pass_execute(PassDevice::Pass& p) {
    need(rad_dev_set(p.device)==RAD_OK && p.ctx->bind_thread()==RAD_OK,
         "full pass context bind failed");
    pass_active=&p;
    RadBatch batch{};
    batch.phase=RAD_PHASE_PREFILL;
    batch.n_tok=p.gen.geom.n_text+p.gen.geom.n_audio+
                p.gen.geom.lat_frames*p.gen.geom.lat_h*p.gen.geom.lat_w;
    batch.n_seq=1;
    const auto t0=std::chrono::steady_clock::now();
    const int rc=p.ctx->run_step(&batch);
    const int sync=rc==RAD_OK ? rad_stream_sync(p.ctx->stream()):rc;
    pass_active=nullptr;
    if(sync!=RAD_OK)
        std::fprintf(stderr,"full pass variant %d on device %d: rc=%d sync=%d error=%s\n",
             p.variant,p.device,rc,sync,p.ctx->step_error());
    need(rc==RAD_OK && sync==RAD_OK,"full pass DiT issue failed");
    PassReceipt ret;
    ret.ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-t0).count();
    const size_t S=(size_t)p.model->maxS,A=(size_t)p.model->maxA;
    ret.video.resize(S*ltx2::kC);
    ret.audio.resize(A*ltx2::kAudioC);
    need(rad_memcpy_async(ret.video.data(),p.ctx->buf_ptr(p.model->b_vel_v[0]),
                          (int64_t)ret.video.size()*2,p.ctx->stream())==RAD_OK &&
         rad_memcpy_async(ret.audio.data(),p.ctx->buf_ptr(p.model->b_vel_a[0]),
                          (int64_t)ret.audio.size()*2,p.ctx->stream())==RAD_OK &&
         rad_stream_sync(p.ctx->stream())==RAD_OK,
         "full pass output copy refused");
    return ret;
}
