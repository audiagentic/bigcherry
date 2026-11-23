// Experimental independent guidance-pass placement using the *native*
// LTX2 DiT declaration and issue path, without a second attention/FFN engine.
// This is a lab entrypoint: no production scheduling or model routing is changed.
#pragma once

struct PassDevice {
    int device=0;
    std::unique_ptr<Registry> reg;
    ltx2::Model model{};
    std::unique_ptr<Builder> builder;
    std::vector<void*> weights;
    struct Pass {
        int device=0, variant=0;
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

static int run_full_pass_placement(const RadFile& model, const std::string& dir,
                                   int64_t width,int64_t height,int64_t frames) {
    ltx_test::require_aql_backend(3);
    need(width%ltx2::kScale==0 && height%ltx2::kScale==0 &&
         frames>=ltx2::kFtail &&
         (frames-ltx2::kFtail)%ltx2::kFpc==0,
         "full pass geometry not supported");
    const int64_t latf=(frames-ltx2::kFtail)/ltx2::kFpc+ltx2::kLtail;
    const int64_t lath=height/ltx2::kScale,latw=width/ltx2::kScale;
    const int64_t S=latf*lath*latw;
    const int64_t A=(frames*ltx2::kAudioRate+ltx2::kFps/2)/ltx2::kFps+1;
    need(S>0 && S<=ltx2::max_video_rows(),"full pass exceeds native video rows");
    RadVideoGeom geom{};
    geom.n_text=ltx2::kTextRows; geom.frames=(int32_t)frames;
    geom.lat_frames=(int32_t)latf; geom.lat_h=(int32_t)lath;
    geom.lat_w=(int32_t)latw; geom.n_audio=(int32_t)A;
    // Discover the larger-memory card and the two smaller cards at runtime,
    // rather than tying guidance ownership to enumeration order.
    std::array<RadDeviceProps,3> cards{};
    for(int dev=0;dev<3;++dev)
        need(rad_dev_props(dev,&cards[(size_t)dev])==RAD_OK,
             "full pass device properties unavailable");
    int larger=0;
    for(int dev=1;dev<3;++dev)
        if(cards[(size_t)dev].vram_bytes>cards[(size_t)larger].vram_bytes)
            larger=dev;
    std::array<int,2> smaller{};
    int pos=0;
    for(int dev=0;dev<3;++dev)
        if(dev!=larger) smaller[(size_t)pos++]=dev;
    need(pos==2 &&
         cards[(size_t)larger].vram_bytes>cards[(size_t)smaller[0]].vram_bytes &&
         cards[(size_t)larger].vram_bytes>cards[(size_t)smaller[1]].vram_bytes,
         "full pass needs one larger and two smaller discrete cards");
    const std::array<int,4> assignment{larger,larger,smaller[0],smaller[1]};
    std::array<std::unique_ptr<PassDevice>,3> devices;
    std::array<PassDevice::Pass*,4> mapped{};
    for(int dev=0;dev<3;++dev) {
        devices[dev]=std::make_unique<PassDevice>();
        auto& d=*devices[dev];
        d.device=dev;
        need(rad_dev_set(dev)==RAD_OK,"full pass device roster refused");
        d.reg=std::make_unique<Registry>();
        need(d.reg->set_code_devices({dev})==RAD_OK,
             "full pass local code-device selection refused");
        const std::string arch=cards[(size_t)dev].arch;
        const std::string native_library=
            arch.rfind("gfx12",0)==0 ? "libr4d" :
            arch.rfind("gfx11",0)==0 ? "libr3" : "";
        need(!native_library.empty(),"full pass lacks a native device kernel build");
        const std::vector<std::string> libraries{native_library,"libref"};
        need(d.reg->load_dir(dir,libraries)==RAD_OK,
             "full pass kernel registry load refused");
        d.builder=pass_dit_graph(model,*d.reg,S,A,ltx2::kTextRows+A+S,
                                 latw*lath,d.model,true);
        need(d.builder->program().device_holes(false).empty(),
             "full pass native DiT has unresolved device kernels");
        pass_load_weights(model,d);
        for(int variant=0;variant<4;++variant)
            if(assignment[variant]==dev) {
                auto pass=pass_init(d,variant,geom);
                mapped[variant]=pass.get();
                d.passes.push_back(std::move(pass));
            }
    }
    // Baseline: all four guidance variants on one R9700, sequentially.
    // Keep a separate placed-serial receipt to distinguish cross-device
    // arithmetic differences from races in the concurrent placement.
    std::array<PassReceipt,4> one_card{},serial{},parallel{};
    auto& reference=*mapped[0];
    for(int v=0;v<4;++v) {
        reference.variant=v;
        one_card[v]=pass_execute(reference);
        std::printf("PASS_FULL_ONE_CARD variant=%d ms=%.3f\n",v,one_card[v].ms);
        std::fflush(stdout);
    }
    reference.variant=0;
    for(int v=0;v<4;++v) {
        serial[v]=pass_execute(*mapped[v]);
        std::printf("PASS_FULL_SERIAL variant=%d device=%d ms=%.3f\n",
                    v,assignment[v],serial[v].ms);
        std::fflush(stdout);
    }
    std::array<std::string,3> errors{};
    const auto t0=std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    for(int dev=0;dev<3;++dev) {
        workers.emplace_back([&,dev]{
            try {
                for(int v=0;v<4;++v)
                    if(assignment[v]==dev) parallel[v]=pass_execute(*mapped[v]);
            } catch(const std::exception& e) {errors[dev]=e.what();}
        });
    }
    for(auto& w:workers) w.join();
    const double elapsed=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-t0).count();
    for(const auto& e:errors) need(e.empty(),"parallel full pass: "+e);
    double one_card_ms=0, placed_serial_ms=0;
    int cross_device_mismatches=0;
    for(int v=0;v<4;++v) {
        const bool exact=parallel[v].video==serial[v].video &&
                         parallel[v].audio==serial[v].audio;
        const auto count_diff=[](const auto& lhs,const auto& rhs) {
            need(lhs.size()==rhs.size(),"full pass output shapes differ");
            size_t n=0;
            for(size_t i=0;i<lhs.size();++i) n+=(lhs[i]!=rhs[i]);
            return n;
        };
        const size_t vdiff=count_diff(one_card[v].video,parallel[v].video);
        const size_t adiff=count_diff(one_card[v].audio,parallel[v].audio);
        cross_device_mismatches+=(vdiff!=0 || adiff!=0);
        const auto nonfinite=[](const auto& data) {
            return std::count_if(data.begin(),data.end(),[](uint16_t x) {
                return (x & 0x7f80U)==0x7f80U;
            });
        };
        std::printf("PASS_FULL_FINITE variant=%d ref_video_bad=%zu placed_video_bad=%zu ref_audio_bad=%zu placed_audio_bad=%zu\n",
            v,nonfinite(one_card[v].video),nonfinite(parallel[v].video),
            nonfinite(one_card[v].audio),nonfinite(parallel[v].audio));
        std::printf("PASS_FULL_PARALLEL variant=%d device=%d ms=%.3f within_device_bit_equal=%d one_card_video_diffs=%zu/%zu one_card_audio_diffs=%zu/%zu\n",
                    v,assignment[v],parallel[v].ms,int(exact),
                    vdiff,parallel[v].video.size(),adiff,parallel[v].audio.size());
        need(exact,"parallel full DiT differs from serial on the same card");
        one_card_ms+=one_card[v].ms;
        placed_serial_ms+=serial[v].ms;
    }
    std::printf("PASS_FULL_RESULT passes=4 one_card_ms=%.3f placed_serial_ms=%.3f parallel_makespan_ms=%.3f one_card_speedup=%.3f placed_serial_speedup=%.3f cross_device_bit_equal=%d\n",
                one_card_ms,placed_serial_ms,elapsed,one_card_ms/elapsed,
                placed_serial_ms/elapsed,int(cross_device_mismatches==0));
    std::fflush(stdout);
    need(cross_device_mismatches==0,"cross-architecture full DiT output differs bit-for-bit");
    return 0;
}
