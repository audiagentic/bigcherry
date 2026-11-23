// Opt-in AQL lab: uneven guidance passes with native block-range execution.
// No production planner, kernel, weight layout or service route changes.
#pragma once
#include <condition_variable>
#include <limits>
#include <mutex>
#include <exception>

struct PinnedDiT {
    std::array<void*,4> host{};
    std::array<size_t,4> bytes{};
    explicit PinnedDiT(const ltx2::Model& m) {
        bytes = {size_t(m.maxS*m.D*2),size_t(m.maxS*m.D*2),
                 size_t(m.maxA*m.Da*2),size_t(m.maxA*m.Da*2)};
        for (size_t i=0;i<host.size();++i) {
            host[i]=rad_dev_alloc((int64_t)bytes[i],RAD_MEM_HOST_PINNED);
            need(host[i]!=nullptr,"native pinned hand-off allocation failed");
        }
    }
    ~PinnedDiT() {
        for (void* p:host) if(p) rad_dev_free(p,RAD_MEM_HOST_PINNED);
    }
    PinnedDiT(const PinnedDiT&)=delete;
    PinnedDiT& operator=(const PinnedDiT&)=delete;
    size_t total_bytes() const {
        size_t n=0; for(auto b:bytes)n+=b; return n;
    }
};

static std::array<rad_buf,4> pass_state(const ltx2::Model& m) {
    return {m.b_hv,m.b_ov,m.b_ha,m.b_oa};
}

// Always device -> pinned host -> device; never enable a peer copy.
static double pass_handoff(PassDevice::Pass& from,PassDevice::Pass& to,
                           PinnedDiT& pinned) {
    const auto a=pass_state(*from.model),b=pass_state(*to.model);
    const auto t0=std::chrono::steady_clock::now();
    need(rad_dev_set(from.device)==RAD_OK,"hand-off source device bind refused");
    need(rad_stream_sync(from.ctx->stream())==RAD_OK,"hand-off source stream not settled");
    for(size_t i=0;i<a.size();++i)
        need(rad_memcpy_async(pinned.host[i],from.ctx->buf_ptr(a[i]),
                              (int64_t)pinned.bytes[i],from.ctx->stream())==RAD_OK,
             "hand-off download failed");
    need(rad_stream_sync(from.ctx->stream())==RAD_OK,"hand-off download sync failed");
    need(rad_dev_set(to.device)==RAD_OK,"hand-off target device bind refused");
    for(size_t i=0;i<b.size();++i)
        need(rad_memcpy_async(to.ctx->buf_ptr(b[i]),pinned.host[i],
                              (int64_t)pinned.bytes[i],to.ctx->stream())==RAD_OK,
             "hand-off upload failed");
    need(rad_stream_sync(to.ctx->stream())==RAD_OK,"hand-off upload sync failed");
    return std::chrono::duration<double,std::milli>(
               std::chrono::steady_clock::now()-t0).count();
}

static double pass_range(PassDevice::Pass& p,int64_t begin,int64_t end) {
    need(begin>=0 && begin<end && end<=p.model->nb,
         "invalid native DiT block range");
    need(rad_dev_set(p.device)==RAD_OK && p.ctx->bind_thread()==RAD_OK,
         "DiT block range thread bind refused");
    p.range_begin=begin; p.range_end=end;
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
    p.range_begin=p.range_end=0;
    if(rc!=RAD_OK || sync!=RAD_OK)
        std::fprintf(stderr,"DiT range [%lld,%lld) failed device=%d rc=%d sync=%d error=%s\n",
                     (long long)begin,(long long)end,p.device,rc,sync,p.ctx->step_error());
    need(rc==RAD_OK && sync==RAD_OK,"native DiT block range failed");
    return std::chrono::duration<double,std::milli>(
               std::chrono::steady_clock::now()-t0).count();
}

static PassReceipt pass_range_final(PassDevice::Pass& p,int64_t begin,int64_t end) {
    need(end==p.model->nb,"range output requires final native projection");
    p.range_begin=begin; p.range_end=end;
    auto result=pass_execute(p);
    p.range_begin=p.range_end=0;
    return result;
}

static size_t pass_diffs(const std::vector<uint16_t>& a,
                         const std::vector<uint16_t>& b) {
    need(a.size()==b.size(),"DiT pass receipt sizes differ");
    size_t n=0;
    for(size_t i=0;i<a.size();++i)n+=a[i]!=b[i];
    return n;
}

static bool pass_equal(const PassReceipt& a,const PassReceipt& b) {
    return a.video==b.video && a.audio==b.audio;
}

static float pass_bf16_float(uint16_t x) {
    const uint32_t bits=uint32_t(x)<<16;
    float f=0;
    std::memcpy(&f,&bits,sizeof f);
    return f;
}
static double pass_rel_rms(const std::vector<uint16_t>& reference,
                           const std::vector<uint16_t>& tested) {
    need(reference.size()==tested.size(),"DiT relative RMS shapes differ");
    double squared=0,base=0;
    for(size_t i=0;i<reference.size();++i) {
        const double a=pass_bf16_float(reference[i]),b=pass_bf16_float(tested[i]);
        need(std::isfinite(a)&&std::isfinite(b),"nonfinite DiT output");
        squared+=(a-b)*(a-b); base+=a*a;
    }
    return std::sqrt(squared/std::max(base,std::numeric_limits<double>::min()));
}

struct PassCutPlan {
    std::array<int64_t,2> cut{};
    std::array<int,2> order{0,1};
    double predicted_ms=std::numeric_limits<double>::infinity();
};

static PassCutPlan pass_choose_cuts(
        int64_t L,double base_ms,
        const std::array<double,2>& xfirst,
        const std::array<double,2>& xblock,
        const std::array<double,2>& xfull,
        const std::array<double,2>& rblock,
        const std::array<double,2>& transfer) {
    need(L>1,"need at least two native DiT blocks");
    PassCutPlan best;
    for(int64_t c0=1;c0<=L;++c0)for(int64_t c1=1;c1<=L;++c1) {
        if(c0==L && c1==L)continue;
        const std::array<int64_t,2> cut{c0,c1};
        std::array<double,2> ready{},work{},front{};
        for(int i=0;i<2;++i) {
            front[i]=cut[i]==L ? xfull[i] : xfirst[i]+(cut[i]-1)*xblock[i];
            ready[i]=front[i]+(cut[i]==L ? 0 : transfer[i]);
            work[i]=(L-cut[i])*rblock[i];
        }
        for(const std::array<int,2>& order:
            {std::array<int,2>{0,1},std::array<int,2>{1,0}}) {
            double finish=base_ms;
            for(int i:order)if(cut[i]<L)
                finish=std::max(finish,ready[i])+work[i];
            const double makespan=std::max({finish,front[0],front[1]});
            if(makespan<best.predicted_ms) {
                best.cut=cut; best.order=order; best.predicted_ms=makespan;
            }
        }
    }
    return best;
}

static int run_full_pass_uneven(const RadFile& model,const std::string& dir,
                                int64_t width,int64_t height,int64_t frames) {
    ltx_test::require_aql_backend(3);
    need(width>0 && height>0 && width%ltx2::kScale==0 &&
         height%ltx2::kScale==0 && frames>=ltx2::kFtail &&
         (frames-ltx2::kFtail)%ltx2::kFpc==0,
         "unsupported LTX uneven lab geometry");
    const int64_t latf=(frames-ltx2::kFtail)/ltx2::kFpc+ltx2::kLtail;
    const int64_t latw=width/ltx2::kScale,lath=height/ltx2::kScale;
    const int64_t S=latf*latw*lath;
    const int64_t A=(frames*ltx2::kAudioRate+ltx2::kFps/2)/ltx2::kFps+1;
    need(S<=ltx2::max_video_rows(),"uneven DiT exceeds native video rows");
    RadVideoGeom geom{};
    geom.n_text=ltx2::kTextRows; geom.frames=(int32_t)frames;
    geom.lat_frames=(int32_t)latf; geom.lat_h=(int32_t)lath;
    geom.lat_w=(int32_t)latw; geom.n_audio=(int32_t)A;

    // Discover the two compatible XTXs and the faster R9700; exclude gfx1030.
    std::array<int,3> card{-1,-1,-1};
    for(int i=0;i<rad_dev_count();++i) {
        RadDeviceProps props{};
        need(rad_dev_props(i,&props)==RAD_OK,"native AQL properties refused");
        if(std::strcmp(props.arch,"gfx1100")==0) {
            if(card[0]<0)card[0]=i;
            else if(card[1]<0)card[1]=i;
        } else if(std::strcmp(props.arch,"gfx1201")==0 && card[2]<0)
            card[2]=i;
    }
    need(std::all_of(card.begin(),card.end(),[](int i){return i>=0;}),
         "uneven DiT requires two gfx1100 and one gfx1201");
    std::printf("UNEVEN_ROSTER xtx_a=%d xtx_b=%d r9700=%d\n",
                card[0],card[1],card[2]);
    std::array<std::unique_ptr<PassDevice>,3> devices{};
    std::array<PassDevice::Pass*,4> rpass{};
    std::array<PassDevice::Pass*,2> xpass{};
    for(int logical=0;logical<3;++logical) {
        devices[logical]=std::make_unique<PassDevice>();
        auto& d=*devices[logical];
        d.device=card[logical];
        need(rad_dev_set(d.device)==RAD_OK,"uneven device bind failed");
        d.reg=std::make_unique<Registry>();
        need(d.reg->set_code_devices({d.device})==RAD_OK,
             "uneven code device selection failed");
        const std::vector<std::string> libraries =
            logical==2 ? std::vector<std::string>{"libr4d","libref"} :
                         std::vector<std::string>{"libr3","libref"};
        need(d.reg->load_dir(dir,libraries)==RAD_OK,
             "uneven native registry load failed");
        d.builder=pass_dit_graph(model,*d.reg,S,A,ltx2::kTextRows+A+S,
                                 latw*lath,d.model,true);
        need(d.builder->program().device_holes(false).empty(),
             "native DiT has unresolved GPU kernels");
        pass_load_weights(model,d);
        for(int v=0;v<4;++v) {
            if(logical!=2 && v!=logical+2)continue;
            auto pass=pass_init(d,v,geom);
            if(logical==2)rpass[v]=pass.get();
            else xpass[logical]=pass.get();
            d.passes.push_back(std::move(pass));
        }
    }
    const int64_t L=devices[2]->model.nb;
    need(L>1,"cannot split single native DiT block");

    // Measured serial reference, including the STG and isolated branches.
    std::array<PassReceipt,4> reference{};
    for(int v=0;v<4;++v) {
        auto& p=*rpass[0]; p.variant=v;
        reference[v]=pass_execute(p);
        std::printf("UNEVEN_REFERENCE variant=%d r9700_ms=%.3f\n",
                    v,reference[v].ms);
    }
    rpass[0]->variant=0;
    std::array<PassReceipt,2> xfull{};
    for(int i=0;i<2;++i) {
        xfull[i]=pass_execute(*xpass[i]);
        std::printf("UNEVEN_XFULL variant=%d ms=%.3f\n",
                    i+2,xfull[i].ms);
    }

    // Obtain actual per-device block timing by executing one native block
    // then the remaining L-1 blocks with final projection, all on the GPU.
    std::array<double,2> xfirst{},xblock{},rblock{},handoff{};
    for(int i=0;i<2;++i) {
        const int v=i+2;
        auto& rp=*rpass[0];rp.variant=v;
        const double rf=pass_range(rp,0,1);
        const auto rr=pass_range_final(rp,1,L);
        need(pass_equal(rr,reference[v]),
             "same-device R9700 segmented blocks changed output");
        rblock[i]=rr.ms/double(L-1);
        xfirst[i]=pass_range(*xpass[i],0,1);
        const auto xr=pass_range_final(*xpass[i],1,L);
        need(pass_equal(xr,xfull[i]),
             "same-device XTX segmented blocks changed output");
        xblock[i]=xr.ms/double(L-1);
        std::printf("UNEVEN_CAL variant=%d r_first_ms=%.3f r_block_ms=%.4f x_first_ms=%.3f x_block_ms=%.4f\n",
                    v,rf,rblock[i],xfirst[i],xblock[i]);
    }
    rpass[0]->variant=0;
    std::array<std::unique_ptr<PinnedDiT>,2> pins{};
    for(int i=0;i<2;++i) {
        pins[i]=std::make_unique<PinnedDiT>(devices[2]->model);
        handoff[i]=pass_handoff(*xpass[i],*rpass[i+2],*pins[i]);
        std::printf("UNEVEN_HANDOFF_CAL variant=%d bytes=%zu ms=%.3f\n",
                    i+2,pins[i]->total_bytes(),handoff[i]);
    }

    const double one_card_ms=reference[0].ms+reference[1].ms+
                             reference[2].ms+reference[3].ms;
    const double base_ms=reference[0].ms+reference[1].ms;
    const auto cuts=pass_choose_cuts(L,base_ms,xfirst,xblock,
                  {xfull[0].ms,xfull[1].ms},rblock,handoff);
    const double capacity_bound_ms=one_card_ms/
        (1.0 + (rblock[0]+rblock[1])/(xblock[0]+xblock[1]) * 2.0);
    std::printf("UNEVEN_CUTS blocks=%lld variant2_xtx_blocks=%lld variant3_xtx_blocks=%lld r_tail_order=%d,%d predicted_ms=%.3f capacity_bound_ms=%.3f\n",
                (long long)L,(long long)cuts.cut[0],(long long)cuts.cut[1],
                cuts.order[0]+2,cuts.order[1]+2,cuts.predicted_ms,capacity_bound_ms);

    // Verify that crossing a context through pinned host preserves *every*
    // BF16 bit on the same GPU at the selected native block boundaries.
    for(int i=0;i<2;++i)if(cuts.cut[i]<L) {
        auto& source=*rpass[0]; source.variant=i+2;
        (void)pass_range(source,0,cuts.cut[i]);
        const double copy_ms=pass_handoff(source,*rpass[i+2],*pins[i]);
        const auto same=pass_range_final(*rpass[i+2],cuts.cut[i],L);
        const bool exact=pass_equal(reference[i+2],same);
        std::printf("UNEVEN_SAME_DEVICE variant=%d cut=%lld handoff_ms=%.3f bit_equal=%d video_diffs=%zu audio_diffs=%zu\n",
                    i+2,(long long)cuts.cut[i],copy_ms,int(exact),
                    pass_diffs(reference[i+2].video,same.video),
                    pass_diffs(reference[i+2].audio,same.audio));
        need(exact,"R9700 same-device pinned hand-off broke DiT exactness");
    }
    rpass[0]->variant=0;

    // Deterministic sequential split receipt, used to detect races below.
    std::array<PassReceipt,4> serial{};
    serial[0]=pass_execute(*rpass[0]);
    serial[1]=pass_execute(*rpass[1]);
    for(int i=0;i<2;++i) {
        if(cuts.cut[i]==L)serial[i+2]=pass_execute(*xpass[i]);
        else {
            (void)pass_range(*xpass[i],0,cuts.cut[i]);
            (void)pass_handoff(*xpass[i],*rpass[i+2],*pins[i]);
            serial[i+2]=pass_range_final(*rpass[i+2],cuts.cut[i],L);
        }
    }

    std::array<PassReceipt,4> parallel{};
    std::array<double,2> actual_handoff{};
    std::array<bool,2> ready{false,false};
    std::array<std::exception_ptr,3> errors{};
    std::mutex mutex;
    std::condition_variable cv;
    const auto t0=std::chrono::steady_clock::now();
    std::array<std::thread,3> workers;
    for(int i=0;i<2;++i)workers[i]=std::thread([&,i]{
        try {
            if(cuts.cut[i]==L)parallel[i+2]=pass_execute(*xpass[i]);
            else {
                (void)pass_range(*xpass[i],0,cuts.cut[i]);
                actual_handoff[i]=pass_handoff(*xpass[i],*rpass[i+2],*pins[i]);
                {std::lock_guard<std::mutex> lock(mutex);ready[i]=true;}
                cv.notify_all();
            }
        }catch(...) {
            {std::lock_guard<std::mutex> lock(mutex);errors[i]=std::current_exception();}
            cv.notify_all();
        }
    });
    workers[2]=std::thread([&]{
        try {
            parallel[0]=pass_execute(*rpass[0]);
            parallel[1]=pass_execute(*rpass[1]);
            for(int i:cuts.order)if(cuts.cut[i]<L) {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock,[&]{return ready[i]||errors[i];});
                if(errors[i])std::rethrow_exception(errors[i]);
                lock.unlock();
                parallel[i+2]=pass_range_final(*rpass[i+2],cuts.cut[i],L);
            }
        }catch(...) {
            {std::lock_guard<std::mutex> lock(mutex);errors[2]=std::current_exception();}
            cv.notify_all();
        }
    });
    for(auto& t:workers)t.join();
    for(const auto& e:errors)if(e)std::rethrow_exception(e);
    const double elapsed=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-t0).count();
    for(int v=0;v<4;++v) {
        const bool exact=pass_equal(serial[v],parallel[v]);
        std::printf("UNEVEN_PARALLEL variant=%d same_device_serial_bit_equal=%d r9700_video_diffs=%zu r9700_audio_diffs=%zu video_rel_rms=%.6f audio_rel_rms=%.6f\n",
            v,int(exact),
            pass_diffs(reference[v].video,parallel[v].video),
            pass_diffs(reference[v].audio,parallel[v].audio),
            pass_rel_rms(reference[v].video,parallel[v].video),
            pass_rel_rms(reference[v].audio,parallel[v].audio));
        need(exact,"parallel uneven placement changed same-device split output");
    }
    std::printf("UNEVEN_RESULT geom=%lldx%lld frames=%lld one_card_ms=%.3f baseline_2_1_1_ms=%.3f uneven_makespan_ms=%.3f improvement_vs_baseline=%.4f one_card_speedup=%.4f capacity_bound_ms=%.3f handoff2_ms=%.3f handoff3_ms=%.3f\n",
        (long long)width,(long long)height,(long long)frames,
        one_card_ms,std::max({base_ms,xfull[0].ms,xfull[1].ms}),
        elapsed,std::max({base_ms,xfull[0].ms,xfull[1].ms})/elapsed,
        one_card_ms/elapsed,capacity_bound_ms,actual_handoff[0],actual_handoff[1]);
    std::fflush(stdout);
    return 0;
}
