// Residency diagnostic for independent LTX guidance-pass placement.
// Uses the exact native DiT builder (weights, operations and activation liveness)
// and uploads every checkpoint DiT weight plane to one AQL device.
#pragma once

static int64_t pass_model_i(const RadFile& f, const char* key) {
    int64_t n=0;
    if (f.get_i(key,&n)) return n;
    const char* s=f.get_s(key,"");
    need(s && *s, std::string("missing checkpoint metadata ")+key);
    return std::stoll(s);
}

static std::unique_ptr<Builder> pass_dit_graph(
        const RadFile& model, Registry& reg, int64_t video, int64_t audio,
        int64_t rows, int64_t ppf, ltx2::Model& m, bool instances) {
    m.nm=ltx2::Names("");
    m.nb=pass_model_i(model,"transformer.num_layers");
    m.H=pass_model_i(model,"transformer.num_attention_heads");
    m.hd=pass_model_i(model,"transformer.attention_head_dim");
    m.Ha=pass_model_i(model,"transformer.audio_num_attention_heads");
    m.hda=pass_model_i(model,"transformer.audio_attention_head_dim");
    m.D=m.H*m.hd; m.Da=m.Ha*m.hda; m.Dc=m.Da;
    m.ffv=4*m.D; m.ffa=4*m.Da;
    m.maxS=video; m.maxA=audio; m.maxPpf=ppf;
    const auto* qkv=model.find("dit.blk.0.attn1.qkv.weight");
    need(qkv!=nullptr, "native DiT checkpoint QKV missing");
    const auto& enc=model.encoding(*qkv);
    need(rad::arch::linear_is_iq8r(enc), "requires native IQ8R DiT");
    m.gd.w_i8h=true;
    m.gd.w_i8_group=rad::arch::linear_w8h_group(enc);
    m.gd.dtype="bf16"; m.gd.act_dtype=RAD_BF16;
    m.gd.rank=0; m.gd.world=1; m.gd.max_tok=rows;
    m.gd.max_seqs=1; m.gd.max_ctx=rows;
    m.gd.n_layer=m.nb;
    RadBuildCtx bc=context(0,1,rows);
    bc.max_ctx=rows;
    auto b=std::make_unique<Builder>(reg,metadata(),bc);
    b->set_sources(rad::rad_container_sources(model));
    // These two connector outputs are external to the DiT, but the
    // native DiT declares all of its own video, audio and text activations.
    m.cv.b_text=ltx2::decl_b(b.get(),m.nm.f("ltx.pass.text_v"),
                            RAD_BF16,{2*ltx2::kTextRows,m.D},RAD_BUF_PERSIST);
    m.ca.b_text=ltx2::decl_b(b.get(),m.nm.f("ltx.pass.text_a"),
                            RAD_BF16,{2*ltx2::kTextRows,m.Da},RAD_BUF_PERSIST);
    m.b_rows=ltx2::decl_b(b.get(),m.nm.f("ltx.pass.text_cu"),RAD_I32,{3},RAD_BUF_PERSIST);
    need(m.cv.b_text && m.ca.b_text && m.b_rows, "connector activation declare refused");
    const int rc=ltx2::declare_dit(b.get(),nullptr,m);
    if(rc!=RAD_OK)
        for(const auto& err:b->errors()) std::fprintf(stderr,"DiT declaration: %s\n",err.c_str());
    need(rc==RAD_OK, "native full DiT declaration refused");
    need((int64_t)m.dit.size()==m.nb, "not all native video+audio blocks declared");
    b->set_instances(instances);
    const int finished=b->finish();
    if(finished!=RAD_OK)
        for(const auto& err:b->errors()) std::fprintf(stderr,"DiT build: %s\n",err.c_str());
    need(finished==RAD_OK && b->errors().empty(),"native DiT arena planning failed");
    return b;
}

static int run_pass_residency(const RadFile& model, Registry& reg,
                              int64_t width, int64_t height, int64_t frames) {
    ltx_test::require_aql_backend(1);
    need(width>0 && height>0 && frames>0 &&
         width%ltx2::kScale==0 && height%ltx2::kScale==0 &&
         frames>=ltx2::kFtail &&
         (frames-ltx2::kFtail)%ltx2::kFpc==0, "invalid native LTX video geometry");
    const int64_t latf = (frames-ltx2::kFtail)/ltx2::kFpc*ltx2::kLpc + ltx2::kLtail;
    const int64_t latw=width/ltx2::kScale, lath=height/ltx2::kScale;
    const int64_t video=latf*latw*lath;
    const int64_t audio=(frames*ltx2::kAudioRate + ltx2::kFps/2)/ltx2::kFps + 1;
    const int64_t rows=ltx2::kTextRows+audio+video;
    need(video<=ltx2::max_video_rows(), "exceeds native DiT's video-row ceiling");
    need(rad_dev_set(0)==RAD_OK, "cannot bind visible AQL device");
    RadDeviceProps before{};
    need(rad_dev_props(0,&before)==RAD_OK && before.vram_free>0,
         "AQL device free-VRAM query failed");

    ltx2::Model m{};
    auto builder=pass_dit_graph(model,reg,video,audio,rows,latw*lath,m,false);
    const Builder& b=*builder;
    const int64_t arena=b.program().arena_bytes;
    int64_t stored=0;
    for(size_t i=1;i<b.program().weights.size();++i)
        stored+=b.program().weights[i].stored_bytes;

    std::vector<const RadFilePlane*> planes;
    int64_t raw_bytes=0, block_entries=0;
    std::vector<int> blocks((size_t)m.nb,0);
    for(int64_t i=0;i<model.entry_count();++i) {
        const auto& e=model.entry(i);
        const std::string name=model.str(e.name);
        if(name.rfind("dit.",0)!=0) continue;
        if(name.rfind("dit.blk.",0)==0) {
            const char* start=name.c_str()+8;
            char* end=nullptr;
            const long idx=std::strtol(start,&end,10);
            need(end!=start && end && *end=='.' && idx>=0 && idx<m.nb,
                 "unexpected native DiT block index");
            ++blocks[(size_t)idx];
            ++block_entries;
        }
        for(uint32_t p=0;p<e.n_planes;++p) {
            const auto& plane=model.plane(e,p);
            raw_bytes+=(int64_t)plane.bytes;
            planes.push_back(&plane);
        }
    }
    need(std::all_of(blocks.begin(),blocks.end(),[](int n){return n>0;}),
         "one or more video+audio DiT blocks have no checkpoint weights");
    std::printf("PASS_RESIDENCY geom=%lldx%lld frames=%lld lat=%lldx%lldx%lld video=%lld audio=%lld text=%d step_rows=%lld blocks=%lld block_weights=%lld planes=%zu\n",
        (long long)width,(long long)height,(long long)frames,
        (long long)latf,(long long)lath,(long long)latw,(long long)video,
        (long long)audio,ltx2::kTextRows,(long long)rows,(long long)m.nb,
        (long long)block_entries,planes.size());
    std::printf("PASS_RESIDENCY plan raw_weight_mib=%.2f native_stored_mib=%.2f arena_mib=%.2f before_free_mib=%.2f device_total_mib=%.2f\n",
        raw_bytes/1048576.,stored/1048576.,arena/1048576.,
        before.vram_free/1048576.,before.vram_bytes/1048576.);
    std::fflush(stdout);

    // Accounting uses the source planes actually uploaded, not the padded
    // file-entry spans. Native stored_bytes is emitted for relayout comparison.
    // The native layout may pad or expand source planes. Reserve its entire
    // stored-byte difference as device memory as well as each original plane.
    const int64_t layout_extra=std::max<int64_t>(0,stored-raw_bytes);
    const bool budget_fits=raw_bytes+layout_extra+arena<=before.vram_free;
    if(!budget_fits) {
        std::printf("PASS_RESIDENCY fit=0 reason=budget headroom_mib=%.2f measured=0\n",
             (before.vram_free-raw_bytes-layout_extra-arena)/1048576.);
        return 0;
    }
    struct Allocation {
        std::vector<void*> weights;
        void* arena=nullptr;
        RadStream stream=nullptr;
        ~Allocation(){
            if(stream) { (void)rad_stream_sync(stream); rad_stream_destroy(stream); }
            if(arena) rad_dev_free(arena,RAD_MEM_DEVICE);
            for(auto p:weights) rad_dev_free(p,RAD_MEM_DEVICE);
        }
    } memory;
    need(rad_stream_create(&memory.stream,0)==RAD_OK, "AQL upload stream refused");
    bool fit=true;
    for(const auto* p:planes) {
        void* d=rad_dev_alloc((int64_t)p->bytes,RAD_MEM_DEVICE);
        if(!d){fit=false;break;}
        memory.weights.push_back(d);
        if(rad_memcpy_async(d,model.data(*p),(int64_t)p->bytes,memory.stream)!=RAD_OK) {
            fit=false;break;
        }
    }
    if(fit && layout_extra>0) {
        void* extra=rad_dev_alloc(layout_extra,RAD_MEM_DEVICE);
        fit=extra!=nullptr;
        if(extra) memory.weights.push_back(extra);
    }
    if(fit) {
        memory.arena=rad_dev_alloc(arena,RAD_MEM_DEVICE);
        fit=memory.arena!=nullptr;
    }
    if(fit) fit=rad_stream_sync(memory.stream)==RAD_OK;
    RadDeviceProps after{};
    need(rad_dev_props(0,&after)==RAD_OK, "AQL post-allocation VRAM query failed");
    std::printf("PASS_RESIDENCY fit=%d measured=1 after_free_mib=%.2f headroom_mib=%.2f used_delta_mib=%.2f uploaded_planes=%zu/%zu\n",
        int(fit),after.vram_free/1048576.,after.vram_free/1048576.,
        (before.vram_free-after.vram_free)/1048576.,
        memory.weights.size() - (layout_extra>0 && fit ? 1 : 0),planes.size());
    std::fflush(stdout);
    return fit?0:1;
}
