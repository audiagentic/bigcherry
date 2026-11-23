// Lab-only real-model native DiT builder, shared by range-placement execution.
// No alternate DiT graph or weight layout.
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
