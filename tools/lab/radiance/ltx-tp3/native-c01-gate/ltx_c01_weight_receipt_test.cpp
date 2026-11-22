/* Diagnostic CPU-only receipt of actual RadBuilder weight-view and shard APIs.
 * Uses the official 2.2.0 core, ABI and IQ8R encoding from a real, read-only RAD2
 * model; no plugin execution, kernel loading, GPU initialization or output claim.
 */
#include "build/rad_build.h"
#include "build/startup.h"
#include "format/radfile.h"
#include <arch/rad_fp8.h>
#include "bc_ltx_tp3_c01.h"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using rad::Builder;
using rad::Program;
using rad::RadFile;
using rad::Registry;
using rad::WeightInfo;
using rad::arch::LinearFP8;
using ltx2::C01Slice;

namespace {
void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}
const RadFileEntry& entry(const RadFile& f, const std::string& name) {
    const auto* e = f.find(name);
    if (!e) throw std::runtime_error("model missing "+name);
    require(e->rank == 2, "expected rank-2 matrix");
    return *e;
}
RadWeightDecl weight_decl(uint32_t dtype, int64_t n, int64_t k, int shard,
                          const char* source, const char* planes, int banks = 0,
                          int64_t width = 0) {
    RadWeightDecl d{};
    d.dtype = dtype;
    d.rank = 2;
    d.shape[0] = n;
    d.shape[1] = k;
    d.shard = shard;
    d.access = RAD_ACCESS_PER_TOKEN;
    d.group = rad_group_layer(0);
    d.source = source;
    d.planes = planes;
    d.n_row_parts = banks;
    for (int i = 0; i < banks; ++i) d.row_parts[i] = width;
    return d;
}
void check_span(const WeightInfo& w, int64_t lo, int64_t hi) {
    require(w.shard_lo == lo && w.shard_hi == hi, "column span receipt differs");
}
void check_parts(const WeightInfo& w, int64_t start, int64_t width,
                 int64_t full, int banks) {
    require(w.n_part_spans == banks, "wrong bank count");
    for (int i = 0; i < banks; ++i) {
        require(w.part_lo[i] == start && w.part_hi[i] == start + width &&
                w.part_total[i] == full, "wrong native bank span");
    }
}
RadBuildCtx context(int rank) {
    RadBuildCtx c{};
    c.rank = rank;
    c.world_size = 3;
    c.max_tok = 128;
    c.max_seqs = 1;
    c.max_ctx = 4096;
    c.scope = "";
    return c;
}

int64_t model_i64(const RadFile& f, const char* name) {
    int64_t i = 0;
    if (f.get_i(name,&i)) return i;
    const char* text = f.get_s(name);
    if (!text || !*text) throw std::runtime_error(std::string("missing metadata: ")+name);
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (!end || end==text || *end || errno == ERANGE) throw std::runtime_error(std::string("noninteger metadata: ")+name);
    return value;
}

RadModelMeta metadata() {
    RadModelMeta m{};
    m.arch_id = "ltx2";
    m.name = "ltx-c01-builder-receipt";
    return m;
}
void row_view(const RadFile& f, Registry& reg, const std::string& name,
              int hd, int banks, const std::array<int,3>& heads, int rank) {
    const auto& e = entry(f, name);
    const RadEncoding& enc = f.encoding(e);
    require(rad::arch::linear_is_iq8r(enc), "source is not actual IQ8R");
    const auto* sc = rad_enc_plane(&enc, "scale");
    require(sc && sc->block[1] > 0, "IQ8R group unavailable");
    const int64_t full = e.shape[0] / banks;
    require(full == hd * (heads[0] + heads[1] + heads[2]) &&
            e.shape[0] == banks * full, "unexpected head count");
    const int64_t start = hd * (rank ? heads[0] + (rank > 1 ? heads[1] : 0) : 0);
    const int64_t width = hd * heads[rank];
    const C01Slice slice = C01Slice::rows(full, start, width, hd, banks);
    require(slice.valid(banks * width, e.shape[1], sc->block[1]), "C01 row admission failed");
    require(e.shape[1] % sc->block[1] == 0, "IQ8R K not scale aligned");

    Builder b(reg, metadata(), context(rank));
    b.set_sources(rad::rad_container_sources(f));
    const auto weight = weight_decl(RAD_I8, banks * width, e.shape[1], RAD_SHARD_ROW,
                                    name.c_str(), "codes", banks, width);
    const auto scale = weight_decl(RAD_U8, banks * width, e.shape[1], RAD_SHARD_ROW,
                                   name.c_str(), "scale,scale.1", banks, width);
    const rad_weight codes = b.decl_weight("c01.codes", &weight);
    const rad_weight scales = b.decl_weight("c01.scales", &scale);
    require(codes && scales, "native builder code/scale declare refused");
    LinearFP8 linear{};
    linear.w = codes;
    linear.ws = scales;
    linear.ws_row_div = 1; // two-level IQ8R view is at full logical [N,K]
    const int64_t lo[3] = {start, start, start};
    const int64_t hi[3] = {start+width, start+width, start+width};
    const int64_t totals[3] = {full, full, full};
    require(linear.span_parts(&b, lo, hi, totals, banks) == RAD_OK,
            "native LinearFP8 code + two-level scale span refused");

    const std::string bias_name = name.substr(0, name.size()-7) + ".bias";
    const auto* be = f.find(bias_name);
    require(be && be->rank == 1 && be->shape[0] == e.shape[0],
            "fused QKV/KV bias missing or wrong");
    RadWeightDecl bd{};
    bd.dtype = RAD_BF16;
    bd.rank = 1;
    bd.shape[0] = banks * width;
    bd.shard = RAD_SHARD_ROW;
    bd.group = rad_group_layer(0);
    bd.access = RAD_ACCESS_PER_TOKEN;
    bd.source = bias_name.c_str();
    bd.n_row_parts = banks;
    for (int i=0;i<banks;++i) bd.row_parts[i] = width;
    const rad_weight bias = b.decl_weight("c01.bias", &bd);
    require(bias && b.weight_shard_span_parts(bias, lo, hi, totals, banks) == RAD_OK,
            "native bias span refused");
    require(b.errors().empty(), "native builder recorded error");
    check_parts(*b.program().weight(codes), start, width, full, banks);
    check_parts(*b.program().weight(scales), start, width, full, banks);
    check_parts(*b.program().weight(bias), start, width, full, banks);
    const WeightInfo* sw = b.program().weight(scales);
    require(sw->enc_known && sw->n_sel == 2 && sw->sel_cols[0] == e.shape[1] / sc->block[1] &&
            sw->sel_cols[1] == 1,
            "two-level scale receipt did not follow C01 logical view");
}
void col_view(const RadFile& f, Registry& reg, const std::string& name,
              const std::array<int,3>& heads, int rank) {
    const auto& e = entry(f, name);
    const RadEncoding& enc = f.encoding(e);
    require(rad::arch::linear_is_iq8r(enc), "output is not IQ8R");
    const int64_t group = rad::arch::linear_w8h_group(enc);
    const int64_t full = e.shape[1];
    const int64_t all_heads = heads[0] + heads[1] + heads[2];
    require(all_heads > 0 && full % all_heads == 0, "nonwhole-head K");
    const int64_t hdim = full / all_heads;
    int64_t start = 0;
    for(int i=0;i<rank;++i) start += heads[i]*hdim;
    const int64_t width = heads[rank]*hdim;
    const auto slice = C01Slice::columns(full, start, width);
    const bool aligned = start % group == 0 && width % group == 0;
    require(slice.valid(e.shape[0], width, group) == aligned, "C01 K geometry mismatch");
    if (!aligned) return;

    // C01 only supports biasless output columns. Probe video FFN-down.
    require(!f.find(name.substr(0,name.size()-7)+".bias"), "output has a bias");
    Builder b(reg, metadata(), context(rank));
    b.set_sources(rad::rad_container_sources(f));
    auto d = weight_decl(RAD_I8, e.shape[0], width, RAD_SHARD_COL,
                         name.c_str(), "codes");
    auto s = weight_decl(RAD_U8, e.shape[0], width, RAD_SHARD_COL,
                         name.c_str(), "scale,scale.1");
    const rad_weight codes = b.decl_weight("c01.out.codes", &d);
    const rad_weight scales = b.decl_weight("c01.out.scale", &s);
    require(codes && scales, "native column weight declaration refused");
    require(b.weight_shard_span(codes, start, start+width) == RAD_OK &&
            b.weight_shard_span(scales, start, start+width) == RAD_OK,
            "native two-level IQ8R K spans refused");
    require(b.errors().empty(), "native builder column errors");
    check_span(*b.program().weight(codes), start, start+width);
    check_span(*b.program().weight(scales), start, start+width);
    require(b.program().weight(scales)->n_sel == 2, "IQ8R secondary scale omitted");
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: ltx_c01_weight_receipt_test MODEL.rad\n");
        return 2;
    }
    try {
        RadFile model;
        require(model.open(argv[1]) == RAD_OK, "RAD2 model open/validate failed");
        Registry reg;
        int row = 0, col = 0, refused = 0;
        const std::array<std::array<int,3>,3> splits = {{{8,8,16},{12,12,8},{10,10,12}}};
        const int64_t blocks = model_i64(model, "transformer.num_layers");
        const int64_t vh = model_i64(model, "transformer.num_attention_heads");
        const int64_t ah = model_i64(model, "transformer.audio_num_attention_heads");
        const int64_t vhd = model_i64(model, "transformer.attention_head_dim");
        const int64_t ahd = model_i64(model, "transformer.audio_attention_head_dim");
        require(blocks > 0 && vh == ah && vh > 0 && vhd > 0 && ahd > 0,
                "unsupported source modality head geometry");
        for(int64_t block=0; block<blocks; ++block) {
            const std::string prefix = "dit.blk." + std::to_string(block) + ".";
            for(const auto& shares : splits) {
                require(shares[0] + shares[1] + shares[2] == vh,
                        "rank head coverage changed");
                for(int rank=0; rank<3; ++rank) {
                    row_view(model,reg,prefix+"attn1.qkv.weight",vhd,3,shares,rank);
                    row_view(model,reg,prefix+"audio_attn1.qkv.weight",ahd,3,shares,rank);
                    row_view(model,reg,prefix+"attn2.kv.weight",vhd,2,shares,rank);
                    row_view(model,reg,prefix+"audio_attn2.kv.weight",ahd,2,shares,rank);
                    row_view(model,reg,prefix+"a2v.kv.weight",ahd,2,shares,rank);
                    row_view(model,reg,prefix+"v2a.kv.weight",ahd,2,shares,rank);
                    row += 6;
                    col_view(model,reg,prefix+"ff_down.weight",shares,rank);
                    col++;
                }
            }
            // Biased outputs require bias-once after all-reduce, not partial bias.
            for(const char* suffix: {"attn1.out.weight", "attn2.out.weight",
                                     "audio_attn1.out.weight", "audio_attn2.out.weight",
                                     "a2v.out.weight", "v2a.out.weight",
                                     "audio_ff_down.weight"}) {
                const std::string name = prefix + suffix;
                const auto& e = entry(model,name);
                const std::string bn = name.substr(0,name.size()-7)+".bias";
                const auto* bias = model.find(bn);
                require(bias && bias->rank == 1 && bias->shape[0] == e.shape[0],
                        "biased output missing model bias");
                ++refused;
            }
            // A 10/10/12 audio attention K cut is non-group aligned.
            const auto& e = entry(model,prefix+"audio_attn1.out.weight");
            const auto& enc = model.encoding(e);
            const int64_t group=rad::arch::linear_w8h_group(enc), hdim=e.shape[1]/ah;
            const int64_t part = splits[2][0] * hdim;
            require(!C01Slice::columns(e.shape[1], 0, part).valid(e.shape[0], part, group),
                    "invalid audio split unexpectedly admitted");
        }
        std::printf("LTX2 native ABI44 IQ8R Builder receipts: rows=%d cols=%d blocked=%d\n",
                    row,col,refused);
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr, "LTX2 C01 native receipt failed: %s\n", e.what());
        return 1;
    }
}
