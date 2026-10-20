#pragma once

#include "moe_lane_common.h"
#include "moe_lane_front.h"
#include "moe_lane_gemm.h"
#include "moe_lane_reduce.h"
#include "moe_lane_host.h"

namespace r3_moe_gpu {

class Runner {
public:
    explicit Runner(const Container& model) : model_(model) {}
    /* The decode GEMV for the weight format in use (W4 nonlinear nibbles or FP8 block-scaled). */
    template <bool TOKEN>
    void launch_gemv(dim3 grid, hipStream_t stream, const uint8_t* aq, const float* as, const int32_t* sorted, const int32_t* off,
                     uint32_t top_k, void* const* code0, void* const* scale0, void* const* code1, void* const* scale1, uint32_t n0,
                     uint32_t n1, uint32_t k0, uint32_t k1, uint32_t a_stride, uint32_t out_stride, uint16_t* out, uint32_t part_pitch) {
        if (weight_format_ == MOE_FMT_FP8_E4M3_BLOCK)
            hipLaunchKernelGGL((lane_moe_gemv<TOKEN, true>), grid, dim3(256), 0, stream, aq, as, sorted, off, top_k, code0, scale0, code1,
                               scale1, n0, n1, k0, k1, a_stride, out_stride, out, part_pitch);
        else
            hipLaunchKernelGGL((lane_moe_gemv<TOKEN, false>), grid, dim3(256), 0, stream, aq, as, sorted, off, top_k, code0, scale0, code1,
                               scale1, n0, n1, k0, k1, a_stride, out_stride, out, part_pitch);
    }
    /* Routed experts per layer, from the container; and the thread count of the one-block routing kernels: a power of two, at least
     * kFuseMaxRoutes and the expert count (0 when the count is beyond what one block can scan). */
    uint32_t experts() const { return model_.experts; }
    uint32_t weight_format() const { return weight_format_; }
    /* The shape the kernels run with: hidden width, routes a token, activation group (RCS09-06). Checked against the capacities the
     * kernels are built for (per-token scratch, the GEMV's LDS row, whole 16-byte activation pieces) and copied to the device. */
    r3::moe_shm::LayoutShape shape_{};
    /* The expert weight format the worker serves (the backend contract's MOE_FMT_*). W4NL64 is what open() prepared; FP8 block-scaled
     * switches to radiance's fp8a8 grouped GEMM and the plain quantisers, reads the container's planes as they are (that kernel's
     * layout is the checkpoint's own row-major bytes) and takes the grouped GEMM for decode too (no FP8 GEMV yet). */
    bool set_weight_format(uint32_t fmt) {
        if (fmt == MOE_FMT_W4NL64) return true;
        if (fmt != MOE_FMT_FP8_E4M3_BLOCK) {
            std::fprintf(stderr, "r3_moe_worker: no kernels for weight format %u\n", fmt);
            return false;
        }
        if (!moe_fp8_ || !moe_fp8_->layout || !quant_plain_ || !gated_plain_) {
            std::fprintf(stderr, "r3_moe_worker: installed libr4d lacks the fp8a8 grouped GEMM or the plain quantiser rows\n");
            return false;
        }
        if (!full_width()) {
            std::fprintf(stderr, "r3_moe_worker: FP8 experts are served full width (one engine rank), not sliced across two\n");
            return false;
        }
        weight_format_ = fmt;
        dtype_ = "fp8a8";   // the decode GEMV (R3_MOE_WORKER_GEMV, default on) now has an FP8 form
        moe_ = moe_fp8_;
        had_quant_ = quant_plain_;
        gated_quant_ = gated_plain_;
        gemv_debug_ = false;   // the grouped-GEMM-vs-GEMV debug comparison is W4-only
        /* Large-lane steps of up to 16 rows take the FP8 GEMV too (35B stream A/B: engine decode +3.5% at 4 streams, +2.7% at 6; 32 rows
         * is worse at 6); R3_MOE_WORKER_GEMV_LARGE_ROWS still overrides. */
        if (!std::getenv("R3_MOE_WORKER_GEMV_LARGE_ROWS")) gemv_large_rows_ = 16;
        rows_ = false;
        return true;
    }
    bool set_shape(const r3::moe_shm::LayoutShape& s) {
        if (!s.hidden || !s.top_k || s.top_k > r3::moe_shm::kMaxTopK || !s.act_group || s.hidden % 16 || s.hidden % s.act_group ||
            ((s.hidden / s.act_group) * 4) % 16 || s.hidden > kGemvMaxK) {
            std::fprintf(stderr, "r3_moe_worker: lane kernels cannot run hidden %u, top-k %u, group %u\n", s.hidden, s.top_k, s.act_group);
            return false;
        }
        if (!full_width() && s.intermediate != kExpertWidth) {   // the two-rank slice split (384 + 256) is defined for width 640 only
            std::fprintf(stderr, "r3_moe_worker: two-rank slicing is defined for expert width %lld, not %u\n", (long long)kExpertWidth, s.intermediate);
            return false;
        }
        shape_ = s;
        const LaneShape ls{s.hidden, s.top_k, s.act_group, s.experts};
        return hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(g_shape), &ls, sizeof ls), "set lane shape");
    }
    uint32_t route_block() const {
        uint32_t n = kFuseMaxRoutes;
        while (n < model_.experts) n <<= 1;
        return n <= 1024 ? n : 0u;
    }
    ~Runner() {
        if (stream_) (void)hipStreamDestroy(stream_);
        for (auto& l : lane_) {
            if (l.sample_start) (void)hipEventDestroy(l.sample_start);
            if (l.sample_stop) (void)hipEventDestroy(l.sample_stop);
            if (l.stream) (void)hipStreamDestroy(l.stream);
        }
        for (void* p : persistent_) if (p) (void)hipFree(p);
        if (plugin_opened_ && plugin_close_) plugin_close_();
        if (so_) dlclose(so_);
    }

    bool open() {
        const char* poison = std::getenv("R3_MOE_WORKER_POISON");
        /* "zero" returns zeros (the same output as dropping the remote routes); anything else non-"0" is 1e3. */
        if (const char* rv = std::getenv("R3_MOE_WORKER_ROWS"); rv && *rv && std::strcmp(rv, "0") != 0) {
            rows_ = true;
            std::fprintf(stderr, "r3_moe_worker: few-rows prefill kernel on the large lane\n");
        }
        /* The decode GEMV is the default (RAD05: decode +37%, KLD inside the noise floor); 0 turns it off,
         * 2 runs the grouped GEMM and compares the two (--compare). */
        {
            const char* gv = std::getenv("R3_MOE_WORKER_GEMV");
            const bool off = gv && std::strcmp(gv, "0") == 0;
            gemv_debug_ = gv && std::strcmp(gv, "2") == 0;
            gemv_ = !off && !gemv_debug_;
        }
        /* R3_MOE_WORKER_GEMV_LARGE_ROWS (RAD06): large-lane requests of at most this many rows take the GEMV too.
         * Replayed real requests show 9-row steps (decode / MTP verify just over the small lane) at ~0.5 ms a
         * grouped-GEMM call for 25-51 routes; the large lane is sized per request, so this costs the small lane
         * nothing (raising R3_MOE_WORKER_SMALL_ROWS pads every pre-queued decode step instead). 0 = off. */
        if (const char* gl = std::getenv("R3_MOE_WORKER_GEMV_LARGE_ROWS"); gl && *gl)
            gemv_large_rows_ = (uint32_t)std::max(0, std::atoi(gl));
        if (gemv_large_rows_)
            std::fprintf(stderr, "r3_moe_worker: large-lane requests up to %u rows take the GEMV\n", gemv_large_rows_);
        poison_ = !poison || !*poison || std::strcmp(poison, "0") == 0 ? 0u
                : std::strcmp(poison, "zero") == 0 ? 2u : 1u;
        if (!hip_ok(hipSetDevice(0), "hipSetDevice(0)")) return false;
        hipDeviceProp_t prop{};
        if (!hip_ok(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties")) return false;
        arch_ = prop.gcnArchName;
        const size_t colon = arch_.find(':');
        if (colon != std::string::npos) arch_.resize(colon);
        /* The expert kernels come from the plugin built for this card: libr4d on gfx1201 (the R9700 helping two
         * XTX ranks), libr3 on gfx1100 (an XTX helping an R9700 engine, RAD02). Both export the same rows. */
        if (arch_ != "gfx1201" && arch_ != "gfx1100") {
            std::fprintf(stderr,
                "r3_moe_worker: visible GPU is %s, want gfx1201 or gfx1100 (set R3_MOE_WORKER_CPU=1 for host path)\n",
                arch_.c_str());
            return false;
        }
        if (!load_plugin()) return false;
        if (!hip_ok(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "hipStreamCreate")) return false;
        for (uint32_t r = 0; r < r3::moe_shm::kLanes; ++r) {
            if (!hip_ok(hipStreamCreateWithFlags(&lane_[r].stream, hipStreamNonBlocking), "hipStreamCreate lane") ||
                !hip_ok(hipEventCreate(&lane_[r].sample_start), "hipEventCreate lane start") ||
                !hip_ok(hipEventCreate(&lane_[r].sample_stop), "hipEventCreate lane stop") ||
                !hip_ok(hipEventCreate(&lane_[r].stage[0]), "hipEventCreate lane stage") ||
                !hip_ok(hipEventCreate(&lane_[r].stage[1]), "hipEventCreate lane stage") ||
                !hip_ok(hipEventCreate(&lane_[r].stage[2]), "hipEventCreate lane stage") ||
                !hip_ok(hipEventCreate(&lane_[r].stage[3]), "hipEventCreate lane stage"))
                return false;
        }
        experts_.resize((size_t)model_.layers * experts());
        tables_.resize(model_.layers);
        /* hidden width and expert count are the container's; row capacity, top-k and group are the protocol's until negotiated */
        return set_shape(r3::moe_shm::LayoutShape{model_.hidden, model_.top_k ? model_.top_k : r3::moe_shm::kTopK, model_.experts, r3::moe_shm::kMaxRows,
                                                  r3::moe_shm::kActGroup, model_.intermediate});
    }

    const std::string& arch() const { return arch_; }

    int64_t expert_bytes(uint32_t layer, uint32_t expert) {
        const RadFileEntry* gu = model_.weight(layer, expert, 0);
        const RadFileEntry* dn = model_.weight(layer, expert, 1);
        if (!gu || !dn) return -1;
        WeightPlan pg, pd;
        if (!plan_weight(gu, true, &pg) || !plan_weight(dn, false, &pd)) return -1;
        return pg.code.bytes + pg.scale.bytes + pd.code.bytes + pd.scale.bytes;
    }

    bool allocate(const std::vector<std::pair<uint32_t, uint32_t>>& owned) {
        constexpr size_t kLoadBounceBytes = 64ull * 1024ull * 1024ull;
        if (!load_bounce_.ensure(kLoadBounceBytes)) return false;
        size_t total = 0;
        for (auto [l, e] : owned) {
            for (uint32_t rank = 0; rank < stored_slices(); ++rank) {
                int64_t cb = 0, sb = 0;
                if (!slice_bytes(model_.weight(l, e, 0), true, 0, rank, &cb, &sb)) return false;
                total += arena_round(cb) + arena_round(sb);
                if (!slice_bytes(model_.weight(l, e, 1), false, 0, rank, &cb, &sb)) return false;
                total += arena_round(cb) + arena_round(sb);
            }
        }
        if (!hip_ok(hipMalloc(&arena_, total ? total : 16), "hipMalloc resident expert arena")) return false;
        persistent_.push_back(arena_);
        arena_size_ = total; arena_used_ = 0;
        layer_loaded_.assign(model_.layers, 0);
        const size_t slot_bytes = (size_t)model_.layers * experts() * sizeof(uint16_t);
        const size_t table_bytes = (size_t)model_.layers * sizeof(DeviceLayerTables);
        return d_expert_slot_.ensure(slot_bytes) &&
               d_layer_tables_.ensure(table_bytes) &&
               hip_ok(hipMemset(d_expert_slot_.p, 0xff, slot_bytes), "initialize expert slot map") &&
               hip_ok(hipMemset(d_layer_tables_.p, 0, table_bytes), "initialize layer table index");
    }

    bool load_layer(uint32_t layer,
                    const std::vector<std::pair<uint32_t, uint32_t>>& owned,
                    const int32_t* slot_of_expert) {
        using namespace r3::moe_shm;
        if (layer >= model_.layers || !slot_of_expert || layer_loaded_.empty()) return false;
        if (layer_loaded_[layer]) return true;

        std::vector<uint8_t> seen(experts(), 0);
        uint32_t nreg = 0;
        for (uint32_t e = 0; e < experts(); ++e) {
            const int32_t es = slot_of_expert[e];
            if (es < 0) continue;
            if (es >= (int32_t)experts() || seen[(uint32_t)es]) return false;
            seen[(uint32_t)es] = 1;
            nreg = std::max(nreg, (uint32_t)es + 1u);
        }
        for (uint32_t es = 0; es < nreg; ++es) if (!seen[es]) return false;

        for (auto [l, e] : owned) {
            if (l != layer || slot_of_expert[e] < 0) continue;
            const uint32_t es = (uint32_t)slot_of_expert[e];
            Expert& x = experts_[(size_t)l * experts() + e];
            if (!upload_weight(model_.weight(l, e, 0), true, es, &x.gu) ||
                !upload_weight(model_.weight(l, e, 1), false, es, &x.dn))
                return false;
            x.live = true;
        }
        if (!build_layer_table(layer, slot_of_expert)) return false;
        layer_loaded_[layer] = 1;
        return true;
    }

    int compute(void* map, r3::moe_shm::Request* q) {
        using namespace r3::moe_shm;
        if (!map || !q || q->rank >= kMaxRanks || q->rank_count != kMaxRanks || q->layer >= model_.layers ||
            q->rows == 0 || q->rows > kMaxRows || q->hidden != shape_.hidden ||
            q->route_count == 0 || q->route_count > kMaxRoutes ||
            q->activation_stride != shape_.hidden * (uint32_t)sizeof(uint16_t)) return -EINVAL;

        const Route* route = ptr<Route>(map, q->route_offset);
        const uint16_t* hact = ptr<uint16_t>(map, q->activation_offset);
        const uint32_t rows = q->rows;
        const uint32_t tr = rows * shape_.top_k;

        std::vector<std::vector<uint32_t>> by_slot(experts());
        std::vector<uint16_t> ew((size_t)rows * shape_.top_k, 0);
        for (uint32_t i = 0; i < q->route_count; ++i) {
            const Route& r = route[i];
            if (r.token < 0 || (uint32_t)r.token >= rows || r.expert < 0 ||
                r.expert >= (int32_t)experts() || !owns(q->layer, (uint32_t)r.expert)) return -EINVAL;
            if (r.slot >= shape_.top_k) return -EINVAL;
            const int32_t es = layer_slot(map, q->layer, (uint32_t)r.expert);
            if (es < 0 || es >= (int32_t)experts()) return -EINVAL;
            const uint32_t flat = (uint32_t)r.token * shape_.top_k + r.slot;
            ew[flat] = r.weight_bf16;
            by_slot[(uint32_t)es].push_back(flat);
        }

        std::vector<int32_t> sorted(tr, -1);
        std::vector<int32_t> off(experts() + 1, 0);
        uint32_t at = 0;
        for (uint32_t es = 0; es < experts(); ++es) {
            off[es] = (int32_t)at;
            for (uint32_t flat : by_slot[es]) sorted[at++] = (int32_t)flat;
        }
        off[experts()] = (int32_t)at;
        if (at != q->route_count) return -EINVAL;

        if (!ensure_work(rows, tr)) return -ENOMEM;
        const size_t act_bytes = (size_t)rows * shape_.hidden * sizeof(uint16_t);
        if (!hip_ok(hipMemcpyAsync(d_act_.p, hact, act_bytes, hipMemcpyHostToDevice, stream_),
                    "upload activations")) return -EIO;
        if (!hip_ok(hipMemcpyAsync(d_sorted_.p, sorted.data(), (size_t)tr * sizeof(int32_t),
                                   hipMemcpyHostToDevice, stream_), "upload sorted routes") ||
            !hip_ok(hipMemcpyAsync(d_off_.p, off.data(), off.size() * sizeof(int32_t),
                                   hipMemcpyHostToDevice, stream_), "upload expert offsets") ||
            !hip_ok(hipMemcpyAsync(d_ew_.p, ew.data(), ew.size() * sizeof(uint16_t),
                                   hipMemcpyHostToDevice, stream_), "upload route weights"))
            return -EIO;

        std::vector<RadParam> aqp = {
            pi("M", rows), pi("n", shape_.hidden), pi("group", kGroup), ps("dtype", "bf16")
        };
        if (launch(had_quant_,
                   { tensor(d_act_.p, RAD_BF16, {rows, shape_.hidden}),
                     tensor(d_aq_.p, RAD_F8E4M3, {rows, shape_.hidden}),
                     tensor(d_as_.p, RAD_F32, {rows, shape_.hidden / kGroup}) }, aqp) < 0)
            return -EIO;

        for (uint32_t rank = 0; rank < kMaxRanks; ++rank) {
            LayerTables& lt = tables_[q->layer];
            const int64_t w0 = rank_slice_width(rank, 0);
            const int64_t w1 = rank_slice_width(rank, 1);

            std::vector<RadParam> gp =
                gemm_params(rows, 2 * w0, shape_.hidden, 2, "token", 2 * w1, shape_.hidden);
            if (launch(moe_,
                       { tensor(d_aq_.p, RAD_F8E4M3, {rows, shape_.hidden}),
                         tensor(d_as_.p, RAD_F32, {rows, shape_.hidden / kGroup}),
                         table_tensor(lt.gu_code[rank][0], lt.gu_code_dtype, 2 * w0, shape_.hidden),
                         table_scale_tensor(lt.gu_scale[rank][0], lt.gu_scale_dtype, 2 * w0, shape_.hidden),
                         tensor(d_sorted_.p, RAD_I32, {tr}),
                         tensor(d_off_.p, RAD_I32, {experts() + 1}),
                         table_tensor(lt.gu_code[rank][1], lt.gu_code_dtype, 2 * w1, shape_.hidden),
                         table_scale_tensor(lt.gu_scale[rank][1], lt.gu_scale_dtype, 2 * w1, shape_.hidden),
                         tensor(d_gu_.p, RAD_BF16, {tr, kRankGateUpWidth}) }, gp) < 0)
                return -EIO;

            std::vector<RadParam> gqp = {
                pi("M", q->route_count), pi("n", kRankWidth), pi("group", kGroup),
                ps("dtype", "bf16"), ps("act", "silu")
            };
            if (launch(gated_quant_,
                       { tensor(d_gu_.p, RAD_BF16, {q->route_count, kRankGateUpWidth}),
                         tensor(d_mid_.p, RAD_F8E4M3, {q->route_count, kRankWidth}),
                         tensor(d_mid_s_.p, RAD_F32, {q->route_count, kRankWidth / kGroup}),
                         absent() }, gqp) < 0)
                return -EIO;

            std::vector<RadParam> dp =
                gemm_params(rows, shape_.hidden, w0, 1, "sorted", shape_.hidden, w1);
            if (launch(moe_,
                       { tensor(d_mid_.p, RAD_F8E4M3, {tr, kRankWidth}),
                         tensor(d_mid_s_.p, RAD_F32, {tr, kRankWidth / kGroup}),
                         table_tensor(lt.dn_code[rank][0], lt.dn_code_dtype, shape_.hidden, w0),
                         table_scale_tensor(lt.dn_scale[rank][0], lt.dn_scale_dtype, shape_.hidden, w0),
                         tensor(d_sorted_.p, RAD_I32, {tr}),
                         tensor(d_off_.p, RAD_I32, {experts() + 1}),
                         table_tensor(lt.dn_code[rank][1], lt.dn_code_dtype, shape_.hidden, w1),
                         table_scale_tensor(lt.dn_scale[rank][1], lt.dn_scale_dtype, shape_.hidden, w1),
                         tensor(d_down_.p, RAD_BF16, {tr, shape_.hidden}) }, dp) < 0)
                return -EIO;

            std::vector<RadParam> mp = {
                pi("M", rows), pi("n", shape_.hidden), pi("top_k", shape_.top_k), ps("dtype", "bf16")
            };
            if (launch(gather_,
                       { tensor(d_down_.p, RAD_BF16, {tr, shape_.hidden}),
                         tensor(d_ew_.p, RAD_BF16, {rows, shape_.top_k}),
                         tensor(d_sorted_.p, RAD_I32, {tr}),
                         absent(), absent(),
                         tensor(d_result_.p, RAD_BF16, {rows, shape_.hidden}) }, mp) < 0)
                return -EIO;

            void* dst = ptr<void>(map, q->result_offset[rank]);
            if (!hip_ok(hipMemcpyAsync(dst, d_result_.p, act_bytes,
                                       hipMemcpyDeviceToHost, stream_), "download TP partial"))
                return -EIO;
        }
        return hip_ok(hipStreamSynchronize(stream_), "helper stream sync") ? 0 : -EIO;
    }

    /* Small lanes are sized to their class's rows and pre-queued; the large lane to large_rows, launched per request.
     * `count` small classes are enabled (ascending `sizes`); the other small lanes are left unprepared (cap_rows 0). */
    bool prepare_lanes(const std::array<uint32_t, r3::moe_shm::kSmallClasses>& sizes, uint32_t count, uint32_t large_rows) {
        using namespace r3::moe_shm;
        if (count == 0 || count > kSmallClasses || large_rows == 0 || large_rows > kMaxRows || !d_layer_tables_.p)
            return false;
        for (uint32_t c = 0; c < count; ++c)
            if (sizes[c] == 0 || sizes[c] > kMaxRows || sizes[c] > large_rows || (c && sizes[c] <= sizes[c - 1]))
                return false;
        class_count_ = count;
        class_rows_ = sizes;
        large_rows_ = large_rows;
        const size_t nclass = experts() / 2;
        for (uint32_t lane = 0; lane < kLanes; ++lane) {
            const uint32_t rank = lane % kMaxRanks;
            const uint32_t cls = lane_class(lane);
            if (cls < kSmallClasses && cls >= class_count_) {   /* a disabled small class: no buffers, no worker */
                lane_[lane].cap_rows = 0;
                continue;
            }
            const uint32_t cap = cls < kSmallClasses ? class_rows_[cls] : large_rows_;
            LaneWork& w = lane_[lane];
            w.cap_rows = cap;
            if (!ensure_lane_work(w, cap, cap * shape_.top_k, cap * shape_.top_k) ||
                !w.d_stats.ensure(3 * sizeof(uint64_t)))
                return false;
            for (uint32_t parity = 0; parity < 2; ++parity) {
                if (!w.gu_code[parity].ensure(nclass * sizeof(void*)) ||
                    !w.gu_scale[parity].ensure(nclass * sizeof(void*)) ||
                    !w.dn_code[parity].ensure(nclass * sizeof(void*)) ||
                    !w.dn_scale[parity].ensure(nclass * sizeof(void*)))
                    return false;
            }
            if (!prepare_lane_scratch(w, rank, cap, cap * shape_.top_k) ||
                !hip_ok(hipMemset(w.d_stats.p, 0, 3 * sizeof(uint64_t)), "zero lane stats"))
                return false;
        }
        return true;
    }

    bool stage_compare_activation(void* host_map, uint32_t lane,
                                  const uint16_t* activation, uint32_t rows) {
        using namespace r3::moe_shm;
        if (!host_map || !activation || lane >= kLanes || rows == 0 || rows > lane_[lane].cap_rows)
            return false;
        const uint32_t rank = lane % kMaxRanks;
        LaneWork& w = lane_[lane];
        const size_t xbytes = (size_t)rows * shape_.hidden * sizeof(uint16_t);
        if (!hip_ok(hipMemcpyAsync(w.d_act.p, activation, xbytes,
                                   hipMemcpyHostToDevice, w.stream),
                    "--compare upload BF16 activation"))
            return false;
        std::vector<RadParam> aqp = {
            pi("M", rows), pi("n", shape_.hidden), pi("group", kGroup), ps("dtype", "bf16")
        };
        if (launch_on(had_quant_,
                      { tensor(w.d_act.p, RAD_BF16, {rows, shape_.hidden}),
                        tensor(w.d_aq.p, RAD_F8E4M3, {rows, shape_.hidden}),
                        tensor(w.d_as.p, RAD_F32, {rows, shape_.hidden / kGroup}) },
                      aqp, w.stream, w.scratch) < 0)
            return false;
        Header* h = static_cast<Header*>(host_map);
        auto* hq = ptr<uint8_t>(host_map, h->activation_offset[rank] + kActivationBf16Bytes);
        auto* hs = ptr<float>(host_map,
            h->activation_offset[rank] + kActivationBf16Bytes + kActivationQBytes);
        if (!hip_ok(hipMemcpyAsync(hq, w.d_aq.p, (size_t)rows * shape_.hidden,
                                   hipMemcpyDeviceToHost, w.stream),
                    "--compare stage exact q") ||
            !hip_ok(hipMemcpyAsync(hs, w.d_as.p,
                                   (size_t)rows * (shape_.hidden / kGroup) * sizeof(float),
                                   hipMemcpyDeviceToHost, w.stream),
                    "--compare stage exact scales"))
            return false;
        return hip_ok(hipStreamSynchronize(w.stream), "--compare stage q/s sync");
    }

    /* rows: the fixed small-lane size (pre-queued), or the request's own row count on a large lane. */
    int enqueue_lane(void* host_map, void* device_map, uint32_t lane, uint32_t sequence,
                     uint32_t rows, bool sample) {
        using namespace r3::moe_shm;
        if (!host_map || !device_map || lane >= kLanes || sequence == 0 || rows == 0 ||
            rows > lane_[lane].cap_rows)
            return -EINVAL;
        const uint32_t rank = lane % kMaxRanks;
        LaneWork& w = lane_[lane];
        auto* hhost = static_cast<Header*>(host_map);
        auto* dbase = static_cast<uint8_t*>(device_map);
        auto dev_at = [&](const void* hp) -> void* {
            return dbase + (static_cast<const uint8_t*>(hp) -
                            static_cast<const uint8_t*>(host_map));
        };

        Request* qhost = lane_request(host_map, lane, sequence);
        if (!qhost) return -EINVAL;
        auto* q = static_cast<Request*>(dev_at(qhost));
        auto* request_seq =
            static_cast<uint32_t*>(dev_at(&hhost->lane[lane].request_sequence));
        auto* done_seq =
            static_cast<uint32_t*>(dev_at(&hhost->lane[lane].done_sequence));
        auto* route = static_cast<const Route*>(
            dev_at(ptr<Route>(host_map, hhost->route_offset[rank])));
        auto* act_q = static_cast<const uint8_t*>(
            dev_at(ptr<uint8_t>(host_map,
                hhost->activation_offset[rank] + kActivationBf16Bytes)));
        auto* act_s = static_cast<const float*>(
            dev_at(ptr<float>(host_map,
                hhost->activation_offset[rank] + kActivationBf16Bytes + kActivationQBytes)));
        auto* dst = static_cast<uint16_t*>(
            dev_at(ptr<uint16_t>(
                host_map, hhost->result_offset[rank * kMaxRanks + rank])));

        if (!hip_ok(hipStreamWaitValue32(w.stream, request_seq, sequence,
                                         hipStreamWaitValueGte, 0xffffffffu),
                    "lane wait request"))
            return -EIO;
        if (sample && !w.sample_pending) {
            if (!hip_ok(hipEventRecord(w.sample_start, w.stream), "lane timing start"))
                return -EIO;
            w.sample_sequence = sequence;
            w.sample_pending = true;
        }

        const uint32_t routes = rows * shape_.top_k;
        const uint32_t slots = routes;
        /* R3_MOE_WORKER_FUSEROUTE (default on): the small lane's routing prologue in one block (lane_route_small); 0 keeps
         * the memsets and the four routing kernels. Read each call so a replay sweep can compare. */
        const char* fuse_env = std::getenv("R3_MOE_WORKER_FUSEROUTE");
        const bool fuse = lane_is_small(lane) && rows <= 8u && routes <= kFuseMaxRoutes &&
                          (!fuse_env || std::strcmp(fuse_env, "0") != 0);
        /* No clear of the result slot: it is mapped host memory, and the merge reads only the tokens this
         * request's routes name, each of which the reduction writes. */
        if (!fuse &&
            (!hip_ok(hipMemsetAsync(w.d_counts.p, 0, experts() * sizeof(uint32_t), w.stream),
                     "clear route counts") ||
             !hip_ok(hipMemsetAsync(w.d_ew.p, 0, routes * sizeof(uint16_t), w.stream),
                     "clear route weights") ||
             !hip_ok(hipMemsetAsync(w.d_sorted.p, 0xff, routes * sizeof(int32_t), w.stream),
                     "clear sorted routes") ||
             !hip_ok(hipMemsetAsync(w.d_inv.p, 0xff, slots * sizeof(int32_t), w.stream),
                     "clear slot inverse")))
            return -EIO;

        const dim3 block(256);
        const dim3 act_grid(
            ((uint64_t)rows * shape_.hidden + block.x - 1) / block.x);
        const dim3 route_grid((routes + block.x - 1) / block.x);
        /* R3_MOE_WORKER_VECIO (default on): 16-byte accesses to the engine's mapped host memory for the activation
         * copy and the result write; 0 keeps the scalar kernels. Read each call so a replay sweep can compare. */
        const char* vio_env = std::getenv("R3_MOE_WORKER_VECIO");
        const bool vecio = !vio_env || std::strcmp(vio_env, "0") != 0;
        if ((vecio || fuse) && ((reinterpret_cast<uintptr_t>(act_q) | reinterpret_cast<uintptr_t>(act_s) |
                       reinterpret_cast<uintptr_t>(dst)) & 15u) != 0) {
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_VECIO needs 16-byte aligned activation and result planes\n");
            return -EINVAL;
        }
        if (fuse) {
            /* the activation copy is folded into lane_front_small below */
        } else if (vecio) {
            const dim3 vgrid(((uint64_t)rows * (shape_.hidden / 16) + block.x - 1) / block.x);
            hipLaunchKernelGGL(lane_prepare_activation_v, vgrid, block, 0, w.stream,
                               q, reinterpret_cast<const uint4*>(act_q), reinterpret_cast<const uint4*>(act_s),
                               reinterpret_cast<uint4*>(w.d_aq.p), reinterpret_cast<uint4*>(w.d_as.p),
                               rank, sequence, rows, model_.layers,
                               static_cast<uint64_t*>(w.d_stats.p));
        } else {
            hipLaunchKernelGGL(lane_prepare_activation, act_grid, block, 0, w.stream,
                               q, act_q, act_s,
                               static_cast<uint8_t*>(w.d_aq.p), static_cast<float*>(w.d_as.p),
                               rank, sequence, rows, model_.layers,
                               static_cast<uint64_t*>(w.d_stats.p));
        }
        if (fuse) {
            auto front = [&](auto block_n) {
                constexpr uint32_t N = decltype(block_n)::value;
                hipLaunchKernelGGL((lane_front_small<N>), dim3(1), dim3(N), 0, w.stream,
                                   route, q, static_cast<const DeviceLayerTables*>(d_layer_tables_.p),
                                   reinterpret_cast<const uint4*>(act_q), reinterpret_cast<const uint4*>(act_s),
                                   reinterpret_cast<uint4*>(w.d_aq.p), reinterpret_cast<uint4*>(w.d_as.p),
                                   static_cast<uint64_t*>(w.d_stats.p),
                                   rank, full_width() ? 0u : rank, sequence, rows, model_.layers, routes,
                                   experts(), static_cast<int32_t*>(w.d_off.p), static_cast<int32_t*>(w.d_sorted.p),
                                   static_cast<int32_t*>(w.d_inv.p), static_cast<uint16_t*>(w.d_ew.p),
                                   static_cast<void**>(w.gu_code[0].p), static_cast<void**>(w.gu_scale[0].p),
                                   static_cast<void**>(w.dn_code[0].p), static_cast<void**>(w.dn_scale[0].p),
                                   static_cast<void**>(w.gu_code[1].p), static_cast<void**>(w.gu_scale[1].p),
                                   static_cast<void**>(w.dn_code[1].p), static_cast<void**>(w.dn_scale[1].p));
            };
            switch (route_block()) {
                case 256: front(std::integral_constant<uint32_t, 256>{}); break;
                case 512: front(std::integral_constant<uint32_t, 512>{}); break;
                case 1024: front(std::integral_constant<uint32_t, 1024>{}); break;
                default: return -EINVAL;
            }
        } else {
        hipLaunchKernelGGL(lane_route_count, route_grid, block, 0, w.stream,
                           route, q, static_cast<const DeviceLayerTables*>(d_layer_tables_.p),
                           rank, sequence, rows, model_.layers, experts(),
                           static_cast<uint32_t*>(w.d_counts.p),
                           static_cast<int32_t*>(w.d_flat.p),
                           static_cast<uint16_t*>(w.d_ew.p));
        auto prefix = [&](auto block_n) {
            constexpr uint32_t N = decltype(block_n)::value;
            hipLaunchKernelGGL((lane_route_prefix<N>), dim3(1), dim3(N), 0, w.stream,
                               static_cast<const uint32_t*>(w.d_counts.p),
                               static_cast<int32_t*>(w.d_off.p),
                               static_cast<uint32_t*>(w.d_cursor.p), experts());
        };
        switch (route_block()) {
            case 256: prefix(std::integral_constant<uint32_t, 256>{}); break;
            case 512: prefix(std::integral_constant<uint32_t, 512>{}); break;
            case 1024: prefix(std::integral_constant<uint32_t, 1024>{}); break;
            default: return -EINVAL;
        }
        hipLaunchKernelGGL(lane_route_scatter, route_grid, block, 0, w.stream,
                           route, q, static_cast<const DeviceLayerTables*>(d_layer_tables_.p),
                           rank, sequence, rows, model_.layers, experts(),
                           static_cast<const int32_t*>(w.d_flat.p),
                           static_cast<uint32_t*>(w.d_cursor.p),
                           static_cast<int32_t*>(w.d_sorted.p),
                           nullptr);
        }
        if (!hip_ok(hipGetLastError(), "lane request kernels")) return -EIO;
        const bool sampling = sample && w.sample_pending && w.sample_sequence == sequence;
        if (sampling && !hip_ok(hipEventRecord(w.stage[0], w.stream), "lane stage 0")) return -EIO;

        /* A two-rank engine asks for its own TP slice; a one-rank engine for the whole expert (full width). */
        {
            const uint32_t slice = full_width() ? 0u : rank;
            void* out_down = w.d_down.p;
            if (!fuse)
            hipLaunchKernelGGL(lane_select_tables,
                               dim3((experts() / 2 + block.x - 1) / block.x), block, 0, w.stream,
                               q, static_cast<const DeviceLayerTables*>(d_layer_tables_.p),
                               model_.layers, rank, slice, sequence, rows, experts(),
                               static_cast<void**>(w.gu_code[0].p),
                               static_cast<void**>(w.gu_scale[0].p),
                               static_cast<void**>(w.dn_code[0].p),
                               static_cast<void**>(w.dn_scale[0].p),
                               static_cast<void**>(w.gu_code[1].p),
                               static_cast<void**>(w.gu_scale[1].p),
                               static_cast<void**>(w.dn_code[1].p),
                               static_cast<void**>(w.dn_scale[1].p));
            const int64_t w0 = slice_w(slice, 0);
            const int64_t w1 = slice_w(slice, 1);
            /* Small lane, decode GEMV (RAD05): the step's routes, one block a (row, 64 outputs). */
            const bool gemv = gemv_ && (lane_is_small(lane) || rows <= gemv_large_rows_);
            /* R3_MOE_WORKER_ROWS_SMALL (RAD06): the few-rows kernel on the small (decode) lane too. A decode request's 8 rows
             * send ~2.4 routes to each distinct expert (the hot experts are owned by the helpers), and the per-route GEMV
             * reads that expert's weights once a route; here they are read once for all the expert's rows. Read each
             * call so a replay sweep can compare; the rows accumulate in a different order, so it is not bit-exact. */
            const char* rs_env = std::getenv("R3_MOE_WORKER_ROWS_SMALL");
            const bool rows_small = weight_format_ == MOE_FMT_W4NL64 && rs_env && std::strcmp(rs_env, "0") != 0 &&
                                    lane_is_small(lane) && rows <= 16u;
            const bool rowk = (rows_ && !lane_is_small(lane)) || rows_small;
            const dim3 rows_grid_gu(experts(), (uint32_t)((2 * std::max(w0, w1) + kRowsTileN - 1) / kRowsTileN));
            const dim3 rows_grid_dn(experts(), (uint32_t)((shape_.hidden + kRowsTileN - 1) / kRowsTileN));
            const dim3 gemv_grid_gu(routes, (uint32_t)((2 * std::max(w0, w1) + kGemvTile - 1) / kGemvTile));
            const dim3 gemv_grid_dn(routes, (uint32_t)((shape_.hidden + kGemvTile - 1) / kGemvTile));
            std::vector<RadParam> gp =
                gemm_params(rows, 2 * w0, shape_.hidden, 2, "token", 2 * w1, shape_.hidden);
            if (rowk) {
                hipLaunchKernelGGL(lane_moe_rows<true>, rows_grid_gu, dim3(256), 0, w.stream,
                                   static_cast<const uint8_t*>(w.d_aq.p), static_cast<const float*>(w.d_as.p),
                                   static_cast<const int32_t*>(w.d_sorted.p), static_cast<const int32_t*>(w.d_off.p),
                                   (uint32_t)shape_.top_k,
                                   static_cast<void* const*>(w.gu_code[0].p), static_cast<void* const*>(w.gu_scale[0].p),
                                   static_cast<void* const*>(w.gu_code[1].p), static_cast<void* const*>(w.gu_scale[1].p),
                                   (uint32_t)(2 * w0), (uint32_t)(2 * w1), (uint32_t)shape_.hidden, (uint32_t)shape_.hidden,
                                   (uint32_t)shape_.hidden, (uint32_t)(2 * mid_w()), static_cast<uint16_t*>(w.d_gu.p),
                                   (uint32_t)mid_w());
                if (!hip_ok(hipGetLastError(), "lane gate/up rows")) return -EIO;
            } else if (gemv) {
                launch_gemv<true>(gemv_grid_gu, w.stream,
                                   static_cast<const uint8_t*>(w.d_aq.p), static_cast<const float*>(w.d_as.p),
                                   static_cast<const int32_t*>(w.d_sorted.p), static_cast<const int32_t*>(w.d_off.p),
                                   (uint32_t)shape_.top_k,
                                   static_cast<void* const*>(w.gu_code[0].p), static_cast<void* const*>(w.gu_scale[0].p),
                                   static_cast<void* const*>(w.gu_code[1].p), static_cast<void* const*>(w.gu_scale[1].p),
                                   (uint32_t)(2 * w0), (uint32_t)(2 * w1), (uint32_t)shape_.hidden, (uint32_t)shape_.hidden,
                                   (uint32_t)shape_.hidden, (uint32_t)(2 * mid_w()), static_cast<uint16_t*>(w.d_gu.p),
                                   (uint32_t)mid_w());
                if (!hip_ok(hipGetLastError(), "lane gate/up GEMV")) return -EIO;
            } else if (launch_on(moe_,
                          { tensor(w.d_aq.p, RAD_F8E4M3, {rows, shape_.hidden}),
                            tensor(w.d_as.p, RAD_F32, {rows, shape_.hidden / kGroup}),
                            table_tensor(w.gu_code[0].p, gu_code_dtype_, 2 * w0, shape_.hidden),
                            table_scale_tensor(w.gu_scale[0].p, gu_scale_dtype_, 2 * w0, shape_.hidden),
                            tensor(w.d_sorted.p, RAD_I32, {routes}),
                            tensor(w.d_off.p, RAD_I32, {experts() + 1}),
                            table_tensor(w.gu_code[1].p, gu_code_dtype_, 2 * w1, shape_.hidden),
                            table_scale_tensor(w.gu_scale[1].p, gu_scale_dtype_, 2 * w1, shape_.hidden),
                            tensor(w.d_gu.p, RAD_BF16, {routes, 2 * mid_w()}) },
                          gp, w.stream, w.scratch) < 0)
                return -EIO;

            if (gemv_debug_ && weight_format_ == MOE_FMT_W4NL64 && lane_is_small(lane)) {
                hipLaunchKernelGGL((lane_moe_gemv<true, false>), gemv_grid_gu, dim3(256), 0, w.stream,
                                   static_cast<const uint8_t*>(w.d_aq.p), static_cast<const float*>(w.d_as.p),
                                   static_cast<const int32_t*>(w.d_sorted.p), static_cast<const int32_t*>(w.d_off.p),
                                   (uint32_t)shape_.top_k,
                                   static_cast<void* const*>(w.gu_code[0].p), static_cast<void* const*>(w.gu_scale[0].p),
                                   static_cast<void* const*>(w.gu_code[1].p), static_cast<void* const*>(w.gu_scale[1].p),
                                   (uint32_t)(2 * w0), (uint32_t)(2 * w1), (uint32_t)shape_.hidden, (uint32_t)shape_.hidden,
                                   (uint32_t)shape_.hidden, (uint32_t)(2 * mid_w()), static_cast<uint16_t*>(w.d_dbg.p),
                                   (uint32_t)mid_w());
                w.dbg_routes = routes;
            }
            if (sampling && !hip_ok(hipEventRecord(w.stage[1], w.stream), "lane stage 1")) return -EIO;
            std::vector<RadParam> gqp = {
                pi("M", routes), pi("n", mid_w()), pi("group", kGroup),
                ps("dtype", "bf16"), ps("act", "silu")
            };
            if (launch_on(gated_quant_,
                          { tensor(w.d_gu.p, RAD_BF16, {routes, 2 * mid_w()}),
                            tensor(w.d_mid.p, RAD_F8E4M3, {routes, mid_w()}),
                            tensor(w.d_mid_s.p, RAD_F32, {routes, mid_w() / kGroup}),
                            absent() }, gqp, w.stream, w.scratch) < 0)
                return -EIO;

            if (sampling && !hip_ok(hipEventRecord(w.stage[2], w.stream), "lane stage 2")) return -EIO;
            std::vector<RadParam> dp =
                gemm_params(rows, shape_.hidden, w0, 1, "sorted", shape_.hidden, w1);
            if (rowk) {
                hipLaunchKernelGGL(lane_moe_rows<false>, rows_grid_dn, dim3(256), 0, w.stream,
                                   static_cast<const uint8_t*>(w.d_mid.p), static_cast<const float*>(w.d_mid_s.p),
                                   static_cast<const int32_t*>(w.d_sorted.p), static_cast<const int32_t*>(w.d_off.p),
                                   (uint32_t)shape_.top_k,
                                   static_cast<void* const*>(w.dn_code[0].p), static_cast<void* const*>(w.dn_scale[0].p),
                                   static_cast<void* const*>(w.dn_code[1].p), static_cast<void* const*>(w.dn_scale[1].p),
                                   (uint32_t)shape_.hidden, (uint32_t)shape_.hidden, (uint32_t)w0, (uint32_t)w1,
                                   (uint32_t)mid_w(), (uint32_t)shape_.hidden, static_cast<uint16_t*>(out_down), 0u);
                if (!hip_ok(hipGetLastError(), "lane down rows")) return -EIO;
            } else if (gemv) {
                launch_gemv<false>(gemv_grid_dn, w.stream,
                                   static_cast<const uint8_t*>(w.d_mid.p), static_cast<const float*>(w.d_mid_s.p),
                                   static_cast<const int32_t*>(w.d_sorted.p), static_cast<const int32_t*>(w.d_off.p),
                                   (uint32_t)shape_.top_k,
                                   static_cast<void* const*>(w.dn_code[0].p), static_cast<void* const*>(w.dn_scale[0].p),
                                   static_cast<void* const*>(w.dn_code[1].p), static_cast<void* const*>(w.dn_scale[1].p),
                                   (uint32_t)shape_.hidden, (uint32_t)shape_.hidden, (uint32_t)w0, (uint32_t)w1,
                                   (uint32_t)mid_w(), (uint32_t)shape_.hidden, static_cast<uint16_t*>(out_down), 0u);
                if (!hip_ok(hipGetLastError(), "lane down GEMV")) return -EIO;
            } else if (launch_on(moe_,
                          { tensor(w.d_mid.p, RAD_F8E4M3, {routes, mid_w()}),
                            tensor(w.d_mid_s.p, RAD_F32, {routes, mid_w() / kGroup}),
                            table_tensor(w.dn_code[0].p, dn_code_dtype_, shape_.hidden, w0),
                            table_scale_tensor(w.dn_scale[0].p, dn_scale_dtype_, shape_.hidden, w0),
                            tensor(w.d_sorted.p, RAD_I32, {routes}),
                            tensor(w.d_off.p, RAD_I32, {experts() + 1}),
                            table_tensor(w.dn_code[1].p, dn_code_dtype_, shape_.hidden, w1),
                            table_scale_tensor(w.dn_scale[1].p, dn_scale_dtype_, shape_.hidden, w1),
                            tensor(out_down, RAD_BF16, {routes, shape_.hidden}) },
                          dp, w.stream, w.scratch) < 0)
                return -EIO;
        }

        if (sampling && !hip_ok(hipEventRecord(w.stage[3], w.stream), "lane stage 3")) return -EIO;
        if (!fuse)
            hipLaunchKernelGGL(lane_route_inverse, route_grid, block, 0, w.stream,
                               static_cast<const int32_t*>(w.d_sorted.p),
                               static_cast<int32_t*>(w.d_inv.p), routes, slots);
        if (vecio)
            hipLaunchKernelGGL(lane_reduce_tokens_v, dim3(rows), block, 0, w.stream,
                               static_cast<const uint16_t*>(w.d_down.p),
                               static_cast<const int32_t*>(w.d_inv.p),
                               static_cast<const uint16_t*>(w.d_ew.p),
                               dst, rows, shape_.hidden, poison_);
        else
            hipLaunchKernelGGL(lane_reduce_tokens, dim3(rows), block, 0, w.stream,
                               static_cast<const uint16_t*>(w.d_down.p),
                               static_cast<const int32_t*>(w.d_inv.p),
                               static_cast<const uint16_t*>(w.d_ew.p),
                               dst, rows, shape_.hidden, poison_);
        if (!hip_ok(hipGetLastError(), "lane reduce tokens")) return -EIO;

        if (sample && w.sample_pending && w.sample_sequence == sequence &&
            !hip_ok(hipEventRecord(w.sample_stop, w.stream), "lane timing stop"))
            return -EIO;
        hipLaunchKernelGGL(lane_publish_done, dim3(1), dim3(1), 0, w.stream,
                           q, done_seq, sequence);
        if (!hip_ok(hipGetLastError(), "lane publish done")) return -EIO;
        return 0;
    }

    bool collect_lane_timing(uint32_t lane, uint32_t done_sequence,
                             double* sampled_us, uint64_t* samples) {
        if (lane >= r3::moe_shm::kLanes || !sampled_us || !samples) return false;
        LaneWork& w = lane_[lane];
        if (!w.sample_pending ||
            (int32_t)(done_sequence - w.sample_sequence) < 0)
            return true;
        float ms = 0.0f;
        if (!hip_ok(hipEventElapsedTime(&ms, w.sample_start, w.sample_stop),
                    "lane timing elapsed"))
            return false;
        *sampled_us += (double)ms * 1000.0;
        ++*samples;
        hipEvent_t mark[LaneWork::kStages + 2] = {w.sample_start, w.stage[0], w.stage[1], w.stage[2], w.stage[3],
                                                  w.sample_stop};
        for (int i = 0; i <= LaneWork::kStages; ++i) {
            float part = 0.0f;
            if (hipEventElapsedTime(&part, mark[i], mark[i + 1]) == hipSuccess) w.stage_us[i] += (double)part * 1000.0;
        }
        w.sample_pending = false;
        /* The sampled request's per-expert row counts (d_counts; this lane has synced): the run-length histogram the
         * grouped GEMM's tile choice depends on. Sampled requests only, so the copy is off the hot path. */
        std::vector<uint32_t> counts(experts());
        if (hipMemcpy(counts.data(), w.d_counts.p, counts.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess) {
            for (uint32_t e = 0; e < experts(); ++e) {
                const uint32_t c = counts[e];
                if (!c) continue;
                const int b = c == 1 ? 0 : c == 2 ? 1 : c <= 4 ? 2 : c <= 8 ? 3 : c <= 16 ? 4 : c <= 32 ? 5 : c <= 64 ? 6 : 7;
                ++w.run_hist[b];
            }
        }
        return true;
    }

    /* Sampled small-lane requests, per stage: activation+routes, gate/up GEMM, gated quant, down GEMM, reduce. */
    void stage_report(uint32_t lane, uint64_t samples) const {
        if (lane >= r3::moe_shm::kLanes || !samples) return;
        const LaneWork& w = lane_[lane];
        std::fprintf(stderr, "r3_moe_worker: lane %u stages (us/request): routes %.1f gate_up %.1f quant %.1f "
                     "down %.1f reduce %.1f\n", lane, w.stage_us[0] / samples, w.stage_us[1] / samples,
                     w.stage_us[2] / samples, w.stage_us[3] / samples, w.stage_us[4] / samples);
        uint64_t runs = 0;
        for (uint64_t h : w.run_hist) runs += h;
        if (runs)
            std::fprintf(stderr, "r3_moe_worker: lane %u experts a request by rows: 1 %.1f, 2 %.1f, 3-4 %.1f, 5-8 %.1f, "
                         "9-16 %.1f, 17-32 %.1f, 33-64 %.1f, 65+ %.1f\n", lane,
                         (double)w.run_hist[0] / samples, (double)w.run_hist[1] / samples, (double)w.run_hist[2] / samples,
                         (double)w.run_hist[3] / samples, (double)w.run_hist[4] / samples, (double)w.run_hist[5] / samples,
                         (double)w.run_hist[6] / samples, (double)w.run_hist[7] / samples);
    }

    /* R3_MOE_WORKER_GEMV=2 (--compare): the grouped GEMM's gate/up beside the GEMV's, after the lane synced. */
    void gemv_debug_report(uint32_t lane) {
        using namespace r3::moe_shm;
        LaneWork& w = lane_[lane];
        const uint32_t routes = w.dbg_routes;
        if (!gemv_debug_ || !routes) return;
        std::vector<uint16_t> g((size_t)routes * 2 * mid_w()), v((size_t)routes * 2 * mid_w());
        int32_t total = 0;
        if (hip_ok(hipMemcpy(g.data(), w.d_gu.p, g.size() * 2, hipMemcpyDeviceToHost), "gemv debug read gemm") &&
            hip_ok(hipMemcpy(v.data(), w.d_dbg.p, v.size() * 2, hipMemcpyDeviceToHost), "gemv debug read gemv") &&
            hip_ok(hipMemcpy(&total, static_cast<int32_t*>(w.d_off.p) + experts(), 4, hipMemcpyDeviceToHost), "gemv debug off")) {
            auto f = [](uint16_t b) { uint32_t u = (uint32_t)b << 16; float x; std::memcpy(&x, &u, 4); return x; };
            double num = 0, den = 0;
            for (int32_t r = 0; r < total; ++r)
                for (int64_t n = 0; n < 2 * mid_w(); ++n) {
                    const double a = f(g[(size_t)r * 2 * mid_w() + n]), b = f(v[(size_t)r * 2 * mid_w() + n]);
                    num += (a - b) * (a - b); den += a * a;
                }
            std::fprintf(stderr, "gemv-debug routes %d rel_l2 %.5g\n  gemm row0:", total,
                         std::sqrt(num / (den + 1e-30)));
            for (int n = 0; n < 8; ++n) std::fprintf(stderr, " %.5g", f(g[n]));
            std::fprintf(stderr, "\n  gemv row0:");
            for (int n = 0; n < 8; ++n) std::fprintf(stderr, " %.5g", f(v[n]));
            std::fprintf(stderr, "\n  gemm row0 n=%lld..:", (long long)mid_w());
            for (int n = 0; n < 4; ++n) std::fprintf(stderr, " %.5g", f(g[mid_w() + n]));
            std::fprintf(stderr, "\n  gemv row0 n=%lld..:", (long long)mid_w());
            for (int n = 0; n < 4; ++n) std::fprintf(stderr, " %.5g", f(v[mid_w() + n]));
            std::fprintf(stderr, "\n");
        }
    }

    bool sync_lane(uint32_t lane) {
        return lane < r3::moe_shm::kLanes &&
               hip_ok(hipStreamSynchronize(lane_[lane].stream), "lane stream sync");
    }

    bool lane_stats(uint32_t lane, uint64_t out[3]) {
        if (lane >= r3::moe_shm::kLanes || !out) return false;
        return hip_ok(hipMemcpy(out, lane_[lane].d_stats.p, 3 * sizeof(uint64_t),
                                hipMemcpyDeviceToHost), "read lane stats");
    }

private:
    const Container& model_;
    std::string arch_;
    void* so_ = nullptr;
    void (*plugin_close_)() = nullptr;
    bool plugin_opened_ = false;
    hipStream_t stream_ = nullptr;
    std::array<LaneWork, r3::moe_shm::kLanes> lane_{};
    uint32_t gu_code_dtype_ = 0, gu_scale_dtype_ = 0;
    uint32_t dn_code_dtype_ = 0, dn_scale_dtype_ = 0;
    const RadKernelInfo* moe_ = nullptr;
    const RadKernelInfo* had_quant_ = nullptr;
    const RadKernelInfo* gated_quant_ = nullptr;
    const RadKernelInfo* gather_ = nullptr;
    /* The plain-E4M3 weight rows (RAD13): radiance's own fp8a8 grouped GEMM and the quantisers without the Hadamard rotation
     * (the rotated twins exist for four-bit weights only). Chosen by set_weight_format. */
    const RadKernelInfo* moe_fp8_ = nullptr;
    const RadKernelInfo* quant_plain_ = nullptr;
    const RadKernelInfo* gated_plain_ = nullptr;
    uint32_t weight_format_ = MOE_FMT_W4NL64;
    const char* dtype_ = kDtype;
    std::vector<Expert> experts_;
    std::vector<LayerTables> tables_;
    std::vector<uint8_t> layer_loaded_;
    std::vector<void*> persistent_;
    void* arena_ = nullptr;
    size_t arena_size_ = 0, arena_used_ = 0;

    HostBounce load_bounce_;
    Buffer d_act_, d_aq_, d_as_, d_sorted_, d_off_, d_ew_, d_gu_;
    Buffer d_mid_, d_mid_s_, d_down_, d_result_, scratch_;
    Buffer d_layer_tables_, d_expert_slot_;
    std::array<uint32_t, r3::moe_shm::kSmallClasses> class_rows_{};   /* enabled small-lane (pre-queued) step sizes */
    uint32_t class_count_ = 0;
public:
    /* Engine tensor-parallel ranks (R3_MOE_WORKER_ENGINE_RANKS): 2 = two XTX ranks, each asking for its TP
     * slice; 1 = one engine (the R9700, RAD02), asking for the full expert width -- both slices, summed. */
    uint32_t engine_ranks_ = r3::moe_shm::kMaxRanks;
private:
    /* FULL WIDTH: a one-rank engine wants each expert whole, so the helper stores it whole (one copy, the
     * relayout's own form) and runs one GEMM chain at the full 640 intermediate width -- instead of two TP
     * halves run one after the other and summed. A two-rank engine keeps one stored slice per rank. */
    bool full_width() const { return engine_ranks_ == 1; }
    uint32_t stored_slices() const { return full_width() ? 1u : r3::moe_shm::kMaxRanks; }
    int64_t slice_lo(uint32_t s, uint32_t parity) const { return full_width() ? 0 : rank_slice_lo(s, parity); }
    int64_t slice_w(uint32_t s, uint32_t parity) const { return full_width() ? (int64_t)shape_.intermediate : rank_slice_width(s, parity); }
    int64_t mid_w() const { return full_width() ? (int64_t)shape_.intermediate : kRankWidth; }
    uint32_t large_rows_ = 0;   /* large-lane capacity */
    uint32_t poison_ = 0;
    bool gemv_ = true;          /* R3_MOE_WORKER_GEMV: decode GEMV on the small lane (RAD05); 0 = off */
    uint32_t gemv_large_rows_ = 0; /* R3_MOE_WORKER_GEMV_LARGE_ROWS: GEMV for large-lane requests this small */
    bool rows_ = false;         /* R3_MOE_WORKER_ROWS: few-rows-a-expert kernel on the large lane (RAD06) */
    bool gemv_debug_ = false;   /* R3_MOE_WORKER_GEMV=2: grouped GEMM AND GEMV gate/up, compared (--compare) */

    template <class T> bool symbol(const char* name, T* out, bool required = true) {
        dlerror();
        void* p = dlsym(so_, name);
        const char* e = dlerror();
        if (!p || e) {
            if (required) std::fprintf(stderr, "r3_moe_worker: libr4d missing %s: %s\n",
                                       name, e ? e : "null");
            return !required;
        }
        std::memcpy(out, &p, sizeof(p));
        return true;
    }

    std::string plugin_path() const {
        if (const char* e = std::getenv("R3_MOE_WORKER_PLUGIN"); e && *e) return e;
        const std::string so = arch_ == "gfx1100" ? "libr3.so" : "libr4d.so";
        if (const char* e = std::getenv("RADIANCE_HOME"); e && *e) {
            std::string h(e);
            const size_t colon = h.find(':');
            if (colon != std::string::npos) h.resize(colon);
            return h + "/kernels/" + so;
        }
#ifdef RAD_DEFAULT_HOME
        return std::string(RAD_DEFAULT_HOME) + "/kernels/" + so;
#else
        return so;
#endif
    }

    bool load_plugin() {
        const std::string path = plugin_path();
        so_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!so_) {
            std::fprintf(stderr, "r3_moe_worker: dlopen %s: %s\n", path.c_str(), dlerror());
            return false;
        }
        uint32_t (*abi)() = nullptr;
        const RadPluginInfo* (*info)() = nullptr;
        int (*count)() = nullptr;
        const RadKernelInfo* (*at)(int) = nullptr;
        int (*open_fn)() = nullptr;
        if (!symbol("rad_plugin_abi_version", &abi) || !symbol("rad_plugin_info", &info) ||
            !symbol("rad_kernel_count", &count) || !symbol("rad_kernel_at", &at)) return false;
        /* rad_plugin_open/close are optional in the ABI (rad_abi.h); libr4d exports neither. */
        (void)symbol("rad_plugin_open", &open_fn, false);
        (void)symbol("rad_plugin_close", &plugin_close_, false);
        if (abi() != RAD_ABI_VERSION) {
            std::fprintf(stderr, "r3_moe_worker: libr4d ABI %u, helper ABI %u\n",
                         abi(), (unsigned)RAD_ABI_VERSION);
            return false;
        }
        const RadPluginInfo* inf = info();
        if (!inf || !inf->build_target || !std::strstr(inf->build_target, arch_.c_str())) {
            std::fprintf(stderr, "r3_moe_worker: %s is not a %s plugin\n", path.c_str(), arch_.c_str());
            return false;
        }
        if (open_fn && open_fn() != RAD_OK) {
            std::fprintf(stderr, "r3_moe_worker: libr4d rad_plugin_open refused this device\n");
            return false;
        }
        plugin_opened_ = true;
        auto find = [&](const char* op, const char* preferred) -> const RadKernelInfo* {
            const RadKernelInfo* fallback = nullptr;
            for (int i = 0; i < count(); ++i) {
                const RadKernelInfo* r = at(i);
                if (!r || !r->op || std::strcmp(r->op, op) != 0) continue;
                if (!fallback) fallback = r;
                if (preferred && r->name && std::strcmp(r->name, preferred) == 0) return r;
            }
            return fallback;
        };
        moe_ = find("moe_gemm_q", "moe_gemm_w4a8");
        had_quant_ = find("had_quant_act_fp8", "had_quant_act_fp8");
        gated_quant_ = find("gated_had_quant_fp8", "gated_had_quant_fp8");
        gather_ = find("moe_gather", "moe_gather_bf16");
        if (const RadKernelInfo* r = find("moe_gemm_q", "moe_gemm_fp8a8"); r && r->name && std::strcmp(r->name, "moe_gemm_fp8a8") == 0)
            moe_fp8_ = r;
        quant_plain_ = find("quant_act_fp8", "quant_act_fp8");
        gated_plain_ = find("gated_quant_fp8", "gated_quant_fp8");
        if (!moe_ || !moe_->layout || !moe_->relayout || !had_quant_ || !gated_quant_ || !gather_) {
            std::fprintf(stderr, "r3_moe_worker: installed libr4d lacks required MoE/quant rows\n");
            return false;
        }
        /* R3_MOE_WORKER_MOE_FORM (RAD06): pin the grouped GEMM's tile form - 1 the 32-row prefill form, 2 the
         * 16-row (tile, run) decode form. libr4d picks by T against all 512 experts, but a helper's rows spread over
         * only the experts it owns (~4.5 rows an expert at prefill), the decode form's regime. */
        if (const char* f = std::getenv("R3_MOE_WORKER_MOE_FORM"); f && *f && std::strcmp(f, "0") != 0) {
            auto pin = reinterpret_cast<void (*)(int)>(dlsym(so_, "r4d_moe_gemm_pin_prefill"));
            const int form = std::atoi(f);
            if (!pin || (form != 1 && form != 2)) {
                std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_MOE_FORM=%s not applied (%s)\n", f,
                             pin ? "want 1 or 2" : "plugin does not export r4d_moe_gemm_pin_prefill");
                return false;
            }
            pin(form);
            std::fprintf(stderr, "r3_moe_worker: grouped MoE GEMM pinned to the %s form\n",
                         form == 2 ? "16-row decode" : "32-row prefill");
        }
        /* RAD06 tile sweep on gfx1100 (libr4d's tiles were tuned on gfx1201): R3_MOE_WORKER_MOE_DOWN_TILE 1 the
         * 64/128-column down tile, 2 the 256-column 16-wave one. */
        if (const char* v = std::getenv("R3_MOE_WORKER_MOE_DOWN_TILE"); v && *v) {
            auto fn = reinterpret_cast<void (*)(int)>(dlsym(so_, "r4d_moe_gemm_pin_down_tile"));
            if (!fn) {
                std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_MOE_DOWN_TILE set but the plugin lacks r4d_moe_gemm_pin_down_tile\n");
                return false;
            }
            fn(std::atoi(v));
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_MOE_DOWN_TILE=%s applied\n", v);
        }
        /* RAD06: runs of this many rows or more take 64-row slices and the MT=4 tile (libr3 native/r4d_moe.hip.rw). */
        if (const char* v = std::getenv("R3_MOE_WORKER_MOE_LONG_ROWS"); v && *v && std::strcmp(v, "0") != 0) {
            auto fn = reinterpret_cast<void (*)(int)>(dlsym(so_, "r4d_moe_gemm_pin_long_rows"));
            if (!fn) {
                std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_MOE_LONG_ROWS set but the plugin lacks r4d_moe_gemm_pin_long_rows\n");
                return false;
            }
            fn(std::atoi(v));
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_MOE_LONG_ROWS=%s applied\n", v);
        }
        return true;
    }

    static size_t arena_round(int64_t n) { return ((size_t)(n > 0 ? n : 1) + 255u) & ~(size_t)255u; }

    bool slice_bytes(const RadFileEntry* e, bool gate_up, uint32_t expert_slot, uint32_t rank,
                     int64_t* code, int64_t* scale) const {
        if (!e || !code || !scale || rank >= r3::moe_shm::kMaxRanks) return false;
        int64_t nr = 1, nc = 1;
        rad_enc_view(e->rank, e->shape, &nr, &nc);
        const int64_t width = slice_w(rank, expert_slot & 1u);
        if (weight_format_ == MOE_FMT_FP8_E4M3_BLOCK) {   // one byte a weight, one bf16 scale a 128x128 tile
            const int64_t rows = gate_up ? 2 * width : nr, cols = gate_up ? nc : width;
            if (gate_up ? (nr != 2 * (int64_t)shape_.intermediate || nc != shape_.hidden)
                        : (nr != shape_.hidden || nc != (int64_t)shape_.intermediate)) return false;
            *code = rows * cols;
            *scale = ((rows + 127) / 128) * ((cols + 127) / 128) * 2;
            return true;
        }
        if (gate_up) {
            if (nr != 2 * (int64_t)shape_.intermediate || nc != shape_.hidden) return false;
            *code = (2 * width) * nc / 2;
            *scale = (2 * width) * (nc / 64);
        } else {
            if (nr != shape_.hidden || nc != (int64_t)shape_.intermediate) return false;
            *code = nr * width / 2;
            *scale = nr * (width / 64);
        }
        return true;
    }

    void* arena_take(size_t n) {
        const size_t r = arena_round((int64_t)n);
        if (!arena_ || arena_used_ + r > arena_size_) {
            std::fprintf(stderr, "r3_moe_worker: resident expert arena overflow\n");
            return nullptr;
        }
        void* p = (uint8_t*)arena_ + arena_used_;
        arena_used_ += r;
        return p;
    }

    void* alloc_persistent(size_t n) {
        void* p = nullptr;
        if (!hip_ok(hipMalloc(&p, n ? n : 16), "hipMalloc resident expert")) return nullptr;
        persistent_.push_back(p);
        return p;
    }

    bool canonical_pair(const RadFileEntry* e, const char* role0, int level0,
                        const char* role1, int level1,
                        std::array<int,2>* sel, std::array<RadTensor,2>* out) const {
        const RadEncoding* enc = model_.encoding(e);
        int s[2] = {-1, -1};
        s[0] = level0 >= 0 ? rad_enc_find_level(enc, role0, level0) : rad_enc_find(enc, role0);
        s[1] = level1 >= 0 ? rad_enc_find_level(enc, role1, level1) : rad_enc_find(enc, role1);
        if (s[0] < 0 || s[1] < 0) return false;
        int64_t rows = 1, cols = 1;
        rad_enc_view(e->rank, e->shape, &rows, &cols);
        for (int i = 0; i < 2; ++i) {
            const RadEncPlane& ep = enc->plane[s[i]];
            int64_t pr = 0, pc = 0;
            rad_enc_plane_dims(&ep, rows, cols, &pr, &pc);
            const RadFilePlane* fp = model_.plane(e, s[i]);
            (*sel)[i] = s[i];
            (*out)[i] = tensor(nullptr, ep.dtype, {pr, pc});
        }
        return true;
    }

    std::vector<RadParam> layout_params(const RadFileEntry* e, bool gate_up) const {
        int64_t nr = 1, nc = 1;
        rad_enc_view(e->rank, e->shape, &nr, &nc);
        return {
            pi("M", 1), pi("N", nr), pi("K", nc), pi("N_odd", nr), pi("K_odd", nc),
            pi("parts", gate_up ? 2 : 1), pi("n_expert", experts()),
            pi("top_k", shape_.top_k), pi("group", kGroup),
            ps("a_order", gate_up ? "token" : "sorted"), ps("dtype", dtype_)
        };
    }

    bool plan_weight(const RadFileEntry* e, bool gate_up, WeightPlan* out) {
        if (!e || !out || e->enc >= model_.h->enc_count) return false;
        const RadEncoding* enc = model_.encoding(e);
        if (!enc || !rad_enc_valid(enc)) return false;
        out->params = layout_params(e, gate_up);
        if (weight_format_ == MOE_FMT_FP8_E4M3_BLOCK) {   // E4M3 codes and one bf16 scale plane, in the layout the kernel reads
            const int c = rad_enc_find(enc, "codes"), sc = rad_enc_find_level(enc, "scale", 0);
            if (c < 0 || sc < 0) return false;
            /* the identity layout is right only if the planes ARE the kernel's: E4M3 codes one byte a weight, a bf16 scale a 128x128 tile,
             * nothing else (no zero point, table, transform or second scale level) */
            if (enc->n_planes != 2 || enc->plane[c].dtype != RAD_F8E4M3 || enc->plane[c].block[1] != 1 ||
                enc->plane[sc].dtype != RAD_BF16 || enc->plane[sc].kind != RAD_PLANE_TILED || enc->plane[sc].block[0] != 128 ||
                enc->plane[sc].block[1] != 128 || std::strlen(enc->transform) != 0) return false;
            int64_t rows = 1, cols = 1;
            rad_enc_view(e->rank, e->shape, &rows, &cols);
            int64_t pr = 0, pc = 0;
            rad_enc_plane_dims(&enc->plane[c], rows, cols, &pr, &pc);
            out->code_sel = {c, -1};
            out->code_planes[0] = tensor(nullptr, enc->plane[c].dtype, {pr, pc});
            rad_enc_plane_dims(&enc->plane[sc], rows, cols, &pr, &pc);
            out->scale_sel = {sc, -1};
            out->scale_planes[0] = tensor(nullptr, enc->plane[sc].dtype, {pr, pc});
            /* The row-major fp8a8 hook checks the planes (dtype, 128x128 tiles, N/K) and then answers RAD_E_UNSUPPORTED on purpose:
             * the kernel reads the checkpoint's own bytes, so there is no layout to compute. Any other answer is a refusal. */
            const int f0 = moe_->layout(out->params.data(), (int)out->params.size(), 2, enc, out->code_sel.data(),
                                        out->code_planes.data(), 1, &out->code);
            const int f1 = moe_->layout(out->params.data(), (int)out->params.size(), 3, enc, out->scale_sel.data(),
                                        out->scale_planes.data(), 1, &out->scale);
            if ((f0 != RAD_OK && f0 != RAD_E_UNSUPPORTED) || (f1 != RAD_OK && f1 != RAD_E_UNSUPPORTED)) return false;
            auto as_stored = [&](RadLayout* l, const RadTensor& pl) {
                std::memset(l, 0, sizeof *l);
                l->dtype = pl.dtype;
                l->rank = 2;
                l->shape[0] = pl.shape[0];
                l->shape[1] = pl.shape[1];
                l->align = 16;
                l->bytes = pl.shape[0] * rad_enc_row_bytes(pl.dtype, pl.shape[1]);
            };
            if (f0 == RAD_E_UNSUPPORTED) as_stored(&out->code, out->code_planes[0]);
            if (f1 == RAD_E_UNSUPPORTED) as_stored(&out->scale, out->scale_planes[0]);
            return out->code.bytes > 0 && out->scale.bytes > 0;
        }
        if (!canonical_pair(e, "codes", -1, "table", -1, &out->code_sel, &out->code_planes) ||
            !canonical_pair(e, "scale", 0, "scale", 1, &out->scale_sel, &out->scale_planes))
            return false;
        const int rc0 = moe_->layout(out->params.data(), (int)out->params.size(), 2, enc,
                                     out->code_sel.data(), out->code_planes.data(), 2, &out->code);
        const int rc1 = moe_->layout(out->params.data(), (int)out->params.size(), 3, enc,
                                     out->scale_sel.data(), out->scale_planes.data(), 2, &out->scale);
        return rc0 == RAD_OK && rc1 == RAD_OK && out->code.bytes > 0 && out->scale.bytes > 0;
    }

    bool load_plane_data(const RadFileEntry* e, int sel, RadTensor* t, std::vector<uint8_t>* out) {
        if (!e || !t || !out || sel < 0 || (uint32_t)sel >= e->n_planes) return false;
        const RadFilePlane* fp = model_.plane(e, sel);
        out->resize((size_t)fp->bytes);
        size_t done = 0;
        while (done < out->size()) {
            const size_t n = std::min(load_bounce_.bytes, out->size() - done);
            if (!model_.pread_full(fp->offset + done, load_bounce_.p, n)) return false;
            std::memcpy(out->data() + done, load_bounce_.p, n);
            done += n;
        }
        t->data = out->data();
        return true;
    }

    bool upload_weight(const RadFileEntry* e, bool gate_up, uint32_t expert_slot,
                       std::array<StoredWeight, r3::moe_shm::kMaxRanks>* out) {
        WeightPlan p;
        if (!plan_weight(e, gate_up, &p)) {
            std::fprintf(stderr, "r3_moe_worker: expert weight is not %s-compatible\n", dtype_);
            return false;
        }
        int64_t nr = 1, nc = 1;
        rad_enc_view(e->rank, e->shape, &nr, &nc);
        if (weight_format_ == MOE_FMT_FP8_E4M3_BLOCK) {
            /* the container's planes ARE the kernel's layout (fp8a8's hook is the checkpoint's own row-major bytes): no relayout */
            std::vector<uint8_t> code, scale;
            RadTensor ct = p.code_planes[0], st = p.scale_planes[0];
            if (!load_plane_data(e, p.code_sel[0], &ct, &code) || !load_plane_data(e, p.scale_sel[0], &st, &scale)) {
                std::fprintf(stderr, "r3_moe_worker: pread expert plane failed\n");
                return false;
            }
            int64_t cb = 0, sb = 0;
            if (!slice_bytes(e, gate_up, expert_slot, 0, &cb, &sb) || (int64_t)code.size() != cb || (int64_t)scale.size() != sb ||
                p.code.bytes != cb || p.scale.bytes != sb) {
                std::fprintf(stderr, "r3_moe_worker: unexpected fp8a8 plane geometry for %s\n", gate_up ? "gate/up" : "down");
                return false;
            }
            StoredWeight& w = (*out)[0];
            w.code = arena_take(code.size());
            w.scale = arena_take(scale.size());
            if (!w.code || !w.scale ||
                !hip_ok(hipMemcpy(w.code, code.data(), code.size(), hipMemcpyHostToDevice), "upload expert codes") ||
                !hip_ok(hipMemcpy(w.scale, scale.data(), scale.size(), hipMemcpyHostToDevice), "upload expert scales"))
                return false;
            w.code_dtype = p.code.dtype;
            w.scale_dtype = p.scale.dtype;
            w.code_bytes = cb;
            w.scale_bytes = sb;
            uint32_t& cd = gate_up ? gu_code_dtype_ : dn_code_dtype_;
            uint32_t& sd = gate_up ? gu_scale_dtype_ : dn_scale_dtype_;
            if ((cd && cd != w.code_dtype) || (sd && sd != w.scale_dtype)) {
                std::fprintf(stderr, "r3_moe_worker: owned expert stored layouts disagree\n");
                return false;
            }
            cd = w.code_dtype;
            sd = w.scale_dtype;
            return true;
        }
        if (p.code.bytes != nr * nc / 2 || p.scale.bytes != nr * (nc / 64)) {
            std::fprintf(stderr, "r3_moe_worker: unexpected %s relayout geometry\n",
                         gate_up ? "gate/up" : "down");
            return false;
        }

        std::array<std::vector<uint8_t>, 2> code_src, scale_src;
        auto code_planes = p.code_planes;
        auto scale_planes = p.scale_planes;
        for (int i = 0; i < 2; ++i) {
            if (!load_plane_data(e, p.code_sel[i], &code_planes[i], &code_src[i]) ||
                !load_plane_data(e, p.scale_sel[i], &scale_planes[i], &scale_src[i])) {
                std::fprintf(stderr, "r3_moe_worker: pread expert plane failed\n");
                return false;
            }
        }
        const RadEncoding* enc = model_.encoding(e);
        std::vector<uint8_t> full_code((size_t)p.code.bytes);
        std::vector<uint8_t> full_scale((size_t)p.scale.bytes);
        if (moe_->relayout(p.params.data(), (int)p.params.size(), 2, enc,
                           p.code_sel.data(), code_planes.data(), 2,
                           full_code.data(), p.code.bytes) != RAD_OK ||
            moe_->relayout(p.params.data(), (int)p.params.size(), 3, enc,
                           p.scale_sel.data(), scale_planes.data(), 2,
                           full_scale.data(), p.scale.bytes) != RAD_OK) {
            std::fprintf(stderr, "r3_moe_worker: libr4d refused expert relayout\n");
            return false;
        }

        for (uint32_t rank = 0; rank < stored_slices(); ++rank) {
            const int64_t lo = slice_lo(rank, expert_slot & 1u);
            const int64_t width = slice_w(rank, expert_slot & 1u);
            int64_t cb = 0, sb = 0;
            if (!slice_bytes(e, gate_up, expert_slot, rank, &cb, &sb)) return false;
            std::vector<uint8_t> code((size_t)cb);
            std::vector<uint8_t> scale((size_t)sb);
            if (gate_up) {
                const size_t crb = (size_t)nc / 2;
                const size_t srb = (size_t)nc / 64;
                for (int part = 0; part < 2; ++part) {
                    const int64_t src0 = part * (int64_t)shape_.intermediate + lo;
                    const int64_t dst0 = part * width;
                    std::memcpy(code.data() + (size_t)dst0 * crb,
                                full_code.data() + (size_t)src0 * crb,
                                (size_t)width * crb);
                    std::memcpy(scale.data() + (size_t)dst0 * srb,
                                full_scale.data() + (size_t)src0 * srb,
                                (size_t)width * srb);
                }
            } else {
                const size_t fcrb = (size_t)nc / 2;
                const size_t fsrb = (size_t)nc / 64;
                const size_t crb = (size_t)width / 2;
                const size_t srb = (size_t)width / 64;
                for (int64_t row = 0; row < nr; ++row) {
                    std::memcpy(code.data() + (size_t)row * crb,
                                full_code.data() + (size_t)row * fcrb + (size_t)lo / 2,
                                crb);
                    std::memcpy(scale.data() + (size_t)row * srb,
                                full_scale.data() + (size_t)row * fsrb + (size_t)lo / 64,
                                srb);
                }
            }

            StoredWeight& w = (*out)[rank];
            w.code = arena_take(code.size());
            w.scale = arena_take(scale.size());
            if (!w.code || !w.scale ||
                !hip_ok(hipMemcpy(w.code, code.data(), code.size(), hipMemcpyHostToDevice),
                        "upload sliced expert codes") ||
                !hip_ok(hipMemcpy(w.scale, scale.data(), scale.size(), hipMemcpyHostToDevice),
                        "upload sliced expert scales"))
                return false;
            w.code_dtype = p.code.dtype;
            w.scale_dtype = p.scale.dtype;
            w.code_bytes = cb;
            w.scale_bytes = sb;
            uint32_t& cd = gate_up ? gu_code_dtype_ : dn_code_dtype_;
            uint32_t& sd = gate_up ? gu_scale_dtype_ : dn_scale_dtype_;
            if ((cd && cd != w.code_dtype) || (sd && sd != w.scale_dtype)) {
                std::fprintf(stderr, "r3_moe_worker: owned expert stored layouts disagree\n");
                return false;
            }
            cd = w.code_dtype;
            sd = w.scale_dtype;
        }
        return true;
    }

    bool build_one_table(const std::vector<void*>& host, void** dev) {
        *dev = alloc_persistent(host.size() * sizeof(void*));
        return *dev && hip_ok(hipMemcpy(*dev, host.data(), host.size() * sizeof(void*),
                                        hipMemcpyHostToDevice), "upload expert pointer table");
    }

    bool build_layer_table(uint32_t l, const int32_t* slot_of_expert) {
        using namespace r3::moe_shm;
        if (l >= model_.layers || !slot_of_expert) return false;
        const size_t nclass = experts() / 2;
        std::vector<uint16_t> slot_map(experts(), 0);
        std::fill(slot_map.begin(), slot_map.end(), (uint16_t)0xffffu);
        for (uint32_t e = 0; e < experts(); ++e) {
            const int32_t es = slot_of_expert[e];
            if (es >= 0) slot_map[e] = (uint16_t)es;
        }

        LayerTables& lt = tables_[l];
        lt.gu_code_dtype = gu_code_dtype_; lt.gu_scale_dtype = gu_scale_dtype_;
        lt.dn_code_dtype = dn_code_dtype_; lt.dn_scale_dtype = dn_scale_dtype_;
        for (uint32_t rank = 0; rank < stored_slices(); ++rank) {
            for (uint32_t parity = 0; parity < 2; ++parity) {
                std::vector<void*> guc(nclass, nullptr), gus(nclass, nullptr);
                std::vector<void*> dnc(nclass, nullptr), dns(nclass, nullptr);
                for (uint32_t e = 0; e < experts(); ++e) {
                    const int32_t es = slot_of_expert[e];
                    if (es < 0 || ((uint32_t)es & 1u) != parity) continue;
                    const Expert& x = experts_[(size_t)l * experts() + e];
                    if (!x.live) continue;
                    const size_t ix = (size_t)es / 2;
                    guc[ix] = x.gu[rank].code; gus[ix] = x.gu[rank].scale;
                    dnc[ix] = x.dn[rank].code; dns[ix] = x.dn[rank].scale;
                }
                if (!build_one_table(guc, &lt.gu_code[rank][parity]) ||
                    !build_one_table(gus, &lt.gu_scale[rank][parity]) ||
                    !build_one_table(dnc, &lt.dn_code[rank][parity]) ||
                    !build_one_table(dns, &lt.dn_scale[rank][parity]))
                    return false;
            }
        }

        const size_t slot_off = (size_t)l * experts() * sizeof(uint16_t);
        if (!hip_ok(hipMemcpy(static_cast<uint8_t*>(d_expert_slot_.p) + slot_off,
                              slot_map.data(), slot_map.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
                    "upload runtime expert slot map"))
            return false;

        DeviceLayerTables dt{};
        for (uint32_t rank = 0; rank < stored_slices(); ++rank) {
            for (uint32_t parity = 0; parity < 2; ++parity) {
                dt.gu_code[rank][parity] = lt.gu_code[rank][parity];
                dt.gu_scale[rank][parity] = lt.gu_scale[rank][parity];
                dt.dn_code[rank][parity] = lt.dn_code[rank][parity];
                dt.dn_scale[rank][parity] = lt.dn_scale[rank][parity];
            }
        }
        dt.expert_slot = static_cast<const uint16_t*>(d_expert_slot_.p) + (size_t)l * experts();
        return hip_ok(hipMemcpy(static_cast<uint8_t*>(d_layer_tables_.p) +
                                    (size_t)l * sizeof(DeviceLayerTables),
                                &dt, sizeof(dt), hipMemcpyHostToDevice),
                      "upload runtime layer table index");
    }

    bool owns(uint32_t layer, uint32_t expert) const {
        return layer < model_.layers && expert < experts() &&
               experts_[(size_t)layer * experts() + expert].live;
    }

    /* The scale table of a weight of logical dims [n, k]: W4 has one E4M3 scale per 64 columns of every row; FP8 block-scaled has one
     * bf16 scale per 128x128 tile (the checkpoint's own plane), so its scale tensor is [ceil(n/128), ceil(k/128)]. */
    RadTensor table_scale_tensor(void* table, uint32_t dtype, int64_t n, int64_t k) const {
        if (weight_format_ == MOE_FMT_FP8_E4M3_BLOCK) return table_tensor(table, dtype, (n + 127) / 128, (k + 127) / 128);
        return table_tensor(table, dtype, n, k / 64);
    }
    RadTensor table_tensor(void* table, uint32_t dtype, int64_t n, int64_t k) const {
        RadTensor t = tensor(table, dtype, {experts() / 2, n, k});
        t.stride[0] = 0; /* ABI table form: data points to a device array of per-entry pointers. */
        return t;
    }

    std::vector<RadParam> gemm_params(uint32_t rows, int64_t n, int64_t k,
                                      int64_t parts, const char* order,
                                      int64_t n_odd = -1, int64_t k_odd = -1) const {
        if (n_odd < 0) n_odd = n;
        if (k_odd < 0) k_odd = k;
        return {
            pi("M", rows), pi("N", n), pi("K", k), pi("N_odd", n_odd), pi("K_odd", k_odd),
            pi("parts", parts), pi("n_expert", experts()),
            pi("top_k", shape_.top_k), pi("group", kGroup),
            ps("a_order", order), ps("dtype", dtype_)
        };
    }

    bool scratch_need(const RadKernelInfo* r, std::vector<RadTensor> ts,
                      std::vector<RadParam> ps, size_t* need) {
        if (!r || !need) return false;
        if (!r->scratch) return true;
        RadArgs a{};
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.rank = 0; a.world_size = 1;
        const int64_t n = r->scratch(&a);
        if (n < 0) return false;
        *need = std::max(*need, (size_t)std::max<int64_t>(n, 1));
        return true;
    }

    bool prepare_lane_scratch(LaneWork& w, uint32_t rank, uint32_t rows, uint32_t routes) {
        using namespace r3::moe_shm;
        size_t need = 1;
        std::vector<RadParam> aqp = {
            pi("M", rows), pi("n", shape_.hidden), pi("group", kGroup), ps("dtype", "bf16")
        };
        if (!scratch_need(had_quant_,
                          { tensor(w.d_act.p, RAD_BF16, {rows, shape_.hidden}),
                            tensor(w.d_aq.p, RAD_F8E4M3, {rows, shape_.hidden}),
                            tensor(w.d_as.p, RAD_F32, {rows, shape_.hidden / kGroup}) },
                          aqp, &need))
            return false;

        const int64_t w0 = slice_w(full_width() ? 0u : rank, 0);
        const int64_t w1 = slice_w(full_width() ? 0u : rank, 1);
        std::vector<RadParam> gp =
            gemm_params(rows, 2 * w0, shape_.hidden, 2, "token", 2 * w1, shape_.hidden);
        if (!scratch_need(moe_,
                          { tensor(w.d_aq.p, RAD_F8E4M3, {rows, shape_.hidden}),
                            tensor(w.d_as.p, RAD_F32, {rows, shape_.hidden / kGroup}),
                            table_tensor(w.gu_code[0].p, gu_code_dtype_, 2 * w0, shape_.hidden),
                            table_scale_tensor(w.gu_scale[0].p, gu_scale_dtype_, 2 * w0, shape_.hidden),
                            tensor(w.d_sorted.p, RAD_I32, {routes}),
                            tensor(w.d_off.p, RAD_I32, {experts() + 1}),
                            table_tensor(w.gu_code[1].p, gu_code_dtype_, 2 * w1, shape_.hidden),
                            table_scale_tensor(w.gu_scale[1].p, gu_scale_dtype_, 2 * w1, shape_.hidden),
                            tensor(w.d_gu.p, RAD_BF16, {routes, 2 * mid_w()}) },
                          gp, &need))
            return false;

        std::vector<RadParam> gqp = {
            pi("M", routes), pi("n", mid_w()), pi("group", kGroup),
            ps("dtype", "bf16"), ps("act", "silu")
        };
        if (!scratch_need(gated_quant_,
                          { tensor(w.d_gu.p, RAD_BF16, {routes, 2 * mid_w()}),
                            tensor(w.d_mid.p, RAD_F8E4M3, {routes, mid_w()}),
                            tensor(w.d_mid_s.p, RAD_F32, {routes, mid_w() / kGroup}),
                            absent() },
                          gqp, &need))
            return false;

        std::vector<RadParam> dp =
            gemm_params(rows, shape_.hidden, w0, 1, "sorted", shape_.hidden, w1);
        if (!scratch_need(moe_,
                          { tensor(w.d_mid.p, RAD_F8E4M3, {routes, mid_w()}),
                            tensor(w.d_mid_s.p, RAD_F32, {routes, mid_w() / kGroup}),
                            table_tensor(w.dn_code[0].p, dn_code_dtype_, shape_.hidden, w0),
                            table_scale_tensor(w.dn_scale[0].p, dn_scale_dtype_, shape_.hidden, w0),
                            tensor(w.d_sorted.p, RAD_I32, {routes}),
                            tensor(w.d_off.p, RAD_I32, {experts() + 1}),
                            table_tensor(w.dn_code[1].p, dn_code_dtype_, shape_.hidden, w1),
                            table_scale_tensor(w.dn_scale[1].p, dn_scale_dtype_, shape_.hidden, w1),
                            tensor(w.d_down.p, RAD_BF16, {routes, shape_.hidden}) },
                          dp, &need))
            return false;

        std::vector<RadParam> mp = {
            pi("M", rows), pi("n", shape_.hidden), pi("top_k", shape_.top_k), ps("dtype", "bf16")
        };
        if (!scratch_need(gather_,
                          { tensor(w.d_down.p, RAD_BF16, {routes, shape_.hidden}),
                            tensor(w.d_ew.p, RAD_BF16, {rows, shape_.top_k}),
                            tensor(w.d_sorted.p, RAD_I32, {routes}),
                            absent(), absent(), tensor(w.d_act.p, RAD_BF16, {rows, shape_.hidden}) },
                          mp, &need))
            return false;
        return w.scratch.ensure(need);
    }

    int launch_on(const RadKernelInfo* r, std::vector<RadTensor> ts, std::vector<RadParam> ps,
                  hipStream_t stream, Buffer& scratch) {
        if (!r || !r->launch) return RAD_E_UNSUPPORTED;
        RadArgs a{};
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.rank = 0; a.world_size = 1;
        if (r->scratch) {
            const int64_t need = r->scratch(&a);
            if (need < 0 || !scratch.ensure((size_t)std::max<int64_t>(need, 1))) return RAD_E_STATE;
            a.scratch = scratch.p; a.scratch_bytes = (int64_t)scratch.bytes;
        }
        const int rc = r->launch(&a, (RadStream)stream);
        if (rc != RAD_OK)
            std::fprintf(stderr, "r3_moe_worker: libr4d %s launch rc=%d\n", r->name, rc);
        return rc;
    }

    int launch(const RadKernelInfo* r, std::vector<RadTensor> ts, std::vector<RadParam> ps) {
        return launch_on(r, std::move(ts), std::move(ps), stream_, scratch_);
    }

    bool ensure_lane_work(LaneWork& w, uint32_t rows, uint32_t routes, uint32_t tr) {
        return w.d_counts.ensure((size_t)experts() * 4) &&
               w.d_cursor.ensure((size_t)experts() * 4) &&
               w.d_act.ensure((size_t)rows * shape_.hidden * 2) &&
               w.d_flat.ensure((size_t)routes * 4) &&
               w.d_sorted.ensure((size_t)tr * 4) &&
               w.d_off.ensure((experts() + 1ull) * 4) &&
               w.d_ew.ensure((size_t)tr * 2) &&
               w.d_inv.ensure((size_t)tr * 4) &&
               w.d_aq.ensure((size_t)rows * shape_.hidden) &&
               w.d_as.ensure((size_t)rows * (shape_.hidden / kGroup) * 4) &&
               w.d_gu.ensure((size_t)tr * 2 * mid_w() * 2) &&
               w.d_mid.ensure((size_t)tr * mid_w()) &&
               w.d_mid_s.ensure((size_t)tr * (mid_w() / kGroup) * 4) &&
               w.d_down.ensure((size_t)tr * shape_.hidden * 2) &&
               w.d_dbg.ensure((size_t)tr * 2 * mid_w() * 2);
    }

    bool ensure_work(uint32_t rows, uint32_t tr) {
        return d_act_.ensure((size_t)rows * shape_.hidden * 2) &&
               d_aq_.ensure((size_t)rows * shape_.hidden) &&
               d_as_.ensure((size_t)rows * (shape_.hidden / kGroup) * 4) &&
               d_sorted_.ensure((size_t)tr * 4) &&
               d_off_.ensure((experts() + 1ull) * 4) &&
               d_ew_.ensure((size_t)rows * shape_.top_k * 2) &&
               d_gu_.ensure((size_t)tr * kRankGateUpWidth * 2) &&
               d_mid_.ensure((size_t)tr * kRankWidth) &&
               d_mid_s_.ensure((size_t)tr * (kRankWidth / kGroup) * 4) &&
               d_down_.ensure((size_t)tr * shape_.hidden * 2) &&
               d_result_.ensure((size_t)rows * shape_.hidden * 2);
    }
};

} // namespace r3_moe_gpu
