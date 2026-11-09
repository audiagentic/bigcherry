#pragma once
#include "worker_plugin_config.h"

#include "moe_lane_common.h"
#include "moe_lane_front.h"
#include "moe_lane_gemm.h"
#include "moe_lane_rows_stock_replay.h"  // diagnostic-only baseline for --replay-sweep
#include "rad46_rows_policy.h"
#include "moe_lane_quant.h"
#include "moe_lane_reduce.h"
#include "moe_lane_host.h"
#include "worker_replay_memory.h"
#include "worker_footprint.h"
#include "worker_slot_pool.h"
#include "worker_move_slots.h"
#include "worker_generation_tables.h"
#include "worker_footprint.h"
#include <memory>
#include "rad_abi_layout.h"

namespace r3_moe_gpu {
/* The quad-DPP rows kernel with the replay-selected W4 decoder (quad_perm_, RAD49): a statement, usable in an else-if chain. */
#define R3_QUADDPP_LAUNCH(T, RB, ...) do { if (quad_perm_) hipLaunchKernelGGL((lane_moe_rows_w4_quaddpp<T, RB, true>), __VA_ARGS__); \
    else hipLaunchKernelGGL((lane_moe_rows_w4_quaddpp<T, RB, false>), __VA_ARGS__); } while (0)

class Runner : public r3::worker_moves::Operations {
public:
    explicit Runner(const Container& model) : model_(model) {}
    /* The decode GEMV for the weight format in use (W4 nonlinear nibbles or FP8 block-scaled). */
    template <bool TOKEN>
    void launch_gemv(dim3 grid, hipStream_t stream, const uint8_t* aq, const float* as, const int32_t* sorted, const int32_t* off,
                     uint32_t top_k, void* const* code0, void* const* scale0, void* const* code1, void* const* scale1, uint32_t n0,
                     uint32_t n1, uint32_t k0, uint32_t k1, uint32_t a_stride, uint32_t out_stride, uint16_t* out, uint32_t part_pitch) {
        /* R3_MOE_WORKER_GEMV_W4H2 is read each call (as FUSEROUTE is) so a replay sweep can compare the two W4 kernels in one process. */
        const bool w4h2 = r3_serving::settings().gemv_w4h2;
        if (weight_format_ == MOE_FMT_W4NL64 && w4h2 && !(k0 & 63u) && !(k1 & 63u))
            hipLaunchKernelGGL((lane_moe_gemv_w4h2<TOKEN>), grid, dim3(256), 0, stream, aq, as, sorted, off, top_k, code0, scale0, code1,
                               scale1, n0, n1, k0, k1, a_stride, out_stride, out, part_pitch, lane_shape_);
        else if (weight_format_ == MOE_FMT_FP8_E4M3_BLOCK && gemv_mr_ && !(k0 & 15u) && !(k1 & 15u))
            hipLaunchKernelGGL((lane_moe_gemv_mr<TOKEN>), dim3(experts(), (std::max(n0, n1) + kMrTile - 1) / kMrTile), dim3(256), 0, stream, aq, as, sorted, off, top_k, code0,
                               scale0, code1, scale1, n0, n1, k0, k1, a_stride, out_stride, out, part_pitch);
        else if (weight_format_ == MOE_FMT_FP8_E4M3_BLOCK && gemv_h2_ == 4 && !(k0 & 15u) && !(k1 & 15u))
            hipLaunchKernelGGL((lane_moe_gemv_h2<TOKEN, 4>), grid, dim3(256), 0, stream, aq, as, sorted, off, top_k, code0, scale0, code1,
                               scale1, n0, n1, k0, k1, a_stride, out_stride, out, part_pitch, lane_shape_);
        else if (weight_format_ == MOE_FMT_FP8_E4M3_BLOCK && gemv_h2_ == 2 && !(k0 & 15u) && !(k1 & 15u))
            hipLaunchKernelGGL((lane_moe_gemv_h2<TOKEN, 2>), grid, dim3(256), 0, stream, aq, as, sorted, off, top_k, code0, scale0, code1,
                               scale1, n0, n1, k0, k1, a_stride, out_stride, out, part_pitch, lane_shape_);
        else if (weight_format_ == MOE_FMT_FP8_E4M3_BLOCK && gemv_h2_ == 1 && !(k0 & 15u) && !(k1 & 15u))
            hipLaunchKernelGGL((lane_moe_gemv_h2<TOKEN, 1>), grid, dim3(256), 0, stream, aq, as, sorted, off, top_k, code0, scale0, code1,
                               scale1, n0, n1, k0, k1, a_stride, out_stride, out, part_pitch, lane_shape_);
        else if (weight_format_ == MOE_FMT_FP8_E4M3_BLOCK && gemv_wide_ && !(k0 & 15u) && !(k1 & 15u))
            hipLaunchKernelGGL((lane_moe_gemv_wide<TOKEN>), grid, dim3(256), 0, stream, aq, as, sorted, off, top_k, code0, scale0, code1,
                               scale1, n0, n1, k0, k1, a_stride, out_stride, out, part_pitch, lane_shape_);
        else if (weight_format_ == MOE_FMT_FP8_E4M3_BLOCK)
            hipLaunchKernelGGL((lane_moe_gemv<TOKEN, true>), grid, dim3(256), 0, stream, aq, as, sorted, off, top_k, code0, scale0, code1,
                               scale1, n0, n1, k0, k1, a_stride, out_stride, out, part_pitch, lane_shape_);
        else
            hipLaunchKernelGGL((lane_moe_gemv<TOKEN, false>), grid, dim3(256), 0, stream, aq, as, sorted, off, top_k, code0, scale0, code1,
                               scale1, n0, n1, k0, k1, a_stride, out_stride, out, part_pitch, lane_shape_);
    }
    /* Routed experts per layer, from the container; and the thread count of the one-block routing kernels: a power of two, at least
     * kFuseMaxRoutes and the expert count (0 when the count is beyond what one block can scan). */
    uint32_t experts() const { return model_.experts; }
    uint32_t weight_format() const { return weight_format_; }
    /* The shape the kernels run with: hidden width, routes a token, activation group (RCS09-06). Checked against the capacities the
     * kernels are built for (per-token scratch, the GEMV's LDS row, whole 16-byte activation pieces) and copied to the device. */
    r3::moe_shm::LayoutShape shape_{};
    LaneShape lane_shape_{};              /* what the lane kernels are launched with (RCS05: an argument, not a device global) */
    /* The expert weight format the worker serves (the backend contract's MOE_FMT_*). W4NL64 is what open() prepared; FP8 block-scaled
     * switches to radiance's fp8a8 grouped GEMM and the plain quantisers, reads the container's planes as they are (that kernel's
     * layout is the checkpoint's own row-major bytes) and takes the grouped GEMM for decode too (no FP8 GEMV yet). */
    bool set_weight_format(uint32_t fmt) {
        if (fmt == MOE_FMT_W4NL64) {
            if (gemv_w4h2_) std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_GEMV_W4H2 on - packed f16 W4 decode GEMV\n");
            return true;
        }
        if (fmt != MOE_FMT_FP8_E4M3_BLOCK) {
            std::fprintf(stderr, "r3_moe_worker: no kernels for weight format %u\n", fmt);
            return false;
        }
        if (!moe_fp8_ || !moe_fp8_->layout || !quant_plain_ || !gated_plain_) {
            std::fprintf(stderr, "r3_moe_worker: installed libr4d lacks the fp8a8 grouped GEMM or the plain quantiser rows\n");
            return false;
        }
        weight_format_ = fmt;
        dtype_ = "fp8a8";   // the decode GEMV (R3_MOE_WORKER_GEMV, default on) now has an FP8 form
        moe_ = moe_fp8_;
        had_quant_ = quant_plain_;
        gated_quant_ = gated_plain_;
        gemv_debug_ = false;   // the grouped-GEMM-vs-GEMV debug comparison is W4-only
        /* Large-lane steps of up to 32 rows take the FP8 GEMV too: with the wide kernel (below) 32 rows is +60-70% engine decode at 6 streams
         * over 16 (35B stream A/B); with the old kernel 32 rows was worse. R3_MOE_WORKER_GEMV_LARGE_ROWS still overrides. */
        gemv_wide_ = gemv_wide_env_ < 0 ? true : gemv_wide_env_ > 0;
        gemv_h2_ = gemv_h2_env_ < 0 ? 2 : gemv_h2_env_;   /* FP8 default: packed f16 dot products, two rows a block */
        if (gemv_h2_) std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_GEMV_H2 on - packed f16 dot products in the FP8 decode GEMV, %d row(s) a block\n", gemv_h2_);
        return true;
    }
    /* The engine width of the segment this runner serves (Header::rank_count), checked on every request at the lane boundary. */
    bool set_engine_world(uint32_t world) {
        if (world < 1 || world > r3::moe_shm::kMaxEngineRanks) return false;
        lane_shape_.world = world;
        return true;
    }
    bool set_shape(const r3::moe_shm::LayoutShape& s) {
        if (!s.hidden || !s.top_k || s.top_k > r3::moe_shm::kMaxTopK || !s.act_group || s.hidden % 16 || s.hidden % s.act_group ||
            ((s.hidden / s.act_group) * 4) % 16 || s.hidden > kGemvMaxK) {
            std::fprintf(stderr, "r3_moe_worker: lane kernels cannot run hidden %u, top-k %u, group %u\n", s.hidden, s.top_k, s.act_group);
            return false;
        }
        shape_ = s;
        lane_shape_ = LaneShape{s.hidden, s.top_k, s.act_group, s.experts, lane_shape_.world};
        return true;
    }
    // Only --replay-sweep calls this method; production never reads RAD46
    // switches and always uses the measured, stage-specific accurate policy.
    bool set_rad46_sweep_variant(const char* name) {
        if (!name) return false;
        quad_perm_ = false;
        if (std::strcmp(name, "stock") == 0) rad46_sweep_ = Rad46Sweep::Stock;
        else if (std::strcmp(name, "output_lane") == 0) rad46_sweep_ = Rad46Sweep::OutputLane;
        else if (std::strcmp(name, "quad") == 0) rad46_sweep_ = Rad46Sweep::Quad;
        else if (std::strcmp(name, "quad_v2") == 0) rad46_sweep_ = Rad46Sweep::QuadV2;
        else if (std::strcmp(name, "production") == 0) rad46_sweep_ = Rad46Sweep::Production;
        else if (std::strcmp(name, "quad_perm") == 0) { rad46_sweep_ = Rad46Sweep::Production; quad_perm_ = true; return true; }
        else if (std::strcmp(name, "rowlane16") == 0) rad46_sweep_ = Rad46Sweep::RowLane16;
        else if (std::strcmp(name, "rowlane8") == 0) rad46_sweep_ = Rad46Sweep::RowLane8;
        else return false;
        return true;
    }
    uint32_t route_block() const {
        uint32_t n = kFuseMaxRoutes;
        while (n < model_.experts) n <<= 1;
        return n <= 1024 ? n : 0u;
    }
    ~Runner() {
        if (copy_event_) (void)hipEventDestroy(copy_event_);
        if (copy_stream_) {
            // Pinned H2D source and resident arena must outlive any queued
            // private-stream read, even if shutdown interrupted a proposal.
            (void)hipStreamSynchronize(copy_stream_);
            (void)hipStreamDestroy(copy_stream_);
        }
        if (move_host_) (void)hipHostFree(move_host_);
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
        const auto& conf = r3_serving::settings();
        gemv_h2_env_ = (int)conf.gemv_h2;
        gemv_w4h2_ = conf.gemv_w4h2;
        gemv_mr_ = conf.gemv_mr;
        if (gemv_mr_)
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_GEMV_MR on - the FP8 decode GEMV batches an expert's rows\n");
        gemv_wide_env_ = conf.gemv_wide ? 1 : 0;
        rows_ = conf.rows;
        if (rows_) std::fprintf(stderr, "r3_moe_worker: few-rows prefill kernel on the large lane\n");
        gemv_debug_ = conf.gemv == 2;
        gemv_ = conf.gemv == 1;
        gemv_large_rows_ = conf.handoff.gemv_large_rows;
        if (gemv_large_rows_)
            std::fprintf(stderr, "r3_moe_worker: large-lane requests up to %u rows take the GEMV\n", gemv_large_rows_);
        // Former R3_MOE_WORKER_POISON: only experiment.helper.poison in the manifest.
        poison_ = conf.poison == "zero" ? 2u : conf.poison == "0" ? 0u : 1u;
        if (!hip_ok(hipSetDevice(0), "hipSetDevice(0)")) return false;
        hipDeviceProp_t prop{};
        if (!hip_ok(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties")) return false;
        arch_ = prop.gcnArchName;
        const size_t colon = arch_.find(':');
        if (colon != std::string::npos) arch_.resize(colon);
        if (!load_plugin()) return false;
        experts_.resize((size_t)model_.layers * experts());
        tables_.resize(size_t(r3::worker_tables::kGenerations) * model_.layers);
        initial_locals_.assign(size_t(model_.layers)*experts(),-1);
        fixed_locals_.assign(size_t(model_.layers)*experts(),-1);
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

    struct SlotPlan {
        uint64_t stride = 0, lanes = 0, tables = 0, slot_map = 0;
        uint32_t expert_capacity = 0;
    };
    // The configured budget is for expert weights ONLY. Prepared lane
    // allocations, the two pointer banks, frozen map and one incoming slot
    // are card-memory requirements and never reduce ownership selection.
    bool slot_plan(uint64_t budget, SlotPlan* out) const {
        if (!out) return false;
        uint64_t largest = 0;
        for (uint32_t l = 0; l < model_.layers; ++l)
            for (uint32_t e = 0; e < experts(); ++e)
                if (model_.weight(l, e, 0) && model_.weight(l, e, 1)) {
                    const uint64_t n = slot_cost(l, e);
                    if (!n) return false;
                    largest = std::max(largest, n);
                }
        const uint64_t n = uint64_t(model_.layers) * experts();
        static_assert(sizeof(DeviceLayerTables) == r3::worker_footprint::kDeviceLayerTablesBytes,
                      "worker_footprint.h counts the pointer banks with this table size");
        const uint64_t tables = r3::worker_footprint::pointer_bank_bytes(footprint_shape());
        uint64_t lanes = 0;
        for (const LaneWork& w : lane_) lanes += lane_bytes(w);
        const auto plan = r3::worker_slots::budget_plan(budget, largest, n);
        *out = {largest, lanes, tables, r3::worker_footprint::slot_map_bytes(footprint_shape()), plan.expert_slots};
        return largest && plan.expert_slots > 0;
    }

    bool allocate(const std::vector<std::pair<uint32_t, uint32_t>>& owned, uint64_t budget,
                  bool replay = false) {
        constexpr size_t kLoadBounceBytes = 64ull * 1024ull * 1024ull;
        if (!load_bounce_.ensure(kLoadBounceBytes) || !budget) return false;
        SlotPlan plan{};
        if (!slot_plan(budget, &plan) || owned.size() > plan.expert_capacity ||
            owned.size() > UINT32_MAX || uint64_t(owned.size()) > UINT64_MAX / plan.stride) {
            std::fprintf(stderr, "r3_moe_worker: selected expert weights exceed expert budget\n");
            return false;
        }
        if (replay) {
            // The manifest budget is an upper bound, not a reason to reserve
            // thousands of empty expert slots during a captured replay.
            uint64_t actual_budget = 0;
            if (!r3::replay::expert_budget(plan.stride, owned.size(), budget, &actual_budget))
                return false;
            budget = actual_budget;
            plan.expert_capacity = uint32_t(owned.size());
        }
        const r3::worker_slots::DeviceMemory memory{
            plan.lanes, plan.tables, plan.slot_map, plan.stride,
            uint64_t(plan.expert_capacity) * plan.stride
        };
        size_t free_now = 0, total_now = 0;
        if (!lane_free_sampled_ ||
            !hip_ok(hipMemGetInfo(&free_now, &total_now), "hipMemGetInfo before expert arena"))
            return false;
        const auto fit = r3::worker_slots::check_device_memory(memory, free_before_lanes_, free_now);
        if (!fit.fits) {
            std::fprintf(stderr, "r3_moe_worker: insufficient free VRAM for unchanged expert selection: "
                         "shortfall %llu bytes, required %llu (arena %llu + lanes %llu + "
                         "pointer banks %llu + slot map %llu + incoming %llu), "
                         "free before lanes %llu, free now %zu, future need %llu\n",
                         (unsigned long long)fit.shortfall, (unsigned long long)fit.total,
                         (unsigned long long)memory.arena, (unsigned long long)memory.lanes,
                         (unsigned long long)memory.tables, (unsigned long long)memory.slot_map,
                         (unsigned long long)memory.staging, (unsigned long long)free_before_lanes_,
                         free_now, (unsigned long long)fit.future);
            return false;
        }
        slots_ = std::make_unique<r3::worker_slots::Pool<HipSlotAllocator>>(
            budget, plan.stride, plan.expert_capacity, slot_allocator_);
        if (!slots_->prepare()) return false;
        const size_t slot_map_bytes = size_t(plan.slot_map);
        const size_t table_bytes = size_t(model_.layers) * sizeof(DeviceLayerTables);
        layer_loaded_.assign(model_.layers, 0);
        return d_expert_slot_.ensure(slot_map_bytes) &&
               d_layer_tables_.ensure(r3::worker_tables::kGenerations * table_bytes) &&
               hip_ok(hipMemset(d_expert_slot_.p, 0xff, slot_map_bytes), "initialize expert slot map") &&
               hip_ok(hipMemset(d_layer_tables_.p, 0,
                                r3::worker_tables::kGenerations * table_bytes),
                      "initialize generation layer tables");
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

        for (uint32_t e=0;e<experts();++e)
            fixed_locals_[size_t(layer)*experts()+e]=slot_of_expert[e];
        for (auto [l, e] : owned) {
            if (l != layer || slot_of_expert[e] < 0) continue;
            initial_locals_[size_t(l)*experts()+e] = slot_of_expert[e];
            const uint32_t es = (uint32_t)slot_of_expert[e];
            Expert& x = experts_[(size_t)l * experts() + e];
            uint32_t id = 0;
            const uint64_t bytes = slot_cost(l, e);
            if (!slots_ || !bytes ||
                slots_->bootstrap(uint32_t(size_t(l) * experts() + e), bytes, 1, &id) !=
                    r3::residency::Result::OK) {
                std::fprintf(stderr, "r3_moe_worker: expert slot capacity refused layer %u expert %u\n", l, e);
                return false;
            }
            loading_base_ = static_cast<uint8_t*>(slots_->address(id));
            loading_size_ = bytes;
            loading_used_ = 0;
            if (!upload_weight(model_.weight(l, e, 0), true, &x.gu) ||
                !upload_weight(model_.weight(l, e, 1), false, &x.dn) ||
                loading_used_ != loading_size_) return false;
            loading_base_ = nullptr;
            loading_size_ = loading_used_ = 0;
            x.live = true;
        }
        for (uint32_t generation = 0; generation < r3::worker_tables::kGenerations; ++generation)
            if (!build_layer_table(layer, slot_of_expert, generation)) return false;
        layer_loaded_[layer] = 1;
        return true;
    }

    /* Replay admission uses the same bootstrapped physical residency ledger
     * and full-width weight bindings that serving will read. */
    bool compare_resident(uint32_t layer, uint32_t expert, uint32_t local) const {
        if (layer >= model_.layers || expert >= experts() || !slots_) return false;
        const size_t key = size_t(layer) * experts() + expert;
        const Expert& x = experts_[key];
        if (!x.live || initial_locals_[key] != int32_t(local) ||
            !x.gu.code || !x.gu.scale ||
            !x.dn.code || !x.dn.scale) return false;
        for (const auto& s : slots_->ledger().slots())
            if (s.expert == key && s.state == r3::residency::State::ACTIVE &&
                s.completed_planes == 15 && s.bytes == slots_->stride())
                return true;
        return false;
    }

    r3::residency::Ledger* move_ledger() {
        return slots_ ? &slots_->ledger() : nullptr;
    }
    const std::vector<int32_t>& initial_locals() const { return initial_locals_; }
    uint64_t expert_stride() const { return slots_ ? slots_->stride() : 0; }
    uint64_t move_bytes_staged() const { return move_bytes_staged_; }
    int32_t preferred_local(uint32_t expert) const override {
        return expert<fixed_locals_.size() ? fixed_locals_[expert] : -1;
    }
    void activate_bank(uint32_t slot) override { active_generation_=slot; }
    bool start_upload(uint32_t expert, uint32_t physical, uint32_t local) override {
        if (!slots_ || copy_pending_ || staging_upload_ || expert>=experts_.size() ||
            local>=experts() || preferred_local(expert)!=int32_t(local) ||
            !slots_->address(physical) ||
            experts_[expert].live) return false;
        if (!copy_stream_ &&
            !hip_ok(hipStreamCreateWithFlags(&copy_stream_,hipStreamNonBlocking),
                    "create expert copy stream")) return false;
        if (!move_host_ &&
            !hip_ok(hipHostMalloc(reinterpret_cast<void**>(&move_host_),
                                  size_t(slots_->stride()),hipHostMallocDefault),
                    "pin incoming expert host staging")) return false;
        const uint32_t l=expert/experts(), e=expert%experts();
        const uint64_t bytes=slot_cost(l,e);
        if (!bytes || bytes>slots_->stride()) return false;
        loading_base_=static_cast<uint8_t*>(slots_->address(physical));
        loading_size_=bytes; loading_used_=0; staging_upload_=true;
        std::memset(move_host_,0,size_t(slots_->stride()));
        Expert& x=experts_[expert];
        const bool encoded=
            upload_weight(model_.weight(l,e,0),true,&x.gu) &&
            upload_weight(model_.weight(l,e,1),false,&x.dn) &&
            loading_used_==bytes;
        staging_upload_=false; loading_base_=nullptr;
        loading_size_=loading_used_=0;
        if (!encoded) { x={}; return false; }
        if (!hip_ok(hipEventCreateWithFlags(&copy_event_,hipEventDisableTiming),
                    "create per-expert copy event")) { x={}; return false; }
        if (!hip_ok(hipMemcpyAsync(slots_->address(physical),move_host_,
                                   size_t(bytes),hipMemcpyHostToDevice,copy_stream_),
                    "upload incoming expert on private stream") ||
            !hip_ok(hipEventRecord(copy_event_,copy_stream_),
                    "record expert copy completion")) {
            (void)hipStreamSynchronize(copy_stream_);
            (void)hipEventDestroy(copy_event_); copy_event_=nullptr; x={}; return false;
        }
        move_expert_=expert; copy_pending_=true;
        move_bytes_staged_ += bytes; // successfully staged on the private stream, never per-request
        return true;
    }
    r3::worker_moves::Event upload_event() override {
        using r3::worker_moves::Event;
        if (!copy_pending_ || !copy_event_) return Event::FAILED;
        const hipError_t status=hipEventQuery(copy_event_);
        if (status==hipErrorNotReady) return Event::PENDING;
        if (status!=hipSuccess) return Event::FAILED;
        (void)hipEventDestroy(copy_event_); copy_event_=nullptr;
        experts_[move_expert_].live=true;
        move_expert_=UINT32_MAX; copy_pending_=false;
        return Event::COMPLETE;
    }
    bool build_bank(uint32_t generation,const std::vector<int32_t>& locals) override {
        if (!slots_ || generation>1 || generation==active_generation_ ||
            copy_pending_ || locals.size()!=size_t(model_.layers)*experts())
            return false; // the active pointer bank is never writable
        for (uint32_t layer=0; layer<model_.layers; ++layer)
            if (!build_layer_table(layer,locals.data()+size_t(layer)*experts(),generation))
                return false;
        return true;
    }
    bool drain_upload() override {
        if (copy_stream_ && !hip_ok(hipStreamSynchronize(copy_stream_),
                                   "drain precommit expert copy")) return false;
        if (copy_event_) { (void)hipEventDestroy(copy_event_); copy_event_=nullptr; }
        move_expert_=UINT32_MAX; copy_pending_=false; staging_upload_=false;
        return true;
    }
    void forget(uint32_t expert) override {
        if (expert<experts_.size()) experts_[expert]={};
    }

    // Resolve the actual plugin weight dtypes before sizing per-lane scratch.
    // A later upload must agree with these types (upload_weight checks it).
    bool prime_weight_layout() {
        for (uint32_t l = 0; l < model_.layers; ++l)
            for (uint32_t e = 0; e < experts(); ++e) {
                const auto* gu_entry = model_.weight(l, e, 0);
                const auto* dn_entry = model_.weight(l, e, 1);
                if (!gu_entry || !dn_entry) continue;
                WeightPlan gu{}, dn{};
                if (!plan_weight(gu_entry, true, &gu) ||
                    !plan_weight(dn_entry, false, &dn)) return false;
                gu_code_dtype_ = gu.code.dtype; gu_scale_dtype_ = gu.scale.dtype;
                dn_code_dtype_ = dn.code.dtype; dn_scale_dtype_ = dn.scale.dtype;
                return true;
            }
        return false;
    }

    /* Small lanes are sized to their class's rows and pre-queued; the large lane to large_rows, launched per request.
     * `count` small classes are enabled (ascending `sizes`); the other small lanes are left unprepared (cap_rows 0). */
    bool prepare_lanes(const std::array<uint32_t, r3::moe_shm::kSmallClasses>& sizes, uint32_t count, uint32_t large_rows,
                       const std::array<bool, r3::moe_shm::kLanes>& open,
                       uint32_t dense_route_slots = 0) {
        using namespace r3::moe_shm;
        if (count == 0 || count > kSmallClasses || large_rows == 0 || large_rows > kMaxRows || !gu_code_dtype_)
            return false;
        for (uint32_t c = 0; c < count; ++c)
            if (sizes[c] == 0 || sizes[c] > kMaxRows || sizes[c] > large_rows || (c && sizes[c] <= sizes[c - 1]))
                return false;
        size_t free_bytes = 0, total_bytes = 0;
        if (!hip_ok(hipMemGetInfo(&free_bytes, &total_bytes), "hipMemGetInfo before lane preparation"))
            return false;
        free_before_lanes_ = free_bytes;
        lane_free_sampled_ = true;
        class_count_ = count;
        class_rows_ = sizes;
        large_rows_ = large_rows;
        const size_t nclass = experts() / 2;
        for (uint32_t lane = 0; lane < kLanes; ++lane) {
            const uint32_t rank = lane % kMaxSlots;
            const uint32_t cls = lane_class(lane);
            LaneWork& w = lane_[lane];
            /* `open`: lanes that get a stream (a hardware queue); buffers only for an open lane of an enabled class. */
            if (open[lane] && !w.stream &&
                (!hip_ok(hipStreamCreateWithFlags(&w.stream, hipStreamNonBlocking), "hipStreamCreate lane") ||
                 !hip_ok(hipEventCreate(&w.sample_start), "hipEventCreate lane start") ||
                 !hip_ok(hipEventCreate(&w.sample_stop), "hipEventCreate lane stop") ||
                 !hip_ok(hipEventCreate(&w.stage[0]), "hipEventCreate lane stage") ||
                 !hip_ok(hipEventCreate(&w.stage[1]), "hipEventCreate lane stage") ||
                 !hip_ok(hipEventCreate(&w.stage[2]), "hipEventCreate lane stage") ||
                 !hip_ok(hipEventCreate(&w.stage[3]), "hipEventCreate lane stage")))
                return false;
            if (!open[lane] || (cls < kSmallClasses && cls >= class_count_)) {   /* not ours or disabled: no buffers, no worker */
                w.cap_rows = 0;
                continue;
            }
            const uint32_t cap = cls < kSmallClasses ? class_rows_[cls] : large_rows_;
            // The lane's sorted index space is dense cap * top_k, even if
            // the capture publishes fewer routes. Compare reserves that
            // request-sized bound, not the manifest's 4096-row serving bound.
            if (dense_route_slots && dense_route_slots > uint64_t(cap) * shape_.top_k)
                return false;
            const uint32_t routes = dense_route_slots ? dense_route_slots : cap * shape_.top_k;
            w.cap_rows = cap;
            if (!ensure_lane_work(w, cap, routes, routes) ||
                !w.d_stats.ensure(3 * sizeof(uint64_t)) || !w.d_publish_limit.ensure(sizeof(uint32_t)) ||
                (cls == kSmallClasses && !w.d_chunk_cnt.ensure(kCombineChunks * sizeof(uint32_t))) ||
                !hip_ok(hipMemset(w.d_publish_limit.p, 0xff, sizeof(uint32_t)), "lane publish limit"))
                return false;
            for (uint32_t parity = 0; parity < 2; ++parity) {
                if (!w.gu_code[parity].ensure(nclass * sizeof(void*)) ||
                    !w.gu_scale[parity].ensure(nclass * sizeof(void*)) ||
                    !w.dn_code[parity].ensure(nclass * sizeof(void*)) ||
                    !w.dn_scale[parity].ensure(nclass * sizeof(void*)))
                    return false;
            }
            if (!prepare_lane_scratch(w, rank, cap, routes) ||
                !hip_ok(hipMemset(w.d_stats.p, 0, 3 * sizeof(uint64_t)), "zero lane stats"))
                return false;
        }
        /* The serving lanes are exactly the shared footprint (worker_footprint.h), the one bc-serve derives an omitted
         * budget-mib with; a drift here would let the planner fill an arena the card cannot hold. Compare re-plans per capture
         * (dense route slots) and is not a serving footprint. */
        if (!dense_route_slots) {
            r3::worker_footprint::Lanes plan{};
            plan.small_rows = sizes; plan.small_count = count; plan.large_rows = large_rows; plan.open = open;
            uint64_t prepared = 0;
            for (const LaneWork& w : lane_) prepared += lane_bytes(w) - w.scratch.bytes;
            const uint64_t want = r3::worker_footprint::lanes_bytes(footprint_shape(), plan);
            if (prepared != want) {
                std::fprintf(stderr, "r3_moe_worker: prepared lanes take %llu bytes, the shared footprint says %llu "
                             "(worker_footprint.h out of step with the lane buffers)\n",
                             (unsigned long long)prepared, (unsigned long long)want);
                return false;
            }
        }
        return true;
    }
    r3::worker_footprint::Shape footprint_shape() const {
        return {model_.layers, experts(), shape_.hidden, shape_.intermediate, shape_.top_k, (uint64_t)kGroup};
    }

    bool release_compare_lanes() {
        // No request may still reference a buffer when it is freed.
        for (LaneWork& w : lane_)
            if (w.stream && !hip_ok(hipStreamSynchronize(w.stream), "sync compare lane"))
                return false;
        bool ok = true;
        for (LaneWork& w : lane_)
            if (!w.release_allocations()) ok = false;
        return ok;
    }

    bool stage_compare_activation(void* host_map, uint32_t lane,
                                  const uint16_t* activation, uint32_t rows) {
        using namespace r3::moe_shm;
        if (!host_map || !activation || lane >= kLanes || rows == 0 || rows > lane_[lane].cap_rows)
            return false;
        const uint32_t rank = lane % kMaxSlots;
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
        const uint32_t rank = lane % kMaxSlots;
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
        /* The slot's consumer plane (RAD49): fixed by the slot and the segment's engine width, so a pre-queued small lane
         * needs no request field. */
        auto* dst = static_cast<uint16_t*>(
            dev_at(ptr<uint16_t>(
                host_map, hhost->result_offset[result_index(rank, slot_consumer(rank, hhost->rank_count))])));

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
        const bool fuse = lane_is_small(lane) && rows <= 8u && routes <= kFuseMaxRoutes &&
                          r3_serving::settings().fuseroute;
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
        const bool vecio = r3_serving::settings().vecio;
        /* R3_MOE_WORKER_SPARSE_ACT (default on): the large lane copies only the rows the request's routes name. The kernels are queued before the request
         * arrives, so the route count is a device-side fact: the mask is built on the device and the copy skips unflagged rows. */
        const bool sparse_act = r3_serving::settings().sparse_act;
        if ((vecio || fuse) && ((reinterpret_cast<uintptr_t>(act_q) | reinterpret_cast<uintptr_t>(act_s) |
                       reinterpret_cast<uintptr_t>(dst)) & 15u) != 0) {
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_VECIO needs 16-byte aligned activation and result planes\n");
            return -EINVAL;
        }
        if (fuse) {
            /* the activation copy is folded into lane_front_small below */
        } else if (vecio && sparse_act && !lane_is_small(lane)) {
            if (!hip_ok(hipMemsetAsync(w.d_rowmask.p, 0, rows, w.stream), "clear row mask")) return -EIO;
            hipLaunchKernelGGL(lane_mark_rows, route_grid, block, 0, w.stream,
                               route, q, static_cast<uint8_t*>(w.d_rowmask.p),
                               rank, sequence, rows, model_.layers, lane_shape_);
            const dim3 vgrid(((uint64_t)rows * (shape_.hidden / 16) + block.x - 1) / block.x);
            hipLaunchKernelGGL(lane_prepare_activation_sparse, vgrid, block, 0, w.stream,
                               q, static_cast<const uint8_t*>(w.d_rowmask.p),
                               reinterpret_cast<const uint4*>(act_q), reinterpret_cast<const uint4*>(act_s),
                               reinterpret_cast<uint4*>(w.d_aq.p), reinterpret_cast<uint4*>(w.d_as.p),
                               rank, sequence, rows, model_.layers,
                               static_cast<uint64_t*>(w.d_stats.p), lane_shape_);
        } else if (vecio) {
            const dim3 vgrid(((uint64_t)rows * (shape_.hidden / 16) + block.x - 1) / block.x);
            hipLaunchKernelGGL(lane_prepare_activation_v, vgrid, block, 0, w.stream,
                               q, reinterpret_cast<const uint4*>(act_q), reinterpret_cast<const uint4*>(act_s),
                               reinterpret_cast<uint4*>(w.d_aq.p), reinterpret_cast<uint4*>(w.d_as.p),
                               rank, sequence, rows, model_.layers,
                               static_cast<uint64_t*>(w.d_stats.p), lane_shape_);
        } else {
            hipLaunchKernelGGL(lane_prepare_activation, act_grid, block, 0, w.stream,
                               q, act_q, act_s,
                               static_cast<uint8_t*>(w.d_aq.p), static_cast<float*>(w.d_as.p),
                               rank, sequence, rows, model_.layers,
                               static_cast<uint64_t*>(w.d_stats.p), lane_shape_);
        }
        if (fuse) {
            auto front = [&](auto block_n) {
                constexpr uint32_t N = decltype(block_n)::value;
                hipLaunchKernelGGL((lane_front_small<N>), dim3(1), dim3(N), 0, w.stream,
                                   route, q, static_cast<const DeviceLayerTables*>(d_layer_tables_.p),
                                   static_cast<const uint16_t*>(d_expert_slot_.p),
                                   reinterpret_cast<const uint4*>(act_q), reinterpret_cast<const uint4*>(act_s),
                                   reinterpret_cast<uint4*>(w.d_aq.p), reinterpret_cast<uint4*>(w.d_as.p),
                                   static_cast<uint64_t*>(w.d_stats.p),
                                   rank, sequence, rows, model_.layers, routes,
                                   experts(), static_cast<int32_t*>(w.d_off.p), static_cast<int32_t*>(w.d_sorted.p),
                                   static_cast<int32_t*>(w.d_inv.p), static_cast<uint16_t*>(w.d_ew.p),
                                   static_cast<void**>(w.gu_code[0].p), static_cast<void**>(w.gu_scale[0].p),
                                   static_cast<void**>(w.dn_code[0].p), static_cast<void**>(w.dn_scale[0].p),
                                   static_cast<void**>(w.gu_code[1].p), static_cast<void**>(w.gu_scale[1].p),
                                   static_cast<void**>(w.dn_code[1].p), static_cast<void**>(w.dn_scale[1].p), lane_shape_);
            };
            switch (route_block()) {
                case 256: front(std::integral_constant<uint32_t, 256>{}); break;
                case 512: front(std::integral_constant<uint32_t, 512>{}); break;
                case 1024: front(std::integral_constant<uint32_t, 1024>{}); break;
                default: return -EINVAL;
            }
        } else {
        hipLaunchKernelGGL(lane_route_count, route_grid, block, 0, w.stream,
                           route, q, static_cast<const uint16_t*>(d_expert_slot_.p),
                           rank, sequence, rows, model_.layers, experts(),
                           static_cast<uint32_t*>(w.d_counts.p),
                           static_cast<int32_t*>(w.d_flat.p),
                           static_cast<uint16_t*>(w.d_ew.p), lane_shape_);
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
                           route, q, static_cast<const uint16_t*>(d_expert_slot_.p),
                           rank, sequence, rows, model_.layers, experts(),
                           static_cast<const int32_t*>(w.d_flat.p),
                           static_cast<uint32_t*>(w.d_cursor.p),
                           static_cast<int32_t*>(w.d_sorted.p),
                           nullptr, lane_shape_);
        }
        if (!hip_ok(hipGetLastError(), "lane request kernels")) return -EIO;
        const bool sampling = sample && w.sample_pending && w.sample_sequence == sequence;
        if (sampling && !hip_ok(hipEventRecord(w.stage[0], w.stream), "lane stage 0")) return -EIO;
        if (trace_) hipLaunchKernelGGL(lane_stage_stamp, dim3(1), dim3(1), 0, w.stream, q, 0u);   /* activations ready */

        /* Every request asks for the whole expert (full width, RAD49). */
        {
            void* out_down = w.d_down.p;
            if (!fuse)
            hipLaunchKernelGGL(lane_select_tables,
                               dim3((experts() / 2 + block.x - 1) / block.x), block, 0, w.stream,
                               q, static_cast<const DeviceLayerTables*>(d_layer_tables_.p),
                               model_.layers, rank, sequence, rows, experts(),
                               static_cast<void**>(w.gu_code[0].p),
                               static_cast<void**>(w.gu_scale[0].p),
                               static_cast<void**>(w.dn_code[0].p),
                               static_cast<void**>(w.dn_scale[0].p),
                               static_cast<void**>(w.gu_code[1].p),
                               static_cast<void**>(w.gu_scale[1].p),
                               static_cast<void**>(w.dn_code[1].p),
                               static_cast<void**>(w.dn_scale[1].p), lane_shape_);
            const int64_t w0 = mid_w();
            const int64_t w1 = mid_w();
            /* Small lane, decode GEMV (RAD05): the step's routes, one block a (row, 64 outputs). */
            const bool gemv = gemv_ && (lane_is_small(lane) || rows <= gemv_large_rows_);
            /* R3_MOE_WORKER_ROWS_SMALL (RAD06): the few-rows kernel on the small (decode) lane too. A decode request's 8 rows
             * send ~2.4 routes to each distinct expert (the hot experts are owned by the helpers), and the per-route GEMV
             * reads that expert's weights once a route; here they are read once for all the expert's rows. Read each
             * call so a replay sweep can compare; the rows accumulate in a different order, so it is not bit-exact. */
            const bool rows_small = r3_serving::settings().rows_small &&
                                    lane_is_small(lane) && rows <= 16u;
            const bool rowk = (rows_ && !lane_is_small(lane)) || rows_small;
            // Accurate W4 rows only. The historical inaccurate fp16-scaled
            // path is absent from production; unsupported tile geometry fails
            // closed instead of quietly restoring the precision regression.
            const bool w4_rows = rowk && weight_format_ == MOE_FMT_W4NL64;
            const bool accurate_geometry =
                shape_.hidden % kOutputLaneK == 0 &&
                w0 > 0 && w1 > 0 && w0 <= mid_w() && w1 <= mid_w() &&
                w0 % kOutputLaneK == 0 && w1 % kOutputLaneK == 0 &&
                mid_w() % kOutputLaneK == 0;
            if (w4_rows && !accurate_geometry) {
                std::fprintf(stderr, "r3_moe_worker: accurate W4 rows need complete K128 groups\n");
                return -EINVAL;
            }
            const Rad46RowsForm gu_form = rad46_rows_form(rad46_sweep_, true, rows);
            const Rad46RowsForm dn_form = rad46_rows_form(rad46_sweep_, false, rows);
            const dim3 rows_grid_gu(experts(), (uint32_t)((2 * std::max(w0, w1) + kRowsTileN - 1) / kRowsTileN));
            const dim3 rows_grid_dn(experts(), (uint32_t)((shape_.hidden + kRowsTileN - 1) / kRowsTileN));
            const dim3 w4_grid_gu(experts(), (uint32_t)((2 * std::max(w0, w1) + 8 * kRowsGuOut - 1) / (8 * kRowsGuOut)));
            const dim3 w4_grid_dn(experts(), (uint32_t)((shape_.hidden + 8 * kRowsDnOut - 1) / (8 * kRowsDnOut)));
            const dim3 gemv_grid_gu(routes, (uint32_t)((2 * std::max(w0, w1) + kGemvTile - 1) / kGemvTile));
            const dim3 gemv_grid_dn(routes, (uint32_t)((shape_.hidden + kGemvTile - 1) / kGemvTile));
            std::vector<RadParam> gp =
                gemm_params(rows, 2 * w0, shape_.hidden, 2, "token", 2 * w1, shape_.hidden);
            if (rowk) {
                if (weight_format_ == MOE_FMT_FP8_E4M3_BLOCK)
                    hipLaunchKernelGGL((lane_moe_rows_fp8<true>), rows_grid_gu, dim3(256), 0, w.stream,
                                   static_cast<const uint8_t*>(w.d_aq.p), static_cast<const float*>(w.d_as.p),
                                   static_cast<const int32_t*>(w.d_sorted.p), static_cast<const int32_t*>(w.d_off.p),
                                   (uint32_t)shape_.top_k,
                                   static_cast<void* const*>(w.gu_code[0].p), static_cast<void* const*>(w.gu_scale[0].p),
                                   static_cast<void* const*>(w.gu_code[1].p), static_cast<void* const*>(w.gu_scale[1].p),
                                   (uint32_t)(2 * w0), (uint32_t)(2 * w1), (uint32_t)shape_.hidden, (uint32_t)shape_.hidden,
                                   (uint32_t)shape_.hidden, (uint32_t)(2 * mid_w()), static_cast<uint16_t*>(w.d_gu.p),
                                   (uint32_t)mid_w());
                else if (gu_form == Rad46RowsForm::Quad8)
                    R3_QUADDPP_LAUNCH(true, kQuadV2GuRB,
                                   dim3(experts(), (uint32_t)(2 * std::max(w0, w1) / kQuadDppN)), dim3(128), 0, w.stream,
                                   static_cast<const uint8_t*>(w.d_aq.p), static_cast<const float*>(w.d_as.p),
                                   static_cast<const int32_t*>(w.d_sorted.p), static_cast<const int32_t*>(w.d_off.p),
                                   (uint32_t)shape_.top_k,
                                   static_cast<void* const*>(w.gu_code[0].p), static_cast<void* const*>(w.gu_scale[0].p),
                                   static_cast<void* const*>(w.gu_code[1].p), static_cast<void* const*>(w.gu_scale[1].p),
                                   (uint32_t)(2 * w0), (uint32_t)(2 * w1), (uint32_t)shape_.hidden, (uint32_t)shape_.hidden,
                                   (uint32_t)shape_.hidden, (uint32_t)(2 * mid_w()), static_cast<uint16_t*>(w.d_gu.p),
                                   (uint32_t)mid_w());
                else if (gu_form == Rad46RowsForm::Quad16)
                    R3_QUADDPP_LAUNCH(true, kQuadDppRB,
                                   dim3(experts(), (uint32_t)(2 * std::max(w0, w1) / kQuadDppN)), dim3(128), 0, w.stream,
                                   static_cast<const uint8_t*>(w.d_aq.p), static_cast<const float*>(w.d_as.p),
                                   static_cast<const int32_t*>(w.d_sorted.p), static_cast<const int32_t*>(w.d_off.p),
                                   (uint32_t)shape_.top_k,
                                   static_cast<void* const*>(w.gu_code[0].p), static_cast<void* const*>(w.gu_scale[0].p),
                                   static_cast<void* const*>(w.gu_code[1].p), static_cast<void* const*>(w.gu_scale[1].p),
                                   (uint32_t)(2 * w0), (uint32_t)(2 * w1), (uint32_t)shape_.hidden, (uint32_t)shape_.hidden,
                                   (uint32_t)shape_.hidden, (uint32_t)(2 * mid_w()), static_cast<uint16_t*>(w.d_gu.p),
                                   (uint32_t)mid_w());
                else if (gu_form == Rad46RowsForm::RowLane16)
                    hipLaunchKernelGGL((lane_moe_rows_w4_rowlane<true, kQuadDppRB>),
                                   dim3(experts(), (uint32_t)(2 * std::max(w0, w1) / kQuadDppN)), dim3(128), 0, w.stream,
                                   static_cast<const uint8_t*>(w.d_aq.p), static_cast<const float*>(w.d_as.p),
                                   static_cast<const int32_t*>(w.d_sorted.p), static_cast<const int32_t*>(w.d_off.p),
                                   (uint32_t)shape_.top_k,
                                   static_cast<void* const*>(w.gu_code[0].p), static_cast<void* const*>(w.gu_scale[0].p),
                                   static_cast<void* const*>(w.gu_code[1].p), static_cast<void* const*>(w.gu_scale[1].p),
                                   (uint32_t)(2 * w0), (uint32_t)(2 * w1), (uint32_t)shape_.hidden, (uint32_t)shape_.hidden,
                                   (uint32_t)shape_.hidden, (uint32_t)(2 * mid_w()), static_cast<uint16_t*>(w.d_gu.p),
                                   (uint32_t)mid_w());
                else if (gu_form == Rad46RowsForm::RowLane8)
                    hipLaunchKernelGGL((lane_moe_rows_w4_rowlane<true, kQuadV2GuRB>),
                                   dim3(experts(), (uint32_t)(2 * std::max(w0, w1) / kQuadDppN)), dim3(128), 0, w.stream,
                                   static_cast<const uint8_t*>(w.d_aq.p), static_cast<const float*>(w.d_as.p),
                                   static_cast<const int32_t*>(w.d_sorted.p), static_cast<const int32_t*>(w.d_off.p),
                                   (uint32_t)shape_.top_k,
                                   static_cast<void* const*>(w.gu_code[0].p), static_cast<void* const*>(w.gu_scale[0].p),
                                   static_cast<void* const*>(w.gu_code[1].p), static_cast<void* const*>(w.gu_scale[1].p),
                                   (uint32_t)(2 * w0), (uint32_t)(2 * w1), (uint32_t)shape_.hidden, (uint32_t)shape_.hidden,
                                   (uint32_t)shape_.hidden, (uint32_t)(2 * mid_w()), static_cast<uint16_t*>(w.d_gu.p),
                                   (uint32_t)mid_w());
                else if (gu_form == Rad46RowsForm::OutputLane)
                    hipLaunchKernelGGL((lane_moe_rows_w4_outputlane<true, kOutputLaneRB>),
                                   dim3(experts(), (uint32_t)(2 * std::max(w0, w1) / kOutputLaneN)), dim3(kOutputLaneN), 0, w.stream,
                                   static_cast<const uint8_t*>(w.d_aq.p), static_cast<const float*>(w.d_as.p),
                                   static_cast<const int32_t*>(w.d_sorted.p), static_cast<const int32_t*>(w.d_off.p),
                                   (uint32_t)shape_.top_k,
                                   static_cast<void* const*>(w.gu_code[0].p), static_cast<void* const*>(w.gu_scale[0].p),
                                   static_cast<void* const*>(w.gu_code[1].p), static_cast<void* const*>(w.gu_scale[1].p),
                                   (uint32_t)(2 * w0), (uint32_t)(2 * w1), (uint32_t)shape_.hidden, (uint32_t)shape_.hidden,
                                   (uint32_t)shape_.hidden, (uint32_t)(2 * mid_w()), static_cast<uint16_t*>(w.d_gu.p),
                                   (uint32_t)mid_w());
                else if (gu_form == Rad46RowsForm::StockReplay)
                    hipLaunchKernelGGL((lane_moe_rows_w4_stock_replay<true, kRowsGuRB, kRowsGuOut>), w4_grid_gu, dim3(256), 0, w.stream,
                                   static_cast<const uint8_t*>(w.d_aq.p), static_cast<const float*>(w.d_as.p),
                                   static_cast<const int32_t*>(w.d_sorted.p), static_cast<const int32_t*>(w.d_off.p),
                                   (uint32_t)shape_.top_k,
                                   static_cast<void* const*>(w.gu_code[0].p), static_cast<void* const*>(w.gu_scale[0].p),
                                   static_cast<void* const*>(w.gu_code[1].p), static_cast<void* const*>(w.gu_scale[1].p),
                                   (uint32_t)(2 * w0), (uint32_t)(2 * w1), (uint32_t)shape_.hidden, (uint32_t)shape_.hidden,
                                   (uint32_t)shape_.hidden, (uint32_t)(2 * mid_w()), static_cast<uint16_t*>(w.d_gu.p),
                                   (uint32_t)mid_w());
                else return -EINVAL;
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
                                   (uint32_t)mid_w(), lane_shape_);
                w.dbg_routes = routes;
            }
            if (sampling && !hip_ok(hipEventRecord(w.stage[1], w.stream), "lane stage 1")) return -EIO;
            std::vector<RadParam> gqp = {
                pi("M", routes), pi("n", mid_w()), pi("group", kGroup),
                ps("dtype", "bf16"), ps("act", "silu")
            };
            /* the plugin's gated quantiser in a lane kernel that stops at the request's route count (moe_lane_quant.h): the
             * plugin's form quantises every row of the lane's capacity. Read each call so a replay can compare. */
            const bool lane_quant = r3_serving::settings().lane_quant &&
                                    (gated_quant_ == gated_had_ || gated_quant_ == gated_plain_) && mid_w() % 4u == 0 &&
                                    (gated_quant_ != gated_had_ || mid_w() % kGroup == 0);   /* the plugin refuses a rotated ragged width too */
            if (lane_quant) {
                const uint32_t nkb = (uint32_t)((mid_w() + kGroup - 1) / kGroup);
                const dim3 qb(std::min<uint32_t>(nkb, 32u) * 32u);
                if (gated_quant_ == gated_had_)
                    hipLaunchKernelGGL((lane_gated_quant<true>), dim3(routes), qb, 0, w.stream,
                                       q, rank, sequence, rows, model_.layers, lane_shape_,
                                       static_cast<const uint16_t*>(w.d_gu.p), static_cast<uint8_t*>(w.d_mid.p),
                                       static_cast<float*>(w.d_mid_s.p), (uint32_t)mid_w(), nkb,
                                       (uint32_t)(2 * mid_w()), (uint32_t)mid_w(), nkb);
                else
                    hipLaunchKernelGGL((lane_gated_quant<false>), dim3(routes), qb, 0, w.stream,
                                       q, rank, sequence, rows, model_.layers, lane_shape_,
                                       static_cast<const uint16_t*>(w.d_gu.p), static_cast<uint8_t*>(w.d_mid.p),
                                       static_cast<float*>(w.d_mid_s.p), (uint32_t)mid_w(), nkb,
                                       (uint32_t)(2 * mid_w()), (uint32_t)mid_w(), nkb);
                if (!hip_ok(hipGetLastError(), "lane gated quant")) return -EIO;
            } else if (launch_on(gated_quant_,
                          { tensor(w.d_gu.p, RAD_BF16, {routes, 2 * mid_w()}),
                            tensor(w.d_mid.p, RAD_F8E4M3, {routes, mid_w()}),
                            tensor(w.d_mid_s.p, RAD_F32, {routes, mid_w() / kGroup}),
                            absent() }, gqp, w.stream, w.scratch) < 0)
                return -EIO;

            if (sampling && !hip_ok(hipEventRecord(w.stage[2], w.stream), "lane stage 2")) return -EIO;
            std::vector<RadParam> dp =
                gemm_params(rows, shape_.hidden, w0, 1, "sorted", shape_.hidden, w1);
            if (rowk) {
                if (weight_format_ == MOE_FMT_FP8_E4M3_BLOCK)
                    hipLaunchKernelGGL((lane_moe_rows_fp8<false>), rows_grid_dn, dim3(256), 0, w.stream,
                                   static_cast<const uint8_t*>(w.d_mid.p), static_cast<const float*>(w.d_mid_s.p),
                                   static_cast<const int32_t*>(w.d_sorted.p), static_cast<const int32_t*>(w.d_off.p),
                                   (uint32_t)shape_.top_k,
                                   static_cast<void* const*>(w.dn_code[0].p), static_cast<void* const*>(w.dn_scale[0].p),
                                   static_cast<void* const*>(w.dn_code[1].p), static_cast<void* const*>(w.dn_scale[1].p),
                                   (uint32_t)shape_.hidden, (uint32_t)shape_.hidden, (uint32_t)w0, (uint32_t)w1,
                                   (uint32_t)mid_w(), (uint32_t)shape_.hidden, static_cast<uint16_t*>(out_down), 0u);
                else if (dn_form == Rad46RowsForm::Quad16 || dn_form == Rad46RowsForm::Quad8)
                    R3_QUADDPP_LAUNCH(false, kQuadDppRB,
                                   dim3(experts(), (uint32_t)(shape_.hidden / kQuadDppN)), dim3(128), 0, w.stream,
                                   static_cast<const uint8_t*>(w.d_mid.p), static_cast<const float*>(w.d_mid_s.p),
                                   static_cast<const int32_t*>(w.d_sorted.p), static_cast<const int32_t*>(w.d_off.p),
                                   (uint32_t)shape_.top_k,
                                   static_cast<void* const*>(w.dn_code[0].p), static_cast<void* const*>(w.dn_scale[0].p),
                                   static_cast<void* const*>(w.dn_code[1].p), static_cast<void* const*>(w.dn_scale[1].p),
                                   (uint32_t)shape_.hidden, (uint32_t)shape_.hidden, (uint32_t)w0, (uint32_t)w1,
                                   (uint32_t)mid_w(), (uint32_t)shape_.hidden, static_cast<uint16_t*>(out_down), 0u);
                else if (dn_form == Rad46RowsForm::OutputLane)
                    hipLaunchKernelGGL((lane_moe_rows_w4_outputlane<false, kOutputLaneRB>),
                                   dim3(experts(), (uint32_t)(shape_.hidden / kOutputLaneN)), dim3(kOutputLaneN), 0, w.stream,
                                   static_cast<const uint8_t*>(w.d_mid.p), static_cast<const float*>(w.d_mid_s.p),
                                   static_cast<const int32_t*>(w.d_sorted.p), static_cast<const int32_t*>(w.d_off.p),
                                   (uint32_t)shape_.top_k,
                                   static_cast<void* const*>(w.dn_code[0].p), static_cast<void* const*>(w.dn_scale[0].p),
                                   static_cast<void* const*>(w.dn_code[1].p), static_cast<void* const*>(w.dn_scale[1].p),
                                   (uint32_t)shape_.hidden, (uint32_t)shape_.hidden, (uint32_t)w0, (uint32_t)w1,
                                   (uint32_t)mid_w(), (uint32_t)shape_.hidden, static_cast<uint16_t*>(out_down), 0u);
                else if (dn_form == Rad46RowsForm::StockReplay)
                    hipLaunchKernelGGL((lane_moe_rows_w4_stock_replay<false, kRowsDnRB, kRowsDnOut>), w4_grid_dn, dim3(256), 0, w.stream,
                                   static_cast<const uint8_t*>(w.d_mid.p), static_cast<const float*>(w.d_mid_s.p),
                                   static_cast<const int32_t*>(w.d_sorted.p), static_cast<const int32_t*>(w.d_off.p),
                                   (uint32_t)shape_.top_k,
                                   static_cast<void* const*>(w.dn_code[0].p), static_cast<void* const*>(w.dn_scale[0].p),
                                   static_cast<void* const*>(w.dn_code[1].p), static_cast<void* const*>(w.dn_scale[1].p),
                                   (uint32_t)shape_.hidden, (uint32_t)shape_.hidden, (uint32_t)w0, (uint32_t)w1,
                                   (uint32_t)mid_w(), (uint32_t)shape_.hidden, static_cast<uint16_t*>(out_down), 0u);
                else return -EINVAL;
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
        if (trace_) hipLaunchKernelGGL(lane_stage_stamp, dim3(1), dim3(1), 0, w.stream, q, 1u);   /* GEMMs done */
        if (!fuse)
            hipLaunchKernelGGL(lane_route_inverse, route_grid, block, 0, w.stream,
                               static_cast<const int32_t*>(w.d_sorted.p),
                               static_cast<int32_t*>(w.d_inv.p), routes, slots);
        /* R3_MOE_WORKER_COMBINE, chunked (RAD42): the large lane publishes its rows chunk by chunk for helper 0's combine. */
        LaneChunks chunks{nullptr, nullptr, 0u};
        if (chunk_helper_ < kCombineHelpers && !lane_is_small(lane)) {
            if (!hip_ok(hipMemsetAsync(w.d_chunk_cnt.p, 0, kCombineChunks * sizeof(uint32_t), w.stream), "lane chunk counters"))
                return -EIO;
            chunks = LaneChunks{static_cast<uint32_t*>(w.d_chunk_cnt.p),
                                static_cast<uint32_t*>(dev_at(&hhost->chunks[slot_consumer(rank, hhost->rank_count)].seq[chunk_helper_][0])),
                                sequence};
        }
        if (vecio)
            hipLaunchKernelGGL(lane_reduce_tokens_v, dim3(rows), block, 0, w.stream,
                               static_cast<const uint16_t*>(w.d_down.p),
                               static_cast<const int32_t*>(w.d_inv.p),
                               static_cast<const uint16_t*>(w.d_ew.p),
                               dst, rows, shape_.hidden, poison_, lane_shape_, chunks);
        else
            hipLaunchKernelGGL(lane_reduce_tokens, dim3(rows), block, 0, w.stream,
                               static_cast<const uint16_t*>(w.d_down.p),
                               static_cast<const int32_t*>(w.d_inv.p),
                               static_cast<const uint16_t*>(w.d_ew.p),
                               dst, rows, shape_.hidden, poison_, lane_shape_, chunks);
        if (!hip_ok(hipGetLastError(), "lane reduce tokens")) return -EIO;

        if (sample && w.sample_pending && w.sample_sequence == sequence &&
            !hip_ok(hipEventRecord(w.sample_stop, w.stream), "lane timing stop"))
            return -EIO;
        hipLaunchKernelGGL(lane_publish_done, dim3(1), dim3(1), 0, w.stream,
                           q, done_seq, sequence, static_cast<const uint32_t*>(w.d_publish_limit.p));
        if (!hip_ok(hipGetLastError(), "lane publish done")) return -EIO;
        return 0;
    }

    bool collect_lane_timing(uint32_t lane, uint32_t done_sequence,
                             double* sampled_us, uint64_t* samples,
                             double* service_us = nullptr, uint64_t* service_rows = nullptr) {
        if (service_us) *service_us = 0;
        if (service_rows) *service_rows = 0;
        if (lane >= r3::moe_shm::kLanes || !sampled_us || !samples) return false;
        LaneWork& w = lane_[lane];
        if (!w.sample_pending || r3::moe_shm::seq_after(w.sample_sequence, done_sequence))
            return true;
        float ms = 0.0f;
        if (!hip_ok(hipEventElapsedTime(&ms, w.sample_start, w.sample_stop),
                    "lane timing elapsed"))
            return false;
        *sampled_us += (double)ms * 1000.0;
        ++*samples;
        hipEvent_t mark[LaneWork::kStages + 2] = {w.sample_start, w.stage[0], w.stage[1], w.stage[2], w.stage[3],
                                                  w.sample_stop};
        double stage_total = 0;
        bool complete_stages = true;
        for (int i = 0; i <= LaneWork::kStages; ++i) {
            float part = 0.0f;
            if (hipEventElapsedTime(&part, mark[i], mark[i + 1]) == hipSuccess) {
                w.stage_us[i] += (double)part * 1000.0;
                stage_total += (double)part * 1000.0;
            } else complete_stages = false;
        }
        w.sample_pending = false;
        /* The sampled request's per-expert row counts (d_counts; this lane has synced): the run-length histogram the
         * grouped GEMM's tile choice depends on. Sampled requests only, so the copy is off the hot path. */
        std::vector<uint32_t> counts(experts());
        if (hipMemcpy(counts.data(), w.d_counts.p, counts.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess) {
            uint64_t routed_rows = 0;
            for (uint32_t e = 0; e < experts(); ++e) {
                const uint32_t c = counts[e];
                routed_rows += c;
                if (!c) continue;
                const int b = c == 1 ? 0 : c == 2 ? 1 : c <= 4 ? 2 : c <= 8 ? 3 : c <= 16 ? 4 : c <= 32 ? 5 : c <= 64 ? 6 : 7;
                ++w.run_hist[b];
            }
            // This exact sum, unlike the coarse run_hist bins, normalises
            // the existing five stage times by routed expert-rows. Publish
            // only when the sampled HIP events and counts are BOTH complete.
            if (complete_stages && routed_rows && service_us && service_rows) {
                *service_us = stage_total;
                *service_rows = routed_rows;
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

    /* RAD20: a sample recorded on a pre-queued wait released at burst end timed no request; forget it. */
    void drop_lane_sample(uint32_t lane) {
        if (lane < r3::moe_shm::kLanes) lane_[lane].sample_pending = false;
    }

    /* RAD20 park: lane_publish_done publishes no sequence above `limit` (the engine's last request) while the lane's
     * unconsumed waits are released; UINT32_MAX again once drained. Device memory, so the check costs no PCIe read. */
    /* R3_MOE_WORKER_COMBINE (RAD42): this helper's index; its large lane then publishes Header::chunks.seq[k]. */
    void set_chunk_helper(uint32_t k) { chunk_helper_ = k; }
    bool set_publish_limit(uint32_t lane, uint32_t limit) {
        return lane < r3::moe_shm::kLanes && lane_[lane].d_publish_limit.p &&
               hip_ok(hipMemcpy(lane_[lane].d_publish_limit.p, &limit, sizeof(limit), hipMemcpyHostToDevice),
                      "lane publish limit");
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
    std::array<LaneWork, r3::moe_shm::kLanes> lane_{};
    uint32_t gu_code_dtype_ = 0, gu_scale_dtype_ = 0;
    uint32_t dn_code_dtype_ = 0, dn_scale_dtype_ = 0;
    const RadKernelInfo* moe_ = nullptr;
    const RadKernelInfo* had_quant_ = nullptr;
    const RadKernelInfo* gated_quant_ = nullptr;
    const RadKernelInfo* gated_had_ = nullptr;   // the rotated form, so the lane kernel knows which it replaces
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
    static uint64_t lane_bytes(const LaneWork& w) {
        const Buffer* b[] = {&w.d_counts, &w.d_cursor, &w.d_flat,
            &w.d_sorted, &w.d_off, &w.d_ew, &w.d_inv, &w.d_dbg,
            &w.d_rowmask, &w.d_act, &w.d_aq, &w.d_as, &w.d_gu,
            &w.d_mid, &w.d_mid_s, &w.d_down, &w.d_stats,
            &w.scratch, &w.d_chunk_cnt, &w.d_publish_limit};
        uint64_t n = 0;
        for (const Buffer* p : b) n += p->bytes;
        for (unsigned i = 0; i < 2; ++i)
            n += w.gu_code[i].bytes + w.gu_scale[i].bytes +
                 w.dn_code[i].bytes + w.dn_scale[i].bytes;
        return n;
    }

    struct HipSlotAllocator {
        bool allocate(uint64_t bytes, void** out) {
            return bytes <= SIZE_MAX && hip_ok(hipMalloc(out, size_t(bytes)), "hipMalloc expert slot");
        }
        void release(void* p) { (void)hipFree(p); }
    } slot_allocator_;
    std::unique_ptr<r3::worker_slots::Pool<HipSlotAllocator>> slots_;
    uint64_t free_before_lanes_ = 0;
    bool lane_free_sampled_ = false;
    std::vector<int32_t> initial_locals_; // bootstrap's retained per-layer logical index
    std::vector<int32_t> fixed_locals_; // engine's immutable per-layer expert map
    uint8_t* move_host_ = nullptr; // pinned mirror of one physical staging stride
    hipStream_t copy_stream_ = nullptr; // never used by serving streams
    hipEvent_t copy_event_ = nullptr; // a fresh event for each gained expert
    uint32_t move_expert_ = UINT32_MAX;
    bool staging_upload_ = false, copy_pending_ = false;
    uint64_t move_bytes_staged_ = 0;
    uint32_t active_generation_ = 0;
    uint8_t* loading_base_ = nullptr;
    uint64_t loading_size_ = 0, loading_used_ = 0;

    HostBounce load_bounce_;
    Buffer d_layer_tables_, d_expert_slot_;
    std::array<uint32_t, r3::moe_shm::kSmallClasses> class_rows_{};   /* enabled small-lane (pre-queued) step sizes */
    uint32_t class_count_ = 0;
    /* FULL WIDTH (RAD49, for every engine width): the helper stores each expert whole (one copy, the relayout's own form)
     * and runs one GEMM chain at the full intermediate width; behind TP the layer's consumer rank adds that row once. */
    int64_t mid_w() const { return (int64_t)shape_.intermediate; }
    uint32_t large_rows_ = 0;   /* large-lane capacity */
    uint32_t poison_ = 0;
    bool gemv_ = true;          /* R3_MOE_WORKER_GEMV: decode GEMV on the small lane (RAD05); 0 = off */
    uint32_t gemv_large_rows_ = 0; /* R3_MOE_WORKER_GEMV_LARGE_ROWS: GEMV for large-lane requests this small */
    int gemv_h2_env_ = -1;      /* R3_MOE_WORKER_GEMV_H2 as set: -1 unset, else 0 (wide kernel), 1, 2 or 4 rows a block */
    int gemv_h2_ = 0;           /* R3_MOE_WORKER_GEMV_H2: 1 packed f16 dot products in the FP8 decode GEMV, 2 the same with two rows a block (RAD14) */
    bool gemv_w4h2_ = true;     /* R3_MOE_WORKER_GEMV_W4H2: packed f16 W4 decode GEMV (RAD14), default on */
    bool gemv_mr_ = false;      /* R3_MOE_WORKER_GEMV_MR: the wide FP8 GEMV batching an expert's rows (RAD14) */
    int gemv_wide_env_ = -1;    /* R3_MOE_WORKER_GEMV_WIDE as set: -1 unset, 0 off, 1 on */
    bool gemv_wide_ = false;    /* R3_MOE_WORKER_GEMV_WIDE: sixteen-byte FP8 loads in the decode GEMV (RAD14) */
    Rad46Sweep rad46_sweep_ = Rad46Sweep::Production; // replay-sweep overrides only between runs
    bool quad_perm_ = false;   // replay-sweep "quad_perm": the production forms with the register W4 decoder (RAD49)
    bool rows_ = false;         /* R3_MOE_WORKER_ROWS: few-rows-a-expert kernel on the large lane (RAD06) */
    uint32_t chunk_helper_ = UINT32_MAX;   /* R3_MOE_WORKER_COMBINE: this helper's index for the chunk words (RAD42) */
    bool trace_ = r3_serving::settings().trace > 0;   /* RAD40 stage stamps */
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

    // Both the plugin file and installation root are manifest-validated.
    std::string plugin_path() const {
        const auto& conf = r3_serving::settings();
        if (!conf.plugin.empty()) return conf.plugin;
        return conf.radiance_home + "/kernels/" + conf.plugin_file;
    }

    bool load_plugin() {
        const std::string path = plugin_path();
        if (path.empty()) return false;
        so_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!so_) {
            std::fprintf(stderr, "r3_moe_worker: dlopen %s: %s\n", path.c_str(), dlerror());
            return false;
        }
        if (!r3_serving::configure_plugin(so_, r3_serving::settings())) return false;
        uint32_t (*abi)() = nullptr;
        const RadPluginInfo* (*info)() = nullptr;
        int (*count)() = nullptr;
        const RadKernelInfo* (*at)(int) = nullptr;
        int (*open_fn)() = nullptr;
        uint64_t (*layout)() = nullptr;
        if (!symbol("rad_plugin_abi_version", &abi) || !symbol("rad_plugin_abi_layout", &layout) ||
            !symbol("rad_plugin_info", &info) || !symbol("rad_kernel_count", &count) ||
            !symbol("rad_kernel_at", &at)) return false;
        /* rad_plugin_open/close are optional in the ABI (rad_abi.h); libr4d exports neither. */
        (void)symbol("rad_plugin_open", &open_fn, false);
        (void)symbol("rad_plugin_close", &plugin_close_, false);
        if (abi() != RAD_ABI_VERSION) {
            std::fprintf(stderr, "r3_moe_worker: libr4d ABI %u, helper ABI %u\n",
                         abi(), (unsigned)RAD_ABI_VERSION);
            return false;
        }
        /* The same version number can name two struct layouts; the engine's loader refuses a plugin whose layout
         * fingerprint differs, and so does the helper. */
        if (layout() != rad_abi_layout()) {
            std::fprintf(stderr, "r3_moe_worker: %s ABI layout %016llx, helper layout %016llx\n", path.c_str(),
                         (unsigned long long)layout(), (unsigned long long)rad_abi_layout());
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
        gated_had_ = gated_quant_;
        gather_ = find("moe_gather", "moe_gather_bf16");
        if (const RadKernelInfo* r = find("moe_gemm_q", "moe_gemm_fp8a8"); r && r->name && std::strcmp(r->name, "moe_gemm_fp8a8") == 0)
            moe_fp8_ = r;
        quant_plain_ = find("quant_act_fp8", "quant_act_fp8");
        gated_plain_ = find("gated_quant_fp8", "gated_quant_fp8");
        if (!moe_ || !moe_->layout || !moe_->relayout || !had_quant_ || !gated_quant_ || !gather_) {
            std::fprintf(stderr, "r3_moe_worker: installed libr4d lacks required MoE/quant rows\n");
            return false;
        }
        const auto& conf = r3_serving::settings();
        if (conf.moe_form) {
            auto pin = reinterpret_cast<void (*)(int)>(dlsym(so_, "r4d_moe_gemm_pin_prefill"));
            if (!pin) {
                std::fprintf(stderr, "r3_moe_worker: plugin lacks r4d_moe_gemm_pin_prefill\n");
                return false;
            }
            pin((int)conf.moe_form);
            std::fprintf(stderr, "r3_moe_worker: grouped MoE GEMM pinned to the %s form\n",
                         conf.moe_form == 2 ? "16-row decode" : "32-row prefill");
        }
        {
            auto fn = reinterpret_cast<void (*)(int)>(dlsym(so_, "r4d_moe_gemm_pin_down_tile"));
            if (!fn) {
                std::fprintf(stderr, "r3_moe_worker: plugin lacks r4d_moe_gemm_pin_down_tile\n");
                return false;
            }
            fn((int)conf.handoff.moe_down_tile);
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_MOE_DOWN_TILE=%u applied\n",
                         conf.handoff.moe_down_tile);
        }
        if (conf.moe_long_rows) {
            auto fn = reinterpret_cast<void (*)(int)>(dlsym(so_, "r4d_moe_gemm_pin_long_rows"));
            if (!fn) {
                std::fprintf(stderr, "r3_moe_worker: plugin lacks r4d_moe_gemm_pin_long_rows\n");
                return false;
            }
            fn((int)conf.moe_long_rows);
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_MOE_LONG_ROWS=%u applied\n",
                         conf.moe_long_rows);
        }
        return true;
    }

    static size_t arena_round(int64_t n) { return ((size_t)(n > 0 ? n : 1) + 255u) & ~(size_t)255u; }

    bool stored_bytes(const RadFileEntry* e, bool gate_up, int64_t* code, int64_t* scale) const {
        if (!e || !code || !scale) return false;
        int64_t nr = 1, nc = 1;
        rad_enc_view(e->rank, e->shape, &nr, &nc);
        const int64_t width = mid_w();
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

    uint64_t slot_cost(uint32_t l, uint32_t e) const {
        uint64_t total = 0;
        for (uint32_t weight = 0; weight < 2; ++weight) {
            int64_t cb = 0, sb = 0;
            if (!stored_bytes(model_.weight(l, e, weight), weight == 0, &cb, &sb) || cb <= 0 || sb <= 0) return 0;
            const uint64_t add = arena_round(cb) + arena_round(sb);
            if (add > UINT64_MAX - total) return 0;
            total += add;
        }
        return total;
    }
    void* arena_take(size_t n) {
        const size_t r = arena_round((int64_t)n);
        if (!loading_base_ || loading_used_ > loading_size_ ||
            r > loading_size_ - loading_used_) {
            std::fprintf(stderr, "r3_moe_worker: resident expert slot overflow\n");
            return nullptr;
        }
        void* p = loading_base_ + loading_used_;
        loading_used_ += r;
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

    bool transfer_plane(void* dst, const void* src, size_t bytes, const char* label) {
        if (!staging_upload_) return hip_ok(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice),label);
        // Device pointers are opaque host addresses. Compare integer offsets,
        // not C++ pointers from unrelated virtual address spaces.
        const uintptr_t base=reinterpret_cast<uintptr_t>(loading_base_);
        const uintptr_t address=reinterpret_cast<uintptr_t>(dst);
        if (!move_host_ || !loading_base_ || address<base ||
            address-base>loading_size_ ||
            bytes>loading_size_-(address-base)) return false;
        std::memcpy(move_host_+(address-base),src,bytes);
        return true;
    }
    bool upload_weight(const RadFileEntry* e, bool gate_up, StoredWeight* out) {
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
            if (!stored_bytes(e, gate_up, &cb, &sb) || (int64_t)code.size() != cb || (int64_t)scale.size() != sb ||
                p.code.bytes != cb || p.scale.bytes != sb) {
                std::fprintf(stderr, "r3_moe_worker: unexpected fp8a8 plane geometry for %s\n", gate_up ? "gate/up" : "down");
                return false;
            }
            StoredWeight& w = *out;
            w.code = arena_take(code.size());
            w.scale = arena_take(scale.size());
            if (!w.code || !w.scale ||
                !transfer_plane(w.code, code.data(), code.size(), "upload expert codes") ||
                !transfer_plane(w.scale, scale.data(), scale.size(), "upload expert scales"))
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

        /* The relayout's own form IS the stored form: full width (RAD49), no TP slice is cut out of it. */
        int64_t cb = 0, sb = 0;
        if (!stored_bytes(e, gate_up, &cb, &sb) || (int64_t)full_code.size() != cb || (int64_t)full_scale.size() != sb)
            return false;
        StoredWeight& w = *out;
        w.code = arena_take(full_code.size());
        w.scale = arena_take(full_scale.size());
        if (!w.code || !w.scale ||
            !transfer_plane(w.code, full_code.data(), full_code.size(), "upload expert codes") ||
            !transfer_plane(w.scale, full_scale.data(), full_scale.size(), "upload expert scales"))
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

    bool build_one_table(const std::vector<void*>& host, void** dev) {
        if (!*dev) *dev = alloc_persistent(host.size() * sizeof(void*));
        return *dev && hip_ok(hipMemcpy(*dev, host.data(), host.size() * sizeof(void*),
                                        hipMemcpyHostToDevice), "upload expert pointer table");
    }

    bool build_layer_table(uint32_t l, const int32_t* slot_of_expert, uint32_t generation) {
        using namespace r3::moe_shm;
        if (l >= model_.layers || !slot_of_expert ||
            generation >= r3::worker_tables::kGenerations) return false;
        const size_t nclass = experts() / 2;
        std::vector<uint16_t> slot_map(experts(), 0);
        std::fill(slot_map.begin(), slot_map.end(), (uint16_t)0xffffu);
        for (uint32_t e = 0; e < experts(); ++e) {
            const int32_t es = slot_of_expert[e];
            if (es >= 0) slot_map[e] = (uint16_t)es;
        }

        LayerTables& lt = tables_[r3::worker_tables::table_index(generation,l,model_.layers)];
        lt.gu_code_dtype = gu_code_dtype_; lt.gu_scale_dtype = gu_scale_dtype_;
        lt.dn_code_dtype = dn_code_dtype_; lt.dn_scale_dtype = dn_scale_dtype_;
        for (uint32_t parity = 0; parity < 2; ++parity) {
            std::vector<void*> guc(nclass, nullptr), gus(nclass, nullptr);
            std::vector<void*> dnc(nclass, nullptr), dns(nclass, nullptr);
            for (uint32_t e = 0; e < experts(); ++e) {
                const int32_t es = slot_of_expert[e];
                if (es < 0 || ((uint32_t)es & 1u) != parity) continue;
                const Expert& x = experts_[(size_t)l * experts() + e];
                if (!x.live) continue;
                const size_t ix = (size_t)es / 2;
                guc[ix] = x.gu.code; gus[ix] = x.gu.scale;
                dnc[ix] = x.dn.code; dns[ix] = x.dn.scale;
            }
            if (!build_one_table(guc, &lt.gu_code[parity]) ||
                !build_one_table(gus, &lt.gu_scale[parity]) ||
                !build_one_table(dnc, &lt.dn_code[parity]) ||
                !build_one_table(dns, &lt.dn_scale[parity]))
                return false;
        }

        const size_t slot_off = (size_t)l * experts() * sizeof(uint16_t);
        if (generation==0 && !layer_loaded_[l] &&
            !hip_ok(hipMemcpy(static_cast<uint8_t*>(d_expert_slot_.p) + slot_off,
                              slot_map.data(),slot_map.size()*sizeof(uint16_t),
                              hipMemcpyHostToDevice),
                    "upload immutable runtime expert slot map"))
            return false;

        DeviceLayerTables dt{};
        for (uint32_t parity = 0; parity < 2; ++parity) {
            dt.gu_code[parity] = lt.gu_code[parity];
            dt.gu_scale[parity] = lt.gu_scale[parity];
            dt.dn_code[parity] = lt.dn_code[parity];
            dt.dn_scale[parity] = lt.dn_scale[parity];
        }
        dt.expert_slot = static_cast<const uint16_t*>(d_expert_slot_.p) +
            size_t(l)*experts();
        return hip_ok(hipMemcpy(static_cast<uint8_t*>(d_layer_tables_.p) +
                                    r3::worker_tables::table_index(generation, l, model_.layers) *
                                        sizeof(DeviceLayerTables),
                                &dt, sizeof(dt), hipMemcpyHostToDevice),
                      "upload immutable generation layer table");
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

        const int64_t w0 = mid_w();
        const int64_t w1 = mid_w();
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

    bool ensure_lane_work(LaneWork& w, uint32_t rows, uint32_t routes, uint32_t tr) {
        return w.d_counts.ensure((size_t)experts() * 4) &&
               w.d_rowmask.ensure((size_t)rows) &&
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
};

} // namespace r3_moe_gpu
