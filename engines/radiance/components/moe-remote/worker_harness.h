#pragma once

// Request capture, trace dump, replay and --compare. Included only by r3_moe_worker.cpp, at the position the code had in the single file (RCS03, pure move).

namespace {
/* REQUEST CAPTURE / REPLAY (RAD06). R3_MOE_WORKER_CAPTURE=<dir> writes real large-lane requests (every
 * R3_MOE_WORKER_CAPTURE_EVERY-th, at most R3_MOE_WORKER_CAPTURE_MAX) as <dir>/req-<lane>-<seq>.bin:
 * CaptureHeader (version 3: the model's shape is in it), the layer's runtime slot map (header.experts int32), the routes, the
 * quantised activation rows (E4M3 codes then an f32 scale a group).
 * `--replay MODEL FILE` runs --compare's machinery on that request instead of synthetic routing. */
struct CaptureHeader {
    uint32_t magic, version, layer, rows, route_count, hidden, experts, top_k, act_group;
};
constexpr uint32_t kCaptureVersion = 3;
constexpr uint32_t kCaptureMagic = 0x52335243u;  // "CR3R"

void capture_request(void* map, const Request* q, uint32_t lane, uint32_t seq) {
    static const char* dir = std::getenv("R3_MOE_WORKER_CAPTURE");
    if (!dir || !*dir) return;
    static const uint32_t every = [] {
        const char* v = std::getenv("R3_MOE_WORKER_CAPTURE_EVERY");
        return v && *v ? std::max(1, std::atoi(v)) : 37;
    }();
    static const uint32_t max_n = [] {
        const char* v = std::getenv("R3_MOE_WORKER_CAPTURE_MAX");
        return v && *v ? std::max(1, std::atoi(v)) : 16;
    }();
    static std::atomic<uint32_t> written{0};
    if (seq % every != 0 || written.load() >= max_n) return;
    const Header* sh = static_cast<const Header*>(map);
    CaptureHeader hd{kCaptureMagic, kCaptureVersion, load(&q->layer), load(&q->rows), load(&q->route_count),
                     sh->hidden, sh->expert_count, sh->top_k, sh->activation_group};
    if (hd.layer >= kMaxLayers || hd.rows == 0 || hd.rows > kMaxRows) return;
    const std::string path = std::string(dir) + "/req-" + std::to_string(lane) + "-" + std::to_string(seq) + ".bin";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fwrite(&hd, sizeof(hd), 1, f);
    std::fwrite(layer_slots(map, hd.layer), sizeof(int32_t), hd.experts, f);
    std::fwrite(ptr<Route>(map, q->route_offset), sizeof(Route), hd.route_count, f);
    /* The engine publishes the quantised activation (E4M3 plus a scale a group), not bf16: the bf16 plane of the map
     * is empty on the large lane. Version 1 captures stored that empty plane and cannot be replayed. */
    std::fwrite(ptr<uint8_t>(map, q->activation_q_offset), 1, (size_t)hd.rows * hd.hidden, f);
    std::fwrite(ptr<float>(map, q->activation_s_offset), sizeof(float), (size_t)hd.rows * (hd.hidden / hd.act_group), f);
    std::fclose(f);
    written.fetch_add(1);
}

/* R3_MOE_WORKER_TRACE=N: print the stamps of every Nth finished request ("TRACE,..." on stderr). The engine writes its
 * wait stamps right after it sees the helper's done word, so a request is read once the lane has moved on to the next
 * one. Engine stamps (0-2) and helper stamps (3-4) come from different devices' clocks and are only ever subtracted
 * within their own group; wall_clock64 runs at hipDeviceAttributeWallClockRate (100 MHz on these cards). */
void trace_dump(void* map, uint32_t lane, uint32_t seq) {
    static const uint32_t every = [] {
        const char* v = std::getenv("R3_MOE_WORKER_TRACE");
        return v && *v ? (uint32_t)std::max(0, std::atoi(v)) : 0u;
    }();
    if (!every || seq == 0 || seq % every != 0) return;
    const Request* q = lane_request(map, lane, seq);
    if (!q || load(&q->sequence) != seq) return;
    uint64_t t[5];
    for (int i = 0; i < 5; ++i) t[i] = q->trace[i];
    if (!t[0] || !t[1] || !t[2] || !t[3] || !t[4]) return;
    const double us = 1.0 / 100.0;   /* a tick is 10 ns */
    /* wall_ns: this host's clock when the line is printed, so a benchmark can bucket lines by the round that produced them. */
    const unsigned long long wall_ns = (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::fprintf(stderr, "TRACE,lane,%u,seq,%u,layer,%u,rows,%u,routes,%u,local_us,%.2f,wait_us,%.2f,helper_gpu_us,%.2f,enter,%llu,wall_ns,%llu\n",
                 lane, seq, q->layer, q->rows, q->route_count,
                 (double)(int64_t)(t[1] - t[0]) * us, (double)(int64_t)(t[2] - t[1]) * us,
                 (double)(int64_t)(t[4] - t[3]) * us, (unsigned long long)t[1], wall_ns);
}

struct ReplayInput {
    CaptureHeader hd{};
    std::vector<int32_t> slots;
    std::vector<Route> routes;
    std::vector<uint8_t> actq;
    std::vector<float> acts;
};
const ReplayInput* g_replay = nullptr;
/* --replay-sweep with R3_MOE_SWEEP_FAST=1: no CPU reference; each variant's output is compared bit for bit with the
 * first variant's of the same capture. */
bool g_sweep_fast = false;

bool load_replay(const char* path, ReplayInput* out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    bool ok = std::fread(&out->hd, sizeof(out->hd), 1, f) == 1 && out->hd.magic == kCaptureMagic && out->hd.version == kCaptureVersion &&
              out->hd.hidden > 0 && out->hd.hidden <= 16384 && out->hd.experts > 0 && out->hd.experts <= 4096 && out->hd.top_k > 0 &&
              out->hd.top_k <= kMaxTopK && out->hd.act_group > 0 && out->hd.hidden % out->hd.act_group == 0 &&
              out->hd.layer < kMaxLayers && out->hd.rows > 0 && out->hd.rows <= kMaxRows && out->hd.route_count <= kMaxRoutes;
    if (ok) {
        out->slots.resize(out->hd.experts);
        ok = std::fread(out->slots.data(), sizeof(int32_t), out->slots.size(), f) == out->slots.size();
    }
    if (ok) {
        out->routes.resize(out->hd.route_count);
        out->actq.resize((size_t)out->hd.rows * out->hd.hidden);
        out->acts.resize((size_t)out->hd.rows * (out->hd.hidden / out->hd.act_group));
        ok = std::fread(out->routes.data(), sizeof(Route), out->routes.size(), f) == out->routes.size() &&
             std::fread(out->actq.data(), 1, out->actq.size(), f) == out->actq.size() &&
             std::fread(out->acts.data(), sizeof(float), out->acts.size(), f) == out->acts.size();
    }
    std::fclose(f);
    return ok;
}

int compare_mode(const Container& model, r3_moe_gpu::Runner& gpu, const Owned& owned,
                 uint32_t layer, uint32_t rows, const SmallClasses& classes, uint32_t lane_large_rows,
                 bool synthetic_protect) {
    /* the request is built for the model's own shape (the lane checks every request against it) */
    const uint32_t hid = gpu.shape_.hidden, topk = gpu.shape_.top_k;
    if (g_replay) {  // a captured real request replaces the synthetic layer, rows, routes and activations
        if (g_replay->hd.hidden != hid || g_replay->hd.experts != model.experts || g_replay->hd.top_k != topk ||
            g_replay->hd.act_group != kActGroup) {
            std::fprintf(stderr, "r3_moe_worker: the capture is for hidden %u, %u experts, top-k %u; this model is hidden %u, %u experts, top-k %u\n",
                         g_replay->hd.hidden, g_replay->hd.experts, g_replay->hd.top_k, hid, model.experts, topk);
            return 1;
        }
        layer = g_replay->hd.layer;
        rows = g_replay->hd.rows;
    }
    if (layer >= model.layers || rows == 0 || rows > lane_large_rows) {
        std::fprintf(stderr, "r3_moe_worker: --compare rows %u above R3_MOE_WORKER_MAX_ROWS %u\n",
                     rows, lane_large_rows);
        return 2;
    }
    std::vector<int32_t> runtime_slot(model.experts, -1);
    static constexpr uint32_t kSyntheticProtected[8] = {1, 17, 63, 127, 191, 255, 319, 511};
    uint32_t next_slot = 0;
    for (uint32_t e = 0; e < model.experts; ++e) {
        if (!model.weight(layer, e, 0) || !model.weight(layer, e, 1)) continue;
        bool protect = false;
        if (synthetic_protect)
            for (uint32_t p : kSyntheticProtected) protect = protect || e == p;
        if (!protect) runtime_slot[e] = (int32_t)next_slot++;
    }
    if (g_replay) runtime_slot = g_replay->slots;

    std::vector<uint32_t> remote_expert, local_expert;
    std::vector<uint8_t> owned_here(model.experts, 0);
    for (auto [l, e] : owned.expert) {
        if (l != layer || runtime_slot[e] < 0) continue;
        remote_expert.push_back(e);
        owned_here[e] = 1;
    }
    for (uint32_t e = 0; e < model.experts; ++e)
        if (!owned_here[e] && model.weight(layer, e, 0) && model.weight(layer, e, 1))
            local_expert.push_back(e);
    if (remote_expert.size() < 2 || local_expert.empty()) {
        std::fprintf(stderr,
                     "r3_moe_worker: --compare layer %u needs >=2 owned and >=1 local experts\n",
                     layer);
        return 1;
    }
    if (synthetic_protect) {
        bool shifted_parity = false;
        for (uint32_t e : remote_expert)
            shifted_parity = shifted_parity ||
                (((uint32_t)runtime_slot[e] & 1u) != (e & 1u));
        if (!shifted_parity) {
            std::fprintf(stderr,
                         "r3_moe_worker: --compare synthetic protection did not shift an owned TP class\n");
            return 1;
        }
    }

    const uint32_t remote_per_token =
        std::min<uint32_t>(4, (uint32_t)remote_expert.size());
    const uint32_t total = rows * topk;
    std::vector<int32_t> full_expert(total, -1);
    std::vector<uint16_t> full_weight(total, 0);
    std::vector<uint8_t> is_remote(total, 0);
    std::vector<uint16_t> activation((size_t)rows * hid);

    uint64_t rng = 0x8f3f73b5cf1c9ad1ull ^ ((uint64_t)layer << 32) ^ rows;
    auto rnd = [&]() -> uint32_t {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        return (uint32_t)(rng >> 16);
    };
    for (size_t i = 0; i < activation.size(); ++i) {
        const float x = ((int32_t)(rnd() & 0xffffu) - 32768) * (1.0f / 32768.0f);
        activation[i] = to_bf16(x);
    }

    std::vector<Route> published;
    published.reserve((size_t)rows * remote_per_token);
    for (uint32_t t = 0; t < rows; ++t) {
        uint32_t ri = 0, li = 0;
        for (uint32_t slot = 0; slot < topk; ++slot) {
            const uint32_t flat = t * topk + slot;
            const bool remote = ((slot + 3 * t) % topk) < remote_per_token;
            const float w = (float)(slot + 1) / 55.0f;
            full_weight[flat] = to_bf16(w);
            if (remote) {
                const uint32_t e = remote_expert[(t * 5u + ri) % remote_expert.size()];
                full_expert[flat] = (int32_t)e;
                is_remote[flat] = 1;
                published.push_back(Route{(int32_t)t, (int32_t)e, full_weight[flat],
                                          (uint16_t)slot});
                ++ri;
            } else {
                const uint32_t e = local_expert[(t * 7u + li) % local_expert.size()];
                full_expert[flat] = (int32_t)e;
                ++li;
            }
        }
    }
    if (g_replay) {
        published = g_replay->routes;
        /* The bf16 plane (the host reference's input) from the captured quantised planes; the lane itself reads the
         * captured q and s as they are. */
        for (size_t i = 0; i < activation.size(); ++i) {
            const uint32_t c = g_replay->actq[i];
            const int e = (int)((c >> 3) & 0xfu), m = (int)(c & 7u);
            const float mag = e ? std::ldexp(1.0f + m / 8.0f, e - 7) : std::ldexp(m / 8.0f, -6);
            activation[i] = to_bf16(((c & 0x80u) ? -mag : mag) * g_replay->acts[i / g_replay->hd.act_group]);
        }
    }
    const uint32_t nr = (uint32_t)published.size();

    r3::moe_shm::LayoutShape shape{};
    Layout lay{};
    if (!segment_shape(model, r3::moe_shm::kLegacySlots, &shape, &lay)) return 1;
    void* map = mmap(nullptr, lay.bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (map == MAP_FAILED) {
        std::fprintf(stderr, "r3_moe_worker: --compare mmap: %s\n", std::strerror(errno));
        return 1;
    }
    if (!init(map, model.layers, shape, r3::moe_shm::kLegacySlots)) return 1;
    publish_owned(map, owned, 0);
    Header* h = static_cast<Header*>(map);
    h->helper_count = 1;
    std::memcpy(layer_slots(map, layer), runtime_slot.data(), runtime_slot.size() * sizeof(int32_t));
    store(&h->layer_map_ready[layer], 1);
    if (!gpu.load_layer(layer, owned.expert, runtime_slot.data())) {
        std::fprintf(stderr, "r3_moe_worker: --compare layer %u runtime table build failed\n", layer);
        munmap(map, lay.bytes);
        return 1;
    }
    store(&h->layer_map_consumed[layer], 1);
    h->small_class_count = classes.count;
    for (uint32_t c = 0; c < classes.count; ++c) h->small_class_rows[c] = classes.rows[c];
    if (!gpu.prepare_lanes(classes.rows, classes.count, lane_large_rows, served_lanes(1, 0, classes.count))) {
        munmap(map, lay.bytes);
        return 1;
    }
    if (synthetic_protect)
        std::printf("synthetic_protect layer %u removed=8 regular=%u\n", layer, next_slot);
    bool pinned = false;
    void* dev_map = nullptr;
    if (!r3_moe_gpu::hip_ok(
            hipHostRegister(map, lay.bytes, hipHostRegisterPortable | hipHostRegisterMapped),
            "--compare hipHostRegister")) {
        munmap(map, lay.bytes);
        return 1;
    }
    pinned = true;
    if (!r3_moe_gpu::hip_ok(hipHostGetDevicePointer(&dev_map, map, 0),
                            "--compare hipHostGetDevicePointer")) {
        (void)hipHostUnregister(map);
        munmap(map, lay.bytes);
        return 1;
    }

    bool bad = false;
    constexpr uint32_t sequence = 1;
    for (uint32_t rank = 0; rank < kMaxSlots; ++rank) {
        const uint32_t lane = lane_of(rank, step_class(h, rows));
        Request* q = lane_request(map, lane, sequence);
        q->completion_sequence = 0;
        q->status = 0;
        q->layer = layer;
        q->rank = rank;
        q->rank_count = kMaxEngineRanks;
        q->rows = rows;
        q->hidden = hid;
        q->route_count = nr;
        q->row_base = 0;
        q->route_offset = h->route_offset[rank];
        q->activation_offset = h->activation_offset[rank];
        q->activation_q_offset = h->activation_offset[rank] + kActivationBf16Bytes;
        q->activation_s_offset =
            h->activation_offset[rank] + kActivationBf16Bytes + kActivationQBytes;
        q->activation_stride = hid * sizeof(uint16_t);
        q->activation_q_stride = hid;
        q->activation_s_stride = (hid / kActGroup) * sizeof(float);
        for (uint32_t r = 0; r < kMaxEngineRanks; ++r) {
            q->result_offset[r] = h->result_offset[result_index(rank, r)];
            q->result_stride[r] = hid * sizeof(uint16_t);
        }
        q->sequence = sequence;
        std::memcpy(ptr<Route>(map, q->route_offset), published.data(),
                    published.size() * sizeof(Route));
        std::memcpy(ptr<uint16_t>(map, q->activation_offset), activation.data(),
                    activation.size() * sizeof(uint16_t));
        if (g_replay) {
            std::memcpy(ptr<uint8_t>(map, q->activation_q_offset), g_replay->actq.data(), g_replay->actq.size());
            std::memcpy(ptr<float>(map, q->activation_s_offset), g_replay->acts.data(),
                        g_replay->acts.size() * sizeof(float));
        } else if (!gpu.stage_compare_activation(map, lane, activation.data(), rows)) {
            std::fprintf(stderr,
                         "r3_moe_worker: --compare rank %u could not stage exact q/s\n", rank);
            bad = true;
            break;
        }

        /* Queue the production lane before publishing the sequence, as the service worker does. */
        const int erc = gpu.enqueue_lane(map, dev_map, lane, sequence,
                                         lane_is_small(lane) ? h->small_class_rows[lane_class(lane)] : rows, false);
        if (erc != 0) {
            std::fprintf(stderr, "r3_moe_worker: --compare lane rank %u enqueue failed rc=%d\n",
                         rank, erc);
            bad = true;
            break;
        }
        store(&h->lane[lane].request_sequence, sequence);
        if (gpu.sync_lane(lane)) gpu.gemv_debug_report(lane);
        if (!gpu.sync_lane(lane) || load(&h->lane[lane].done_sequence) != sequence || q->status != 0) {
            std::fprintf(stderr,
                         "r3_moe_worker: --compare lane rank %u completion failed status=%d done=%u\n",
                         rank, q->status, load(&h->lane[lane].done_sequence));
            bad = true;
            break;
        }
        /* The helper's per-token rows, taken before the host reference rewrites the same slot. */
        const uint16_t* res = ptr<uint16_t>(map, h->result_offset[result_index(rank, rank)]);
        const std::vector<uint16_t> got(res, res + (size_t)rows * hid);
        {   /* FNV-1a over the helper's result bytes: two builds that print the same hash for the same capture and settings produced
             * bit-identical output (the cross-build identity check of RCS05/RAD14; rel_l2 below only ever bounds the error). */
            uint64_t fnv = 1469598103934665603ull;
            for (uint16_t v : got) { fnv = (fnv ^ (v & 0xffu)) * 1099511628211ull; fnv = (fnv ^ (v >> 8)) * 1099511628211ull; }
            std::printf("replay_hash layer %u rank %u rows %u fnv64=%016llx\n", layer, rank, rows,
                        (unsigned long long)fnv);
        }
        if (g_sweep_fast) {
            static std::map<std::pair<const ReplayInput*, uint32_t>, std::vector<uint16_t>> base;
            double sum_abs = 0.0;
            size_t nonzero = 0;
            for (uint16_t v : got) { const double x = std::abs((double)bf16(v)); sum_abs += x; nonzero += v != 0; }
            std::printf("sweep_sum layer %u rank %u sum_abs=%.6g nonzero=%zu of %zu\n", layer, rank, sum_abs, nonzero, got.size());
            const auto key = std::make_pair(g_replay, rank);
            const auto it = base.find(key);
            if (it == base.end()) {
                base[key] = got;
                std::printf("sweep_cmp layer %u rank %u baseline mismatches=0\n", layer, rank);
            } else {
                size_t mism = 0;
                long double dn = 0.0L, dd = 0.0L;
                double dmax = 0.0;
                for (size_t i = 0; i < got.size(); ++i) {
                    mism += got[i] != it->second[i];
                    const double a = bf16(got[i]), b = bf16(it->second[i]);   /* this variant against the first one */
                    dn += (long double)(a - b) * (a - b);
                    dd += (long double)b * b;
                    dmax = std::max(dmax, std::abs(a - b));
                }
                std::printf("sweep_cmp layer %u rank %u mismatches=%zu of %zu rel_l2=%.6g max_abs=%.6g\n", layer, rank, mism, got.size(),
                            std::sqrt((double)(dn / std::max<long double>(dd, 1.0e-30L))), dmax);
            }
            continue;
        }

        std::vector<uint16_t> want_route;
        const int hrc = gpu.weight_format() == MOE_FMT_FP8_E4M3_BLOCK ? compute_host_engine_fp8(model, map, q, &want_route)
                                                                      : compute_host_engine(model, map, q, &want_route);
        if (hrc != 0 || want_route.size() != (size_t)nr * hid) {
            std::fprintf(stderr,
                         "r3_moe_worker: --compare local-path host rank %u failed rc=%d\n",
                         rank, hrc);
            bad = true;
            break;
        }
        /* Host reference per token: the same slot-order f32 sum over this request's routes. */
        std::vector<int32_t> route_of((size_t)rows * topk, -1);
        for (uint32_t j = 0; j < nr; ++j)
            route_of[(size_t)published[j].token * topk + published[j].slot] = (int32_t)j;
        long double num = 0.0L, den = 0.0L;
        double max_abs = 0.0;
        uint32_t tokens = 0;
        for (uint32_t t = 0; t < rows; ++t) {
            bool any = false;
            for (uint32_t slot = 0; slot < topk; ++slot) any = any || route_of[(size_t)t * topk + slot] >= 0;
            if (!any) continue;
            ++tokens;
            for (uint32_t col = 0; col < hid; ++col) {
                float acc = 0.0f;
                for (uint32_t slot = 0; slot < topk; ++slot) {
                    const int32_t j = route_of[(size_t)t * topk + slot];
                    if (j < 0) continue;
                    acc += bf16(published[(uint32_t)j].weight_bf16) *
                           bf16(want_route[(size_t)(uint32_t)j * hid + col]);
                }
                const double b = bf16(to_bf16(acc));
                const double a = bf16(got[(size_t)t * hid + col]);
                const double d = a - b;
                num += (long double)d * d;
                den += (long double)b * b;
                max_abs = std::max(max_abs, std::abs(d));
            }
        }
        const double rel =
            std::sqrt((double)(num / std::max<long double>(den, 1.0e-30L)));
        std::printf("engine_join layer %u rank %u lane %u routes=%u tokens=%u rel_l2=%.8g max_abs=%.8g\n",
                    layer, rank, lane, nr, tokens, rel, max_abs);
        bad = bad || tokens == 0 || !std::isfinite(rel) || rel >= 5.0e-3;
    }

    if (pinned) (void)hipHostUnregister(map);
    munmap(map, lay.bytes);
    return bad ? 1 : 0;
}

} // namespace
