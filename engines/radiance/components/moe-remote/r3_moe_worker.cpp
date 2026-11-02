#include "moe_backend.h"
#include "moe_remote_shm.h"
#include "rad_format.h"
#include "rad_encoding.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <memory>
#include <set>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "worker_container.h"
#include "worker_hostref.h"

#include "moe_remote_gpu.h"

#include "worker_shape.h"
#include "worker_share.h"
#include "worker_owners.h"
#include "worker_harness.h"
#include "worker_selftest.h"
#include "worker_combine.h"
#include "worker_config.h"
#include "worker_cpu.h"
#include "worker_segment.h"
#include "worker_lanes.h"
#include "worker_queue_warm.h"

int main(int argc, char** argv) {
    /* EVERY LANE STREAM ON ITS OWN HARDWARE QUEUE. A small lane pre-queues "wait for the next request" packets;
     * a stream sharing a hardware queue with one sits behind that wait until a small request arrives. With
     * every step large (KL mode's 64-token steps over an 8-row small lane) none ever does, and the large lane's
     * work never runs. HIP maps streams onto GPU_MAX_HW_QUEUES queues (4 by default) and the helper opens one
     * stream for each lane it serves (served_lanes: its rank slot's enabled classes and large lane) plus its own,
     * so the count is derived before HIP initialises, with two spare; an explicit setting wins. No more: every
     * extra queue is one more the gfx11 MES schedules, and past its hardware slots it oversubscribes and keeps
     * MES busy (~95 W) with nothing to run (RAD20). With power save off (R3_MOE_WORKER_POWER_SAVE=0), every protocol
     * lane gets one (lanes_open): the extra queues keep MES and the command processor awake, which dispatches ~0.3%
     * faster in decode at ~95 W per GPU at idle. The environment is only peeked here; it is validated below. */
    auto lanes_open = [](bool save, uint32_t count, uint32_t index, uint32_t classes) {
        std::array<bool, kLanes> all;
        all.fill(true);
        return save ? served_lanes(count, index, classes) : all;
    };
    {
        uint32_t lanes = 0;
        for (bool s : lanes_open(WorkerConfig::peek_flag("R3_MOE_WORKER_POWER_SAVE", true),
                                 WorkerConfig::peek_u32("R3_MOE_WORKER_COUNT", 1),
                                 WorkerConfig::peek_u32("R3_MOE_WORKER_INDEX", 0), helper_small_classes().count))
            lanes += s;
        (void)::setenv("GPU_MAX_HW_QUEUES", std::to_string(lanes + 3).c_str(), 0);
    }
    /* The helper's grouped MoE GEMM (prefill) on gfx11's native bf16 WMMA path: R9700 engine + two XTX helpers,
     * Flash-Next, prefill 2,518 -> 2,665 tok/s and helper gate/up -25%, KLD 0.064 (noise floor). Read by libr3
     * when the plugin is loaded, so set before it; an explicit setting wins. */
    (void)::setenv("R3_MOE_NATIVE", "1", 0);
    /* The same MoE GEMM with E4M3 widened to f16 for the f16 WMMA: bit-identical output (replay rel_l2 0 on four
     * captured requests, KLD unchanged), grouped GEMM 1.7-3.2x faster, prefill 2,130 -> 3,150 tok/s with decode
     * unchanged (RAD06 2026-10-18). Read by libr3 at launch; 0 turns it off. */
    (void)::setenv("R3_MOE_FP16", "1", 0);
    /* And the activation tile widened to half once at staging instead of by every wave that reads it: bit-identical,
     * a further 7-8% off the grouped GEMM on four real captured requests (RAD06 2026-10-18). 0 turns it off. */
    (void)::setenv("R3_MOE_AHALF", "1", 0);
    /* The activation markers component.toml names: one line a flag that is on, printed here and only when on, so a
     * capture's evidence check finds it in the arm that sets the flag and not in the control. FP16 and AHALF are read by
     * libr3's launcher at each launch; VECIO and FUSEROUTE by the lane at each request (default on unless "0"). */
    {
        auto is_on = [](const char* name, bool dflt) {
            const char* v = std::getenv(name);
            return v && *v ? std::strcmp(v, "0") != 0 : dflt;
        };
        const bool native = is_on("R3_MOE_NATIVE", false);
        if (native && is_on("R3_MOE_FP16", false))
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_FP16 on - f16 WMMA MoE GEMM\n");
        if (native && is_on("R3_MOE_FP16", false) && is_on("R3_MOE_AHALF", false))
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_AHALF on - activation tile widened to half at staging\n");
        if (is_on("R3_MOE_WORKER_VECIO", true))
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_VECIO on - 16-byte lane I/O\n");
        if (is_on("R3_MOE_WORKER_LANE_QUANT", true))
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_LANE_QUANT on - gated quantise bounded by the route count\n");
        if (is_on("R3_MOE_WORKER_SPARSE_ACT", true))
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_SPARSE_ACT on - sparse large-lane activation copy\n");
        if (is_on("R3_MOE_WORKER_FUSEROUTE", true))
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_FUSEROUTE on - fused small-lane front end\n");
    }
    /* --replay-sweep MODEL VARIANTS FILE...: one process, one model load and one expert allocation for every capture,
     * then each capture is replayed once per variant. VARIANTS is ';'-separated, each "NAME=value,NAME=value" (empty
     * = the environment as launched); the variant's variables are set (and the previous variant's unset) before the
     * replay, so anything libr3 reads at launch (R3_MOE_FP16, R3_MOE_PROBE, R3_MOE_PITCH16) can be compared with
     * no further model load. Each run prints "SWEEP run i capture c layer L variant [..]" and its engine_join line. */
    if (argc >= 2 && std::strcmp(argv[1], "--route-selftest") == 0) return route_selftest();
    /* --describe MODEL ARCH: print what the container's experts are and whether the backend for ARCH (gfx1100, gfx1030) can run
     * them, then exit (0 supported, 3 refused). No GPU, no allocation, no shared memory. */
    if (argc >= 4 && std::strcmp(argv[1], "--describe") == 0) {
        Container model;
        if (!model.open(argv[2])) {
            std::fprintf(stderr, "r3_moe_worker: --describe %s: not a container\n", argv[2]);
            return 2;
        }
        return backend_accepts(model, argv[3], true) ? 0 : 3;
    }
    static std::vector<ReplayInput> sweep_replays;
    static std::vector<std::string> sweep_variants;
    if (argc >= 5 && std::strcmp(argv[1], "--replay-sweep") == 0) {
        for (const char* q = argv[3];;) {
            const char* e = std::strchr(q, ';');
            sweep_variants.emplace_back(q, e ? (size_t)(e - q) : std::strlen(q));
            if (!e) break;
            q = e + 1;
        }
        sweep_replays.resize((size_t)(argc - 4));
        for (int i = 4; i < argc; ++i)
            if (!load_replay(argv[i], &sweep_replays[(size_t)(i - 4)])) {
                std::fprintf(stderr, "r3_moe_worker: --replay-sweep %s: not a capture file\n", argv[i]);
                return 2;
            }
        { const char* f = std::getenv("R3_MOE_SWEEP_FAST"); g_sweep_fast = f && *f && std::strcmp(f, "0") != 0; }
        static std::string sweep_layer, sweep_rows;
        g_replay = &sweep_replays[0];
        sweep_layer = std::to_string(sweep_replays[0].hd.layer);
        sweep_rows = std::to_string(sweep_replays[0].hd.rows);
        static const char* sweep_argv[8];
        const char* rebuilt[] = {argv[0], "--compare", argv[2], "--layer", sweep_layer.c_str(), "--rows",
                                 sweep_rows.c_str(), nullptr};
        std::copy(std::begin(rebuilt), std::end(rebuilt), sweep_argv);
        argv = const_cast<char**>(sweep_argv);
        argc = 7;
    }
    /* --replay MODEL FILE: --compare on a captured real request (layer and rows come from the file). */
    static ReplayInput replay;
    static std::string replay_layer, replay_rows;
    static const char* replay_argv[8];
    if (sweep_replays.empty() && argc == 4 && std::strcmp(argv[1], "--replay") == 0) {
        if (!load_replay(argv[3], &replay)) {
            std::fprintf(stderr, "r3_moe_worker: --replay %s: not a capture file\n", argv[3]);
            return 2;
        }
        g_replay = &replay;
        replay_layer = std::to_string(replay.hd.layer);
        replay_rows = std::to_string(replay.hd.rows);
        const char* rebuilt[] = {argv[0], "--compare", argv[2], "--layer", replay_layer.c_str(), "--rows",
                                 replay_rows.c_str(), nullptr};
        std::copy(std::begin(rebuilt), std::end(rebuilt), replay_argv);
        argv = const_cast<char**>(replay_argv);
        argc = 7;
        std::fprintf(stderr, "r3_moe_worker: replay layer %u rows %u routes %u\n", replay.hd.layer, replay.hd.rows,
                     replay.hd.route_count);
    }
    bool compare =argc > 1 && std::strcmp(argv[1], "--compare") == 0;
    bool synthetic_protect = compare && argc > 2 &&
        std::strcmp(argv[argc - 1], "--synthetic-protect") == 0;
    const int carg = argc - (synthetic_protect ? 1 : 0);
    const char* model_path = nullptr;
    uint32_t compare_layer = 0, compare_rows = 0;
    bool compare_one_layer = false;
    if (compare) {
        if (carg == 5 && std::strcmp(argv[3], "--rows") == 0) {
            if (!parse_u32(argv[4], 1, kMaxRows, &compare_rows)) {
                std::fprintf(stderr, "usage: r3_moe_worker --compare MODEL.rad --rows M\n");
                return 2;
            }
        } else if (carg == 7 && std::strcmp(argv[3], "--layer") == 0 &&
                   std::strcmp(argv[5], "--rows") == 0 &&
                   parse_u32(argv[4], 0, kMaxLayers - 1, &compare_layer) &&
                   parse_u32(argv[6], 1, kMaxRows, &compare_rows)) {
            compare_one_layer = true;
        } else {
            std::fprintf(stderr,
                "usage: r3_moe_worker --compare MODEL.rad [--layer L] --rows M [--synthetic-protect]\n");
            return 2;
        }
        model_path = argv[2];
    } else {
        if (argc != 2) {
            std::fprintf(stderr, "usage: r3_moe_worker MODEL.rad\n");
            return 2;
        }
        model_path = argv[1];
    }

    WorkerConfig cfg;
    if (std::string err; !cfg.from_environment(kMaxHelpers, &err)) {
        std::fprintf(stderr, "r3_moe_worker: %s\n", err.c_str());
        return 2;
    }

    Container model;
    if (!model.open(model_path) || model.layers > kMaxLayers) {
        std::fprintf(stderr, "r3_moe_worker: cannot open/validate %s\n", model_path);
        return 1;
    }

    const bool cpu = cfg.cpu && !compare;
    std::unique_ptr<r3_moe_gpu::Runner> gpu;
    if (!cpu) {
        gpu = std::make_unique<r3_moe_gpu::Runner>(model);
        if (cfg.engine_ranks) gpu->engine_ranks_ = cfg.engine_ranks;
        if (!gpu->open()) return 1;
        {   // the container's own weight format picks the kernels (RAD13); an unsupported one was refused by name above
            MoeExpertDesc first{};
            for (uint32_t l = 0; l < model.layers && !first.weight.format; ++l)
                for (uint32_t e = 0; e < model.experts && !first.weight.format; ++e)
                    if (model.weight(l, e, 0) && model.weight(l, e, 1)) (void)describe_expert(model, l, e, &first);
            if (!first.weight.format) {
                std::fprintf(stderr, "r3_moe_worker: no expert of this container has a weight format the worker knows\n");
                return 1;
            }
            for (uint32_t l = 0; l < model.layers; ++l)   // one table of experts takes one encoding
                for (uint32_t e = 0; e < model.experts; ++e) {
                    MoeExpertDesc d{};
                    if (model.weight(l, e, 0) && model.weight(l, e, 1) && describe_expert(model, l, e, &d) &&
                        (d.weight.format != first.weight.format || d.weight.block_k != first.weight.block_k ||
                         d.weight.scale_dtype != first.weight.scale_dtype)) {
                        std::fprintf(stderr, "r3_moe_worker: experts (%u,%u) and the first differ in weight format\n", l, e);
                        return 1;
                    }
                }
            if (!gpu->set_weight_format(first.weight.format)) return 1;
        }
    }

    /* R3_MOE_WORKER_COUNT=2 with R3_MOE_WORKER_INDEX=0/1: two helpers (the XTX cards) behind a one-rank engine
     * (the R9700) share one segment; helper 0 creates it, helper 1 attaches, each owns half the candidates and
     * serves its own rank slot. */
    const uint32_t helper_index = cfg.helper_index, helper_count = cfg.helper_count;
    if (helper_count > 1 && (!gpu || gpu->engine_ranks_ != 1 || compare)) {
        std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_INDEX %u of R3_MOE_WORKER_COUNT %u: want index < count <= %u, "
                     "and two helpers only on GPU behind a one-rank engine (R3_MOE_WORKER_ENGINE_RANKS=1), not --compare\n",
                     helper_index, helper_count, kMaxHelpers);
        return 2;
    }
    uint64_t budgets[r3::worker_share::kMaxHelperShares] = {};
    if (!r3::worker_share::parse_budgets(std::getenv("R3_MOE_WORKER_MIBS"), cfg.helper_count, helper_budget_bytes(), budgets)) {
        std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_MIBS must list one budget in MiB for each of the %u helpers\n", cfg.helper_count);
        return 2;
    }
    Owned owned;
    if (g_replay) {
        /* A replay owns exactly the experts the captured request routes to: the capturing helper's ownership
         * (one of two, by heat) is not this process's, and an unowned route would never complete. */
        std::set<std::pair<uint32_t, uint32_t>> seen;
        auto own_routes = [&](const ReplayInput& ri) {
            for (const Route& r : ri.routes)
                if (r.expert >= 0 && r.expert < (int32_t)model.experts &&
                    seen.emplace(ri.hd.layer, (uint32_t)r.expert).second)
                    owned.expert.emplace_back(ri.hd.layer, (uint32_t)r.expert);
        };
        if (sweep_replays.empty()) own_routes(*g_replay);
        else for (const ReplayInput& ri : sweep_replays) own_routes(ri);
    } else if (!select_owned(model, gpu.get(), budgets, helper_index, helper_count, &owned)) {
        return 1;
    }
    if (gpu && !gpu->allocate(owned.expert)) return 1;
    const SmallClasses lane_classes = helper_small_classes();
    if (!lane_classes.count) return 1;
    const uint32_t lane_large_rows = helper_max_rows();
    std::string lane_list;                       /* "4,8,16,24", for the log */
    for (uint32_t c = 0; c < lane_classes.count; ++c)
        lane_list += (c ? "," : "") + std::to_string(lane_classes.rows[c]);
    if (const char* sl = std::getenv("R3_MOE_WORKER_SMALL_LANES"); sl && *sl)
        std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_SMALL_LANES on - small lanes %s\n", lane_list.c_str());
    long rss_pages = 0;
    if (FILE* sf = std::fopen("/proc/self/statm", "r")) {
        long total_pages = 0;
        if (std::fscanf(sf, "%ld %ld", &total_pages, &rss_pages) != 2) rss_pages = 0;
        std::fclose(sf);
    }
    const double rss_mib = rss_pages > 0 ? (double)rss_pages * (double)sysconf(_SC_PAGESIZE) / (1024.0 * 1024.0) : -1.0;
    std::fprintf(stderr, "r3_moe_worker: selected %zu expert candidates, %.3f GiB; host RSS %.1f MiB\n",
                 owned.expert.size(), (double)owned.bytes / (1024.0 * 1024.0 * 1024.0), rss_mib);

    if (compare && !sweep_replays.empty()) {
        std::set<std::string> touched;
        int rc = 0, run = 0;
        for (size_t c = 0; c < sweep_replays.size(); ++c) {
            g_replay = &sweep_replays[c];
            for (const std::string& v : sweep_variants) {
                for (const std::string& k : touched) (void)::unsetenv(k.c_str());
                touched.clear();
                for (size_t pos = 0; pos < v.size();) {
                    const size_t comma = v.find(',', pos);
                    const std::string kv = v.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                    const size_t eq = kv.find('=');
                    if (eq != std::string::npos && eq > 0) {
                        (void)::setenv(kv.substr(0, eq).c_str(), kv.c_str() + eq + 1, 1);
                        touched.insert(kv.substr(0, eq));
                    }
                    if (comma == std::string::npos) break;
                    pos = comma + 1;
                }
                std::printf("SWEEP run %d capture %zu layer %u variant [%s]\n", run++, c, g_replay->hd.layer, v.c_str());
                std::fflush(stdout);
                rc |= compare_mode(model, *gpu, owned, g_replay->hd.layer, g_replay->hd.rows, lane_classes,
                                   lane_large_rows, false);
                std::fflush(stdout);
            }
        }
        return rc;
    }
    if (compare) {
        if (compare_one_layer)
            return compare_mode(model, *gpu, owned, compare_layer, compare_rows, lane_classes,
                                lane_large_rows, synthetic_protect);
        int rc = 0;
        for (uint32_t layer : {0u, 20u, 34u, 47u}) {
            if (layer >= model.layers) {
                std::fprintf(stderr, "r3_moe_worker: --compare layer %u unavailable (layers=%u)\n",
                             layer, model.layers);
                rc = 1;
                continue;
            }
            std::printf("layer %u\n", layer);
            rc |= compare_mode(model, *gpu, owned, layer, compare_rows, lane_classes,
                               lane_large_rows, synthetic_protect);
        }
        return rc;
    }

    r3::moe_shm::LayoutShape shape{};
    Layout lay{};
    if (!segment_shape(model, helper_count, &shape, &lay)) return 1;
    /* The mailbox segment: helper 0 creates it, the others attach (up to 30 minutes while the first loads). Owned by
     * `seg` from here on; every return below releases it, and only helper 0 unlinks the name. */
    /* Before the first wait (segment attach, the engine's maps): a stop signal must end every one of them. No
     * SA_RESTART (std::signal sets it): futex waits return EINTR and their loops see the stop word. */
    struct sigaction sa{};
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr); sigaction(SIGTERM, &sa, nullptr);
    Segment seg;
    if (!seg.open(shm_name(), lay.bytes, helper_index == 0, 1800)) return 1;
    void* const map = seg.host();
    Header* const h = seg.header();
    const char* const name = seg.name();
    if (helper_index == 0) {
        if (!init(map, model.layers, shape, helper_count)) return 1;
        h->small_class_count = lane_classes.count;
        for (uint32_t c = 0; c < lane_classes.count; ++c) h->small_class_rows[c] = lane_classes.rows[c];
        /* Every helper reads the same step activations: stage them once, in slot 0. */
        for (uint32_t r = 1; r < helper_count; ++r) h->activation_offset[r] = h->activation_offset[0];
        store(&h->helper_count, helper_count);   /* last: an attaching helper waits for it */
        (void)futex_wake(&h->helper_count);
    } else {
        /* Woken by helper 0's publication; the one-second bound keeps a signal-driven stop prompt. */
        for (int i = 0; i < 1800 && !stopping(); ++i) {
            const uint32_t seen = load(&h->helper_count);
            if (seen == helper_count) break;
            struct timespec ts{1, 0};
            (void)::syscall(SYS_futex, &h->helper_count, FUTEX_WAIT, seen, &ts, nullptr, 0);
        }
        bool same_lanes = load(&h->helper_count) == helper_count && valid(h, lay.bytes, shape) && h->small_class_count == lane_classes.count;
        for (uint32_t c = 0; same_lanes && c < lane_classes.count; ++c)
            same_lanes = h->small_class_rows[c] == lane_classes.rows[c];
        if (!same_lanes) {
            std::fprintf(stderr, "r3_moe_worker: segment %s is not helper 0's %u-helper segment (small lanes %u, ours %s)\n",
                         name, helper_count, (unsigned)h->small_class_count, lane_list.c_str());
            return 1;
        }
    }
    /* The engine watches this process from here on (Header::helper_pid): if it dies before it can set helper_fault,
     * the engine sets it. */
    if (!publish_owned(map, owned, helper_index)) {
        raise_fault(h);
        return 1;
    }
    /* The pid publication is the engine registration barrier: publish ownership first. */
    h->helper_start[helper_index] = process_start_ticks((int)::getpid());
    __atomic_store_n(&h->helper_pid[helper_index], (int32_t)::getpid(), __ATOMIC_RELEASE);
    notify(h);
    if (!cpu && !seg.register_gpu()) return 1;
    void* const dev_map = seg.device();

    std::fprintf(stderr, "r3_moe_worker: waiting for engine runtime expert maps\n");
    /* The first map waits as long as the engine takes to start. After that the engine declares its
     * layers back to back, so a layer whose map has not come within 10 s is one this engine never
     * declares (the MTP layer in a run without speculation, e.g. KL mode): drop its ownership. */
    bool any_map = false;
    for (uint32_t layer = 0; layer < model.layers; ++layer) {
        /* One-second sleeps: a stop signal may land on another thread (HIP's), so EINTR alone cannot be relied on. */
        for (uint32_t waited = 0; !load(&h->layer_map_ready[layer]);) {
            if (stopping()) return 1;
            if (any_map && waited++ == 10) break;
            struct timespec ts{1, 0};
            (void)::syscall(SYS_futex, &h->layer_map_ready[layer], FUTEX_WAIT, 0, &ts, nullptr, 0);
        }
        if (!load(&h->layer_map_ready[layer])) {
            std::fprintf(stderr, "r3_moe_worker: layer %u has no runtime map (not declared by the engine); not owned\n",
                         layer);
            for (auto [l, e] : owned.expert)
                if (l == layer) (void)drop_bootstrap_owner(map, l, e, helper_index);
            continue;
        }
        any_map = true;
        const int32_t* slots = layer_slots(map, layer);
        if (!slots) return 1;
        for (auto [l, e] : owned.expert)
            if (l == layer && slots[e] < 0) (void)drop_bootstrap_owner(map, l, e, helper_index);
        if (gpu && !gpu->load_layer(layer, owned.expert, slots)) {
            std::fprintf(stderr, "r3_moe_worker: layer %u runtime table build failed\n", layer);
            raise_fault(h);   /* not consumed: the engine's map wait fails instead of using a map this helper lacks */
            return 1;
        }
        store(&h->layer_map_consumed[layer], 1);
        notify(h);
    }
    Owned runtime_owned;
    for (auto [l, e] : owned.expert) {
        if (!load(&h->layer_map_ready[l]) || layer_slot(map, l, e) < 0) continue;
        runtime_owned.expert.emplace_back(l, e);
        const int64_t eb = laid_out_bytes(model, gpu.get(), l, e);
        if (eb > 0) runtime_owned.bytes += (uint64_t)eb;
    }
    owned = std::move(runtime_owned);
    const auto lanes_served = served_lanes(helper_count, helper_index, lane_classes.count);
    std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_POWER_SAVE %s\n", cfg.power_save ? "on" : "off");
    if (gpu && !gpu->prepare_lanes(lane_classes.rows, lane_classes.count, lane_large_rows,
                                   lanes_open(cfg.power_save, helper_count, helper_index, lane_classes.count)))
        return 1;
    if (gpu) (void)posix_fadvise(model.fd, 0, 0, POSIX_FADV_DONTNEED);
    std::fprintf(stderr, "r3_moe_worker: runtime map owns %zu experts, %.3f GiB\n",
                 owned.expert.size(), (double)owned.bytes / (1024.0 * 1024.0 * 1024.0));

    std::fprintf(stderr,
                 "r3_moe_worker: ready shm=%s layers=%u bytes=%llu (%s, small lane rows <= %s, large <= %u)\n",
                 name, model.layers, (unsigned long long)lay.bytes,
                 cpu ? "host fp32 reference" : gpu->arch().c_str(), lane_list.c_str(), lane_large_rows);

    uint64_t stat_req = 0, stat_rows = 0, stat_routes = 0;
    double stat_us = 0.0;
    if (cpu) {
        uint32_t seq = 1;
        while (!stopping()) {
            Request* q = request(map, seq);
            while (!stopping() && load(&q->sequence) != seq) {
                const uint32_t seen = load(&h->publish_sequence);
                if (load(&q->sequence) != seq) (void)futex_wait(&h->publish_sequence, seen);
            }
            if (stopping()) break;
            const auto t0 = std::chrono::steady_clock::now();
            q->status = compute_host(model, map, q);
            stat_us += std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count();
            ++stat_req; stat_rows += q->rows; stat_routes += q->route_count;
            store(&q->completion_sequence, seq);
            (void)futex_wake(&q->completion_sequence);
            if (++seq == 0) break;
        }
    } else {
        /* R3_MOE_WORKER_COMBINE=1 (default off), two to kCombineHelpers helpers: helper 0 adds every helper's rows for the
         * engine's large lane on its large-lane thread after its stream sync (worker_combine.h); each other helper wakes it
         * after its own sync.
         * The engine asks only once combine.enabled is set. */
        Combine combine;
        if (cfg.combine && helper_count >= 2 && helper_count <= kCombineHelpers) {
            if (!combine_cpu_supported()) {
                std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_COMBINE needs a CPU with AVX2 and F16C\n");
                return 1;
            }
            combine.on = true;
            gpu->set_chunk_helper(helper_index);   /* every helper publishes its large-lane rows chunk by chunk (RAD42) */
            if (helper_index == 0) {
                const uint32_t threads = std::min(cfg.combine_threads, std::max(1u, std::thread::hardware_concurrency()));
                combine.pool = std::make_unique<CombinePool>(threads);
                combine.mask.resize(h->max_rows);
                h->combine.min_rows = cfg.combine_min_rows;
                store(&h->combine.enabled, 1u);
                std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_COMBINE on - %u CPU thread(s), large-lane steps of %u+ rows\n",
                             threads, cfg.combine_min_rows);
            }
        }
        /* Power save: oversubscribe the GPU only inside bursts, for fast dispatch (worker_queue_warm.h). Started after
         * the lane streams exist, so it counts their queues. Power save off is oversubscribed already. Declared before
         * the lanes, so it is destroyed after they have published the stop. */
        QueueWarm warm;
        if (!warm.start(h, cfg.power_save && cfg.queue_warm)) return 1;
        Lanes lanes(h, map, dev_map, *gpu, lane_classes, cfg, lanes_served, combine);
        if (!lanes.start()) return 1;
        for (uint32_t v; (v = g_stop_word.load(std::memory_order_acquire)) == 0;)
            (void)::syscall(SYS_futex, reinterpret_cast<uint32_t*>(&g_stop_word), FUTEX_WAIT_PRIVATE, v, nullptr, nullptr, 0);
        lanes.stop_and_join();
        warm.stop();
        if (combine.pool) {
            store(&h->combine.enabled, 0u);
            combine.pool.reset();
            std::fprintf(stderr, "r3_moe_worker: combine steps %llu, CPU %.1f ms (%.1f us a step)\n",
                         (unsigned long long)combine.stats.steps, combine.stats.combine_us / 1000.0,
                         combine.stats.steps ? combine.stats.combine_us / (double)combine.stats.steps : 0.0);
        }
        lanes.report(&stat_req, &stat_rows, &stat_routes, &stat_us);
    }

    std::fprintf(stderr, "r3_moe_worker: served %llu requests, %llu rows, %llu routes, compute %.1f ms (%.1f us a request)\n",
                 (unsigned long long)stat_req, (unsigned long long)stat_rows, (unsigned long long)stat_routes,
                 stat_us / 1000.0, stat_req ? stat_us / (double)stat_req : 0.0);
    return 0;   /* seg releases the segment */
}
