#pragma once

// --route-selftest. Included only by r3_moe_worker.cpp, at the position the code had in the single file (RCS03, pure move).


/* --route-selftest: the one-block routing kernels at several expert counts against a CPU model (RCS09-05). No container, no shared
 * memory, no model: synthetic routes and a slot table, run on the first visible device. The unfused count/prefix/scatter path is
 * compared as sets per expert slot (its scatter order is atomic); the fused small-lane front end must match the deterministic
 * order exactly. Exit status is the number of failing cases. */
int route_selftest() {
    using namespace r3::moe_shm;
    using namespace r3_moe_gpu;
    uint32_t rng = 12345u;
    auto next = [&rng] { rng = rng * 1664525u + 1013904223u; return rng >> 8; };
    auto hip = [](hipError_t e, const char* what) {
        if (e != hipSuccess) std::fprintf(stderr, "route-selftest: %s: %s\n", what, hipGetErrorString(e));
        return e == hipSuccess;
    };
    int failures = 0;
    const uint32_t kMaxRowsFused = 8;
    for (uint32_t experts : {128u, 256u, 384u, 512u, 640u, 1000u}) {
        uint32_t blockn = 256;
        while (blockn < experts) blockn <<= 1;
        for (uint32_t rows : {1u, 8u}) {
            for (int permuted = 0; permuted < 2; ++permuted) {
                const uint32_t nr = rows * kTopK;
                std::vector<Route> routes(nr);
                std::vector<uint16_t> slot(experts);
                for (uint32_t e = 0; e < experts; ++e) slot[e] = permuted ? (uint16_t)((e * 7u + 3u) % experts) : (uint16_t)e;
                for (uint32_t t = 0; t < rows; ++t)
                    for (uint32_t k = 0; k < kTopK; ++k)
                        routes[t * kTopK + k] = Route{(int32_t)t, (int32_t)(next() % experts), (uint16_t)(0x3f80u + k), (uint16_t)k};
                const LaneShape lane_shape{kHidden, kTopK, kActGroup, experts};   // the kernels read the model shape from a device constant
                if (!hip(hipMemcpyToSymbol(HIP_SYMBOL(g_shape), &lane_shape, sizeof lane_shape), "set lane shape")) { ++failures; continue; }
                std::vector<uint32_t> counts(experts, 0), excl(experts + 1, 0);
                std::vector<std::vector<int32_t>> want(experts);
                for (uint32_t i = 0; i < nr; ++i) {
                    const uint32_t es = slot[(uint32_t)routes[i].expert];
                    ++counts[es];
                    want[es].push_back((int32_t)((uint32_t)routes[i].token * kTopK + routes[i].slot));
                }
                for (uint32_t e = 0; e < experts; ++e) excl[e + 1] = excl[e] + counts[e];

                Request q{};
                q.sequence = 1; q.rank = 0; q.rank_count = 1; q.layer = 0; q.rows = rows; q.hidden = kHidden; q.route_count = nr;
                q.activation_stride = kHidden * (uint32_t)sizeof(uint16_t);
                q.activation_q_stride = kHidden;
                q.activation_s_stride = (kHidden / kActGroup) * (uint32_t)sizeof(float);
                DeviceLayerTables tl{};
                const size_t half = experts / 2;
                std::vector<void*> tables(8, nullptr);
                void *d_route = nullptr, *d_q = nullptr, *d_slot = nullptr, *d_tl = nullptr, *d_counts = nullptr, *d_off = nullptr;
                void *d_cursor = nullptr, *d_flat = nullptr, *d_ew = nullptr, *d_sorted = nullptr, *d_inv = nullptr, *d_stats = nullptr;
                void *d_srcq = nullptr, *d_srcs = nullptr, *d_dstq = nullptr, *d_dsts = nullptr;
                const size_t q_per_row = kHidden / 16, s_per_row = (kHidden / kActGroup) * 4 / 16;
                bool good = hip(hipMalloc(&d_route, nr * sizeof(Route)), "alloc routes") &&
                            hip(hipMalloc(&d_q, sizeof(Request)), "alloc request") &&
                            hip(hipMalloc(&d_slot, experts * sizeof(uint16_t)), "alloc slots") &&
                            hip(hipMalloc(&d_tl, sizeof(DeviceLayerTables)), "alloc tables") &&
                            hip(hipMalloc(&d_counts, experts * 4), "alloc counts") &&
                            hip(hipMalloc(&d_off, (experts + 1) * 4), "alloc offsets") &&
                            hip(hipMalloc(&d_cursor, experts * 4), "alloc cursor") &&
                            hip(hipMalloc(&d_flat, nr * 4), "alloc flat") && hip(hipMalloc(&d_ew, nr * 2), "alloc weights") &&
                            hip(hipMalloc(&d_sorted, nr * 4), "alloc sorted") && hip(hipMalloc(&d_inv, nr * 4), "alloc inverse") &&
                            hip(hipMalloc(&d_stats, 3 * 8), "alloc stats") &&
                            hip(hipMalloc(&d_srcq, kMaxRowsFused * q_per_row * 16), "alloc src q") &&
                            hip(hipMalloc(&d_srcs, kMaxRowsFused * s_per_row * 16), "alloc src s") &&
                            hip(hipMalloc(&d_dstq, kMaxRowsFused * q_per_row * 16), "alloc dst q") &&
                            hip(hipMalloc(&d_dsts, kMaxRowsFused * s_per_row * 16), "alloc dst s");
                for (auto& t : tables) good = good && hip(hipMalloc(&t, (half + 1) * sizeof(void*)), "alloc table");
                if (good) {
                    for (auto& t : tables) good = good && hip(hipMemset(t, 0, (half + 1) * sizeof(void*)), "zero table");
                    for (int c = 0; c < 2; ++c) {
                        tl.gu_code[0][c] = tables[c * 4 + 0]; tl.gu_scale[0][c] = tables[c * 4 + 1];
                        tl.dn_code[0][c] = tables[c * 4 + 2]; tl.dn_scale[0][c] = tables[c * 4 + 3];
                    }
                    tl.expert_slot = static_cast<const uint16_t*>(d_slot);
                    good = good && hip(hipMemcpy(d_route, routes.data(), nr * sizeof(Route), hipMemcpyHostToDevice), "upload routes") &&
                           hip(hipMemcpy(d_q, &q, sizeof q, hipMemcpyHostToDevice), "upload request") &&
                           hip(hipMemcpy(d_slot, slot.data(), experts * 2, hipMemcpyHostToDevice), "upload slots") &&
                           hip(hipMemcpy(d_tl, &tl, sizeof tl, hipMemcpyHostToDevice), "upload tables") &&
                           hip(hipMemset(d_counts, 0, experts * 4), "zero counts") && hip(hipMemset(d_stats, 0, 24), "zero stats") &&
                           hip(hipMemset(d_srcq, 0, kMaxRowsFused * q_per_row * 16), "zero src q") &&
                           hip(hipMemset(d_srcs, 0, kMaxRowsFused * s_per_row * 16), "zero src s");
                }
                const auto* rq = static_cast<const Request*>(d_q);
                const auto* rt = static_cast<const Route*>(d_route);
                const auto* lt = static_cast<const DeviceLayerTables*>(d_tl);
                const uint32_t grid = (nr + 255) / 256;
                auto run_prefix = [&](auto n_const) {
                    constexpr uint32_t N = decltype(n_const)::value;
                    hipLaunchKernelGGL((lane_route_prefix<N>), dim3(1), dim3(N), 0, 0, static_cast<const uint32_t*>(d_counts),
                                       static_cast<int32_t*>(d_off), static_cast<uint32_t*>(d_cursor), experts);
                };
                auto run_front = [&](auto n_const) {
                    constexpr uint32_t N = decltype(n_const)::value;
                    hipLaunchKernelGGL((lane_front_small<N>), dim3(1), dim3(N), 0, 0, rt, const_cast<Request*>(rq), lt,
                                       static_cast<const uint4*>(d_srcq), static_cast<const uint4*>(d_srcs),
                                       static_cast<uint4*>(d_dstq), static_cast<uint4*>(d_dsts), static_cast<uint64_t*>(d_stats),
                                       0u, 0u, 1u, kMaxRowsFused, 1u, nr, experts, static_cast<int32_t*>(d_off),
                                       static_cast<int32_t*>(d_sorted), static_cast<int32_t*>(d_inv), static_cast<uint16_t*>(d_ew),
                                       static_cast<void**>(tables[0]), static_cast<void**>(tables[1]), static_cast<void**>(tables[2]),
                                       static_cast<void**>(tables[3]), static_cast<void**>(tables[4]), static_cast<void**>(tables[5]),
                                       static_cast<void**>(tables[6]), static_cast<void**>(tables[7]));
                };
                std::vector<int32_t> off(experts + 1), sorted(nr), inv(nr);
                std::vector<uint16_t> ew(nr);
                std::vector<uint32_t> got_counts(experts);
                std::string why;
                if (!good) why = "device setup";
                if (good) {   // the unfused path
                    hipLaunchKernelGGL(lane_route_count, dim3(grid), dim3(256), 0, 0, rt, const_cast<Request*>(rq), lt, 0u, 1u,
                                       rows, 1u, experts, static_cast<uint32_t*>(d_counts), static_cast<int32_t*>(d_flat),
                                       static_cast<uint16_t*>(d_ew));
                    switch (blockn) {
                        case 256: run_prefix(std::integral_constant<uint32_t, 256>{}); break;
                        case 512: run_prefix(std::integral_constant<uint32_t, 512>{}); break;
                        case 1024: run_prefix(std::integral_constant<uint32_t, 1024>{}); break;
                        default: why = "no block size";
                    }
                    hipLaunchKernelGGL(lane_route_scatter, dim3(grid), dim3(256), 0, 0, rt, rq, lt, 0u, 1u, rows, 1u, experts,
                                       static_cast<const int32_t*>(d_flat), static_cast<uint32_t*>(d_cursor),
                                       static_cast<int32_t*>(d_sorted), static_cast<int32_t*>(nullptr));
                    good = hip(hipDeviceSynchronize(), "unfused kernels") &&
                           hip(hipMemcpy(got_counts.data(), d_counts, experts * 4, hipMemcpyDeviceToHost), "read counts") &&
                           hip(hipMemcpy(off.data(), d_off, (experts + 1) * 4, hipMemcpyDeviceToHost), "read offsets") &&
                           hip(hipMemcpy(sorted.data(), d_sorted, nr * 4, hipMemcpyDeviceToHost), "read sorted") &&
                           hip(hipMemcpy(ew.data(), d_ew, nr * 2, hipMemcpyDeviceToHost), "read weights");
                    if (!good && why.empty()) why = "unfused run";
                    if (why.empty()) {
                        if (got_counts != counts) why = "counts";
                        for (uint32_t e = 0; why.empty() && e <= experts; ++e)
                            if ((uint32_t)off[e] != excl[e]) why = "offsets";
                        for (uint32_t e = 0; why.empty() && e < experts; ++e) {
                            std::vector<int32_t> g(sorted.begin() + off[e], sorted.begin() + off[e + 1]), w = want[e];
                            std::sort(g.begin(), g.end());
                            std::sort(w.begin(), w.end());
                            if (g != w) why = "sorted routes";
                        }
                        for (uint32_t i = 0; why.empty() && i < nr; ++i)
                            if (ew[(uint32_t)routes[i].token * kTopK + routes[i].slot] != routes[i].weight_bf16) why = "weights";
                    }
                }
                if (good && why.empty() && rows <= kMaxRowsFused) {   // the fused front end: deterministic order
                    std::fill(off.begin(), off.end(), -1);
                    switch (blockn) {
                        case 256: run_front(std::integral_constant<uint32_t, 256>{}); break;
                        case 512: run_front(std::integral_constant<uint32_t, 512>{}); break;
                        case 1024: run_front(std::integral_constant<uint32_t, 1024>{}); break;
                        default: why = "no block size";
                    }
                    int32_t status = 0;
                    good = hip(hipDeviceSynchronize(), "fused kernel") &&
                           hip(hipMemcpy(off.data(), d_off, (experts + 1) * 4, hipMemcpyDeviceToHost), "read fused offsets") &&
                           hip(hipMemcpy(sorted.data(), d_sorted, nr * 4, hipMemcpyDeviceToHost), "read fused sorted") &&
                           hip(hipMemcpy(inv.data(), d_inv, nr * 4, hipMemcpyDeviceToHost), "read fused inverse") &&
                           hip(hipMemcpy(&status, static_cast<uint8_t*>(d_q) + offsetof(Request, status), 4, hipMemcpyDeviceToHost),
                               "read status");
                    if (!good && why.empty()) why = "fused run";
                    if (why.empty() && status != 0) why = "fused status";
                    for (uint32_t e = 0; why.empty() && e <= experts; ++e)
                        if ((uint32_t)off[e] != excl[e]) why = "fused offsets";
                    for (uint32_t e = 0; why.empty() && e < experts; ++e)
                        for (uint32_t j = 0; why.empty() && j < counts[e]; ++j) {
                            if (sorted[excl[e] + j] != want[e][j]) why = "fused order";
                            else if (inv[(uint32_t)want[e][j]] != (int32_t)(excl[e] + j)) why = "fused inverse";
                        }
                }
                for (void* p : {d_route, d_q, d_slot, d_tl, d_counts, d_off, d_cursor, d_flat, d_ew, d_sorted, d_inv, d_stats, d_srcq, d_srcs,
                                d_dstq, d_dsts})
                    if (p) (void)hipFree(p);
                for (void* p : tables)
                    if (p) (void)hipFree(p);
                const bool pass = good && why.empty();
                std::printf("route-selftest experts %u block %u rows %u %s: %s%s%s\n", experts, blockn, rows,
                            permuted ? "permuted" : "identity", pass ? "PASS" : "FAIL", pass ? "" : " ", pass ? "" : why.c_str());
                if (!pass) ++failures;
            }
        }
    }
    std::printf("route-selftest: %d failing case(s)\n", failures);
    return failures;
}
