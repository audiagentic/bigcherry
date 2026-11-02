// Generates owned_golden.txt from the PRE-MOVE selection (worker_owners.h at commit 6a3d11433): every case in
// golden_cases.h, every helper, no GPU. Built and run as golden/README.md says. Writes the file named by argv[1].
#include "moe_backend.h"
#include "moe_remote_shm.h"
#include "rad_format.h"
#include "rad_encoding.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <thread>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "worker_container.h"
#include "worker_hostref.h"
#include "moe_remote_gpu.h"
#include "worker_shape.h"
#include "worker_share.h"
#include "worker_owners.h"

#include "golden_cases.h"
#include "test_rad_fixture.h"

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: gen_golden_main <out.txt> <scratch dir>\n"); return 2; }
    const std::string scratch = argv[2];
    std::filesystem::create_directories(scratch);
    const std::string model = scratch + "/owners.rad", heat = scratch + "/owners.hot-first";
    bc::test::write_rad(model, 4, 16, 128, 128);
    std::vector<uint32_t> h(128 * 16, 0);
    for (uint32_t l = 0; l < 4; ++l)
        for (uint32_t e = 0; e < 16; ++e) h[l * 16 + e] = (e * 7 + l * 3) % 16 + 1;
    std::ofstream(heat, std::ios::binary).write(reinterpret_cast<const char*>(h.data()), (std::streamsize)(h.size() * 4));
    Container f;
    if (!f.open(model.c_str())) { std::fprintf(stderr, "cannot open %s\n", model.c_str()); return 1; }
    std::string out;
    for (const golden::Case& c : golden::cases()) {
        for (const char* k : {"HEAT", "PLACEMENT", "FORMATS", "SHARES", "BALANCE", "BALANCE_SPEED"})
            unsetenv((std::string("R3_MOE_WORKER_") + k).c_str());
        setenv("R3_MOE_WORKER_FORMATS", "w4nl64", 1);   // the card's formats are required; a case may override
        for (const auto& [k, v] : c.env)
            setenv((std::string("R3_MOE_WORKER_") + k).c_str(), v == "@heat" ? heat.c_str() : v.c_str(), 1);
        std::vector<uint64_t> budgets;
        for (uint64_t n : c.budget_experts) budgets.push_back(n * golden::kExpertBytes);
        for (uint32_t i = 0; i < c.helpers; ++i) {
            Owned o;
            const bool ok = select_owned(f, nullptr, budgets.data(), i, c.helpers, &o);
            out += std::string(c.name) + " " + std::to_string(i) + " " + (ok ? "ok" : "refused") + " " + std::to_string(ok ? o.bytes : 0);
            if (ok) for (auto [l, e] : o.expert) out += " " + std::to_string(l) + ":" + std::to_string(e);
            out += "\n";
        }
    }
    std::ofstream(argv[1]) << out;
    return 0;
}
