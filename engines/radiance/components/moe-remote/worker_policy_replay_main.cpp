// r3_policy_replay: offline replay of the live placement policy over a recorded series of heat snapshots.
//   r3_policy_replay --series DIR --start-heat FILE --file-layers 128 --file-experts 512 --layers 49 --experts 512
//       --budgets-mib 22528,22528,14336 --stride-bytes N --shares 1,1,1 --half-life-observations X --margin M
//       --max-moves N --max-moved-bytes B --min-samples S --period-s P --min-residency-s R --cadence N
// DIR holds the snapshots, one file each (the engine's route-count file, uint32 [file-layers][file-experts],
// cumulative), replayed in name order. FILE is the launch's heat ranking (uint32 [file-layers][file-experts], value =
// max - count, so smaller is hotter); the starting owners are the static placer's plan over it. Every value is an
// explicit argument: nothing here has a default. See worker_policy_replay.h for what the replay does not model.
#include "worker_policy_replay.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

using namespace r3::policy;

static bool read_u32(const std::string& path, std::vector<uint32_t>* out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize bytes = f.tellg();
    if (bytes <= 0 || bytes % 4) return false;
    out->resize(size_t(bytes) / 4);
    f.seekg(0);
    return bool(f.read(reinterpret_cast<char*>(out->data()), bytes));
}

static std::vector<std::string> list_dir(const std::string& dir) {
    std::vector<std::string> names;
    if (DIR* d = opendir(dir.c_str())) {
        while (const dirent* e = readdir(d)) if (e->d_name[0] != '.') names.push_back(e->d_name);
        closedir(d);
    }
    std::sort(names.begin(), names.end());
    return names;
}

static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out; size_t a = 0;
    while (a <= s.size()) { const size_t b = s.find(',', a); out.push_back(s.substr(a, b == std::string::npos ? b : b - a)); if (b == std::string::npos) break; a = b + 1; }
    return out;
}

int main(int argc, char** argv) {
    std::map<std::string, std::string> arg;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strncmp(argv[i], "--", 2)) { std::fprintf(stderr, "r3_policy_replay: unexpected argument %s\n", argv[i]); return 2; }
        arg[argv[i] + 2] = argv[i + 1];
    }
    for (const char* k : {"series", "start-heat", "file-layers", "file-experts", "layers", "experts", "budgets-mib", "stride-bytes", "shares",
                          "half-life-observations", "margin", "max-moves", "max-moved-bytes", "min-samples", "period-s", "min-residency-s", "cadence"})
        if (!arg.count(k)) { std::fprintf(stderr, "r3_policy_replay: --%s is required (no defaults)\n", k); return 2; }
    const uint32_t fl = uint32_t(std::stoul(arg["file-layers"])), fe = uint32_t(std::stoul(arg["file-experts"]));
    ReplayConfig cfg;
    cfg.layers = uint32_t(std::stoul(arg["layers"])); cfg.experts = uint32_t(std::stoul(arg["experts"]));
    if (cfg.layers > fl || cfg.experts > fe) { std::fprintf(stderr, "r3_policy_replay: layers x experts exceed the file shape\n"); return 2; }
    for (const auto& m : split(arg["budgets-mib"])) cfg.budgets.push_back(uint64_t(std::stoull(m)) << 20);
    for (const auto& w : split(arg["shares"])) cfg.shares.push_back(uint32_t(std::stoul(w)));
    cfg.stride = std::stoull(arg["stride-bytes"]);
    cfg.half_life_observations = std::stod(arg["half-life-observations"]);
    cfg.margin = std::stod(arg["margin"]);
    cfg.max_moves = uint32_t(std::stoul(arg["max-moves"]));
    cfg.max_moved_bytes = std::stoull(arg["max-moved-bytes"]);
    cfg.min_samples = std::stoull(arg["min-samples"]);
    cfg.period_ns = uint64_t(std::stod(arg["period-s"]) * 1e9);
    cfg.min_residency_ns = uint64_t(std::stod(arg["min-residency-s"]) * 1e9);
    cfg.cadence_snapshots = std::stoull(arg["cadence"]);
    const size_t n = size_t(cfg.layers) * cfg.experts;
    auto crop = [&](const std::vector<uint32_t>& full) {
        std::vector<uint32_t> v(n);
        for (uint32_t l = 0; l < cfg.layers; ++l) for (uint32_t e = 0; e < cfg.experts; ++e) v[size_t(l) * cfg.experts + e] = full[size_t(l) * fe + e];
        return v;
    };
    std::vector<uint32_t> start;
    if (!read_u32(arg["start-heat"], &start) || start.size() != size_t(fl) * fe) { std::fprintf(stderr, "r3_policy_replay: cannot read --start-heat as uint32 [%u][%u]\n", fl, fe); return 1; }
    const std::vector<uint32_t> ranking = crop(start);
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), size_t(0));
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return ranking[a] < ranking[b]; });   // max - count: smaller is hotter
    std::vector<int64_t> bytes(n, int64_t(cfg.stride));
    uint64_t budgets[r3::worker_share::kMaxHelperShares] = {};
    if (cfg.budgets.size() > r3::worker_share::kMaxHelperShares || cfg.shares.size() != cfg.budgets.size()) { std::fprintf(stderr, "r3_policy_replay: one share for each helper\n"); return 2; }
    for (size_t d = 0; d < cfg.budgets.size(); ++d) budgets[d] = cfg.budgets[d];
    const std::vector<int32_t> placed = r3::worker_share::plan_tail(bytes, uint32_t(cfg.budgets.size()), cfg.shares.data(), budgets);
    std::vector<int32_t> owner(n, -1);
    for (size_t k = 0; k < n; ++k) owner[order[k]] = placed[k];
    std::vector<std::vector<uint32_t>> snapshots;
    for (const std::string& name : list_dir(arg["series"])) {
        std::vector<uint32_t> full;
        if (!read_u32(arg["series"] + "/" + name, &full) || full.size() != size_t(fl) * fe) { std::fprintf(stderr, "r3_policy_replay: %s is not uint32 [%u][%u]\n", name.c_str(), fl, fe); return 1; }
        snapshots.push_back(crop(full));
    }
    const ReplayResult r = replay(cfg, snapshots, owner);
    if (!r.valid) { std::fprintf(stderr, "r3_policy_replay: replay refused (need two or more snapshots, a valid starting layout inside the budgets, and one share a helper)\n"); return 1; }
    std::printf("snapshot\tobservations\ttail_share_before\ttail_share_after\tmoved\n");
    for (const ReplayStep& s : r.steps)
        std::printf("%zu\t%llu\t%.6f\t%.6f\t%zu\n", s.snapshot, (unsigned long long)s.observations, s.tail_share_before, s.tail_share_after, s.moved);
    std::printf("# snapshots=%zu transactions=%zu experts_moved=%zu moved_bytes=%llu hot_set_k=%zu drift_half_life_observations=%llu marginal_relative_error=%.6f\n",
                snapshots.size(), r.transactions, r.experts_moved, (unsigned long long)(r.experts_moved * cfg.stride), r.hot_set_k,
                (unsigned long long)r.drift_half_life_observations, r.marginal_error);
    return 0;
}
