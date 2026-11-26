/* bc_serving_managed_bind.cpp -- bc_serving_managed_bind (RSC02, rad0070): the part of the managed native entry that
 * reads and writes the engine's parsed Config, so it is built into rad_core only with the patch that calls it. */
#include "bc_serving_managed.h"

#include "engine_admission.h"

#ifdef BC_RAD_NATIVE_FLEET_CONFIG
#include "fleet/config.h"
#include "engine.h"
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <filesystem>
#include <cstdint>
#include <string>
#include <vector>

namespace {
int refuse(const std::string& why) { return bc_serving_managed_refuse(why); }

#ifdef BC_RAD_NATIVE_FLEET_CONFIG
// A KL score that works with helpers is run only by the owner that holds this very admission: its parent is the matched
// runtime's bc-serve and keeps the same sealed document open.
bool matched_quality_owner(const std::string& home, int admission_fd) {
    namespace fs = std::filesystem;
    const pid_t owner = ::getppid();
    if (owner <= 1) return false;
    const fs::path expected = fs::path(home) / "bin/bc-serve";
    const fs::path parent = fs::path("/proc") / std::to_string(owner);
    std::error_code error;
    if (!expected.is_absolute() || fs::canonical(expected, error) != expected || error) return false;
    if (fs::read_symlink(parent / "exe", error) != expected || error) return false;
    struct stat local {}, retained {}, runtime {}, executable {};
    return ::fstat(admission_fd, &local) == 0 &&
           ::stat((parent / "fd" / std::to_string(admission_fd)).c_str(), &retained) == 0 &&
           local.st_dev == retained.st_dev && local.st_ino == retained.st_ino &&
           ::lstat(expected.c_str(), &runtime) == 0 && S_ISREG(runtime.st_mode) &&
           ::stat((parent / "exe").c_str(), &executable) == 0 && runtime.st_dev == executable.st_dev &&
           runtime.st_ino == executable.st_ino && ::getppid() == owner;
}
#endif
}  // namespace

#ifdef BC_RAD_NATIVE_FLEET_CONFIG
int bc_serving_managed_bind(rad::fleet::FleetConfig* fleet) {
    const bc::serving::EngineAdmission* admitted = bc_serving_managed_admission();
    if (!admitted) return 0;   // the stock entry
    const bc::serving::EngineAdmission& a = *admitted;
    // The single-model contract of a managed start. Anything else is a different product (Fleet's broker, parking, a
    // second model) that this owner does not admit.
    if (fleet->models.size() != 1) return refuse("exactly one model is served; the file names " + std::to_string(fleet->models.size()));
    if (!fleet->card_headroom_mib.empty())
        return refuse("memory.card-headroom-mib is not honoured by a single-model managed start on this runtime; remove it");
    rad::fleet::ModelEntry& model = fleet->models.front();
    rad::Config& cfg = model.cfg;
    if (!a.quality && (cfg.kld() || cfg.calib() || cfg.diagnose_mode() || cfg.debug_graph))
        return refuse("the file selects an offline or debug mode; a managed start serves");
    if (cfg.host != "127.0.0.1") return refuse("only the managed loopback listener is supported (server.host is " + cfg.host + ")");
    if (static_cast<size_t>(std::max(cfg.tp, 1)) != a.bdfs.size())
        return refuse("tp " + std::to_string(cfg.tp) + " differs from the " + std::to_string(a.bdfs.size()) + " admitted engine devices");
    std::vector<int> ordinals(a.bdfs.size());
    for (size_t i = 0; i < ordinals.size(); ++i) ordinals[i] = static_cast<int>(i);
    if (!fleet->cards.empty() && fleet->cards != ordinals) return refuse("the file's cards differ from the selected rank order");
    // rank-devices is a machine fact: a file that omits it takes the admitted ordered selection, one that states it must match.
    if (a.rank_devices) {
        if (cfg.given.find("--rank-devices") == cfg.given.end()) {
            cfg.rank_devices = *a.rank_devices;
            cfg.given.insert_or_assign("--rank-devices", "engine admission");
        } else if (cfg.rank_devices != *a.rank_devices) {
            return refuse("rank-devices in the file differ from the admitted ordered BDF selection");
        }
    }
    // A derived host size applies to an absent (or auto) option, or agrees exactly; a disagreement names both.
    const auto bind_host = [&](const char* option, const char* key, int64_t& value, int64_t derived, bool allow_auto) {
        const auto authored = cfg.given.find(option);
        if (authored != cfg.given.end() && !(allow_auto && value == -1)) {
            if (value == derived) return true;
            refuse(std::string(key) + " in the file is " + std::to_string(value) + " but the owner derived " + std::to_string(derived));
            return false;
        }
        value = derived;
        cfg.given.insert_or_assign(option, "engine admission");
        return true;
    };
    if (!bind_host("--host-pool-mib", "memory.host-pool-mib (per rank)", cfg.host_pool_mib, a.host_pool_mib_per_rank, false)) return 2;
    if (a.quality) {
        // An offline KL run scores the trunk's own rows: no managed serving cache or checkpoint store, its own files as
        // the upstream flags the run would have been given, and the ranks as the visible devices.
        std::vector<std::string> flags = {"radiance"};
        if (a.quality->record)
            flags.insert(flags.end(), {"--kld-record", a.quality->reference, "--kld-corpus", a.quality->corpus});
        else
            flags.insert(flags.end(), {"--kld-ref", a.quality->reference, "--kld-out", a.quality->output});
        std::vector<char*> argv;
        for (std::string& f : flags) argv.push_back(f.data());
        argv.push_back(nullptr);
        if (rad::config_parse(static_cast<int>(flags.size()), argv.data(), &cfg) != RAD_OK || !cfg.kld())
            return refuse("the KL run files are not a valid record or score invocation");
        if (a.handoff && !matched_quality_owner(cfg.radiance_home, bc_serving_managed_admission_fd()))
            return refuse("KL scoring with helpers requires its matched owner and its retained admission");
        // A KL run scores the trunk's own rows: the engine refuses an explicit speculative depth rather than ignoring it
        // ("leave it at auto or 0"), and a serving config states one (the bundled configs say 3). Quality-only, explicit and
        // recorded: the one authored option a KL run overrides (the engine itself turns the drafter, the prefix cache and
        // multimodal patches off for it). The retired manifest path forced nothing: KL runs there took a config that already
        // said 0 or auto. A serving start never reaches this branch.
        cfg.n_spec = 0;
        cfg.given.insert_or_assign("--num-speculative-tokens", "engine admission: a KL run scores the trunk");
        cfg.devices = ordinals;
        bc_serving_managed_mark();
        return 0;
    }
    if (a.host_kv_mib > INT32_MAX) return refuse("the admitted host KV tier exceeds the native prefix-cache-mib range");
    if (!bind_host("--prefix-cache-mib", "prefix-cache-mib", cfg.prefix_cache_mib, a.host_kv_mib, false) ||
        !bind_host("--host-headroom-mib", "host-headroom-mib", cfg.host_headroom_mib, a.host_reserve_mib, false))
        return 2;
    if (a.checkpoint_slots &&
        !bind_host("--checkpoint-slots", "checkpoint-slots", cfg.checkpoint_slots, *a.checkpoint_slots, true))
        return 2;
    bc_serving_managed_mark();
    return 0;
}
#else
int bc_serving_managed_bind(rad::fleet::FleetConfig*) { return 0; }
#endif
