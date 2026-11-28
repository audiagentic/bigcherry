/* The managed native entry (RSC02, rad0070), card-free: the stock entry is untouched without the flag, a bad admission is
 * refused before anything starts, bind applies derived values only where the file leaves them to the owner, and the chosen
 * cards must be the admitted ranks. The process-global managed flag is set once by the last case (it is never cleared). */
#include "bc_serving_managed.h"

#include "engine_admission.h"
#include "fleet/config.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {
using Json = nlohmann::json;

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

int sealed_memfd(const std::string& bytes, bool seal) {
    const int fd = ::memfd_create("bc-admission-test", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    check(fd >= 3 && ::write(fd, bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size()), "memfd");
    if (seal) check(::fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) == 0, "seal");
    return fd;
}

bc::serving::EngineAdmission admission() {
    bc::serving::EngineAdmission a;
    a.measuring = true;   // no expected plan to build here: the plan check is rad0011's and has its own tests
    a.bdfs = {"0000:03:00.0", "0000:06:00.0"};
    a.rank_devices = std::vector<std::string>{"0000:03:00.0", "0000:06:00.0"};
    a.host_pool_mib_per_rank = 2048;
    a.host_kv_mib = 4096;
    a.host_reserve_mib = 1024;
    a.checkpoint_slots = 6;
    a.kv_backing_floor_ppm = 1000;
    a.shm = "/r3-test";
    return a;
}

// Runs the take-flag hook on a fresh argv carrying `admission`; returns its exit code.
int take(const bc::serving::EngineAdmission& a, std::vector<std::string> extra = {}, bool seal = true,
         const std::string& override_text = {}) {
    const std::string text = override_text.empty() ? bc::serving::engine_admission_to_json(a).dump() : override_text;
    const int fd = sealed_memfd(text, seal);
    std::vector<std::string> args = {"radiance", "--config", "/proc/self/fd/9", "--bc-admission-fd", std::to_string(fd)};
    args.insert(args.end(), extra.begin(), extra.end());
    std::vector<char*> argv;
    for (std::string& s : args) argv.push_back(s.data());
    argv.push_back(nullptr);
    int argc = static_cast<int>(args.size());
    char** pointer = argv.data();
    const int rc = bc_serving_managed_take_flag(&argc, &pointer);
    if (rc == 0) {   // the owner's flags are gone, the stock ones remain
        check(argc == 3 && std::string(pointer[1]) == "--config" && pointer[3] == nullptr, "the stock arguments changed");
    }
    ::close(fd);
    return rc;
}

rad::fleet::FleetConfig fleet(int tp = 2) {
    rad::fleet::FleetConfig f;
    f.path = "test.yaml";
    rad::fleet::ModelEntry m;
    m.name = "m";
    m.cfg.tp = tp;
    m.cfg.model = "m.rad";   // an authored file always names its model (the KL overlay parse needs it)
    m.cfg.host = "127.0.0.1";
    f.models.push_back(m);
    return f;
}
void expect_refused(rad::fleet::FleetConfig f, const char* needle) {
    check(bc_serving_managed_bind(&f) == 2, std::string("accepted: ") + needle);
    check(bc_serving_managed_last_refusal().find(needle) != std::string::npos,
          std::string("refusal does not name '") + needle + "': " + bc_serving_managed_last_refusal());
    check(!bc_serving_managed(), "a refused bind marked the process managed");
}
}  // namespace

int main() {
    try {
        // Stock: without the flag nothing changes and nothing is managed.
        {
            std::vector<std::string> args = {"radiance", "--config", "x.yaml", "--check"};
            std::vector<char*> argv;
            for (std::string& s : args) argv.push_back(s.data());
            argv.push_back(nullptr);
            int argc = 4;
            char** pointer = argv.data();
            check(bc_serving_managed_take_flag(&argc, &pointer) == 0 && argc == 4 && pointer == argv.data() &&
                      std::string(pointer[3]) == "--check",
                  "the stock argv was touched");
            rad::fleet::FleetConfig f = fleet();
            const rad::fleet::FleetConfig before = f;
            check(bc_serving_managed_bind(&f) == 0 && f.models[0].cfg.host_pool_mib == before.models[0].cfg.host_pool_mib &&
                      f.models[0].cfg.given.empty() && !bc_serving_managed(),
                  "the stock bind changed something");
            check(bc_serving_managed_chosen({1, 0}).empty(), "the stock entry checked the chosen cards");
            check(bc_serving_managed_usable({0, 1}) == std::vector<int>({0, 1}) && bc_serving_managed_usable({0}) == std::vector<int>({0}),
                  "the stock entry changed the usable cards");
        }
        // Refused documents and flags: nothing installed, exit 2.
        const bc::serving::EngineAdmission good = admission();
        check(take(good, {}, false) == 2 && bc_serving_managed_last_refusal().find("sealed") != std::string::npos, "an unsealed descriptor");
        check(take(good, {}, true, "{\"kind\":\"other\"}") == 2, "a foreign document");
        check(take(good, {"--bc-heat-launch-nonce", std::string(32, 'a')}) == 2, "a heat nonce for a measuring start");
        check(take(good, {"--bc-step-phase-dir", "relative"}) == 2, "a relative step-phase dir");
        check(take(good, {"--bc-heat-launch-nonce", "x", "--bc-heat-launch-nonce", "y"}) == 2, "a repeated flag");
        // Accepted document, then the bind matrix (each refusal leaves the process unmanaged).
        check(take(good) == 0, "a good admission was refused");
        {   // mixed architectures (2x gfx1100 + gfx1201): Radiance's one-architecture filter leaves [0,1]; the admitted ranks are all three
            bc::serving::EngineAdmission mixed = good;
            mixed.bdfs = {"0000:03:00.0", "0000:06:00.0", "0000:09:00.0"};
            mixed.rank_devices = mixed.bdfs;
            check(take(mixed) == 0, "a three-rank admission was refused");
            check(bc_serving_managed_usable({0, 1}) == std::vector<int>({0, 1, 2}), "a mixed-architecture start lost its third rank");
            check(take(good) == 0, "the two-rank admission was refused again");
            check(bc_serving_managed_usable({0}) == std::vector<int>({0, 1}), "the usable cards are not the admitted ranks");
        }
        {
            rad::fleet::FleetConfig two = fleet();
            two.models.push_back(two.models[0]);
            expect_refused(two, "exactly one model");
            rad::fleet::FleetConfig headroom = fleet();
            headroom.card_headroom_mib[0] = 512;
            expect_refused(headroom, "card-headroom");
            rad::fleet::FleetConfig waived = fleet();
            waived.models[0].cfg.proceed_incompatible = true;
            expect_refused(waived, "cannot waive engine capability checks");
            rad::fleet::FleetConfig exposed = fleet();
            exposed.models[0].cfg.host = "0.0.0.0";
            expect_refused(exposed, "loopback");
            expect_refused(fleet(3), "tp 3");
            rad::fleet::FleetConfig cards = fleet();
            cards.cards = {1, 0};
            expect_refused(cards, "rank order");
            rad::fleet::FleetConfig ranks = fleet();
            ranks.models[0].cfg.rank_devices = {"0000:06:00.0", "0000:03:00.0"};
            ranks.models[0].cfg.given["--rank-devices"] = "yaml";
            expect_refused(ranks, "rank-devices");
            rad::fleet::FleetConfig pool = fleet();
            pool.models[0].cfg.host_pool_mib = 1000;
            pool.models[0].cfg.given["--host-pool-mib"] = "yaml";
            expect_refused(pool, "1000");   // names the authored value ...
            check(bc_serving_managed_last_refusal().find("2048") != std::string::npos, "... and the derived one");
        }
        check(bc_serving_managed_chosen({0, 1}).empty(), "the admitted order was refused");
        check(bc_serving_managed_chosen({1, 0}).find("rank order") != std::string::npos, "a reversed order was admitted");
        check(!bc_serving_managed_chosen({0}).empty(), "a short choice was admitted");
        // The accepted bind: absent values are bound and recorded; an authored value that agrees stays; auto checkpoint binds.
        {
            rad::fleet::FleetConfig f = fleet();
            f.models[0].cfg.host_pool_mib = 2048;
            f.models[0].cfg.given["--host-pool-mib"] = "yaml";
            f.models[0].cfg.checkpoint_slots = -1;
            f.models[0].cfg.given["--checkpoint-slots"] = "yaml";
            check(bc_serving_managed_bind(&f) == 0, "the good bind was refused: " + bc_serving_managed_last_refusal());
            const rad::Config& c = f.models[0].cfg;
            check(c.host_pool_mib == 2048 && c.prefix_cache_mib == 4096 && c.host_headroom_mib == 1024 && c.checkpoint_slots == 6,
                  "derived values were not bound");
            check(c.rank_devices == std::vector<std::string>({"0000:03:00.0", "0000:06:00.0"}), "rank devices were not bound");
            check(c.given.at("--prefix-cache-mib") == "engine admission", "the source of a bound value is not recorded");
            check(bc_serving_managed(), "an accepted bind did not mark the process managed");
        }
        {   // A KL run scores the trunk: its start overrides the authored speculative depth (3 in the bundled configs) with 0,
            // recorded; a serving start keeps what the file says.
            bc::serving::EngineAdmission kl = admission();
            kl.measuring = false;
            kl.quality = bc::serving::EngineAdmission::Quality{false, "/ref", "", "/out.json"};
            check(take(kl) == 0, "a KL admission was refused");
            rad::fleet::FleetConfig f = fleet();
            f.models[0].cfg.n_spec = 3;
            f.models[0].cfg.given["--num-speculative-tokens"] = "yaml";
            check(bc_serving_managed_bind(&f) == 0, "the KL bind was refused: " + bc_serving_managed_last_refusal());
            check(f.models[0].cfg.n_spec == 0 && f.models[0].cfg.kld() &&
                      f.models[0].cfg.given.at("--num-speculative-tokens").find("KL run") != std::string::npos,
                  "a KL start kept the authored speculative depth");
            check(take(admission()) == 0, "the serving admission was refused");
            rad::fleet::FleetConfig serving = fleet();
            serving.models[0].cfg.n_spec = 3;
            serving.models[0].cfg.given["--num-speculative-tokens"] = "yaml";
            check(bc_serving_managed_bind(&serving) == 0 && serving.models[0].cfg.n_spec == 3 && !serving.models[0].cfg.kld(),
                  "a serving start changed the authored speculative depth");
        }
        check(bc_serving_managed_engine_exited() == false, "engine exit flagged early");
        bc_serving_managed_note_engine_exit();
        check(bc_serving_managed_engine_exited(), "engine exit not recorded");
        std::cout << "bc_native_managed_test: passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bc_native_managed_test: " << error.what() << "\n";
        return 1;
    }
}
