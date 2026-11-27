// RSC02 step 3: replay and heat guard semantics. No GPU.
#include "../moe-remote/heat_publication.h"
#include "../moe-remote/worker_heat_source.h"
#include "runtime_guards.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;
using namespace bc::serving::session;

namespace {
void expect(bool ok, const char* description) {
    if (!ok)
        throw std::runtime_error(description);
}
struct Temporary {
    fs::path root;
    Temporary() {
        std::string name = (fs::temp_directory_path() / "bc-rsc02-guards-XXXXXX").string();
        std::vector<char> buffer(name.begin(), name.end());
        buffer.push_back('\0');
        const auto* made = ::mkdtemp(buffer.data());
        if (!made)
            throw std::runtime_error("mkdtemp failed");
        root = made;
    }
    ~Temporary() {
        std::error_code unused;
        fs::remove_all(root, unused);
    }
    void write(std::string_view name, std::string_view contents) const {
        std::ofstream(root / name) << contents;
    }
};

constexpr std::string_view kLaunchNonce = "0123456789abcdef0123456789abcdef";
constexpr std::string_view kOtherNonce = "fedcba9876543210fedcba9876543210";

void publish_like_engine(const fs::path& rank, const std::vector<uint32_t>& counts,
                         std::string_view nonce = kLaunchNonce) {
    const std::string filename = rank.string();
    expect(filename.size() > 6 && filename.substr(filename.size() - 6) == ".rank0",
           "producer fixture must use the actual rank path");
    expect(r3::heat_publication::publish(filename.substr(0, filename.size() - 6), 0,
                                         static_cast<uint32_t>(counts.size() / r3::moe_shm::kMaxLayers), counts,
                                         nonce),
           "the engine producer's shared publication function must succeed");
}

void producer_contract_case(std::string_view which) {
    using r3::moe_shm::kMaxLayers;
    Temporary tmp;
    const fs::path rank = tmp.root / "engine.rank0";
    const auto now = std::chrono::system_clock::now() + std::chrono::seconds(1);
    const auto launch = now - std::chrono::seconds(10);
    std::vector<uint32_t> counts(static_cast<size_t>(kMaxLayers) * 2, 0);
    counts[0] = 20;
    publish_like_engine(rank, counts, which == "foreign" ? std::string_view{} : kLaunchNonce);

    if (which == "layers") {
        const HistogramExpectation model{4, 2, launch, {}, std::string(kLaunchNonce)};
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), model) == HistogramVerdict::fresh,
               "real 128-layer producer record must pass for a four-layer model");
        // The unchanged worker reader must accept the producer's original
        // binary layout even when the new optional proof sidecar is present.
        r3::policy::HistogramFileSource worker(rank.string(), 4, 2);
        r3::policy::HeatDelta delta;
        std::string worker_error;
        const auto realtime_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
        expect(worker.poll(1, realtime_ns, &delta, &worker_error) == r3::policy::HeatPoll::BASELINE &&
                   delta.complete_cells && delta.layers == 4 && delta.experts == 2,
               "legacy worker reader still accepts raw producer data with optional sidecar");

        counts[size_t(4) * 2U] = 1;
        publish_like_engine(rank, counts);
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), model) == HistogramVerdict::unsafe,
               "counts in padded layers cannot qualify the authenticated model");
    } else if (which == "rename") {
        const HistogramExpectation model{kMaxLayers, 2, launch, {}, std::string(kLaunchNonce)};
        HistogramIdentity first{};
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), model, &first) == HistogramVerdict::fresh,
               "initial producer snapshot must pass");
        counts[0] += 4;
        publish_like_engine(rank, counts);
        auto next = model;
        next.identity = first;
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), next) == HistogramVerdict::fresh,
               "atomic producer refresh must not fail due to changed inode");
    } else if (which == "reset") {
        const HistogramExpectation model{kMaxLayers, 2, launch, {}, std::string(kLaunchNonce)};
        HistogramIdentity first{};
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), model, &first) == HistogramVerdict::fresh,
               "initial counter snapshot must pass");
        counts[0] = 1;
        publish_like_engine(rank, counts);
        auto next = model;
        next.identity = first;
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), next) == HistogramVerdict::unsafe,
               "counter reset/wrap must not qualify as a new monotonic sample");
    } else if (which == "foreign") {
        const HistogramExpectation model{kMaxLayers, 2, launch, {}, std::string(kLaunchNonce)};
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), model) == HistogramVerdict::unsafe,
               "post-launch histogram with no launch nonce must be refused");
        publish_like_engine(rank, counts);
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), model) == HistogramVerdict::fresh,
               "matching launch-bound proof qualifies");
        publish_like_engine(rank, counts, kOtherNonce);
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), model) == HistogramVerdict::unsafe,
               "a different launch nonce cannot authenticate the file");
        // An unrelated writer may atomically replace the raw file, but cannot
        // emit a matching companion with the secret nonce.
        std::ofstream unrelated(rank.string() + ".tmp", std::ios::binary | std::ios::trunc);
        unrelated.write(reinterpret_cast<const char*>(counts.data()),
                        static_cast<std::streamsize>(counts.size() * sizeof(uint32_t)));
        unrelated.close();
        fs::rename(rank.string() + ".tmp", rank);
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), model) == HistogramVerdict::unsafe,
               "raw publication with stale sidecar cannot hijack this session");
    } else if (which == "mtp_layer") {
        // Flash-Next's container has 49 routed layers (0..48): layer 48 is the MTP layer and the engine counts routes
        // in it. A 48-layer expectation would call those counts padding; the 49-layer one the container gives accepts
        // them, and the padding check beyond it stays strict.
        counts[size_t(48) * 2U] = 300;
        publish_like_engine(rank, counts);
        const HistogramExpectation container{49, 2, launch, {}, std::string(kLaunchNonce)};
        std::string why;
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), container, nullptr, &why) == HistogramVerdict::fresh,
               "counts in the last (MTP) layer of a 49-layer container are demand");
        const HistogramExpectation short_by_one{48, 2, launch, {}, std::string(kLaunchNonce)};
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), short_by_one, nullptr, &why) == HistogramVerdict::unsafe &&
                   why.find("beyond the model's routed layers") != std::string::npos,
               "the same counts are padding to a 48-layer expectation, and the reason says so");
        counts[size_t(49) * 2U] = 1;
        publish_like_engine(rank, counts);
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), container, nullptr, &why) == HistogramVerdict::unsafe,
               "a count beyond the 49th layer is still padding");
    } else if (which == "proof_lag") {
        // A heavily loaded disk stretches the gap between the producer's two renames (raw file, then its proof): a poll
        // inside the gap sees the new raw file beside the previous generation's proof. Within the grace it is re-read
        // and qualifies; past it the pair is unsafe and the reason names what differs.
        HistogramExpectation model{49, 2, launch, {}, std::string(kLaunchNonce)};
        counts[0] = 9;
        publish_like_engine(rank, counts);                       // generation 1: raw + matching proof
        counts[0] = 10;
        publish_like_engine(rank, counts, kOtherNonce);          // generation 2 raw, a proof of another launch: the gap
        std::thread producer([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            publish_like_engine(rank, counts);                   // the proof catches up
        });
        std::string why;
        model.proof_grace = std::chrono::seconds(5);
        // join before checking: a failed expect must report itself, not unwind past a joinable thread (std::terminate)
        const bool caught_up = histogram_verdict(rank, now, std::chrono::seconds(30), model, nullptr, &why) == HistogramVerdict::fresh;
        producer.join();
        expect(caught_up, "a proof that catches up within the grace qualifies");
        publish_like_engine(rank, counts, kOtherNonce);
        model.proof_grace = std::chrono::milliseconds(60);
        const auto t0 = std::chrono::steady_clock::now();
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), model, nullptr, &why) == HistogramVerdict::unsafe &&
                   why.find("launch nonce") != std::string::npos && why.find("grace") != std::string::npos &&
                   std::chrono::steady_clock::now() - t0 >= std::chrono::milliseconds(60),
               "a proof that never catches up is unsafe after the grace and the reason says which field differs");
    } else if (which == "startup_wait") {
        const HistogramExpectation model{49, 2, launch, {}, std::string(kLaunchNonce)};
        std::vector<uint32_t> zero(static_cast<size_t>(kMaxLayers) * 2, 0);
        publish_like_engine(rank, zero);
        // an authentic all-zero file is waited for: the engine's next snapshot carries the canary's counts
        std::thread engine([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            auto later = zero;
            later[size_t(48) * 2U] = 7;
            publish_like_engine(rank, later);
        });
        const auto before = std::chrono::steady_clock::now();
        std::string why;
        const bool qualified = histogram_verdict_startup(rank, std::chrono::seconds(30), model, std::chrono::seconds(5),
                                                         nullptr, &why) == HistogramVerdict::fresh;
        const bool within_bound = std::chrono::steady_clock::now() - before < std::chrono::seconds(4);
        engine.join();
        expect(qualified, "the startup check waits for the first counted snapshot and then qualifies it");
        expect(within_bound, "and does not wait out the bound");
        // still empty at the end of the wait: pending (the caller refuses, with the reason)
        publish_like_engine(rank, zero);
        const auto t0 = std::chrono::steady_clock::now();
        expect(histogram_verdict_startup(rank, std::chrono::seconds(30), model, std::chrono::milliseconds(200), nullptr,
                                         &why) == HistogramVerdict::pending &&
                   std::chrono::steady_clock::now() - t0 >= std::chrono::milliseconds(200),
               "an all-zero file that stays empty is waited for exactly the bound, then pending");
        // no wait declared: decided at once
        const auto t1 = std::chrono::steady_clock::now();
        expect(histogram_verdict_startup(rank, std::chrono::seconds(30), model, std::chrono::nanoseconds{0}, nullptr) ==
                   HistogramVerdict::pending && std::chrono::steady_clock::now() - t1 < std::chrono::milliseconds(150),
               "no declared wait means no wait");
        // every other failure is immediate, however long the bound
        publish_like_engine(rank, zero, kOtherNonce);
        const auto t2 = std::chrono::steady_clock::now();
        expect(histogram_verdict_startup(rank, std::chrono::seconds(30), model, std::chrono::seconds(5), nullptr, &why) ==
                       HistogramVerdict::unsafe &&
                   std::chrono::steady_clock::now() - t2 < std::chrono::seconds(1) &&
                   why.find("proof") != std::string::npos,
               "a wrong launch nonce is refused at once, never waited for");
        auto padded = zero;
        padded[size_t(60) * 2U] = 1;
        publish_like_engine(rank, padded);
        const auto t3 = std::chrono::steady_clock::now();
        expect(histogram_verdict_startup(rank, std::chrono::seconds(30), model, std::chrono::seconds(5), nullptr) ==
                       HistogramVerdict::unsafe &&
                   std::chrono::steady_clock::now() - t3 < std::chrono::seconds(1),
               "padded counts are refused at once, never waited for");
    } else {
        throw std::runtime_error("unknown heat regression case");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2) {
            producer_contract_case(argv[1]);
            return 0;
        }
        expect(argc == 1, "invalid test arguments");
        for (const char* check : {"layers", "rename", "reset", "foreign", "mtp_layer", "startup_wait", "proof_lag"})
            producer_contract_case(check);
        using bc::serving::v2::LivePolicy;
        expect(replay_verdict(1, false, "") == ReplayVerdict::disabled, "MTP depth <= 1 is exempt");
        expect(replay_verdict(4, true, "") == ReplayVerdict::profiled, "profile-ops is exempt");
        expect(replay_verdict(4, false, "recorded pass: draft pass 0 \n") == ReplayVerdict::absent,
               "only draft pass zero is the serial replay regression");
        expect(replay_verdict(4, false, "recorded pass: draft pass -1 token\n") == ReplayVerdict::observed,
               "negative nonzero pass counts as replay");
        expect(replay_verdict(4, false, "recorded pass: draft pass 8 token\n") == ReplayVerdict::observed,
               "positive nonzero pass counts as replay");
        expect(replay_verdict(4, false, "recorded pass: draft pass x token\n") == ReplayVerdict::absent,
               "nonnumeric pass cannot satisfy the gate");

        RuntimeGuards runtime;
        expect(runtime.poll().has_value(), "unbound live policy guard must fail closed");
        expect(runtime.startup({}, "missing-log", {}).has_value(),
               "unauthenticated manifest cannot pass the startup gate");

        Temporary tmp;
        // The synthetic observations are published after test setup; allow
        // one second of clock margin rather than misclassifying them as future.
        const auto now = std::chrono::system_clock::now() + std::chrono::seconds(1);
        const auto rank = tmp.root / "test.rank0";
        const auto launch = now - std::chrono::minutes(1);
        const HistogramExpectation expected{4, 2, launch, {}, std::string(kLaunchNonce)};
        std::vector<uint32_t> sample(size_t(r3::moe_shm::kMaxLayers) * 2, 0);
        publish_like_engine(rank, sample);
        std::string zero_reason;
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), expected, nullptr, &zero_reason) ==
                       HistogramVerdict::pending &&
                   zero_reason.find("no route counts yet") != std::string::npos,
               "all-zero heat is authentic but no evidence yet: pending, never fresh");
        sample[0] = 3;
        publish_like_engine(rank, sample);
        const std::string observed(reinterpret_cast<const char*>(sample.data()),
                                   sample.size() * sizeof(uint32_t));
        HistogramIdentity identity{};
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), expected, &identity) ==
                       HistogramVerdict::fresh &&
                   identity.inode != 0,
               "a complete and nonzero current-engine histogram may qualify");
        auto bound = expected;
        bound.identity = identity;
        expect(histogram_verdict(rank, now + std::chrono::minutes(5), std::chrono::seconds(30), bound) ==
                   HistogramVerdict::stale,
               "an unchanged heat file cannot remain live forever");
        expect(histogram_verdict(rank, now - std::chrono::minutes(5), std::chrono::seconds(30), bound) ==
                   HistogramVerdict::future,
               "future-dated heat cannot be reported fresh");
        expect(histogram_verdict(tmp.root / "missing", now, std::chrono::seconds(30), expected) ==
                   HistogramVerdict::missing,
               "an absent engine heat stream must not be fabricated");
        tmp.write("empty.rank0", "");
        expect(histogram_verdict(tmp.root / "empty.rank0", now, std::chrono::seconds(30), expected) ==
                   HistogramVerdict::unsafe,
               "an empty stream is not an observation");
        tmp.write("partial.rank0", observed.substr(0, 12));
        expect(histogram_verdict(tmp.root / "partial.rank0", now, std::chrono::seconds(30), expected) ==
                   HistogramVerdict::unsafe,
               "partial rank0 histogram cannot pass layout dimensions");
        auto wrong_dimensions = expected;
        wrong_dimensions.experts = 3;
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), wrong_dimensions) ==
                   HistogramVerdict::unsafe,
               "wrong layout histogram dimensions fail");
        auto wrong_run = expected;
        wrong_run.launched_at = now + std::chrono::milliseconds(1);
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), wrong_run) == HistogramVerdict::unsafe,
               "prelaunch histogram cannot represent this engine run");
        tmp.write("replacement.rank0", observed);
        fs::rename(tmp.root / "replacement.rank0", rank);
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), bound) == HistogramVerdict::unsafe,
               "a replacement raw record without matching versioned proof cannot hijack the stream");
        fs::create_symlink(rank, tmp.root / "symlink.rank0");
        expect(histogram_verdict(tmp.root / "symlink.rank0", now, std::chrono::seconds(30), expected) ==
                   HistogramVerdict::unsafe,
               "rank histogram must not be redirected through a symlink");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RSC02 step 3 test failed: " << error.what() << '\n';
        return 1;
    }
}
