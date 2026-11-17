// RSC02 step 3: native bounded ABBA lab and replay guard semantics. No GPU.
#include "calibration_lab.h"
#include "runtime_guards.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
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
} // namespace

int main() {
    try {
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
        const HistogramExpectation expected{2, 2, launch, {}};
        tmp.write("test.rank0", std::string(16, '\0'));
        expect(histogram_verdict(rank, now, std::chrono::seconds(30), expected) == HistogramVerdict::unsafe,
               "all-zero heat is no recorded rank demand");
        std::string observed(16, '\0');
        observed[0] = 3; // four native-endian cumulative uint32 counters, one with demand
        tmp.write("test.rank0", observed);
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
               "a replacement inode cannot hijack the bound engine heat stream");
        fs::create_symlink(rank, tmp.root / "symlink.rank0");
        expect(histogram_verdict(tmp.root / "symlink.rank0", now, std::chrono::seconds(30), expected) ==
                   HistogramVerdict::unsafe,
               "rank histogram must not be redirected through a symlink");
        tmp.write("a.txt", "\n A prompt \n \t \n");
        tmp.write("b.txt", "B prompt\n");
        auto prompts = read_calibration_prompts(tmp.root / "a.txt", tmp.root / "b.txt");
        expect(bool(prompts) && prompts.value().a.size() == 1 && prompts.value().b.size() == 1,
               "ABBA needs the actual nonempty prompts, skipping whitespace");
        tmp.write("same_a.txt", "A prompt\n");
        tmp.write("same_b.txt", "A prompt\n");
        expect(!read_calibration_prompts(tmp.root / "same_a.txt", tmp.root / "same_b.txt"),
               "byte-identical A/B workload must be refused");
        tmp.write("same_b.txt", "  \nA prompt\n\t");
        expect(!read_calibration_prompts(tmp.root / "same_a.txt", tmp.root / "same_b.txt"),
               "identical effective nonblank batches must be refused");
        tmp.write("b.txt", "   \n");
        expect(!read_calibration_prompts(tmp.root / "a.txt", tmp.root / "b.txt"),
               "blank B workload must refuse, not run indefinitely");

        // Generated by tools/lab/radiance/calibration_watchdog.py::bounded_deadline,
        // not recomputed from the C++ implementation under test.
        std::ifstream golden_file(BC_RSC02_CALIBRATION_GOLDEN);
        expect(bool(golden_file), "Python watchdog goldens must be checked in");
        const auto fixtures = nlohmann::json::parse(golden_file);
        expect(fixtures.at("cases").size() >= 6, "watchdog oracle fixture coverage");
        for (const auto& row : fixtures.at("cases")) {
            LivePolicy fixture_policy{};
            const auto& p = row.at("policy");
            fixture_policy.cadence_ns = p.at("cadence_ns").get<int64_t>();
            fixture_policy.max_heat_age_ns = p.at("max_heat_age_ns").get<int64_t>();
            fixture_policy.min_residency_ns = p.at("min_residency_ns").get<int64_t>();
            fixture_policy.min_samples = p.at("min_samples").get<int64_t>();
            std::optional<DriftProgress> fixture_drift;
            if (row.at("drift").is_object()) {
                const auto& d = row.at("drift");
                fixture_drift = DriftProgress{d.at("capacity").get<uint64_t>(),
                                              d.at("intervals").get<uint64_t>(),
                                              d.at("horizon_observations").get<uint64_t>(),
                                              d.at("observation_rate_per_s").get<uint64_t>(),
                                              d.at("accepted_gap_ns").get<uint64_t>(),
                                              d.at("accepted_heat_deltas").get<uint64_t>()};
            }
            const auto envelope = calibration_envelope(
                fixture_policy, row.at("a_count").get<size_t>(), row.at("b_count").get<size_t>(),
                std::chrono::seconds(row.at("request_timeout_s").get<int64_t>()), fixture_drift);
            expect(bool(envelope), "watchdog golden must derive a valid bounded envelope");
            expect(envelope.value().limit == std::chrono::seconds(row.at("expected_limit_s").get<int64_t>()),
                   "native bounded deadline differs from Python watchdog golden");
            expect(envelope.value().prefeed ==
                       std::chrono::seconds(row.at("expected_prefeed_s").get<int64_t>()),
                   "prefeed deadline differs from Python watchdog golden");
            expect(envelope.value().report_period ==
                       std::chrono::seconds(row.at("expected_report_s").get<int64_t>()),
                   "report cadence differs from Python watchdog golden");
        }

        LivePolicy policy{};
        policy.cadence_ns = 1000000000;
        policy.max_heat_age_ns = 2000000000;
        policy.min_residency_ns = 3000000000;
        policy.min_samples = 4;
        auto base = calibration_envelope(policy, 1, 1, std::chrono::seconds(10), {});
        expect(bool(base), "valid positive policy derives a finite envelope");
        expect(base.value().prefeed == std::chrono::seconds(40), "four phases of observed policy cadence");
        expect(base.value().limit == std::chrono::seconds(80), "ABBA workload capped by actual prompts");
        expect(base.value().report_period == std::chrono::seconds(2) && !base.value().accepted_feed,
               "unmeasured feed never qualifies by default");
        DriftProgress drift{};
        drift.capacity = 10;
        drift.intervals = 7;
        drift.horizon_observations = 30;
        drift.observation_rate_per_s = 2;
        drift.accepted_gap_ns = 3000000000ULL;
        drift.accepted_heat_deltas = 4;
        auto measured = calibration_envelope(policy, 1, 1, std::chrono::seconds(10), drift);
        expect(bool(measured) && measured.value().accepted_feed, "measured accepted gaps qualify feed");
        expect(measured.value().limit == std::chrono::seconds(120),
               "watchdog golden: four phase drift extensions capped by one fixed workload budget");
        drift.capacity = 1;
        drift.horizon_observations = 0;
        drift.accepted_gap_ns = 1000000000ULL;
        auto short_drift = calibration_envelope(policy, 1, 1, std::chrono::seconds(10), drift);
        expect(bool(short_drift) && short_drift.value().limit == std::chrono::seconds(88),
               "watchdog golden: short drift contributes four two-second phases");
        drift.capacity = 100;
        drift.horizon_observations = 30;
        drift.accepted_gap_ns = 3000000000ULL;
        auto capped = calibration_envelope(policy, 1, 1, std::chrono::seconds(10), drift);
        expect(bool(capped) && capped.value().limit == std::chrono::seconds(120),
               "coarsening ring cannot slide deadline beyond the fixed two-workload cap");
        const auto preserved = monotonic_calibration_envelope(capped.value(), short_drift.value());
        expect(preserved.limit == std::chrono::seconds(120) && preserved.prefeed == base.value().prefeed &&
                   preserved.accepted_feed,
               "shortening drift never retracts admitted deadline or restarts prefeed clock");
        policy.cadence_ns = -1;
        expect(!calibration_envelope(policy, 1, 1, std::chrono::seconds(10), drift),
               "negative policy cannot wrap the finite horizon");

        tmp.write("helper.log", "helper: starting\n");
        const auto missing = read_calibration_drift(tmp.root / "helper.log");
        expect(bool(missing) && !missing.value(), "absent measured drift is not fabricated");
        expect(bool(complete_calibration_marker(tmp.root / "helper.log")) &&
                   !complete_calibration_marker(tmp.root / "helper.log").value(),
               "no settled marker means continue bounded collection");
        tmp.write("helper.log", "RHP19_DRIFT_PROGRESS_JSON "
                                "{\"capacity\":10,\"intervals\":7,\"horizon_observations\":30,"
                                "\"observation_rate_per_s\":2,\"accepted_gap_ns\":3000000000,"
                                "\"accepted_heat_deltas\":4}\n");
        const auto present = read_calibration_drift(tmp.root / "helper.log");
        expect(bool(present) && present.value() && present.value()->capacity == 10,
               "latest measured drift is decoded, not guessed from cadence");
        tmp.write("helper.log", "RHP19_DRIFT_PROGRESS_JSON {\"capacity\":0}\n");
        expect(!read_calibration_drift(tmp.root / "helper.log"),
               "incomplete/malformed measured drift cannot extend deadline");

        tmp.write("helper.log", "RHP16_CALIBRATION_JSON {\"measurements\":{}}\n");
        expect(!complete_calibration_marker(tmp.root / "helper.log"),
               "partial calibration marker cannot activate a production record");
        tmp.write("helper.log", "RHP16_CALIBRATION_JSON not-json\n");
        expect(!complete_calibration_marker(tmp.root / "helper.log"), "malformed settled marker must refuse");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RSC02 step 3 test failed: " << error.what() << '\n';
        return 1;
    }
}
