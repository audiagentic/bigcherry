// RSC02 step 3: native bounded ABBA lab and replay guard semantics. No GPU.
#include "calibration_lab.h"
#include "runtime_guards.h"

#include <filesystem>
#include <fstream>
#include <iostream>
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
        const auto now = std::chrono::system_clock::now();
        const auto rank = tmp.root / "test.rank0";
        tmp.write("test.rank0", std::string(16, '\0'));
        expect(histogram_verdict(rank, now, std::chrono::seconds(30)) == HistogramVerdict::fresh,
               "fresh rank histogram is valid bounded heat evidence");
        expect(histogram_verdict(rank, now + std::chrono::minutes(5), std::chrono::seconds(30)) ==
                   HistogramVerdict::stale,
               "an unchanged heat file cannot remain live forever");
        expect(histogram_verdict(rank, now - std::chrono::minutes(5), std::chrono::seconds(30)) ==
                   HistogramVerdict::future,
               "future-dated heat cannot be reported fresh");
        expect(histogram_verdict(tmp.root / "missing", now, std::chrono::seconds(30)) ==
                   HistogramVerdict::missing,
               "an absent engine heat stream must not be fabricated");
        tmp.write("empty.rank0", "");
        expect(histogram_verdict(tmp.root / "empty.rank0", now, std::chrono::seconds(30)) ==
                   HistogramVerdict::unsafe,
               "an empty stream is not an observation");
        fs::create_symlink(rank, tmp.root / "symlink.rank0");
        expect(histogram_verdict(tmp.root / "symlink.rank0", now, std::chrono::seconds(30)) ==
                   HistogramVerdict::unsafe,
               "rank histogram must not be redirected through a symlink");
        tmp.write("a.txt", "\n A prompt \n \t \n");
        tmp.write("b.txt", "B prompt\n");
        auto prompts = read_calibration_prompts(tmp.root / "a.txt", tmp.root / "b.txt");
        expect(bool(prompts) && prompts.value().a.size() == 1 && prompts.value().b.size() == 1,
               "ABBA needs the actual nonempty prompts, skipping whitespace");
        tmp.write("b.txt", "   \n");
        expect(!read_calibration_prompts(tmp.root / "a.txt", tmp.root / "b.txt"),
               "blank B workload must refuse, not run indefinitely");

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
        expect(measured.value().limit == std::chrono::seconds(110),
               "drift retention extends the baseline, not beyond the fixed workload ceiling");
        drift.capacity = 100;
        auto capped = calibration_envelope(policy, 1, 1, std::chrono::seconds(10), drift);
        expect(bool(capped) && capped.value().limit == std::chrono::seconds(120),
               "coarsening ring cannot slide deadline beyond the fixed two-workload cap");
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
