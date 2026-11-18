#include "calibration_lab.h"
#include "v2/strict_json.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <sstream>
#include <string_view>
#include <utility>

namespace bc::serving::session {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
v2::ConfigError invalid(std::string_view stage, std::string detail) {
    return v2::make_error(v2::ConfigErrorCode::calibration_invalid, "session.calibration." + std::string(stage),
                          std::move(detail));
}
v2::Result<std::string> text_of(const fs::path& path) {
    std::ifstream input(path);
    if (!input)
        return v2::make_error(v2::ConfigErrorCode::io_failure, "session.calibration.log",
                              "cannot read required calibration input " + path.filename().string());
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}
bool safe_add(uint64_t lhs, uint64_t rhs, uint64_t* output) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
        return false;
    *output = lhs + rhs;
    return true;
}
bool safe_mul(uint64_t lhs, uint64_t rhs, uint64_t* output) {
    if (rhs && lhs > std::numeric_limits<uint64_t>::max() / rhs)
        return false;
    *output = lhs * rhs;
    return true;
}
uint64_t ceil_seconds(uint64_t ns) {
    return (ns / 1000000000ULL) + (ns % 1000000000ULL != 0 ? 1ULL : 0ULL);
}
v2::Result<uint64_t> positive(int64_t value, std::string_view name) {
    if (value <= 0)
        return invalid("policy", std::string(name) + " must be positive");
    return static_cast<uint64_t>(value);
}
v2::Result<Json> parse_marker(std::string_view text, std::string_view stage) {
    auto result = v2::parse_strict_json(text, "calibration " + std::string(stage));
    if (!result)
        return result.error();
    if (!result.value().is_object())
        return invalid(stage, "calibration marker must contain a JSON object");
    return result.value();
}
v2::Result<uint64_t> unsigned_field(const Json& row, const char* key) {
    if (!row.contains(key) || !row.at(key).is_number_integer())
        return invalid("drift", std::string(key) + " must be a nonnegative integer");
    const auto& value = row.at(key);
    if (value.is_number_unsigned()) {
        const auto number = value.get<uint64_t>();
        if (number < (uint64_t{1} << 63U))
            return number;
    } else if (value.get<int64_t>() >= 0) {
        return static_cast<uint64_t>(value.get<int64_t>());
    }
    return invalid("drift", std::string(key) + " exceeds the safe measured range");
}
std::vector<std::string> marker_payloads(const std::string& bytes, std::string_view prefix) {
    std::istringstream input(bytes);
    std::vector<std::string> out;
    for (std::string line; std::getline(input, line);) {
        if (line.starts_with(prefix))
            out.push_back(line.substr(prefix.size()));
    }
    return out;
}
} // namespace

v2::Result<CalibrationPrompts> read_calibration_prompts(const fs::path& a, const fs::path& b) {
    CalibrationPrompts prompts;
    std::string raw_a, raw_b;
    const auto read = [&](const fs::path& path, std::vector<std::string>* out, std::string* raw) -> v2::Status {
        auto file = text_of(path);
        if (!file)
            return file.error();
        *raw = std::move(file.value());
        std::istringstream input(*raw);
        for (std::string line; std::getline(input, line);) {
            if (std::any_of(line.begin(), line.end(), [](unsigned char ch) { return !std::isspace(ch); }))
                out->push_back(std::move(line));
        }
        if (out->empty())
            return invalid("prompts", path.filename().string() + ": a workload needs nonempty prompts");
        return std::nullopt;
    };
    if (auto error = read(a, &prompts.a, &raw_a))
        return *error;
    if (auto error = read(b, &prompts.b, &raw_b))
        return *error;
    if (raw_a == raw_b || prompts.a == prompts.b)
        return invalid("prompts", "A and B must be distinct raw inputs and distinct effective prompt batches");
    return prompts;
}

v2::Result<std::optional<DriftProgress>> read_calibration_drift(const fs::path& helper_log) {
    const auto file = text_of(helper_log);
    if (!file)
        return file.error();
    const auto entries = marker_payloads(file.value(), "RHP19_DRIFT_PROGRESS_JSON ");
    if (entries.empty())
        return std::optional<DriftProgress>{};
    auto decoded = parse_marker(entries.back(), "drift");
    if (!decoded)
        return decoded.error();
    const auto& row = decoded.value();
    DriftProgress progress;
    const auto assign = [&](const char* name, uint64_t* destination) -> v2::Status {
        auto value = unsigned_field(row, name);
        if (!value)
            return value.error();
        *destination = value.value();
        return std::nullopt;
    };
    if (auto error = assign("capacity", &progress.capacity))
        return *error;
    if (auto error = assign("intervals", &progress.intervals))
        return *error;
    if (auto error = assign("horizon_observations", &progress.horizon_observations))
        return *error;
    if (auto error = assign("observation_rate_per_s", &progress.observation_rate_per_s))
        return *error;
    if (auto error = assign("accepted_gap_ns", &progress.accepted_gap_ns))
        return *error;
    if (auto error = assign("accepted_heat_deltas", &progress.accepted_heat_deltas))
        return *error;
    if (progress.capacity == 0 || progress.intervals > progress.capacity)
        return invalid("drift", "invalid measured ring capacity or occupied intervals");
    return std::optional<DriftProgress>{progress};
}

v2::Result<std::chrono::seconds> calibration_heat_feed_deadline(const v2::LivePolicy& policy,
                                                               std::chrono::seconds histogram_period) {
    const auto cadence = positive(policy.cadence_ns, "cadence_ns");
    const auto age = positive(policy.max_heat_age_ns, "max_heat_age_ns");
    const auto samples = positive(policy.min_samples, "min_samples");
    if (!cadence)
        return cadence.error();
    if (!age)
        return age.error();
    if (!samples)
        return samples.error();
    if (histogram_period.count() <= 0)
        return invalid("heat", "calibration requires a positive histogram publication period");
    uint64_t periods = 0, period_ns = 0, sample_ns = 0, deadline_ns = 0;
    // One file snapshot establishes the baseline; subsequent snapshots
    // can contribute the admitted observation budget. Never equate the
    // min_samples counter with policy proposal-cadence intervals.
    if (!safe_add(samples.value(), 1, &periods) ||
        !safe_mul(static_cast<uint64_t>(histogram_period.count()), 1000000000ULL, &period_ns) ||
        !safe_mul(periods, period_ns, &sample_ns) ||
        !safe_add(sample_ns, std::max(cadence.value(), age.value()), &deadline_ns) ||
        ceil_seconds(deadline_ns) > static_cast<uint64_t>(std::chrono::seconds::max().count()))
        return invalid("heat", "derived histogram heat feed deadline overflows");
    return std::chrono::seconds(ceil_seconds(deadline_ns));
}

v2::Result<CalibrationEnvelope> calibration_envelope(const v2::LivePolicy& policy, size_t a_count,
                                                     size_t b_count, std::chrono::seconds request_timeout,
                                                     const std::optional<DriftProgress>& drift) {
    const auto cadence = positive(policy.cadence_ns, "cadence_ns");
    const auto age = positive(policy.max_heat_age_ns, "max_heat_age_ns");
    const auto residence = positive(policy.min_residency_ns, "min_residency_ns");
    const auto samples = positive(policy.min_samples, "min_samples");
    if (!cadence)
        return cadence.error();
    if (!age)
        return age.error();
    if (!residence)
        return residence.error();
    if (!samples)
        return samples.error();
    if (!a_count || !b_count || request_timeout.count() <= 0)
        return invalid("workload", "calibration requires two nonempty workloads and a finite request timeout");

    uint64_t phase_ns = 0, count_ns = 0, add_ns = 0, baseline_ns = 0;
    if (!safe_add(samples.value(), 1, &count_ns) || !safe_mul(count_ns, cadence.value(), &phase_ns) ||
        !safe_add(phase_ns, age.value(), &add_ns) || !safe_add(add_ns, residence.value(), &phase_ns) ||
        !safe_mul(4, phase_ns, &baseline_ns))
        return invalid("policy", "calibration horizon overflows");
    const uint64_t baseline = ceil_seconds(baseline_ns);
    const uint64_t report = ceil_seconds(std::max(cadence.value(), age.value()));
    const auto timeout = static_cast<uint64_t>(request_timeout.count());
    uint64_t count = 0, work = 0;
    if (a_count > std::numeric_limits<uint64_t>::max() || b_count > std::numeric_limits<uint64_t>::max() ||
        !safe_add(static_cast<uint64_t>(a_count), static_cast<uint64_t>(b_count), &count) ||
        !safe_mul(count, 2, &count) || !safe_mul(count, timeout, &work))
        return invalid("workload", "calibration workload duration overflows");

    uint64_t extension_s = 0;
    const bool feed = drift && drift->accepted_heat_deltas > 1 && drift->accepted_gap_ns;
    if (feed) {
        uint64_t drift_ns = 0;
        if (!safe_mul(drift->capacity, std::max({cadence.value(), age.value(), drift->accepted_gap_ns}),
                      &drift_ns))
            return invalid("drift", "measured drift horizon overflows");
        if (drift->observation_rate_per_s && drift->horizon_observations) {
            uint64_t observed_ns = 0;
            if (!safe_mul(drift->horizon_observations, 1000000000ULL, &observed_ns))
                return invalid("drift", "measured observation horizon overflows");
            const uint64_t q = observed_ns / drift->observation_rate_per_s;
            const uint64_t r = observed_ns % drift->observation_rate_per_s;
            uint64_t rounded = 0;
            if (!safe_add(q, r ? 1U : 0U, &rounded))
                return invalid("drift", "observation horizon overflows");
            drift_ns = std::max(drift_ns, rounded);
        }
        // Match the watchdog's four phase horizons without accumulating
        // independent ceil() rounding; add checked drift nanoseconds before
        // rounding the full bound to whole seconds.
        uint64_t total_drift_ns = 0, extended_ns = 0;
        if (!safe_mul(4, drift_ns, &total_drift_ns) || !safe_add(baseline_ns, total_drift_ns, &extended_ns))
            return invalid("drift", "four-phase measured drift horizon overflows");
        extension_s = ceil_seconds(extended_ns) - baseline;
    }
    uint64_t limit = 0;
    if (!safe_add(baseline, work, &limit) || !safe_add(limit, std::min(extension_s, work), &limit) ||
        limit > static_cast<uint64_t>(std::chrono::seconds::max().count()))
        return invalid("workload", "derived calibration deadline exceeds representable time");
    return CalibrationEnvelope{std::chrono::seconds(limit), std::chrono::seconds(baseline),
                               std::chrono::seconds(report), feed};
}

CalibrationEnvelope monotonic_calibration_envelope(const CalibrationEnvelope& admitted,
                                                   const CalibrationEnvelope& observed) {
    return CalibrationEnvelope{std::max(admitted.limit, observed.limit), admitted.prefeed,
                               observed.report_period, admitted.accepted_feed || observed.accepted_feed};
}

v2::Result<bool> complete_calibration_marker(const fs::path& helper_log) {
    const auto file = text_of(helper_log);
    if (!file)
        return file.error();
    const auto markers = marker_payloads(file.value(), "RHP16_CALIBRATION_JSON ");
    if (markers.empty())
        return false;
    if (markers.size() != 1)
        return invalid("record", "more than one settled calibration marker");
    auto decoded = parse_marker(markers.front(), "record");
    if (!decoded)
        return decoded.error();
    if (!decoded.value().contains("measurements") || !decoded.value().at("measurements").is_object() ||
        !decoded.value().contains("absent") || !decoded.value().at("absent").is_object())
        return invalid("record", "settled marker requires measurements and explicit absent reasons");
    const auto& measurements = decoded.value().at("measurements");
    const auto& absent = decoded.value().at("absent");
    static const std::set<std::string> required = {"txn_p50_ns",
                                                   "txn_p95_ns",
                                                   "copy_bytes_per_s",
                                                   "heat_gap_max_ns",
                                                   "heat_watermarked",
                                                   "hot_set_k",
                                                   "marginal_error_ppm",
                                                   "marginal_observations",
                                                   "drift_half_life_observations",
                                                   "observation_rate_per_s",
                                                   "tail_share_gain_ppm",
                                                   "tail_lane_service_ns_per_s",
                                                   "tail_lane_row_ns",
                                                   "lane_lost_ns",
                                                   "effect_rows_per_s",
                                                   "effect_noise_max_ns",
                                                   "effect_row_noise_ns",
                                                   "effect_window_noise_ladder",
                                                   "abort_quiescent_ns"};
    for (const auto& key : required) {
        if (measurements.contains(key) == absent.contains(key))
            return invalid("record", "partial or contradictory calibration marker: " + key);
    }
    return true;
}
} // namespace bc::serving::session
