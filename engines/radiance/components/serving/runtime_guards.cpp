#include "runtime_guards.h"
#include "experts.h"
#include "v2/runtime_receipt.h"

#include <array>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace bc::serving::session {
namespace {
namespace fs = std::filesystem;
v2::ConfigError refuse(std::string_view stage, std::string reason) {
    return v2::make_error(v2::ConfigErrorCode::admission_refused, "session.runtime." + std::string(stage),
                          std::move(reason));
}
v2::Result<std::string> read_engine_log(const fs::path& path) {
    std::ifstream file(path);
    if (!file)
        return v2::make_error(v2::ConfigErrorCode::io_failure, "session.runtime.log",
                              "cannot read the engine log for this launch");
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}
bool draft_replayed(std::string_view log) {
    constexpr std::string_view marker = "recorded pass: draft pass ";
    size_t from = 0;
    while (true) {
        const size_t pos = log.find(marker, from);
        if (pos == std::string_view::npos)
            break;
        const size_t start = pos + marker.size();
        const size_t end = log.find(' ', start);
        if (end == std::string_view::npos)
            break;
        int64_t pass = 0;
        const auto parsed = std::from_chars(log.data() + start, log.data() + end, pass);
        if (parsed.ec == std::errc{} && parsed.ptr == log.data() + end && pass != 0)
            return true;
        from = end;
    }
    return false;
}
class Descriptor final {
  public:
    explicit Descriptor(int fd) : fd_(fd) {}
    ~Descriptor() {
        if (fd_ >= 0)
            ::close(fd_);
    }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    int get() const {
        return fd_;
    }

  private:
    int fd_;
};
class ReadMapping final {
  public:
    ReadMapping(int fd, size_t length) : length_(length) {
        data_ = ::mmap(nullptr, length_, PROT_READ, MAP_SHARED, fd, 0);
    }
    ~ReadMapping() {
        if (data_ != MAP_FAILED)
            ::munmap(data_, length_);
    }
    ReadMapping(const ReadMapping&) = delete;
    ReadMapping& operator=(const ReadMapping&) = delete;
    bool valid() const noexcept {
        return data_ != MAP_FAILED;
    }
    const void* data() const {
        return data_;
    }

  private:
    void* data_ = MAP_FAILED;
    size_t length_;
};
} // namespace

HistogramVerdict histogram_verdict(const fs::path& rank_file, std::chrono::system_clock::time_point now,
                                   std::chrono::nanoseconds max_age, const HistogramExpectation& expected,
                                   HistogramIdentity* observed) {
    // Do not trust the file's own length as its shape: that makes partial
    // histograms and a prior run's unrelated counters look like observations.
    constexpr uint64_t max_bytes = 64ULL * 1024 * 1024;
    if (expected.layers == 0 || expected.experts == 0 ||
        expected.layers > max_bytes / sizeof(uint32_t) / expected.experts ||
        expected.launched_at.time_since_epoch().count() <= 0 || max_age.count() <= 0)
        return HistogramVerdict::unsafe;
    const uint64_t required = expected.layers * expected.experts * sizeof(uint32_t);
    const Descriptor fd(::open(rank_file.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (fd.get() < 0)
        return errno == ENOENT ? HistogramVerdict::missing : HistogramVerdict::unsafe;
    struct stat before{}, after{};
    if (::fstat(fd.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
        before.st_size != static_cast<off_t>(required) || before.st_mtim.tv_sec < 0 ||
        before.st_mtim.tv_nsec < 0 || before.st_mtim.tv_nsec >= 1000000000L ||
        before.st_mtim.tv_sec > std::numeric_limits<int64_t>::max() / 1000000000LL)
        return HistogramVerdict::unsafe;
    const HistogramIdentity identity{static_cast<uint64_t>(before.st_dev),
                                     static_cast<uint64_t>(before.st_ino)};
    if (expected.identity &&
        (expected.identity->inode != identity.inode || expected.identity->device != identity.device))
        return HistogramVerdict::unsafe;
    const int64_t modified = (static_cast<int64_t>(before.st_mtim.tv_sec) * 1000000000LL) +
                             static_cast<int64_t>(before.st_mtim.tv_nsec);
    const auto launch_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(expected.launched_at.time_since_epoch()).count();
    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    if (modified <= launch_ns)
        return HistogramVerdict::unsafe; // previous run, even if it is otherwise still fresh
    if (modified > now_ns)
        return HistogramVerdict::future;
    if (now_ns - modified > max_age.count())
        return HistogramVerdict::stale;

    // Read the *whole* fixed-size producer record before declaring evidence.
    // Check metadata after reading, so truncated/in-flight writes fail closed.
    std::array<uint32_t, 1024> counts{};
    uint64_t bytes_read = 0;
    bool observed_demand = false;
    while (bytes_read < required) {
        const size_t requested = static_cast<size_t>(std::min<uint64_t>(required - bytes_read, sizeof(counts)));
        const ssize_t n = ::pread(fd.get(), counts.data(), requested, static_cast<off_t>(bytes_read));
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0 || static_cast<size_t>(n) % sizeof(uint32_t) != 0)
            return HistogramVerdict::unsafe;
        for (size_t i = 0; i < static_cast<size_t>(n) / sizeof(uint32_t); ++i)
            observed_demand = observed_demand || counts[i] != 0;
        bytes_read += static_cast<uint64_t>(n);
    }
    if (!observed_demand || ::fstat(fd.get(), &after) != 0 || before.st_ino != after.st_ino ||
        before.st_dev != after.st_dev || before.st_size != after.st_size ||
        before.st_mtim.tv_sec != after.st_mtim.tv_sec || before.st_mtim.tv_nsec != after.st_mtim.tv_nsec)
        return HistogramVerdict::unsafe;
    if (observed)
        *observed = identity;
    return HistogramVerdict::fresh;
}

ReplayVerdict replay_verdict(int64_t manifest_depth, bool profiled, std::string_view log) {
    if (manifest_depth <= 1)
        return ReplayVerdict::disabled;
    if (profiled)
        return ReplayVerdict::profiled;
    return draft_replayed(log) ? ReplayVerdict::observed : ReplayVerdict::absent;
}

v2::Status RuntimeGuards::startup(const EngineStart& plan, const fs::path& engine_log, const Retry& retry) {
    if (manifest_)
        return refuse("startup", "runtime guard was already bound");
    auto loaded = v2::load_manifest(plan.manifest, v2::PlanUse::serving);
    if (!loaded)
        return loaded.error();
    auto manifest = std::move(loaded.value());
    const bool profiled = manifest.spec.experiment && manifest.spec.experiment->engine &&
                          manifest.spec.experiment->engine->profile_ops.has_value();
    auto log = read_engine_log(engine_log);
    if (!log)
        return log.error();
    auto verdict =
        replay_verdict(manifest.spec.processes.engine.execution.num_speculative_tokens, profiled, log.value());
    if (verdict == ReplayVerdict::absent) {
        if (!retry)
            return refuse("mtp", "nonzero MTP draft replay missing and retry is unavailable");
        const auto retried = retry.invoke("session.runtime.mtp.retry");
        if (!retried)
            return retried.error();
        if (retried.value())
            return retried.value();
        log = read_engine_log(engine_log);
        if (!log)
            return log.error();
        verdict = replay_verdict(manifest.spec.processes.engine.execution.num_speculative_tokens, profiled,
                                 log.value());
        if (verdict == ReplayVerdict::absent)
            return refuse("mtp",
                          "MTP depth > 1 but no nonzero draft pass was replayed after the second canary");
    }
    if (manifest.spec.handoff) {
        const auto& handoff = *manifest.spec.handoff;
        const auto& state = handoff.placement.live_policy_state;
        if (state.requested == v2::LivePolicyRequested::kOn && !state.configured &&
            !(lab_pending_ && manifest.spec.intent.service.profile == v2::Profile::kLab))
            return refuse("policy", "live placement was requested but the startup decision disabled it: " +
                                        state.reason_code);
        policy_enabled_ = handoff.placement.live_policy.has_value();
        if (policy_enabled_ && !state.configured && !lab_pending_)
            return refuse("policy", "configured live-policy settings lack a qualified startup decision");
        if (handoff.telemetry.histogram_file) {
            if (handoff.telemetry.histogram_file->empty())
                return refuse("heat", "declared histogram file is empty");
            histogram_ = fs::path(*handoff.telemetry.histogram_file + ".rank0");
            // The real engine records raw counts indexed by the authenticated
            // model's routed layer and expert dimensions, not a generic blob.
            bc::experts::Container model;
            if (!model.open(manifest.spec.intent.model.c_str()))
                return refuse("heat", "cannot derive histogram shape from authenticated model container");
            heat_expectation_ = HistogramExpectation{model.layers, model.experts, plan.engine_launched_at, {}};
            if (heat_expectation_.launched_at.time_since_epoch().count() <= 0)
                return refuse("heat", "owner did not bind heat publication to this engine launch");
            // Startup heat age is bounded by the policy and the engine's
            // publication cadence; a temporarily idle engine gets a short
            // scheduling margin without turning an old file into evidence.
            const int64_t age_ns = state.max_heat_age_ns > 0 ? state.max_heat_age_ns : 0;
            const int64_t period_ns =
                handoff.telemetry.period_s > 0 &&
                        handoff.telemetry.period_s < std::numeric_limits<int64_t>::max() / 3000000000LL
                    ? handoff.telemetry.period_s * 3000000000LL
                    : 0;
            heat_age_budget_ = std::chrono::nanoseconds(std::max({age_ns, period_ns, int64_t(5000000000LL)}));
        }
        if (policy_enabled_ && !histogram_)
            return refuse("policy", "live placement needs a recorded engine heat stream");
        if (plan.handoff_shm != handoff.transport.shm)
            return refuse("policy", "owner's live handoff segment differs from the authenticated manifest");
        segment_ = handoff.transport.shm;
    }
    engine_log_ = engine_log;
    manifest_ = std::move(manifest);
    next_poll_ = std::chrono::steady_clock::now();
    return poll();
}

v2::Status RuntimeGuards::inspect_heat() {
    if (!histogram_)
        return std::nullopt;
    HistogramIdentity identity{};
    const auto evidence = histogram_verdict(*histogram_, std::chrono::system_clock::now(), heat_age_budget_,
                                            heat_expectation_, &identity);
    if (evidence != HistogramVerdict::fresh)
        return refuse(
            "heat", "manifest-declared engine rank0 heat histogram is absent, unsafe, stale or future-dated: " +
                        std::to_string(static_cast<int>(evidence)));
    if (!heat_expectation_.identity)
        heat_expectation_.identity = identity;
    return std::nullopt;
}

v2::Status RuntimeGuards::inspect_policy() {
    if (!policy_enabled_)
        return std::nullopt;
    if (!manifest_ || !manifest_->spec.handoff)
        return refuse("policy", "live policy has no authenticated manifest");
    if (segment_.size() < 2U || segment_.front() != '/' || segment_.find('/', 1) != std::string::npos)
        return refuse("policy", "live segment name is not a single POSIX object");
    const Descriptor fd(::shm_open(segment_.c_str(), O_RDONLY | O_CLOEXEC, 0));
    if (fd.get() < 0)
        return refuse("policy", "worker-owned live placement transport is unavailable");
    struct stat st{};
    if (::fstat(fd.get(), &st) != 0 || st.st_size <= 0 ||
        static_cast<uint64_t>(st.st_size) > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        return refuse("policy", "worker-owned live placement transport size is invalid");
    SegmentIdentity now{static_cast<uint64_t>(st.st_dev), static_cast<uint64_t>(st.st_ino)};
    if (segment_identity_ && (segment_identity_->device != now.device || segment_identity_->inode != now.inode))
        return refuse("policy", "worker transport object changed after its launch binding");
    ReadMapping mapping(fd.get(), static_cast<size_t>(st.st_size));
    if (!mapping.valid())
        return refuse("policy", "cannot map live worker telemetry read-only");
    const auto report = v2::publish_live_runtime_receipt(
        manifest_->metadata.instance_id, manifest_->spec.handoff->placement.live_policy_state,
        manifest_->metadata.content_digest, mapping.data(), static_cast<size_t>(st.st_size));
    if (!report)
        return report.error();
    segment_identity_ = now;
    const auto status = report.value().at("runtime_status").get<std::string>();
    if (status == "degraded" || status == "placement_stopped" || status == "missing_evidence")
        return refuse("policy", "live worker placement is " + status);
    if (status != "active" && status != "transitioning")
        return refuse("policy", "unknown live worker placement verdict");
    return std::nullopt;
}

v2::Status RuntimeGuards::prove_lab() {
    if (!lab_pending_)
        return refuse("lab", "lab evidence gate called without an active calibration");
    lab_pending_ = false;
    next_poll_ = std::chrono::steady_clock::now();
    return poll();
}

v2::Status RuntimeGuards::poll() {
    if (!manifest_)
        return refuse("poll", "runtime guard has not passed the startup gate");
    if (lab_pending_ || std::chrono::steady_clock::now() < next_poll_)
        return std::nullopt;
    if (auto error = inspect_heat())
        return error;
    if (auto error = inspect_policy())
        return error;
    next_poll_ = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    return std::nullopt;
}
} // namespace bc::serving::session
