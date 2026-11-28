#include "runtime_guards.h"
#include "../moe-remote/heat_publication.h"
#include "experts.h"
#include "v2/runtime_receipt.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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

// The companion is optional for legacy engines and deliberately invisible to
// workers. A session owner, unlike a worker, requires an exact matching proof.
bool read_heat_proof(const fs::path& rank_file, r3::heat_publication::Proof* proof) {
    const std::string path = r3::heat_publication::proof_path(rank_file.string());
    const Descriptor fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (fd.get() < 0)
        return false;
    struct stat before{}, after{};
    if (::fstat(fd.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
        before.st_size != static_cast<off_t>(sizeof(*proof)))
        return false;
    size_t read = 0;
    while (read < sizeof(*proof)) {
        const ssize_t n = ::pread(fd.get(), reinterpret_cast<char*>(proof) + read, sizeof(*proof) - read,
                                  static_cast<off_t>(read));
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            return false;
        read += static_cast<size_t>(n);
    }
    return ::fstat(fd.get(), &after) == 0 && before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
           before.st_size == after.st_size && before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
           before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
           proof->magic == r3::heat_publication::kProofMagic &&
           proof->version == r3::heat_publication::kProofVersion && proof->record_size == sizeof(*proof);
}

HistogramVerdict histogram_verdict(const fs::path& rank_file, std::chrono::system_clock::time_point now,
                                   std::chrono::nanoseconds max_age, const HistogramExpectation& expected,
                                   HistogramIdentity* observed, std::string* reason) {
    using r3::moe_shm::kMaxHeatLdsBytes;
    using r3::moe_shm::kMaxLayers;
    const auto verdict = [&](HistogramVerdict v, const char* why) {
        if (reason)
            *reason = why;
        return v;
    };
    constexpr uint64_t nanoseconds_per_second = 1000000000ULL;
    if (expected.layers == 0 || expected.layers > kMaxLayers || expected.experts == 0 ||
        expected.experts > kMaxHeatLdsBytes / sizeof(uint32_t) ||
        !r3::heat_publication::valid_nonce(expected.launch_nonce) ||
        expected.launched_at.time_since_epoch().count() <= 0 || max_age.count() <= 0)
        return verdict(HistogramVerdict::unsafe, "the expected shape, launch nonce, launch time or age bound is invalid");
    const size_t cells = size_t(kMaxLayers) * static_cast<size_t>(expected.experts);
    const size_t required = cells * sizeof(uint32_t);
    if (expected.identity && expected.identity->counts.size() != cells)
        return verdict(HistogramVerdict::unsafe, "the previous sample has a different size");

    // The producer must rename the raw histogram, then rename its companion.
    // A poll in that small gap may see two different generations. Re-read
    // briefly rather than mistaking normal atomic publication for degradation.
    const auto grace_end = std::chrono::steady_clock::now() + expected.proof_grace;
    for (;;) {
        const Descriptor fd(::open(rank_file.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
        if (fd.get() < 0)
            return errno == ENOENT ? verdict(HistogramVerdict::missing, "the histogram file does not exist")
                                   : verdict(HistogramVerdict::unsafe, "the histogram file cannot be opened (a symlink or no access)");
        struct stat before{}, after{};
        if (::fstat(fd.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
            before.st_size != static_cast<off_t>(required) || before.st_mtim.tv_sec < 0 ||
            before.st_mtim.tv_nsec < 0 || before.st_mtim.tv_nsec >= 1000000000L ||
            static_cast<uint64_t>(before.st_mtim.tv_sec) >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / nanoseconds_per_second)
            return verdict(HistogramVerdict::unsafe, "the histogram file is not a regular file of the expected size");

        const int64_t modified = (static_cast<int64_t>(before.st_mtim.tv_sec) * 1000000000LL) +
                                 static_cast<int64_t>(before.st_mtim.tv_nsec);
        const auto launch_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(expected.launched_at.time_since_epoch())
                .count();
        const auto now_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        if (modified <= launch_ns || (expected.identity && modified < expected.identity->modified_ns))
            return verdict(HistogramVerdict::unsafe, modified <= launch_ns
                                                         ? "the histogram was last written before this engine launched"
                                                         : "the histogram is older than the previous sample");
        if (modified > now_ns)
            return verdict(HistogramVerdict::future, "the histogram is dated in the future");
        if (now_ns - modified > max_age.count())
            return verdict(HistogramVerdict::stale, "the histogram is older than the allowed heat age");

        std::vector<uint32_t> values(cells);
        size_t copied = 0;
        while (copied < required) {
            const ssize_t n = ::pread(fd.get(), reinterpret_cast<char*>(values.data()) + copied,
                                      required - copied, static_cast<off_t>(copied));
            if (n < 0 && errno == EINTR)
                continue;
            if (n <= 0)
                return verdict(HistogramVerdict::unsafe, "the histogram file could not be read in full");
            copied += static_cast<size_t>(n);
        }
        if (::fstat(fd.get(), &after) != 0 || before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
            before.st_size != after.st_size || before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
            before.st_mtim.tv_nsec != after.st_mtim.tv_nsec)
            return verdict(HistogramVerdict::unsafe, "the histogram file changed while it was read");

        r3::heat_publication::Proof proof{};
        const bool proof_read = read_heat_proof(rank_file, &proof);
        std::string mismatch;
        if (!proof_read) mismatch = "no readable proof file";
        else {
            const auto add = [&](const char* what) { mismatch += (mismatch.empty() ? "" : ", "); mismatch += what; };
            if (std::string_view(proof.launch_nonce.data(), proof.launch_nonce.size()) != expected.launch_nonce) add("launch nonce");
            if (proof.device != static_cast<uint64_t>(before.st_dev) || proof.inode != static_cast<uint64_t>(before.st_ino)) add("file identity");
            if (proof.size_bytes != static_cast<uint64_t>(required)) add("size");
            if (proof.modified_sec != static_cast<int64_t>(before.st_mtim.tv_sec) ||
                proof.modified_nsec != static_cast<int64_t>(before.st_mtim.tv_nsec)) add("modification time");
        }
        if (!mismatch.empty()) {
            if (std::chrono::steady_clock::now() < grace_end) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            return verdict(HistogramVerdict::unsafe,
                           ("the launch-bound proof beside the histogram does not match this launch and file (" + mismatch +
                            ") for longer than the publication grace")
                               .c_str());
        }

        bool demand = false;
        const size_t model_cells = static_cast<size_t>(expected.layers) * static_cast<size_t>(expected.experts);
        for (size_t i = 0; i < cells; ++i) {
            if (i >= model_cells) {
                if (values[i] != 0)
                    return verdict(HistogramVerdict::unsafe,
                                   "counts in a layer beyond the model's routed layers (padding must stay zero)"); // never accept counts in padded layers
            } else {
                demand = demand || values[i] != 0;
            }
            if (expected.identity && values[i] < expected.identity->counts[i])
                return verdict(HistogramVerdict::unsafe,
                               "a count went down since the previous sample (restart or wrap)"); // fail closed, never synthesize a delta
        }
        // Authentic and well-formed but nothing routed yet: no evidence, no fault (the caller decides how long to wait).
        if (!demand)
            return verdict(HistogramVerdict::pending, "the histogram is authentic but holds no route counts yet");
        if (observed)
            *observed = HistogramIdentity{static_cast<uint64_t>(before.st_dev),
                                          static_cast<uint64_t>(before.st_ino), modified, std::move(values)};
        return verdict(HistogramVerdict::fresh, "");
    }
    return verdict(HistogramVerdict::unsafe, "the histogram was never read consistently");
}

HistogramVerdict histogram_verdict_startup(const fs::path& rank_file, std::chrono::nanoseconds max_age,
                                           const HistogramExpectation& expected, std::chrono::nanoseconds wait,
                                           HistogramIdentity* observed, std::string* reason) {
    const auto give_up = std::chrono::steady_clock::now() + wait;
    auto verdict = histogram_verdict(rank_file, std::chrono::system_clock::now(), max_age, expected, observed, reason);
    while (verdict == HistogramVerdict::pending && std::chrono::steady_clock::now() < give_up) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        verdict = histogram_verdict(rank_file, std::chrono::system_clock::now(), max_age, expected, observed, reason);
    }
    return verdict;
}

bool start_is_profiled(const v2::Manifest& manifest, bool native_profile_ops) {
    return native_profile_ops || (manifest.spec.experiment && manifest.spec.experiment->engine &&
                                  manifest.spec.experiment->engine->profile_ops.has_value());
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
    auto loaded = plan.admitted_manifest();
    if (!loaded)
        return loaded.error();
    auto manifest = std::move(loaded.value());
    const bool profiled = start_is_profiled(manifest, plan.native_profile_ops);
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
        if (state.requested == v2::LivePolicyRequested::kOn && !state.configured)
            return refuse("policy", "live placement was requested but the startup decision disabled it: " +
                                        state.reason_code);
        policy_enabled_ = handoff.placement.live_policy.has_value();
        if (policy_enabled_ && !state.configured)
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
            heat_expectation_ = HistogramExpectation{
                model.layers, model.experts, plan.engine_launched_at, {}, plan.heat_session_nonce};
            if (model.layers == 0 || model.layers > r3::moe_shm::kMaxLayers || model.experts == 0 ||
                model.experts > r3::moe_shm::kMaxHeatLdsBytes / sizeof(uint32_t))
                return refuse("heat", "model dimensions exceed the engine histogram producer contract");
            if (!r3::heat_publication::valid_nonce(heat_expectation_.launch_nonce))
                return refuse("heat", "owner did not provide a session-bound heat publication nonce");
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
            // The startup guard runs right after the canary, before the engine's next histogram snapshot has to have
            // counted it: an authentic all-zero file is waited for up to the declared publication period, or the
            // policy's allowed heat age if that is longer (both from the manifest; neither declared: no wait).
            const int64_t declared_period_ns =
                handoff.telemetry.period_s > 0 &&
                        handoff.telemetry.period_s < std::numeric_limits<int64_t>::max() / 1000000000LL
                    ? handoff.telemetry.period_s * 1000000000LL
                    : 0;
            heat_startup_wait_ = std::chrono::nanoseconds(std::max(declared_period_ns, age_ns));
            // The raw file and its proof are two renames apart: a poll that lands between them re-reads for up to
            // one declared publication period (never less than the default) before it calls the pair unsafe.
            heat_expectation_.proof_grace = std::chrono::nanoseconds(std::max<int64_t>(declared_period_ns, 100000000LL));
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
    // Recording is observational: an unavailable or temporarily inconsistent
    // histogram must not shut down inference. Strict freshness, nonce, identity
    // and shape checks remain mandatory when live placement policy is active.
    if (!policy_enabled_ || !histogram_)
        return std::nullopt;
    HistogramIdentity identity{};
    std::string why;
    // Only the first check after startup waits, and only for an authentic file with no counts yet; every other
    // verdict, and every later check, is decided at once.
    const auto wait = heat_startup_wait_;
    heat_startup_wait_ = std::chrono::nanoseconds{0};
    const auto evidence = histogram_verdict_startup(*histogram_, heat_age_budget_, heat_expectation_, wait,
                                                    &identity, &why);
    if (evidence != HistogramVerdict::fresh)
        return refuse(
            "heat", "manifest-declared engine rank0 heat histogram is absent, unsafe, stale, future-dated or still empty: " +
                        std::to_string(static_cast<int>(evidence)) + " (" + why + ")");
    heat_expectation_.identity = std::move(identity);
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
        manifest_->metadata.content_digest, manifest_->spec.handoff->transport.shm, mapping.data(),
        static_cast<size_t>(st.st_size));
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

v2::Status RuntimeGuards::poll() {
    if (!manifest_)
        return refuse("poll", "runtime guard has not passed the startup gate");
    if (std::chrono::steady_clock::now() < next_poll_)
        return std::nullopt;
    if (auto error = inspect_heat())
        return error;
    if (auto error = inspect_policy())
        return error;
    next_poll_ = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    return std::nullopt;
}
} // namespace bc::serving::session
