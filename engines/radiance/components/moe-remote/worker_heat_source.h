#pragma once
// Stage-1 heat-source contract. All values are model-global (layer, expert)
// rather than runtime slot IDs or helper-local expert IDs. Poll is a CPU-only
// operation for a background controller, NEVER for a request or lane thread.
#include "moe_remote_shm.h"
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace r3::policy {
enum class HeatProvenance : uint8_t { UNKNOWN, ENGINE_HISTOGRAM_FILE };
struct HeatDelta {
    HeatProvenance provenance = HeatProvenance::UNKNOWN;
    uint32_t layers = 0, experts = 0;
    std::vector<int32_t> counts;       // interval deltas; row-major model-global IDs
    uint64_t observation_count = 0;   // cumulative routed selections (not engine steps)
    uint64_t interval_observations = 0;
    uint64_t age_ns = 0;              // age of the source file at read time
    bool complete_cells = false;     // exact byte shape, no counter regression
    bool complete_observations = false; // step/bank provenance proved by producer
};
enum class HeatPoll : uint8_t { BASELINE, UNCHANGED, DELTA, INVALID };
class HeatSource {
public:
    virtual ~HeatSource() = default;
    // Both timestamps MUST be sampled by the caller at poll time:
    // steady clock for residency, realtime for filesystem mtimes.
    virtual HeatPoll poll(uint64_t monotonic_now_ns, uint64_t realtime_now_ns,
                          HeatDelta* out, std::string* error) = 0;
};

// Interim engine-side diagnostic source: R3_MOE_WORKER_HISTOGRAM is written
// atomically as raw native-endian uint32_t[kMaxLayers][expert_count] to
// <prefix>.rank<rank>. Neither step IDs nor a complete observation watermark
// are in this format. Never claim complete_observations or infer steps from
// sum(counts). RHP02 replaces this reader with the authoritative engine feed.
class HistogramFileSource final : public HeatSource {
    std::string path_;
    uint32_t layers_, experts_;
    std::vector<uint32_t> previous_;
    uint64_t total_ = 0;
    bool baseline_ = false;
    struct FileStamp {
        uint64_t mtime_ns = 0;
        ino_t inode = 0;
        dev_t device = 0;
    } stamp_{};
    static bool timestamp(const struct stat& st, uint64_t* out) {
        if (st.st_mtim.tv_sec < 0 || st.st_mtim.tv_nsec < 0 ||
            st.st_mtim.tv_nsec >= 1000000000L ||
            uint64_t(st.st_mtim.tv_sec) > (UINT64_MAX - uint64_t(st.st_mtim.tv_nsec)) / 1000000000ULL)
            return false;
        *out = uint64_t(st.st_mtim.tv_sec) * 1000000000ULL + uint64_t(st.st_mtim.tv_nsec);
        return true;
    }
public:
    HistogramFileSource(std::string path, uint32_t layers, uint32_t experts)
        : path_(std::move(path)), layers_(layers), experts_(experts) {}
    HeatPoll poll(uint64_t mono, uint64_t realtime, HeatDelta* out,
                  std::string* error) override {
        auto invalid = [&](const char* why) {
            if (error) *error = why;
            return HeatPoll::INVALID;
        };
        if (!out || !mono || !realtime || !layers_ ||
            layers_ > moe_shm::kMaxLayers || !experts_ ||
            experts_ > moe_shm::kExperts || path_.empty())
            return invalid("histogram: invalid domain, timestamp or output");
        const size_t count = size_t(moe_shm::kMaxLayers) * experts_;
        const size_t bytes = count * sizeof(uint32_t);
        const int fd = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) return invalid("histogram: cannot open rank file");
        struct stat st{};
        const bool shape = ::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) &&
                           st.st_size >= 0 && uint64_t(st.st_size) == bytes;
        uint64_t mtime = 0;
        if (!shape || !timestamp(st, &mtime) || mtime > realtime) {
            ::close(fd);
            return invalid("histogram: invalid file size or timestamp");
        }
        std::vector<uint32_t> values(count);
        size_t copied = 0;
        while (copied < bytes) {
            const ssize_t n = ::read(fd, reinterpret_cast<char*>(values.data()) + copied, bytes - copied);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            copied += size_t(n);
        }
        struct stat after{};
        const bool stable = copied == bytes && ::fstat(fd, &after) == 0 &&
                            after.st_size == st.st_size && after.st_ino == st.st_ino &&
                            after.st_dev == st.st_dev && after.st_mtim.tv_sec == st.st_mtim.tv_sec &&
                            after.st_mtim.tv_nsec == st.st_mtim.tv_nsec;
        ::close(fd);
        if (!stable) return invalid("histogram: changed or truncated during read");
        const uint64_t age = realtime - mtime;
        *out = {};
        out->layers = layers_; out->experts = experts_; out->age_ns = age;
        out->provenance = HeatProvenance::ENGINE_HISTOGRAM_FILE;
        out->observation_count = total_;
        out->counts.assign(size_t(layers_) * experts_, 0);
        out->complete_cells = true;
        // Files are atomically renamed, but counts are sampled from a live
        // GPU-updated array without a step watermark. No per-step coherence.
        out->complete_observations = false;
        const FileStamp incoming{mtime, st.st_ino, st.st_dev};
        if (!baseline_) {
            previous_ = std::move(values);
            stamp_ = incoming;
            baseline_ = true;
            return HeatPoll::BASELINE;
        }
        if (stamp_.inode == incoming.inode && stamp_.device == incoming.device &&
            stamp_.mtime_ns == incoming.mtime_ns)
            return HeatPoll::UNCHANGED;
        uint64_t interval = 0;
        for (size_t i = 0; i < count; ++i) {
            if (values[i] < previous_[i]) {
                // Restart or uint32 wrap: do not synthesize a huge positive
                // delta. A new baseline needs a separate fresh window.
                baseline_ = false;
                previous_.clear();
                return invalid("histogram: counter regression (restart or wrap)");
            }
            const uint32_t d = values[i] - previous_[i];
            if (i / experts_ >= layers_ && d)
                return invalid("histogram: counter outside model layer domain");
            if (d > uint32_t(INT32_MAX) || interval > UINT64_MAX - d ||
                total_ > UINT64_MAX - (interval + uint64_t(d)))
                return invalid("histogram: delta or cumulative observation overflow");
            interval += d;
            if (i / experts_ < layers_)
                out->counts[(i / experts_) * experts_ + i % experts_] = int32_t(d);
        }
        previous_ = std::move(values);
        stamp_ = incoming;
        total_ += interval;
        out->observation_count = total_;
        out->interval_observations = interval;
        return interval ? HeatPoll::DELTA : HeatPoll::UNCHANGED;
    }
};
} // namespace r3::policy
