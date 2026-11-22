#include "calibration_watch.h"

#include <fstream>
#include <stdexcept>
#include <utility>

namespace bc::serving {
namespace {
constexpr const char* kMarker = "RHP16_CALIBRATION_JSON ";
} // namespace

CalibrationWatch::CalibrationWatch(std::filesystem::path helper_log, std::filesystem::path store,
                                   LivePolicyTarget target)
    : log_(std::move(helper_log)), store_(std::move(store)), target_(std::move(target)) {}

bool CalibrationWatch::poll() {
    if (state_ != State::waiting) return false;
    std::ifstream in(log_, std::ios::binary);
    if (!in) return false;   // the helper has not created its log yet
    in.seekg(0, std::ios::end);
    const auto size = uint64_t(in.tellg());
    if (size < offset_) {    // truncated or replaced: read it from the start again
        offset_ = 0;
        pending_.clear();
    }
    if (size == offset_) return false;
    in.seekg(std::streamoff(offset_));
    std::string chunk(size - offset_, '\0');
    in.read(chunk.data(), std::streamsize(chunk.size()));
    chunk.resize(size_t(in.gcount()));
    offset_ += chunk.size();
    pending_ += chunk;
    size_t from = 0;
    for (size_t nl; (nl = pending_.find('\n', from)) != std::string::npos; from = nl + 1) {
        if (pending_.compare(from, std::char_traits<char>::length(kMarker), kMarker) != 0) continue;
        const size_t begin = from + std::char_traits<char>::length(kMarker);
        return decide(pending_.substr(begin, nl - begin));
    }
    pending_.erase(0, from);   // complete non-marker lines are dropped; only the open tail is kept
    return false;
}

bool CalibrationWatch::decide(const std::string& payload) {
    try {
        ingest_live_policy_marker(payload, store_, target_.plan_key, target_.layout, target_.objective,
                                  target_.store_key, target_.provenance);
        state_ = State::stored;
        detail_ = "stored the helper's calibration record under " + target_.store_key;
    } catch (const std::exception& e) {
        // A record the store already holds (complete, or partial against a partial) is settled, not an error;
        // anything else, including an unwritable store, is only reported.
        const std::string what = e.what();
        const bool duplicate = what.find("refuse duplicate") != std::string::npos;
        state_ = duplicate ? State::settled_elsewhere : State::refused;
        detail_ = duplicate ? "the calibration store already holds this key: " + what
                            : "calibration record not stored: " + what;
    }
    pending_.clear();
    return true;
}

} // namespace bc::serving
