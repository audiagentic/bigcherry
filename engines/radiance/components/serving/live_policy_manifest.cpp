#include "live_policy.h"
#include "manifest_writer.h"
#include "experts.h"
#include "plan_cache.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace bc::serving {
std::optional<DerivedLivePolicy> derive_manifest_live_policy(
    const ManifestInputs& in, const bc::experts::Container& container) {
    if (!in.measured || in.calibration_cache.empty()) return std::nullopt;
    const std::string key = plan_key(in.measured->key_inputs);
    const auto record = read_live_policy_record(in.calibration_cache, key);
    if (!record) return std::nullopt;

    // The GPU backend allocates fixed-size slots. Select the largest *device
    // layout* expert among the container's supported expert descriptors, not
    // their compressed file bytes, and reject a layout with no usable experts.
    uint64_t stride = 0;
    for (uint32_t layer = 0; layer < container.layers; ++layer)
        for (uint32_t expert = 0; expert < container.experts; ++expert) {
            if (!container.weight(layer, expert, 0) || !container.weight(layer, expert, 1))
                continue;
            const int64_t bytes = bc::experts::laid_out_bytes(container, true, layer, expert);
            if (bytes <= 0) throw std::runtime_error("live policy: cannot derive a device-layout expert stride");
            stride = std::max(stride, uint64_t(bytes));
        }
    if (!stride) throw std::runtime_error("live policy: no valid experts in the container");
    std::vector<uint64_t> budgets;
    for (const auto& h : in.resolved.manifest.at("process_helpers")) {
        const auto mib = h.at("budget_mib").get<uint64_t>();
        if (!mib || mib > std::numeric_limits<uint64_t>::max() / 1048576ull)
            throw std::runtime_error("live policy: invalid helper budget");
        budgets.push_back(mib * 1048576ull);
    }
    const CalibrationMatch match{"", in.resolved.values.at("layout").at("name").get<std::string>(),
                                  in.resolved.values.at("service").at("objective").get<std::string>()};
    return derive_live_policy(*record, Calibration::standard(in.calibration_cache), match, stride, budgets);
}
} // namespace bc::serving
