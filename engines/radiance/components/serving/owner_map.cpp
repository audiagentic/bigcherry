#include "owner_map.h"

#include <cmath>
#include <map>

namespace bc::serving {
namespace {
using C = v1::ConfigErrorCode;

v1::Status mismatch(const std::string& path, const std::string& message) {
    return v1::make_error(C::ownership_mismatch, path, message);
}

std::string join(const std::vector<std::string>& parts) {
    std::string s;
    for (size_t i = 0; i < parts.size(); ++i) s += (i ? "," : "") + parts[i];
    return s;
}
}  // namespace

std::vector<bc::experts::OwnerPolicy> owner_policies(const v1::Manifest& m) {
    const auto& helpers = m.spec.processes.helpers;
    std::vector<std::string> shares;
    for (const v1::HelperProcess& h : helpers) shares.push_back(std::to_string(h.share));
    std::vector<bc::experts::OwnerPolicy> out;
    for (size_t i = 0; i < helpers.size(); ++i) {
        const v1::HelperProcess& h = helpers[i];
        bc::experts::OwnerPolicy p;
        p.heat = m.spec.handoff.placement.heat_file;
        p.placement = std::string(v1::kPlacementStrategyNames[static_cast<size_t>(m.spec.handoff.placement.strategy)]);
        std::vector<std::string> formats;
        for (v1::ExpertFormat f : h.formats) formats.push_back(std::string(v1::kExpertFormatNames[static_cast<size_t>(f)]));
        p.formats = join(formats);
        p.shares = join(shares);
        const v1::Result<v1::HelperView> view = v1::helper_view(m, h.role);
        p.balance = view.ok() && view.value().settings.placement.balance;
        p.balance_speeds = {1.0, 1.0};
        if (p.balance && view.value().settings.placement.balance_speeds_milli) {
            const auto& speeds = *view.value().settings.placement.balance_speeds_milli;
            p.balance_speeds.clear();
            for (int64_t s : speeds) p.balance_speeds.push_back(static_cast<double>(s) / 1000.0);
        }
        out.push_back(std::move(p));
    }
    return out;
}

std::vector<std::string> helper_archs(const v1::Manifest& m) {
    std::map<std::string, std::string> arch;
    for (const v1::GpuFact& g : m.spec.facts.gpus) arch[g.bdf] = g.arch;
    std::vector<std::string> out;
    for (const v1::HelperProcess& h : m.spec.processes.helpers) out.push_back(arch.count(h.gpu) ? arch.at(h.gpu) : std::string());
    return out;
}

std::vector<uint64_t> helper_budgets(const v1::Manifest& m) {
    std::vector<uint64_t> out;
    for (const v1::HelperProcess& h : m.spec.processes.helpers) out.push_back(static_cast<uint64_t>(h.vram_budget_mib) << 20);
    return out;
}

v1::Status verify_owner_map(const v1::Manifest& m, const bc::experts::Container& model) {
    bc::experts::OwnerMap planned;
    std::string why;
    if (!bc::experts::plan_owners(model, owner_policies(m), helper_archs(m), helper_budgets(m), &planned, &why))
        return mismatch("spec.plan.owner_map", "the declared placement cannot be planned for this model: " + why);
    const auto& got = m.spec.plan.owner_map.helpers;
    if (planned.helpers.size() != got.size())
        return mismatch("spec.plan.owner_map.helpers", std::to_string(got.size()) + " owned set(s) in the manifest, " +
                                                          std::to_string(planned.helpers.size()) + " planned");
    for (size_t i = 0; i < got.size(); ++i) {
        const std::string at = "spec.plan.owner_map.helpers[" + std::to_string(i) + "]";
        const bc::experts::Owned& want = planned.helpers[i];
        if (want.expert.size() != got[i].experts.size())
            return mismatch(at + ".experts", "the manifest owns " + std::to_string(got[i].experts.size()) + " expert(s), the planner selects " +
                                                 std::to_string(want.expert.size()));
        for (size_t k = 0; k < want.expert.size(); ++k)
            if (static_cast<int64_t>(want.expert[k].first) != got[i].experts[k][0] ||
                static_cast<int64_t>(want.expert[k].second) != got[i].experts[k][1])
                return mismatch(at + ".experts[" + std::to_string(k) + "]",
                                "the manifest has [" + std::to_string(got[i].experts[k][0]) + ", " + std::to_string(got[i].experts[k][1]) +
                                    "], the planner selects [" + std::to_string(want.expert[k].first) + ", " +
                                    std::to_string(want.expert[k].second) + "] at this position");
        if (static_cast<int64_t>(want.bytes) != got[i].bytes)
            return mismatch(at + ".bytes", "the manifest says " + std::to_string(got[i].bytes) + " device bytes, the experts take " +
                                               std::to_string(want.bytes));
    }
    return std::nullopt;
}

}  // namespace bc::serving
