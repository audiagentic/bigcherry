#include "owner_map.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>

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
        p.heat = m.spec.handoff->placement.heat_file;
        p.placement = std::string(v1::kPlacementStrategyNames[static_cast<size_t>(m.spec.handoff->placement.strategy)]);
        std::vector<std::string> formats;
        for (v1::ExpertFormat f : h.formats) formats.push_back(std::string(v1::kExpertFormatNames[static_cast<size_t>(f)]));
        p.formats = join(formats);
        p.shares = join(shares);
        const v1::HelperCommon settings = v1::effective_helper_settings(m, i);
        p.balance = settings.placement.balance;
        p.balance_speeds = {1.0, 1.0};
        if (p.balance && settings.placement.balance_speeds_milli) {
            const auto& speeds = *settings.placement.balance_speeds_milli;
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
    // never wraps: a budget outside what 64 bits of bytes can count (negative or too large) is clamped to the range (verify_owner_map refuses it before it gets here)
    for (const v1::HelperProcess& h : m.spec.processes.helpers)
        out.push_back(static_cast<uint64_t>(std::clamp<int64_t>(h.vram_budget_mib, 0, static_cast<int64_t>(UINT64_MAX >> 20))) << 20);
    return out;
}

namespace {
v1::Status invalid(const std::string& path, const std::string& message) { return v1::make_error(C::invalid_value, path, message); }

/* What the verifier needs of the manifest before it plans anything: it is a public entry point for producers, so it does
 * not rely on the reader having run. Every refusal names the field that is wrong. */
v1::Status preconditions(const v1::Manifest& m) {
    const auto& helpers = m.spec.processes.helpers;
    const auto& got = m.spec.plan.owner_map.helpers;
    if (got.size() != helpers.size())
        return mismatch("spec.plan.owner_map.helpers", std::to_string(got.size()) + " owned set(s) in the manifest, " +
                                                          std::to_string(helpers.size()) + " helper(s)");
    std::map<std::string, std::string> arch;
    for (const v1::GpuFact& g : m.spec.facts.gpus) arch[g.bdf] = g.arch;
    const bool tail = m.spec.handoff->placement.strategy == v1::PlacementStrategy::kTail;
    if (tail && helpers.size() < 2)
        return invalid("spec.handoff.placement.strategy", "the tail strategy needs at least two helpers, not " + std::to_string(helpers.size()));
    std::set<std::string> used(m.spec.processes.engine.device.gpus.begin(), m.spec.processes.engine.device.gpus.end());
    uint64_t total_share = 0;
    bool any_balance = false;
    const bool first_balance = v1::effective_helper_settings(m, 0).placement.balance;
    for (size_t i = 0; i < helpers.size(); ++i) {
        const v1::HelperProcess& h = helpers[i];
        const std::string at = "spec.processes.helpers[" + std::to_string(i) + "]";
        if (!arch.count(h.gpu))
            return v1::make_error(C::unresolved_reference, at + ".gpu", "GPU " + h.gpu + " is not among spec.facts.gpus");
        if (!used.insert(h.gpu).second)
            return v1::make_error(C::duplicate_field, at + ".gpu", "GPU " + h.gpu + " is already used by the engine or an earlier helper");
        if (h.vram_budget_mib <= 0)
            return invalid(at + ".vram_budget_mib", std::to_string(h.vram_budget_mib) + " MiB is not a budget");
        if (h.vram_budget_mib > static_cast<int64_t>(UINT64_MAX >> 20))
            return invalid(at + ".vram_budget_mib", std::to_string(h.vram_budget_mib) + " MiB does not fit 64 bits of bytes");
        if (h.share < 0) return invalid(at + ".share", std::to_string(h.share) + " is negative");
        if (h.share > 1000000)
            return invalid(at + ".share", std::to_string(h.share) + " is above 1000000, the largest share the ownership selector accepts");
        total_share += static_cast<uint64_t>(h.share);
        // the declared tier is the arrangement the placement strategy implies: tail gives the cold end to the last helper
        const v1::HelperTier want = tail && i + 1 == helpers.size() && helpers.size() > 1 ? v1::HelperTier::kTail : v1::HelperTier::kHead;
        if (h.tier != want)
            return mismatch(at + ".tier", std::string("the placement strategy ") +
                                              std::string(v1::kPlacementStrategyNames[static_cast<size_t>(m.spec.handoff->placement.strategy)]) +
                                              " makes this helper a " + (want == v1::HelperTier::kTail ? "tail" : "head") + " helper");
        const v1::HelperCommon effective = v1::effective_helper_settings(m, i);
        // the reader's effective-setting invariants (spec 2.2): speeds absent when balance is off, one positive speed per helper when on
        const std::string speeds_at = at + " (effective) placement.balance_speeds_milli";
        if (!effective.placement.balance && effective.placement.balance_speeds_milli)
            return invalid(speeds_at, "balance_speeds_milli must be absent when balance is false");
        if (effective.placement.balance) {
            const auto& speeds = effective.placement.balance_speeds_milli;
            if (!speeds || speeds->size() != helpers.size())
                return invalid(speeds_at, "balance needs one speed per helper (" + std::to_string(helpers.size()) + ")");
            for (int64_t s : *speeds)
                if (s < 1) return invalid(speeds_at, "a balance speed is at least 1 (1000 = 1.0), not " + std::to_string(s));
        }
        if (effective.placement.balance != first_balance || effective.placement.balance_speeds_milli != v1::effective_helper_settings(m, 0).placement.balance_speeds_milli)
            return invalid(at + " (effective) placement.balance", "balance and its speeds must be the same for every helper (set by helper_common, a role's settings or the experiment shape)");
        any_balance = any_balance || effective.placement.balance;
    }
    if (any_balance) {
        // the selector splits the experts of exactly two equal helpers, by round robin, from a heat file; the effective
        // setting is named at the helper (it may come from helper_common, a role patch or the experiment shape)
        const std::string at = "spec.processes.helpers[0] (effective) placement.balance";
        if (helpers.size() != 2) return invalid(at, "balance splits the experts of exactly two helpers, not " + std::to_string(helpers.size()));
        if (helpers[0].vram_budget_mib != helpers[1].vram_budget_mib)
            return invalid(at, "balance needs one budget for both helpers");
        if (tail) return invalid(at, "balance does not combine with the tail placement strategy");
    } else if (total_share == 0) {
        return invalid("spec.processes.helpers", "every helper's share is 0: nobody would own an expert");
    }
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i].count != static_cast<int64_t>(got[i].experts.size()))
            return mismatch("spec.plan.owner_map.helpers[" + std::to_string(i) + "].count",
                            "count " + std::to_string(got[i].count) + " but " + std::to_string(got[i].experts.size()) + " owned experts");
    return std::nullopt;
}
}  // namespace

v1::Status verify_owner_map(const v1::Manifest& m, const bc::experts::Container& model) {
    if (m.spec.processes.helpers.empty())   // engine-only: nothing to own, and the map must say so
        return m.spec.plan.owner_map.helpers.empty() ? v1::Status{} : mismatch("spec.plan.owner_map.helpers", "owned sets for an engine-only configuration");
    if (auto e = preconditions(m)) return e;
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
