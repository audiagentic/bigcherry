#include "ledger.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace bc::serving {
namespace {
using Json = nlohmann::json;
constexpr uint64_t kMiB = 1048576;

uint64_t mib_up(uint64_t bytes) { return (bytes + kMiB - 1) / kMiB; }

void exact(const Json& j, const std::set<std::string>& keys, const std::string& where) {
    if (!j.is_object()) throw std::runtime_error(where + ": expected an object");
    for (auto it = j.begin(); it != j.end(); ++it)
        if (!keys.count(it.key())) throw std::runtime_error(where + ": unknown key " + it.key());
    for (const auto& k : keys)
        if (!j.contains(k)) throw std::runtime_error(where + ": missing " + k);
}

std::string text(const Json& j, const char* key, const std::string& where) {
    if (!j.at(key).is_string() || j.at(key).get_ref<const std::string&>().empty())
        throw std::runtime_error(where + "." + key + " must be a non-empty string");
    return j.at(key).get<std::string>();
}

uint64_t count(const Json& j, const char* key, const std::string& where) {
    const Json& v = j.at(key);
    if (!v.is_number_integer() || (!v.is_number_unsigned() && v.get<int64_t>() < 0))
        throw std::runtime_error(where + "." + key + " must be a byte count");
    return v.get<uint64_t>();
}

Residence residence_of(const std::string& name, const std::string& where) {
    if (name == "pinned") return Residence::pinned;
    if (name == "mlocked") return Residence::mlocked;
    if (name == "pageable") return Residence::pageable;
    throw std::runtime_error(where + ".residence must be pinned, mlocked or pageable, not " + name);
}

const char* residence_name(Residence r) {
    return r == Residence::pinned ? "pinned" : r == Residence::mlocked ? "mlocked" : "pageable";
}

struct Limit {
    std::string name;
    uint64_t room = 0;           // bytes
    bool all = true;             // counts every consumer, the tier and the reserve
    Residence only = Residence::pageable;   // when !all: the residence it counts
};
}  // namespace

EnginePlan parse_engine_plan(const Json& plan) {
    exact(plan, {"plan_version", "producer", "inputs_sha256", "host", "kv_bytes_per_token"}, "engine plan");
    if (!plan.at("plan_version").is_number_integer() || plan.at("plan_version").get<int64_t>() != 1)
        throw std::runtime_error("engine plan: plan_version must be 1");
    EnginePlan p;
    p.producer = text(plan, "producer", "engine plan");
    p.inputs_sha256 = text(plan, "inputs_sha256", "engine plan");
    if (!plan.at("host").is_array()) throw std::runtime_error("engine plan.host must be an array");
    std::set<std::string> names;
    for (size_t i = 0; i < plan.at("host").size(); ++i) {
        const Json& c = plan.at("host").at(i);
        const std::string where = "engine plan.host[" + std::to_string(i) + "]";
        exact(c, {"name", "bytes", "residence", "from", "remedy"}, where);
        HostConsumer h{text(c, "name", where), count(c, "bytes", where),
                       residence_of(text(c, "residence", where), where), text(c, "from", where), {}};
        if (!c.at("remedy").is_string()) throw std::runtime_error(where + ".remedy must be a string");
        h.remedy = c.at("remedy").get<std::string>();
        if (!names.insert(h.name).second) throw std::runtime_error(where + ": duplicate consumer " + h.name);
        p.host.push_back(std::move(h));
    }
    const Json& kv = plan.at("kv_bytes_per_token");
    exact(kv, {"bytes", "from"}, "engine plan.kv_bytes_per_token");
    p.kv_bytes_per_token = count(kv, "bytes", "engine plan.kv_bytes_per_token");
    p.kv_from = text(kv, "from", "engine plan.kv_bytes_per_token");
    if (!p.kv_bytes_per_token) throw std::runtime_error("engine plan.kv_bytes_per_token.bytes must be positive");
    return p;
}

Ledger host_ledger(const LedgerInputs& in) {
    const Json& h = in.host_facts;
    std::vector<Limit> limits;
    limits.push_back({"MemAvailable", h.at("mem_available_mib").get<uint64_t>() * kMiB});
    for (const Json& cg : h.at("memory_cgroups"))
        limits.push_back({"memory cgroup " + cg.at("dir").get<std::string>(), cg.at("room_mib").get<uint64_t>() * kMiB});
    const Json& pin = h.at("amdgpu_pin");
    if (pin.at("loaded").get<bool>() && !pin.at("lifted").get<bool>())
        limits.push_back({"the amdgpu pin cap", pin.at("cap_mib").get<uint64_t>() * kMiB, false, Residence::pinned});
    const Json& lock = h.at("memlock");
    if (!lock.at("limit_mib").is_null() && !lock.at("cap_ipc_lock").get<bool>())
        limits.push_back({"RLIMIT_MEMLOCK", lock.at("limit_mib").get<uint64_t>() * kMiB, false, Residence::mlocked});

    const uint64_t reserve = in.reserve_mib * kMiB;
    Json consumers = Json::array();
    for (const auto& c : in.consumers)
        consumers.push_back({{"name", c.name}, {"mib", mib_up(c.bytes)}, {"residence", residence_name(c.residence)},
                             {"from", c.from}});

    // The tier is pinned host memory (the engine's prefix cache tier); it counts where pinned memory counts.
    uint64_t tier_room = std::numeric_limits<uint64_t>::max();
    std::string tier_limit;
    Json limit_rows = Json::array();
    for (const Limit& l : limits) {
        uint64_t used = l.all ? reserve : 0;
        const HostConsumer* largest = nullptr;
        for (const auto& c : in.consumers) {
            if (!l.all && c.residence != l.only) continue;
            used += c.bytes;
            if (!largest || c.bytes > largest->bytes) largest = &c;
        }
        limit_rows.push_back({{"limit", l.name}, {"room_mib", l.room / kMiB}, {"planned_mib", mib_up(used)},
                              {"counts", l.all ? "every consumer and the reserve" :
                                         std::string(residence_name(l.only)) + " consumers"}});
        if (used > l.room) {
            std::string why = "host memory plan refused: " + std::to_string(mib_up(used)) + " MiB planned under " +
                              l.name + " (" + std::to_string(l.room / kMiB) + " MiB)";
            if (l.all) why += ", including the " + std::to_string(in.reserve_mib) + " MiB reserve";
            if (largest)
                why += "; the largest consumer is " + largest->name + " (" + std::to_string(mib_up(largest->bytes)) +
                       " MiB, " + largest->from + ")" + (largest->remedy.empty() ? "" : ": " + largest->remedy);
            throw std::runtime_error(why);
        }
        const bool tier_counts = l.all || l.only == Residence::pinned;
        if (tier_counts && l.room - used < tier_room) {
            tier_room = l.room - used;
            tier_limit = l.name;
        }
    }

    Ledger out;
    std::string tier_why;
    if (!in.retention_keep) {
        tier_why = "service.retention = evict: finished sessions are not kept";
    } else if (tier_room < in.min_tier_bytes) {
        tier_why = "the room left (" + std::to_string(tier_room / kMiB) + " MiB under " + tier_limit +
                   ") is less than one full-length session's KV (" + std::to_string(mib_up(in.min_tier_bytes)) +
                   " MiB), so the tier would hold no whole session";
    } else {
        out.host_kv_mib = tier_room / kMiB;
        tier_why = "the room left under " + tier_limit + " after every consumer and the reserve";
    }
    out.explain = {{"limits", limit_rows}, {"consumers", consumers}, {"reserve_mib", in.reserve_mib},
                   {"host_kv_mib", out.host_kv_mib}, {"host_kv_why", tier_why},
                   {"min_tier_mib", mib_up(in.min_tier_bytes)}};
    return out;
}

}  // namespace bc::serving
