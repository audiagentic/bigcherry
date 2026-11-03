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
    const CheckpointStore& ck = in.checkpoints;
    if (ck.share != "want" && ck.share != "floor" && ck.share != "tier_first")
        throw std::runtime_error("calibration.checkpoint_share must be want, floor or tier_first, not " + ck.share);
    if (ck.floor_slots > ck.want_slots)
        throw std::runtime_error("checkpoint store: floor_slots exceeds want_slots");
    Json consumers = Json::array();
    for (const auto& c : in.consumers)
        consumers.push_back({{"name", c.name}, {"mib", mib_up(c.bytes)}, {"residence", residence_name(c.residence)},
                             {"from", c.from}});

    // The room each limit leaves for the tier with the store at `slots`; refuses when even that does not fit.
    // The store and the tier are pinned host memory; they count where pinned memory counts.
    struct Room { uint64_t bytes = 0; std::string limit; Json rows; };
    auto room_with = [&](uint64_t slots) {
        Room r{std::numeric_limits<uint64_t>::max(), "", Json::array()};
        const uint64_t store = slots * ck.slot_bytes;
        for (const Limit& l : limits) {
            const bool pinned_counts = l.all || l.only == Residence::pinned;
            uint64_t used = (l.all ? reserve : 0) + (pinned_counts ? store : 0);
            const HostConsumer* largest = nullptr;
            for (const auto& c : in.consumers) {
                if (!l.all && c.residence != l.only) continue;
                used += c.bytes;
                if (!largest || c.bytes > largest->bytes) largest = &c;
            }
            r.rows.push_back({{"limit", l.name}, {"room_mib", l.room / kMiB}, {"planned_mib", mib_up(used)},
                              {"counts", l.all ? "every consumer, the checkpoint store and the reserve" :
                                         l.only == Residence::pinned ? "pinned consumers and the checkpoint store" :
                                         std::string(residence_name(l.only)) + " consumers"}});
            if (used > l.room) {
                std::string why = "host memory plan refused: " + std::to_string(mib_up(used)) + " MiB planned under " +
                                  l.name + " (" + std::to_string(l.room / kMiB) + " MiB)";
                if (l.all) why += ", including the " + std::to_string(in.reserve_mib) + " MiB reserve";
                if (pinned_counts && store)
                    why += " and the checkpoint floor of " + std::to_string(slots) + " slot(s), " +
                           std::to_string(mib_up(store)) + " MiB";
                if (largest)
                    why += "; the largest consumer is " + largest->name + " (" +
                           std::to_string(mib_up(largest->bytes)) + " MiB, " + largest->from + ")" +
                           (largest->remedy.empty() ? "" : ": " + largest->remedy);
                throw std::runtime_error(why);
            }
            if (pinned_counts && l.room - used < r.bytes) { r.bytes = l.room - used; r.limit = l.name; }
        }
        return r;
    };

    const bool tier_first = ck.share == "tier_first";
    const uint64_t planned = ck.share == "want" ? ck.want_slots : ck.floor_slots;
    uint64_t slots = planned;
    std::string ck_why = "the calibrated share (" + ck.share + "): " + ck.share_from;
    // The floor must fit whatever happens: checking it first makes a refusal name the floor, not the want.
    (void)room_with(ck.floor_slots);
    Room room = room_with(slots);
    const bool keep = in.retention_keep;
    // tier_first: the tier takes its calibrated target out of the room the floor leaves, and the store the remainder,
    // between its floor and its want, as the engine's own auto-size gives it.
    uint64_t tier_bytes = 0;
    if (tier_first) {
        const Room at_floor = room_with(ck.floor_slots);
        tier_bytes = keep ? std::min<uint64_t>(in.tier_target_bytes, at_floor.bytes) : 0;
        if (tier_bytes < in.min_tier_bytes) tier_bytes = 0;
        const uint64_t left = at_floor.bytes - tier_bytes;
        slots = std::min<uint64_t>(ck.want_slots, ck.floor_slots + (ck.slot_bytes ? left / ck.slot_bytes : 0));
        room = room_with(slots);
        ck_why = "the calibrated share (tier_first: " + ck.share_from + "): the host KV tier takes " +
                 std::to_string(tier_bytes / kMiB) + " MiB first and the store the remainder, " + std::to_string(slots) +
                 " of a want of " + std::to_string(ck.want_slots) + " (floor " + std::to_string(ck.floor_slots) + ")";
    }
    if (!tier_first && keep && room.bytes < in.min_tier_bytes && slots > ck.floor_slots && ck.slot_bytes) {
        // Step 1: give the tier its minimum out of the store, never below the floor.
        const uint64_t short_by = in.min_tier_bytes - room.bytes;
        const uint64_t give = std::min<uint64_t>(slots - ck.floor_slots, (short_by + ck.slot_bytes - 1) / ck.slot_bytes);
        slots -= give;
        room = room_with(slots);
        ck_why = "shrunk from " + std::to_string(planned) + " (" + ck.share + ") to " + std::to_string(slots) +
                 " slot(s) so the host KV tier keeps one full-length session" +
                 (slots == ck.floor_slots ? ", reaching the floor" : "");
    }

    Ledger out;
    out.checkpoint_slots = slots;
    std::string tier_why;
    if (!keep) {
        tier_why = "service.retention = evict: finished sessions are not kept";
    } else if (tier_first ? tier_bytes == 0 : room.bytes < in.min_tier_bytes) {
        tier_why = "the room left (" + std::to_string(room.bytes / kMiB) + " MiB under " + room.limit +
                   ", with the checkpoint store at its floor of " + std::to_string(slots) +
                   " slot(s)) is less than one full-length session's KV (" + std::to_string(mib_up(in.min_tier_bytes)) +
                   " MiB), so the tier would hold no whole session";
    } else if (tier_first) {
        out.host_kv_mib = tier_bytes / kMiB;
        tier_why = tier_bytes == in.tier_target_bytes
                       ? "its calibrated target (calibration.host_kv_tier.target_mib), taken before the checkpoint store"
                       : "the room left under " + room.limit + " with the store at its floor, below its calibrated target";
    } else {
        out.host_kv_mib = room.bytes / kMiB;
        tier_why = "the room left under " + room.limit + " after every consumer, the checkpoint store and the reserve";
    }
    out.explain = {{"limits", room.rows}, {"consumers", consumers}, {"reserve_mib", in.reserve_mib},
                   {"checkpoints", {{"slots", slots}, {"floor_slots", ck.floor_slots}, {"want_slots", ck.want_slots},
                                    {"slot_mib", mib_up(ck.slot_bytes)}, {"mib", mib_up(slots * ck.slot_bytes)},
                                    {"from", ck.from}, {"why", ck_why}}},
                   {"host_kv_mib", out.host_kv_mib}, {"host_kv_why", tier_why},
                   {"min_tier_mib", mib_up(in.min_tier_bytes)}};
    return out;
}

}  // namespace bc::serving
