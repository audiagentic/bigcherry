#include "xbigcherry.h"

#include <map>
#include <set>
#include <stdexcept>

namespace bc::serving {
namespace {

using Json = nlohmann::json;
enum class Kind { text, integer, boolean, integers, scalar };

[[noreturn]] void refuse(const std::string& path, const std::string& what) {
    throw std::invalid_argument("x-bigcherry: " + path + " " + what);
}

const std::map<std::string, std::map<std::string, Kind>>& sections() {
    static const std::map<std::string, std::map<std::string, Kind>> table = {
        {"power", {{"profile", Kind::text}}},
        {"service", {{"objective", Kind::text}, {"quality", Kind::text}, {"retention", Kind::text}}},
        {"memory", {{"host-pool-mib", Kind::integer}, {"host-kv-mib", Kind::integer},
                    {"host-reserve-mib", Kind::integer}, {"tail-budget-mib", Kind::integer},
                    {"helper-budgets-mib", Kind::integers}}},
        {"handoff", {{"heat", Kind::text}, {"histogram", Kind::text}, {"histogram-period-s", Kind::integer},
                     {"moe-down-tile", Kind::boolean}, {"skip-owned", Kind::boolean},
                     {"gemv-large-rows", Kind::integer}, {"small-rows", Kind::integer},
                     {"max-rows", Kind::integer}, {"placement", Kind::text}, {"shm", Kind::text}}},
        {"helper", {{"combine", Kind::boolean}, {"combine-min-rows", Kind::integer},
                    {"combine-threads", Kind::integer}, {"cpu", Kind::boolean}, {"cpu-set-auto", Kind::boolean},
                    {"power-save", Kind::boolean}, {"queue-warm", Kind::boolean}, {"count", Kind::integer},
                    {"engine-ranks", Kind::integer}, {"index", Kind::integer}}},
        {"kernels", {{"all-reduce-wire4", Kind::scalar}, {"attn-native", Kind::scalar}, {"dec11", Kind::scalar},
                     {"fuse-stage", Kind::scalar}, {"gemv", Kind::scalar}, {"gfx12-moe-gemv", Kind::scalar},
                     {"moe-ahalf", Kind::scalar}, {"moe-fp16", Kind::scalar}, {"moe-native", Kind::scalar}}},
    };
    return table;
}

void require_object(const Json& value, const std::string& path) {
    if (!value.is_object()) refuse(path, "must be a mapping");
}

void closed(const Json& object, const std::string& path, const std::set<std::string>& keys) {
    for (const auto& [key, value] : object.items())
        if (!keys.contains(key)) refuse(path + "." + key, "is not a key of this section");
}

std::int64_t whole(const Json& value, const std::string& path, std::int64_t lowest) {
    if (!value.is_number_integer() || value.get<std::int64_t>() < lowest)
        refuse(path, "must be an integer of at least " + std::to_string(lowest));
    return value.get<std::int64_t>();
}

std::string name(const Json& value, const std::string& path) {
    if (!value.is_string() || value.get_ref<const std::string&>().empty()) refuse(path, "must be a nonempty string");
    return value.get<std::string>();
}

Json section(const Json& source, const std::string& key) {
    const std::string path = "x-bigcherry." + key;
    const Json& body = source.at(key);
    require_object(body, path);
    std::set<std::string> names;
    for (const auto& [field, kind] : sections().at(key)) names.insert(field);
    closed(body, path, names);
    for (const auto& [field, value] : body.items()) {
        const std::string at = path + "." + field;
        switch (sections().at(key).at(field)) {
            case Kind::text: name(value, at); break;
            case Kind::integer: whole(value, at, 0); break;
            case Kind::boolean:
                if (!value.is_boolean()) refuse(at, "must be true or false (quote nothing, coerce nothing)");
                break;
            case Kind::integers:
                if (!value.is_array() || value.empty()) refuse(at, "must be a nonempty list of integers");
                for (std::size_t i = 0; i < value.size(); ++i) whole(value[i], at + "[" + std::to_string(i) + "]", 0);
                break;
            case Kind::scalar:
                if (!value.is_boolean() && !value.is_string() && !value.is_number_integer())
                    refuse(at, "must be a boolean, integer or string");
                break;
        }
    }
    return body;
}

}  // namespace

std::int64_t XBigcherry::rank_count() const {
    std::int64_t total = 0;
    for (const XRank& rank : ranks) total += rank.count;
    return total;
}

XBigcherry decode_x_bigcherry(const Json& source) {
    if (source.is_null()) refuse("(section)", "is required: a serving config file carries an x-bigcherry section");
    require_object(source, "x-bigcherry");
    closed(source, "x-bigcherry",
           {"schema", "description", "variant-of", "differs", "ranks", "helpers", "power", "service", "memory",
            "handoff", "helper", "kernels"});
    for (const char* required : {"schema", "description", "ranks"})
        if (!source.contains(required)) refuse(std::string("x-bigcherry.") + required, "is required");
    if (whole(source.at("schema"), "x-bigcherry.schema", 0) != 1)
        refuse("x-bigcherry.schema", "is " + source.at("schema").dump() + "; this build reads schema 1");
    XBigcherry out;
    out.description = name(source.at("description"), "x-bigcherry.description");
    if (source.contains("variant-of")) out.variant_of = name(source.at("variant-of"), "x-bigcherry.variant-of");
    if (source.contains("differs")) {
        const Json& list = source.at("differs");
        if (!out.variant_of) refuse("x-bigcherry.differs", "needs variant-of");
        if (!list.is_array()) refuse("x-bigcherry.differs", "must be a list of YAML paths");
        for (std::size_t i = 0; i < list.size(); ++i) {
            const std::string path = name(list[i], "x-bigcherry.differs[" + std::to_string(i) + "]");
            if (!path.starts_with("models.") && !path.starts_with("x-bigcherry.") && path != "cards" &&
                !path.starts_with("server."))
                refuse("x-bigcherry.differs[" + std::to_string(i) + "]", "is not a YAML path of this file: " + path);
            out.differs.push_back(path);
        }
    } else if (out.variant_of) {
        refuse("x-bigcherry.variant-of", "needs differs (an empty list when the files differ in nothing else)");
    }
    const Json& ranks = source.at("ranks");
    if (!ranks.is_array() || ranks.empty()) refuse("x-bigcherry.ranks", "must be a nonempty list of {arch, count}");
    for (std::size_t i = 0; i < ranks.size(); ++i) {
        const std::string at = "x-bigcherry.ranks[" + std::to_string(i) + "]";
        require_object(ranks[i], at);
        closed(ranks[i], at, {"arch", "count"});
        if (!ranks[i].contains("arch") || !ranks[i].contains("count")) refuse(at, "needs arch and count");
        out.ranks.push_back({name(ranks[i].at("arch"), at + ".arch"), whole(ranks[i].at("count"), at + ".count", 1)});
    }
    if (source.contains("helpers")) {
        const Json& groups = source.at("helpers");
        if (!groups.is_array() || groups.empty()) refuse("x-bigcherry.helpers", "must be a nonempty list of groups");
        for (std::size_t i = 0; i < groups.size(); ++i) {
            const std::string at = "x-bigcherry.helpers[" + std::to_string(i) + "]";
            require_object(groups[i], at);
            closed(groups[i], at, {"arch", "count", "placement", "budget-mib", "share"});
            for (const char* key : {"arch", "count", "placement"})
                if (!groups[i].contains(key)) refuse(at, std::string("needs ") + key);
            XHelperGroup group;
            group.arch = name(groups[i].at("arch"), at + ".arch");
            const Json& count = groups[i].at("count");
            if (count.is_string() && count.get<std::string>() == "all") group.count = -1;
            else group.count = whole(count, at + ".count", 1);
            group.placement = name(groups[i].at("placement"), at + ".placement");
            if (group.placement != "head" && group.placement != "tail")
                refuse(at + ".placement", "must be head or tail");
            if (groups[i].contains("budget-mib")) group.budget_mib = whole(groups[i].at("budget-mib"), at + ".budget-mib", 1);
            if (groups[i].contains("share")) group.share = whole(groups[i].at("share"), at + ".share", 1);
            out.helpers.push_back(std::move(group));
        }
    }
    for (Json XBigcherry::*field : {&XBigcherry::power, &XBigcherry::service, &XBigcherry::memory,
                                    &XBigcherry::handoff, &XBigcherry::helper, &XBigcherry::kernels}) {
        const char* key = field == &XBigcherry::power ? "power" : field == &XBigcherry::service ? "service"
                          : field == &XBigcherry::memory ? "memory" : field == &XBigcherry::handoff ? "handoff"
                          : field == &XBigcherry::helper ? "helper" : "kernels";
        if (source.contains(key)) out.*field = section(source, key);
    }
    return out;
}

}  // namespace bc::serving
