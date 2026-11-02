#include "calibration.h"
#include "third_party/toml.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace bc::serving {
namespace {
using Json = nlohmann::json;

// calibration.toml, embedded at build time (CMakeLists.txt writes calibration_seed.inc).
constexpr const char kSeed[] =
#include "calibration_seed.inc"
    ;

Json value_of(const toml::node& node, const std::string& where) {
    if (auto s = node.as_string()) return s->get();
    if (auto n = node.as_integer()) return n->get();
    if (auto d = node.as_floating_point()) return d->get();
    if (auto b = node.as_boolean()) return b->get();
    throw std::runtime_error(where + ": a calibration value is a number, string or boolean");
}

std::string text_of(const toml::table& t, const char* key, const std::string& where) {
    const auto* v = t.get(key);
    if (!v || !v->as_string() || v->as_string()->get().empty())
        throw std::runtime_error(where + ": " + key + " must be a non-empty string");
    return v->as_string()->get();
}

int specificity(const Json& match) { return int(match.size()); }

bool matches(const Json& match, const CalibrationMatch& at) {
    for (auto it = match.begin(); it != match.end(); ++it) {
        const std::string& want = it.value().get_ref<const std::string&>();
        const std::string& have = it.key() == "model" ? at.model : it.key() == "layout" ? at.layout : at.objective;
        if (want != have) return false;
    }
    return true;
}
}  // namespace

std::string Calibrated::describe(const std::string& key) const {
    return "calibrated (" + source + ": " + seeded + ") " + key + ": " + evidence;
}

void Calibration::parse(const std::string& text, const std::string& where, const std::string& source,
                        std::vector<Entry>& into) {
    toml::table root;
    try { root = toml::parse(text, where); }
    catch (const toml::parse_error& e) { throw std::runtime_error(where + ": TOML parse error: " + e.what()); }
    for (const auto& [k, v] : root)
        if (k.str() != "schema_version" && k.str() != "entry")
            throw std::runtime_error(where + ": unknown key " + std::string(k.str()));
    const auto* version = root.get("schema_version");
    if (!version || !version->as_integer() || version->as_integer()->get() != 1)
        throw std::runtime_error(where + ": schema_version must be 1");
    const auto* list = root.get("entry");
    if (!list) return;
    if (!list->as_array()) throw std::runtime_error(where + ": entry must be an array of tables");
    size_t index = 0;
    for (const auto& node : *list->as_array()) {
        const std::string at = where + ": entry[" + std::to_string(index++) + "]";
        const auto* t = node.as_table();
        if (!t) throw std::runtime_error(at + " must be a table");
        for (const auto& [k, v] : *t)
            if (k.str() != "key" && k.str() != "match" && k.str() != "value" && k.str() != "seeded" &&
                k.str() != "evidence")
                throw std::runtime_error(at + ": unknown key " + std::string(k.str()));
        Entry e;
        e.key = text_of(*t, "key", at);
        if (e.key.rfind("calibration.", 0) != 0)
            throw std::runtime_error(at + ": key must start with calibration.");
        e.seeded = text_of(*t, "seeded", at);
        e.evidence = text_of(*t, "evidence", at);
        e.source = source;
        const auto* value = t->get("value");
        if (!value) throw std::runtime_error(at + ": missing value");
        e.value = value_of(*value, at);
        e.match = Json::object();
        if (const auto* m = t->get("match")) {
            if (!m->as_table()) throw std::runtime_error(at + ": match must be a table");
            for (const auto& [mk, mv] : *m->as_table()) {
                const std::string name(mk.str());
                if (name != "model" && name != "layout" && name != "objective")
                    throw std::runtime_error(at + ": match." + name + " (allowed: model, layout, objective)");
                if (!mv.as_string() || mv.as_string()->get().empty())
                    throw std::runtime_error(at + ": match." + name + " must be a non-empty string");
                e.match[name] = mv.as_string()->get();
            }
        }
        for (const Entry& other : into)
            if (other.key == e.key && other.match == e.match)
                throw std::runtime_error(at + ": duplicate entry for " + e.key + " " + e.match.dump());
        into.push_back(std::move(e));
    }
}

Calibration Calibration::load(const std::string& seed_toml, const std::filesystem::path& cache) {
    Calibration c;
    parse(seed_toml, "calibration.toml (seed)", "seed", c.entries_);
    if (!cache.empty()) {
        std::ifstream in(cache);
        if (!in) throw std::runtime_error("cannot read calibration cache " + cache.string());
        std::ostringstream text;
        text << in.rdbuf();
        std::vector<Entry> refits;
        parse(text.str(), cache.string(), "cache", refits);
        for (auto& e : refits) c.entries_.push_back(std::move(e));
    }
    return c;
}

Calibration Calibration::standard(const std::filesystem::path& cache) { return load(kSeed, cache); }

Calibrated Calibration::get(const std::string& key, const CalibrationMatch& at) const {
    const Entry* best = nullptr;
    for (const Entry& e : entries_) {
        if (e.key != key || !matches(e.match, at)) continue;
        if (!best) { best = &e; continue; }
        const int a = specificity(e.match), b = specificity(best->match);
        if (a > b || (a == b && e.source == "cache" && best->source == "seed")) best = &e;
        else if (a == b && e.source == best->source)
            throw std::runtime_error("calibration: ambiguous entries for " + key + ": " + e.match.dump() +
                                     " and " + best->match.dump());
    }
    if (!best)
        throw std::runtime_error("calibration: no entry for " + key + " (model " + at.model + ", layout " +
                                 at.layout + ", objective " + at.objective +
                                 "); calibrate this configuration or add a seed with its evidence");
    return {best->value, best->source, best->seeded, best->evidence, best->match};
}

double Calibration::number(const std::string& key, const CalibrationMatch& at) const {
    const Calibrated c = get(key, at);
    if (!c.value.is_number()) throw std::runtime_error("calibration: " + key + " must be a number");
    return c.value.get<double>();
}

}  // namespace bc::serving
