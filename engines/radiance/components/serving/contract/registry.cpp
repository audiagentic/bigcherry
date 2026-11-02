#include "registry.h"

#include "jcs.h"
#include "sha256.h"
#include "third_party/toml.hpp"

#include <algorithm>
#include <functional>
#include <set>
#include <sstream>
#include <stdexcept>

namespace bc::contract {
namespace {

using Json = nlohmann::json;

[[noreturn]] void fault(const std::string& where, const std::string& what) {
    throw std::runtime_error(where + ": " + what);
}

// ---- TOML access, strict ---------------------------------------------------------------------------------------------

/* Every number in the registry is an integer (spec §8, §11 amendment 3): a float anywhere is refused, naming it. */
void refuse_floats(const toml::node& n, const std::string& at) {
    if (n.is_floating_point()) fault(at, "non-integer number (the v1 contract has integers only)");
    if (const auto* t = n.as_table())
        for (const auto& [k, v] : *t) refuse_floats(v, at + "." + std::string(k.str()));
    if (const auto* a = n.as_array())
        for (size_t i = 0; i < a->size(); ++i) refuse_floats(*a->get(i), at + "[" + std::to_string(i) + "]");
}

void only_keys(const toml::table& t, const std::set<std::string>& allowed, const std::string& at) {
    for (const auto& [k, v] : t)
        if (!allowed.count(std::string(k.str()))) fault(at, "unknown key '" + std::string(k.str()) + "'");
}

std::string str(const toml::table& t, const char* key, const std::string& at, bool required = true) {
    const auto* n = t.get(key);
    if (!n) {
        if (required) fault(at, std::string("missing '") + key + "'");
        return "";
    }
    if (!n->is_string()) fault(at, std::string("'") + key + "' must be a string");
    return std::string(**n->as_string());
}

std::optional<int64_t> integer(const toml::table& t, const char* key, const std::string& at) {
    const auto* n = t.get(key);
    if (!n) return std::nullopt;
    if (!n->is_integer()) fault(at, std::string("'") + key + "' must be an integer");
    return **n->as_integer();
}

bool boolean(const toml::table& t, const char* key, const std::string& at) {
    const auto* n = t.get(key);
    if (!n) return false;
    if (!n->is_boolean()) fault(at, std::string("'") + key + "' must be a boolean");
    return **n->as_boolean();
}

Json to_json(const toml::node& n, const std::string& at) {
    if (n.is_string()) return std::string(**n.as_string());
    if (n.is_integer()) return **n.as_integer();
    if (n.is_boolean()) return **n.as_boolean();
    if (const auto* a = n.as_array()) {
        Json out = Json::array();
        for (size_t i = 0; i < a->size(); ++i) out.push_back(to_json(*a->get(i), at + "[" + std::to_string(i) + "]"));
        return out;
    }
    fault(at, "unsupported value (strings, integers, booleans and arrays of them only)");
}

std::vector<Json> list(const toml::table& t, const char* key, const std::string& at) {
    const auto* n = t.get(key);
    if (!n) return {};
    if (!n->is_array()) fault(at, std::string("'") + key + "' must be an array");
    const Json j = to_json(*n, at + "." + key);
    return {j.begin(), j.end()};
}

std::vector<std::string> strings(const toml::table& t, const char* key, const std::string& at) {
    std::vector<std::string> out;
    for (const Json& v : list(t, key, at)) {
        if (!v.is_string()) fault(at, std::string("'") + key + "' must hold strings");
        out.push_back(v.get<std::string>());
    }
    return out;
}

// ---- names -----------------------------------------------------------------------------------------------------------

const std::set<std::string>& keywords() {
    static const std::set<std::string> k = {
        "alignas", "alignof", "and", "asm", "auto", "bool", "break", "case", "catch", "char", "class", "concept",
        "const", "consteval", "constexpr", "constinit", "continue", "co_await", "co_return", "co_yield", "decltype",
        "default", "delete", "do", "double", "else", "enum", "explicit", "export", "extern", "false", "float", "for",
        "friend", "goto", "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "nullptr",
        "operator", "or", "private", "protected", "public", "register", "requires", "return", "short", "signed",
        "sizeof", "static", "struct", "switch", "template", "this", "throw", "true", "try", "typedef", "typeid",
        "typename", "union", "unsigned", "using", "virtual", "void", "volatile", "while", "xor"};
    return k;
}

bool identifier(const std::string& s) {
    if (s.empty() || !(std::isalpha(static_cast<unsigned char>(s[0])) || s[0] == '_')) return false;
    for (char c : s)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) return false;
    return !keywords().count(s);
}

bool type_name(const std::string& s) {
    return identifier(s) && std::isupper(static_cast<unsigned char>(s[0]));
}

/* "fp8_e4m3_block" -> "kFp8E4m3Block", "host_reserve.floor_mib" -> "kHostReserveFloorMib". */
std::string enumerator(const std::string& wire) {
    std::string out = "k";
    bool up = true;
    for (char c : wire) {
        if (c == '_' || c == '.' || c == '-') { up = true; continue; }
        out += up ? static_cast<char>(std::toupper(static_cast<unsigned char>(c))) : c;
        up = false;
    }
    return out;
}

// ---- loading ---------------------------------------------------------------------------------------------------------

const std::set<std::string> kKinds = {"object", "map", "array", "integer", "string", "boolean", "enum", "int_enum",
                                      "ref", "calibration_key", "calibration_value"};
const std::set<std::string> kClasses = {"envelope", "policy", "fact", "derived", "tuned", "diagnostic"};
const std::set<std::string> kScalarAttrs = {"minimum", "maximum", "exclusive_maximum", "min_length", "pattern",
                                            "format", "const", "values", "ref", "cpp_type"};

/* The scalar attributes of a node, from `t` with every key prefixed by `prefix` ("", "item_" or "value_"). */
void scalar_attrs(const toml::table& t, const std::string& prefix, const std::string& at, Node* n) {
    auto k = [&](const char* name) { return prefix + name; };
    n->minimum = integer(t, k("minimum").c_str(), at);
    n->maximum = integer(t, k("maximum").c_str(), at);
    n->exclusive_maximum = integer(t, k("exclusive_maximum").c_str(), at);
    n->min_length = integer(t, k("min_length").c_str(), at);
    n->pattern = str(t, k("pattern").c_str(), at, false);
    n->format = str(t, k("format").c_str(), at, false);
    if (const auto* c = t.get(k("const"))) n->constant = to_json(*c, at + "." + k("const"));
    n->values = list(t, k("values").c_str(), at);
    n->ref = str(t, k("ref").c_str(), at, false);
}

/* What each kind may carry; anything else set on it is a fault. */
void check_scalar(const Node& n, const std::string& at) {
    const bool num = n.minimum || n.maximum || n.exclusive_maximum;
    const bool text = n.min_length || !n.pattern.empty() || !n.format.empty();
    auto refuse = [&](bool bad, const char* what) {
        if (bad) fault(at, std::string("kind '") + n.kind + "' cannot have " + what);
    };
    if (n.kind == "number") fault(at, "kind 'number': the v1 contract has integers only");
    if (n.min_length && *n.min_length < 0) fault(at, "min_length is not negative");
    if (n.values.size() > 256) fault(at, "an enum has at most 256 values (uint8_t)");
    if (!kKinds.count(n.kind) && n.kind != "tuple") fault(at, "unknown kind '" + n.kind + "'");
    refuse(n.kind != "integer" && num, "a numeric bound");
    refuse(n.kind != "string" && text, "a string constraint");
    refuse(n.kind != "enum" && n.kind != "int_enum" && !n.values.empty(), "values");
    refuse(n.kind != "ref" && !n.ref.empty(), "a ref");
    refuse(n.kind != "enum" && !n.cpp_type.empty(), "a cpp_type");
    refuse(n.kind != "integer" && n.kind != "string" && n.constant.has_value(), "a const");
    if (n.constant) {
        refuse(num || text, "constraints beside a const");
        if (n.kind == "integer" && !n.constant->is_number_integer()) fault(at, "an integer const must be an integer");
        if (n.kind == "string" && !n.constant->is_string()) fault(at, "a string const must be a string");
    }
    if (n.kind == "ref" && n.ref.empty()) fault(at, "kind 'ref' needs a ref");
    if (n.kind == "enum" || n.kind == "int_enum") {
        if (n.values.empty()) fault(at, "an enum needs values");
        std::set<std::string> seen;
        for (const Json& v : n.values) {
            if (n.kind == "enum" && !v.is_string()) fault(at, "enum values are strings (int_enum for integers)");
            if (n.kind == "int_enum" && !v.is_number_integer()) fault(at, "int_enum values are integers");
            if (!seen.insert(v.dump()).second) fault(at, "repeated enum value " + v.dump());
        }
        if (n.kind == "enum" && !type_name(n.cpp_type)) fault(at, "an enum needs cpp_type, a C++ type name");
        if (n.kind == "enum") {
            std::set<std::string> names;
            for (const Json& v : n.values) {
                const std::string e = enumerator(v.get<std::string>());
                if (!identifier(e)) fault(at, "enum value " + v.dump() + " gives the invalid C++ enumerator " + e);
                if (!names.insert(e).second)
                    fault(at, "enum values " + v.dump() + " and another map to the same C++ enumerator");
            }
        }
    }
}

std::string parent_of(const std::string& path) {
    const size_t slash = path.rfind('/');
    if (slash == std::string::npos) return "";
    return path.substr(0, slash);
}

struct Loader {
    Registry r;

    size_t add(Node n, const std::string& at) {
        if (r.by_path.count(n.path)) fault(at, "path '" + n.path + "' is defined twice");
        r.nodes.push_back(std::move(n));
        const size_t i = r.nodes.size() - 1;
        r.by_path[r.nodes[i].path] = i;
        return i;
    }

    /* Attach node i to its parent, which must already exist and accept it. */
    void attach(size_t i, const std::string& at) {
        Node& n = r.nodes[i];
        const std::string parent = parent_of(n.path);
        n.key = n.path.substr(parent.size() + 1);
        const auto it = r.by_path.find(parent);
        if (it == r.by_path.end()) fault(at, "'" + n.path + "' comes before (or without) its parent '" + parent + "'");
        Node& p = r.nodes[it->second];
        if (n.key == "[]") {
            if (p.kind != "array" || !p.children.empty()) fault(at, "'[]' needs an array parent that has no item yet");
        } else if (n.key == "*") {
            if (p.kind != "map" || !p.children.empty()) fault(at, "'*' needs a map parent that has no value yet");
        } else {
            if (p.kind != "object") fault(at, "'" + parent + "' is not an object");
            if (!identifier(n.cpp_member.empty() ? n.key : n.cpp_member))
                fault(at, "member '" + n.key + "' is not a C++ identifier; set cpp_member");
        }
        if ((n.key == "[]" || n.key == "*") && n.optional) fault(at, "an item or map value cannot be optional");
        p.children.push_back(i);
    }

    void field(const toml::table& t, const std::string& at) {
        only_keys(t, {"path", "kind", "optional", "nullable", "doc", "cpp", "cpp_type", "cpp_member", "ref", "const",
                      "values", "minimum", "maximum", "exclusive_maximum", "min_length", "pattern", "format",
                      "item_kind", "item_ref", "item_min_length", "item_minimum", "item_values", "item_pattern",
                      "min_items", "max_items", "unique_items", "tuple", "key_ref", "required_keys", "min_properties",
                      "constraints", "radiance_option", "radiance_config"},
                  at);
        Node n;
        n.path = str(t, "path", at);
        const std::string where = at + " (" + (n.path.empty() ? "\"\"" : n.path) + ")";
        n.kind = str(t, "kind", where);
        n.doc = str(t, "doc", where);
        if (n.doc.empty()) fault(where, "empty doc");
        n.optional = boolean(t, "optional", where);
        n.nullable = boolean(t, "nullable", where);
        n.cpp = str(t, "cpp", where, false);
        n.cpp_type = str(t, "cpp_type", where, false);
        n.cpp_member = str(t, "cpp_member", where, false);
        n.radiance_option = str(t, "radiance_option", where, false);
        n.radiance_config = str(t, "radiance_config", where, false);
        if (n.radiance_option.empty() != n.radiance_config.empty())
            fault(where, "radiance_option and radiance_config go together");
        if (!n.cpp_member.empty() && !identifier(n.cpp_member)) fault(where, "cpp_member is not a C++ identifier");
        n.min_properties = integer(t, "min_properties", where);
        n.min_items = integer(t, "min_items", where);
        n.max_items = integer(t, "max_items", where);
        for (const auto* v : {&n.min_properties, &n.min_items, &n.max_items})
            if (*v && **v < 0) fault(where, "item and member counts are not negative");
        if (n.min_items && n.max_items && *n.min_items > *n.max_items) fault(where, "min_items exceeds max_items");
        n.unique_items = boolean(t, "unique_items", where);
        if (const auto* c = t.get("constraints")) {
            if (!c->is_string()) fault(where, "constraints is a JSON array, as a string");
            try { n.constraints = Json::parse(std::string(**c->as_string())); }
            catch (const std::exception& e) { fault(where, std::string("constraints is not JSON: ") + e.what()); }
            if (!n.constraints->is_array() || n.constraints->empty()) fault(where, "constraints is a nonempty array");
            const Json flat = n.constraints->flatten();
            for (const auto& [p, v] : flat.items())
                if (v.is_number_float()) fault(where, "constraints hold a non-integer number at " + p);
        }
        const bool container = n.kind == "object" || n.kind == "map" || n.kind == "array";
        if (n.kind == "object") {
            if (!type_name(n.cpp)) fault(where, "an object needs cpp, a C++ type name");
        } else if (!n.cpp.empty()) {
            fault(where, "only objects have cpp");
        }
        if (n.constraints && n.kind != "object") fault(where, "only objects have constraints");
        if (n.min_properties && n.kind != "object" && n.kind != "map") fault(where, "min_properties is for objects");
        if ((n.min_items || n.max_items || n.unique_items) && n.kind != "array") fault(where, "item counts are for arrays");
        if (n.nullable && container) fault(where, "containers are not nullable");
        if (!container) {
            scalar_attrs(t, "", where, &n);
            check_scalar(n, where);
        } else {
            for (const char* k : {"minimum", "maximum", "exclusive_maximum", "min_length", "pattern", "format", "const",
                                  "values", "ref"})
                if (t.get(k)) fault(where, std::string("a container cannot have '") + k + "'");
        }
        if (n.kind != "array")
            for (const char* k : {"item_kind", "item_ref", "item_min_length", "item_minimum", "item_values",
                                  "item_pattern", "tuple"})
                if (t.get(k)) fault(where, std::string("only arrays have '") + k + "'");
        if (n.kind == "array" && !n.cpp_type.empty() && str(t, "item_kind", where) != "enum")
            fault(where, "cpp_type on an array names its enum items");
        if (n.kind != "map" && (t.get("key_ref") || t.get("required_keys"))) fault(where, "only maps have keys");
        if (n.kind == "map") {
            n.key_ref = str(t, "key_ref", where);
            n.required_keys = strings(t, "required_keys", where);
        }
        std::string item_cpp_type;
        if (n.kind == "array") { item_cpp_type = n.cpp_type; n.cpp_type.clear(); }
        const size_t i = add(n, where);
        if (r.nodes[i].path.empty()) {
            if (r.root != SIZE_MAX) fault(where, "two roots");
            if (r.nodes[i].kind != "object" || r.nodes[i].optional) fault(where, "the root is a required object");
            r.root = i;
        } else {
            attach(i, where);
        }
        if (r.nodes[i].kind == "array") array_item(t, i, item_cpp_type, where);
    }

    /* An array's item: an object item comes as its own [[field]] "<path>/[]"; a scalar or tuple item is built here. */
    void array_item(const toml::table& t, size_t array, const std::string& cpp_type, const std::string& at) {
        const std::string item_kind = str(t, "item_kind", at);
        if (item_kind == "object") {
            for (const char* k : {"item_ref", "item_min_length", "item_minimum", "item_values", "item_pattern", "tuple"})
                if (t.get(k)) fault(at, std::string("an object item is its own [[field]]; '") + k + "' does not apply");
            return;   // its [[field]] must follow; checked in finish()
        }
        Node item;
        item.path = r.nodes[array].path + "/[]";
        item.kind = item_kind;
        if (item_kind == "tuple") {
            for (const char* k : {"item_ref", "item_min_length", "item_minimum", "item_values", "item_pattern"})
                if (t.get(k)) fault(at, std::string("a tuple item takes only 'tuple'; '") + k + "' does not apply");
            if (!cpp_type.empty()) fault(at, "a tuple item has no cpp_type");
            item.tuple = strings(t, "tuple", at);
            if (item.tuple.empty()) fault(at, "a tuple item needs its element defs");
        } else {
            if (t.get("tuple")) fault(at, "'tuple' is for tuple items");
            Node probe;
            scalar_attrs(t, "item_", at, &probe);
            item.minimum = probe.minimum; item.maximum = probe.maximum; item.min_length = probe.min_length;
            item.pattern = probe.pattern; item.values = probe.values; item.ref = probe.ref;
            item.cpp_type = cpp_type;
            if (item_kind == "object" || item_kind == "map" || item_kind == "array")
                fault(at, "item_kind '" + item_kind + "' is not supported");
            check_scalar(item, at + " item");
        }
        const size_t i = add(item, at);
        attach(i, at);
    }

    void def(const toml::table& t, const std::string& at) {
        Def d;
        d.name = str(t, "name", at);
        const std::string where = at + " (" + d.name + ")";
        d.kind = str(t, "kind", where);
        d.doc = str(t, "doc", where);
        if (d.doc.empty()) fault(where, "empty doc");
        if (!identifier(d.name)) fault(where, "a def name is an identifier");
        for (const Def& o : r.defs)
            if (o.name == d.name) fault(where, "def '" + d.name + "' is defined twice");
        Node n;
        n.path = "$defs/" + d.name;
        n.doc = d.doc;
        if (d.kind == "object" || d.kind == "patch") {
            only_keys(t, {"name", "kind", "doc", "cpp", "of"}, where);
            d.cpp = str(t, "cpp", where);
            if (!type_name(d.cpp)) fault(where, "cpp must be a C++ type name");
            n.kind = "object";
            n.cpp = d.cpp;
            if (d.kind == "patch") {
                d.of = str(t, "of", where);
                n.patch = true;
                n.min_properties = 1;
            } else if (t.get("of")) {
                fault(where, "'of' is for patch defs");
            }
        } else {
            only_keys(t, {"name", "kind", "doc", "minimum", "maximum", "exclusive_maximum", "min_length", "pattern",
                          "format"},
                      where);
            n.kind = d.kind;
            scalar_attrs(t, "", where, &n);
            check_scalar(n, where);
            if (n.kind != "integer" && n.kind != "string") fault(where, "a scalar def is an integer or a string");
        }
        d.node = add(n, where);
        r.defs.push_back(d);
    }

    void calibration_key(const toml::table& t, const std::string& at) {
        only_keys(t, {"key", "doc", "value_kind", "value_ref", "value_minimum", "value_maximum",
                      "value_exclusive_maximum", "value_min_length", "value_pattern", "value_values"},
                  at);
        CalibrationKey c;
        c.key = str(t, "key", at);
        const std::string where = at + " (" + c.key + ")";
        c.doc = str(t, "doc", where);
        if (c.doc.empty()) fault(where, "empty doc");
        for (const CalibrationKey& o : r.calibration)
            if (o.key == c.key) fault(where, "calibration key '" + c.key + "' is defined twice");
        Node v;
        v.path = "calibration:" + c.key;
        v.kind = str(t, "value_kind", where);
        scalar_attrs(t, "value_", where, &v);
        if (v.kind == "enum") v.cpp_type = "CalibrationChoice";   // never a C++ type: checked as JSON only
        if (v.kind != "ref" && v.kind != "integer" && v.kind != "string" && v.kind != "enum" && v.kind != "boolean")
            fault(where, "a calibration value is a ref, integer, string, enum or boolean");
        check_scalar(v, where);
        c.value = add(v, where);
        r.calibration.push_back(c);
    }

    /* Clone `from`'s members under `to` as a patch: every member optional, every object a nonempty partial map. */
    void clone_patch(size_t from, size_t to, const std::string& at) {
        const std::vector<size_t> kids = r.nodes[from].children;
        for (size_t c : kids) {
            Node n = r.nodes[c];
            n.path = r.nodes[to].path + "/" + n.key;
            n.children.clear();
            n.patch = true;
            if (n.key != "[]" && n.key != "*") n.optional = true;
            if (n.kind == "object") {
                n.cpp += "Patch";
                n.min_properties = 1;
            }
            const size_t i = add(n, at);
            attach(i, at);
            clone_patch(c, i, at);
        }
    }

    void finish() {
        if (r.root == SIZE_MAX) fault("registry", "no root [[field]] (path \"\")");
        std::set<std::string> def_names;
        for (const Def& d : r.defs) def_names.insert(d.name);
        for (const Def& d : r.defs) {
            if (d.kind != "patch") continue;
            const auto it = std::find_if(r.defs.begin(), r.defs.end(), [&](const Def& o) { return o.name == d.of; });
            if (it == r.defs.end() || it->kind != "object") fault("def " + d.name, "'of' must name an object def");
            if (!r.nodes[d.node].children.empty()) fault("def " + d.name, "a patch def has no fields of its own");
            clone_patch(it->node, d.node, "def " + d.name);
        }
        for (const Node& n : r.nodes) {
            const std::string at = n.path.empty() ? "\"\"" : n.path;
            if (n.kind == "ref" && !def_names.count(n.ref)) fault(at, "ref '" + n.ref + "' names no def");
            for (const std::string& t : n.tuple)
                if (!def_names.count(t)) fault(at, "tuple element '" + t + "' names no def");
            if (n.kind == "map" && !def_names.count(n.key_ref)) fault(at, "key_ref '" + n.key_ref + "' names no def");
            if ((n.kind == "array" || n.kind == "map") && n.children.size() != 1)
                fault(at, n.kind == "array" ? "an array needs exactly one item" : "a map needs its '*' value");
            if (n.kind == "calibration_key" || n.kind == "calibration_value") {
                if (n.path != "/spec/calibration/[]/key" && n.path != "/spec/calibration/[]/value")
                    fault(at, "calibration kinds belong to /spec/calibration/[]");
            }
        }
        if (r.calibration.empty()) fault("registry", "no [[calibration_key]]");
        if (r.calibration.size() > 256) fault("registry", "at most 256 calibration keys (uint8_t)");
        std::set<std::string> enumerators;
        for (const CalibrationKey& c : r.calibration) {
            const std::string e = enumerator(c.key);
            if (!identifier(e)) fault("calibration_key " + c.key, "gives the invalid C++ enumerator " + e);
            if (!enumerators.insert(e).second) fault("calibration_key " + c.key, "maps to the C++ enumerator " + e + " twice");
        }
        auto def_kind = [&](const std::string& name) {
            for (const Def& d : r.defs)
                if (d.name == name) return d.kind;
            return std::string();
        };
        for (const Node& n : r.nodes) {
            const std::string at = n.path.empty() ? "\"\"" : n.path;
            if (n.kind == "map" && def_kind(n.key_ref) != "string") fault(at, "key_ref must name a string def");
            for (const std::string& t : n.tuple)
                if (def_kind(t) != "integer") fault(at, "tuple elements must be integer defs");
            if (n.path.rfind("calibration:", 0) == 0 && n.kind == "ref" && def_kind(n.ref) != "integer" &&
                def_kind(n.ref) != "string")
                fault(at, "a calibration value refers to an integer or string def");
            if (n.kind == "object") {
                std::set<std::string> members;
                for (size_t c : n.children) {
                    const Node& m = r.nodes[c];
                    const std::string name = m.cpp_member.empty() ? m.key : m.cpp_member;
                    if (!members.insert(name).second) fault(m.path, "C++ member " + name + " is repeated in " + n.cpp);
                }
            }
        }
        for (const char* p : {"/spec/calibration/[]/key", "/spec/calibration/[]/value"})
            if (!r.by_path.count(p)) fault("registry", std::string("missing ") + p);
        // object members in a patch def or a ref to an object def are fine; C++ type names must not collide
        std::map<std::string, std::string> types;   // name -> what defines it
        for (const Node& n : r.nodes) {
            if (n.kind == "object") {
                const auto [it, fresh] = types.emplace(n.cpp, "struct " + n.path);
                if (!fresh) fault(n.path, "C++ type " + n.cpp + " is also " + it->second);
            }
        }
        std::map<std::string, Json> enums;
        for (const Node& n : r.nodes) {
            if (n.kind != "enum" || n.path.rfind("calibration:", 0) == 0) continue;
            const Json vals = n.values;
            const auto [it, fresh] = enums.emplace(n.cpp_type, vals);
            if (!fresh && it->second != vals) fault(n.path, "enum " + n.cpp_type + " is defined with other values");
            if (types.count(n.cpp_type)) fault(n.path, "enum " + n.cpp_type + " collides with a struct");
        }
        for (const char* reserved : {"CalibrationKey", "CalibrationValue", "FieldInfo"})
            if (types.count(reserved) || enums.count(reserved)) fault("registry", std::string(reserved) + " is reserved");
    }

    void classes(const toml::array* defaults) {
        std::vector<std::pair<std::string, std::string>> rules;
        if (defaults)
            for (size_t i = 0; i < defaults->size(); ++i) {
                const std::string at = "class_default[" + std::to_string(i) + "]";
                const auto* t = defaults->get(i)->as_table();
                if (!t) fault(at, "must be a table");
                only_keys(*t, {"path", "class"}, at);
                const std::string path = str(*t, "path", at), cls = str(*t, "class", at);
                if (!kClasses.count(cls)) fault(at, "unknown class '" + cls + "'");
                if (!r.by_path.count(path)) fault(at, "path '" + path + "' is not a field");
                for (const auto& [p, c] : rules)
                    if (p == path) fault(at, "path '" + path + "' has two class defaults");
                rules.emplace_back(path, cls);
            }
        for (Node& n : r.nodes) {
            if (n.path.rfind("calibration:", 0) == 0) continue;
            size_t best = 0;
            for (const auto& [p, c] : rules) {
                const bool covers = n.path == p || (n.path.rfind(p + "/", 0) == 0);
                if (covers && p.size() + 1 > best) { best = p.size() + 1; n.cls = c; }
            }
        }
        // every value in the document has a class; a container may hold several (a def's fields take its subtree rule)
        for (const Node& n : r.nodes) {
            const bool container = n.kind == "object" || n.kind == "array" || n.kind == "map";
            if (n.cls.empty() && !container && n.path.rfind("calibration:", 0) != 0 && n.path.rfind("$defs/", 0) != 0)
                fault(n.path, "no [[class_default]] covers this field");
        }
    }
};

// ---- the schema ------------------------------------------------------------------------------------------------------

struct SchemaWriter {
    const Registry& r;

    OrderedJson ref(const std::string& name) const { return OrderedJson{{"$ref", "#/$defs/" + name}}; }

    OrderedJson scalar(const Node& n) const {
        OrderedJson s = OrderedJson::object();
        if (n.kind == "ref") return ref(n.ref);
        if (n.constant) { s["const"] = *n.constant; return s; }
        if (n.kind == "enum" || n.kind == "int_enum") {
            s["enum"] = OrderedJson::array();
            for (const Json& v : n.values) s["enum"].push_back(OrderedJson::parse(v.dump()));
            return s;
        }
        if (n.kind == "calibration_key") {
            s["enum"] = OrderedJson::array();
            for (const CalibrationKey& c : r.calibration) s["enum"].push_back(c.key);
            return s;
        }
        if (n.kind == "calibration_value") {
            // spec §11, amendment 3: a calibration value is an integer, a string or a boolean; each key narrows it
            s["anyOf"] = OrderedJson::array({OrderedJson{{"type", "integer"}, {"minimum", 0}},
                                             OrderedJson{{"type", "string"}, {"minLength", 1}},
                                             OrderedJson{{"type", "boolean"}}});
            return s;
        }
        s["type"] = n.kind;
        if (n.minimum) s["minimum"] = *n.minimum;
        if (n.maximum) s["maximum"] = *n.maximum;
        if (n.exclusive_maximum) s["exclusiveMaximum"] = *n.exclusive_maximum;
        if (n.min_length) s["minLength"] = *n.min_length;
        if (!n.pattern.empty()) s["pattern"] = n.pattern;
        if (!n.format.empty()) s["format"] = n.format;
        return s;
    }

    OrderedJson node(size_t i, bool described) const {
        const Node& n = r.nodes[i];
        OrderedJson body;
        if (n.kind == "object") {
            body = OrderedJson{{"type", "object"}, {"additionalProperties", false}};
            if (!n.patch) {
                OrderedJson req = OrderedJson::array();
                for (size_t c : n.children)
                    if (!r.nodes[c].optional) req.push_back(r.nodes[c].key);
                body["required"] = req;
            }
            OrderedJson props = OrderedJson::object();
            for (size_t c : n.children) props[r.nodes[c].key] = node(c, true);
            body["properties"] = props;
            if (n.min_properties) body["minProperties"] = *n.min_properties;
            OrderedJson all = OrderedJson::array();
            if (n.path == "/spec/calibration/[]") calibration_rules(&all);
            if (n.constraints)
                for (const Json& c : *n.constraints) all.push_back(OrderedJson::parse(c.dump()));
            if (!all.empty()) body["allOf"] = all;
        } else if (n.kind == "map") {
            body = OrderedJson{{"type", "object"}};
            OrderedJson req = OrderedJson::array();
            for (const std::string& k : n.required_keys) req.push_back(k);
            body["required"] = req;
            if (n.min_properties) body["minProperties"] = *n.min_properties;
            body["propertyNames"] = ref(n.key_ref);
            body["additionalProperties"] = node(n.children.at(0), true);
        } else if (n.kind == "array") {
            body = OrderedJson{{"type", "array"}, {"items", node(n.children.at(0), false)}};
            if (n.min_items) body["minItems"] = *n.min_items;
            if (n.max_items) body["maxItems"] = *n.max_items;
            if (n.unique_items) body["uniqueItems"] = true;
        } else if (n.kind == "tuple") {
            OrderedJson pre = OrderedJson::array();
            for (const std::string& t : n.tuple) pre.push_back(ref(t));
            body = OrderedJson{{"type", "array"}, {"prefixItems", pre}, {"items", false},
                               {"minItems", n.tuple.size()}, {"maxItems", n.tuple.size()}};
        } else {
            body = scalar(n);
        }
        if (n.nullable) body = OrderedJson{{"oneOf", OrderedJson::array({body, OrderedJson{{"type", "null"}}})}};
        if (!described || n.doc.empty()) return body;
        OrderedJson out = OrderedJson{{"description", n.doc}};
        for (auto& [k, v] : body.items()) out[k] = v;
        return out;
    }

    /* Each calibration key's value type, as an if/then on the key. */
    void calibration_rules(OrderedJson* all) const {
        for (const CalibrationKey& c : r.calibration) {
            const Node& v = r.nodes[c.value];
            OrderedJson value = scalar(v);
            all->push_back(OrderedJson{
                {"if", OrderedJson{{"properties", OrderedJson{{"key", OrderedJson{{"const", c.key}}}}},
                                   {"required", OrderedJson::array({"key"})}}},
                {"then", OrderedJson{{"properties", OrderedJson{{"value", value}}}}}});
        }
    }

    OrderedJson document() const {
        OrderedJson s = OrderedJson{{"$schema", r.schema_dialect}, {"$id", r.schema_id}, {"title", r.title}};
        const OrderedJson root = node(r.root, true);
        for (const auto& [k, v] : root.items()) s[k] = v;
        OrderedJson defs = OrderedJson::object();
        for (const Def& d : r.defs) defs[d.name] = node(d.node, true);
        s["$defs"] = defs;
        return s;
    }
};

/* Remove annotations from a schema, visiting only subschema positions (JSON Schema 2020-12 applicators): literal
 * instance data under const, enum, required and the like is never entered, so changing it changes the fingerprint. */
const std::set<std::string> kAnnotations = {"title", "description", "$comment", "examples", "default", "deprecated",
                                            "readOnly", "writeOnly"};
void strip_annotations(Json* schema) {
    if (!schema->is_object()) return;   // a boolean schema
    for (const std::string& k : kAnnotations) schema->erase(k);
    for (auto& [k, v] : schema->items()) {
        if (k == "properties" || k == "$defs" || k == "patternProperties" || k == "dependentSchemas") {
            for (auto& [name, sub] : v.items()) strip_annotations(&sub);
        } else if (k == "allOf" || k == "anyOf" || k == "oneOf" || k == "prefixItems") {
            for (Json& sub : v) strip_annotations(&sub);
        } else if (k == "items" || k == "additionalProperties" || k == "propertyNames" || k == "not" || k == "if" ||
                   k == "then" || k == "else" || k == "contains" || k == "unevaluatedItems" ||
                   k == "unevaluatedProperties") {
            strip_annotations(&v);
        }
    }
}

// ---- C++ -------------------------------------------------------------------------------------------------------------

std::string literal(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out + "\"";
}

/* `enum class Name : uint8_t {...};` and its wire-name table, one entry per line once a single line would pass 120. */
void enum_decl(std::ostringstream& o, const std::string& name, const std::vector<std::string>& wire) {
    std::string one = "enum class " + name + " : uint8_t {";
    for (size_t i = 0; i < wire.size(); ++i) one += (i ? ", " : " ") + enumerator(wire[i]);
    one += " };";
    std::string names = "inline constexpr std::array<std::string_view, " + std::to_string(wire.size()) + "> k" + name +
                        "Names = {";
    std::string flat = names;
    for (size_t i = 0; i < wire.size(); ++i) flat += (i ? ", " : "") + literal(wire[i]);
    flat += "};";
    if (one.size() <= 120) {
        o << one << "\n";
    } else {
        o << "enum class " << name << " : uint8_t {\n";
        for (const std::string& w : wire) o << "    " << enumerator(w) << ",\n";
        o << "};\n";
    }
    if (flat.size() <= 120) {
        o << flat << "\n";
    } else {
        o << names << "\n";
        for (const std::string& w : wire) o << "    " << literal(w) << ",\n";
        o << "};\n";
    }
}

/* Wrap `text` as comment lines with the given prefix, at most 120 columns. */
void comment(std::ostringstream& o, const std::string& indent, const std::string& text) {
    std::istringstream words(text);
    std::string w, line;
    while (words >> w) {
        if (!line.empty() && indent.size() + 3 + line.size() + 1 + w.size() > 120) {
            o << indent << "// " << line << "\n";
            line.clear();
        }
        line += (line.empty() ? "" : " ") + w;
    }
    if (!line.empty()) o << indent << "// " << line << "\n";
}

struct TypesWriter {
    const Registry& r;

    const Def& def(const std::string& name) const {
        for (const Def& d : r.defs)
            if (d.name == name) return d;
        throw std::logic_error("def " + name);
    }

    std::string base(const Node& n) const {
        if (n.kind == "object") return n.cpp;
        if (n.kind == "ref") {
            const Def& d = def(n.ref);
            if (d.kind == "object" || d.kind == "patch") return d.cpp;
            return base(r.nodes[d.node]);
        }
        if (n.kind == "integer" || n.kind == "int_enum") return "int64_t";
        if (n.kind == "string") return "std::string";
        if (n.kind == "boolean") return "bool";
        if (n.kind == "enum") return n.cpp_type;
        if (n.kind == "calibration_key") return "CalibrationKey";
        if (n.kind == "calibration_value") return "CalibrationValue";
        if (n.kind == "array") return "std::vector<" + type(r.nodes[n.children.at(0)]) + ">";
        if (n.kind == "map") return "std::map<std::string, " + type(r.nodes[n.children.at(0)]) + ">";
        if (n.kind == "tuple") {
            for (const std::string& t : n.tuple)
                if (r.nodes[def(t).node].kind != "integer") throw std::runtime_error(n.path + ": tuples hold integers");
            return "std::array<int64_t, " + std::to_string(n.tuple.size()) + ">";
        }
        throw std::logic_error("kind " + n.kind);
    }

    std::string type(const Node& n) const {
        std::string t = base(n);
        if (n.nullable) t = "std::optional<" + t + ">";
        if (n.optional) t = "std::optional<" + t + ">";
        return t;
    }

    void enums(std::ostringstream& o) const {
        o << "// Calibrated parameters (contract [[calibration_key]]), in wire order.\n";
        std::vector<std::string> keys;
        for (const CalibrationKey& c : r.calibration) keys.push_back(c.key);
        enum_decl(o, "CalibrationKey", keys);
        o << "\n";
        std::set<std::string> done;
        for (const Node& n : r.nodes) {
            if (n.kind != "enum" || n.path.rfind("calibration:", 0) == 0 || !done.insert(n.cpp_type).second) continue;
            comment(o, "", n.cpp_type + ": " + n.doc);
            std::vector<std::string> wire;
            for (const Json& v : n.values) wire.push_back(v.get<std::string>());
            enum_decl(o, n.cpp_type, wire);
            o << "\n";
        }
    }

    void structs(std::ostringstream& o) const {
        std::set<std::string> done;
        std::function<void(size_t)> emit = [&](size_t i) {
            const Node& n = r.nodes[i];
            for (size_t c : n.children) emit(c);
            if (n.kind != "object" || !done.insert(n.cpp).second) return;
            comment(o, "", (n.path.empty() ? std::string("(document)") : n.path) + ": " + n.doc);
            o << "struct " << n.cpp << " {\n";
            for (size_t c : n.children) {
                const Node& m = r.nodes[c];
                comment(o, "    ", m.doc);
                o << "    " << type(m) << " " << (m.cpp_member.empty() ? m.key : m.cpp_member) << ";\n";
            }
            o << "};\n\n";
        };
        for (const Def& d : r.defs)
            if (d.kind == "object" || d.kind == "patch") emit(d.node);
        emit(r.root);
    }

    void fields(std::ostringstream& o) const {
        size_t count = 0;
        std::ostringstream rows;
        for (const Node& n : r.nodes) {
            if (n.path.rfind("calibration:", 0) == 0 || n.doc.empty() || n.patch) continue;
            rows << "    {" << literal(n.path) << ", " << literal(n.kind) << ", " << literal(n.cls) << ", "
                 << (n.optional ? "false" : "true") << ", " << (n.nullable ? "true" : "false") << ",\n     "
                 << literal(n.radiance_option) << ", " << literal(n.radiance_config) << ",\n     " << literal(n.doc)
                 << "},\n";
            ++count;
        }
        o << "// Every registry field (patch copies excepted), in registry order: the metadata explain and provenance use.\n";
        o << "struct FieldInfo {\n"
             "    std::string_view path;             // JSON Pointer, or \"$defs/<name>/...\" inside a shared type\n"
             "    std::string_view kind;\n"
             "    std::string_view value_class;      // provenance class of the subtree (contract [[class_default]])\n"
             "    bool required;\n"
             "    bool nullable;\n"
             "    std::string_view radiance_option;  // the engine flag the field binds to, or empty\n"
             "    std::string_view radiance_config;  // its native Config destination, or empty\n"
             "    std::string_view doc;\n"
             "};\n";
        o << "inline constexpr std::array<FieldInfo, " << count << "> kFields = {{\n" << rows.str() << "}};\n";
    }
};

std::string md_cell(std::string s) {
    std::string out;
    for (char c : s) {
        if (c == '|') out += "\\|";
        else if (c == '\n') out += ' ';
        else out += c;
    }
    return out;
}

struct FieldsWriter {
    const Registry& r;
    TypesWriter types{r};

    std::string describe(const Node& n) const {
        std::string t;
        if (n.kind == "ref") t = "`" + n.ref + "`";
        else if (n.kind == "object") t = "object `" + n.cpp + "`";
        else if (n.kind == "map") t = "map `" + n.key_ref + "` → " + describe(r.nodes[n.children.at(0)]);
        else if (n.kind == "array") {
            t = "array of " + describe(r.nodes[n.children.at(0)]);
            if (n.min_items || n.max_items)
                t += " (" + std::to_string(n.min_items.value_or(0)) + "–" +
                     (n.max_items ? std::to_string(*n.max_items) : std::string("")) + ")";
            if (n.unique_items) t += ", unique";
        } else if (n.kind == "tuple") {
            t = "[";
            for (size_t i = 0; i < n.tuple.size(); ++i) t += (i ? ", " : "") + ("`" + n.tuple[i] + "`");
            t += "]";
        } else if (n.kind == "enum" || n.kind == "int_enum") {
            t = "one of ";
            for (size_t i = 0; i < n.values.size(); ++i) t += (i ? ", " : "") + ("`" + n.values[i].dump() + "`");
        } else if (n.kind == "calibration_key") t = "a calibration key";
        else if (n.kind == "calibration_value") t = "integer, string or boolean, by key";
        else if (n.constant) t = "exactly `" + n.constant->dump() + "`";
        else {
            t = n.kind;
            if (n.minimum || n.maximum || n.exclusive_maximum) {
                t += " " + (n.minimum ? std::to_string(*n.minimum) : std::string("")) + "..";
                if (n.maximum) t += std::to_string(*n.maximum);
                if (n.exclusive_maximum) t += "<" + std::to_string(*n.exclusive_maximum);
            }
            if (n.min_length) t += ", length ≥ " + std::to_string(*n.min_length);
            if (!n.pattern.empty()) t += ", `" + n.pattern + "`";
            if (!n.format.empty()) t += ", " + n.format;
        }
        if (n.nullable) t += " or null";
        return t;
    }

    void rows(std::ostringstream& o, size_t i) const {
        const Node& n = r.nodes[i];
        if (!n.doc.empty() && !n.path.empty()) {
            const std::string binding = n.radiance_option.empty()
                                            ? std::string()
                                            : "`" + n.radiance_option + "` / " + n.radiance_config;
            o << "| `" << md_cell(n.path) << "` | " << md_cell(describe(n)) << " | " << (n.optional ? "" : "yes")
              << " | " << n.cls << " | " << md_cell(binding) << " | " << md_cell(n.doc) << " |\n";
        }
        for (size_t c : n.children) rows(o, c);
    }

    std::string write() const {
        std::ostringstream o;
        o << "<!-- Generated by contract/generate.cpp from contract/serving-v1.toml. Do not edit. -->\n";
        o << "# Serving manifest " << r.api_version << " fields\n\n";
        o << "Contract fingerprint `" << fingerprint(r) << "`. The specification is "
             "`docs/design/serving-manifest-v1/spec.md`; every value below is an integer, string, boolean, array or "
             "object (no non-integer numbers). Required = the member must be present; class = the provenance class of "
             "the subtree; every value's lifecycle is `" << r.lifecycle << "`.\n\n";
        const std::string head = "| Path | Type | Required | Class | Radiance | Description |\n|---|---|---|---|---|---|\n";
        o << "## Document\n\n" << head;
        rows(o, r.root);
        o << "\n## Shared types (`$defs`)\n\n";
        for (const Def& d : r.defs) {
            const Node& n = r.nodes[d.node];
            o << "### `" << d.name << "`\n\n" << md_cell(d.doc) << " ";
            if (d.kind == "patch") o << "Every member of `" << d.of << "`, each optional; a present group is nonempty.\n\n";
            else if (d.kind == "object") o << "C++ `" << d.cpp << "`.\n\n" << head;
            else o << "Type: " << md_cell(describe(n)) << ".\n\n";
            if (d.kind == "object") {
                for (size_t c : n.children) rows(o, c);
                o << "\n";
            }
        }
        o << "## Calibration keys\n\n| Key | Value | Description |\n|---|---|---|\n";
        for (const CalibrationKey& c : r.calibration)
            o << "| `" << c.key << "` | " << md_cell(describe(r.nodes[c.value])) << " | " << md_cell(c.doc) << " |\n";
        return o.str();
    }
};

}  // namespace

bool load_registry(const std::string& toml_text, const std::string& source_name, Registry* out, std::string* why) {
    try {
        toml::table doc;
        try { doc = toml::parse(toml_text, source_name); }
        catch (const toml::parse_error& e) {
            std::ostringstream m;
            m << e.description() << " (line " << e.source().begin.line << ")";
            fault(source_name, m.str());
        }
        refuse_floats(doc, source_name);
        only_keys(doc, {"contract", "class_default", "calibration_key", "def", "field"}, source_name);
        Loader l;
        const auto* c = doc["contract"].as_table();
        if (!c) fault(source_name, "missing [contract]");
        only_keys(*c, {"api_version", "kind", "contract_version", "schema_dialect", "schema_id", "title", "namespace",
                       "lifecycle"},
                  "[contract]");
        l.r.api_version = str(*c, "api_version", "[contract]");
        l.r.kind = str(*c, "kind", "[contract]");
        l.r.schema_dialect = str(*c, "schema_dialect", "[contract]");
        l.r.schema_id = str(*c, "schema_id", "[contract]");
        l.r.title = str(*c, "title", "[contract]");
        l.r.ns = str(*c, "namespace", "[contract]");
        {
            std::string part;
            size_t parts = 0;
            for (size_t at = 0, next; (next = l.r.ns.find("::", at)) != std::string::npos || at <= l.r.ns.size();) {
                part = l.r.ns.substr(at, next == std::string::npos ? std::string::npos : next - at);
                if (!identifier(part)) fault("[contract]", "namespace must be identifiers joined by ::, is " + l.r.ns);
                ++parts;
                if (next == std::string::npos) break;
                at = next + 2;
            }
            if (!parts) fault("[contract]", "empty namespace");
        }
        l.r.lifecycle = str(*c, "lifecycle", "[contract]");
        const auto version = integer(*c, "contract_version", "[contract]");
        if (!version || *version < 1) fault("[contract]", "contract_version is a positive integer");
        l.r.contract_version = *version;
        if (l.r.api_version != "bigcherry/v" + std::to_string(*version))
            fault("[contract]", "api_version must end in the contract_version");
        auto each = [&](const char* name, auto&& fn) {
            const auto* a = doc[name].as_array();
            if (!a) return;
            for (size_t i = 0; i < a->size(); ++i) {
                const std::string at = std::string(name) + "[" + std::to_string(i) + "]";
                const auto* t = a->get(i)->as_table();
                if (!t) fault(at, "must be a table");
                fn(*t, at);
            }
        };
        each("calibration_key", [&](const toml::table& t, const std::string& at) { l.calibration_key(t, at); });
        each("def", [&](const toml::table& t, const std::string& at) { l.def(t, at); });
        each("field", [&](const toml::table& t, const std::string& at) { l.field(t, at); });
        l.finish();
        l.classes(doc["class_default"].as_array());
        *out = std::move(l.r);
        return true;
    } catch (const std::exception& e) {
        if (why) *why = e.what();
        return false;
    }
}

std::string emit_schema(const Registry& r) {
    return SchemaWriter{r}.document().dump(2) + "\n";
}

namespace {
Json validation_schema(const Registry& r) {
    Json schema = Json::parse(SchemaWriter{r}.document().dump());
    strip_annotations(&schema);
    return schema;
}

std::string jcs_sha256(const Json& j, const char* what) {
    std::string bytes, why;
    if (!bc::serving::jcs_canonicalize(j, &bytes, &why)) throw std::runtime_error(std::string(what) + " JCS: " + why);
    return "sha256:" + bc::serving::sha256_hex(bytes);
}
}  // namespace

std::string fingerprint(const Registry& r) { return jcs_sha256(validation_schema(r), "schema"); }

std::string semantic_fingerprint(const Registry& r) {
    // The validation schema plus everything else the contract freezes that the schema does not express: each field's
    // provenance class and lifecycle, its Radiance binding, the calibration key order. C++
    // names and docs are excluded: renaming generated code is not a wire change, and a consumer would not compile.
    Json fields = Json::array();
    for (const Node& n : r.nodes) {
        if (n.path.rfind("calibration:", 0) == 0) continue;
        fields.push_back(Json{{"path", n.path}, {"class", n.cls}, {"radiance_option", n.radiance_option},
                              {"radiance_config", n.radiance_config}});
    }
    Json keys = Json::array();
    for (const CalibrationKey& c : r.calibration) keys.push_back(c.key);
    return jcs_sha256(Json{{"schema", validation_schema(r)}, {"lifecycle", r.lifecycle}, {"fields", fields},
                           {"calibration_keys", keys}},
                      "semantics");
}

std::string emit_types(const Registry& r) {
    TypesWriter t{r};
    std::ostringstream o;
    o << "// Generated by contract/generate.cpp from contract/serving-v1.toml. Do not edit.\n"
         "// The typed model of the serving manifest "
      << r.api_version
      << " (docs/design/serving-manifest-v1).\n"
         "// Integers are int64_t (every v1 number is an integer within +/-(2^53 - 1)). An optional member is\n"
         "// std::optional, a nullable one std::optional, and one that is both std::optional<std::optional<T>>.\n"
         "#pragma once\n"
         "#include <array>\n#include <cstdint>\n#include <map>\n#include <optional>\n#include <string>\n"
         "#include <string_view>\n#include <variant>\n#include <vector>\n\n";
    o << "namespace " << r.ns << " {\n\n";
    o << "inline constexpr std::string_view kApiVersion = " << literal(r.api_version) << ";\n";
    o << "inline constexpr std::string_view kKind = " << literal(r.kind) << ";\n";
    o << "inline constexpr int64_t kContractVersion = " << r.contract_version << ";\n";
    o << "// sha256 of the JCS validation schema without annotations (contract/README.md); the build pins it.\n";
    o << "inline constexpr std::string_view kFingerprint = " << literal(fingerprint(r)) << ";\n";
    o << "// sha256 of the schema plus the frozen semantics it does not express (classes, lifecycle, Radiance bindings);\n"
         "// README.md.\n";
    o << "inline constexpr std::string_view kSemanticFingerprint = " << literal(semantic_fingerprint(r)) << ";\n\n";
    t.enums(o);
    o << "// A calibration value: an integer, a string or a boolean; each key fixes which (contract [[calibration_key]]).\n";
    o << "using CalibrationValue = std::variant<int64_t, std::string, bool>;\n\n";
    t.structs(o);
    t.fields(o);
    o << "\n}  // namespace " << r.ns << "\n";
    return o.str();
}

std::string emit_fields_md(const Registry& r) {
    return FieldsWriter{r}.write();
}

std::map<std::string, std::string> generate_all(const Registry& r) {
    return {{"serving-v1.schema.json", emit_schema(r)},
            {"serving_types.h", emit_types(r)},
            {"serving-fields.md", emit_fields_md(r)},
            {"contract.fingerprint", "schema " + fingerprint(r) + "\nsemantics " + semantic_fingerprint(r) + "\n"}};
}

}  // namespace bc::contract
