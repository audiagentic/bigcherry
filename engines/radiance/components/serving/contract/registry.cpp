#include "registry.h"

#include "jcs.h"
#include "sha256.h"
#include "third_party/toml.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <optional>
#include <regex>
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
/* v1/error.h codes a registry may assign to a subtree or a const mismatch. */
const std::set<std::string> kErrorClasses = {"unsupported_version", "invalid_value", "provenance_invalid",
                                             "calibration_invalid", "ownership_mismatch", "plan_mismatch"};
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
    n->max_length = integer(t, k("max_length").c_str(), at);
    n->pattern = str(t, k("pattern").c_str(), at, false);
    n->format = str(t, k("format").c_str(), at, false);
    if (const auto* c = t.get(k("const"))) n->constant = to_json(*c, at + "." + k("const"));
    n->values = list(t, k("values").c_str(), at);
    n->ref = str(t, k("ref").c_str(), at, false);
}

/* YYYY-MM-DD naming a real day, leap years included: the rule the generated reader applies to a date at run time. */
bool default_is_calendar_date(const std::string& s) {
    if (s.size() != 10 || s[4] != '-' || s[7] != '-') return false;
    for (size_t i : {0u, 1u, 2u, 3u, 5u, 6u, 8u, 9u})
        if (s[i] < '0' || s[i] > '9') return false;
    const int y = std::stoi(s.substr(0, 4)), m = std::stoi(s.substr(5, 2)), d = std::stoi(s.substr(8, 2));
    static const int kDays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (m < 1 || m > 12 || d < 1) return false;
    const bool leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    return d <= kDays[m - 1] + (m == 2 && leap ? 1 : 0);
}

/* A default is a literal of the field's own kind that its schema accepts, from a named place (spec section 7). */
void check_default(const Node& field, const Node& n, const std::string& at) {
    // `field` carries the default; `n` is the node whose kind and bounds it must fit (the field itself, or the scalar def it refs)
    if (!field.default_value) return;
    const Json& d = *field.default_value;
    if (n.kind == "integer") {
        if (!d.is_number_integer()) fault(at, "an integer default must be an integer");
        const int64_t v = d.get<int64_t>();
        if ((n.minimum && v < *n.minimum) || (n.maximum && v > *n.maximum) ||
            (n.exclusive_maximum && v >= *n.exclusive_maximum))
            fault(at, "the default " + d.dump() + " is outside the field's numeric bounds");
    } else if (n.kind == "boolean") {
        if (!d.is_boolean()) fault(at, "a boolean default must be true or false");
    } else if (n.kind == "string") {
        if (!d.is_string()) fault(at, "a string default must be a string");
        const std::string& text = d.get_ref<const std::string&>();
        // length is counted in characters, as JSON Schema counts it (UTF-8 continuation bytes are not characters)
        int64_t len = 0;
        for (unsigned char c : text) len += (c & 0xC0) != 0x80;
        if ((n.min_length && len < *n.min_length) || (n.max_length && len > *n.max_length))
            fault(at, "the default is outside the field's length bounds");
        if (!n.pattern.empty() && !std::regex_search(text, std::regex(n.pattern)))
            fault(at, "the default " + d.dump() + " does not match the field's pattern");
        if (n.format == "date" && !default_is_calendar_date(text))
            fault(at, "the default " + d.dump() + " is not a date (YYYY-MM-DD)");
    } else if (n.kind == "enum" || n.kind == "int_enum") {
        if (std::find(n.values.begin(), n.values.end(), d) == n.values.end())
            fault(at, "the default " + d.dump() + " is not one of the field's values");
    } else {
        fault(at, std::string("kind '") + n.kind + "' cannot have a default");
    }
    // the default must be what the schema accepts: a const field has one value
    if (const Json* c = field.constant ? &*field.constant : (n.constant ? &*n.constant : nullptr))
        if (d != *c) fault(at, "the default " + d.dump() + " is not the field's const " + c->dump());
    const std::string& s = field.default_source;
    if (s.rfind("radiance:Config.", 0) == 0) {
        if (field.radiance_config != s.substr(std::string("radiance:").size()))
            fault(at, "default_source " + s + " must name the field's own radiance_config (" +
                          (field.radiance_config.empty() ? std::string("none") : field.radiance_config) + ")");
    } else if (s.rfind("setting:R3_", 0) != 0 || s.size() <= std::string("setting:R3_").size()) {
        fault(at, "default_source is \"radiance:Config.<member>\" or \"setting:R3_<NAME>\"");
    }
}

/* What each kind may carry; anything else set on it is a fault. */
void check_scalar(const Node& n, const std::string& at) {
    const bool num = n.minimum || n.maximum || n.exclusive_maximum;
    const bool text = n.min_length || n.max_length || !n.pattern.empty() || !n.format.empty();
    auto refuse = [&](bool bad, const char* what) {
        if (bad) fault(at, std::string("kind '") + n.kind + "' cannot have " + what);
    };
    if (n.kind == "number") fault(at, "kind 'number': the v1 contract has integers only");
    if (n.min_length && *n.min_length < 0) fault(at, "min_length is not negative");
    // spec §11 amendment 7: std::regex recurses per repetition, so a pattern only ever sees a bounded string
    if (!n.pattern.empty() && !n.max_length) fault(at, "a pattern needs a max_length (spec §11 amendment 7)");
    if (!n.pattern.empty() && n.max_length && *n.max_length > 1024)
        fault(at, "a pattern's max_length is at most 1024 characters (4,096 UTF-8 bytes, 5x under the measured-safe "
                  "20,000; spec §11 amendment 7)");
    if (n.max_length && (*n.max_length < 1 || (n.min_length && *n.min_length > *n.max_length)))
        fault(at, "max_length is positive and not below min_length");
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
    if (n.radiance_switch && (n.kind != "enum" || n.values.size() != 1 || n.values[0] != "on"))
        fault(at, "a radiance_switch is an enum with the one value \"on\" (presence is the flag)");
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
                      "values", "minimum", "maximum", "exclusive_maximum", "min_length", "max_length", "pattern", "format",
                      "item_kind", "item_ref", "item_min_length", "item_max_length", "item_minimum", "item_values",
                      "item_pattern",
                      "min_items", "max_items", "unique_items", "tuple", "key_ref", "required_keys", "min_properties",
                      "constraints", "radiance_option", "radiance_config", "radiance_switch", "error_class", "mismatch", "default",
                      "default_source"},
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
        n.radiance_switch = boolean(t, "radiance_switch", where);
        if (n.radiance_switch && n.radiance_option.empty()) fault(where, "radiance_switch needs a radiance_option");
        if (!n.cpp_member.empty() && !identifier(n.cpp_member)) fault(where, "cpp_member is not a C++ identifier");
        n.error_class = str(t, "error_class", where, false);
        n.mismatch = str(t, "mismatch", where, false);
        if (!n.error_class.empty() && !kErrorClasses.count(n.error_class))
            fault(where, "unknown error_class '" + n.error_class + "'");
        if (!n.mismatch.empty() && !kErrorClasses.count(n.mismatch)) fault(where, "unknown mismatch '" + n.mismatch + "'");
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
        if (const auto* d = t.get("default")) n.default_value = to_json(*d, where + ".default");
        n.default_source = str(t, "default_source", where, false);
        if (n.default_value.has_value() != !n.default_source.empty())
            fault(where, "default and default_source go together: a default names the place it comes from");
        if (!container) {
            scalar_attrs(t, "", where, &n);
            check_scalar(n, where);
        } else {
            if (n.default_value) fault(where, "a container cannot have a default");
            for (const char* k : {"minimum", "maximum", "exclusive_maximum", "min_length", "max_length", "pattern",
                                  "format", "const", "values", "ref"})
                if (t.get(k)) fault(where, std::string("a container cannot have '") + k + "'");
        }
        if (n.kind != "array")
            for (const char* k : {"item_kind", "item_ref", "item_min_length", "item_max_length", "item_minimum",
                                  "item_values", "item_pattern", "tuple"})
                if (t.get(k)) fault(where, std::string("only arrays have '") + k + "'");
        if (n.kind == "array" && !n.cpp_type.empty() && str(t, "item_kind", where) != "enum")
            fault(where, "cpp_type on an array names its enum items");
        if (n.kind != "map" && (t.get("key_ref") || t.get("required_keys"))) fault(where, "only maps have keys");
        if (!n.mismatch.empty() && !n.constant) fault(where, "mismatch is for const fields");
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
            for (const char* k : {"item_ref", "item_min_length", "item_max_length", "item_minimum", "item_values",
                                  "item_pattern", "tuple"})
                if (t.get(k)) fault(at, std::string("an object item is its own [[field]]; '") + k + "' does not apply");
            return;   // its [[field]] must follow; checked in finish()
        }
        Node item;
        item.path = r.nodes[array].path + "/[]";
        item.kind = item_kind;
        if (item_kind == "tuple") {
            for (const char* k : {"item_ref", "item_min_length", "item_max_length", "item_minimum", "item_values",
                                  "item_pattern"})
                if (t.get(k)) fault(at, std::string("a tuple item takes only 'tuple'; '") + k + "' does not apply");
            if (!cpp_type.empty()) fault(at, "a tuple item has no cpp_type");
            item.tuple = strings(t, "tuple", at);
            if (item.tuple.empty()) fault(at, "a tuple item needs its element defs");
        } else {
            if (t.get("tuple")) fault(at, "'tuple' is for tuple items");
            Node probe;
            scalar_attrs(t, "item_", at, &probe);
            item.minimum = probe.minimum; item.maximum = probe.maximum; item.min_length = probe.min_length;
            item.max_length = probe.max_length;
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
            only_keys(t, {"name", "kind", "doc", "minimum", "maximum", "exclusive_maximum", "min_length", "max_length",
                          "pattern", "format"},
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
                      "value_exclusive_maximum", "value_min_length", "value_max_length", "value_pattern",
                      "value_values"},
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
            n.patch_of = c;
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
            r.nodes[d.node].patch_of = it->node;
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
        const std::pair<const char*, Json> identity[] = {{"/apiVersion", r.api_version},
                                                          {"/kind", r.kind},
                                                          {"/metadata/contract_version", r.contract_version}};
        for (const auto& [path, want] : identity) {
            const auto it = r.by_path.find(path);
            if (it == r.by_path.end() || !r.nodes[it->second].constant || *r.nodes[it->second].constant != want)
                fault(path, "must be a const equal to the [contract] identity " + want.dump());
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

    /* spec §11 amendment 8: the closed schema bounds how deep a valid document nests; a container is one level, a ref
     * to an object def continues into it, and a ref that reaches itself makes the bound undefined, so it is refused. */
    int64_t depth(size_t i, std::set<std::string>* visiting, std::map<size_t, int64_t>* memo) {
        if (const auto it = memo->find(i); it != memo->end()) return it->second;
        const Node& n = r.nodes[i];
        int64_t d = 0;
        if (n.kind == "ref") {
            const auto def = std::find_if(r.defs.begin(), r.defs.end(), [&](const Def& x) { return x.name == n.ref; });
            if (def != r.defs.end() && (def->kind == "object" || def->kind == "patch")) {
                if (!visiting->insert(def->name).second)
                    fault(n.path.empty() ? "\"\"" : n.path, "ref '" + n.ref + "' reaches itself: the nesting depth would be unbounded");
                d = depth(def->node, visiting, memo);
                visiting->erase(def->name);
            }
        } else if (n.kind == "object" || n.kind == "map" || n.kind == "array") {
            int64_t inner = 0;
            for (size_t c : n.children) inner = std::max(inner, depth(c, visiting, memo));
            d = 1 + inner;
        } else if (n.kind == "tuple") {
            d = 1;
        }
        (*memo)[i] = d;
        return d;
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
        r.class_defaults = rules;
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
        OrderedJson s = scalar_body(n);
        if (n.default_value && !n.patch) s["default"] = OrderedJson::parse(n.default_value->dump());
        return s;
    }

    OrderedJson scalar_body(const Node& n) const {
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
        if (n.max_length) s["maxLength"] = *n.max_length;
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
                                            "readOnly", "writeOnly", "contentEncoding", "contentMediaType"};
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
                   k == "unevaluatedProperties" || k == "contentSchema") {
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
                 << literal(n.radiance_option) << ", " << literal(n.radiance_config) << ", " << (n.radiance_switch ? "true" : "false")
                 << ",\n     " << literal(n.doc)
                 << ", " << literal(n.default_value ? n.default_value->dump() : std::string()) << ", "
                 << literal(n.default_source) << ", " << literal(n.ref) << "},\n";
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
             "    bool radiance_switch;              // the flag takes no value: the field's presence is the flag (an enum with the one value \"on\")\n"
             "    std::string_view doc;\n"
             "    std::string_view default_json;     // the schema default as JSON, or empty (spec section 7)\n"
             "    std::string_view default_source;   // where the default comes from, or empty\n"
             "    std::string_view ref_def;          // kind \"ref\": the shared type it uses, else empty\n"
             "};\n";
        o << "inline constexpr std::array<FieldInfo, " << count << "> kFields = {{\n" << rows.str() << "}};\n";
        // A patch def copies the members of the object def it patches: a field below one is described by the copied member.
        std::ostringstream patches;
        size_t patch_count = 0;
        for (const Def& d : r.defs)
            if (d.kind == "patch") {
                patches << "    {" << literal(d.name) << ", " << literal(d.of) << "},\n";
                ++patch_count;
            }
        // A value's provenance class follows where it sits in the document, not the shared type it uses: the longest
        // matching position prefix wins ("[]" stands for an array index).
        std::ostringstream classes;
        for (const auto& [path, cls] : r.class_defaults) classes << "    {" << literal(path) << ", " << literal(cls) << "},\n";
        o << "\n// [[class_default]] rules: the provenance class of everything at or under a document position.\n";
        o << "inline constexpr std::array<std::array<std::string_view, 2>, " << r.class_defaults.size()
          << "> kClassDefaults = {{\n" << classes.str() << "}};\n";
        o << "\n// The patch defs and the object def each one patches (spec section 2.2).\n";
        o << "inline constexpr std::array<std::array<std::string_view, 2>, " << patch_count << "> kPatchDefs = {{\n"
          << patches.str() << "}};\n";
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
            if (n.max_length) t += ", length ≤ " + std::to_string(*n.max_length);
            if (!n.pattern.empty()) t += ", `" + n.pattern + "`";
            if (!n.format.empty()) t += ", " + n.format;
        }
        if (n.nullable) t += " or null";
        if (n.default_value) t += "; default `" + n.default_value->dump() + "` (" + n.default_source + ")";
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

// ---- the codec -------------------------------------------------------------------------------------------------------

struct CodecWriter {
    const Registry& r;
    TypesWriter types{r};
    std::map<size_t, std::string> cls;   // node -> effective error_class

    void classes_from(size_t i, const std::string& inherited) {
        const std::string c = r.nodes[i].error_class.empty() ? inherited : r.nodes[i].error_class;
        cls[i] = c;
        for (size_t k : r.nodes[i].children) classes_from(k, c);
    }
    CodecWriter(const Registry& reg) : r(reg) {
        classes_from(r.root, "");
        for (const Def& d : r.defs) classes_from(d.node, "");
        const std::string cal = cls.at(r.by_path.at("/spec/calibration/[]/value"));
        for (const CalibrationKey& k : r.calibration) cls[k.value] = cal;
    }

    std::string code(size_t i, const char* fallback) const {
        const std::string& c = cls.at(i);
        return "C::" + (c.empty() ? std::string(fallback) : c);
    }
    std::string mismatch(size_t i) const {
        const Node& n = r.nodes[i];
        if (!n.mismatch.empty()) return "C::" + n.mismatch;
        return code(i, "invalid_value");
    }

    const Def& def(const std::string& name) const { return types.def(name); }
    bool object_ref(const Node& n) const {
        if (n.kind != "ref") return false;
        const Def& d = def(n.ref);
        return d.kind == "object" || d.kind == "patch";
    }
    /* The C++ type a node's decoder writes (its member type without optional wrapping). */
    std::string value_type(size_t i) const {
        const Node& n = r.nodes[i];
        if (n.path.rfind("calibration:", 0) == 0) {
            if (n.kind == "enum" || n.kind == "string") return "std::string";
            if (n.kind == "boolean") return "bool";
            if (n.kind == "ref") return types.base(r.nodes[def(n.ref).node]);
            return "int64_t";
        }
        return types.base(n);
    }
    /* The call that decodes node i from `j` into `out`. */
    std::string call(size_t i, const std::string& j, const std::string& at, const std::string& out) const {
        const Node& n = r.nodes[i];
        if (n.kind == "object" || object_ref(n)) return "decode(" + j + ", " + at + ", " + out + ")";
        return "d" + std::to_string(i) + "(" + j + ", " + at + ", " + out + ")";
    }
    static std::string raw(const std::string& s) { return "R\"re(" + s + ")re\""; }

    /* Scalar checks for node i, using the constraints of `spec` (i itself, or the scalar def it refers to). */
    void scalar_body(std::ostringstream& o, size_t i, const Node& spec) const {
        const std::string tc = code(i, "invalid_type"), vc = code(i, "invalid_value");
        if (spec.kind == "integer") {
            if (spec.constant) {
                o << "    if (!j.is_number_integer()) return type_fault(" << tc << ", at, \"an integer\", j);\n"
                  << "    if (j.get<int64_t>() != " << spec.constant->get<int64_t>() << ") return fault(" << mismatch(i)
                  << ", at, \"must be " << spec.constant->get<int64_t>() << ", is \" + j.dump());\n"
                  << "    *out = j.get<int64_t>();\n    return std::nullopt;\n";
                return;
            }
            auto opt = [](const std::optional<int64_t>& v) { return v ? std::to_string(*v) : std::string("std::nullopt"); };
            o << "    return int_in(j, at, " << tc << ", " << vc << ", " << opt(spec.minimum) << ", " << opt(spec.maximum)
              << ", " << opt(spec.exclusive_maximum) << ", out);\n";
        } else if (spec.kind == "string") {
            if (spec.constant) {
                o << "    if (!j.is_string()) return type_fault(" << tc << ", at, \"a string\", j);\n"
                  << "    if (j.get_ref<const std::string&>() != " << literal(spec.constant->get<std::string>())
                  << ") return fault(" << mismatch(i) << ", at, \"must be \\\"" << spec.constant->get<std::string>()
                  << "\\\", is \" + j.dump());\n"
                  << "    *out = j.get<std::string>();\n    return std::nullopt;\n";
                return;
            }
            if (!spec.pattern.empty())
                o << "    static const std::regex re(" << raw(spec.pattern) << ", std::regex::ECMAScript);\n";
            o << "    return str_in(j, at, " << tc << ", " << vc << ", "
              << (spec.min_length ? std::to_string(*spec.min_length) : std::string("-1")) << ", "
              << (spec.max_length ? std::to_string(*spec.max_length) : std::string("-1")) << ", "
              << (spec.pattern.empty() ? std::string("nullptr") : std::string("&re")) << ", "
              << (spec.pattern.empty() ? std::string("\"\"") : literal(spec.pattern)) << ", "
              << (spec.format == "date" ? "true" : "false") << ", out);\n";
        } else if (spec.kind == "boolean") {
            o << "    if (!j.is_boolean()) return type_fault(" << tc << ", at, \"a boolean\", j);\n"
              << "    *out = j.get<bool>();\n    return std::nullopt;\n";
        } else if (spec.kind == "enum" && r.nodes[i].path.rfind("calibration:", 0) == 0) {
            o << "    return choice_in(j, at, " << tc << ", " << vc << ", {";
            for (size_t k = 0; k < spec.values.size(); ++k) o << (k ? ", " : "") << literal(spec.values[k].get<std::string>());
            o << "}, out);\n";
        } else if (spec.kind == "enum") {
            o << "    return enum_in(j, at, " << tc << ", " << vc << ", k" << spec.cpp_type << "Names, out);\n";
        } else if (spec.kind == "int_enum") {
            o << "    return int_enum_in(j, at, " << tc << ", " << vc << ", {";
            for (size_t k = 0; k < spec.values.size(); ++k) o << (k ? ", " : "") << spec.values[k].get<int64_t>();
            o << "}, out);\n";
        } else if (spec.kind == "calibration_key") {
            o << "    return enum_in(j, at, " << tc << ", " << vc << ", kCalibrationKeyNames, out);\n";
        } else {
            throw std::logic_error("scalar kind " + spec.kind);
        }
    }

    void node_fn(std::ostringstream& o, size_t i) const {
        const Node& n = r.nodes[i];
        const std::string tc = code(i, "invalid_type"), vc = code(i, "invalid_value");
        o << "Status d" << i << "(const Json& j, const std::string& at, " << value_type(i) << "* out) {  // "
          << (n.path.empty() ? "\"\"" : n.path) << "\n";
        if (n.kind == "array") {
            const size_t item = n.children.at(0);
            o << "    if (!j.is_array()) return type_fault(" << tc << ", at, \"an array\", j);\n";
            if (n.min_items && *n.min_items > 0)
                o << "    if (j.size() < " << *n.min_items << ") return fault(" << vc << ", at, \"needs at least "
                  << *n.min_items << " item(s), has \" + std::to_string(j.size()));\n";
            if (n.max_items)
                o << "    if (j.size() > " << *n.max_items << ") return fault(" << vc << ", at, \"allows at most "
                  << *n.max_items << " item(s), has \" + std::to_string(j.size()));\n";
            // every item is decoded (and so validated) before uniqueness, which is a set lookup per item
            o << "    out->clear();\n    out->reserve(j.size());\n"
              << "    for (size_t k = 0; k < j.size(); ++k) {\n        " << value_type(item) << " v{};\n"
              << "        if (auto e = " << call(item, "j[k]", "index(at, k)", "&v") << ") return e;\n"
              << "        out->push_back(std::move(v));\n    }\n";
            if (n.unique_items)
                o << "    std::map<std::string, size_t> seen;\n"
                     "    for (size_t k = 0; k < j.size(); ++k)\n"
                     "        if (const auto [it, fresh] = seen.emplace(j[k].dump(), k); !fresh)\n"
                     "            return fault("
                  << code(i, "duplicate_field") << ", index(at, k), \"repeats item \" + std::to_string(it->second));\n";
            o << "    return std::nullopt;\n";
        } else if (n.kind == "tuple") {
            o << "    if (!j.is_array()) return type_fault(" << tc << ", at, \"an array\", j);\n"
              << "    if (j.size() != " << n.tuple.size() << ") return fault(" << vc << ", at, \"must have exactly "
              << n.tuple.size() << " items, has \" + std::to_string(j.size()));\n";
            for (size_t k = 0; k < n.tuple.size(); ++k)
                o << "    if (auto e = d" << def(n.tuple[k]).node << "(j[" << k << "], index(at, " << k << "), &(*out)["
                  << k << "])) return e;\n";
            o << "    return std::nullopt;\n";
        } else if (n.kind == "map") {
            const size_t value = n.children.at(0);
            const Node& key = r.nodes[def(n.key_ref).node];
            o << "    if (!j.is_object()) return type_fault(" << tc << ", at, \"an object\", j);\n";
            for (const std::string& k : n.required_keys)
                o << "    if (!j.contains(" << literal(k) << ")) return fault(" << code(i, "missing_field")
                  << ", keyed(at, " << literal(k) << "), \"missing required entry\");\n";
            if (n.min_properties)
                o << "    if (j.size() < " << *n.min_properties << ") return fault(" << vc << ", at, \"needs at least "
                  << *n.min_properties << " entr(ies)\");\n";
            if (!key.pattern.empty()) o << "    static const std::regex re(" << raw(key.pattern) << ", std::regex::ECMAScript);\n";
            o << "    out->clear();\n    for (const auto& [k, v] : j.items()) {\n";
            if (key.max_length)
                o << "        if (characters(k) > " << *key.max_length << ") return fault(" << vc
                  << ", keyed(at, k.substr(0, 64) + \"...\"), \"key is \" + std::to_string(characters(k)) + "
                     "\" characters, longer than " << *key.max_length << "\");\n";
            if (!key.pattern.empty())
                o << "        if (!std::regex_search(k, re)) return fault(" << vc
                  << ", keyed(at, k), \"key is not a valid " << n.key_ref << "\");\n";
            o << "        " << value_type(value) << " x{};\n"
              << "        if (auto e = " << call(value, "v", "keyed(at, k)", "&x") << ") return e;\n"
              << "        (*out)[k] = std::move(x);\n    }\n    return std::nullopt;\n";
        } else if (n.kind == "ref") {
            scalar_body(o, i, r.nodes[def(n.ref).node]);
        } else {
            scalar_body(o, i, n);
        }
        o << "}\n\n";
    }

    void struct_fn(std::ostringstream& o, size_t i) const {
        const Node& n = r.nodes[i];
        const std::string tc = code(i, "invalid_type"), vc = code(i, "invalid_value"), uc = code(i, "unknown_field"),
                          mc = code(i, "missing_field");
        o << "Status decode(const Json& j, const std::string& at, " << n.cpp << "* out) {  // "
          << (n.path.empty() ? "\"\"" : n.path) << "\n";
        o << "    *out = " << n.cpp << "{};   // the result replaces the destination; nothing stale survives\n";
        o << "    if (auto e = closed(j, at, " << tc << ", " << uc << ", {";
        for (size_t k = 0; k < n.children.size(); ++k) o << (k ? ", " : "") << literal(r.nodes[n.children[k]].key);
        o << "})) return e;\n";
        if (n.min_properties)
            o << "    if (j.size() < " << *n.min_properties << ") return fault(" << vc << ", at, "
              << (n.patch ? "\"an override group must not be empty\"" : "\"must not be empty\"") << ");\n";
        bool key_seen = false;
        for (size_t c : n.children) {
            const Node& m = r.nodes[c];
            const std::string name = m.cpp_member.empty() ? m.key : m.cpp_member;
            const std::string at = "member(at, " + literal(m.key) + ")";
            o << "    if (const auto it = j.find(" << literal(m.key) << "); it == j.end()) {\n";
            if (!m.optional) o << "        return fault(" << mc << ", " << at << ", \"missing required member\");\n";
            o << "    } else";
            if (m.nullable)
                o << " if (it->is_null()) {\n        out->" << name
                  << (m.optional ? ".emplace(std::nullopt);\n" : " = std::nullopt;\n") << "    } else";
            o << " {\n";
            if (m.kind == "calibration_key") key_seen = true;
            if (m.kind == "calibration_value") {
                if (!key_seen) throw std::runtime_error(m.path + ": the calibration value must follow its key");
                o << "        switch (out->key) {\n";
                for (size_t k = 0; k < r.calibration.size(); ++k) {
                    const size_t v = r.calibration[k].value;
                    o << "        case CalibrationKey::" << enumerator(r.calibration[k].key) << ": {\n"
                      << "            " << value_type(v) << " v{};\n"
                      << "            if (auto e = d" << v << "(*it, " << at << ", &v)) return e;\n"
                      << "            out->" << name << " = std::move(v);\n            break;\n        }\n";
                }
                o << "        }\n    }\n";
                continue;
            }
            o << "        " << value_type(c) << " v{};\n"
              << "        if (auto e = " << call(c, "*it", at, "&v") << ") return e;\n"
              << "        out->" << name << (m.optional && m.nullable ? ".emplace(std::move(v));\n" : " = std::move(v);\n")
              << "    }\n";
        }
        o << "    return std::nullopt;\n}\n\n";
    }

    /* A C++ expression giving the JSON of node i's value `v` (its base type; optional wrapping handled by the caller). */
    std::string enc(size_t i, const std::string& v, int depth = 0) const {
        const Node& n = r.nodes[i];
        const std::string x = "x" + std::to_string(depth);
        if (n.kind == "object" || object_ref(n)) return "encode(" + v + ")";
        if (n.kind == "enum") return "Json(k" + n.cpp_type + "Names[static_cast<size_t>(" + v + ")])";
        if (n.kind == "calibration_key") return "Json(kCalibrationKeyNames[static_cast<size_t>(" + v + ")])";
        if (n.kind == "calibration_value") return "std::visit([](const auto& cv) { return Json(cv); }, " + v + ")";
        if (n.kind == "array")
            return "[&] { Json a = Json::array(); for (const auto& " + x + " : " + v + ") a.push_back(" +
                   enc(n.children.at(0), x, depth + 1) + "); return a; }()";
        if (n.kind == "map")
            return "[&] { Json o = Json::object(); for (const auto& [k" + std::to_string(depth) + ", " + x + "] : " + v +
                   ") o[k" + std::to_string(depth) + "] = " + enc(n.children.at(0), x, depth + 1) + "; return o; }()";
        return "Json(" + v + ")";   // integers, strings, booleans, int enums, tuples
    }

    std::vector<size_t> structs() const {
        std::vector<size_t> out;
        std::set<std::string> seen;
        std::function<void(size_t)> walk = [&](size_t i) {
            for (size_t c : r.nodes[i].children) walk(c);
            if (r.nodes[i].kind == "object" && seen.insert(r.nodes[i].cpp).second) out.push_back(i);
        };
        for (const Def& d : r.defs)
            if (d.kind == "object" || d.kind == "patch") walk(d.node);
        walk(r.root);
        return out;
    }
    std::vector<size_t> scalars() const {
        std::vector<size_t> out;
        std::set<size_t> seen;
        std::function<void(size_t)> walk = [&](size_t i) {
            for (size_t c : r.nodes[i].children) walk(c);
            const Node& n = r.nodes[i];
            if (n.kind != "object" && !object_ref(n) && n.kind != "calibration_value" && seen.insert(i).second)
                out.push_back(i);
        };
        for (const Def& d : r.defs) walk(d.node);
        walk(r.root);
        for (const CalibrationKey& k : r.calibration) out.push_back(k.value);
        return out;
    }

    std::string header() const {
        std::ostringstream o;
        o << "// Generated by contract/generate.cpp from contract/serving-v1.toml. Do not edit.\n"
             "// Strict decoders for the typed model (serving_types.h): a closed object refuses an unknown member, a\n"
             "// required one must be present, every value has its schema type and constraints. Faults use v1/error.h\n"
             "// codes (a subtree's registry error_class replaces the per-fault code) and the human dotted path.\n"
             "// Cross-field rules (contract `constraints`) and semantics are v1/manifest.cpp's.\n"
             "#pragma once\n#include \"serving_types.h\"\n#include \"v1/error.h\"\n\n#include <string>\n\n"
             "#include <nlohmann/json.hpp>\n\n";
        o << "namespace " << r.ns << " {\n\n";
        for (size_t i : structs()) {
            o << "Status decode(const nlohmann::json& j, const std::string& path, " << r.nodes[i].cpp << "* out);\n";
        }
        o << "\n// The writer: each struct's JSON (absent optionals omitted, nullable nulls written as null).\n";
        for (size_t i : structs()) o << "nlohmann::json encode(const " << r.nodes[i].cpp << "& value);\n";
        o << "\n// Patches (spec §2.2): every present leaf replaces the base's; present groups recurse.\n";
        for (size_t i : structs())
            if (r.nodes[i].patch)
                o << "void apply(" << r.nodes[r.nodes[i].patch_of].cpp << "& base, const " << r.nodes[i].cpp
                  << "& patch);\n";
        o << "\n}  // namespace " << r.ns << "\n";
        return o.str();
    }

    std::string source() const {
        std::ostringstream o;
        o << "// Generated by contract/generate.cpp from contract/serving-v1.toml. Do not edit.\n"
             "#include \"serving_codec.h\"\n\n#include <map>\n#include <optional>\n#include <regex>\n\n";
        o << "namespace " << r.ns << " {\nnamespace {\n\n"
          << R"cpp(using Json = nlohmann::json;
using C = ConfigErrorCode;
const char* const kHint = "regenerate the manifest with bc-serve; a v1 manifest is never edited by hand";

std::string member(const std::string& at, const std::string& key) { return at.empty() ? key : at + "." + key; }
std::string index(const std::string& at, size_t i) { return at + "[" + std::to_string(i) + "]"; }
std::string keyed(const std::string& at, const std::string& key) { return at + "[" + Json(key).dump() + "]"; }

const char* kind_of(const Json& j) {
    switch (j.type()) {
        case Json::value_t::null: return "null";
        case Json::value_t::boolean: return "a boolean";
        case Json::value_t::number_integer:
        case Json::value_t::number_unsigned: return "an integer";
        case Json::value_t::number_float: return "a non-integer number";
        case Json::value_t::string: return "a string";
        case Json::value_t::array: return "an array";
        case Json::value_t::object: return "an object";
        default: return "an unsupported value";
    }
}

ConfigError fault(C c, const std::string& at, std::string message) { return make_error(c, at, std::move(message), kHint); }
Status type_fault(C c, const std::string& at, const char* want, const Json& j) {
    return fault(c, at, std::string("expected ") + want + ", found " + kind_of(j));
}

Status closed(const Json& j, const std::string& at, C tc, C uc, std::initializer_list<std::string_view> known) {
    if (!j.is_object()) return type_fault(tc, at, "an object", j);
    for (const auto& [k, v] : j.items()) {
        bool ok = false;
        for (std::string_view name : known) ok = ok || name == k;
        if (!ok) return fault(uc, member(at, k), "unknown member");
    }
    return std::nullopt;
}

Status int_in(const Json& j, const std::string& at, C tc, C vc, std::optional<int64_t> min, std::optional<int64_t> max,
              std::optional<int64_t> below, int64_t* out) {
    if (!j.is_number_integer()) return type_fault(tc, at, "an integer", j);
    const int64_t v = j.get<int64_t>();
    if (min && v < *min) return fault(vc, at, std::to_string(v) + " is below the minimum " + std::to_string(*min));
    if (max && v > *max) return fault(vc, at, std::to_string(v) + " is above the maximum " + std::to_string(*max));
    if (below && v >= *below) return fault(vc, at, std::to_string(v) + " must be below " + std::to_string(*below));
    *out = v;
    return std::nullopt;
}

bool calendar_date(const std::string& s) {
    if (s.size() != 10 || s[4] != '-' || s[7] != '-') return false;
    for (size_t i : {0u, 1u, 2u, 3u, 5u, 6u, 8u, 9u})
        if (s[i] < '0' || s[i] > '9') return false;
    const int y = std::stoi(s.substr(0, 4)), m = std::stoi(s.substr(5, 2)), d = std::stoi(s.substr(8, 2));
    static const int kDays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (m < 1 || m > 12 || d < 1) return false;
    const bool leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    return d <= kDays[m - 1] + (m == 2 && leap ? 1 : 0);
}

int64_t characters(const std::string& s) {
    int64_t n = 0;
    for (unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
}

Status str_in(const Json& j, const std::string& at, C tc, C vc, int64_t min_length, int64_t max_length,
              const std::regex* re, const char* pattern, bool date, std::string* out) {
    if (!j.is_string()) return type_fault(tc, at, "a string", j);
    const std::string& v = j.get_ref<const std::string&>();
    if (min_length >= 0 && static_cast<int64_t>(v.size()) < min_length)
        return fault(vc, at, min_length == 1 ? "must not be empty" : "is shorter than " + std::to_string(min_length));
    // before the pattern: std::regex recurses per repetition, so it only sees bounded strings (amendment 7); like JSON
    // Schema, the bound counts characters (the input is valid UTF-8, so code points are the non-continuation bytes)
    if (max_length >= 0 && characters(v) > max_length)
        return fault(vc, at, "is " + std::to_string(characters(v)) + " characters, longer than " + std::to_string(max_length));
    if (re && !std::regex_search(v, *re)) return fault(vc, at, Json(v).dump() + " does not match " + pattern);
    if (date && !calendar_date(v)) return fault(vc, at, Json(v).dump() + " is not a YYYY-MM-DD date");
    *out = v;
    return std::nullopt;
}

template <class E, size_t N>
Status enum_in(const Json& j, const std::string& at, C tc, C vc, const std::array<std::string_view, N>& names, E* out) {
    if (!j.is_string()) return type_fault(tc, at, "a string", j);
    const std::string& v = j.get_ref<const std::string&>();
    for (size_t i = 0; i < N; ++i)
        if (names[i] == v) {
            *out = static_cast<E>(i);
            return std::nullopt;
        }
    std::string allowed;
    for (size_t i = 0; i < N; ++i) allowed += (i ? ", " : "") + std::string(names[i]);
    return fault(vc, at, Json(v).dump() + " is not one of " + allowed);
}

Status choice_in(const Json& j, const std::string& at, C tc, C vc, std::initializer_list<std::string_view> names,
                 std::string* out) {
    if (!j.is_string()) return type_fault(tc, at, "a string", j);
    const std::string& v = j.get_ref<const std::string&>();
    std::string allowed;
    for (std::string_view n : names) {
        if (n == v) {
            *out = v;
            return std::nullopt;
        }
        allowed += (allowed.empty() ? "" : ", ") + std::string(n);
    }
    return fault(vc, at, Json(v).dump() + " is not one of " + allowed);
}

Status int_enum_in(const Json& j, const std::string& at, C tc, C vc, std::initializer_list<int64_t> values, int64_t* out) {
    if (!j.is_number_integer()) return type_fault(tc, at, "an integer", j);
    const int64_t v = j.get<int64_t>();
    std::string allowed;
    for (int64_t a : values) {
        if (a == v) {
            *out = v;
            return std::nullopt;
        }
        allowed += (allowed.empty() ? "" : ", ") + std::to_string(a);
    }
    return fault(vc, at, std::to_string(v) + " is not one of " + allowed);
}

)cpp";
        const std::vector<size_t> sc = scalars();
        for (size_t i : sc) o << "[[maybe_unused]] Status d" << i << "(const Json& j, const std::string& at, " << value_type(i) << "* out);\n";
        o << "\n";
        for (size_t i : sc) node_fn(o, i);
        o << "}  // namespace\n\n";
        for (size_t i : structs()) struct_fn(o, i);
        for (size_t i : structs()) {
            const Node& n = r.nodes[i];
            o << "nlohmann::json encode(const " << n.cpp << "& value) {\n    Json j = Json::object();\n";
            for (size_t c : n.children) {
                const Node& m = r.nodes[c];
                const std::string name = "value." + (m.cpp_member.empty() ? m.key : m.cpp_member);
                const std::string key = literal(m.key);
                if (m.optional && m.nullable)
                    o << "    if (" << name << ") j[" << key << "] = *" << name << " ? " << enc(c, "**" + name)
                      << " : Json(nullptr);\n";
                else if (m.optional)
                    o << "    if (" << name << ") j[" << key << "] = " << enc(c, "*" + name) << ";\n";
                else if (m.nullable)
                    o << "    j[" << key << "] = " << name << " ? " << enc(c, "*" + name) << " : Json(nullptr);\n";
                else
                    o << "    j[" << key << "] = " << enc(c, name) << ";\n";
            }
            o << "    return j;\n}\n\n";
        }
        for (size_t i : structs()) {
            const Node& n = r.nodes[i];
            if (!n.patch) continue;
            o << "void apply(" << r.nodes[n.patch_of].cpp << "& base, const " << n.cpp << "& patch) {\n";
            for (size_t c : n.children) {
                const Node& m = r.nodes[c];
                const std::string name = m.cpp_member.empty() ? m.key : m.cpp_member;
                if (m.kind == "object")
                    o << "    if (patch." << name << ") apply(base." << name << ", *patch." << name << ");\n";
                else
                    o << "    if (patch." << name << ") base." << name << " = *patch." << name << ";\n";
            }
            o << "}\n\n";
        }
        o << "}  // namespace " << r.ns << "\n";
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
        for (const Node& n : l.r.nodes) {   // every default fits the scalar it is a default of (a ref field: the def's scalar)
            if (!n.default_value) continue;
            const std::string where = n.path.empty() ? "\"\"" : n.path;
            if (n.kind == "ref") {
                const auto def = std::find_if(l.r.defs.begin(), l.r.defs.end(), [&](const Def& x) { return x.name == n.ref; });
                if (def == l.r.defs.end()) fault(where, "unknown ref '" + n.ref + "'");
                check_default(n, l.r.nodes[def->node], where);
            } else {
                check_default(n, n, where);
            }
        }
        {
            std::set<std::string> visiting;
            std::map<size_t, int64_t> memo;
            for (const Def& d : l.r.defs)
                if (d.kind == "object" || d.kind == "patch") {
                    visiting.insert(d.name);
                    l.depth(d.node, &visiting, &memo);
                    visiting.erase(d.name);
                }
            memo.clear();
            l.r.max_depth = l.depth(l.r.root, &visiting, &memo);
        }
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

// ---- defaults against their sources ----------------------------------------------------------------------------------

namespace {
std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

/* The length of the raw-string prefix (R, u8R, uR, UR, LR) right before the quote at src[quote], or 0 when the quote does
 * not open a raw string (the R must not be the tail of a longer identifier). */
size_t raw_prefix_length(const std::string& src, size_t quote) {
    if (quote == 0 || src[quote - 1] != 'R') return 0;
    size_t len = 1;
    for (const char* prefix : {"u8", "u", "U", "L"}) {
        const size_t n = std::char_traits<char>::length(prefix);
        if (quote >= 1 + n && src.compare(quote - 1 - n, n, prefix) == 0) { len = 1 + n; break; }
    }
    if (quote > len) {
        const unsigned char before = static_cast<unsigned char>(src[quote - len - 1]);
        if (std::isalnum(before) || before == '_') return 0;
    }
    return len;
}

/* The header as code only. Comments become spaces (newlines kept). A string literal's text moves to `literals` and the
 * code keeps only `"<index>"` in the quotes, so a `;`, `=`, brace or declaration inside a literal is never read as code,
 * while a legitimate string initialiser can still be looked up. A character literal is emptied to its quotes. A digit
 * separator (1'000) is not a character literal, and a raw string R"d(...)d" is a string literal. */
std::string code_only(const std::string& src, std::vector<std::string>* literals) {
    std::string out;
    out.reserve(src.size());
    auto keep = [&](std::string body) {
        literals->push_back(std::move(body));
        return "\"" + std::to_string(literals->size() - 1) + "\"";
    };
    for (size_t i = 0; i < src.size();) {
        const char c = src[i];
        if (src.compare(i, 2, "//") == 0) {
            while (i < src.size() && src[i] != '\n') { out += ' '; ++i; }
        } else if (src.compare(i, 2, "/*") == 0) {
            const size_t end = src.find("*/", i + 2);
            const size_t stop = end == std::string::npos ? src.size() : end + 2;
            for (; i < stop; ++i) out += src[i] == '\n' ? '\n' : ' ';
        } else if (c == '"' && raw_prefix_length(src, i) > 0) {
            // raw string: R"delim( ... )delim" (optionally u8R, uR, UR or LR)
            const size_t open = src.find('(', i + 1);
            const std::string delim = open == std::string::npos ? std::string() : src.substr(i + 1, open - i - 1);
            const size_t close = open == std::string::npos ? std::string::npos : src.find(")" + delim + "\"", open + 1);
            if (close == std::string::npos) { out += c; ++i; continue; }
            out.resize(out.size() - raw_prefix_length(src, i));   // the R and its encoding prefix
            out += keep(src.substr(open + 1, close - open - 1));
            for (size_t k = i; k < close + delim.size() + 2; ++k) if (src[k] == '\n') out += '\n';
            i = close + delim.size() + 2;
        } else if (c == '"') {
            std::string body;
            ++i;
            while (i < src.size() && src[i] != '"' && src[i] != '\n') {
                if (src[i] == '\\' && i + 1 < src.size()) body += src[i++];
                body += src[i++];
            }
            if (i < src.size() && src[i] == '"') ++i;
            // only a plain literal (no escapes) is usable as a value; any other is kept as text no initialiser matches
            out += keep(body.find('\\') == std::string::npos ? body : std::string("\\"));
        } else if (c == '\'' && i > 0 && std::isalnum(static_cast<unsigned char>(src[i - 1])) && i + 1 < src.size() &&
                   std::isalnum(static_cast<unsigned char>(src[i + 1]))) {
            ++i;   // a digit separator inside a number: dropped, so 1'000 reads as 1000
        } else if (c == '\'') {
            ++i;
            while (i < src.size() && src[i] != '\'' && src[i] != '\n') { if (src[i] == '\\') ++i; ++i; }
            if (i < src.size() && src[i] == '\'') ++i;
            out += "''";
        } else {
            out += c;
            ++i;
        }
    }
    return out;
}

/* The text between the braces of `struct Config {`, as code only. */
std::string config_body(const std::string& source, std::vector<std::string>* literals) {
    const std::string header = code_only(source, literals);
    const size_t at = header.find("struct Config {");
    if (at == std::string::npos) throw std::runtime_error("no `struct Config {` in core/rad_core.h");
    size_t i = header.find('{', at), depth = 0;
    const size_t start = i + 1;
    for (; i < header.size(); ++i) {
        if (header.compare(i, 2, "/*") == 0) { i = header.find("*/", i); if (i == std::string::npos) break; ++i; continue; }
        if (header.compare(i, 2, "//") == 0) { i = header.find('\n', i); if (i == std::string::npos) break; continue; }
        if (header[i] == '{') ++depth;
        else if (header[i] == '}' && --depth == 0) return header.substr(start, i - start);
    }
    throw std::runtime_error("unterminated `struct Config` in core/rad_core.h");
}

/* The simple literal an initialiser is: an integer, true/false, or a plain string. Anything else is no source. */
std::optional<Json> simple_literal(std::string text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) text.pop_back();
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) text.erase(text.begin());
    if (text == "true") return Json(true);
    if (text == "false") return Json(false);
    if (text.size() >= 2 && text.front() == '"' && text.back() == '"' && text.find('"', 1) == text.size() - 1 &&
        text.find('\\') == std::string::npos)
        return Json(text.substr(1, text.size() - 2));
    size_t i = text[0] == '-' ? 1 : 0;
    if (i >= text.size()) return std::nullopt;
    for (size_t k = i; k < text.size(); ++k)
        if (!std::isdigit(static_cast<unsigned char>(text[k]))) return std::nullopt;
    try { return Json(static_cast<int64_t>(std::stoll(text))); } catch (...) { return std::nullopt; }
}

/* Does the source value `src` equal the registry default `want` for a field of `kind`? Integers 0/1 stand for a boolean
 * (radiance declares many flags as int). */
bool same_value(const std::string& kind, const Json& want, const Json& src) {
    if (kind == "boolean") {
        if (src.is_boolean()) return src == want;
        if (src.is_number_integer() && (src.get<int64_t>() == 0 || src.get<int64_t>() == 1))
            return (src.get<int64_t>() == 1) == want.get<bool>();
        return false;
    }
    return src == want;
}
}  // namespace

bool verify_defaults(const Registry& r, const std::string& radiance_src, const std::vector<std::string>& component_tomls,
                     std::string* why) {
    try {
        std::string config;   // read once, on first use
        std::vector<std::string> literals;   // the text of the string literals in it, by index
        std::vector<toml::table> settings;
        bool settings_read = false;
        for (const Node& n : r.nodes) {
            if (!n.default_value) continue;
            const std::string at = n.path + " default " + n.default_value->dump() + " from " + n.default_source;
            std::string kind = n.kind;   // a ref field is compared as the scalar def it uses
            if (kind == "ref") {
                const auto def = std::find_if(r.defs.begin(), r.defs.end(), [&](const Def& x) { return x.name == n.ref; });
                if (def != r.defs.end()) kind = r.nodes[def->node].kind;
            }
            Json source;
            if (n.default_source.rfind("radiance:Config.", 0) == 0) {
                if (config.empty()) config = config_body(read_text_file(radiance_src + "/core/rad_core.h"), &literals);
                const std::string member = n.default_source.substr(std::string("radiance:Config.").size());
                const std::regex declared("(?:^|[;{}\\n])[^;{}=()]*[ \\t*&]" + member + R"(\s*=\s*([^;]*);)");
                std::smatch m;
                if (!std::regex_search(config, m, declared))
                    throw std::runtime_error(at + ": Config::" + member + " has no `= <literal>;` initialiser in core/rad_core.h");
                // two readings of the same name are ambiguous: refuse rather than certify the first
                const std::string rest = m.suffix().str();
                if (std::regex_search(rest, declared))
                    throw std::runtime_error(at + ": Config::" + member + " is declared more than once in core/rad_core.h");
                std::string init = m[1].str();
                // a string initialiser is `"<index>"` in the code-only text: give it back its text
                static const std::regex quoted(R"re(^\s*"(\d+)"\s*$)re");
                std::smatch q;
                if (std::regex_match(init, q, quoted) && std::stoul(q[1].str()) < literals.size())
                    init = "\"" + literals[std::stoul(q[1].str())] + "\"";
                const auto lit = simple_literal(init);
                if (!lit) throw std::runtime_error(at + ": the initialiser `" + m[1].str() + "` is not a simple literal; leave the default absent");
                source = *lit;
            } else {
                const std::string name = n.default_source.substr(std::string("setting:").size());
                if (!settings_read) {
                    for (const std::string& f : component_tomls) settings.push_back(toml::parse(read_text_file(f), f));
                    settings_read = true;
                }
                const toml::table* found = nullptr;
                for (const toml::table& doc : settings)
                    if (const auto* s = doc["setting"][name].as_table()) found = s;
                if (!found) throw std::runtime_error(at + ": no [setting." + name + "] in the component files");
                const auto* d = (*found)["default"].as_string();
                if (!d) throw std::runtime_error(at + ": [setting." + name + "] has no string default");
                const std::string text = **d;
                if (kind == "integer" || kind == "int_enum") {
                    const auto lit = simple_literal(text);
                    if (!lit || !lit->is_number_integer()) throw std::runtime_error(at + ": the setting default '" + text + "' is not an integer");
                    source = *lit;
                } else if (kind == "boolean") {
                    if (text == "1" || text == "true") source = true;
                    else if (text == "0" || text == "false") source = false;
                    else throw std::runtime_error(at + ": the setting default '" + text + "' is not 1/0/true/false");
                } else {
                    source = text;
                }
            }
            if (!same_value(kind, *n.default_value, source))
                throw std::runtime_error(at + ": the source says " + source.dump() + " (the registry and its source diverge)");
        }
        return true;
    } catch (const std::exception& e) {
        if (why) *why = e.what();
        return false;
    }
}

std::string semantic_fingerprint(const Registry& r) {
    // The validation schema plus everything else the contract freezes that the schema does not express: each field's
    // provenance class and lifecycle, its Radiance binding, the reader's error codes, the calibration key order. C++
    // names and docs are excluded: renaming generated code is not a wire change, and a consumer would not compile.
    Json fields = Json::array();
    for (const Node& n : r.nodes) {
        if (n.path.rfind("calibration:", 0) == 0) continue;
        Json f = Json{{"path", n.path}, {"class", n.cls}, {"radiance_option", n.radiance_option},
                      {"radiance_config", n.radiance_config}, {"error_class", n.error_class},
                      {"mismatch", n.mismatch}};
        if (n.radiance_switch) f["radiance_switch"] = true;   // present only where a field is a value-less flag
        // spec section 7 / section 11 amendment 9: a default decides which leaves need a provenance record, so it is
        // contract semantics (present only where a field has one)
        if (n.default_value) {
            f["default"] = *n.default_value;
            f["default_source"] = n.default_source;
        }
        fields.push_back(std::move(f));
    }
    Json keys = Json::array();
    for (const CalibrationKey& c : r.calibration) keys.push_back(c.key);
    return jcs_sha256(Json{{"identity", Json{{"api_version", r.api_version}, {"kind", r.kind},
                                             {"contract_version", r.contract_version}}},
                           {"schema", validation_schema(r)}, {"lifecycle", r.lifecycle}, {"fields", fields},
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
    o << "// sha256 of the schema plus the frozen semantics it does not express (classes, lifecycle, Radiance bindings,\n"
         "// reader error codes); README.md.\n";
    o << "inline constexpr std::string_view kSemanticFingerprint = " << literal(semantic_fingerprint(r)) << ";\n";
    o << "// spec §11 amendment 8: the deepest container nesting a valid document can have (the root object is 1),\n"
         "// derived from the closed schema by the generator. The strict reader refuses anything deeper.\n";
    o << "inline constexpr size_t kMaxDepth = " << r.max_depth << ";\n\n";
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

std::string emit_codec_h(const Registry& r) { return CodecWriter(r).header(); }
std::string emit_codec_cpp(const Registry& r) { return CodecWriter(r).source(); }

std::map<std::string, std::string> generate_all(const Registry& r) {
    return {{"serving-v1.schema.json", emit_schema(r)},
            {"serving_codec.h", emit_codec_h(r)},
            {"serving_codec.cpp", emit_codec_cpp(r)},
            {"serving_types.h", emit_types(r)},
            {"serving-fields.md", emit_fields_md(r)},
            {"contract.fingerprint", "schema " + fingerprint(r) + "\nsemantics " + semantic_fingerprint(r) + "\n"}};
}

}  // namespace bc::contract
