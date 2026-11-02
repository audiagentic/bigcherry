#pragma once
/* The serving manifest v1 field registry (contract/serving-v1.toml) and what is generated from it: the JSON Schema, the
 * typed C++ model, the field reference and the contract fingerprint. See contract/README.md for the registry format.
 *
 * GPU-free and independent of bc_serving, so the generator builds before the library that will consume its output. */
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace bc::contract {

using OrderedJson = nlohmann::ordered_json;

/* One schema node: a [[field]], a scalar [[def]], an object [[def]]'s root, or an array's synthetic item. */
struct Node {
    std::string path;                 // "/spec/intent", "$defs/helperCommon/cpu", ".../[]", ".../*"
    std::string key;                  // member name within the parent ("" for the root, "[]" items, "*" map values)
    std::string kind;                 // object, map, array, integer, string, boolean, enum, int_enum, ref,
                                      // calibration_key, calibration_value, tuple
    bool optional = false;            // not in the parent's "required"
    bool nullable = false;            // may be JSON null
    std::string doc;
    std::string cpp;                  // object nodes: the struct name
    std::string cpp_type;             // enum nodes: the enum class name
    std::string cpp_member;           // member name when the JSON name is not a C++ identifier
    std::string ref;                  // ref nodes: the def name
    std::optional<nlohmann::json> constant;
    std::vector<nlohmann::json> values;          // enum / int_enum
    std::optional<int64_t> minimum, maximum, exclusive_maximum, min_length, max_length, min_items, max_items,
        min_properties;
    std::string pattern, format;
    bool unique_items = false;
    std::vector<std::string> tuple;              // tuple nodes: each element's def name
    std::string key_ref;                         // map nodes: the def their keys must satisfy
    std::vector<std::string> required_keys;      // map nodes
    std::optional<nlohmann::json> constraints;   // object nodes: extra allOf rules, verbatim
    std::string radiance_option, radiance_config;
    std::string cls;                             // provenance class (resolved from [[class_default]])
    std::vector<size_t> children;                // object members in order; arrays: their one item; maps: the value
    bool patch = false;                          // inside a patch def: every member optional, objects nonempty
    size_t patch_of = SIZE_MAX;                  // patch nodes: the node they copy
    std::string error_class;                     // the decoder's code for this subtree's faults ("" = by fault)
    std::string mismatch;                        // const nodes: the code when the value differs ("" = invalid_value)
};

struct Def {
    std::string name, kind, doc, cpp, of;        // kind: a scalar kind, "object" or "patch" (of = the def it patches)
    size_t node = SIZE_MAX;                      // the def's root node
};

struct CalibrationKey {
    std::string key, doc;
    size_t value = SIZE_MAX;                     // node describing the value's type
};

struct Registry {
    std::string api_version, kind, schema_dialect, schema_id, title, ns, lifecycle;
    int64_t contract_version = 0;
    std::vector<Node> nodes;                     // nodes[root] is the document
    size_t root = SIZE_MAX;
    std::vector<Def> defs;                       // in file order
    std::vector<CalibrationKey> calibration;
    std::map<std::string, size_t> by_path;
    int64_t max_depth = 0;   // the deepest legal nesting of JSON containers (the root object is 1)
};

/* Parse and check a registry. False with `why` naming the first fault (and its path) on: any non-integer number, an
 * unknown table or key, a missing doc, a duplicate or orphaned path, a child before its parent, an unknown kind, def or
 * class, a "number" type, a non-identifier C++ name, duplicate C++ type names, or kinds and keys that do not fit. */
bool load_registry(const std::string& toml_text, const std::string& source_name, Registry* out, std::string* why);

/* The JSON Schema 2020-12 document, canonical key order, two-space indent, trailing newline. */
std::string emit_schema(const Registry& r);
/* "sha256:" + SHA-256 of the RFC 8785 JCS bytes of the schema with every annotation (title, description, $comment,
 * examples, default, deprecated, readOnly, writeOnly) removed from schema positions only: documentation edits do not
 * change it; any change to what validates does, including literal const/enum data. */
std::string fingerprint(const Registry& r);
/* "sha256:" + SHA-256 of the JCS of the annotation-free schema plus the semantics the schema does not express: every
 * field's provenance class, the lifecycle, the Radiance bindings, the reader's error codes and the calibration keys.
 * The closed-v1 identity: either fingerprint changing is a bigcherry/v2 change. */
std::string semantic_fingerprint(const Registry& r);
/* serving_types.h: the typed model (namespace from [contract]), its enums with their wire names, the field table. */
std::string emit_types(const Registry& r);
/* serving-fields.md: the field reference. */
std::string emit_fields_md(const Registry& r);
/* serving_codec.h / serving_codec.cpp: one strict decoder per struct (v1/error.h codes: unknown, missing, type,
 * value, by subtree error_class), and apply() for each patch def (spec §2.2: present leaves replace, objects recurse). */
std::string emit_codec_h(const Registry& r);
std::string emit_codec_cpp(const Registry& r);

/* Every generated file, by its name in generated/. */
std::map<std::string, std::string> generate_all(const Registry& r);

}  // namespace bc::contract
