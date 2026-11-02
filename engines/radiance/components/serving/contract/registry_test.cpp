// bc_contract_test: the registry loader refuses what the v1 contract forbids, naming the fault, and the emitters keep
// their promises (fingerprint ignores documentation, patch defs are partial, the real registry loads).
#include "registry.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#ifndef BC_CONTRACT_SOURCE_DIR
#error "BC_CONTRACT_SOURCE_DIR must name engines/radiance/components/serving"
#endif

namespace {
int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++failures;
    }
}

const std::string kBase = R"(
[contract]
api_version = "bigcherry/v1"
kind = "ServingManifest"
contract_version = 1
schema_dialect = "https://json-schema.org/draft/2020-12/schema"
schema_id = "https://example.invalid/test.json"
title = "test"
namespace = "bc::serving::test"
lifecycle = "startup_only"

[[class_default]]
path = "/spec"
class = "policy"

[[calibration_key]]
key = "a.b_mib"
value_kind = "ref"
value_ref = "uint"
doc = "A calibrated size."

[[def]]
name = "uint"
kind = "integer"
minimum = 0
doc = "Unsigned."

[[def]]
name = "group"
kind = "object"
cpp = "Group"
doc = "A group."

[[field]]
path = "$defs/group/on"
kind = "boolean"
doc = "On."

[[def]]
name = "groupPatch"
kind = "patch"
of = "group"
cpp = "GroupPatch"
doc = "A partial group."

[[class_default]]
path = "/apiVersion"
class = "envelope"

[[class_default]]
path = "/kind"
class = "envelope"

[[class_default]]
path = "/metadata"
class = "envelope"

[[field]]
path = ""
kind = "object"
cpp = "Doc"
doc = "The document."

[[field]]
path = "/apiVersion"
kind = "string"
const = "bigcherry/v1"
cpp_member = "api_version"
doc = "Identity."

[[field]]
path = "/kind"
kind = "string"
const = "ServingManifest"
doc = "Kind."

[[field]]
path = "/metadata"
kind = "object"
cpp = "Meta"
doc = "Metadata."

[[field]]
path = "/metadata/contract_version"
kind = "integer"
const = 1
doc = "Version."

[[field]]
path = "/spec"
kind = "object"
cpp = "Spec"
doc = "Spec."

[[field]]
path = "/spec/calibration"
kind = "array"
item_kind = "object"
doc = "Calibration."

[[field]]
path = "/spec/calibration/[]"
kind = "object"
cpp = "Cal"
doc = "One entry."

[[field]]
path = "/spec/calibration/[]/key"
kind = "calibration_key"
doc = "Key."

[[field]]
path = "/spec/calibration/[]/value"
kind = "calibration_value"
doc = "Value."

[[field]]
path = "/spec/n"
kind = "ref"
ref = "uint"
doc = "A count."

[[field]]
path = "/spec/patch"
kind = "ref"
ref = "groupPatch"
optional = true
doc = "A patch."
)";

std::string replaced(const std::string& from, const std::string& to) {
    std::string s = kBase;
    const size_t at = s.find(from);
    if (at == std::string::npos) {
        std::printf("FAIL: test fixture lacks '%s'\n", from.c_str());
        ++failures;
        return s;
    }
    return s.replace(at, from.size(), to);
}

void refused(const std::string& text, const std::string& needle, const std::string& what) {
    bc::contract::Registry r;
    std::string why;
    const bool ok = bc::contract::load_registry(text, "test.toml", &r, &why);
    check(!ok && why.find(needle) != std::string::npos, what + ": " + (ok ? "loaded" : why));
}
}  // namespace

int main() {
    bc::contract::Registry r;
    std::string why;
    check(bc::contract::load_registry(kBase, "test.toml", &r, &why), "the base registry loads: " + why);
    if (failures) return 1;
    const auto schema = nlohmann::json::parse(bc::contract::emit_schema(r));
    check(schema["properties"]["spec"]["required"] == nlohmann::json::array({"calibration", "n"}),
          "optional members are not required: " + schema["properties"]["spec"]["required"].dump());
    check(schema["additionalProperties"] == false && schema["properties"]["spec"]["additionalProperties"] == false,
          "objects are closed");
    const auto& patch = schema["$defs"]["groupPatch"];
    check(patch["minProperties"] == 1 && !patch.contains("required") && patch["properties"].contains("on"),
          "a patch def is a nonempty partial copy: " + patch.dump());
    const auto& cal = schema["properties"]["spec"]["properties"]["calibration"]["items"];
    check(cal["properties"]["key"]["enum"] == nlohmann::json::array({"a.b_mib"}) &&
              cal["allOf"][0]["then"]["properties"]["value"]["$ref"] == "#/$defs/uint",
          "calibration keys and their value types come from [[calibration_key]]: " + cal.dump());
    const std::string types = bc::contract::emit_types(r);
    check(types.find("std::optional<GroupPatch> patch;") != std::string::npos &&
              types.find("std::optional<bool> on;") != std::string::npos &&
              types.find("enum class CalibrationKey : uint8_t { kABMib };") != std::string::npos,
          "typed model: optional members, patch members optional, calibration keys");

    // the fingerprint follows what validates, not the documentation
    bc::contract::Registry docs, bound;
    check(bc::contract::load_registry(replaced("doc = \"A count.\"", "doc = \"Another count.\""), "t", &docs, &why),
          "doc edit loads");
    check(bc::contract::load_registry(replaced("minimum = 0\n", "minimum = 1\n"), "t", &bound, &why), "bound edit loads");
    check(bc::contract::fingerprint(docs) == bc::contract::fingerprint(r), "a doc edit keeps the fingerprint");
    check(bc::contract::fingerprint(bound) != bc::contract::fingerprint(r), "a bound edit changes the fingerprint");

    refused(replaced("minimum = 0\n", "minimum = 0.5\n"), "non-integer number", "a float bound");
    refused(replaced("kind = \"integer\"", "kind = \"number\""), "integers only", "a number type");
    refused(replaced("doc = \"A count.\"", "doc = \"A count.\"\nunits = \"MiB\""), "unknown key 'units'", "unknown key");
    refused(replaced("doc = \"A count.\"", ""), "missing 'doc'", "missing doc");
    refused(replaced("doc = \"A count.\"", "doc = \"\""), "empty doc", "empty doc");
    refused(kBase + "\n[[field]]\npath = \"/spec/n\"\nkind = \"boolean\"\ndoc = \"x\"\n", "defined twice", "duplicate path");
    refused(kBase + "\n[[field]]\npath = \"/spec/x/y\"\nkind = \"boolean\"\ndoc = \"x\"\n", "comes before", "orphan");
    refused(replaced("ref = \"uint\"\ndoc = \"A count.\"", "ref = \"nat\"\ndoc = \"A count.\""), "names no def",
            "unknown ref");
    refused(kBase + "\n[[field]]\npath = \"/spec/e\"\nkind = \"enum\"\nvalues = [\"a\"]\ndoc = \"x\"\n", "cpp_type",
            "an enum without a C++ type");
    refused(kBase + "\n[[field]]\npath = \"/spec/e\"\nkind = \"enum\"\nvalues = [\"a\", \"a\"]\ncpp_type = \"E\"\ndoc = \"x\"\n",
            "repeated enum value", "a repeated enum value");
    refused(kBase + "\n[[field]]\npath = \"/spec/class\"\nkind = \"boolean\"\ndoc = \"x\"\n", "cpp_member",
            "a keyword member");
    refused(kBase + "\n[[field]]\npath = \"/spec/o\"\nkind = \"object\"\ncpp = \"Spec\"\ndoc = \"x\"\n", "C++ type Spec",
            "a repeated struct name");
    refused(replaced("cpp = \"Spec\"\ndoc = \"Spec.\"",
                     "cpp = \"Spec\"\nconstraints = '[{\"if\": {\"minimum\": 1.5}}]'\ndoc = \"Spec.\""),
            "non-integer number", "a float inside constraints");
    refused(replaced("api_version = \"bigcherry/v1\"", "api_version = \"bigcherry/v2\""), "contract_version",
            "an api_version that disagrees with contract_version");
    refused(kBase + "\n[[unit]]\nname = \"MiB\"\n", "unknown key 'unit'", "an unknown table");
    refused(kBase + "\n[[field]]\npath = \"/spec/o\"\nkind = \"object\"\nnullable = true\ncpp = \"O\"\ndoc = \"x\"\n",
            "not nullable", "a nullable object");
    refused(replaced("path = \"/spec\"\nclass = \"policy\"", "path = \"/spec\"\nclass = \"vibes\""), "unknown class",
            "an unknown class");
    refused(replaced("value_ref = \"uint\"", "value_ref = \"nat\""), "names no def", "a calibration value of no def");

    // review of #535: the fingerprint follows literal const data inside constraints, not annotations; the semantic
    // fingerprint also follows classes and bindings
    {
        const std::string with_a = replaced("cpp = \"Spec\"\ndoc = \"Spec.\"",
                                            "cpp = \"Spec\"\nconstraints = '[{\"not\": {\"const\": {\"description\": \"A\"}}}]'\ndoc = \"Spec.\"");
        const std::string with_b = replaced("cpp = \"Spec\"\ndoc = \"Spec.\"",
                                            "cpp = \"Spec\"\nconstraints = '[{\"not\": {\"const\": {\"description\": \"B\"}}}]'\ndoc = \"Spec.\"");
        bc::contract::Registry a, b, titled, classed;
        check(bc::contract::load_registry(with_a, "t", &a, &why) && bc::contract::load_registry(with_b, "t", &b, &why),
              "const fixtures load: " + why);
        check(bc::contract::fingerprint(a) != bc::contract::fingerprint(b),
              "a change to object-valued const data changes the fingerprint");
        check(bc::contract::load_registry(replaced("title = \"test\"", "title = \"renamed\""), "t", &titled, &why), "title");
        check(bc::contract::fingerprint(titled) == bc::contract::fingerprint(r) &&
                  bc::contract::semantic_fingerprint(titled) == bc::contract::semantic_fingerprint(r),
              "a title change keeps both fingerprints");
        check(bc::contract::load_registry(replaced("class = \"policy\"", "class = \"derived\""), "t", &classed, &why), "class");
        check(bc::contract::fingerprint(classed) == bc::contract::fingerprint(r) &&
                  bc::contract::semantic_fingerprint(classed) != bc::contract::semantic_fingerprint(r),
              "a class change keeps the schema fingerprint and changes the semantic one");
    }
    refused(replaced("kind = \"ServingManifest\"\ncontract_version", "kind = \"ChangedManifest\"\ncontract_version"),
            "[contract] identity", "a [contract] kind that is not the document's /kind");
    {
        bc::contract::Registry ca, cb;
        check(bc::contract::load_registry(replaced("cpp = \"Spec\"\ndoc = \"Spec.\"",
                                                   "cpp = \"Spec\"\nconstraints = '[{\"contentSchema\": {\"description\": \"A\"}}]'\ndoc = \"Spec.\""),
                                          "t", &ca, &why) &&
                  bc::contract::load_registry(replaced("cpp = \"Spec\"\ndoc = \"Spec.\"",
                                                       "cpp = \"Spec\"\nconstraints = '[{\"contentSchema\": {\"description\": \"B\"}}]'\ndoc = \"Spec.\""),
                                              "t", &cb, &why),
              "contentSchema fixtures load: " + why);
        check(bc::contract::fingerprint(ca) == bc::contract::fingerprint(cb), "an annotation inside contentSchema is stripped");
    }
    refused(replaced("value_ref = \"uint\"", "value_ref = \"group\""), "integer or string def",
            "a calibration value referring to an object def");
    refused(kBase + "\n[[field]]\npath = \"/spec/t\"\nkind = \"array\"\nitem_kind = \"tuple\"\ntuple = [\"group\", \"uint\"]\n"
                    "doc = \"x\"\n",
            "tuple elements must be integer defs", "a tuple element that is not an integer def");
    refused(kBase + "\n[[field]]\npath = \"/spec/s\"\nkind = \"string\"\npattern = \"^a+$\"\ndoc = \"x\"\n",
            "a pattern needs a max_length", "a pattern without a max_length (spec §11 amendment 7)");
    refused(kBase + "\n[[field]]\npath = \"/spec/s\"\nkind = \"string\"\nmin_length = -1\ndoc = \"x\"\n",
            "min_length is not negative", "a negative string length");
    refused(kBase + "\n[[field]]\npath = \"/spec/e\"\nkind = \"enum\"\nvalues = [\"a.b\", \"a_b\"]\ncpp_type = \"E\"\ndoc = \"x\"\n",
            "same C++ enumerator", "two enum values with one C++ enumerator");
    {
        std::string many = "[";
        for (int i = 0; i < 257; ++i) many += (i ? ", \"v" : "\"v") + std::to_string(i) + "\"";
        many += "]";
        refused(kBase + "\n[[field]]\npath = \"/spec/e\"\nkind = \"enum\"\nvalues = " + many + "\ncpp_type = \"E\"\ndoc = \"x\"\n",
                "at most 256 values", "an enum past 256 values");
        std::string keys;
        for (int i = 0; i < 257; ++i)
            keys += "\n[[calibration_key]]\nkey = \"k" + std::to_string(i) + "\"\nvalue_kind = \"ref\"\nvalue_ref = \"uint\"\ndoc = \"x\"\n";
        refused(replaced("[[def]]\nname = \"uint\"", keys + "\n[[def]]\nname = \"uint\""), "at most 256 calibration keys",
                "calibration keys past 256");
    }
    refused(kBase + "\n[[field]]\npath = \"/spec/a\"\nkind = \"array\"\nitem_kind = \"boolean\"\nmin_items = 3\nmax_items = 2\n"
                    "doc = \"x\"\n",
            "min_items exceeds max_items", "min_items above max_items");
    refused(kBase + "\n[[field]]\npath = \"/spec/t\"\nkind = \"array\"\nitem_kind = \"tuple\"\ntuple = [\"uint\", \"uint\"]\n"
                    "item_minimum = 1\ndoc = \"x\"\n",
            "takes only 'tuple'", "a scalar attribute on a tuple item");
    refused(kBase + "\n[[field]]\npath = \"/spec/m\"\nkind = \"map\"\nkey_ref = \"uint\"\nrequired_keys = []\ndoc = \"x\"\n"
                    "\n[[field]]\npath = \"/spec/m/*\"\nkind = \"boolean\"\ndoc = \"x\"\n",
            "key_ref must name a string def", "a map keyed by an integer def");
    refused(kBase + "\n[[field]]\npath = \"/spec/a\"\nkind = \"array\"\nitem_kind = \"boolean\"\nmin_items = -1\ndoc = \"x\"\n",
            "not negative", "a negative count");
    refused(kBase + "\n[[field]]\npath = \"/spec/x\"\nkind = \"boolean\"\ncpp_member = \"n\"\ndoc = \"x\"\n",
            "C++ member n is repeated", "two members with one C++ name");
    refused(replaced("[[def]]\nname = \"uint\"", "[[calibration_key]]\nkey = \"a_b.mib\"\nvalue_kind = \"ref\"\n"
                                                  "value_ref = \"uint\"\ndoc = \"x\"\n\n[[def]]\nname = \"uint\""),
            "twice", "calibration keys with one C++ enumerator");
    refused(kBase + "\n[[field]]\npath = \"/spec/e\"\nkind = \"enum\"\nvalues = [\"a/b\"]\ncpp_type = \"E\"\ndoc = \"x\"\n",
            "invalid C++ enumerator", "an enum value that is no C++ identifier");
    refused(replaced("namespace = \"bc::serving::test\"", "namespace = \"bc::1x\""), "namespace", "a bad namespace");
    refused(replaced("[[class_default]]\npath = \"/spec\"\nclass = \"policy\"",
                     "[[class_default]]\npath = \"/spec\"\nclass = \"policy\"\n\n[[class_default]]\npath = \"/spec\"\nclass = \"fact\""),
            "two class defaults", "a repeated class default");

    // the real registry loads and generates
    std::ifstream in(std::string(BC_CONTRACT_SOURCE_DIR) + "/contract/serving-v1.toml", std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    bc::contract::Registry real;
    check(bc::contract::load_registry(text.str(), "serving-v1.toml", &real, &why), "the v1 registry loads: " + why);
    if (failures) return 1;
    check(bc::contract::generate_all(real).size() == 6, "six generated files");
    std::printf("ok\n");
    return 0;
}
