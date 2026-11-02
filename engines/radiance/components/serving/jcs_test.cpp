// RFC 8785 (JCS) for the serving manifest v1 subset: known-answer vectors, refusals, and the committed examples'
// content digests (docs/design/serving-manifest-v1, cross-checked there by an independent canonicaliser).
#include "jcs.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#ifndef BC_CONTRACT_DIR
#error "BC_CONTRACT_DIR must name docs/design/serving-manifest-v1"
#endif

namespace {
using Json = nlohmann::json;
int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++failures;
    }
}

std::string canon(const Json& j) {
    std::string out, why;
    if (!bc::serving::jcs_canonicalize(j, &out, &why)) return "<refused: " + why + ">";
    return out;
}

std::string refusal(const Json& j) {
    std::string out, why;
    return bc::serving::jcs_canonicalize(j, &out, &why) ? std::string() : why;
}

Json read_json(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream s;
    s << in.rdbuf();
    return Json::parse(s.str());
}
}  // namespace

int main() {
    // RFC 8785 §3.2.3: members sort by UTF-16 code units, so U+1F600 (D83D DE00) precedes U+FB33.
    const Json sorting = Json::parse("{\"\\u20ac\":\"Euro Sign\",\"\\r\":\"Carriage Return\",\"\\ufb33\":\"Hebrew Letter "
                                     "Dalet With Dagesh\",\"1\":\"One\",\"\\ud83d\\ude00\":\"Emoji: Grinning Face\","
                                     "\"\\u0080\":\"Control\",\"\\u00f6\":\"Latin Small Letter O With Diaeresis\"}");
    const std::string sorted = "{\"\\r\":\"Carriage Return\",\"1\":\"One\",\"\xc2\x80\":\"Control\","
                               "\"\xc3\xb6\":\"Latin Small Letter O With Diaeresis\",\"\xe2\x82\xac\":\"Euro Sign\","
                               "\"\xf0\x9f\x98\x80\":\"Emoji: Grinning Face\","
                               "\"\xef\xac\xb3\":\"Hebrew Letter Dalet With Dagesh\"}";
    check(canon(sorting) == sorted, "RFC 8785 3.2.3 member order: " + canon(sorting));

    // Escapes: the six short forms, other controls as \u00xx lowercase, '/' and DEL and non-ASCII unescaped.
    const Json escapes = std::string("\x00\x07\x1f\"\\/\b\f\n\r\t\x7f\xc3\xb6", 14);
    check(canon(escapes) == std::string("\"\\u0000\\u0007\\u001f\\\"\\\\/\\b\\f\\n\\r\\t\x7f\xc3\xb6\""),
          "escapes: " + canon(escapes));

    // Structure without whitespace; integers in plain decimal at the I-JSON bounds.
    const Json nested = Json::parse(R"({"b":[1,{"z":true,"a":null}],"a":{"y":-9007199254740991,"x":9007199254740991}})");
    check(canon(nested) == R"({"a":{"x":9007199254740991,"y":-9007199254740991},"b":[1,{"a":null,"z":true}]})",
          "nested: " + canon(nested));
    check(canon(Json::parse("[]")) == "[]" && canon(Json::parse("{}")) == "{}" && canon(Json("")) == "\"\"",
          "empty containers and string");

    // Refusals, each naming the JSON Pointer of the offending value.
    check(refusal(Json::parse(R"({"a":[1,0.5]})")).find("/a/1: non-integer number") == 0, "float refused");
    check(refusal(Json::parse(R"({"n":9007199254740992})")).find("/n: integer outside") == 0, "2^53 refused");
    check(refusal(Json::parse(R"({"n":-9007199254740992})")).find("/n: integer outside") == 0, "-2^53 refused");
    check(refusal(Json(std::string("\xc3\x28", 2))).find("not valid UTF-8") != std::string::npos, "bad UTF-8 refused");
    check(refusal(Json(std::string("\xed\xa0\x80", 3))).find("not valid UTF-8") != std::string::npos,
          "encoded surrogate refused");
    check(refusal(Json(std::string("\xc0\xaf", 2))).find("not valid UTF-8") != std::string::npos, "overlong refused");
    check(refusal(Json::parse(R"({"a/b":{"c~d":1.5}})")).find("/a~1b/c~0d:") == 0, "pointer escaping in refusals");

    // U+2028 and U+2029 are left unescaped (RFC 8785 escapes only what JSON requires).
    check(canon(Json("\xe2\x80\xa8\xe2\x80\xa9")) == "\"\xe2\x80\xa8\xe2\x80\xa9\"", "U+2028/U+2029 unescaped");
    // More invalid UTF-8: truncated multibyte, a 4-byte overlong, a code point above U+10FFFF, in a member name too.
    check(refusal(Json(std::string("\xe2\x82", 2))).find("not valid UTF-8") != std::string::npos, "truncated refused");
    check(refusal(Json(std::string("\xf0\x82\x82\xac", 4))).find("not valid UTF-8") != std::string::npos,
          "4-byte overlong refused");
    check(refusal(Json(std::string("\xf4\x90\x80\x80", 4))).find("not valid UTF-8") != std::string::npos,
          "above U+10FFFF refused");
    {
        Json bad_name = Json::object();
        bad_name[std::string("k\xc3", 2)] = 1;
        check(refusal(bad_name).find("member name is not valid UTF-8") != std::string::npos, "invalid member name refused");
    }
    // An unsigned value above INT64_MAX (and so above 2^53 - 1) is refused, not wrapped.
    check(refusal(Json(uint64_t(9223372036854775813ull))).find("integer outside") != std::string::npos,
          "unsigned above INT64_MAX refused");
    // The bare token -0 parses as the integer 0 and canonicalizes to 0, consistent with RFC 8785 (ECMAScript renders
    // -0 as "0"). Refusing the -0 spelling would be the token-aware v1 reader's job, and only if the spec required it.
    check(Json::parse("-0").is_number_integer() && canon(Json::parse("-0")) == "0", "-0 canonicalizes to 0");
    // content_digest: absent is fine, a string is replaced, any other type is refused naming the member.
    {
        Json m = Json::parse(R"({"metadata":{"x":1},"spec":{}})");
        std::string d, why;
        check(bc::serving::manifest_content_digest(m, &d, &why), "absent content_digest digests: " + why);
        m["metadata"]["content_digest"] = "sha256:anything";
        std::string d2;
        check(bc::serving::manifest_content_digest(m, &d2, &why) && d2 == d, "a string content_digest is excluded");
        for (const Json& wrong : {Json(nullptr), Json(5), Json::object(), Json::array()}) {
            m["metadata"]["content_digest"] = wrong;
            why.clear();
            check(!bc::serving::manifest_content_digest(m, &d2, &why) &&
                      why == "/metadata/content_digest: must be a string when present",
                  "content_digest of type " + std::string(wrong.type_name()) + " refused: " + why);
        }
    }

    // The committed examples: their recorded content_digest is the JCS SHA-256 without that member.
    for (const char* name : {"example.flashnext-4gpu.json", "example.lab-experiment.json"}) {
        const Json m = read_json(std::string(BC_CONTRACT_DIR) + "/" + name);
        std::string digest, why;
        check(bc::serving::manifest_content_digest(m, &digest, &why), std::string(name) + ": " + why);
        check(digest == m.at("metadata").at("content_digest").get<std::string>(),
              std::string(name) + ": digest " + digest);
        Json changed = m;
        changed["spec"]["intent"]["service"]["max_sessions"] = 7;
        std::string other;
        check(bc::serving::manifest_content_digest(changed, &other, &why) && other != digest,
              std::string(name) + ": a changed value changes the digest");
    }

    if (failures) return 1;
    std::printf("ok\n");
    return 0;
}
