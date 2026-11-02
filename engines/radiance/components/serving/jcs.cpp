#include "jcs.h"
#include "serving.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace bc::serving {
namespace {
using Json = nlohmann::json;
constexpr int64_t kMaxSafe = (int64_t(1) << 53) - 1;

std::string pointer_token(const std::string& key) {
    std::string t;
    for (char c : key) {
        if (c == '~') t += "~0";
        else if (c == '/') t += "~1";
        else t += c;
    }
    return t;
}

// UTF-8 to UTF-16 code units, refusing malformed, overlong and surrogate encodings.
bool utf16(const std::string& s, std::u16string* out) {
    out->clear();
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        uint32_t cp = 0;
        size_t n = 0;
        if (c < 0x80) { cp = c; n = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; n = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; n = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; n = 4; }
        else return false;
        if (i + n > s.size()) return false;
        for (size_t k = 1; k < n; ++k) {
            const unsigned char d = static_cast<unsigned char>(s[i + k]);
            if ((d & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (d & 0x3F);
        }
        static const uint32_t kMin[5] = {0, 0, 0x80, 0x800, 0x10000};
        if (cp < kMin[n] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
        if (cp >= 0x10000) {
            cp -= 0x10000;
            out->push_back(char16_t(0xD800 + (cp >> 10)));
            out->push_back(char16_t(0xDC00 + (cp & 0x3FF)));
        } else {
            out->push_back(char16_t(cp));
        }
        i += n;
    }
    return true;
}

bool string_out(const std::string& s, std::string* out) {
    std::u16string check;
    if (!utf16(s, &check)) return false;
    static const char kHex[] = "0123456789abcdef";
    *out += '"';
    for (char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"': *out += "\\\""; break;
            case '\\': *out += "\\\\"; break;
            case '\b': *out += "\\b"; break;
            case '\f': *out += "\\f"; break;
            case '\n': *out += "\\n"; break;
            case '\r': *out += "\\r"; break;
            case '\t': *out += "\\t"; break;
            default:
                if (c < 0x20) {
                    *out += "\\u00";
                    *out += kHex[c >> 4];
                    *out += kHex[c & 0xF];
                } else {
                    *out += ch;
                }
        }
    }
    *out += '"';
    return true;
}

bool emit(const Json& v, const std::string& at, std::string* out, std::string* why) {
    auto fail = [&](const std::string& reason) {
        if (why) *why = (at.empty() ? std::string("(root)") : at) + ": " + reason;
        return false;
    };
    switch (v.type()) {
        case Json::value_t::null: *out += "null"; return true;
        case Json::value_t::boolean: *out += v.get<bool>() ? "true" : "false"; return true;
        case Json::value_t::number_integer: {
            const int64_t n = v.get<int64_t>();
            if (n > kMaxSafe || n < -kMaxSafe) return fail("integer outside I-JSON's +/-(2^53 - 1)");
            *out += std::to_string(n);
            return true;
        }
        case Json::value_t::number_unsigned: {
            const uint64_t n = v.get<uint64_t>();
            if (n > uint64_t(kMaxSafe)) return fail("integer outside I-JSON's +/-(2^53 - 1)");
            *out += std::to_string(n);
            return true;
        }
        case Json::value_t::number_float:
            return fail("non-integer number (v1 numbers are integers; a fraction is a fixed-scale integer)");
        case Json::value_t::string:
            if (!string_out(v.get_ref<const std::string&>(), out)) return fail("string is not valid UTF-8");
            return true;
        case Json::value_t::array: {
            *out += '[';
            size_t i = 0;
            for (const auto& e : v) {
                if (i) *out += ',';
                if (!emit(e, at + "/" + std::to_string(i), out, why)) return false;
                ++i;
            }
            *out += ']';
            return true;
        }
        case Json::value_t::object: {
            std::vector<std::pair<std::u16string, const std::string*>> keys;
            keys.reserve(v.size());
            for (auto it = v.begin(); it != v.end(); ++it) {
                std::u16string k16;
                if (!utf16(it.key(), &k16)) return fail("member name is not valid UTF-8");
                keys.emplace_back(std::move(k16), &it.key());
            }
            std::sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            *out += '{';
            for (size_t i = 0; i < keys.size(); ++i) {
                if (i) *out += ',';
                const std::string& name = *keys[i].second;
                if (!string_out(name, out)) return fail("member name is not valid UTF-8");
                *out += ':';
                if (!emit(v.at(name), at + "/" + pointer_token(name), out, why)) return false;
            }
            *out += '}';
            return true;
        }
        default:
            return fail("value type has no JSON form (binary or discarded)");
    }
}
}  // namespace

bool jcs_canonicalize(const Json& value, std::string* out, std::string* why) {
    std::string text;
    if (!emit(value, "", &text, why)) return false;
    *out = std::move(text);
    return true;
}

bool manifest_content_digest(const Json& manifest, std::string* digest, std::string* why) {
    if (!manifest.is_object() || !manifest.contains("metadata") || !manifest.at("metadata").is_object()) {
        if (why) *why = "(root): a manifest is an object with a metadata object";
        return false;
    }
    Json copy = manifest;
    copy["metadata"].erase("content_digest");
    std::string bytes;
    if (!jcs_canonicalize(copy, &bytes, why)) return false;
    *digest = "sha256:" + sha256_hex(bytes);
    return true;
}

}  // namespace bc::serving
