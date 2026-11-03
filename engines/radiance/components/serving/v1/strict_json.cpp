#include "strict_json.h"

#include <cstdint>
#include <set>
#include <vector>

namespace bc::serving::v1 {
namespace {

using Json = nlohmann::json;
constexpr int64_t kMaxSafe = (int64_t(1) << 53) - 1;
const char* const kHint = "regenerate the manifest with bc-serve; a v1 manifest is never edited by hand";

/* The offset of the first byte that is not well-formed UTF-8 (no overlongs, surrogates or code points past U+10FFFF),
 * or npos. */
size_t bad_utf8(std::string_view s) {
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        size_t n = 0;
        uint32_t cp = 0;
        if (c < 0x80) { ++i; continue; }
        if ((c & 0xE0) == 0xC0) { n = 2; cp = c & 0x1F; }
        else if ((c & 0xF0) == 0xE0) { n = 3; cp = c & 0x0F; }
        else if ((c & 0xF8) == 0xF0) { n = 4; cp = c & 0x07; }
        else return i;
        if (i + n > s.size()) return i;
        for (size_t k = 1; k < n; ++k) {
            const unsigned char d = static_cast<unsigned char>(s[i + k]);
            if ((d & 0xC0) != 0x80) return i;
            cp = (cp << 6) | (d & 0x3F);
        }
        static const uint32_t kMin[5] = {0, 0, 0x80, 0x800, 0x10000};
        if (cp < kMin[n] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return i;
        i += n;
    }
    return std::string_view::npos;
}

void locate(std::string_view bytes, size_t offset, ConfigError* e) {
    uint32_t line = 1, column = 1;
    for (size_t i = 0; i < offset && i < bytes.size(); ++i) {
        if (bytes[i] == '\n') { ++line; column = 1; }
        else ++column;
    }
    e->line = line;
    e->column = column;
}

/* Builds the document while refusing what v1 forbids. Containers are held by pointer: an open container's parent is
 * never modified while it is open, so the pointers stay valid. */
class Sax : public nlohmann::json_sax<Json> {
public:
    Json root;
    std::optional<ConfigError> error;
    size_t error_offset = std::string::npos;

    bool null() override { return place(Json(nullptr)); }
    bool boolean(bool v) override { return place(Json(v)); }
    bool number_integer(number_integer_t v) override {
        if (v > kMaxSafe || v < -kMaxSafe) return fail_number(std::to_string(v));
        return place(Json(v));
    }
    bool number_unsigned(number_unsigned_t v) override {
        if (v > static_cast<uint64_t>(kMaxSafe)) return fail_number(std::to_string(v));
        return place(Json(v));
    }
    bool number_float(number_float_t, const string_t& text) override {
        return fail(ConfigErrorCode::invalid_format, child_path(),
                    "non-integer number " + text + " (every v1 number is an integer within +/-(2^53 - 1))");
    }
    bool string(string_t& v) override { return place(Json(std::move(v))); }
    bool binary(binary_t&) override { return fail(ConfigErrorCode::invalid_format, child_path(), "binary value"); }
    bool start_object(std::size_t) override { return open(Json::object()); }
    bool key(string_t& k) override {
        if (!keys_.back().insert(k).second)
            return fail(ConfigErrorCode::duplicate_field, paths_.back(),
                        "duplicate member \"" + k + "\" in " + (paths_.back().empty() ? "the document" : paths_.back()));
        key_ = k;
        return true;
    }
    bool end_object() override { return close(); }
    bool start_array(std::size_t) override { return open(Json::array()); }
    bool end_array() override { return close(); }
    bool parse_error(std::size_t position, const std::string&, const nlohmann::detail::exception& ex) override {
        if (!error) {
            error = make_error(ConfigErrorCode::invalid_format, "", std::string("malformed JSON: ") + ex.what(), kHint);
            error_offset = position > 0 ? position - 1 : 0;
        }
        return false;
    }

private:
    std::vector<Json*> stack_;
    std::vector<std::set<std::string>> keys_;
    std::vector<std::string> paths_;
    std::string key_;

    std::string child_path() const {
        if (stack_.empty()) return "";
        const Json& top = *stack_.back();
        const std::string& at = paths_.back();
        if (top.is_array()) return at + "[" + std::to_string(top.size()) + "]";
        return at.empty() ? key_ : at + "." + key_;
    }

    Json* put(Json v) {
        if (stack_.empty()) {
            root = std::move(v);
            return &root;
        }
        Json& top = *stack_.back();
        if (top.is_array()) {
            top.push_back(std::move(v));
            return &top.back();
        }
        top[key_] = std::move(v);
        return &top[key_];
    }

    bool place(Json v) {
        put(std::move(v));
        return true;
    }
    bool open(Json container) {
        if (stack_.size() >= kMaxDepth)
            return fail(ConfigErrorCode::invalid_format, stack_.empty() ? std::string() : paths_.back(),
                        "nested deeper than " + std::to_string(kMaxDepth) +
                            " levels, the deepest a v1 document can be (spec §11 amendment 8)");
        const std::string path = child_path();
        stack_.push_back(put(std::move(container)));
        keys_.emplace_back();
        paths_.push_back(path);
        return true;
    }
    bool close() {
        stack_.pop_back();
        keys_.pop_back();
        paths_.pop_back();
        return true;
    }
    bool fail_number(const std::string& text) {
        return fail(ConfigErrorCode::invalid_format, child_path(),
                    "integer " + text + " is outside I-JSON's +/-(2^53 - 1)");
    }
    bool fail(ConfigErrorCode code, const std::string& path, const std::string& message) {
        if (!error) error = make_error(code, path, message, kHint);
        return false;
    }
};

}  // namespace

Result<nlohmann::json> parse_strict_json(std::string_view bytes, const std::string& source) {
    auto with_source = [&](ConfigError e) {
        e.source_file = source;
        return e;
    };
    if (bytes.size() > kMaxDocumentBytes)
        return with_source(make_error(ConfigErrorCode::invalid_format, "",
                                      "document is " + std::to_string(bytes.size()) + " bytes, over the " +
                                          std::to_string(kMaxDocumentBytes) + "-byte limit",
                                      kHint));
    if (bytes.size() >= 3 && static_cast<unsigned char>(bytes[0]) == 0xEF &&
        static_cast<unsigned char>(bytes[1]) == 0xBB && static_cast<unsigned char>(bytes[2]) == 0xBF)
        return with_source(make_error(ConfigErrorCode::invalid_format, "", "document starts with a byte-order mark",
                                      kHint));
    if (const size_t at = bad_utf8(bytes); at != std::string_view::npos) {
        ConfigError e = with_source(make_error(ConfigErrorCode::invalid_format, "",
                                               "document is not valid UTF-8 (byte " + std::to_string(at) + ")", kHint));
        locate(bytes, at, &e);
        return e;
    }
    Sax sax;
    const bool ok = Json::sax_parse(bytes.begin(), bytes.end(), &sax, nlohmann::json::input_format_t::json, true);
    if (sax.error) {
        ConfigError e = with_source(*sax.error);
        if (sax.error_offset != std::string::npos) locate(bytes, sax.error_offset, &e);
        return e;
    }
    if (!ok)
        return with_source(make_error(ConfigErrorCode::invalid_format, "", "malformed JSON", kHint));
    return std::move(sax.root);
}

}  // namespace bc::serving::v1
