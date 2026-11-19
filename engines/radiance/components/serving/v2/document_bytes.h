#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace bc::serving::v2 {

inline constexpr size_t kMaxDocumentBytes = 16u * 1024u * 1024u;

/* The offset of the first byte that is not well-formed UTF-8 (no overlongs, surrogates or code points past U+10FFFF),
 * or npos. */
inline size_t bad_utf8(std::string_view s) {
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

}  // namespace bc::serving::v2
