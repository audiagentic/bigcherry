#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace bc::serving::v2 {

inline constexpr size_t kMaxDocumentBytes = 16u * 1024u * 1024u;

inline std::string read_sealed_document(int fd, std::string_view label) {
    if (fd < 3) throw std::invalid_argument(std::string(label) + " must be greater than 2");
    constexpr int required_seals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
    const int seals = ::fcntl(fd, F_GET_SEALS);
    if (seals < 0 || (seals & required_seals) != required_seals)
        throw std::invalid_argument(std::string(label) + " must be a fully sealed memfd");
    struct stat info {};
    if (::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size <= 0 ||
        static_cast<uintmax_t>(info.st_size) > kMaxDocumentBytes)
        throw std::invalid_argument(std::string(label) + " must be a bounded regular file");
    std::string bytes(static_cast<size_t>(info.st_size), '\0');
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = ::pread(fd, bytes.data() + offset, bytes.size() - offset,
                                      static_cast<off_t>(offset));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("could not read the complete " + std::string(label));
        offset += static_cast<size_t>(count);
    }
    return bytes;
}

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
