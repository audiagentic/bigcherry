#pragma once
// What a file's bytes cannot change without changing: size, inode, modification and change time. The change time is set
// by the kernel on every write and cannot be restored by a caller, so an unchanged stamp means unchanged content. One
// mechanism for everything that skips a hash because "nothing moved" (the engine identity digest, a release install's
// verification): stamp BEFORE hashing, store the stamps with the result, and trust the result only while every stamp
// still matches; any difference, or a record that cannot be read, hashes again.
#include <sys/stat.h>

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace bc::serving {

struct FileStamp {
    uint64_t size = 0, inode = 0;
    int64_t mtime_ns = 0, ctime_ns = 0;
    bool operator==(const FileStamp& o) const {
        return size == o.size && inode == o.inode && mtime_ns == o.mtime_ns && ctime_ns == o.ctime_ns;
    }
};

inline int64_t timespec_ns(const timespec& t) { return int64_t{t.tv_sec} * 1000000000 + t.tv_nsec; }

inline FileStamp stamp_of(const struct stat& st) {
    return {uint64_t(st.st_size), uint64_t(st.st_ino), timespec_ns(st.st_mtim), timespec_ns(st.st_ctim)};
}

// The stamp of the file's content (a symbolic link is followed).
inline FileStamp stamp_file(const std::filesystem::path& path) {
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) throw std::runtime_error("cannot stat " + path.string());
    return stamp_of(st);
}

}  // namespace bc::serving
