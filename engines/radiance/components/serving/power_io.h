#pragma once

// Private movable RAII wrapper for kernel descriptors used by power adapters.
#include <unistd.h>
#include <utility>
#include <filesystem>
#include <stdexcept>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>

namespace bc::serving::power::detail {
class FileDescriptor {
  public:
    explicit FileDescriptor(int value = -1) : value_(value) {}
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    FileDescriptor(FileDescriptor&& other) noexcept : value_(std::exchange(other.value_, -1)) {}
    FileDescriptor& operator=(FileDescriptor&& other) noexcept {
        if (this != &other) {
            close();
            value_ = std::exchange(other.value_, -1);
        }
        return *this;
    }
    ~FileDescriptor() {
        close();
    }
    explicit operator bool() const {
        return value_ >= 0;
    }
    int get() const {
        return value_;
    }

  private:
    void close() noexcept {
        if (value_ >= 0)
            (void)::close(std::exchange(value_, -1));
    }
    int value_;
};
inline FileDescriptor lock_power_owner(const std::filesystem::path& root) {
    const auto directory = root / "power";
    std::filesystem::create_directories(directory);
    if (std::filesystem::is_symlink(directory))
        throw std::runtime_error("power owner directory is symlinked");
    FileDescriptor fd(
        ::open((directory / "owner.lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600));
    struct stat status{};
    if (!fd || ::fstat(fd.get(), &status) != 0 || !S_ISREG(status.st_mode) ||
        ::flock(fd.get(), LOCK_EX | LOCK_NB) != 0)
        throw std::runtime_error("power owner lock unavailable or busy");
    return fd;
}

} // namespace bc::serving::power::detail
