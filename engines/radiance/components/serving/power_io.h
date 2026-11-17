#pragma once

// Private movable RAII wrapper for kernel descriptors used by power adapters.
#include <unistd.h>
#include <utility>

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
} // namespace bc::serving::power::detail
