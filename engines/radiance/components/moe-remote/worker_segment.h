#pragma once
// Included only by r3_moe_worker.cpp, after worker_container.h (stopping()).
//
// The shared mailbox segment, owned by one object: shm_open (helper 0 creates and sizes it, the others attach), mmap,
// and for the GPU path hipHostRegister + the device pointer. The destructor undoes exactly what was done, in reverse;
// only the creator unlinks the name. Everything that uses the mapping (lane threads, the queue warm-up) must be stopped
// and joined before this object is destroyed, which declaring it first in the serving scope guarantees.
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

class Segment {
public:
    Segment() = default;
    Segment(const Segment&) = delete;
    Segment& operator=(const Segment&) = delete;
    ~Segment() { close_all(); }

    /* Helper 0 creates a fresh segment of `bytes` (failing if the name exists: a stale segment is the launcher's to
     * remove); another helper attaches, waiting up to `attach_wait_s` for helper 0 to create it (stop-aware). */
    bool open(const char* name, size_t bytes, bool creator, int attach_wait_s) {
        name_ = name;
        bytes_ = bytes;
        if (creator) {
            fd_ = shm_open(name, O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
            if (fd_ < 0) return fail("shm_open");
            unlink_ = true;
            if (ftruncate(fd_, (off_t)bytes) != 0) return fail("ftruncate");
        } else {
            /* Nothing to wait on before the file exists, so a bounded retry: startup only. */
            for (int i = 0; i < attach_wait_s && fd_ < 0 && !stopping(); ++i) {
                fd_ = shm_open(name, O_RDWR | O_CLOEXEC, 0);
                if (fd_ < 0) ::sleep(1);
            }
            if (fd_ < 0) return fail("shm_open");
        }
        void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (m == MAP_FAILED) return fail("mmap");
        map_ = m;
        return true;
    }

    /* The GPU path: pin and map the segment for this device. */
    bool register_gpu() {
        if (!r3_moe_gpu::hip_ok(hipHostRegister(map_, bytes_, hipHostRegisterPortable | hipHostRegisterMapped),
                                "hipHostRegister mailbox"))
            return false;
        registered_ = true;
        return r3_moe_gpu::hip_ok(hipHostGetDevicePointer(&dev_map_, map_, 0), "hipHostGetDevicePointer mailbox");
    }

    void* host() const { return map_; }
    void* device() const { return dev_map_; }
    r3::moe_shm::Header* header() const { return static_cast<r3::moe_shm::Header*>(map_); }
    size_t bytes() const { return bytes_; }
    const char* name() const { return name_; }

private:
    bool fail(const char* what) {
        std::fprintf(stderr, "r3_moe_worker: %s %s: %s\n", what, name_, std::strerror(errno));
        return false;
    }
    void close_all() {
        if (registered_) (void)hipHostUnregister(map_);
        if (map_) (void)munmap(map_, bytes_);
        if (fd_ >= 0) (void)::close(fd_);
        if (unlink_) (void)shm_unlink(name_);
        registered_ = false;
        map_ = nullptr;
        fd_ = -1;
        unlink_ = false;
    }

    const char* name_ = "";
    size_t bytes_ = 0;
    bool unlink_ = false;
    int fd_ = -1;
    void* map_ = nullptr;
    void* dev_map_ = nullptr;
    bool registered_ = false;
};
