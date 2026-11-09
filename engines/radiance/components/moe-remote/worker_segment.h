#pragma once
// Included only by r3_moe_worker.cpp, after worker_container.h (stopping()).
//
// The shared mailbox segment, owned by one object: shm_open (helper 0 creates and sizes it, the others attach), mmap,
// and for the GPU path hipHostRegister + the device pointer. The destructor undoes exactly what was done, in reverse;
// only the creator unlinks the name. Everything that uses the mapping (lane threads, the queue warm-up) must be stopped
// and joined before this object is destroyed, which declaring it first in the serving scope guarantees.
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
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
        // One startup deadline covers name creation, sizing, and header publication.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(attach_wait_s);
        if (creator) {
            fd_ = shm_open(name, O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
            if (fd_ < 0) return fail("shm_open");
            unlink_ = true;
            struct stat made {};
            if (fstat(fd_, &made) == 0) { dev_ = made.st_dev; ino_ = made.st_ino; }
            if (ftruncate(fd_, (off_t)bytes) != 0) return fail("ftruncate");
        } else {
            // Opening the name is not enough: the creator has not necessarily ftruncated yet.
            while (fd_ < 0 && !stopping() && std::chrono::steady_clock::now() < deadline) {
                fd_ = shm_open(name, O_RDWR | O_CLOEXEC, 0);
                if (fd_ < 0) ::sleep(1);
            }
            if (fd_ < 0) return fail("shm_open");
            bool sized = false;
            while (!stopping() && std::chrono::steady_clock::now() < deadline) {
                struct stat st {};
                if (fstat(fd_, &st) != 0) return fail("fstat");
                if (st.st_size == static_cast<off_t>(bytes)) { sized = true; break; }
                if (st.st_size > static_cast<off_t>(bytes)) {
                    std::fprintf(stderr, "r3_moe_worker: mailbox size exceeds expected layout\n");
                    return false;
                }
                ::sleep(1);
            }
            if (!sized) {
                std::fprintf(stderr, "r3_moe_worker: mailbox not sized before attach deadline\n");
                return false;
            }
        }
        void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (m == MAP_FAILED) return fail("mmap");
        map_ = m;
        // An attacher never accepts a legacy or uninitialised segment.
        // The creator's zero-filled mapping is initialized to v23 by main
        // before publishing helper_count. Full shape validation follows
        // that startup barrier (and again before GPU registration).
        if (!creator) {
            while (!stopping() && std::chrono::steady_clock::now() < deadline) {
                const uint32_t version=__atomic_load_n(&header()->version,__ATOMIC_ACQUIRE);
                if (version && version!=r3::moe_shm::kVersion) {
                    std::fprintf(stderr,"r3_moe_worker: refusing non-v23 mailbox\n");
                    return false;
                }
                if (version==r3::moe_shm::kVersion &&
                    __atomic_load_n(&header()->magic,__ATOMIC_ACQUIRE)==r3::moe_shm::kMagic) return true;
                ::sleep(1);
            }
            std::fprintf(stderr,"r3_moe_worker: no valid v23 mailbox before attach deadline\n");
            return false;
        }
        return true;
    }

    /* The GPU path: pin and map the segment for this device. */
    bool register_gpu() {
        /* Validate the complete v23 layout before pinning the host mapping. */
        if (!map_ || !r3::moe_shm::valid_self(header(), bytes_)) {
            std::fprintf(stderr, "r3_moe_worker: invalid v23 mailbox before GPU registration\n");
            return false;
        }
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
    /* Remove the name only if it still names the segment this object created. A later start may already have made a new
     * segment under the same name (the launcher removes a stale name and the next helper 0 recreates it): unlinking by name
     * alone would remove THAT one, and the engine then finds no hand-off. Compare the object behind the name with ours
     * (device and inode, from fstat at create) and leave it alone when it differs. */
    void unlink_if_ours() {
        const int now = shm_open(name_, O_RDONLY | O_CLOEXEC, 0);
        if (now < 0) return;                       // already gone: nothing to remove
        struct stat s {};
        const bool ours = fstat(now, &s) == 0 && s.st_dev == dev_ && s.st_ino == ino_;
        (void)::close(now);
        if (ours) (void)shm_unlink(name_);
    }
    void close_all() {
        if (registered_) (void)hipHostUnregister(map_);
        if (map_) (void)munmap(map_, bytes_);
        if (fd_ >= 0) (void)::close(fd_);
        if (unlink_) unlink_if_ours();
        registered_ = false;
        map_ = nullptr;
        fd_ = -1;
        unlink_ = false;
    }

    const char* name_ = "";
    size_t bytes_ = 0;
    bool unlink_ = false;
    dev_t dev_ = 0;     // the object this creator made (fstat at create), so unlink_if_ours can tell it from a later one
    ino_t ino_ = 0;
    int fd_ = -1;
    void* map_ = nullptr;
    void* dev_map_ = nullptr;
    bool registered_ = false;
};
