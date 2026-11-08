// RHP11 F: Segment::open refuses a non-v24 shared mailbox before
// registration. HIP functions are stubs: this test uses NO GPU.
#include "moe_remote_shm.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <thread>
#include <cstdlib>
#include <string>
#include <unistd.h>
namespace { bool stopping() {return false;} }
static constexpr unsigned hipHostRegisterPortable=1,hipHostRegisterMapped=2;
static unsigned registers=0;
inline int hipHostRegister(void*,size_t,unsigned) {++registers;return 0;}
inline int hipHostGetDevicePointer(void** device,void* host,unsigned) {
    *device=host;return 0;
}
inline int hipHostUnregister(void*) {return 0;}
namespace r3_moe_gpu {
inline bool hip_ok(int code,const char*) {return code==0;}
}
#include "worker_segment.h"
static void check(bool x,const char* message) {
    if(!x){std::fprintf(stderr,"segment_version_test: %s\n",message);std::exit(1);}
}
int main() {
    using namespace r3::moe_shm;
    const LayoutShape shape{128,2,3,8,128,64};
    Layout layout{};
    check(layout_for(shape,2,slots_for(2),&layout),"layout");
    const std::string name="/bc-v24-"+std::to_string(getpid());
    Segment creator;
    check(creator.open(name.c_str(),layout.bytes,true,1),"create");
    check(init(creator.host(),2,shape,2),"valid v24 init");
    auto* h=creator.header();
    h->helper_count=2;
    h->version=23;
    {
        Segment legacy;
        check(!legacy.open(name.c_str(),layout.bytes,false,1),
              "legacy version refused before HIP registration");
    }
    check(registers==0,"legacy mailbox never registered");
    h->version=kVersion;
    {
        Segment attached;
        check(attached.open(name.c_str(),layout.bytes,false,1),
              "v24 accepted by actual Segment::open");
        check(attached.register_gpu() && registers==1,
              "validated v24 passed to fake register");
        auto* g=generation_header(creator.host(),0);
        const uint64_t saved=g->digest;
        g->digest^=1;
        check(!creator.register_gpu() && registers==1,
              "bad v24 generation never registered");
        g->digest=saved;
    }
    // A creator unlinks only the segment it made. Its name can be taken over by a later start's segment before it is destroyed
    // (a late exit): destroying the old creator must leave the new one in place, and the new one still removes its own.
    {
        const std::string own = "/bc-v24-own-" + std::to_string(getpid());
        auto exists = [&] {
            const int fd = shm_open(own.c_str(), O_RDONLY | O_CLOEXEC, 0);
            if (fd >= 0) ::close(fd);
            return fd >= 0;
        };
        Segment* first = new Segment;
        check(first->open(own.c_str(), layout.bytes, true, 1), "create the first segment");
        check(shm_unlink(own.c_str()) == 0, "the first segment's name is removed behind its back");
        {
            Segment second;
            check(second.open(own.c_str(), layout.bytes, true, 1), "a later start makes a new segment under the same name");
            delete first;                                  // the late exit of the first creator
            check(exists(), "the late creator did not remove the newer segment");
        }
        check(!exists(), "the newer creator removes its own name");
        // and the ordinary case: a creator whose name is still its own removes it
        {
            Segment only;
            check(only.open(own.c_str(), layout.bytes, true, 1), "create");
            check(exists(), "exists while open");
        }
        check(!exists(), "an ordinary creator removes its name");
    }
    // The name can exist before ftruncate: an attacher must not mmap and
    // read a zero-length object (SIGBUS), even when shm_open succeeds.
    const std::string delayed_name = "/bc-v24-sizing-" + std::to_string(getpid());
    const int delayed_fd = shm_open(delayed_name.c_str(),
                                   O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    check(delayed_fd >= 0, "open unsized mailbox");
    std::atomic<bool> entered{false}, finished{false}, accepted{false};
    std::thread attacher([&] {
        Segment segment;
        entered.store(true, std::memory_order_release);
        const bool ok = segment.open(delayed_name.c_str(), layout.bytes, false, 5);
        accepted.store(ok && segment.header() && segment.header()->version == kVersion,
                       std::memory_order_release);
        finished.store(true, std::memory_order_release);
    });
    while (!entered.load(std::memory_order_acquire)) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    check(!finished.load(std::memory_order_acquire),
          "unsized segment cannot finish attaching or access the header");
    check(ftruncate(delayed_fd, (off_t)layout.bytes) == 0, "late ftruncate");
    void* delayed_map = mmap(nullptr, layout.bytes, PROT_READ | PROT_WRITE,
                             MAP_SHARED, delayed_fd, 0);
    check(delayed_map != MAP_FAILED, "map sized creator segment");
    check(init(delayed_map, 2, shape, 2), "initialize late-sized v24");
    static_cast<Header*>(delayed_map)->helper_count = 2;
    attacher.join();
    check(accepted.load(std::memory_order_acquire), "attacher waits for sizing and v24");
    check(munmap(delayed_map, layout.bytes) == 0, "unmap late segment");
    check(::close(delayed_fd) == 0 && shm_unlink(delayed_name.c_str()) == 0,
          "remove late segment");

    const std::string short_name = "/bc-v24-short-" + std::to_string(getpid());
    const int short_fd = shm_open(short_name.c_str(),
                                  O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    check(short_fd >= 0 && ftruncate(short_fd, (off_t)layout.bytes - 1) == 0,
          "create undersized segment");
    {
        Segment short_attach;
        check(!short_attach.open(short_name.c_str(), layout.bytes, false, 1) &&
              !short_attach.host(), "undersized segment must time out before mmap");
    }
    check(::close(short_fd) == 0 && shm_unlink(short_name.c_str()) == 0,
          "remove undersized segment");
    std::puts("r3_segment_version_test: ok");
}
