// RHP11 F: Segment::open refuses a non-v23 shared mailbox before
// registration. HIP functions are stubs: this test uses NO GPU.
#include "moe_remote_shm.h"
#include <cstdio>
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
    const std::string name="/bc-v23-"+std::to_string(getpid());
    Segment creator;
    check(creator.open(name.c_str(),layout.bytes,true,1),"create");
    check(init(creator.host(),2,shape,2),"valid v23 init");
    auto* h=creator.header();
    h->helper_count=2;
    h->version=22;
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
              "v23 accepted by actual Segment::open");
        check(attached.register_gpu() && registers==1,
              "validated v23 passed to fake register");
        auto* g=generation_header(creator.host(),0);
        const uint64_t saved=g->digest;
        g->digest^=1;
        check(!creator.register_gpu() && registers==1,
              "bad v23 generation never registered");
        g->digest=saved;
    }
    std::puts("r3_segment_version_test: ok");
}
