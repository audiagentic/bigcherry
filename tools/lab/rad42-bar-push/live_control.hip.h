// RAD50 Gate 4: observed cross-process GPU-live visibility canary.
// Included inside the lab-only A/B runner, with same real FP16 oracle.
// This is hardware evidence, not a general ROCm memory-order guarantee.
// IMPORTANT: exactly ONE GPU lane polls the CPU-host control address.
// Having 256 lanes independently poll pinned host memory caused a 768-row
// readiness timeout in the first hardware trial; reducing host reads is
// a hypothesis for the next controlled diagnostic, not a validated fix.
struct alignas(64) LiveControl {
    uint32_t armed=0,ready=0,done=0,errors=0;
    uint32_t gpu_seen_ready=0,spin_loops=0;
    uint64_t gpu_wait_cycles=0; // diagnostic clock64 delta, not elapsed microseconds
};
__global__ void live_combine_reader(const uint16_t* gpu_local,
                                    uint32_t count,uint32_t gen,
                                    LiveControl* ctrl,uint64_t deadline_wall_ticks) {
    if(blockIdx.x!=0)return;
    __shared__ uint32_t shared_bad[256];
    __shared__ uint32_t wait_failed;
    if(threadIdx.x==0){
        __hip_atomic_store(&ctrl->armed,gen,__ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
        // clock64() is unreliable on GFX11. wall_clock64() runs at the
        // frequency reported by hipDeviceAttributeWallClockRate.
        const uint64_t start=static_cast<uint64_t>(wall_clock64());
        uint32_t spins=0;
        uint32_t seen=0;
        while((seen=__hip_atomic_load(&ctrl->ready,__ATOMIC_ACQUIRE,
                                     __HIP_MEMORY_SCOPE_SYSTEM))!=gen){
            __builtin_amdgcn_s_sleep(8);
            if(++spins>100000000u ||
               static_cast<uint64_t>(wall_clock64())-start>deadline_wall_ticks)
                break;
        }
        seen=__hip_atomic_load(&ctrl->ready,__ATOMIC_ACQUIRE,
                               __HIP_MEMORY_SCOPE_SYSTEM);
        wait_failed=seen!=gen;
        __hip_atomic_store(&ctrl->gpu_seen_ready,seen,__ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(&ctrl->spin_loops,spins,__ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();
    unsigned bad=wait_failed?1u:0u;
    if(!wait_failed) {
        for(uint32_t i=threadIdx.x;i<count;i+=blockDim.x) {
            const uint32_t profile=(i/COMBINE_WIDTH+gen)&3u;
            bad+=gpu_local[i]!=expected_combine_word(profile,i%COMBINE_WIDTH);
        }
    }
    shared_bad[threadIdx.x]=bad;
    __syncthreads();
    for(unsigned step=blockDim.x/2;step>0;step>>=1){
        if(threadIdx.x<step)shared_bad[threadIdx.x]+=shared_bad[threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x==0){
        __hip_atomic_store(&ctrl->errors,
                           wait_failed?0xffffffffu:shared_bad[0],
                           __ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
        __threadfence_system();
        __hip_atomic_store(&ctrl->done,gen,__ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
    }
}
bool host_wait_ticket(const uint32_t* at,uint32_t ticket,int millis){
    const auto until=std::chrono::steady_clock::now()+
                     std::chrono::milliseconds(millis);
    while(__atomic_load_n(at,__ATOMIC_ACQUIRE)!=ticket){
        if(std::chrono::steady_clock::now()>until)return false;
        std::this_thread::yield();
    }
    return true;
}
