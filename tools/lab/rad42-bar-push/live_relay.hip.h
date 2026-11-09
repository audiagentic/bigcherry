// RAD50 Gate 4 slice isolation: coherent parent relay + separate GPU waiter /
// parallel FP16 payload verification. Lab-only; not an engine protocol.
//
// The independent CPU helper writes into GPU-local DMA-BUF and executes
// SYNC_END before its socket ACK. The GPU parent publishes ready only
// after consuming that ACK, using a separate hipHostMallocCoherent|Mapped
// control allocation. This specifically tests control visibility without
// changing payload transport or adding a second GPU memory copy.
//
// The GPU waiter runs BEFORE CPU writes. Verification and done publication
// are enqueued behind it in the SAME stream; verification uses 256 GPU blocks
// rather than one block scanning the entire payload.
__global__ void relay_waiter(LiveControl* control,uint32_t generation,
                             unsigned* timeout_flag,uint64_t deadline_wall_ticks) {
    if(blockIdx.x || threadIdx.x)return;
    __hip_atomic_store(&control->armed,generation,__ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
    // HIP documents clock64() as broken on gfx11; wall_clock64() is
    // fixed frequency and its rate is queried on the host.
    const uint64_t start=static_cast<uint64_t>(wall_clock64());
    unsigned spins=0;
    uint32_t seen=0;
    unsigned reason=0; // 0=matched, 1=spin cap, 2=clock deadline
    while((seen=__hip_atomic_load(&control->ready,__ATOMIC_ACQUIRE,
                                  __HIP_MEMORY_SCOPE_SYSTEM))!=generation) {
        __builtin_amdgcn_s_sleep(8);
        if(++spins>100000000u){reason=1;break;}
        if(static_cast<uint64_t>(wall_clock64())-start > deadline_wall_ticks){
            reason=2;break;
        }
    }
    const uint32_t exit_seen=seen;
    const uint32_t reread=__hip_atomic_load(&control->ready,__ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
    // Device-local diagnostics are reliable even if the mapped-host
    // control records are not coherent in a running kernel.
    timeout_flag[0]=(reason || exit_seen!=generation || reread!=generation)?1u:0u;
    timeout_flag[1]=exit_seen;
    timeout_flag[2]=reread;
    timeout_flag[3]=reason;
    timeout_flag[4]=spins;
    control->gpu_wait_cycles=static_cast<uint64_t>(wall_clock64())-start;
    seen=exit_seen;
    __hip_atomic_store(&control->gpu_seen_ready,seen,__ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(&control->spin_loops,spins,__ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
}
__global__ void relay_verify(const uint16_t* output,uint32_t count,
                             uint32_t generation,const unsigned* timeout_flag,
                             unsigned* errors) {
    if(timeout_flag[0])return;
    const uint32_t id=blockIdx.x*blockDim.x+threadIdx.x;
    const uint32_t step=blockDim.x*gridDim.x;
    unsigned bad=0;
    for(uint32_t i=id;i<count;i+=step) {
        const uint32_t profile=(i/COMBINE_WIDTH+generation)&3u;
        bad+=output[i]!=expected_combine_word(profile,i%COMBINE_WIDTH);
    }
    if(bad)atomicAdd(errors,bad);
}
__global__ void relay_publish_done(LiveControl* control,uint32_t generation,
                                   const unsigned* timeout_flag,
                                   const unsigned* mismatches) {
    if(blockIdx.x || threadIdx.x)return;
    const unsigned bad=timeout_flag[0]?0xffffffffu:mismatches[0];
    __hip_atomic_store(&control->errors,bad,__ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
    __threadfence_system();
    __hip_atomic_store(&control->done,generation,__ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
}
bool relay_rounds(int sock,int device,unsigned iterations,uint32_t rows,
                  LiveControl* host,LiveControl* device_control,
                  const void* gpu_local,unsigned* errors,unsigned* wait_timeout,
                  uint64_t deadline_wall_ticks,bool parent_publish) {
    if(!host || !device_control || !gpu_local || !errors || !wait_timeout ||
       !rows || rows>768 || !iterations || iterations>200 ||
       deadline_wall_ticks==0)return false;
    const uint32_t bytes=rows*COMBINE_ROW_BYTES;
    if(bytes>MAX_BYTES)return false;
    std::vector<double> total,combine,cpu_ack_us,gpu_after_ready_us,arm_us;
    total.reserve(iterations);
    for(unsigned n=0;n<iterations;++n) {
        const uint32_t generation=100000u+rows*1000u+n+1u;
        // The GPU has fully drained the previous generation before reuse.
        // Keep ready stale intentionally to prove exact-ticket matching.
        __atomic_store_n(&host->armed,0u,__ATOMIC_RELAXED);
        __atomic_store_n(&host->done,0u,__ATOMIC_RELAXED);
        __atomic_store_n(&host->errors,0u,__ATOMIC_RELAXED);
        __atomic_store_n(&host->gpu_seen_ready,0u,__ATOMIC_RELAXED);
        __atomic_store_n(&host->spin_loops,0u,__ATOMIC_RELAXED);
        host->gpu_wait_cycles=0u;
        if(!check_hip(hipMemset(errors,0,sizeof(unsigned)),
                      "relay mismatch buffer reset") ||
           !check_hip(hipMemset(wait_timeout,0,5*sizeof(unsigned)),
                      "relay timeout buffer reset"))return false;
        const double start=tick_us();
        hipLaunchKernelGGL(relay_waiter,dim3(1),dim3(1),0,0,
                           device_control,generation,wait_timeout,
                           deadline_wall_ticks);
        hipLaunchKernelGGL(relay_verify,dim3(256),dim3(256),0,0,
                           static_cast<const uint16_t*>(gpu_local),
                           bytes/2,generation,wait_timeout,errors);
        hipLaunchKernelGGL(relay_publish_done,dim3(1),dim3(1),0,0,
                           device_control,generation,wait_timeout,errors);
        if(!check_hip(hipGetLastError(),"relay queued GPU consumer"))return false;
        // The first GPU kernel must be live BEFORE we command CPU work.
        if(!host_wait_ticket(&host->armed,generation,3500)) {
            std::fprintf(stderr,
                "RAD50_RELAY_FAIL device=%d rows=%u gen=%u reason=arm-timeout\n",
                device,rows,generation);
            return false;
        }
        const double armed=tick_us();
        const Command cmd{MAGIC,2u,bytes,generation,2u};
        Ack reply{};
        if(!write_exact(sock,&cmd,sizeof(cmd)) ||
           !read_exact(sock,&reply,sizeof(reply)) ||
           reply.magic!=MAGIC || reply.generation!=generation || reply.status!=0) {
            std::fprintf(stderr,
                "RAD50_RELAY_FAIL device=%d rows=%u gen=%u reason=child-ack status=%u\n",
                device,rows,generation,reply.status);
            return false;
        }
        const double acked=tick_us();
        // Parent is the ONLY writer of the GPU-visible coherent ready flag.
        // This happens after the helper completed CPU combine+SYNC_END.
        if(parent_publish)
            __atomic_store_n(&host->ready,generation,__ATOMIC_RELEASE);
        else if(__atomic_load_n(&host->ready,__ATOMIC_ACQUIRE)!=generation) {
            std::fprintf(stderr,
                "RAD50_PARALLEL_FAIL device=%d rows=%u gen=%u reason=child-ready-not-visible-to-parent\n",
                device,rows,generation);
            return false;
        }
        const double published=tick_us();
        if(!host_wait_ticket(&host->done,generation,5000)){
            std::fprintf(stderr,
                "RAD50_RELAY_FAIL device=%d rows=%u gen=%u reason=done-timeout "
                "host_ready=%u gpu_seen_ready=%u\n",
                device,rows,generation,
                __atomic_load_n(&host->ready,__ATOMIC_ACQUIRE),
                __atomic_load_n(&host->gpu_seen_ready,__ATOMIC_ACQUIRE));
            return false;
        }
        const bool drained=check_hip(hipDeviceSynchronize(),
                                     "relay GPU kernel drain");
        const unsigned bad=__atomic_load_n(&host->errors,__ATOMIC_ACQUIRE);
        const unsigned gpu_seen=__atomic_load_n(&host->gpu_seen_ready,
                                                 __ATOMIC_ACQUIRE);
        const unsigned spins=__atomic_load_n(&host->spin_loops,__ATOMIC_ACQUIRE);
        unsigned gpu_bad=0;
        unsigned wait_diag[5]{};
        const bool copied=check_hip(hipMemcpy(&gpu_bad,errors,sizeof(unsigned),
                              hipMemcpyDeviceToHost),"relay GPU verify errors") &&
                          check_hip(hipMemcpy(wait_diag,wait_timeout,
                              sizeof(wait_diag),hipMemcpyDeviceToHost),
                              "relay GPU wait timeout");
        const double end=tick_us();
        if(!drained || !copied || bad || gpu_bad || wait_diag[0] ||
           gpu_seen!=generation){
            std::fprintf(stderr,
                "RAD50_RELAY_FAIL device=%d rows=%u gen=%u reason=ready-or-data "
                "bad=%u gpu_bad=%u gpu_timeout=%u host_ready=%u seen=%u spins=%u "
                "gpu_local_seen=%u reread=%u exit_reason=%u local_spins=%u "
                "gpu_wait_wall_ticks=%llu child_ack_after_arm_us=%.2f "
                "gpu_after_publish_us=%.2f total_us=%.2f\n",
                device,rows,generation,bad,gpu_bad,wait_diag[0],
                __atomic_load_n(&host->ready,__ATOMIC_ACQUIRE),
                gpu_seen,spins,wait_diag[1],wait_diag[2],wait_diag[3],wait_diag[4],
                (unsigned long long)host->gpu_wait_cycles,
                acked-armed,end-published,end-start);
            return false;
        }
        total.push_back(end-start);
        combine.push_back(reply.write_us);
        cpu_ack_us.push_back(acked-armed);
        arm_us.push_back(armed-start);
        gpu_after_ready_us.push_back(end-published);
        // Long runs produce bounded stdout. Keep early, every tenth and final
        // samples plus all failure records; retain every measurement in
        // in-process percentile vectors, never suppress a failure.
        if(iterations<=15 || n<3u || (n+1u)%10u==0u || n+1u==iterations) {
        std::printf("RAD50_%s_SAMPLE device=%d rows=%u bytes=%u gen=%u "
                    "control=%s delivery=%s gpu_parallel_blocks=256 "
                    "arm_us=%.2f cpu_combine_us=%.2f dma_sync_us=%.2f "
                    "helper_ack_us=%.2f post_ack_complete_us=%.2f "
                    "serial_us=%.2f gpu_seen_ready=%u wait_loops=%u "
                    "mismatches=0 status=0\n",
                    parent_publish?"RELAY":"PARALLEL",device,rows,bytes,generation,
                    parent_publish?"hip-host-coherent":"hip-register-mapped",
                    parent_publish?"parent-after-socket-ack":"child-after-dma-sync-end",
                    armed-start,reply.write_us,
                    reply.sync_start_us+reply.sync_end_us,acked-armed,
                    end-published,end-start,gpu_seen,spins);
        std::fflush(stdout); // Partial logs remain useful.
        }
    }
    auto median=[](std::vector<double> v){
        std::sort(v.begin(),v.end());
        return v[(v.size()-1)/2];
    };
    auto quantile=[](std::vector<double> v,double q){
        std::sort(v.begin(),v.end());
        return v[static_cast<size_t>(q*(v.size()-1))];
    };
    std::printf("RAD50_%s_SUMMARY device=%d rows=%u bytes=%u "
                "iterations=%u verified=%u p50_us=%.2f p95_us=%.2f "
                "p99_us=%.2f max_us=%.2f emitted=%u "
                "combine_p50_us=%.2f arm_p50_us=%.2f ack_p50_us=%.2f "
                "post_ack_complete_p50_us=%.2f status=0\n",
                parent_publish?"RELAY":"PARALLEL",device,rows,bytes,
                iterations,iterations,median(total),quantile(total,.95),
                quantile(total,.99),*std::max_element(total.begin(),total.end()),
                iterations<=15?iterations:(3u+iterations/10u+(iterations%10u?1u:0u)),
                median(combine),median(arm_us),median(cpu_ack_us),
                median(gpu_after_ready_us));
    return true;
}
