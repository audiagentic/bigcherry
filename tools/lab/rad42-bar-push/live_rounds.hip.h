// Cross-process live GPU waiter after protocol and worker() declarations.
// This is a bounded, fail-closed hardware canary, not a production protocol.
bool live_rounds(int sock,int device,unsigned iterations,uint32_t rows,
                 LiveControl* ctrl,LiveControl* device_ctrl,
                 const void* device_va,uint64_t deadline_wall_ticks) {
    if(!rows || rows>768 || !ctrl || !device_ctrl || !device_va ||
       deadline_wall_ticks==0)return false;
    const uint32_t bytes=rows*COMBINE_ROW_BYTES;
    if(bytes>MAX_BYTES)return false;
    std::vector<double> latency,write_times,sync_times;
    latency.reserve(iterations);
    for(uint32_t n=0;n<iterations;++n) {
        // All reset/reuse happens ONLY after the previous iteration's
        // GPU acknowledgment AND hipDeviceSynchronize. Keep ready stale
        // on purpose: a fresh kernel must match the exact new ticket.
        const uint32_t ticket=100000u+rows*1000u+n+1u;
        __atomic_store_n(&ctrl->armed,0u,__ATOMIC_RELAXED);
        __atomic_store_n(&ctrl->done,0u,__ATOMIC_RELAXED);
        __atomic_store_n(&ctrl->errors,0u,__ATOMIC_RELAXED);
        __atomic_store_n(&ctrl->gpu_seen_ready,0u,__ATOMIC_RELAXED);
        __atomic_store_n(&ctrl->spin_loops,0u,__ATOMIC_RELAXED);
        const auto* source=static_cast<const uint16_t*>(device_va);
        const double begin=tick_us();
        hipLaunchKernelGGL(live_combine_reader,dim3(1),dim3(256),0,0,
                           source,bytes/2,ticket,device_ctrl,deadline_wall_ticks);
        if(!check_hip(hipGetLastError(),"RAD50 live GPU launch"))return false;
        // No HIP device synchronization before the CPU child writes!
        // The GPU kernel must already have reached its wait loop.
        if(!host_wait_ticket(&ctrl->armed,ticket,3500)) {
            std::fprintf(stderr,
                "RAD50_LIVE_FAIL device=%d rows=%u gen=%u reason=arm-timeout\n",
                device,rows,ticket);
            return false;
        }
        const double armed=tick_us();
        const Command cmd{MAGIC,2u,bytes,ticket,2u};
        Ack ack{};
        if(!write_exact(sock,&cmd,sizeof(cmd)) ||
           !read_exact(sock,&ack,sizeof(ack)) ||
           ack.magic!=MAGIC || ack.generation!=ticket || ack.status!=0){
            std::fprintf(stderr,
                "RAD50_LIVE_FAIL device=%d rows=%u gen=%u reason=cpu-helper-ack status=%u\n",
                device,rows,ticket,ack.status);
            return false;
        }
        const double child_acked=tick_us();
        if(!host_wait_ticket(&ctrl->done,ticket,5000)) {
            std::fprintf(stderr,
                "RAD50_LIVE_FAIL device=%d rows=%u gen=%u reason=done-timeout\n",
                device,rows,ticket);
            return false;
        }
        const bool drained=check_hip(hipDeviceSynchronize(),"RAD50 live kernel drain");
        const unsigned bad=__atomic_load_n(&ctrl->errors,__ATOMIC_ACQUIRE);
        const unsigned observed=__atomic_load_n(&ctrl->gpu_seen_ready,__ATOMIC_ACQUIRE);
        const unsigned spins=__atomic_load_n(&ctrl->spin_loops,__ATOMIC_ACQUIRE);
        const unsigned host_ready=__atomic_load_n(&ctrl->ready,__ATOMIC_ACQUIRE);
        if(!drained||bad) {
            std::fprintf(stderr,
                "RAD50_LIVE_FAIL device=%d rows=%u gen=%u "
                "reason=wrong-fp16-or-ready-timeout errors=%u "
                "host_ready=%u gpu_seen_ready=%u wait_loops=%u "
                "helper_ack_after_arm_us=%.2f\n",
                device,rows,ticket,bad,host_ready,observed,spins,
                child_acked-armed);
            return false;
        }
        const double completed=tick_us();
        const double sync=ack.sync_start_us+ack.sync_end_us;
        const double elapsed=completed-begin;
        latency.push_back(elapsed);write_times.push_back(ack.write_us);
        sync_times.push_back(sync);
        std::printf(
            "RAD50_LIVE_SAMPLE device=%d rows=%u bytes=%u gen=%u "
            "armed_before_helper=1 stale_ready_retained=1 "
            "cpu_combine_us=%.2f dma_sync_us=%.2f "
            "helper_ack_after_arm_us=%.2f serial_us=%.2f "
            "gpu_seen_ready=%u wait_loops=%u mismatches=0 status=0\n",
            device,rows,bytes,ticket,ack.write_us,sync,
            child_acked-armed,elapsed,observed,spins);
        std::fflush(stdout); // Preserve even partial trials on failure.
    }
    auto quant=[](std::vector<double> v,double q) {
        std::sort(v.begin(),v.end());
        return v[static_cast<size_t>((v.size()-1)*q)];
    };
    std::printf("RAD50_LIVE_SUMMARY device=%d rows=%u bytes=%u "
                "iterations=%u verified=%u total_p50_us=%.2f "
                "total_p95_us=%.2f cpu_combine_p50_us=%.2f "
                "sync_p50_us=%.2f status=0\n",
                device,rows,bytes,iterations,iterations,
                quant(latency,.5),quant(latency,.95),
                quant(write_times,.5),quant(sync_times,.5));
    return true;
}
