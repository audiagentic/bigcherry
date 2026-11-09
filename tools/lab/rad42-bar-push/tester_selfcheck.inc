// Included inside the test runner's unnamed namespace.
// Card-free parser regression cases; no HIP/HSA allocations or test dispatch.
int cli_self_check() {
    int failures=0;
    int checks=0;
    auto check=[&](std::initializer_list<const char*> arguments,
                  bool expected, const char* label) {
        std::vector<char*> av;
        for (const char* s:arguments) av.push_back(const_cast<char*>(s));
        Args args;
        const bool actual=parse(static_cast<int>(av.size()),av.data(),&args);
        ++checks;
        if (actual!=expected) {
            ++failures;
            std::fprintf(stderr,"RAD50 CLI FAIL: %s (got=%d expected=%d)\n",
                         label,(int)actual,(int)expected);
        }
    };
    check({"rad50-tester","--test","combine-ref"},true,"card-free reference");
    check({"rad50-tester","--test","pools","--lease-confirmed"},true,"HSA inventory");
    check({"rad50-tester","--test","serial","--device","0","--lease-confirmed"},
          true,"XTX serial");
    check({"rad50-tester","--test","hip-smoke","--device","2","--lease-confirmed"},
          true,"R9700 HIP diagnosis");
    check({"rad50-tester","--test","ipc","--device","1","--lease-confirmed"},
          true,"IPC exact mode");
    check({"rad50-tester","--test","ipc-map","--device","0","--lease-confirmed"},
          true,"IPC GPU-only attach diagnostics");
    check({"rad50-tester","--test","ipc-gpu-cpu","--device","0","--lease-confirmed"},
          true,"IPC GPU-first CPU BAR verification");
    check({"rad50-tester","--test","dmabuf-full","--device","1","--lease-confirmed"},
          true,"DMA-BUF 4MiB CPU write and GPU verification");
    check({"rad50-tester","--test","dmabuf-full","--device","0"},
          false,"DMA-BUF full without lease");
    check({"rad50-tester","--test","dmabuf-write","--device","0","--lease-confirmed"},
          true,"DMA-BUF 4KiB CPU write and GPU verification");
    check({"rad50-tester","--test","dmabuf-write","--device","0"},
          false,"DMA-BUF write without lease");
    check({"rad50-tester","--test","dmabuf-map","--device","0","--lease-confirmed"},
          true,"DMA-BUF mmap capability mode");
    check({"rad50-tester","--test","dmabuf-map","--device","0"},
          false,"DMA-BUF mode without lease");
    check({"rad50-tester","--test","vmem-cap","--device","0","--lease-confirmed"},
          true,"GPU-local VMM export capability");
    check({"rad50-tester","--test","dmabuf-ab","--device","0","--iterations","10",
           "--mode","direct","--bytes","4096","--lease-confirmed"},
          true,"short 4KiB A/B");
    check({"rad50-tester","--test","dmabuf-ab","--device","1","--iterations","10",
           "--mode","staged","--bytes","4194304","--lease-confirmed"},
          true,"short 4MiB A/B");
    check({"rad50-tester","--test","dmabuf-ab","--device","0","--iterations","10",
           "--mode","direct","--bytes","8000","--lease-confirmed"},
          false,"A/B invalid payload size");
    check({"rad50-tester","--test","dmabuf-ab","--device","0","--iterations","10",
           "--mode","direct","--bytes","4096","--bytes","65536","--lease-confirmed"},
          false,"A/B duplicate bytes selector");
    check({"rad50-tester","--test","serial","--device","0","--bytes","4096",
           "--lease-confirmed"},false,"A/B payload selector on different test");
    check({"rad50-tester","--test","dmabuf-ab","--device","0","--iterations","30",
           "--mode","direct","--lease-confirmed"},true,"two-process direct A/B");
    check({"rad50-tester","--test","dmabuf-ab","--device","1","--iterations","10",
           "--mode","staged","--lease-confirmed"},true,"two-process staged A/B");
    check({"rad50-tester","--test","dmabuf-ab","--device","0","--iterations","30",
           "--lease-confirmed"},false,"A/B requires explicit method");
    check({"rad50-tester","--test","dmabuf-ab","--device","0","--iterations","30",
           "--mode","yield","--lease-confirmed"},false,"A/B rejects CPU handoff mode");
    check({"rad50-tester","--test","dmabuf-ab","--device","0","--iterations","500",
           "--mode","direct","--lease-confirmed"},false,"A/B bounded iterations");
    check({"rad50-tester","--test","cpu-handoff","--mode","direct",
           "--samples","3000","--lease-confirmed"},false,"handoff rejects transfer mode");
    check({"rad50-tester","--test","transfer","--device","0","--iterations","30",
           "--lease-confirmed"},true,"A/B bounded mode");
    check({"rad50-tester","--test","live-ready","--device","0","--lease-confirmed"},
          true,"live waiter");
    check({"rad50-tester","--test","cpu-handoff","--mode","yield",
           "--samples","3000","--lease-confirmed"},true,"CPU latency mode");
    check({"rad50-tester","--test","serial","--device","0"},false,"missing lease");
    check({"rad50-tester","--test","ipc","--lease-confirmed"},false,"missing device");
    check({"rad50-tester","--test","combine-ref","--lease-confirmed"},
          false,"unneeded lease ack");
    check({"rad50-tester","--test","transfer","--device","0","--iterations","201",
           "--lease-confirmed"},false,"too many iterations");
    check({"rad50-tester","--test","transfer","--device","0","--lease-confirmed"},
          false,"missing iteration count");
    check({"rad50-tester","--test","serial","--device","-1","--lease-confirmed"},
          false,"negative GPU index");
    check({"rad50-tester","--test","serial","--device","0","--device","1",
           "--lease-confirmed"},false,"duplicate GPU index");
    check({"rad50-tester","--test","cpu-handoff","--mode","invalid","--samples","3000",
           "--lease-confirmed"},false,"unknown wake strategy");
    check({"rad50-tester","--test","unknown","--lease-confirmed"},false,"unknown test");
    check({"rad50-tester","--test","serial","--device","0","--program","/bin/sh",
           "--lease-confirmed"},false,"arbitrary command disallowed");
    check({"rad50-tester","--helper","3"},false,"internal mode not public parser");
    #include "combine_cli_selfcheck.inc"
    #include "live_cli_selfcheck.inc"
    std::printf("RAD50 CLI self-check: %d/%d passed (no GPU)\n",checks-failures,checks);
    return failures?1:0;
}
