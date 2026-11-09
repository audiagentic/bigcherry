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
    std::printf("RAD50 CLI self-check: %d/%d passed (no GPU)\n",checks-failures,checks);
    return failures?1:0;
}
