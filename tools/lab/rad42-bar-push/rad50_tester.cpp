// RAD50 single, auditable test runner. All tests are compiled into THIS binary.
// Never execute another user-named program, script, or shell command.
// The IPC test alone re-executes this same executable as a fixed child
// process with a pre-opened Unix-domain socket (the existing lab protocol).
// Test implementations remain lab-only and must be compiled into this binary.
// No production engine code, model serving, or vendor changes.
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <initializer_list>
#include <vector>
#include <fcntl.h>
#include <sys/socket.h>

// Original standalone entrypoints, renamed only at *compile* time.
// This deliberately reuses the exact tested implementations, not copies.
int rad50_pools_entry();
int rad50_serial_entry(int, char**);
int rad50_ipc_entry(int, char**);
int rad50_transfer_entry(int, char**);
int rad50_live_ready_entry(int, char**);
int rad50_vmem_cap_entry(int, char**);
int rad50_dmabuf_map_entry(int, char**);
int rad50_dmabuf_ab_entry(int, char**);
int rad50_combine_entry();
int rad50_cpu_handoff_entry(int, char**);

namespace {
enum class Kind {
    Pools, HipSmoke, Serial, Ipc, IpcMap, IpcGpuCpu, VmemCap, DmaBufMap, DmaBufWrite, DmaBufFull, DmaBufAB, CombineAB, DmaBufLive, DmaBufLiveFG, DmaBufLiveRelay, DmaBufLiveParallel, Transfer, LiveReady, Combine, CpuHandoff
};
struct Test {
    const char* name;
    Kind kind;
    bool needs_device;
    bool needs_lease;
    const char* description;
};
constexpr Test tests[] = {
    {"pools", Kind::Pools, false, true, "HSA GPU-local pool/CPU-access inventory (Gate 1)"},
    {"hip-smoke", Kind::HipSmoke, true, true, "HIP-only kernel launch; no BAR or HSA allocation (R9700 diagnostic)"},
    {"serial", Kind::Serial, true, true, "CPU store into HSA GPU-local VRAM, then GPU full-word check (Gate 2)"},
    {"ipc", Kind::Ipc, true, true, "CPU-only child imports GPU VRAM and writes; parent GPU verifies (Gate 2b)"},
    {"ipc-map", Kind::IpcMap, true, true, "GPU-agent-only same-device IPC import, CPU grant metadata; no writes"},
    {"ipc-gpu-cpu", Kind::IpcGpuCpu, true, true, "GPU-agent IPC import then CPU writes 4 KiB; parent GPU verifies"},
    {"vmem-cap", Kind::VmemCap, true, true, "GPU-local VMM DMA-BUF export and local CPU VMA capability"},
    {"dmabuf-map", Kind::DmaBufMap, true, true, "DMA-BUF FD passed to CPU-only child; mmap and SYNC ioctl only, NO writes"},
    {"dmabuf-write", Kind::DmaBufWrite, true, true, "CPU-only child writes 4KiB DMA-BUF; parent GPU verifies every word"},
    {"dmabuf-full", Kind::DmaBufFull, true, true, "CPU-only child writes 4MiB DMA-BUF; parent GPU verifies every word"},
    {"dmabuf-ab", Kind::DmaBufAB, true, true, "Persistent two-process GPU-pull vs CPU BAR-push A/B, exact GPU read and stage latency"},
    {"combine-ab", Kind::CombineAB, true, true, "Production AVX2/F16C FP16 combine into host vs DMA-BUF GPU VRAM; full GPU bit check"},
    {"dmabuf-live", Kind::DmaBufLive, true, true, "Cross-process CPU FP16 write while GPU is already waiting; system-scope generation/ACK"},
    {"dmabuf-live-fg", Kind::DmaBufLiveFG, true, true, "Experimental fine-grained host-register control; XTX-only verification candidate"},
    {"dmabuf-live-relay", Kind::DmaBufLiveRelay, true, true, "Coherent parent ready relay; GPU waiter + 256-block verifier; lab only"},
    {"dmabuf-live-parallel", Kind::DmaBufLiveParallel, true, true, "Direct child host-ready publication; GPU waiter + 256-block verifier; lab only"},
    {"transfer", Kind::Transfer, true, true, "Uncontended CPU BAR write versus mapped-host GPU pull (Gate 3)"},
    {"live-ready", Kind::LiveReady, true, true, "GPU already waiting before CPU BAR data and ready flag (Gate 4 canary)"},
    {"combine-ref", Kind::Combine, false, false, "Card-free FP16 combine/sentinel reference checks"},
    {"cpu-handoff", Kind::CpuHandoff, false, true, "Host-only CPU-to-CPU wake latency; NOT GPU notification"}
};
struct Args {
    const Test* test = nullptr;
    bool lease_ack = false;
    int device = -1;
    int iterations = -1;
    int selected_bytes = -1;
    int selected_rows = -1;
    int verifier_blocks = 256;
    int samples = -1;
    std::string mode;
};
void print_help(const char* binary) {
    std::printf(
        "RAD50 isolated fixed-test binary (no arbitrary command runner)\n"
        "usage: %s --list | --help | --self-check\n"
        "       %s --test NAME [--device HIP_ORDINAL] [--iterations 10..200]\n"
        "          [--mode spin|yield|sleep1|condvar|direct|staged]\n"
        "          [--samples 100..100000] [--bytes 4096|65536|262144|1048576|4194304]\n"
        "          [--rows 1|16|64|256|768]\n"
        "          [--verifier-blocks 1|32|64|128|256] (live parallel/relay only)\n"
        "          [--lease-confirmed]\n"
        "GPU and host-latency tests require --lease-confirmed. The flag is\n"
        "an operator declaration, not an automatic lease acquisition/check.\n"
        "Acquire the exclusive gpu lease with agent-lease and ensure no\n"
        "foreign GPU process exists BEFORE invoking this binary.\n"
        "The helper child mode is private to the fixed IPC protocol.\n",
        binary, binary);
}
void print_list() {
    std::puts("RAD50 compiled test cases:");
    for (const auto& t : tests)
        std::printf("  %-13s %s%s\n", t.name, t.description,
                    t.needs_lease ? " [exclusive lease required]" : "");
}
bool parse_int(const char* str, int lower, int upper, int* value) {
    if (!str || !*str) return false;
    if (*str == '-' || *str == '+') return false;
    errno = 0;
    char* end = nullptr;
    long n = std::strtol(str, &end, 10);
    if (errno || *end || n < lower || n > upper) return false;
    *value = static_cast<int>(n);
    return true;
}
const Test* find_test(const char* name) {
    for (const auto& t : tests)
        if (std::strcmp(t.name, name) == 0) return &t;
    return nullptr;
}
bool parse(int argc, char** argv, Args* result) {
    bool seen_test = false, seen_device = false, seen_iterations = false;
    bool seen_samples = false, seen_mode = false, seen_ack = false;
    bool seen_bytes = false, seen_rows = false, seen_verifier_blocks = false;
    for (int i=1; i<argc; ++i) {
        const char* a=argv[i];
        if (!std::strcmp(a,"--test")) {
            if (seen_test || i+1 >= argc) return false;
            result->test=find_test(argv[++i]);
            if (!result->test) return false;
            seen_test=true;
        } else if (!std::strcmp(a,"--device")) {
            if (seen_device || i+1>=argc ||
                !parse_int(argv[++i],0,127,&result->device)) return false;
            seen_device=true;
        } else if (!std::strcmp(a,"--iterations")) {
            if (seen_iterations || i+1>=argc ||
                !parse_int(argv[++i],10,200,&result->iterations)) return false;
            seen_iterations=true;
        } else if (!std::strcmp(a,"--bytes")) {
            if (seen_bytes || i+1>=argc ||
                !parse_int(argv[++i],4096,4194304,&result->selected_bytes)) return false;
            seen_bytes=true;
            if (result->selected_bytes!=4096 && result->selected_bytes!=65536 &&
                result->selected_bytes!=262144 && result->selected_bytes!=1048576 &&
                result->selected_bytes!=4194304) return false;
        } else if (!std::strcmp(a,"--rows")) {
            if (seen_rows || i+1>=argc ||
                !parse_int(argv[++i],1,768,&result->selected_rows)) return false;
            seen_rows=true;
            if (result->selected_rows!=1 && result->selected_rows!=16 &&
                result->selected_rows!=64 && result->selected_rows!=256 &&
                result->selected_rows!=768) return false;
        } else if (!std::strcmp(a,"--verifier-blocks")) {
            if (seen_verifier_blocks || i+1>=argc ||
                !parse_int(argv[++i],1,256,&result->verifier_blocks)) return false;
            seen_verifier_blocks=true;
            if (result->verifier_blocks!=1 && result->verifier_blocks!=32 &&
                result->verifier_blocks!=64 && result->verifier_blocks!=128 &&
                result->verifier_blocks!=256) return false;
        } else if (!std::strcmp(a,"--samples")) {
            if (seen_samples || i+1>=argc ||
                !parse_int(argv[++i],100,100000,&result->samples)) return false;
            seen_samples=true;
        } else if (!std::strcmp(a,"--mode")) {
            if (seen_mode || i+1>=argc) return false;
            result->mode=argv[++i]; seen_mode=true;
            if (result->mode!="spin" && result->mode!="yield" &&
                result->mode!="sleep1" && result->mode!="condvar" &&
                result->mode!="direct" && result->mode!="staged") return false;
        } else if (!std::strcmp(a,"--lease-confirmed")) {
            if (seen_ack) return false;
            result->lease_ack=true;seen_ack=true;
        } else return false; // No unknown flags, commands, or filenames.
    }
    if (!seen_test) return false;
    const Test& t=*result->test;
    if (t.needs_device != seen_device) return false;
    if (t.needs_lease && !seen_ack) return false;
    if (!t.needs_lease && seen_ack) return false;
    if ((t.kind==Kind::Transfer || t.kind==Kind::DmaBufAB ||
         t.kind==Kind::CombineAB || t.kind==Kind::DmaBufLive ||
         t.kind==Kind::DmaBufLiveFG || t.kind==Kind::DmaBufLiveRelay ||
         t.kind==Kind::DmaBufLiveParallel)!=seen_iterations) return false;
    if (seen_bytes && t.kind!=Kind::DmaBufAB) return false;
    if (seen_verifier_blocks && t.kind!=Kind::DmaBufLiveRelay &&
        t.kind!=Kind::DmaBufLiveParallel) return false;
    if (seen_rows && t.kind!=Kind::CombineAB &&
        t.kind!=Kind::DmaBufLive && t.kind!=Kind::DmaBufLiveFG &&
        t.kind!=Kind::DmaBufLiveRelay && t.kind!=Kind::DmaBufLiveParallel) return false;
    if ((t.kind==Kind::DmaBufLive || t.kind==Kind::DmaBufLiveFG ||
         t.kind==Kind::DmaBufLiveRelay || t.kind==Kind::DmaBufLiveParallel) && !seen_rows) return false;
    if ((t.kind==Kind::CpuHandoff || t.kind==Kind::DmaBufAB ||
         t.kind==Kind::CombineAB)!=seen_mode ||
        (t.kind==Kind::CpuHandoff)!=seen_samples) return false;
    if ((t.kind==Kind::DmaBufAB || t.kind==Kind::CombineAB) &&
        result->mode!="direct" && result->mode!="staged")return false;
    if (t.kind==Kind::CpuHandoff && (result->mode=="direct" || result->mode=="staged"))return false;
    return true;
}
int call_entry(int(*entry)(int,char**), const std::vector<std::string>& strings) {
    std::vector<char*> pointers;
    pointers.reserve(strings.size()+1);
    for (const std::string& s:strings)
        pointers.push_back(const_cast<char*>(s.c_str()));
    pointers.push_back(nullptr);
    return entry(static_cast<int>(strings.size()), pointers.data());
}
int run(const Args& a,const char* self) {
    const Test& t=*a.test;
    // The dispatch is closed over compiled entrypoints; no shell, arbitrary
    // path, user-defined program, script, or dynamic module is accepted.
    if (t.needs_lease)
        std::fprintf(stderr,
            "RAD50 lease acknowledged by operator; runner cannot verify lease ownership.\n");
    const std::string path(self);
    std::vector<std::string> args{path};
    switch(t.kind) {
      case Kind::Pools:
        return rad50_pools_entry();
      case Kind::HipSmoke:
        args.insert(args.end(),{"--device",std::to_string(a.device),"--probe-hip-only"});
        return call_entry(rad50_serial_entry,args);
      case Kind::Serial:
        args.insert(args.end(),{"--device",std::to_string(a.device)});
        return call_entry(rad50_serial_entry,args);
      case Kind::Ipc:
        args.insert(args.end(),{"--device",std::to_string(a.device)});
        return call_entry(rad50_ipc_entry,args);
      case Kind::IpcMap:
        args.insert(args.end(),{"--device",std::to_string(a.device),"--probe-attach-gpu-only"});
        return call_entry(rad50_ipc_entry,args);
      case Kind::IpcGpuCpu:
        args.insert(args.end(),{"--device",std::to_string(a.device),"--probe-gpu-then-cpu"});
        return call_entry(rad50_ipc_entry,args);
      case Kind::VmemCap:
        args.insert(args.end(),{"--device",std::to_string(a.device)});
        return call_entry(rad50_vmem_cap_entry,args);
      case Kind::DmaBufMap:
        args.insert(args.end(),{"--device",std::to_string(a.device)});
        return call_entry(rad50_dmabuf_map_entry,args);
      case Kind::DmaBufWrite:
        args.insert(args.end(),{"--device",std::to_string(a.device),"--write-verify"});
        return call_entry(rad50_dmabuf_map_entry,args);
      case Kind::DmaBufFull:
        args.insert(args.end(),{"--device",std::to_string(a.device),"--full-verify"});
        return call_entry(rad50_dmabuf_map_entry,args);
      case Kind::DmaBufAB:
        args.insert(args.end(),{"--device",std::to_string(a.device),
                                "--iterations",std::to_string(a.iterations),
                                "--mode",a.mode});
        if (a.selected_bytes>0) args.insert(args.end(),{"--bytes",std::to_string(a.selected_bytes)});
        return call_entry(rad50_dmabuf_ab_entry,args);
      case Kind::CombineAB:
        args.insert(args.end(),{"--device",std::to_string(a.device),
                                "--iterations",std::to_string(a.iterations),
                                "--mode",a.mode=="direct"?"combine-direct":"combine-staged"});
        if (a.selected_rows>0)
            args.insert(args.end(),{"--bytes",std::to_string(a.selected_rows*5120)});
        return call_entry(rad50_dmabuf_ab_entry,args);
      case Kind::DmaBufLive:
        args.insert(args.end(),{"--live","--device",std::to_string(a.device),
                                "--iterations",std::to_string(a.iterations),
                                "--rows",std::to_string(a.selected_rows)});
        return call_entry(rad50_dmabuf_ab_entry,args);
      case Kind::DmaBufLiveFG:
        args.insert(args.end(),{"--live-fg","--device",std::to_string(a.device),
                                "--iterations",std::to_string(a.iterations),
                                "--rows",std::to_string(a.selected_rows)});
        return call_entry(rad50_dmabuf_ab_entry,args);
      case Kind::DmaBufLiveRelay:
        args.insert(args.end(),{"--live-relay","--device",std::to_string(a.device),
                                "--iterations",std::to_string(a.iterations),
                                "--rows",std::to_string(a.selected_rows),
                                "--verifier-blocks",std::to_string(a.verifier_blocks)});
        return call_entry(rad50_dmabuf_ab_entry,args);
      case Kind::DmaBufLiveParallel:
        args.insert(args.end(),{"--live-parallel","--device",std::to_string(a.device),
                                "--iterations",std::to_string(a.iterations),
                                "--rows",std::to_string(a.selected_rows),
                                "--verifier-blocks",std::to_string(a.verifier_blocks)});
        return call_entry(rad50_dmabuf_ab_entry,args);
      case Kind::Transfer:
        args.insert(args.end(),{"--device",std::to_string(a.device),
                                "--iterations",std::to_string(a.iterations)});
        return call_entry(rad50_transfer_entry,args);
      case Kind::LiveReady:
        args.insert(args.end(),{"--device",std::to_string(a.device)});
        return call_entry(rad50_live_ready_entry,args);
      case Kind::Combine:
        return rad50_combine_entry();
      case Kind::CpuHandoff:
        args.insert(args.end(),{"--mode",a.mode,"--samples",std::to_string(a.samples)});
        return call_entry(rad50_cpu_handoff_entry,args);
    }
    return 2;
}
bool is_fixed_ipc_child(int argc,char** argv) {
    if (argc!=3 || std::strcmp(argv[1],"--helper") ||
        std::strcmp(argv[2],"3")) return false;
    // Never start child protocol unless the parent passed an actual
    // Unix-domain SOCK_STREAM on descriptor 3.
    int type=0;
    socklen_t n=sizeof(type);
    return fcntl(3,F_GETFD)!=-1 &&
           getsockopt(3,SOL_SOCKET,SO_TYPE,&type,&n)==0 &&
           n==sizeof(type) && type==SOCK_STREAM;
}
#include "tester_selfcheck.inc"
} // namespace

int main(int argc,char** argv) {
    if (argc==2 && (!std::strcmp(argv[1],"--help") ||
                    !std::strcmp(argv[1],"-h"))) {
        print_help(argv[0]); return 0;
    }
    if (argc==2 && !std::strcmp(argv[1],"--list")) {
        print_list(); return 0;
    }
    if (argc==2 && !std::strcmp(argv[1],"--self-check")) return cli_self_check();
    if (is_fixed_ipc_child(argc,argv))
        return rad50_ipc_entry(argc,argv);
    if (argc==3 && !std::strcmp(argv[1],"--dmabuf-child") &&
        !std::strcmp(argv[2],"3") && fcntl(3,F_GETFD)!=-1) {
        int type=0; socklen_t size=sizeof(type);
        if (getsockopt(3,SOL_SOCKET,SO_TYPE,&type,&size)==0 &&
            size==sizeof(type) && type==SOCK_STREAM)
            return rad50_dmabuf_map_entry(argc,argv);
    }
    if (argc==3 && !std::strcmp(argv[1],"--bench-child") &&
        !std::strcmp(argv[2],"3") && fcntl(3,F_GETFD)!=-1) {
        int type=0; socklen_t n=sizeof(type);
        if (getsockopt(3,SOL_SOCKET,SO_TYPE,&type,&n)==0 &&
            n==sizeof(type) && type==SOCK_STREAM)
            return rad50_dmabuf_ab_entry(argc,argv);
    }
    Args args;
    if (!parse(argc,argv,&args)) {
        std::fputs("RAD50: unknown test or invalid arguments; use --help/--list.\n",
                   stderr);
        return 2;
    }
    std::printf("RAD50 case=%s device=%d (no production engine integration)\n",
                args.test->name,args.device);
    std::fflush(stdout);
    const int result=run(args,argv[0]);
    std::fprintf(stderr,"RAD50 RESULT test=%s exit=%d\n",
                 args.test->name,result);
    return result;
}
