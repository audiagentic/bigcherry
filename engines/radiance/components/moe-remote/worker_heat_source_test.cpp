#include "worker_heat_source.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>

using namespace r3::policy;
static void check(bool ok, const char* reason) {
    if (!ok) { std::fprintf(stderr, "r3_heat_source_test: FAIL %s\n", reason); std::exit(1); }
}
static uint64_t realtime() {
    timespec t{}; check(clock_gettime(CLOCK_REALTIME, &t) == 0, "clock");
    return uint64_t(t.tv_sec)*1000000000ULL+uint64_t(t.tv_nsec);
}
struct Fixture {
    std::string dir, path;
    std::vector<uint32_t> counts;
    Fixture() : counts(size_t(r3::moe_shm::kMaxLayers)*4,0) {
        const char* env = std::getenv("TMPDIR");
        std::string tmp = std::string(env && *env ? env : "/tmp") + "/r3heat-XXXXXX";
        std::vector<char> name(tmp.begin(),tmp.end()); name.push_back('\0');
        char* made = ::mkdtemp(name.data()); check(made != nullptr,"mkdtemp");
        dir=made; path=dir+"/routes.rank0";
    }
    ~Fixture() { ::unlink(path.c_str()); ::unlink((path+".tmp").c_str()); ::rmdir(dir.c_str()); }
    void write() {
        const std::string tmp = path+".tmp";
        FILE* f = std::fopen(tmp.c_str(),"wb"); check(f != nullptr,"open");
        check(std::fwrite(counts.data(),sizeof(uint32_t),counts.size(),f)==counts.size(),"write");
        check(std::fclose(f)==0 && std::rename(tmp.c_str(),path.c_str())==0,"atomic rename");
    }
};
int main() {
    Fixture f;
    HistogramFileSource src(f.path,2,4);
    HeatDelta delta;
    std::string err;
    const uint64_t now = realtime();
    check(src.poll(1000000000ULL,now,&delta,&err)==HeatPoll::INVALID,"missing file rejected");
    f.write();
    check(src.poll(1000000000ULL,realtime(),&delta,&err)==HeatPoll::BASELINE,"baseline");
    check(delta.complete_cells && !delta.complete_observations &&
          delta.counts.size()==8 && delta.observation_count==0,"file coverage not falsely credited");
    check(src.poll(1000000000ULL,realtime(),&delta,&err)==HeatPoll::UNCHANGED,"same inode unchanged");
    f.counts[0]=30; f.counts[3]=7; f.counts[4]=11;
    f.write();
    check(src.poll(1000000000ULL,realtime(),&delta,&err)==HeatPoll::DELTA,"changed file");
    check(delta.counts[0]==30 && delta.counts[3]==7 && delta.counts[4]==11 &&
          delta.interval_observations==48 && delta.observation_count==48,
          "model-global deltas and monotonic routed selections");
    f.counts[0]=32;
    f.write();
    check(src.poll(1000000000ULL,realtime(),&delta,&err)==HeatPoll::DELTA &&
          delta.counts[0]==2 && delta.observation_count==50,"interval not repeated");
    f.counts[0]=1;
    f.write();
    check(src.poll(1000000000ULL,realtime(),&delta,&err)==HeatPoll::INVALID &&
          err.find("regression")!=std::string::npos,"reset or wrap must invalidate");
    check(src.poll(1000000000ULL,realtime(),&delta,&err)==HeatPoll::BASELINE,
          "fresh baseline after reset");
    check(src.poll(1000000000ULL,realtime(),nullptr,&err)==HeatPoll::INVALID,"null output");
    check(src.poll(1000000000ULL,1,&delta,&err)==HeatPoll::INVALID,"future file timestamp rejected");

    HistogramFileSource outside(f.path,2,4);
    check(outside.poll(1000000000ULL,realtime(),&delta,&err)==HeatPoll::BASELINE,
          "separate baseline for layer domain check");
    f.counts[9]=1; f.write();
    check(outside.poll(1000000000ULL,realtime(),&delta,&err)==HeatPoll::INVALID &&
          err.find("outside")!=std::string::npos, "out-of-domain routes not counted");
    HistogramFileSource oversized(f.path,2,4);
    check(oversized.poll(1000000000ULL,realtime(),&delta,&err)==HeatPoll::BASELINE,
          "overflow baseline");
    f.counts[0]=UINT32_MAX; f.write();
    check(oversized.poll(1000000000ULL,realtime(),&delta,&err)==HeatPoll::INVALID &&
          err.find("overflow")!=std::string::npos, "oversized signed delta rejected");
    HistogramFileSource malformed(f.path,2,4);
    check(::truncate(f.path.c_str(), 7)==0,"truncate");
    check(malformed.poll(1000000000ULL,realtime(),&delta,&err)==HeatPoll::INVALID &&
          err.find("size")!=std::string::npos,"wrong production-shaped binary size rejected");
    HistogramFileSource wrong_domain(f.path,129,4);
    check(wrong_domain.poll(1000000000ULL,realtime(),&delta,&err)==HeatPoll::INVALID,
          "model domain limit rejected");
    std::puts("r3_heat_source_test: ok");
}
