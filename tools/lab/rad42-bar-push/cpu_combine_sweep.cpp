// RAD50 S1 CPU-only isolate: actual AVX2/F16C helper combine into ordinary
// host RAM, versus combine into staging RAM followed by a host memcpy.
// No HIP, HSA, GPU, DMA-BUF, GPU lease, shell/subprocess or model inference.
// Timings are diagnostics, not VRAM/PCIe or serving-performance evidence.
#include "combine_f16.h"
#include "moe_remote_shm.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr unsigned WIDTH=2560;
constexpr unsigned PROFILES=6;
using Clock=std::chrono::steady_clock;
double elapsed_us(Clock::time_point a,Clock::time_point b) {
    return std::chrono::duration<double,std::micro>(b-a).count();
}
bool allowed_rows(unsigned n) {
    return n==1 || n==16 || n==64 || n==256 || n==768;
}
uint16_t word(unsigned profile,unsigned helper) {
    switch(profile) {
      case 0: { constexpr uint16_t values[]={0x3f80u,0x4000u,0x3f00u,0xbe80u};
                return values[helper]; } // 1,2,0.5,-0.25
      case 1: { constexpr uint16_t values[]={0x3f80u,0x4000u,0x4040u,0x4080u};
                return values[helper]; } // one-hot mask
      case 2: return helper==0?0x4780u:0x3f80u; // 65536 + helpers
      case 3: return helper==0?0x7fc0u:0u;     // canonical quiet BF16 NaN
      case 4: return helper==0?0x8000u:0u;     // signed zero
      default:return helper==0?0x0001u:(helper==1?0x8001u:0u); // subnormal BF16
    }
}
uint8_t row_mask(unsigned profile,unsigned helpers) {
    return profile==1?static_cast<uint8_t>(1u<<(helpers-1)):
                      static_cast<uint8_t>((1u<<helpers)-1u);
}
struct Input {
    unsigned rows,helpers;
    bool unique;
    std::array<std::vector<uint16_t>,4> planes;
    std::vector<uint16_t> expected;
    Input(unsigned row_count,unsigned nhelpers,bool own_rows):
        rows(row_count),helpers(nhelpers),unique(own_rows),
        expected(size_t(rows)*WIDTH) {
        const unsigned n=unique?rows:PROFILES;
        for(unsigned k=0;k<helpers;++k) {
            planes[k].resize(size_t(n)*WIDTH);
            for(unsigned row=0;row<n;++row) {
                const uint16_t value=word(row%PROFILES,k);
                std::fill_n(planes[k].data()+size_t(row)*WIDTH,WIDTH,value);
            }
        }
        // Independent scalar oracle, helper order matches the real worker.
        for(unsigned row=0;row<rows;++row) {
            const unsigned profile=row%PROFILES;
            const uint8_t mask=row_mask(profile,helpers);
            for(unsigned col=0;col<WIDTH;++col) {
                const size_t srcrow=unique?row:profile;
                float v=(mask&1u)?
                    combine_bf16_to_f32(planes[0][srcrow*WIDTH+col]):0.0f;
                for(unsigned k=1;k<helpers;++k)
                    if(mask & (1u<<k))
                        v+=combine_bf16_to_f32(planes[k][srcrow*WIDTH+col]);
                bool over=false;
                const uint16_t f16=combine_f32_f16_rn(v,&over);
                expected[size_t(row)*WIDTH+col]=
                    (col==0 && over)?r3::moe_shm::kCombineFallback:f16;
            }
        }
    }
    void combine(uint16_t* output)const {
        for(unsigned row=0;row<rows;++row) {
            const unsigned profile=row%PROFILES;
            const size_t srcrow=unique?row:profile;
            const uint16_t* src[4]{};
            for(unsigned k=0;k<helpers;++k)
                src[k]=planes[k].data()+srcrow*WIDTH;
            uint16_t* dst=output+size_t(row)*WIDTH;
            if(combine_row_f16(src,helpers,row_mask(profile,helpers),dst,WIDTH))
                dst[0]=r3::moe_shm::kCombineFallback;
        }
    }
};
double quant(std::vector<double> sorted,double q) {
    std::sort(sorted.begin(),sorted.end());
    return sorted[static_cast<size_t>(q*(sorted.size()-1))];
}
struct Sample {
    double direct=0,stage_combine=0,stage_copy=0,stage_total=0;
};
bool measure(const Input& in,unsigned iterations) {
    const size_t n=size_t(in.rows)*WIDTH;
    const size_t bytes=n*sizeof(uint16_t);
    std::vector<uint16_t> direct(n),stage(n),copied(n);
    std::vector<Sample> samples;
    samples.reserve(iterations);
    std::printf("RAD50_CPU_CONTEXT rows=%u width=%u helpers=%u "
                "layout=%s iterations=%u paths=host-direct,host-stage-copy "
                "source=deterministic_bf16 bitwise_oracle=1 gpu=0\n",
                in.rows,WIDTH,in.helpers,in.unique?"unique":"hot",iterations);
    std::fflush(stdout);
    for(unsigned i=0;i<iterations+3;++i) {
        std::fill(direct.begin(),direct.end(),0xa5a5u);
        std::fill(stage.begin(),stage.end(),0xa5a5u);
        std::fill(copied.begin(),copied.end(),0xa5a5u);
        Sample s;
        auto direct_run=[&]() {
            const auto start=Clock::now();
            in.combine(direct.data());
            s.direct=elapsed_us(start,Clock::now());
        };
        auto staged_run=[&]() {
            const auto begin=Clock::now();
            in.combine(stage.data());
            const auto halfway=Clock::now();
            std::memcpy(copied.data(),stage.data(),bytes);
            const auto end=Clock::now();
            s.stage_combine=elapsed_us(begin,halfway);
            s.stage_copy=elapsed_us(halfway,end);
            s.stage_total=elapsed_us(begin,end);
        };
        // Alternate order, including warmup, to avoid a fixed thermal bias.
        if(i&1u){staged_run();direct_run();}
        else {direct_run();staged_run();}
        if(std::memcmp(direct.data(),in.expected.data(),bytes)!=0 ||
           std::memcmp(copied.data(),in.expected.data(),bytes)!=0) {
            std::fprintf(stderr,
                "RAD50_CPU_FAIL rows=%u helpers=%u iteration=%u reason=wrong-fp16\n",
                in.rows,in.helpers,i);
            return false;
        }
        if(i<3u)continue;
        const unsigned n=i-3;
        samples.push_back(s);
        if(iterations<=15 || n<3 || (n+1)%10==0 || n+1==iterations) {
            std::printf("RAD50_CPU_SAMPLE rows=%u helpers=%u layout=%s "
                        "iteration=%u direct_us=%.3f stage_combine_us=%.3f "
                        "stage_copy_us=%.3f stage_total_us=%.3f "
                        "mismatches=0 status=0\n",in.rows,in.helpers,
                        in.unique?"unique":"hot",n+1,
                        s.direct,s.stage_combine,s.stage_copy,s.stage_total);
            std::fflush(stdout);
        }
    }
    std::vector<double> d,sc,cp,st;
    for(const Sample& s:samples) {
        d.push_back(s.direct);
        sc.push_back(s.stage_combine);
        cp.push_back(s.stage_copy);
        st.push_back(s.stage_total);
    }
    std::printf("RAD50_CPU_SUMMARY rows=%u helpers=%u layout=%s "
                "iterations=%u verified=%u emitted=%u direct_p50_us=%.3f "
                "direct_p95_us=%.3f direct_max_us=%.3f stage_combine_p50_us=%.3f "
                "stage_copy_p50_us=%.3f stage_total_p50_us=%.3f "
                "stage_total_p95_us=%.3f stage_total_max_us=%.3f status=0\n",
                in.rows,in.helpers,in.unique?"unique":"hot",iterations,iterations,
                iterations<=15?iterations:(3u+iterations/10u+(iterations%10u?1u:0u)),
                quant(d,.5),quant(d,.95),*std::max_element(d.begin(),d.end()),
                quant(sc,.5),quant(cp,.5),quant(st,.5),quant(st,.95),
                *std::max_element(st.begin(),st.end()));
    std::printf("RAD50_CPU_COMPLETE rows=%u helpers=%u layout=%s "
                "verified=%u status=0\n",in.rows,in.helpers,
                in.unique?"unique":"hot",iterations);
    return true;
}
bool strict_num(const char* text,unsigned min,unsigned max,unsigned* n) {
    if(!text || !*text)return false;
    unsigned v=0;
    for(const char* p=text;*p;++p) {
        if(*p<'0'||*p>'9')return false;
        v=v*10u+unsigned(*p-'0');
        if(v>max)return false;
    }
    if(v<min)return false;
    *n=v;return true;
}
} // namespace
int main(int argc,char** argv) {
    // Called only by the single fixed RAD50 test dispatcher, not an
    // arbitrary CLI execution surface or independent GPU-permission bypass.
    unsigned rows=0,iterations=0,helpers=0;
    if(argc!=9 || std::strcmp(argv[1],"--rows") ||
       std::strcmp(argv[3],"--iterations") ||
       std::strcmp(argv[5],"--helpers") ||
       std::strcmp(argv[7],"--mode") ||
       !strict_num(argv[2],1,768,&rows) || !allowed_rows(rows) ||
       !strict_num(argv[4],10,200,&iterations) ||
       !strict_num(argv[6],1,4,&helpers) ||
       (helpers!=1 && helpers!=2 && helpers!=4) ||
       (std::strcmp(argv[8],"hot") && std::strcmp(argv[8],"unique")))return 2;
    if(!combine_cpu_supported()) {
        std::fputs("RAD50_CPU_FAIL reason=cpu-avx2-f16c-unsupported\n",stderr);
        return 3;
    }
    const Input in(rows,helpers,!std::strcmp(argv[8],"unique"));
    return measure(in,iterations)?0:3;
}
