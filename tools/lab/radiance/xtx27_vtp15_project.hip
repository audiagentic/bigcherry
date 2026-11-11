// VTP15 original BF16 Qwen3.8 GDN input projections on 3 physical GPUs.
//
// Each physical rank executes full original QKV/A/B/Z weight GEMVs on its
// own GPU stream. QKV is packed [Q(local)|K(local)|V(local)], while A/B
// and Z are split by independent value heads. The input is the original
// layer's weighted RMSNorm applied to a synthetic BF16 hidden state.
//
// All weights and independent CPU outputs come from immutable, hash-verified
// checkpoint-derived receipts supplied by the Python driver.
// This is a simple row/warp BF16 projection kernel, NOT native Radiance
// optimized dispatch, not a complete GDN layer or model serving.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define HC(op) do {const hipError_t e_=(op);if(e_!=hipSuccess){ \
 std::fprintf(stderr,"VTP15 HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e_));std::exit(2); \
 }} while(0)
constexpr int K=5120;
constexpr std::array<int,3> NKEYS{4,4,8};
constexpr std::array<int,3> NVALUES{12,12,24};
constexpr std::array<const char*,3> ROLES{"xtx0","xtx1","r9700"};
constexpr std::array<const char*,4> OPS{"qkv","a","b","z"};
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point begin) {
  return std::chrono::duration<double,std::micro>(Clock::now()-begin).count();
}
static double percentile(std::vector<double> data,double frac) {
  std::sort(data.begin(),data.end());
  const auto rank=std::clamp(static_cast<size_t>(std::ceil(data.size()*frac)),
                             size_t(1),data.size());
  return data[rank-1];
}
static const char* arg(int argc,char** argv,const char* name) {
  for(int i=1;i<argc;++i)if(std::strcmp(argv[i],name)==0) {
    if(i+1==argc)throw std::runtime_error(std::string("missing ")+name);
    return argv[i+1];
  }
  throw std::runtime_error(std::string("missing ")+name);
}
static int bounded(int argc,char** argv,const char* name,int limit) {
  const char* src=arg(argc,argv,name);
  char* end=nullptr;const long value=std::strtol(src,&end,10);
  if(!*src || !end || *end || value<1 || value>limit)
    throw std::runtime_error(std::string("invalid ")+name);
  return static_cast<int>(value);
}
template<typename T>static std::vector<T> read_exact(const std::string& path,size_t count) {
  std::ifstream file(path,std::ios::binary|std::ios::ate);
  if(!file || file.tellg()<0 || static_cast<size_t>(file.tellg())!=count*sizeof(T))
    throw std::runtime_error(std::string("wrong exact byte count: ")+path);
  file.seekg(0);
  std::vector<T> data(count);
  file.read(reinterpret_cast<char*>(data.data()),static_cast<std::streamsize>(count*sizeof(T)));
  if(!file) throw std::runtime_error(std::string("truncated file: ")+path);
  return data;
}
static std::array<int,3> physical_devices() {
  int count=0;HC(hipGetDeviceCount(&count));
  std::vector<int> xtx,r9700;
  for(int i=0;i<count;++i) {
    hipDeviceProp_t prop{};HC(hipGetDeviceProperties(&prop,i));
    if(std::strncmp(prop.gcnArchName,"gfx1100",7)==0)xtx.push_back(i);
    if(std::strncmp(prop.gcnArchName,"gfx1201",7)==0)r9700.push_back(i);
  }
  if(xtx.size()!=2 || r9700.size()!=1)
    throw std::runtime_error("need exactly two visible gfx1100 and one gfx1201 GPUs");
  return {xtx[0],xtx[1],r9700[0]};
}
__device__ static float bf16_float(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits)<<16);
}
__device__ static uint16_t bf16_rne(float x) {
  uint32_t bits=__float_as_uint(x);
  return static_cast<uint16_t>((bits+0x7FFFu+((bits>>16)&1u))>>16);
}
__global__ void original_bf16_gemv(const uint16_t* x,const uint16_t* weights,
                                   float* out32,uint16_t* out16,int rows,int n) {
  const int wave=warpSize;
  const int lane=threadIdx.x%wave;
  const int output=blockIdx.x*(blockDim.x/wave)+threadIdx.x/wave;
  const int row=blockIdx.y;
  float accum=0.0f;
  if(output<n && row<rows) {
    const uint16_t* w=weights+static_cast<size_t>(output)*K;
    const uint16_t* input=x+static_cast<size_t>(row)*K;
    for(int k=lane;k<K;k+=wave)
      accum+=bf16_float(w[k])*bf16_float(input[k]);
  }
  for(int offset=wave/2;offset>=1;offset/=2)
    accum+=__shfl_down(accum,offset);
  if(lane==0 && output<n && row<rows) {
    const size_t pos=static_cast<size_t>(row)*n+output;
    out32[pos]=accum;out16[pos]=bf16_rne(accum);
  }
}
struct Op {
  int n=0;
  uint16_t* weights=nullptr;
  uint16_t* y16=nullptr;
  float* y32=nullptr;
  std::vector<uint16_t> expected16;
  std::vector<float> expected32;
  std::vector<uint16_t> actual16;
  std::vector<float> actual32;
  double maxabs=0.0, maxrel=0.0, max_bf16=0.0;
  int exact_bf16=0, one_ulp=0, total=0;
};
struct Rank {
  int id=-1,rows=0,role=0;
  uint16_t* x=nullptr;
  hipStream_t stream=nullptr;
  hipEvent_t begin=nullptr,ready=nullptr;
  std::array<Op,4> ops{};
  std::vector<double> times;
};
static std::string path(const std::string& root,const std::string& prefix,
                        const char* role,const std::string& ext) {
  return root+"/"+prefix+"-"+role+"."+ext;
}
static void setup(Rank& rank,const std::string& pack,const std::string& qkv,
                  const std::string& ref) {
  HC(hipSetDevice(rank.id));
  HC(hipStreamCreateWithFlags(&rank.stream,hipStreamNonBlocking));
  HC(hipEventCreate(&rank.begin));HC(hipEventCreate(&rank.ready));
  const auto x=read_exact<uint16_t>(pack+"/normed_input.bf16",
                                     static_cast<size_t>(rank.rows)*K);
  HC(hipMalloc(reinterpret_cast<void**>(&rank.x),x.size()*sizeof(uint16_t)));
  HC(hipMemcpyAsync(rank.x,x.data(),x.size()*sizeof(uint16_t),
                    hipMemcpyHostToDevice,rank.stream));
  for(int i=0;i<4;++i) {
    Op& o=rank.ops[i];
    o.n=i==0?(2*NKEYS[rank.role]+NVALUES[rank.role])*128:
        (i==3?NVALUES[rank.role]*128:NVALUES[rank.role]);
    const char* role=ROLES[rank.role];
    const std::string op=OPS[i];
    const auto weights=read_exact<uint16_t>(
        i==0?path(qkv,"in_proj_qkv",role,"bf16"):path(pack,op,role,"bf16"),
        static_cast<size_t>(o.n)*K);
    const size_t count=static_cast<size_t>(o.n)*rank.rows;
    o.expected16=read_exact<uint16_t>(path(ref,op,role,"bf16"),count);
    o.expected32=read_exact<float>(path(ref,op,role,"f32"),count);
    o.actual16.resize(count);o.actual32.resize(count);
    HC(hipMalloc(reinterpret_cast<void**>(&o.weights),weights.size()*sizeof(uint16_t)));
    HC(hipMalloc(reinterpret_cast<void**>(&o.y16),count*sizeof(uint16_t)));
    HC(hipMalloc(reinterpret_cast<void**>(&o.y32),count*sizeof(float)));
    HC(hipMemcpyAsync(o.weights,weights.data(),weights.size()*sizeof(uint16_t),
                      hipMemcpyHostToDevice,rank.stream));
    HC(hipStreamSynchronize(rank.stream));
  }
}
static void launch(Rank& rank) {
  HC(hipSetDevice(rank.id));
  HC(hipEventRecord(rank.begin,rank.stream));
  for(auto& o:rank.ops) {
    const int waves=256/32;
    original_bf16_gemv<<<dim3((o.n+waves-1)/waves,rank.rows),256,0,rank.stream>>>(
        rank.x,o.weights,o.y32,o.y16,rank.rows,o.n);
  }
  HC(hipGetLastError());
  HC(hipEventRecord(rank.ready,rank.stream));
}
static void verify(Rank& rank) {
  HC(hipSetDevice(rank.id));
  for(auto& o:rank.ops) {
    HC(hipMemcpy(o.actual16.data(),o.y16,o.actual16.size()*sizeof(uint16_t),
                 hipMemcpyDeviceToHost));
    HC(hipMemcpy(o.actual32.data(),o.y32,o.actual32.size()*sizeof(float),
                 hipMemcpyDeviceToHost));
    for(size_t i=0;i<o.expected16.size();++i) {
      const double expected=o.expected32[i],actual=o.actual32[i];
      const double err=std::abs(expected-actual);
      o.maxabs=std::max(o.maxabs,err);
      o.maxrel=std::max(o.maxrel,err/(1.0+std::abs(expected)));
      if(!std::isfinite(actual) || err>0.0005+0.002*std::abs(expected))
        throw std::runtime_error("original BF16 Qwen GDN projection FP32 oracle FAILED");
      const double eb=std::abs(
        static_cast<double>(o.expected16[i])-static_cast<double>(o.actual16[i]));
      o.max_bf16=std::max(o.max_bf16,eb);
      if(o.expected16[i]==o.actual16[i])o.exact_bf16++;
      if(eb<=1)o.one_ulp++;
      o.total++;
      // The high-precision FP32 oracle above is authoritative: no
      // categorical failure for BF16 ties where FP32 order differs.
    }
  }
}
static void destroy(Rank& rank) {
  HC(hipSetDevice(rank.id));HC(hipStreamSynchronize(rank.stream));
  for(auto& o:rank.ops) {
    HC(hipFree(o.weights));HC(hipFree(o.y16));HC(hipFree(o.y32));
  }
  HC(hipFree(rank.x));
  HC(hipEventDestroy(rank.begin));HC(hipEventDestroy(rank.ready));
  HC(hipStreamDestroy(rank.stream));
}
int main(int argc,char** argv) {
  try {
    const int rows=bounded(argc,argv,"--rows",8);
    const int iters=bounded(argc,argv,"--iterations",30);
    const auto pack=std::string(arg(argc,argv,"--pack-dir"));
    const auto qkv=std::string(arg(argc,argv,"--qkv-dir"));
    const auto ref=std::string(arg(argc,argv,"--reference-dir"));
    const auto ids=physical_devices();
    std::array<Rank,3> rank{};
    for(int i=0;i<3;++i) {
      rank[i].id=ids[i];rank[i].rows=rows;rank[i].role=i;
      setup(rank[i],pack,qkv,ref);
    }
    std::vector<double> wall;
    for(int i=-2;i<iters;++i) {
      const auto start=Clock::now();
      for(auto& r:rank)launch(r);
      for(auto& r:rank) {
        HC(hipSetDevice(r.id));HC(hipEventSynchronize(r.ready));
      }
      if(i<0)continue;
      wall.push_back(elapsed(start));
      for(auto& r:rank) {
        HC(hipSetDevice(r.id));float ms=0;
        HC(hipEventElapsedTime(&ms,r.begin,r.ready));
        r.times.push_back(static_cast<double>(ms)*1000.);
      }
    }
    for(auto& r:rank)verify(r);
    std::printf("mode,rows,wall_us_p50,wall_us_p95,xtx0_us_p50,xtx1_us_p50,r9700_us_p50,max_fp32_abs,max_fp32_relative,bf16_exact,bf16_within_one_ulp,bf16_total,oracle\n");
    double maxabs=0.0,maxrel=0.0;
    int exact=0,within=0,total=0;
    for(auto& r:rank)for(auto& o:r.ops) {
      maxabs=std::max(maxabs,o.maxabs);
      maxrel=std::max(maxrel,o.maxrel);
      exact+=o.exact_bf16;within+=o.one_ulp;total+=o.total;
    }
    std::printf("original_bf16_gdn_input_gemvs_tp3,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.8g,%.8g,%d,%d,%d,PASS\n",
      rows,percentile(wall,.5),percentile(wall,.95),
      percentile(rank[0].times,.5),percentile(rank[1].times,.5),
      percentile(rank[2].times,.5),maxabs,maxrel,exact,within,total);
    std::printf("physical_ids=%d,%d,%d; source=original_checkpoint_BF16_weights; input=original_weighted_RMSNorm_synthetic_hidden; QKV_a_b_z_all_native_GPU_GEMV=YES; recurrence_attached=NO; optimized_kernel=NO; three_physical_contexts=YES; p2p=NO\n",
                ids[0],ids[1],ids[2]);
    for(auto& r:rank)destroy(r);
    return 0;
  } catch(const std::exception& e) {
    std::fprintf(stderr,"VTP15 original BF16 three-GPU GEMV FAILED: %s\n",e.what());
    return 1;
  }
}
